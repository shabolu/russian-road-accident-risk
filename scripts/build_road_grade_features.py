"""Признаки продольного уклона дорог по ячейкам H3 из Copernicus DEM GLO-30.

Всё считается ВДОЛЬ дорог (полилинии OSM, шаг 20 м), а не по рельефу ячейки:
ячейка может стоять на склоне, а дорога идти по горизонтали.

Уклон в точке — наклон линейной подгонки высоты в окне ±100 м вдоль дороги
(итого 200 м). Окно выбрано по замеру шума: на заведомо плоской пойме p90 |уклона|
при окне 200 м ≈ 2.9 %, при 20 м ≈ 6.6 %, при 100 м ≈ 4.7 %.

Признаки на ячейку (веса — длина отрезков):
  road_grade_abs_mean_weighted   средний |уклон|, %
  road_grade_abs_p90             90-й перцентиль |уклона|, % (не max — шум DEM)
  steep_road_share               доля длины с |уклоном| >= 5 %
  major_road_grade_abs_mean      средний |уклон| только по магистралям, %
  junction_on_grade_count        перекрёстки с локальным |уклоном| >= 3 %
  grade_change_density_per_km    переходы «плоско(<2%) <-> круто(>5%)» на км

DEM: Copernicus_DSM_COG_10_N{lat}_00_E{lon}_00_DEM с AWS Open Data, без авторизации.
Тайлы кэшируются в data/raw/dem/.

Запуск:
    venv/bin/python scripts/build_road_grade_features.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.paths import PARQUET_DIR, PROJECT_ROOT  # noqa: E402

DEM_DIR = PROJECT_ROOT / "data" / "raw" / "dem"
DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
STEP_M = 20.0
WINDOW_M = 200.0
STEEP_PCT = 5.0
FLAT_PCT = 2.0
JUNCTION_GRADE_PCT = 3.0
JUNCTION_RADIUS_M = 40.0
H3_RES = 9
ROAD_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link", "service",
}


def tile_name(lat0: int, lon0: int) -> str:
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat0):02d}_00_{ew}{abs(lon0):03d}_00_DEM"


def fetch_tile(lat0: int, lon0: int) -> Path:
    name = tile_name(lat0, lon0)
    path = DEM_DIR / f"{name}.tif"
    url = f"{DEM_BUCKET}/{name}/{name}.tif"
    if not path.exists():
        DEM_DIR.mkdir(parents=True, exist_ok=True)
        print(f"скачиваю {name} ...", file=sys.stderr)
        for attempt in range(6):  # S3 иногда рвёт соединение — докачиваем
            try:
                have = path.stat().st_size if path.exists() else 0
                req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
                with urllib.request.urlopen(req, timeout=300) as r, open(path, "ab") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        f.write(b)
                tifffile.TiffFile(path).close()
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
    return path


class Dem:
    """Мозаика тайлов 1°×1°. Тайл: 3600 строк × 2400 столбцов, tiepoint — левый верхний угол."""

    def __init__(self, lat_min: float, lon_min: float, lat_max: float, lon_max: float):
        self.tiles: dict[tuple[int, int], np.ndarray] = {}
        for lat0 in range(math.floor(lat_min), math.floor(lat_max) + 1):
            for lon0 in range(math.floor(lon_min), math.floor(lon_max) + 1):
                self.tiles[(lat0, lon0)] = tifffile.imread(fetch_tile(lat0, lon0))
        shape = next(iter(self.tiles.values())).shape
        self.rows, self.cols = shape
        self.res_lat = 1.0 / self.rows
        self.res_lon = 1.0 / self.cols

    def elev(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        out = np.full(len(lat), np.nan)
        lat0 = np.floor(lat).astype(int)
        lon0 = np.floor(lon).astype(int)
        for key, arr in self.tiles.items():
            m = (lat0 == key[0]) & (lon0 == key[1])
            if not m.any():
                continue
            r = ((key[0] + 1 - lat[m]) / self.res_lat).astype(int).clip(0, self.rows - 1)
            c = ((lon[m] - key[1]) / self.res_lon).astype(int).clip(0, self.cols - 1)
            out[m] = arr[r, c]
        return out


def resample(coords: list[list[float]], lat_ref: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Точки вдоль полилинии с шагом STEP_M: lat, lon, дистанция, вес (длина отрезка)."""
    k = math.cos(math.radians(lat_ref))
    lats, lons, dist, w = [], [], [], []
    acc = 0.0
    for (a, b), (c, d) in zip(coords, coords[1:]):
        seg = math.hypot((d - b) * 111320 * k, (c - a) * 110540)
        if seg <= 0:
            continue
        n = max(1, int(seg // STEP_M))
        for i in range(n):
            t = i / n
            lats.append(a + (c - a) * t)
            lons.append(b + (d - b) * t)
            dist.append(acc + seg * t)
            w.append(seg / n)
        acc += seg
    return np.array(lats), np.array(lons), np.array(dist), np.array(w)


def window_grade(s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """|уклон| в % для каждой точки: наклон линейной подгонки в окне ±WINDOW_M/2."""
    n = len(s)
    g = np.full(n, np.nan)
    half = WINDOW_M / 2
    lo = np.searchsorted(s, s - half, side="left")
    hi = np.searchsorted(s, s + half, side="right")
    for i in range(n):
        a, b = lo[i], hi[i]
        if b - a < 3:
            continue
        xs, ys = s[a:b], z[a:b]
        if np.isnan(ys).any():
            continue
        xm, ym = xs.mean(), ys.mean()
        den = ((xs - xm) ** 2).sum()
        if den == 0:
            continue
        g[i] = abs(((xs - xm) * (ys - ym)).sum() / den) * 100
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default="51.40,45.80,51.70,46.20", help="lat_min,lng_min,lat_max,lng_max")
    ap.add_argument("--out", default=str(PARQUET_DIR / "h3_road_grade_features.parquet"))
    ap.add_argument("--samples-out", default=str(PARQUET_DIR / "osm_road_grade_samples.parquet"))
    args = ap.parse_args()
    lat_min, lon_min, lat_max, lon_max = (float(x) for x in args.bbox.split(","))
    lat_ref = (lat_min + lat_max) / 2

    dem = Dem(lat_min, lon_min, lat_max, lon_max)
    raw = pd.read_parquet(PARQUET_DIR / "osm_roads_raw.parquet",
                          columns=["osm_id", "highway", "is_major", "coords_json"])
    raw = raw[raw["highway"].isin(ROAD_CLASSES)]
    print(f"дорог: {len(raw):,}", file=sys.stderr)

    parts = []
    for row in raw.itertuples(index=False):
        coords = json.loads(row.coords_json)
        if len(coords) < 2:
            continue
        la, lo, s, w = resample(coords, lat_ref)
        if len(la) < 3:
            continue
        z = dem.elev(la, lo)
        g = window_grade(s, z)
        parts.append(pd.DataFrame({
            "osm_id": row.osm_id, "is_major": bool(row.is_major),
            "lat": la, "lng": lo, "dist_m": s, "w": w, "elev": z, "grade_abs_pct": g,
        }))
    smp = pd.concat(parts, ignore_index=True)
    inside = (smp.lat.between(lat_min, lat_max)) & (smp.lng.between(lon_min, lon_max))
    smp = smp[inside & smp.grade_abs_pct.notna()].copy()
    smp["h3"] = [h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(smp.lat, smp.lng)]
    print(f"точек с уклоном: {len(smp):,}", file=sys.stderr)

    # переходы плоско<->круто вдоль каждой дороги
    smp = smp.sort_values(["osm_id", "dist_m"]).reset_index(drop=True)
    state = np.where(smp.grade_abs_pct >= STEEP_PCT, 1, np.where(smp.grade_abs_pct < FLAT_PCT, -1, 0))
    change = np.zeros(len(smp), dtype=bool)
    last = defaultdict(int)  # osm_id -> последнее ненулевое состояние
    for i, (oid, st) in enumerate(zip(smp.osm_id.to_numpy(), state)):
        if st == 0:
            continue
        prev = last[oid]
        if prev != 0 and st != prev:
            change[i] = True
        last[oid] = st
    smp["grade_change"] = change

    # агрегация в ячейки
    def wq(x: pd.Series, w: pd.Series, q: float) -> float:
        o = np.argsort(x.to_numpy()); xs = x.to_numpy()[o]; ws = w.to_numpy()[o]
        c = np.cumsum(ws) / ws.sum()
        return float(xs[np.searchsorted(c, q)])

    rows = []
    for cell, g in smp.groupby("h3"):
        W = g.w.sum()
        maj = g[g.is_major]
        rows.append({
            "h3": cell,
            "road_grade_abs_mean_weighted": float((g.grade_abs_pct * g.w).sum() / W),
            "road_grade_abs_p90": wq(g.grade_abs_pct, g.w, 0.90),
            "steep_road_share": float(g.w[g.grade_abs_pct >= STEEP_PCT].sum() / W),
            "major_road_grade_abs_mean": (float((maj.grade_abs_pct * maj.w).sum() / maj.w.sum())
                                          if len(maj) else np.nan),
            "grade_change_density_per_km": float(g.grade_change.sum() / (W / 1000.0)),
            "road_grade_sample_count": int(len(g)),
        })
    feat = pd.DataFrame(rows)

    # перекрёстки на уклоне: max |уклон| среди точек дорог в радиусе JUNCTION_RADIUS_M
    nodes = pd.read_parquet(PARQUET_DIR / "osm_road_graph_nodes.parquet", columns=["node_id", "h3", "lat", "lng"])
    k = math.cos(math.radians(lat_ref))
    by_cell = {c: g for c, g in smp.groupby("h3")}
    jcount: dict[str, int] = defaultdict(int)
    for n in nodes.itertuples(index=False):
        cand = [by_cell[c] for c in h3.grid_disk(n.h3, 1) if c in by_cell]
        if not cand:
            continue
        cg = pd.concat(cand)
        d = np.hypot((cg.lng.to_numpy() - n.lng) * 111320 * k, (cg.lat.to_numpy() - n.lat) * 110540)
        near = cg.grade_abs_pct.to_numpy()[d <= JUNCTION_RADIUS_M]
        if len(near) and near.max() >= JUNCTION_GRADE_PCT:
            jcount[n.h3] += 1
    feat["junction_on_grade_count"] = feat["h3"].map(jcount).fillna(0).astype(int)

    feat.to_parquet(args.out, index=False)
    smp[["osm_id", "h3", "lat", "lng", "w", "elev", "grade_abs_pct", "is_major", "grade_change"]].to_parquet(
        args.samples_out, index=False)
    print(f"ячеек: {len(feat):,}  ->  {args.out}", file=sys.stderr)
    print(feat.describe().T[["mean", "50%", "max"]].round(3).to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
