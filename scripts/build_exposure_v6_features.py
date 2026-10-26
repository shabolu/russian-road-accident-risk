"""Этап 6.2–6.5 (запуск 6a): признаки положения, POI, застройки и ОТ по ячейкам H3 res 9.

Источники — таблицы 6.1 (parquet_output/osm_v6_*.parquet), узлы этапа 2 (osm_junction_v2_nodes) для светофоров.
Ячейки — все h3 панели h3_month_road_features. Выход (новый файл, старые не трогаются):
    parquet_output/h3_exposure_v6_features.parquet
    docs/analysis/features/stage6_exposure/blocks_a/tables/{poi_peaks,stops_dedup,build_stats}.csv
Определения и пороги — src/road_accident_risk/exposure_v6.py и README блока.

    PYTHONPATH=src venv/bin/python scripts/build_exposure_v6_features.py
"""
from __future__ import annotations

import json
import time

import h3
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from road_accident_risk import exposure_v6 as ev
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PARQUET_DIR, PROJECT_ROOT
from road_accident_risk.street_graph_v2 import allocate_coords_to_h3, utm38n_forward

OUT = PARQUET_DIR / "h3_exposure_v6_features.parquet"
TAB = PROJECT_ROOT / "docs/analysis/features/stage6_exposure/blocks_a/tables"
RES = 9
BBOX = (51.40, 45.80, 51.70, 46.20)  # lat_min, lon_min, lat_max, lon_max


def utm_inverse(x: float, y: float) -> tuple[float, float]:
    """Обратная проекция UTM 38N -> (lat, lon) Ньютоном по прямой (для отчёта о точках пиков)."""
    lat, lon = 51.5, 46.0
    for _ in range(20):
        fx, fy = utm38n_forward(lat, lon)
        e = 1e-5
        dx_lat, dy_lat = [(a - b) / e for a, b in zip(utm38n_forward(lat + e, lon), (fx, fy))]
        dx_lon, dy_lon = [(a - b) / e for a, b in zip(utm38n_forward(lat, lon + e), (fx, fy))]
        J = np.array([[float(dx_lat), float(dx_lon)], [float(dy_lat), float(dy_lon)]])
        d = np.linalg.solve(J, np.array([x - float(fx), y - float(fy)]))
        lat += d[0]; lon += d[1]
        if abs(d).max() < 1e-10:
            break
    return lat, lon


def xy(lat, lon) -> np.ndarray:
    x, y = utm38n_forward(np.asarray(lat, float), np.asarray(lon, float))
    return np.column_stack([x, y])


def sample_lines(coord_lists, step=ev.LINE_SAMPLE_M) -> np.ndarray:
    """Точки вдоль полилиний (lat, lon) с шагом ≤ step м, в UTM."""
    pts = []
    for coords in coord_lists:
        c = xy([p[0] for p in coords], [p[1] for p in coords])
        for a, b in zip(c[:-1], c[1:]):
            n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
            t = np.linspace(0, 1, n + 1)[:, None]
            pts.append(a + (b - a) * t)
    return np.vstack(pts)


def attractor_mask(p: pd.DataFrame) -> pd.Series:
    am = p["amenity"].fillna("")
    return (p["shop"].notna() & ~p["shop"].isin(ev.SHOP_EXCLUDED)) | p["office"].notna() | am.isin(ev.ATTRACTOR_AMENITY)


def dedup_stops(s: pd.DataFrame) -> pd.DataFrame:
    """Single linkage в радиусе 30 м (как EDA 6.1); остановка = центроид кластера."""
    P = xy(s.lat.values, s.lon.values)
    pairs = cKDTree(P).query_pairs(ev.STOP_DEDUP_M, output_type="ndarray")
    n = len(s)
    g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    s = s.assign(cluster=connected_components(g, directed=False)[1])
    return s.groupby("cluster").agg(lat=("lat", "mean"), lon=("lon", "mean"), n_objects=("lat", "size")).reset_index()


def point_in_ring(px: float, py: float, ring: np.ndarray) -> bool:
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cond = (y > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xin = x + (py - y) * (x2 - x) / (y2 - y)
    return bool(np.count_nonzero(cond & (px < xin)) % 2)


def main() -> None:
    t0 = time.time()
    TAB.mkdir(parents=True, exist_ok=True)
    stats: list[dict] = []
    cells = pd.DataFrame({"h3": pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET, columns=["h3"]).h3.unique()})
    ll = np.array([h3.cell_to_latlng(c) for c in cells.h3])
    cells["lat"], cells["lon"] = ll[:, 0], ll[:, 1]
    C = xy(cells.lat, cells.lon)
    print(f"ячеек {len(cells)}", flush=True)

    # ---------------- E1: положение
    def dist_to_points(pts_ll):
        return cKDTree(xy([p[0] for p in pts_ll], [p[1] for p in pts_ll])).query(C)[0]

    cells["dist_to_center_two_m"] = dist_to_points([ev.SARATOV_CENTER, ev.ENGELS_CENTER])

    poi = pd.read_parquet(PARQUET_DIR / "osm_v6_poi.parquet")
    poi = poi[poi.region == "saratov_buffer"].reset_index(drop=True)
    poi["is_attr"] = attractor_mask(poi)
    poi["is_school"] = poi.amenity.isin(ev.SCHOOL_AMENITY)
    poi["is_health"] = poi.healthcare.notna() | poi.amenity.isin(ev.HEALTH_AMENITY)
    attr = poi[poi.is_attr].reset_index(drop=True)
    A = xy(attr.lat, attr.lon)
    g = ev.POI_PEAK_GRID_M
    x0, y0 = C.min(axis=0) - 5000; x1, y1 = C.max(axis=0) + 5000
    ex = np.arange(x0, x1 + g, g); ey = np.arange(y0, y1 + g, g)
    H, _, _ = np.histogram2d(A[:, 0], A[:, 1], bins=[ex, ey])
    D = ndimage.gaussian_filter(H, ev.POI_PEAK_SIGMA_M / g)
    mx = ndimage.maximum_filter(D, size=int(ev.POI_PEAK_WINDOW_M / g))
    pk = np.argwhere((D == mx) & (D >= ev.POI_PEAK_MIN_REL * D.max()))
    peaks = pd.DataFrame({"x": ex[pk[:, 0]] + g / 2, "y": ey[pk[:, 1]] + g / 2, "rel_density": D[pk[:, 0], pk[:, 1]] / D.max()})
    peaks[["lat", "lon"]] = [utm_inverse(a, b) for a, b in zip(peaks.x, peaks.y)]
    sc = xy(*ev.SARATOV_CENTER); en = xy(*ev.ENGELS_CENTER)
    peaks["dist_to_saratov_center_m"] = np.hypot(peaks.x - sc[0, 0], peaks.y - sc[0, 1])
    peaks["dist_to_engels_center_m"] = np.hypot(peaks.x - en[0, 0], peaks.y - en[0, 1])
    peaks = peaks.sort_values("rel_density", ascending=False).reset_index(drop=True)
    peaks.to_csv(TAB / "poi_peaks.csv", index=False)
    # чувствительность к σ (только для README)
    sens = []
    for sig in (500, 1000, 1500):
        for rel in (0.15, 0.20, 0.30):
            Ds = ndimage.gaussian_filter(H, sig / g)
            m = ndimage.maximum_filter(Ds, size=int(ev.POI_PEAK_WINDOW_M / g))
            sens.append({"sigma_m": sig, "min_rel": rel, "n_peaks": int(((Ds == m) & (Ds >= rel * Ds.max())).sum())})
    pd.DataFrame(sens).to_csv(TAB / "poi_peaks_sensitivity.csv", index=False)
    np.save(TAB / "poi_density_grid.npy", D.astype(np.float32))
    pd.DataFrame({"ex0": [ex[0]], "ey0": [ey[0]], "step": [g], "nx": [len(ex) - 1], "ny": [len(ey) - 1]}).to_csv(
        TAB / "poi_density_grid_meta.csv", index=False)
    cells["dist_to_center_poi_m"] = cKDTree(peaks[["x", "y"]].to_numpy()).query(C)[0]
    print(peaks.round(3).to_string(), flush=True)

    streets = pd.read_parquet(PARQUET_DIR / "osm_v6_streets.parquet", columns=["way_id", "highway", "coords_json"])
    major = streets[streets.highway.isin(ev.MAJOR_CLASSES)]
    M = sample_lines([json.loads(c) for c in major.coords_json])
    cells["dist_to_nearest_major_m"] = cKDTree(M).query(C)[0]
    j = pd.read_parquet(PARQUET_DIR / "osm_junction_v2_nodes.parquet", columns=["lat", "lng", "is_signalized_junction"])
    j = j[j.is_signalized_junction]
    cells["dist_to_nearest_signal_m"] = cKDTree(xy(j.lat, j.lng)).query(C)[0]
    stats += [{"what": "major_edges_ways", "value": len(major)}, {"what": "major_samples", "value": len(M)},
              {"what": "signalized_junctions", "value": len(j)}, {"what": "poi_peaks", "value": len(peaks)}]
    print(f"E1 готов [{time.time() - t0:.0f} с]", flush=True)

    # ---------------- E2: POI
    poi["h3"] = [h3.latlng_to_cell(a, b, RES) for a, b in zip(poi.lat, poi.lon)]
    P = xy(poi.lat, poi.lon)
    for flag, name in (("is_attr", "poi_attractor_count"), ("is_school", "school_kindergarten_count"),
                       ("is_health", "healthcare_count")):
        sub = poi[poi[flag]]
        cells[name] = cells.h3.map(sub.groupby("h3").size()).fillna(0).astype(float)
        tree = cKDTree(P[poi[flag].to_numpy()])
        for r in ev.POI_BUFFERS_M:
            cells[f"{name}_r{r}"] = np.array([len(v) for v in tree.query_ball_point(C, r)], dtype=float)
        stats.append({"what": f"poi_{flag}", "value": int(poi[flag].sum())})
    print(f"E2 готов [{time.time() - t0:.0f} с]", flush=True)

    # ---------------- E3: застройка (здание -> ячейка по центроиду)
    b = pd.read_parquet(PARQUET_DIR / "osm_v6_buildings.parquet",
                        columns=["osm_type", "osm_id", "region", "lat", "lon", "levels", "footprint_m2", "rings_json"])
    b = b[b.region == "saratov_buffer"].reset_index(drop=True)
    b["h3"] = [h3.latlng_to_cell(a, c, RES) for a, c in zip(b.lat, b.lon)]
    B = xy(b.lat, b.lon)
    # здания с POI-аттрактором внутри: точка (узел или центроид площади POI) в полигоне здания
    btree = cKDTree(B)
    has_attr = np.zeros(len(b), bool)
    rings_cache: dict[int, list] = {}
    n_tests = 0
    for (px, py) in A:
        idx = btree.query_ball_point((px, py), 400.0)
        if not idx:
            continue
        idx = sorted(idx, key=lambda i: (B[i, 0] - px) ** 2 + (B[i, 1] - py) ** 2)[:40]
        for i in idx:
            if i not in rings_cache:
                rings_cache[i] = [(s, xy([q[0] for q in r], [q[1] for q in r])) for s, r in json.loads(b.rings_json[i])]
            rings = rings_cache[i]
            n_tests += 1
            inside = any(point_in_ring(px, py, r) for s, r in rings if s == 1) and \
                not any(point_in_ring(px, py, r) for s, r in rings if s == -1)
            if inside:
                has_attr[i] = True
                break
    b["has_attr"] = has_attr
    stats += [{"what": "attractor_buildings", "value": int(has_attr.sum())}, {"what": "pip_tests", "value": n_tests},
              {"what": "attractors_in_building_share", "value": round(float(has_attr.sum()) / len(A), 4)}]
    b["small_fp"] = b.footprint_m2.where(b.footprint_m2 < ev.SMALL_BUILDING_M2, 0.0)
    b["attr_fp"] = b.footprint_m2.where(b.has_attr, 0.0)
    b["levels_fp"] = b.footprint_m2.where(b.levels.notna(), 0.0)
    gb = b.groupby("h3").agg(building_count=("footprint_m2", "size"), building_footprint_m2=("footprint_m2", "sum"),
                             small_fp=("small_fp", "sum"), attractor_building_footprint_m2=("attr_fp", "sum"),
                             levels_fp=("levels_fp", "sum"))
    cells = cells.merge(gb, left_on="h3", right_index=True, how="left")
    for c in ("building_count", "building_footprint_m2", "attractor_building_footprint_m2"):
        cells[c] = cells[c].fillna(0.0).astype(float)
    den = cells.building_footprint_m2.replace(0, np.nan)
    cells["small_building_share"] = cells.small_fp / den
    cells["levels_tagged_share"] = cells.levels_fp / den
    cells = cells.drop(columns=["small_fp", "levels_fp"])
    print(f"E3 готов [{time.time() - t0:.0f} с]", flush=True)

    # ---------------- E4: остановки, трамвай, маршруты
    s = pd.read_parquet(PARQUET_DIR / "osm_v6_pt_stops.parquet")
    s = s[s.region == "saratov_buffer"].reset_index(drop=True)
    s["h3"] = [h3.latlng_to_cell(a, c, RES) for a, c in zip(s.lat, s.lon)]
    cells["pt_stop_raw_count"] = cells.h3.map(s.groupby("h3").size()).fillna(0).astype(float)
    st = dedup_stops(s)
    st["h3"] = [h3.latlng_to_cell(a, c, RES) for a, c in zip(st.lat, st.lon)]
    st.to_csv(TAB / "stops_dedup.csv", index=False)
    cells["pt_stop_count_v6"] = cells.h3.map(st.groupby("h3").size()).fillna(0).astype(float)
    cells["dist_to_nearest_stop_m"] = cKDTree(xy(st.lat, st.lon)).query(C)[0]
    in_bbox = st.lat.between(BBOX[0], BBOX[2]) & st.lon.between(BBOX[1], BBOX[3])
    stats += [{"what": "stop_objects_buffer", "value": len(s)}, {"what": "stop_clusters_buffer", "value": len(st)},
              {"what": "stop_clusters_bbox", "value": int(in_bbox.sum())}]

    t = pd.read_parquet(PARQUET_DIR / "osm_v6_tram_tracks.parquet")
    t = t[(t.region == "saratov_buffer") & ~t.service.isin(ev.TRAM_SERVICE_EXCLUDED)]
    tram: dict[str, float] = {}
    for c in t.coords_json:
        for k, v in allocate_coords_to_h3([tuple(p) for p in json.loads(c)], resolution=RES).items():
            tram[k] = tram.get(k, 0.0) + v
    cells["tram_track_length_m"] = cells.h3.map(tram).fillna(0.0)
    stats.append({"what": "tram_km_used", "value": round(t.length_m.sum() / 1000, 1)})

    r = pd.read_parquet(PARQUET_DIR / "osm_v6_pt_routes.parquet")
    r = r[(r.region == "saratov_buffer") & r.route.isin(ev.ROUTE_TYPES)].copy()
    r["key"] = r.route + ":" + r.ref.fillna(r.name.fillna(r.rel_id.astype(str)))
    geo = dict(zip(streets.way_id, streets.coords_json))
    way_cells: dict[int, set] = {}
    found = total = 0
    route_cells: dict[str, set] = {}
    for key, ways in zip(r.key, r.way_ids):
        acc = route_cells.setdefault(key, set())
        for w in ways:
            total += 1
            if w not in geo:
                continue
            found += 1
            if w not in way_cells:
                way_cells[w] = set(allocate_coords_to_h3([tuple(p) for p in json.loads(geo[w])], resolution=RES))
            acc |= way_cells[w]
    cnt: dict[str, int] = {}
    for key, cs in route_cells.items():
        for c in cs:
            cnt[c] = cnt.get(c, 0) + 1
    cells["pt_route_count"] = cells.h3.map(cnt).fillna(0).astype(float)
    stats += [{"what": "routes_relations", "value": len(r)}, {"what": "routes_unique_keys", "value": r.key.nunique()},
              {"what": "route_way_refs_with_geometry_share", "value": round(found / total, 4)}]
    print(f"E4 готов [{time.time() - t0:.0f} с]", flush=True)

    out = cells[["h3", "lat", "lon", *ev.EXPOSURE_V6_FEATURES]]
    assert list(out.columns[3:]) == ev.EXPOSURE_V6_FEATURES
    out.to_parquet(OUT, index=False)
    pd.DataFrame(stats).to_csv(TAB / "build_stats.csv", index=False)
    print(pd.DataFrame(stats).to_string(index=False))
    print(out.describe().T.round(2).to_string())
    print(f"-> {OUT} {out.shape} [{time.time() - t0:.0f} с]")


if __name__ == "__main__":
    main()
