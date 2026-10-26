"""Road+Geo v2, этап 5 (2026-10-16): паспортные атрибуты ФГИС СКДФ в ячейках H3.

Источник — parquet_output/skdf_traffic_saratov.geojson (слой skdf_open:lyr_road_conditions_traffic,
5 594 участка, 44 поля). Агрегация как у h3_skdf_features.parquet (EDA 2026-09-16): сэмплы ~20 м
вдоль линий, вес — длина куска участка в ячейке. Старый файл не трогается; новый —
H3_SKDF_V2_FEATURES_PARQUET. Итог: docs/analysis/features/stage5_skdf/README.md.

Расшифровка gid — по легенде стилей WFS (GetLegendGraphic, JSON), см. stage5_skdf/LOG.md §1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from road_accident_risk.street_graph_v2 import allocate_coords_to_h3

# --- расшифровка gid -----------------------------------------------------------------------------
VALUE_RANK = {83717: 3, 83718: 2, 83719: 1}  # федеральная / региональная / местная
VALUE_NAME = {83717: "федеральная", 83718: "региональная", 83719: "местная"}
# категория по СНиП: больше — выше (IA 7 … V 1)
SNIP_RANK = {51845: 7, 51846: 6, 51847: 5, 51848: 4, 51849: 3, 51850: 2, 51851: 1}
SNIP_NAME = {51845: "IA", 51846: "IБ", 51847: "IВ", 51848: "II", 51849: "III", 51850: "IV", 51851: "V"}
# тип дорожной одежды (легенда coating); 2043912/2043913 нет в легенде — только рег./фед. IБ–III,
# ширина 12–14 м -> капитальная (предположение, README этапа 5)
COATING_CLASS = {
    **{g: "капитальная" for g in (51833, 51834, 51835, 51836, 2041399, 2041356, 1643500, 2043912, 2043913)},
    51837: "облегчённая", 51838: "облегчённая",
    1643501: "переходная", 1643484: "переходная",
    51839: "низшая", 1643483: "низшая", 2041355: "низшая",
}
COATING_LOW = {"переходная", "низшая"}  # грунт / щебень / гравий
# категория городской улицы по СП 42.13330 (легенда category_sp); 2043903–06 нет в легенде:
# все местные, 97 % — V по СНиП, 30–40 км/ч, 1–2 полосы -> низший уровень (проезды / жилая застройка)
SP_CLASS = {
    1643508: "мгд", 1643509: "мгд",
    1643510: "муоз", 1643511: "муоз", 1643512: "муоз",
    1643513: "мурз",
    **{g: "местн" for g in (1643514, 1643518, 1643519, 1643525, 1643526, 1643527)},
    1643517: "пеш",
    **{g: "проезд" for g in (2043903, 2043904, 2043905, 2043906)},
}
SP_RANK = {"мгд": 5, "муоз": 4, "мурз": 3, "местн": 2, "проезд": 1}
SP_NAME = {"мгд": "магистральные городские дороги", "муоз": "магистр. улицы общегородского знач.",
           "мурз": "магистр. улицы районного знач.", "местн": "улицы и дороги местного знач.",
           "проезд": "проезды / жилая застройка (код 20439xx)"}
AXLE_LOAD_T = {1553237: 6.0, 1553213: 10.0, 1553214: 11.5}
TRAFFIC_PLACEHOLDER = 380.0
MIN_KNOWN_M = 0.0  # кусок с известным атрибутом достаточен; порог не вводим (паспорт полон на 94–99 %)

# --- признаки ячейки -----------------------------------------------------------------------------
# кандидаты в модель (дедуп и абляция — stage5_skdf/README.md)
SKDF_V2_CANDIDATE_FEATURES = [
    "skdf_roadway_width_mean_w",
    "skdf_width_per_lane_mean_w",
    "skdf_lanes_mean_w",
    "skdf_cat_sp_rank_mean_w",
    "skdf_cat_sp_share_magistral",
    "skdf_cat_sp_share_mgd",
    "skdf_cat_sp_share_muoz",
    "skdf_cat_sp_share_murz",
    "skdf_cat_sp_share_local",
    "skdf_cat_sp_share_proezd",
    "skdf_snip_rank_mean_w",
    "skdf_value_rank_mean_w",
    "skdf_low_coating_share",
    "skdf_speed_mean_w",
    "skdf_axle_load_mean_w",
    "skdf_real_traffic_max",
    "skdf_veh_km_real",
]
# только диагностика
SKDF_V2_DIAGNOSTIC_FEATURES = [
    "skdf_len_m",
    "skdf_coverage_share",
    "skdf_placeholder_share",
    "skdf_width_known_len_m",
    "skdf_sp_known_len_m",
]
SKDF_V2_FEATURES = [*SKDF_V2_CANDIDATE_FEATURES, *SKDF_V2_DIAGNOSTIC_FEATURES]


def _gid(v) -> int | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)


def load_skdf_segments(path: Path) -> pd.DataFrame:
    """Участки СКДФ с расшифрованными атрибутами; coords — список линий [(lat, lng), …]."""
    feats = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    rows = []
    for f in feats:
        p = f["properties"]; g = f["geometry"]
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        coating = COATING_CLASS.get(_gid(p.get("coating_gid")))
        sp = SP_CLASS.get(_gid(p.get("category_snip_gid")))
        lanes = p.get("lanes"); width = p.get("roadway_width"); traffic = p.get("traffic")
        lanes = float(lanes) if lanes else np.nan
        width = float(width) if width else np.nan
        traffic = float(traffic) if traffic else np.nan
        rows.append({
            "skdf_id": int(p["id"]), "road_id": p.get("road_id"), "road_name": p.get("road_name"),
            "value_gid": _gid(p.get("value_of_the_road_gid")),
            "value_rank": VALUE_RANK.get(_gid(p.get("value_of_the_road_gid")), np.nan),
            "snip_gid": _gid(p.get("category_gid")), "snip_rank": SNIP_RANK.get(_gid(p.get("category_gid")), np.nan),
            "sp_gid": _gid(p.get("category_snip_gid")), "sp_class": sp,
            "sp_rank": SP_RANK.get(sp, np.nan) if sp else np.nan,
            "coating_gid": _gid(p.get("coating_gid")), "coating_class": coating,
            "is_low_coating": (coating in COATING_LOW) if coating else np.nan,
            "lanes": lanes, "roadway_width": width,
            "width_per_lane": width / lanes if (width == width and lanes == lanes and lanes > 0) else np.nan,
            "top_speed": float(p["top_speed"]) if p.get("top_speed") else np.nan,
            "axle_load_t": AXLE_LOAD_T.get(_gid(p.get("os_gid")), np.nan),
            "traffic": traffic,
            "is_placeholder": (traffic == TRAFFIC_PLACEHOLDER) if traffic == traffic else np.nan,
            "real_traffic": traffic if (traffic == traffic and traffic != TRAFFIC_PLACEHOLDER) else np.nan,
            "coords": [[(lat, lng) for lng, lat in line] for line in lines],
        })
    return pd.DataFrame(rows)


def allocate_skdf_to_h3(seg: pd.DataFrame) -> pd.DataFrame:
    """Куски участок × ячейка (сэмплы ~20 м), длина куска — piece_length_m."""
    out = []
    for sid, lines in zip(seg["skdf_id"], seg["coords"]):
        acc: dict[str, float] = {}
        for line in lines:
            for cell, L in allocate_coords_to_h3(line).items():
                acc[cell] = acc.get(cell, 0.0) + L
        out += [(sid, c, L) for c, L in acc.items()]
    return pd.DataFrame(out, columns=["skdf_id", "h3", "piece_length_m"])


def _wmean(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    known = df[col].notna()
    L = df["piece_length_m"].where(known, 0.0)
    num = (df[col].fillna(0) * L).groupby(df["h3"]).sum()
    den = L.groupby(df["h3"]).sum()
    return (num / den).where(den > MIN_KNOWN_M), den


def aggregate_h3_skdf_v2_features(seg: pd.DataFrame, pieces: pd.DataFrame,
                                  street_length: pd.Series | None = None) -> pd.DataFrame:
    """Признаки ячейки. Среднее/доля — только по длине с известным атрибутом (иначе NaN).

    skdf_roadway_width_mean_w   — Σ ширина·L / Σ L, м;   skdf_width_per_lane_mean_w — то же для ширины/полосу;
    skdf_lanes_mean_w           — полосность паспорта;    skdf_cat_sp_rank_mean_w — ранг категории по СП (5 мгд … 1 проезд);
    skdf_cat_sp_share_*         — доли длины категорий СП среди длины с известной категорией;
    skdf_snip_rank_mean_w       — ранг категории СНиП (7 IA … 1 V); skdf_value_rank_mean_w — 3 фед. / 2 рег. / 1 мест.;
    skdf_low_coating_share      — доля переходной + низшей одежды среди длины с известной одеждой;
    skdf_speed_mean_w, skdf_axle_load_mean_w — паспортная скорость и осевая нагрузка;
    skdf_real_traffic_max       — максимум traffic без заглушки 380 (NaN, если настоящих нет);
    skdf_veh_km_real            — Σ L·traffic / 1000 только по настоящим значениям (0, если их нет);
    skdf_len_m, skdf_coverage_share (= skdf_len_m / street_length_m OSM), skdf_placeholder_share — диагностика.
    """
    d = pieces.merge(seg.drop(columns=["coords"]), on="skdf_id", how="left", validate="many_to_one")
    g = pd.DataFrame(index=pd.Index(sorted(d["h3"].unique()), name="h3"))
    g["skdf_len_m"] = d.groupby("h3")["piece_length_m"].sum()
    for col, name in [("roadway_width", "skdf_roadway_width_mean_w"), ("width_per_lane", "skdf_width_per_lane_mean_w"),
                      ("lanes", "skdf_lanes_mean_w"), ("sp_rank", "skdf_cat_sp_rank_mean_w"),
                      ("snip_rank", "skdf_snip_rank_mean_w"), ("value_rank", "skdf_value_rank_mean_w"),
                      ("top_speed", "skdf_speed_mean_w"), ("axle_load_t", "skdf_axle_load_mean_w")]:
        g[name], den = _wmean(d, col)
        if col == "roadway_width":
            g["skdf_width_known_len_m"] = den
        if col == "sp_rank":
            g["skdf_sp_known_len_m"] = den
    d["_low"] = d["is_low_coating"].astype(float)
    g["skdf_low_coating_share"], _ = _wmean(d, "_low")
    d["_ph"] = d["is_placeholder"].astype(float)
    g["skdf_placeholder_share"], _ = _wmean(d, "_ph")
    for cls, name in [("мгд", "mgd"), ("муоз", "muoz"), ("мурз", "murz"), ("местн", "local"), ("проезд", "proezd")]:
        d["_x"] = np.where(d["sp_class"].isna(), np.nan, (d["sp_class"] == cls).astype(float))
        g[f"skdf_cat_sp_share_{name}"], _ = _wmean(d, "_x")
    d["_x"] = np.where(d["sp_class"].isna(), np.nan, d["sp_class"].isin(["мгд", "муоз", "мурз"]).astype(float))
    g["skdf_cat_sp_share_magistral"], _ = _wmean(d, "_x")
    g["skdf_real_traffic_max"] = d.groupby("h3")["real_traffic"].max()
    g["skdf_veh_km_real"] = (d["real_traffic"].fillna(0) * d["piece_length_m"] / 1000).groupby(d["h3"]).sum()
    if street_length is not None:
        sl = street_length.reindex(g.index)
        g["skdf_coverage_share"] = (g["skdf_len_m"] / sl).where(sl > 0)
    else:
        g["skdf_coverage_share"] = np.nan
    return g[SKDF_V2_FEATURES].reset_index()


# --- сопоставление СКДФ ↔ OSM (валидация, фигуры 3–5; в признаки не входит) --------------------
MATCH_STEP_M = 10.0      # шаг точек вдоль линий
MATCH_RADIUS_M = 20.0    # ближайший параллельный way OSM не дальше
PARALLEL_RADIUS_M = 30.0  # «сечение»: все параллельные ways OSM (две проезжие части)
PARALLEL_MAX_DEG = 25.0  # допуск по направлению
_LAT0 = 51.55


def _xy(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    return np.c_[lng * 111_320.0 * np.cos(np.radians(_LAT0)), lat * 110_574.0]


def sample_lines(lines_by_id: list[tuple[int, list[list[tuple[float, float]]]]],
                 step_m: float = MATCH_STEP_M) -> pd.DataFrame:
    """Точки через ~step_m вдоль линий: id, lat, lng, x, y, bearing (0..180°), вес — длина куска."""
    ids, la, ln, br, w = [], [], [], [], []
    for oid, lines in lines_by_id:
        for line in lines:
            a = np.asarray(line, float)
            if len(a) < 2:
                continue
            xy = _xy(a[:, 0], a[:, 1])
            d = np.diff(xy, axis=0); seg = np.hypot(d[:, 0], d[:, 1])
            for k in np.nonzero(seg > 0)[0]:
                n = max(1, int(np.ceil(seg[k] / step_m)))
                f = (np.arange(n) + 0.5) / n
                ids += [oid] * n; w += [seg[k] / n] * n
                la += list(a[k, 0] + (a[k + 1, 0] - a[k, 0]) * f); ln += list(a[k, 1] + (a[k + 1, 1] - a[k, 1]) * f)
                br += [np.degrees(np.arctan2(d[k, 0], d[k, 1])) % 180.0] * n
    la = np.asarray(la); ln = np.asarray(ln); xy = _xy(la, ln)
    return pd.DataFrame({"id": ids, "lat": la, "lng": ln, "x": xy[:, 0], "y": xy[:, 1], "bearing": br, "w": w})


def match_points_to_osm(sk_pts: pd.DataFrame, osm_pts: pd.DataFrame) -> pd.DataFrame:
    """Для каждой точки СКДФ: ближайший параллельный way OSM (≤ MATCH_RADIUS_M) и множество параллельных
    ways в сечении (≤ PARALLEL_RADIUS_M). Колонки: osm_idx, dist_m, parallel_ids (tuple)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(osm_pts[["x", "y"]].to_numpy())
    q = sk_pts[["x", "y"]].to_numpy(); b = sk_pts["bearing"].to_numpy()
    ob = osm_pts["bearing"].to_numpy(); oid = osm_pts["id"].to_numpy(); oxy = osm_pts[["x", "y"]].to_numpy()
    near, dist, par = [], [], []
    for i, cand in enumerate(tree.query_ball_point(q, PARALLEL_RADIUS_M)):
        if not cand:
            near.append(-1); dist.append(np.nan); par.append(()); continue
        c = np.asarray(cand)
        dang = np.abs(ob[c] - b[i]); dang = np.minimum(dang, 180 - dang)
        c = c[dang <= PARALLEL_MAX_DEG]
        if len(c) == 0:
            near.append(-1); dist.append(np.nan); par.append(()); continue
        dd = np.hypot(oxy[c, 0] - q[i, 0], oxy[c, 1] - q[i, 1])
        j = int(np.argmin(dd))
        near.append(int(oid[c[j]]) if dd[j] <= MATCH_RADIUS_M else -1)
        dist.append(float(dd[j]) if dd[j] <= MATCH_RADIUS_M else np.nan)
        par.append(tuple(sorted(set(oid[c].tolist()))))
    out = sk_pts.copy()
    out["osm_idx"] = near; out["dist_m"] = dist; out["parallel_ids"] = par
    return out


# производные признаки (строятся в catboost_baseline.attach_skdf_v2_features, нужен OSM-тег lanes)
SKDF_V2_DERIVED_FEATURES = ["skdf_has_passport", "lanes_skdf_or_osm"]
SKDF_V2_ALL_FEATURES = [*SKDF_V2_FEATURES, *SKDF_V2_DERIVED_FEATURES]
