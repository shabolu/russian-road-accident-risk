"""Этап 6.1: таблица «ячейка панели × величины новых слоёв OSM» для EDA (без признаков и моделей).

Ячейки — H3 res 9 панели с дорогами (road_length_total_m > 0), bbox Саратова.
Выход: docs/analysis/features/stage6_exposure/osm_eda/cells_osm_v6_eda.csv
(описательные величины; в модель не идут).

    PYTHONPATH=src venv/bin/python scripts/eda_osm_v6_cells.py
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, H3_TAGS_V2_FEATURES_PARQUET, PARQUET_DIR, PROJECT_ROOT
from road_accident_risk.street_graph_v2 import utm38n_forward

OUT_DIR = PROJECT_ROOT / "docs/analysis/features/stage6_exposure/osm_eda"
RES = 9

# ВРЕМЕННЫЕ точки центра (для EDA; выбор центра — пункт 6.2).
# Координаты — из узлов OSM в выборке (см. LOG §3): Театральная площадь Саратова и узел place=city Энгельса.
SARATOV_CENTER = (51.53338, 46.03414)  # node 3620668550 place=neighbourhood «Театральная площадь»
ENGELS_CENTER = (51.50138, 46.12331)  # node 101480194 place=city «Энгельс»

HOUSE_TYPES = {"house", "detached", "semidetached_house", "bungalow", "terrace", "cabin"}
APART_TYPES = {"apartments", "residential", "dormitory"}
COMMERCIAL_TYPES = {"commercial", "retail", "office", "supermarket", "mall", "kiosk"}
NONRES_TYPES = {"garage", "garages", "shed", "hut", "roof", "carport", "greenhouse"}
ATTR_AMENITY = {"restaurant", "cafe", "fast_food", "pharmacy", "bank", "hospital", "clinic",
                "marketplace", "bar", "pub"}
SCHOOL_AMENITY = {"school", "kindergarten"}
NIGHT_AMENITY = {"bar", "pub", "nightclub"}
STOP_DEDUP_M = 30.0


def dist_m(lat, lon, c):
    x, y = utm38n_forward(np.asarray(lat), np.asarray(lon))
    cx, cy = utm38n_forward(np.array([c[0]]), np.array([c[1]]))
    return np.hypot(x - cx[0], y - cy[0])


def dedup_points(lat: np.ndarray, lon: np.ndarray, radius_m: float = STOP_DEDUP_M) -> np.ndarray:
    """Жадная кластеризация: single linkage в радиусе r (KD-дерево). Возвращает метку кластера."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    x, y = utm38n_forward(lat, lon)
    tree = cKDTree(np.column_stack([x, y]))
    pairs = tree.query_pairs(radius_m, output_type="ndarray")
    import scipy.sparse as sp
    n = len(lat)
    g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else sp.coo_matrix((n, n))
    return connected_components(g, directed=False)[1]


def poi_classes(p: pd.DataFrame) -> pd.DataFrame:
    am = p["amenity"].fillna("")
    p = p.copy()
    p["is_attractor"] = p["shop"].notna() | p["office"].notna() | am.isin(ATTR_AMENITY)
    p["is_school"] = am.isin(SCHOOL_AMENITY)
    p["is_health"] = p["healthcare"].notna() | am.isin({"hospital", "clinic", "doctors", "dentist", "pharmacy"})
    p["is_night"] = am.isin(NIGHT_AMENITY)
    p["has_name"] = p["name"].notna()
    return p


def main() -> None:
    panel = pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET,
                            columns=["h3", "month", "target_had_accident_next_month", "road_length_total_m"])
    panel = panel[panel.road_length_total_m > 0]
    cells = panel.groupby("h3").agg(dtp_share=("target_had_accident_next_month", "mean"),
                                    n_months=("month", "size")).reset_index()
    ll = np.array([h3.cell_to_latlng(c) for c in cells.h3])
    cells["lat"], cells["lon"] = ll[:, 0], ll[:, 1]
    cells["dist_center_m"] = dist_m(cells.lat, cells.lon, SARATOV_CENTER)
    cells["dist_engels_center_m"] = dist_m(cells.lat, cells.lon, ENGELS_CENTER)
    tags = pd.read_parquet(H3_TAGS_V2_FEATURES_PARQUET,
                           columns=["h3", "street_length_m", "major_road_length_m", "surface_tagged_share"])
    cells = cells.merge(tags, on="h3", how="left")

    # здания
    b = pd.read_parquet(PARQUET_DIR / "osm_v6_buildings.parquet",
                        columns=["lat", "lon", "in_bbox", "region", "building", "levels", "footprint_m2", "flats"])
    b = b[b.in_bbox]
    b["h3"] = [h3.latlng_to_cell(a, c, RES) for a, c in zip(b.lat, b.lon)]
    b["has_levels"] = b.levels.notna()
    b["typed"] = b.building != "yes"
    b["is_house"] = b.building.isin(HOUSE_TYPES)
    b["is_apart"] = b.building.isin(APART_TYPES)
    b["is_comm"] = b.building.isin(COMMERCIAL_TYPES)
    b["small"] = b.footprint_m2 < 200
    b["fp_levels"] = b.footprint_m2.where(b.has_levels, 0)
    b["floor_m2"] = (b.footprint_m2 * b.levels).fillna(0)
    gb = b.groupby("h3").agg(
        bld_n=("lat", "size"), bld_fp_m2=("footprint_m2", "sum"), bld_fp_median_m2=("footprint_m2", "median"),
        bld_levels_n=("has_levels", "sum"), bld_typed_n=("typed", "sum"), bld_house_n=("is_house", "sum"),
        bld_apart_n=("is_apart", "sum"), bld_comm_n=("is_comm", "sum"), bld_small_n=("small", "sum"),
        bld_fp_with_levels_m2=("fp_levels", "sum"), bld_floor_known_m2=("floor_m2", "sum"),
        bld_levels_median=("levels", "median"),
    ).reset_index()
    comm = b[b.is_comm].groupby("h3").agg(comm_fp_m2=("footprint_m2", "sum"),
                                          comm_floor_known_m2=("floor_m2", "sum")).reset_index()
    cells = cells.merge(gb, on="h3", how="left").merge(comm, on="h3", how="left")

    # POI
    p = poi_classes(pd.read_parquet(PARQUET_DIR / "osm_v6_poi.parquet"))
    p = p[p.in_bbox]
    p["h3"] = [h3.latlng_to_cell(a, c, RES) for a, c in zip(p.lat, p.lon)]
    gp = p.groupby("h3").agg(poi_n=("lat", "size"), poi_attr_n=("is_attractor", "sum"),
                             poi_school_n=("is_school", "sum"), poi_health_n=("is_health", "sum"),
                             poi_night_n=("is_night", "sum"), poi_named_n=("has_name", "sum")).reset_index()
    cells = cells.merge(gp, on="h3", how="left")

    # остановки после дедупа 30 м
    s = pd.read_parquet(PARQUET_DIR / "osm_v6_pt_stops.parquet")
    s = s[s.region == "saratov_buffer"].reset_index(drop=True)
    s["cluster"] = dedup_points(s.lat.values, s.lon.values)
    sc = s.groupby("cluster").agg(lat=("lat", "mean"), lon=("lon", "mean"))
    sc = sc[(sc.lat.between(51.40, 51.70)) & (sc.lon.between(45.80, 46.20))]
    sc["h3"] = [h3.latlng_to_cell(a, c, RES) for a, c in zip(sc.lat, sc.lon)]
    cells = cells.merge(sc.groupby("h3").size().rename("stop_n").reset_index(), on="h3", how="left")
    from scipy.spatial import cKDTree
    sx, sy = utm38n_forward(sc.lat.values, sc.lon.values)
    cx, cy = utm38n_forward(cells.lat.values, cells.lon.values)
    cells["dist_stop_m"] = cKDTree(np.column_stack([sx, sy])).query(np.column_stack([cx, cy]))[0]

    cnt_cols = [c for c in cells.columns if c.endswith("_n") or c.endswith("_m2") and c != "bld_fp_median_m2"]
    for c in cnt_cols:
        if c != "bld_fp_median_m2":
            cells[c] = cells[c].fillna(0)

    cells["bld_levels_share"] = cells.bld_levels_n / cells.bld_n.replace(0, np.nan)
    cells["bld_levels_area_share"] = cells.bld_fp_with_levels_m2 / cells.bld_fp_m2.replace(0, np.nan)
    cells["bld_typed_share"] = cells.bld_typed_n / cells.bld_n.replace(0, np.nan)
    cells["bld_house_share"] = cells.bld_house_n / cells.bld_n.replace(0, np.nan)
    cells["bld_small_share"] = cells.bld_small_n / cells.bld_n.replace(0, np.nan)
    cells["poi_per_km_street"] = cells.poi_n / (cells.street_length_m / 1000).replace(0, np.nan)
    cells["poi_named_share"] = cells.poi_named_n / cells.poi_n.replace(0, np.nan)
    cell_area_m2 = h3.average_hexagon_area(RES, unit="m^2")
    cells["bld_footprint_share"] = cells.bld_fp_m2 / cell_area_m2

    # районы (воспроизводимое правило, только геометрия контуров — не теги типа/этажности)
    engels = (cells.dist_engels_center_m < cells.dist_center_m) & (cells.lon > 46.07)
    d = np.full(len(cells), "смешанная застройка", dtype=object)
    d[(cells.bld_n >= 10) & (cells.bld_small_share >= 0.7)] = "частный сектор"
    d[(cells.bld_n >= 5) & (cells.bld_fp_median_m2 >= 400)] = "многоэтажная / крупная застройка"
    d[(cells.bld_n < 5)] = "малозастроенные окраины"
    d[(cells.dist_center_m > 12000) & ~engels] = "малозастроенные окраины"
    d[engels] = "Энгельс"
    d[cells.dist_center_m <= 2500] = "центр Саратова"
    cells["district"] = d
    cells.to_csv(OUT_DIR / "cells_osm_v6_eda.csv", index=False)
    print(cells.district.value_counts())
    print(cells.describe().T.to_string())


if __name__ == "__main__":
    main()
