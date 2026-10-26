"""Этап 2 плана docs/analysis/features/PLAN.md: перекрёстки с честной топологией.

Строит из osm_roads_raw.parquet (+ светофоры из osm_road_point_nodes.parquet):
    parquet_output/osm_junction_v2_nodes.parquet   — узлы с истинной степенью (полная и уличная сеть)
    parquet_output/h3_junction_v2_features.parquet — признаки ячейки (JUNCTION_V2_FEATURES)
Старые parquet (osm_road_graph_nodes, h3_road_features, панель) не трогаются.

Запуск: PYTHONPATH=src venv/bin/python scripts/build_junction_v2_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.osm_road_features import (  # noqa: E402
    JUNCTION_V2_FEATURES, aggregate_h3_junction_v2_features, build_junction_v2_node_stats,
)
from road_accident_risk.paths import (  # noqa: E402
    H3_JUNCTION_V2_FEATURES_PARQUET, H3_ROAD_FEATURES_PARQUET, OSM_JUNCTION_V2_NODES_PARQUET,
    OSM_ROAD_POINT_NODES_PARQUET, OSM_ROADS_RAW_PARQUET,
)


def main() -> None:
    t0 = time.time()
    roads = pd.read_parquet(OSM_ROADS_RAW_PARQUET)
    points = pd.read_parquet(OSM_ROAD_POINT_NODES_PARQUET)
    sig = points[points["is_traffic_signal"]]
    signal_points = list(zip(sig["lat"].astype(float), sig["lng"].astype(float)))
    nodes = build_junction_v2_node_stats(roads, signal_points)
    nodes.to_parquet(OSM_JUNCTION_V2_NODES_PARQUET, index=False)
    cells = pd.read_parquet(H3_ROAD_FEATURES_PARQUET, columns=["h3"])["h3"]
    feats = aggregate_h3_junction_v2_features(nodes, cells=cells)
    feats.to_parquet(H3_JUNCTION_V2_FEATURES_PARQUET, index=False)
    print(f"ways: {len(roads):,}; светофоров: {len(signal_points)}")
    print(f"узлов (>=2 ways или >=3 подходов): {len(nodes):,} -> {OSM_JUNCTION_V2_NODES_PARQUET}")
    print(f"уличных перекрёстков: {int(nodes.is_street_junction.sum()):,} "
          f"(3-leg {int(nodes.is_street_3leg.sum()):,}, 4+-leg {int(nodes.is_street_4leg.sum()):,}); "
          f"въездов: {int(nodes.is_driveway.sum()):,}")
    print(f"ячеек: {len(feats):,} -> {H3_JUNCTION_V2_FEATURES_PARQUET}")
    print(feats[JUNCTION_V2_FEATURES].describe().T.to_string())
    print(f"{time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
