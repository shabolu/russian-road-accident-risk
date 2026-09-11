from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root.

    ROAD_ACCIDENT_RISK_ROOT can override this when running the package from a
    different working directory.
    """
    override = os.environ.get("ROAD_ACCIDENT_RISK_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
DATASET_JSON = PROJECT_ROOT / "dataset.json"
DATASET_SCHEMA_JSON = PROJECT_ROOT / "dataset_schema.json"
PARQUET_DIR = PROJECT_ROOT / "parquet_output"
DATA_DIR = PROJECT_ROOT / "data"
RAW_OSM_DIR = DATA_DIR / "raw" / "osm"

CRASHES_PARQUET = PARQUET_DIR / "crashes.parquet"
VEHICLES_PARQUET = PARQUET_DIR / "vehicles.parquet"
PERSONS_PARQUET = PARQUET_DIR / "persons.parquet"
CRASHES_SLIM_PARQUET = PARQUET_DIR / "crashes_slim.parquet"
H3_MONTH_PANEL_PARQUET = PARQUET_DIR / "h3_month_panel.parquet"
CALENDAR_BASELINE_PARQUET = PARQUET_DIR / "calendar_baseline.parquet"
OSM_SARATOV_ROADS_JSON = RAW_OSM_DIR / "saratov_roads_overpass.json"
OSM_VOLGA_PBF = RAW_OSM_DIR / "volga-fed-district-260603.osm.pbf"
OSM_ROADS_RAW_PARQUET = PARQUET_DIR / "osm_roads_raw.parquet"
OSM_ROAD_POINT_NODES_PARQUET = PARQUET_DIR / "osm_road_point_nodes.parquet"
OSM_ROAD_H3_SEGMENTS_PARQUET = PARQUET_DIR / "osm_road_h3_segments.parquet"
OSM_ROAD_GRAPH_NODES_PARQUET = PARQUET_DIR / "osm_road_graph_nodes.parquet"
H3_ROAD_FEATURES_PARQUET = PARQUET_DIR / "h3_road_features.parquet"
H3_MONTH_ROAD_FEATURES_PARQUET = PARQUET_DIR / "h3_month_road_features.parquet"


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
