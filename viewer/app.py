from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import duckdb
import h3
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "viewer" / "static"
PANEL_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_panel.parquet"
ROAD_FEATURES_PATH = PROJECT_ROOT / "parquet_output" / "h3_road_features.parquet"
ROAD_GRAPH_NODES_PATH = PROJECT_ROOT / "parquet_output" / "osm_road_graph_nodes.parquet"
ROAD_H3_SEGMENTS_PATH = PROJECT_ROOT / "parquet_output" / "osm_road_h3_segments.parquet"
ROAD_RAW_PATH = PROJECT_ROOT / "parquet_output" / "osm_roads_raw.parquet"
LOGREG_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_logreg_predictions.parquet"
CATBOOST_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_predictions.parquet"
CATBOOST_NO_CALENDAR_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_no_calendar_predictions.parquet"
CATBOOST_ROAD_ONLY_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_predictions.parquet"
CATBOOST_ROAD_ONLY_2_0_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_2_0_predictions.parquet"
CATBOOST_ROAD_ONLY_2_5_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_2_5_predictions.parquet"
CATBOOST_ROAD_ONLY_3_0_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_3_0_predictions.parquet"
CATBOOST_ROAD_ONLY_3_5_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_3_5_predictions.parquet"
CATBOOST_ROAD_ONLY_4_0_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_predictions.parquet"
CATBOOST_ROAD_ONLY_4_0_MIN_2000_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_min_2000_predictions.parquet"
CATBOOST_ROAD_ONLY_4_1_PREDICTIONS_PATH = PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_1_log_predictions.parquet"


CATBOOST_ROAD_ONLY_4_0_TARGET_LAYER_CONFIGS = [
    {
        "key": "count_next_3m",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_count_next_3m_predictions.parquet",
        "score_column": "catboost_road_only_4_0_count_next_3m_score",
        "percentile_column": "catboost_road_only_4_0_count_next_3m_score_percentile",
        "rank_column": "catboost_road_only_4_0_count_next_3m_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_count_next_3m_score_latest",
        "latest_month": "catboost_road_only_4_0_count_next_3m_score_month",
    },
    {
        "key": "count_next_6m",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_count_next_6m_predictions.parquet",
        "score_column": "catboost_road_only_4_0_count_next_6m_score",
        "percentile_column": "catboost_road_only_4_0_count_next_6m_score_percentile",
        "rank_column": "catboost_road_only_4_0_count_next_6m_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_count_next_6m_score_latest",
        "latest_month": "catboost_road_only_4_0_count_next_6m_score_month",
    },
    {
        "key": "binary_next_3m",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_binary_next_3m_predictions.parquet",
        "score_column": "catboost_road_only_4_0_binary_next_3m_score",
        "percentile_column": "catboost_road_only_4_0_binary_next_3m_score_percentile",
        "rank_column": "catboost_road_only_4_0_binary_next_3m_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_binary_next_3m_score_latest",
        "latest_month": "catboost_road_only_4_0_binary_next_3m_score_month",
    },
    {
        "key": "binary_next_6m_weight_none",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_binary_next_6m_weight_none_predictions.parquet",
        "score_column": "catboost_road_only_4_0_binary_next_6m_weight_none_score",
        "percentile_column": "catboost_road_only_4_0_binary_next_6m_weight_none_score_percentile",
        "rank_column": "catboost_road_only_4_0_binary_next_6m_weight_none_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_binary_next_6m_weight_none_score_latest",
        "latest_month": "catboost_road_only_4_0_binary_next_6m_weight_none_score_month",
    },
    {
        "key": "binary_next_6m_xgboost",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_road_only_4_0_binary_next_6m_xgboost_predictions.parquet",
        "score_column": "road_only_4_0_binary_next_6m_xgboost_score",
        "percentile_column": "road_only_4_0_binary_next_6m_xgboost_score_percentile",
        "rank_column": "road_only_4_0_binary_next_6m_xgboost_rank_in_month",
        "latest_metric": "road_only_4_0_binary_next_6m_xgboost_score_latest",
        "latest_month": "road_only_4_0_binary_next_6m_xgboost_score_month",
    },
    {
        "key": "binary_next_6m_lightgbm",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_road_only_4_0_binary_next_6m_lightgbm_predictions.parquet",
        "score_column": "road_only_4_0_binary_next_6m_lightgbm_score",
        "percentile_column": "road_only_4_0_binary_next_6m_lightgbm_score_percentile",
        "rank_column": "road_only_4_0_binary_next_6m_lightgbm_rank_in_month",
        "latest_metric": "road_only_4_0_binary_next_6m_lightgbm_score_latest",
        "latest_month": "road_only_4_0_binary_next_6m_lightgbm_score_month",
    },
    {
        "key": "binary_next_6m_hist_gradient_boosting",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_road_only_4_0_binary_next_6m_hist_gradient_boosting_predictions.parquet",
        "score_column": "road_only_4_0_binary_next_6m_hist_gradient_boosting_score",
        "percentile_column": "road_only_4_0_binary_next_6m_hist_gradient_boosting_score_percentile",
        "rank_column": "road_only_4_0_binary_next_6m_hist_gradient_boosting_rank_in_month",
        "latest_metric": "road_only_4_0_binary_next_6m_hist_gradient_boosting_score_latest",
        "latest_month": "road_only_4_0_binary_next_6m_hist_gradient_boosting_score_month",
    },
    {
        "key": "binary_next_6m_random_forest",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_road_only_4_0_binary_next_6m_random_forest_predictions.parquet",
        "score_column": "road_only_4_0_binary_next_6m_random_forest_score",
        "percentile_column": "road_only_4_0_binary_next_6m_random_forest_score_percentile",
        "rank_column": "road_only_4_0_binary_next_6m_random_forest_rank_in_month",
        "latest_metric": "road_only_4_0_binary_next_6m_random_forest_score_latest",
        "latest_month": "road_only_4_0_binary_next_6m_random_forest_score_month",
    },
    {
        "key": "binary_next_6m_weight_2",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_binary_next_6m_weight_2_predictions.parquet",
        "score_column": "catboost_road_only_4_0_binary_next_6m_weight_2_score",
        "percentile_column": "catboost_road_only_4_0_binary_next_6m_weight_2_score_percentile",
        "rank_column": "catboost_road_only_4_0_binary_next_6m_weight_2_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_binary_next_6m_weight_2_score_latest",
        "latest_month": "catboost_road_only_4_0_binary_next_6m_weight_2_score_month",
    },
    {
        "key": "binary_next_6m_weight_4",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_binary_next_6m_weight_4_predictions.parquet",
        "score_column": "catboost_road_only_4_0_binary_next_6m_weight_4_score",
        "percentile_column": "catboost_road_only_4_0_binary_next_6m_weight_4_score_percentile",
        "rank_column": "catboost_road_only_4_0_binary_next_6m_weight_4_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_binary_next_6m_weight_4_score_latest",
        "latest_month": "catboost_road_only_4_0_binary_next_6m_weight_4_score_month",
    },
    {
        "key": "binary_next_6m_weight_6",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_binary_next_6m_weight_6_predictions.parquet",
        "score_column": "catboost_road_only_4_0_binary_next_6m_weight_6_score",
        "percentile_column": "catboost_road_only_4_0_binary_next_6m_weight_6_score_percentile",
        "rank_column": "catboost_road_only_4_0_binary_next_6m_weight_6_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_binary_next_6m_weight_6_score_latest",
        "latest_month": "catboost_road_only_4_0_binary_next_6m_weight_6_score_month",
    },
    {
        "key": "binary_next_6m_weight_balanced",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_binary_next_6m_weight_balanced_predictions.parquet",
        "score_column": "catboost_road_only_4_0_binary_next_6m_weight_balanced_score",
        "percentile_column": "catboost_road_only_4_0_binary_next_6m_weight_balanced_score_percentile",
        "rank_column": "catboost_road_only_4_0_binary_next_6m_weight_balanced_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_binary_next_6m_weight_balanced_score_latest",
        "latest_month": "catboost_road_only_4_0_binary_next_6m_weight_balanced_score_month",
    },
    {
        "key": "grade2",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_grade2_predictions.parquet",
        "score_column": "catboost_road_only_4_0_grade2_score",
        "percentile_column": "catboost_road_only_4_0_grade2_score_percentile",
        "rank_column": "catboost_road_only_4_0_grade2_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_grade2_score_latest",
        "latest_month": "catboost_road_only_4_0_grade2_score_month",
    },
    {
        "key": "grade2_delta",
        "path": PROJECT_ROOT / "parquet_output" / "h3_month_catboost_road_only_4_0_grade2_delta_predictions.parquet",
        "score_column": "catboost_road_only_4_0_grade2_delta_score",
        "percentile_column": "catboost_road_only_4_0_grade2_delta_score_percentile",
        "rank_column": "catboost_road_only_4_0_grade2_delta_rank_in_month",
        "latest_metric": "catboost_road_only_4_0_grade2_delta_score_latest",
        "latest_month": "catboost_road_only_4_0_grade2_delta_score_month",
    },
]

ROAD_FEATURE_COLUMNS = [
    "road_length_total_m",
    "road_way_count",
    "road_segment_h3_count",
    "major_road_length_m",
    "link_road_length_m",
    "oneway_length_m",
    "lit_road_length_m",
    "bridge_length_m",
    "tunnel_length_m",
    "bad_surface_length_m",
    "asphalt_surface_length_m",
    "major_road_share",
    "link_road_share",
    "oneway_share",
    "lit_share",
    "bridge_share",
    "tunnel_share",
    "bad_surface_share",
    "road_share_motorway",
    "road_share_motorway_link",
    "road_share_trunk",
    "road_share_trunk_link",
    "road_share_primary",
    "road_share_primary_link",
    "road_share_secondary",
    "road_share_secondary_link",
    "road_share_tertiary",
    "road_share_tertiary_link",
    "road_share_unclassified",
    "road_share_residential",
    "road_share_living_street",
    "road_share_service",
    "maxspeed_mean_weighted",
    "maxspeed_max",
    "maxspeed_tagged_share",
    "maxspeed_is_imputed_share",
    "lanes_mean_weighted",
    "lanes_max",
    "lane_meters_estimate",
    "lanes_tagged_share",
    "junction_density_per_km",
    "minor_to_major_junction_density_per_km",
    "junction_count",
    "junction_degree_mean",
    "junction_degree_max",
    "three_way_junction_count",
    "four_plus_way_junction_count",
    "major_junction_count",
    "minor_to_major_junction_count",
    "legacy_major_minor_junction_count",
    "class_delta_junction_count",
    "class_delta_mean",
    "class_delta_max",
    "class_delta_sum",
    "signalized_class_delta_junction_count",
    "unsignalized_class_delta_junction_count",
    "unsignalized_high_delta_junction_count",
    "class_delta_junction_density_per_km",
    "unsignalized_class_delta_junction_density_per_km",
    "traffic_signal_count",
    "traffic_signal_density_per_km",
    "crossing_count",
    "crossing_signalized_count",
    "crossing_signalized_density_per_km",
    "crossing_unsignalized_count",
    "crossing_density_per_km",
    "crossing_signalized_share",
    "crossing_unsignalized_share",
    "crossing_unsignalized_density_per_km",
    "stop_count",
    "stop_density_per_km",
    "give_way_count",
    "give_way_density_per_km",
    "priority_sign_count",
    "priority_sign_density_per_km",
    "stop_share",
    "give_way_share",
    "traffic_calming_count",
    "traffic_calming_density_per_km",
    "public_transport_stop_count",
    "public_transport_stop_density_per_km",
    "bus_stop_count",
    "tram_stop_count",
    "bus_stop_share",
    "tram_stop_share",
    "railway_crossing_count",
    "railway_crossing_density_per_km",
    "mini_roundabout_density_per_km",
    "roundabout_total_count",
    "roundabout_total_density_per_km",
    "mini_roundabout_share",
    "full_roundabout_share",
    "roundabout_count",
    "roundabout_density_per_km",
    "motorway_junction_density_per_km",
    "road_curvature_mean_weighted",
    "road_curvature_max",
    "curvature_deg_per_km",
    "sharp_turn_count",
    "sharp_turn_density_per_km",
    "road_segment_length_median",
    "short_segment_share",
    "road_orientation_entropy",
    "turn_event_count",
    "road_segment_count",
    "road_segment_length_mean",
    "short_segment_count",
    "dead_end_count",
    "turn_event_density_per_km",
    "sharp_turn_share",
    "road_segment_density_per_km",
    "dead_end_density_per_km",
]


app = FastAPI(title="Road Accident H3 Viewer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:", read_only=False)


def parquet_sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def cell_boundary(cell: str) -> list[list[float]]:
    return [
        [round(float(lat), 6), round(float(lng), 6)]
        for lat, lng in h3.cell_to_boundary(cell)
    ]


def cell_center(cell: str) -> tuple[float, float]:
    lat, lng = h3.cell_to_latlng(cell)
    return round(float(lat), 6), round(float(lng), 6)


def rows_to_dicts(cursor: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def interpolate(lat1: float, lng1: float, lat2: float, lng2: float, fraction: float) -> tuple[float, float]:
    return lat1 + (lat2 - lat1) * fraction, lng1 + (lng2 - lng1) * fraction


def road_parts_in_cell(
    coords: list[list[float]],
    cell: str,
    sample_step_m: float = 12.0,
) -> list[dict[str, Any]]:
    resolution = h3.get_resolution(cell)
    parts: list[dict[str, Any]] = []
    current_coords: list[list[float]] = []
    current_length = 0.0

    for point1, point2 in zip(coords, coords[1:]):
        lat1, lng1 = float(point1[0]), float(point1[1])
        lat2, lng2 = float(point2[0]), float(point2[1])
        segment_len = haversine_m(lat1, lng1, lat2, lng2)
        if segment_len <= 0:
            continue

        pieces = max(1, math.ceil(segment_len / sample_step_m))
        piece_len = segment_len / pieces
        for i in range(pieces):
            start_fraction = i / pieces
            end_fraction = (i + 1) / pieces
            mid_fraction = (i + 0.5) / pieces
            start = interpolate(lat1, lng1, lat2, lng2, start_fraction)
            end = interpolate(lat1, lng1, lat2, lng2, end_fraction)
            mid = interpolate(lat1, lng1, lat2, lng2, mid_fraction)
            in_cell = h3.latlng_to_cell(mid[0], mid[1], resolution) == cell

            if in_cell:
                start_pair = [round(start[0], 7), round(start[1], 7)]
                end_pair = [round(end[0], 7), round(end[1], 7)]
                if not current_coords:
                    current_coords.append(start_pair)
                current_coords.append(end_pair)
                current_length += piece_len
            elif current_coords:
                parts.append({"coords": current_coords, "length_m": current_length})
                current_coords = []
                current_length = 0.0

    if current_coords:
        parts.append({"coords": current_coords, "length_m": current_length})

    return parts


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    if not PANEL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Missing panel parquet: {PANEL_PATH}")

    with connect() as con:
        panel = parquet_sql_path(PANEL_PATH)
        row = con.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT h3) AS h3_count,
                MIN(month)::VARCHAR AS min_month,
                MAX(month)::VARCHAR AS max_month,
                SUM(accident_count)::BIGINT AS total_accidents,
                SUM(CASE WHEN target_had_accident_next_month THEN 1 ELSE 0 END)::BIGINT AS target_positive_rows
            FROM parquet_scan('{panel}')
            """
        ).fetchone()

    return {
        "panel": str(PANEL_PATH),
        "road_features": str(ROAD_FEATURES_PATH),
        "logreg_predictions": str(LOGREG_PREDICTIONS_PATH) if LOGREG_PREDICTIONS_PATH.exists() else None,
        "catboost_predictions": str(CATBOOST_PREDICTIONS_PATH) if CATBOOST_PREDICTIONS_PATH.exists() else None,
        "catboost_no_calendar_predictions": (
            str(CATBOOST_NO_CALENDAR_PREDICTIONS_PATH)
            if CATBOOST_NO_CALENDAR_PREDICTIONS_PATH.exists()
            else None
        ),
        "catboost_road_only_predictions": (
            str(CATBOOST_ROAD_ONLY_PREDICTIONS_PATH)
            if CATBOOST_ROAD_ONLY_PREDICTIONS_PATH.exists()
            else None
        ),
        "catboost_road_only_2_0_predictions": (
            str(CATBOOST_ROAD_ONLY_2_0_PREDICTIONS_PATH)
            if CATBOOST_ROAD_ONLY_2_0_PREDICTIONS_PATH.exists()
            else None
        ),
        "catboost_road_only_2_5_predictions": (
            str(CATBOOST_ROAD_ONLY_2_5_PREDICTIONS_PATH)
            if CATBOOST_ROAD_ONLY_2_5_PREDICTIONS_PATH.exists()
            else None
        ),
        "catboost_road_only_3_0_predictions": (
            str(CATBOOST_ROAD_ONLY_3_0_PREDICTIONS_PATH)
            if CATBOOST_ROAD_ONLY_3_0_PREDICTIONS_PATH.exists()
            else None
        ),
        "catboost_road_only_3_5_predictions": (
            str(CATBOOST_ROAD_ONLY_3_5_PREDICTIONS_PATH)
            if CATBOOST_ROAD_ONLY_3_5_PREDICTIONS_PATH.exists()
            else None
        ),
        "catboost_road_only_4_0_predictions": (
            str(CATBOOST_ROAD_ONLY_4_0_PREDICTIONS_PATH)
            if CATBOOST_ROAD_ONLY_4_0_PREDICTIONS_PATH.exists()
            else None
        ),
        "catboost_road_only_4_0_min_2000_predictions": (
            str(CATBOOST_ROAD_ONLY_4_0_MIN_2000_PREDICTIONS_PATH)
            if CATBOOST_ROAD_ONLY_4_0_MIN_2000_PREDICTIONS_PATH.exists()
            else None
        ),
        "catboost_road_only_4_1_log_predictions": (
            str(CATBOOST_ROAD_ONLY_4_1_PREDICTIONS_PATH)
            if CATBOOST_ROAD_ONLY_4_1_PREDICTIONS_PATH.exists()
            else None
        ),
        "catboost_road_only_4_0_target_predictions": {
            config["key"]: str(config["path"]) if config["path"].exists() else None
            for config in CATBOOST_ROAD_ONLY_4_0_TARGET_LAYER_CONFIGS
        },
        "rows": int(row[0]),
        "h3_count": int(row[1]),
        "min_month": row[2][:10],
        "max_month": row[3][:10],
        "total_accidents": int(row[4]),
        "target_positive_rows": int(row[5]),
    }


@app.get("/api/cells")
def cells(metric: str = "total_accidents") -> dict[str, Any]:
    target_layer_by_metric = {
        config["latest_metric"]: config
        for config in CATBOOST_ROAD_ONLY_4_0_TARGET_LAYER_CONFIGS
    }
    supported_metrics = {
        "total_accidents",
        "road_length_total_m",
        "junction_count",
        "crossing_count",
        "logreg_score_latest",
        "catboost_score_latest",
        "catboost_no_calendar_score_latest",
        "catboost_road_only_score_latest",
        "catboost_road_only_2_0_score_latest",
        "catboost_road_only_2_5_score_latest",
        "catboost_road_only_3_0_score_latest",
        "catboost_road_only_3_5_score_latest",
        "catboost_road_only_4_0_score_latest",
        "catboost_road_only_4_0_min_2000_score_latest",
        "catboost_road_only_4_1_log_score_latest",
        *target_layer_by_metric.keys(),
    }
    if metric not in supported_metrics:
        raise HTTPException(status_code=400, detail="Unsupported metric")
    if metric == "logreg_score_latest" and not LOGREG_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing logreg predictions parquet: {LOGREG_PREDICTIONS_PATH}",
        )
    if metric == "catboost_score_latest" and not CATBOOST_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost predictions parquet: {CATBOOST_PREDICTIONS_PATH}",
        )
    if metric == "catboost_no_calendar_score_latest" and not CATBOOST_NO_CALENDAR_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost no-calendar predictions parquet: {CATBOOST_NO_CALENDAR_PREDICTIONS_PATH}",
        )
    if metric == "catboost_road_only_score_latest" and not CATBOOST_ROAD_ONLY_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost road-only predictions parquet: {CATBOOST_ROAD_ONLY_PREDICTIONS_PATH}",
        )
    if metric == "catboost_road_only_2_0_score_latest" and not CATBOOST_ROAD_ONLY_2_0_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost road-only 2.0 predictions parquet: {CATBOOST_ROAD_ONLY_2_0_PREDICTIONS_PATH}",
        )
    if metric == "catboost_road_only_2_5_score_latest" and not CATBOOST_ROAD_ONLY_2_5_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost road-only 2.5 predictions parquet: {CATBOOST_ROAD_ONLY_2_5_PREDICTIONS_PATH}",
        )
    if metric == "catboost_road_only_3_0_score_latest" and not CATBOOST_ROAD_ONLY_3_0_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost road-only 3.0 predictions parquet: {CATBOOST_ROAD_ONLY_3_0_PREDICTIONS_PATH}",
        )
    if metric == "catboost_road_only_3_5_score_latest" and not CATBOOST_ROAD_ONLY_3_5_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost road-only 3.5 predictions parquet: {CATBOOST_ROAD_ONLY_3_5_PREDICTIONS_PATH}",
        )
    if metric == "catboost_road_only_4_0_score_latest" and not CATBOOST_ROAD_ONLY_4_0_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost road-only 4.0 predictions parquet: {CATBOOST_ROAD_ONLY_4_0_PREDICTIONS_PATH}",
        )
    if metric == "catboost_road_only_4_0_min_2000_score_latest" and not CATBOOST_ROAD_ONLY_4_0_MIN_2000_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost road-only 4.0 >2000m predictions parquet: {CATBOOST_ROAD_ONLY_4_0_MIN_2000_PREDICTIONS_PATH}",
        )
    if metric == "catboost_road_only_4_1_log_score_latest" and not CATBOOST_ROAD_ONLY_4_1_PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost road-only 4.1 log predictions parquet: {CATBOOST_ROAD_ONLY_4_1_PREDICTIONS_PATH}",
        )
    if metric in target_layer_by_metric and not target_layer_by_metric[metric]["path"].exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing CatBoost road-only 4.0 target predictions parquet: {target_layer_by_metric[metric]['path']}",
        )

    panel = parquet_sql_path(PANEL_PATH)
    roads = parquet_sql_path(ROAD_FEATURES_PATH)
    logreg_predictions = parquet_sql_path(LOGREG_PREDICTIONS_PATH)
    catboost_predictions = parquet_sql_path(CATBOOST_PREDICTIONS_PATH)
    catboost_no_calendar_predictions = parquet_sql_path(CATBOOST_NO_CALENDAR_PREDICTIONS_PATH)
    catboost_road_only_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_PREDICTIONS_PATH)
    catboost_road_only_2_0_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_2_0_PREDICTIONS_PATH)
    catboost_road_only_2_5_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_2_5_PREDICTIONS_PATH)
    catboost_road_only_3_0_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_3_0_PREDICTIONS_PATH)
    catboost_road_only_3_5_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_3_5_PREDICTIONS_PATH)
    catboost_road_only_4_0_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_4_0_PREDICTIONS_PATH)
    catboost_road_only_4_0_min_2000_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_4_0_MIN_2000_PREDICTIONS_PATH)
    catboost_road_only_4_1_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_4_1_PREDICTIONS_PATH)
    prediction_ctes: list[str] = []
    prediction_joins: list[str] = []
    prediction_selects = [
        "0.0 AS logreg_score_latest",
        "NULL::VARCHAR AS logreg_score_month",
        "0.0 AS catboost_score_latest",
        "NULL::VARCHAR AS catboost_score_month",
        "0.0 AS catboost_no_calendar_score_latest",
        "NULL::VARCHAR AS catboost_no_calendar_score_month",
        "0.0 AS catboost_road_only_score_latest",
        "NULL::VARCHAR AS catboost_road_only_score_month",
        "0.0 AS catboost_road_only_2_0_score_latest",
        "NULL::VARCHAR AS catboost_road_only_2_0_score_month",
        "0.0 AS catboost_road_only_2_5_score_latest",
        "NULL::VARCHAR AS catboost_road_only_2_5_score_month",
        "0.0 AS catboost_road_only_3_0_score_latest",
        "NULL::VARCHAR AS catboost_road_only_3_0_score_month",
        "0.0 AS catboost_road_only_3_5_score_latest",
        "NULL::VARCHAR AS catboost_road_only_3_5_score_month",
        "0.0 AS catboost_road_only_4_0_score_latest",
        "NULL::VARCHAR AS catboost_road_only_4_0_score_month",
        "0.0 AS catboost_road_only_4_0_min_2000_score_latest",
        "NULL::VARCHAR AS catboost_road_only_4_0_min_2000_score_month",
        "0.0 AS catboost_road_only_4_1_log_score_latest",
        "NULL::VARCHAR AS catboost_road_only_4_1_log_score_month",
    ]
    if LOGREG_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{logreg_predictions}')
            ),
            latest_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS logreg_score_month,
                    p.logreg_score AS logreg_score_latest
                FROM parquet_scan('{logreg_predictions}') p
                JOIN latest_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_predictions lp USING (h3)")
        prediction_selects[0] = "COALESCE(lp.logreg_score_latest, 0) AS logreg_score_latest"
        prediction_selects[1] = "lp.logreg_score_month AS logreg_score_month"
    if CATBOOST_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_predictions}')
            ),
            latest_catboost_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_score_month,
                    p.catboost_score AS catboost_score_latest
                FROM parquet_scan('{catboost_predictions}') p
                JOIN latest_catboost_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_predictions cp USING (h3)")
        prediction_selects[2] = "COALESCE(cp.catboost_score_latest, 0) AS catboost_score_latest"
        prediction_selects[3] = "cp.catboost_score_month AS catboost_score_month"
    if CATBOOST_NO_CALENDAR_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_no_calendar_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_no_calendar_predictions}')
            ),
            latest_catboost_no_calendar_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_no_calendar_score_month,
                    p.catboost_no_calendar_score AS catboost_no_calendar_score_latest
                FROM parquet_scan('{catboost_no_calendar_predictions}') p
                JOIN latest_catboost_no_calendar_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_no_calendar_predictions cnp USING (h3)")
        prediction_selects[4] = "COALESCE(cnp.catboost_no_calendar_score_latest, 0) AS catboost_no_calendar_score_latest"
        prediction_selects[5] = "cnp.catboost_no_calendar_score_month AS catboost_no_calendar_score_month"
    if CATBOOST_ROAD_ONLY_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_road_only_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_road_only_predictions}')
            ),
            latest_catboost_road_only_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_road_only_score_month,
                    p.catboost_road_only_score AS catboost_road_only_score_latest
                FROM parquet_scan('{catboost_road_only_predictions}') p
                JOIN latest_catboost_road_only_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_road_only_predictions crop USING (h3)")
        prediction_selects[6] = "COALESCE(crop.catboost_road_only_score_latest, 0) AS catboost_road_only_score_latest"
        prediction_selects[7] = "crop.catboost_road_only_score_month AS catboost_road_only_score_month"
    if CATBOOST_ROAD_ONLY_2_0_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_road_only_2_0_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_road_only_2_0_predictions}')
            ),
            latest_catboost_road_only_2_0_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_road_only_2_0_score_month,
                    p.catboost_road_only_2_0_score AS catboost_road_only_2_0_score_latest
                FROM parquet_scan('{catboost_road_only_2_0_predictions}') p
                JOIN latest_catboost_road_only_2_0_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_road_only_2_0_predictions cro2 USING (h3)")
        prediction_selects[8] = "COALESCE(cro2.catboost_road_only_2_0_score_latest, 0) AS catboost_road_only_2_0_score_latest"
        prediction_selects[9] = "cro2.catboost_road_only_2_0_score_month AS catboost_road_only_2_0_score_month"
    if CATBOOST_ROAD_ONLY_2_5_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_road_only_2_5_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_road_only_2_5_predictions}')
            ),
            latest_catboost_road_only_2_5_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_road_only_2_5_score_month,
                    p.catboost_road_only_2_5_score AS catboost_road_only_2_5_score_latest
                FROM parquet_scan('{catboost_road_only_2_5_predictions}') p
                JOIN latest_catboost_road_only_2_5_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_road_only_2_5_predictions cro25 USING (h3)")
        prediction_selects[10] = "COALESCE(cro25.catboost_road_only_2_5_score_latest, 0) AS catboost_road_only_2_5_score_latest"
        prediction_selects[11] = "cro25.catboost_road_only_2_5_score_month AS catboost_road_only_2_5_score_month"
    if CATBOOST_ROAD_ONLY_3_0_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_road_only_3_0_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_road_only_3_0_predictions}')
            ),
            latest_catboost_road_only_3_0_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_road_only_3_0_score_month,
                    p.catboost_road_only_3_0_score AS catboost_road_only_3_0_score_latest
                FROM parquet_scan('{catboost_road_only_3_0_predictions}') p
                JOIN latest_catboost_road_only_3_0_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_road_only_3_0_predictions cro30 USING (h3)")
        prediction_selects[12] = "COALESCE(cro30.catboost_road_only_3_0_score_latest, 0) AS catboost_road_only_3_0_score_latest"
        prediction_selects[13] = "cro30.catboost_road_only_3_0_score_month AS catboost_road_only_3_0_score_month"
    if CATBOOST_ROAD_ONLY_3_5_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_road_only_3_5_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_road_only_3_5_predictions}')
            ),
            latest_catboost_road_only_3_5_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_road_only_3_5_score_month,
                    p.catboost_road_only_3_5_score AS catboost_road_only_3_5_score_latest
                FROM parquet_scan('{catboost_road_only_3_5_predictions}') p
                JOIN latest_catboost_road_only_3_5_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_road_only_3_5_predictions cro35 USING (h3)")
        prediction_selects[14] = "COALESCE(cro35.catboost_road_only_3_5_score_latest, 0) AS catboost_road_only_3_5_score_latest"
        prediction_selects[15] = "cro35.catboost_road_only_3_5_score_month AS catboost_road_only_3_5_score_month"
    if CATBOOST_ROAD_ONLY_4_0_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_road_only_4_0_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_road_only_4_0_predictions}')
            ),
            latest_catboost_road_only_4_0_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_road_only_4_0_score_month,
                    p.catboost_road_only_4_0_score AS catboost_road_only_4_0_score_latest
                FROM parquet_scan('{catboost_road_only_4_0_predictions}') p
                JOIN latest_catboost_road_only_4_0_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_road_only_4_0_predictions cro40 USING (h3)")
        prediction_selects[16] = "COALESCE(cro40.catboost_road_only_4_0_score_latest, 0) AS catboost_road_only_4_0_score_latest"
        prediction_selects[17] = "cro40.catboost_road_only_4_0_score_month AS catboost_road_only_4_0_score_month"
    if CATBOOST_ROAD_ONLY_4_1_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_road_only_4_1_log_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_road_only_4_1_predictions}')
            ),
            latest_catboost_road_only_4_1_log_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_road_only_4_1_log_score_month,
                    p.catboost_road_only_4_1_log_score AS catboost_road_only_4_1_log_score_latest
                FROM parquet_scan('{catboost_road_only_4_1_predictions}') p
                JOIN latest_catboost_road_only_4_1_log_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_road_only_4_1_log_predictions cro41 USING (h3)")
        prediction_selects[20] = "COALESCE(cro41.catboost_road_only_4_1_log_score_latest, 0) AS catboost_road_only_4_1_log_score_latest"
        prediction_selects[21] = "cro41.catboost_road_only_4_1_log_score_month AS catboost_road_only_4_1_log_score_month"
    if CATBOOST_ROAD_ONLY_4_0_MIN_2000_PREDICTIONS_PATH.exists():
        prediction_ctes.append(f"""
            latest_catboost_road_only_4_0_min_2000_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{catboost_road_only_4_0_min_2000_predictions}')
            ),
            latest_catboost_road_only_4_0_min_2000_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS catboost_road_only_4_0_min_2000_score_month,
                    p.catboost_road_only_4_0_min_2000_score AS catboost_road_only_4_0_min_2000_score_latest
                FROM parquet_scan('{catboost_road_only_4_0_min_2000_predictions}') p
                JOIN latest_catboost_road_only_4_0_min_2000_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append("LEFT JOIN latest_catboost_road_only_4_0_min_2000_predictions cro402k USING (h3)")
        prediction_selects[18] = "COALESCE(cro402k.catboost_road_only_4_0_min_2000_score_latest, 0) AS catboost_road_only_4_0_min_2000_score_latest"
        prediction_selects[19] = "cro402k.catboost_road_only_4_0_min_2000_score_month AS catboost_road_only_4_0_min_2000_score_month"

    for config in CATBOOST_ROAD_ONLY_4_0_TARGET_LAYER_CONFIGS:
        latest_metric = config["latest_metric"]
        latest_month = config["latest_month"]
        prediction_selects.append(f"0.0 AS {latest_metric}")
        prediction_selects.append(f"NULL::VARCHAR AS {latest_month}")
        if not config["path"].exists():
            continue

        key = config["key"]
        alias = f"target_{key}"
        prediction_path = parquet_sql_path(config["path"])
        prediction_ctes.append(f"""
            latest_{key}_prediction_month AS (
                SELECT MAX(month) AS month
                FROM parquet_scan('{prediction_path}')
            ),
            latest_{key}_predictions AS (
                SELECT
                    p.h3,
                    p.month::VARCHAR AS {latest_month},
                    p.{config["score_column"]} AS {latest_metric}
                FROM parquet_scan('{prediction_path}') p
                JOIN latest_{key}_prediction_month m ON p.month = m.month
            )
        """)
        prediction_joins.append(f"LEFT JOIN latest_{key}_predictions {alias} USING (h3)")
        prediction_selects[-2] = f"COALESCE({alias}.{latest_metric}, 0) AS {latest_metric}"
        prediction_selects[-1] = f"{alias}.{latest_month} AS {latest_month}"

    prediction_cte_sql = ""
    if prediction_ctes:
        prediction_cte_sql = ",\n" + ",\n".join(prediction_ctes)
    prediction_join_sql = "\n".join(prediction_joins)
    prediction_select_sql = ",\n                ".join(prediction_selects)
    road_only_metric_names = {
        "catboost_road_only_score_latest",
        "catboost_road_only_2_0_score_latest",
        "catboost_road_only_2_5_score_latest",
        "catboost_road_only_3_0_score_latest",
        "catboost_road_only_3_5_score_latest",
        "catboost_road_only_4_0_score_latest",
        "catboost_road_only_4_1_log_score_latest",
        *target_layer_by_metric.keys(),
    }
    road_only_where_sql = ""
    if metric == "catboost_road_only_4_0_min_2000_score_latest":
        road_only_where_sql = "WHERE COALESCE(r.road_length_total_m, 0) > 2000"
    elif metric in road_only_metric_names:
        road_only_where_sql = "WHERE COALESCE(r.road_length_total_m, 0) > 0"

    with connect() as con:
        rows = rows_to_dicts(con.execute(
            f"""
            WITH summary AS (
                SELECT
                    h3,
                    SUM(accident_count)::BIGINT AS total_accidents,
                    SUM(CASE WHEN accident_count > 0 THEN 1 ELSE 0 END)::BIGINT AS active_months,
                    SUM(CASE WHEN target_had_accident_next_month THEN 1 ELSE 0 END)::BIGINT AS target_positive_months,
                    MAX(accident_count)::BIGINT AS max_month_accidents
                FROM parquet_scan('{panel}')
                GROUP BY h3
            )
            {prediction_cte_sql}
            SELECT
                s.h3,
                s.total_accidents,
                s.active_months,
                s.target_positive_months,
                s.max_month_accidents,
                COALESCE(r.road_length_total_m, 0) AS road_length_total_m,
                COALESCE(r.junction_count, 0) AS junction_count,
                COALESCE(r.crossing_count, 0) AS crossing_count,
                {prediction_select_sql}
            FROM summary s
            LEFT JOIN parquet_scan('{roads}') r USING (h3)
            {prediction_join_sql}
            {road_only_where_sql}
            ORDER BY s.h3
            """
        ))

    if not rows:
        return {"center": [0, 0], "cells": [], "metric": metric, "max_metric": 0}

    centers = [cell_center(row["h3"]) for row in rows]
    max_metric = max(safe_float(row[metric]) for row in rows)
    cells_payload = []

    for row, center in zip(rows, centers):
        summary = {
            "total_accidents": int(row["total_accidents"]),
            "active_months": int(row["active_months"]),
            "target_positive_months": int(row["target_positive_months"]),
            "max_month_accidents": int(row["max_month_accidents"]),
            "road_length_total_m": safe_float(row["road_length_total_m"]),
            "junction_count": safe_float(row["junction_count"]),
            "crossing_count": safe_float(row["crossing_count"]),
            "logreg_score_latest": safe_float(row["logreg_score_latest"]),
            "logreg_score_month": row["logreg_score_month"][:10] if row["logreg_score_month"] else None,
            "catboost_score_latest": safe_float(row["catboost_score_latest"]),
            "catboost_score_month": row["catboost_score_month"][:10] if row["catboost_score_month"] else None,
            "catboost_no_calendar_score_latest": safe_float(row["catboost_no_calendar_score_latest"]),
            "catboost_no_calendar_score_month": (
                row["catboost_no_calendar_score_month"][:10]
                if row["catboost_no_calendar_score_month"]
                else None
            ),
            "catboost_road_only_score_latest": safe_float(row["catboost_road_only_score_latest"]),
            "catboost_road_only_score_month": (
                row["catboost_road_only_score_month"][:10]
                if row["catboost_road_only_score_month"]
                else None
            ),
            "catboost_road_only_2_0_score_latest": safe_float(row["catboost_road_only_2_0_score_latest"]),
            "catboost_road_only_2_0_score_month": (
                row["catboost_road_only_2_0_score_month"][:10]
                if row["catboost_road_only_2_0_score_month"]
                else None
            ),
            "catboost_road_only_2_5_score_latest": safe_float(row["catboost_road_only_2_5_score_latest"]),
            "catboost_road_only_2_5_score_month": (
                row["catboost_road_only_2_5_score_month"][:10]
                if row["catboost_road_only_2_5_score_month"]
                else None
            ),
            "catboost_road_only_3_0_score_latest": safe_float(row["catboost_road_only_3_0_score_latest"]),
            "catboost_road_only_3_0_score_month": (
                row["catboost_road_only_3_0_score_month"][:10]
                if row["catboost_road_only_3_0_score_month"]
                else None
            ),
            "catboost_road_only_3_5_score_latest": safe_float(row["catboost_road_only_3_5_score_latest"]),
            "catboost_road_only_3_5_score_month": (
                row["catboost_road_only_3_5_score_month"][:10]
                if row["catboost_road_only_3_5_score_month"]
                else None
            ),
            "catboost_road_only_4_0_score_latest": safe_float(row["catboost_road_only_4_0_score_latest"]),
            "catboost_road_only_4_0_score_month": (
                row["catboost_road_only_4_0_score_month"][:10]
                if row["catboost_road_only_4_0_score_month"]
                else None
            ),
            "catboost_road_only_4_0_min_2000_score_latest": safe_float(row["catboost_road_only_4_0_min_2000_score_latest"]),
            "catboost_road_only_4_0_min_2000_score_month": (
                row["catboost_road_only_4_0_min_2000_score_month"][:10]
                if row["catboost_road_only_4_0_min_2000_score_month"]
                else None
            ),
            "catboost_road_only_4_1_log_score_latest": safe_float(row["catboost_road_only_4_1_log_score_latest"]),
            "catboost_road_only_4_1_log_score_month": (
                row["catboost_road_only_4_1_log_score_month"][:10]
                if row["catboost_road_only_4_1_log_score_month"]
                else None
            ),
        }
        for config in CATBOOST_ROAD_ONLY_4_0_TARGET_LAYER_CONFIGS:
            latest_metric = config["latest_metric"]
            latest_month = config["latest_month"]
            summary[latest_metric] = safe_float(row[latest_metric])
            summary[latest_month] = row[latest_month][:10] if row[latest_month] else None

        cells_payload.append({
            "h3": row["h3"],
            "center": center,
            "boundary": cell_boundary(row["h3"]),
            "summary": summary,
            "metric_value": safe_float(row[metric]),
        })

    center_lat = sum(lat for lat, _ in centers) / len(centers)
    center_lng = sum(lng for _, lng in centers) / len(centers)

    return {
        "metric": metric,
        "max_metric": max_metric,
        "center": [center_lat, center_lng],
        "cells": cells_payload,
    }


@app.get("/api/cell/{cell}")
def cell_detail(cell: str) -> dict[str, Any]:
    panel = parquet_sql_path(PANEL_PATH)
    roads = parquet_sql_path(ROAD_FEATURES_PATH)
    logreg_predictions = parquet_sql_path(LOGREG_PREDICTIONS_PATH)
    catboost_predictions = parquet_sql_path(CATBOOST_PREDICTIONS_PATH)
    catboost_no_calendar_predictions = parquet_sql_path(CATBOOST_NO_CALENDAR_PREDICTIONS_PATH)
    catboost_road_only_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_PREDICTIONS_PATH)
    catboost_road_only_2_0_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_2_0_PREDICTIONS_PATH)
    catboost_road_only_2_5_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_2_5_PREDICTIONS_PATH)
    catboost_road_only_3_0_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_3_0_PREDICTIONS_PATH)
    catboost_road_only_3_5_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_3_5_PREDICTIONS_PATH)
    catboost_road_only_4_0_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_4_0_PREDICTIONS_PATH)
    catboost_road_only_4_0_min_2000_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_4_0_MIN_2000_PREDICTIONS_PATH)
    catboost_road_only_4_1_predictions = parquet_sql_path(CATBOOST_ROAD_ONLY_4_1_PREDICTIONS_PATH)
    prediction_selects = [
        "NULL AS logreg_score",
        "NULL AS logreg_score_percentile",
        "NULL AS logreg_rank_in_month",
        "NULL AS catboost_score",
        "NULL AS catboost_score_percentile",
        "NULL AS catboost_rank_in_month",
        "NULL AS catboost_no_calendar_score",
        "NULL AS catboost_no_calendar_score_percentile",
        "NULL AS catboost_no_calendar_rank_in_month",
        "NULL AS catboost_road_only_score",
        "NULL AS catboost_road_only_score_percentile",
        "NULL AS catboost_road_only_rank_in_month",
        "NULL AS catboost_road_only_2_0_score",
        "NULL AS catboost_road_only_2_0_score_percentile",
        "NULL AS catboost_road_only_2_0_rank_in_month",
        "NULL AS catboost_road_only_2_5_score",
        "NULL AS catboost_road_only_2_5_score_percentile",
        "NULL AS catboost_road_only_2_5_rank_in_month",
        "NULL AS catboost_road_only_3_0_score",
        "NULL AS catboost_road_only_3_0_score_percentile",
        "NULL AS catboost_road_only_3_0_rank_in_month",
        "NULL AS catboost_road_only_3_5_score",
        "NULL AS catboost_road_only_3_5_score_percentile",
        "NULL AS catboost_road_only_3_5_rank_in_month",
        "NULL AS catboost_road_only_4_0_score",
        "NULL AS catboost_road_only_4_0_score_percentile",
        "NULL AS catboost_road_only_4_0_rank_in_month",
        "NULL AS catboost_road_only_4_0_min_2000_score",
        "NULL AS catboost_road_only_4_0_min_2000_score_percentile",
        "NULL AS catboost_road_only_4_0_min_2000_rank_in_month",
        "NULL AS catboost_road_only_4_1_log_score",
        "NULL AS catboost_road_only_4_1_log_score_percentile",
        "NULL AS catboost_road_only_4_1_log_rank_in_month",
    ]
    prediction_joins: list[str] = []
    if LOGREG_PREDICTIONS_PATH.exists():
        prediction_selects[0] = "pred.logreg_score"
        prediction_selects[1] = "pred.logreg_score_percentile"
        prediction_selects[2] = "pred.logreg_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{logreg_predictions}') pred
                ON p.h3 = pred.h3 AND p.month = pred.month
        """)
    if CATBOOST_PREDICTIONS_PATH.exists():
        prediction_selects[3] = "catpred.catboost_score"
        prediction_selects[4] = "catpred.catboost_score_percentile"
        prediction_selects[5] = "catpred.catboost_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_predictions}') catpred
                ON p.h3 = catpred.h3 AND p.month = catpred.month
        """)
    if CATBOOST_NO_CALENDAR_PREDICTIONS_PATH.exists():
        prediction_selects[6] = "catnocal.catboost_no_calendar_score"
        prediction_selects[7] = "catnocal.catboost_no_calendar_score_percentile"
        prediction_selects[8] = "catnocal.catboost_no_calendar_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_no_calendar_predictions}') catnocal
                ON p.h3 = catnocal.h3 AND p.month = catnocal.month
        """)
    if CATBOOST_ROAD_ONLY_PREDICTIONS_PATH.exists():
        prediction_selects[9] = "catroad.catboost_road_only_score"
        prediction_selects[10] = "catroad.catboost_road_only_score_percentile"
        prediction_selects[11] = "catroad.catboost_road_only_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_road_only_predictions}') catroad
                ON p.h3 = catroad.h3 AND p.month = catroad.month
        """)
    if CATBOOST_ROAD_ONLY_2_0_PREDICTIONS_PATH.exists():
        prediction_selects[12] = "catroad2.catboost_road_only_2_0_score"
        prediction_selects[13] = "catroad2.catboost_road_only_2_0_score_percentile"
        prediction_selects[14] = "catroad2.catboost_road_only_2_0_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_road_only_2_0_predictions}') catroad2
                ON p.h3 = catroad2.h3 AND p.month = catroad2.month
        """)
    if CATBOOST_ROAD_ONLY_2_5_PREDICTIONS_PATH.exists():
        prediction_selects[15] = "catroad25.catboost_road_only_2_5_score"
        prediction_selects[16] = "catroad25.catboost_road_only_2_5_score_percentile"
        prediction_selects[17] = "catroad25.catboost_road_only_2_5_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_road_only_2_5_predictions}') catroad25
                ON p.h3 = catroad25.h3 AND p.month = catroad25.month
        """)
    if CATBOOST_ROAD_ONLY_3_0_PREDICTIONS_PATH.exists():
        prediction_selects[18] = "catroad30.catboost_road_only_3_0_score"
        prediction_selects[19] = "catroad30.catboost_road_only_3_0_score_percentile"
        prediction_selects[20] = "catroad30.catboost_road_only_3_0_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_road_only_3_0_predictions}') catroad30
                ON p.h3 = catroad30.h3 AND p.month = catroad30.month
        """)
    if CATBOOST_ROAD_ONLY_3_5_PREDICTIONS_PATH.exists():
        prediction_selects[21] = "catroad35.catboost_road_only_3_5_score"
        prediction_selects[22] = "catroad35.catboost_road_only_3_5_score_percentile"
        prediction_selects[23] = "catroad35.catboost_road_only_3_5_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_road_only_3_5_predictions}') catroad35
                ON p.h3 = catroad35.h3 AND p.month = catroad35.month
        """)
    if CATBOOST_ROAD_ONLY_4_0_PREDICTIONS_PATH.exists():
        prediction_selects[24] = "catroad40.catboost_road_only_4_0_score"
        prediction_selects[25] = "catroad40.catboost_road_only_4_0_score_percentile"
        prediction_selects[26] = "catroad40.catboost_road_only_4_0_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_road_only_4_0_predictions}') catroad40
                ON p.h3 = catroad40.h3 AND p.month = catroad40.month
        """)
    if CATBOOST_ROAD_ONLY_4_1_PREDICTIONS_PATH.exists():
        prediction_selects[30] = "catroad41.catboost_road_only_4_1_log_score"
        prediction_selects[31] = "catroad41.catboost_road_only_4_1_log_score_percentile"
        prediction_selects[32] = "catroad41.catboost_road_only_4_1_log_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_road_only_4_1_predictions}') catroad41
                ON p.h3 = catroad41.h3 AND p.month = catroad41.month
        """)
    if CATBOOST_ROAD_ONLY_4_0_MIN_2000_PREDICTIONS_PATH.exists():
        prediction_selects[27] = "catroad402k.catboost_road_only_4_0_min_2000_score"
        prediction_selects[28] = "catroad402k.catboost_road_only_4_0_min_2000_score_percentile"
        prediction_selects[29] = "catroad402k.catboost_road_only_4_0_min_2000_rank_in_month"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{catboost_road_only_4_0_min_2000_predictions}') catroad402k
                ON p.h3 = catroad402k.h3 AND p.month = catroad402k.month
        """)
    for config in CATBOOST_ROAD_ONLY_4_0_TARGET_LAYER_CONFIGS:
        score_column = config["score_column"]
        percentile_column = config["percentile_column"]
        rank_column = config["rank_column"]
        prediction_selects.extend([
            f"NULL AS {score_column}",
            f"NULL AS {percentile_column}",
            f"NULL AS {rank_column}",
        ])
        if not config["path"].exists():
            continue

        alias = f"target_{config['key']}"
        prediction_path = parquet_sql_path(config["path"])
        prediction_selects[-3] = f"{alias}.{score_column}"
        prediction_selects[-2] = f"{alias}.{percentile_column}"
        prediction_selects[-1] = f"{alias}.{rank_column}"
        prediction_joins.append(f"""
            LEFT JOIN parquet_scan('{prediction_path}') {alias}
                ON p.h3 = {alias}.h3 AND p.month = {alias}.month
        """)
    prediction_select_sql = ",\n                ".join(prediction_selects)
    prediction_join_sql = "\n".join(prediction_joins)

    with connect() as con:
        months = rows_to_dicts(con.execute(
            f"""
            SELECT
                p.month::VARCHAR AS month,
                p.accident_count,
                p.had_accident,
                p.accidents_prev_1m,
                p.accidents_prev_3m,
                p.accidents_prev_6m,
                p.accidents_prev_12m,
                p.target_count_next_month,
                p.target_had_accident_next_month,
                {prediction_select_sql}
            FROM parquet_scan('{panel}') p
            {prediction_join_sql}
            WHERE p.h3 = ?
            ORDER BY p.month
            """,
            [cell],
        ))

        if not months:
            raise HTTPException(status_code=404, detail="Unknown H3 cell")

        road_row = con.execute(
            f"""
            SELECT {", ".join(ROAD_FEATURE_COLUMNS)}
            FROM parquet_scan('{roads}')
            WHERE h3 = ?
            """,
            [cell],
        ).fetchone()

    if road_row is None:
        road_features = {column: 0.0 for column in ROAD_FEATURE_COLUMNS}
    else:
        road_features = {
            column: safe_float(value)
            for column, value in zip(ROAD_FEATURE_COLUMNS, road_row)
        }

    for row in months:
        row["month"] = row["month"][:7]
        row["had_accident"] = bool(row["had_accident"])
        row["target_had_accident_next_month"] = bool(row["target_had_accident_next_month"])
        row["logreg_score"] = safe_float(row["logreg_score"])
        row["logreg_score_percentile"] = safe_float(row["logreg_score_percentile"])
        row["logreg_rank_in_month"] = int(row["logreg_rank_in_month"]) if row["logreg_rank_in_month"] is not None else None
        row["catboost_score"] = safe_float(row["catboost_score"])
        row["catboost_score_percentile"] = safe_float(row["catboost_score_percentile"])
        row["catboost_rank_in_month"] = int(row["catboost_rank_in_month"]) if row["catboost_rank_in_month"] is not None else None
        row["catboost_no_calendar_score"] = safe_float(row["catboost_no_calendar_score"])
        row["catboost_no_calendar_score_percentile"] = safe_float(row["catboost_no_calendar_score_percentile"])
        row["catboost_no_calendar_rank_in_month"] = (
            int(row["catboost_no_calendar_rank_in_month"])
            if row["catboost_no_calendar_rank_in_month"] is not None
            else None
        )
        row["catboost_road_only_score"] = safe_float(row["catboost_road_only_score"])
        row["catboost_road_only_score_percentile"] = safe_float(row["catboost_road_only_score_percentile"])
        row["catboost_road_only_rank_in_month"] = (
            int(row["catboost_road_only_rank_in_month"])
            if row["catboost_road_only_rank_in_month"] is not None
            else None
        )
        row["catboost_road_only_2_0_score"] = safe_float(row["catboost_road_only_2_0_score"])
        row["catboost_road_only_2_0_score_percentile"] = safe_float(row["catboost_road_only_2_0_score_percentile"])
        row["catboost_road_only_2_0_rank_in_month"] = (
            int(row["catboost_road_only_2_0_rank_in_month"])
            if row["catboost_road_only_2_0_rank_in_month"] is not None
            else None
        )
        row["catboost_road_only_2_5_score"] = safe_float(row["catboost_road_only_2_5_score"])
        row["catboost_road_only_2_5_score_percentile"] = safe_float(row["catboost_road_only_2_5_score_percentile"])
        row["catboost_road_only_2_5_rank_in_month"] = (
            int(row["catboost_road_only_2_5_rank_in_month"])
            if row["catboost_road_only_2_5_rank_in_month"] is not None
            else None
        )
        row["catboost_road_only_3_0_score"] = safe_float(row["catboost_road_only_3_0_score"])
        row["catboost_road_only_3_0_score_percentile"] = safe_float(row["catboost_road_only_3_0_score_percentile"])
        row["catboost_road_only_3_0_rank_in_month"] = (
            int(row["catboost_road_only_3_0_rank_in_month"])
            if row["catboost_road_only_3_0_rank_in_month"] is not None
            else None
        )
        row["catboost_road_only_3_5_score"] = safe_float(row["catboost_road_only_3_5_score"])
        row["catboost_road_only_3_5_score_percentile"] = safe_float(row["catboost_road_only_3_5_score_percentile"])
        row["catboost_road_only_3_5_rank_in_month"] = (
            int(row["catboost_road_only_3_5_rank_in_month"])
            if row["catboost_road_only_3_5_rank_in_month"] is not None
            else None
        )
        row["catboost_road_only_4_0_score"] = safe_float(row["catboost_road_only_4_0_score"])
        row["catboost_road_only_4_0_score_percentile"] = safe_float(row["catboost_road_only_4_0_score_percentile"])
        row["catboost_road_only_4_0_rank_in_month"] = (
            int(row["catboost_road_only_4_0_rank_in_month"])
            if row["catboost_road_only_4_0_rank_in_month"] is not None
            else None
        )
        row["catboost_road_only_4_0_min_2000_score"] = safe_float(row["catboost_road_only_4_0_min_2000_score"])
        row["catboost_road_only_4_0_min_2000_score_percentile"] = safe_float(row["catboost_road_only_4_0_min_2000_score_percentile"])
        row["catboost_road_only_4_0_min_2000_rank_in_month"] = (
            int(row["catboost_road_only_4_0_min_2000_rank_in_month"])
            if row["catboost_road_only_4_0_min_2000_rank_in_month"] is not None
            else None
        )
        row["catboost_road_only_4_1_log_score"] = safe_float(row["catboost_road_only_4_1_log_score"])
        row["catboost_road_only_4_1_log_score_percentile"] = safe_float(row["catboost_road_only_4_1_log_score_percentile"])
        row["catboost_road_only_4_1_log_rank_in_month"] = (
            int(row["catboost_road_only_4_1_log_rank_in_month"])
            if row["catboost_road_only_4_1_log_rank_in_month"] is not None
            else None
        )
        for config in CATBOOST_ROAD_ONLY_4_0_TARGET_LAYER_CONFIGS:
            score_column = config["score_column"]
            percentile_column = config["percentile_column"]
            rank_column = config["rank_column"]
            row[score_column] = safe_float(row[score_column])
            row[percentile_column] = safe_float(row[percentile_column])
            row[rank_column] = (
                int(row[rank_column])
                if row[rank_column] is not None
                else None
            )

    return {
        "h3": cell,
        "center": cell_center(cell),
        "boundary": cell_boundary(cell),
        "road_features": road_features,
        "months": months,
    }


@app.get("/api/cell/{cell}/intersections")
def cell_intersections(cell: str) -> dict[str, Any]:
    if not ROAD_GRAPH_NODES_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing road graph nodes parquet: {ROAD_GRAPH_NODES_PATH}",
        )

    graph_nodes = parquet_sql_path(ROAD_GRAPH_NODES_PATH)
    with connect() as con:
        rows = rows_to_dicts(con.execute(
            f"""
            SELECT
                node_id,
                lat,
                lng,
                junction_way_count,
                junction_degree,
                is_three_plus_way,
                is_class_change,
                is_name_change,
                is_major_junction,
                is_minor_to_major_junction,
                is_class_delta_junction,
                class_rank_delta,
                has_traffic_signal_nearby,
                nearest_traffic_signal_m,
                connected_highway_classes,
                connected_names
            FROM parquet_scan('{graph_nodes}')
            WHERE h3 = ?
            ORDER BY is_minor_to_major_junction DESC, junction_way_count DESC, node_id
            """,
            [cell],
        ))

    for row in rows:
        row["node_id"] = int(row["node_id"])
        row["lat"] = safe_float(row["lat"])
        row["lng"] = safe_float(row["lng"])
        row["junction_way_count"] = int(row["junction_way_count"])
        row["junction_degree"] = int(row["junction_degree"])
        row["is_three_plus_way"] = bool(row["is_three_plus_way"])
        row["is_class_change"] = bool(row["is_class_change"])
        row["is_name_change"] = bool(row["is_name_change"])
        row["is_major_junction"] = bool(row["is_major_junction"])
        row["is_minor_to_major_junction"] = bool(row["is_minor_to_major_junction"])
        row["is_class_delta_junction"] = bool(row["is_class_delta_junction"])
        row["class_rank_delta"] = int(row["class_rank_delta"])
        row["has_traffic_signal_nearby"] = bool(row["has_traffic_signal_nearby"])
        if row["nearest_traffic_signal_m"] is not None:
            row["nearest_traffic_signal_m"] = safe_float(row["nearest_traffic_signal_m"])

    return {
        "h3": cell,
        "count": len(rows),
        "intersections": rows,
    }


@app.get("/api/cell/{cell}/roads")
def cell_roads(cell: str) -> dict[str, Any]:
    if not ROAD_H3_SEGMENTS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing road H3 segments parquet: {ROAD_H3_SEGMENTS_PATH}",
        )
    if not ROAD_RAW_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing raw roads parquet: {ROAD_RAW_PATH}",
        )

    segments = parquet_sql_path(ROAD_H3_SEGMENTS_PATH)
    raw_roads = parquet_sql_path(ROAD_RAW_PATH)
    with connect() as con:
        rows = rows_to_dicts(con.execute(
            f"""
            SELECT
                s.osm_id,
                s.highway,
                s.segment_length_m,
                s.maxspeed_num,
                s.lanes_num,
                s.oneway,
                s.surface,
                s.is_major,
                s.is_link,
                r.name,
                r.ref,
                r.coords_json
            FROM parquet_scan('{segments}') s
            JOIN parquet_scan('{raw_roads}') r USING (osm_id)
            WHERE s.h3 = ?
            ORDER BY s.is_major DESC, s.segment_length_m DESC, s.highway, s.osm_id
            """,
            [cell],
        ))

    roads = []
    for row in rows:
        coords = parse_json(row["coords_json"]) or []
        parts = road_parts_in_cell(coords, cell)
        if not parts:
            continue
        roads.append({
            "osm_id": int(row["osm_id"]),
            "highway": row["highway"],
            "name": row["name"],
            "ref": row["ref"],
            "length_m": safe_float(row["segment_length_m"]),
            "maxspeed_num": safe_float(row["maxspeed_num"]),
            "lanes_num": safe_float(row["lanes_num"]),
            "oneway": bool(row["oneway"]),
            "surface": row["surface"],
            "is_major": bool(row["is_major"]),
            "is_link": bool(row["is_link"]),
            "parts": parts,
        })

    return {
        "h3": cell,
        "count": len(roads),
        "total_length_m": sum(road["length_m"] for road in roads),
        "roads": roads,
    }
