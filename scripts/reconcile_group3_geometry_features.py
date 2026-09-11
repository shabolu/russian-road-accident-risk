from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import h3
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.paths import (
    H3_MONTH_PANEL_PARQUET,
    H3_MONTH_ROAD_FEATURES_PARQUET,
    H3_ROAD_FEATURES_PARQUET,
    OSM_ROADS_RAW_PARQUET,
)


SHARP_TURN_DEG = 45.0
SHORT_SEGMENT_M = 50.0
ORIENTATION_BINS = 8
GEOMETRY_COLUMNS = [
    "road_curvature_mean_weighted",
    "road_curvature_max",
    "curvature_deg_per_km",
    "turn_event_count",
    "turn_event_density_per_km",
    "sharp_turn_count",
    "sharp_turn_density_per_km",
    "sharp_turn_share",
    "road_segment_count",
    "road_segment_density_per_km",
    "road_segment_length_mean",
    "road_segment_length_median",
    "short_segment_count",
    "short_segment_share",
    "road_orientation_entropy",
    "dead_end_count",
    "dead_end_density_per_km",
]


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def midpoint(lat1: float, lng1: float, lat2: float, lng2: float) -> tuple[float, float]:
    return (lat1 + lat2) / 2, (lng1 + lng2) / 2


def bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lng2 - lng1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    angle = math.degrees(math.atan2(y, x))
    return (angle + 360.0) % 360.0


def turn_angle_deg(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    b1 = bearing_deg(p0[0], p0[1], p1[0], p1[1])
    b2 = bearing_deg(p1[0], p1[1], p2[0], p2[1])
    diff = abs(b2 - b1)
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def orientation_bin(angle_deg: float, bins: int = ORIENTATION_BINS) -> int:
    undirected = angle_deg % 180.0
    width = 180.0 / bins
    return min(int(undirected / width), bins - 1)


def orientation_entropy(weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for weight in weights:
        if weight <= 0:
            continue
        p = weight / total
        entropy -= p * math.log(p, 2)
    return entropy


def reconcile_group3_geometry_features() -> None:
    features = pd.read_parquet(H3_ROAD_FEATURES_PARQUET)
    raw = pd.read_parquet(OSM_ROADS_RAW_PARQUET, columns=["nodes_json", "coords_json"])

    node_coords: dict[int, tuple[float, float]] = {}
    adjacency: dict[int, set[int]] = defaultdict(set)
    agg: dict[str, dict[str, object]] = defaultdict(lambda: {
        "turn_angle_sum_deg": 0.0,
        "turn_angle_weighted_sum": 0.0,
        "turn_weight_sum": 0.0,
        "road_curvature_max": 0.0,
        "turn_event_count": 0,
        "sharp_turn_count": 0,
        "road_segment_count": 0,
        "road_segment_length_sum": 0.0,
        "road_segment_lengths": [],
        "short_segment_count": 0,
        "orientation_bin_weights": [0.0] * ORIENTATION_BINS,
        "dead_end_count": 0,
    })

    for row in raw.itertuples(index=False):
        node_ids = json.loads(row.nodes_json) if isinstance(row.nodes_json, str) else []
        coords = json.loads(row.coords_json) if isinstance(row.coords_json, str) else []
        if len(node_ids) != len(coords) or len(coords) < 2:
            continue

        points: list[tuple[float, float]] = []
        for node_id, coord in zip(node_ids, coords):
            lat = float(coord[0])
            lng = float(coord[1])
            node_id = int(node_id)
            node_coords[node_id] = (lat, lng)
            points.append((lat, lng))

        for left, right in zip(node_ids, node_ids[1:]):
            left_id = int(left)
            right_id = int(right)
            if left_id == right_id:
                continue
            adjacency[left_id].add(right_id)
            adjacency[right_id].add(left_id)

        segment_lengths: list[float] = []
        for p0, p1 in zip(points, points[1:]):
            seg_len = haversine_m(p0[0], p0[1], p1[0], p1[1])
            segment_lengths.append(seg_len)
            if seg_len <= 0:
                continue
            mid = midpoint(p0[0], p0[1], p1[0], p1[1])
            cell = h3.latlng_to_cell(mid[0], mid[1], 9)
            cell_agg = agg[cell]
            cell_agg["road_segment_count"] = int(cell_agg["road_segment_count"]) + 1
            cell_agg["road_segment_length_sum"] = float(cell_agg["road_segment_length_sum"]) + seg_len
            cell_agg["road_segment_lengths"].append(seg_len)
            if seg_len < SHORT_SEGMENT_M:
                cell_agg["short_segment_count"] = int(cell_agg["short_segment_count"]) + 1
            b = bearing_deg(p0[0], p0[1], p1[0], p1[1])
            cell_agg["orientation_bin_weights"][orientation_bin(b)] += seg_len

        for idx in range(1, len(points) - 1):
            turn = turn_angle_deg(points[idx - 1], points[idx], points[idx + 1])
            if turn <= 0:
                continue
            local_len = (segment_lengths[idx - 1] + segment_lengths[idx]) / 2
            cell = h3.latlng_to_cell(points[idx][0], points[idx][1], 9)
            cell_agg = agg[cell]
            cell_agg["turn_angle_sum_deg"] = float(cell_agg["turn_angle_sum_deg"]) + turn
            cell_agg["turn_angle_weighted_sum"] = float(cell_agg["turn_angle_weighted_sum"]) + turn * local_len
            cell_agg["turn_weight_sum"] = float(cell_agg["turn_weight_sum"]) + local_len
            cell_agg["turn_event_count"] = int(cell_agg["turn_event_count"]) + 1
            cell_agg["road_curvature_max"] = max(float(cell_agg["road_curvature_max"]), turn)
            if turn >= SHARP_TURN_DEG:
                cell_agg["sharp_turn_count"] = int(cell_agg["sharp_turn_count"]) + 1

    for node_id, neighbors in adjacency.items():
        if len(neighbors) != 1:
            continue
        coord = node_coords.get(node_id)
        if coord is None:
            continue
        cell = h3.latlng_to_cell(coord[0], coord[1], 9)
        agg[cell]["dead_end_count"] = int(agg[cell]["dead_end_count"]) + 1

    rows: list[dict[str, float | int | str]] = []
    for cell, values in agg.items():
        road_segment_count = int(values["road_segment_count"])
        turn_event_count = int(values["turn_event_count"])
        short_segment_count = int(values["short_segment_count"])
        turn_weight_sum = float(values["turn_weight_sum"])
        road_segment_lengths = list(values["road_segment_lengths"])
        rows.append({
            "h3": cell,
            "road_curvature_mean_weighted": (
                float(values["turn_angle_weighted_sum"]) / turn_weight_sum if turn_weight_sum > 0 else 0.0
            ),
            "road_curvature_max": float(values["road_curvature_max"]),
            "turn_angle_sum_deg": float(values["turn_angle_sum_deg"]),
            "turn_event_count": turn_event_count,
            "sharp_turn_count": int(values["sharp_turn_count"]),
            "road_segment_count": road_segment_count,
            "road_segment_length_mean": (
                float(values["road_segment_length_sum"]) / road_segment_count if road_segment_count > 0 else 0.0
            ),
            "road_segment_length_median": float(pd.Series(road_segment_lengths).median()) if road_segment_lengths else 0.0,
            "short_segment_count": short_segment_count,
            "road_orientation_entropy": orientation_entropy(values["orientation_bin_weights"]),
            "dead_end_count": int(values["dead_end_count"]),
        })

    geometry = pd.DataFrame(rows)
    features = features.drop(columns=[column for column in GEOMETRY_COLUMNS if column in features.columns], errors="ignore")
    features = features.drop(columns=["turn_angle_sum_deg"], errors="ignore")
    features = features.merge(geometry, on="h3", how="left")

    for column in GEOMETRY_COLUMNS + ["turn_angle_sum_deg"]:
        if column not in features.columns:
            features[column] = 0.0
    features[GEOMETRY_COLUMNS + ["turn_angle_sum_deg"]] = features[GEOMETRY_COLUMNS + ["turn_angle_sum_deg"]].fillna(0)

    road_length_km = features["road_length_total_m"] / 1000
    features["curvature_deg_per_km"] = features["turn_angle_sum_deg"] / road_length_km
    features["turn_event_density_per_km"] = features["turn_event_count"] / road_length_km
    features["sharp_turn_density_per_km"] = features["sharp_turn_count"] / road_length_km
    features["road_segment_density_per_km"] = features["road_segment_count"] / road_length_km
    features["short_segment_share"] = features["short_segment_count"] / features["road_segment_count"]
    features["sharp_turn_share"] = features["sharp_turn_count"] / features["turn_event_count"]
    features["dead_end_density_per_km"] = features["dead_end_count"] / road_length_km

    ratio_columns = [
        "curvature_deg_per_km",
        "turn_event_density_per_km",
        "sharp_turn_density_per_km",
        "road_segment_density_per_km",
        "short_segment_share",
        "sharp_turn_share",
        "dead_end_density_per_km",
    ]
    features[ratio_columns] = features[ratio_columns].replace([float("inf"), -float("inf")], 0).fillna(0)
    features = features.drop(columns=["turn_angle_sum_deg"])

    features.to_parquet(H3_ROAD_FEATURES_PARQUET, index=False)

    panel = pd.read_parquet(
        H3_MONTH_PANEL_PARQUET,
        columns=["h3", "month", "year", "month_num", "target_had_accident_next_month"],
    )
    h3_month = panel.merge(features, on="h3", how="left")
    feature_columns = [column for column in features.columns if column != "h3"]
    h3_month[feature_columns] = h3_month[feature_columns].fillna(0)
    h3_month.to_parquet(H3_MONTH_ROAD_FEATURES_PARQUET, index=False)

    print(f"Saved: {H3_ROAD_FEATURES_PARQUET}")
    print(f"Saved: {H3_MONTH_ROAD_FEATURES_PARQUET}")
    print("Added/updated group 3 geometry columns:")
    for column in GEOMETRY_COLUMNS:
        print(f"- {column}: sum={features[column].sum():,.3f}")


if __name__ == "__main__":
    reconcile_group3_geometry_features()
