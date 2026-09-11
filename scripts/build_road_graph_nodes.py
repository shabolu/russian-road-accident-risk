from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.osm_road_features import (
    MAJOR_ROAD_CLASSES,
    TRAFFIC_SIGNAL_RADIUS_M,
    latlng_to_h3_cell,
    nearest_signal_distance_m,
    road_class_delta,
)
from road_accident_risk.paths import (
    OSM_ROAD_GRAPH_NODES_PARQUET,
    OSM_ROAD_POINT_NODES_PARQUET,
    OSM_ROADS_RAW_PARQUET,
)


def parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def build_graph_nodes(
    input_path: Path = OSM_ROADS_RAW_PARQUET,
    point_nodes_path: Path = OSM_ROAD_POINT_NODES_PARQUET,
    output_path: Path = OSM_ROAD_GRAPH_NODES_PARQUET,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing raw roads parquet: {input_path}")

    roads = pd.read_parquet(input_path)
    signal_points: list[tuple[float, float]] = []
    if point_nodes_path.exists():
        points = pd.read_parquet(point_nodes_path)
        if not points.empty:
            signals = points[points["is_traffic_signal"]]
            signal_points = list(zip(signals["lat"].astype(float), signals["lng"].astype(float)))

    node_coords: dict[int, tuple[float, float]] = {}
    node_way_ids: dict[int, set[int]] = defaultdict(set)
    node_classes: dict[int, list[str]] = defaultdict(list)
    node_names: dict[int, set[str]] = defaultdict(set)

    for row in roads.to_dict(orient="records"):
        osm_id = int(row["osm_id"])
        highway = str(row["highway"])
        name = row.get("name")
        nodes = parse_json(row["nodes_json"]) or []
        coords = parse_json(row["coords_json"]) or []

        for node_id, coord in zip(nodes, coords):
            node_id = int(node_id)
            lat = float(coord[0])
            lng = float(coord[1])
            node_coords[node_id] = (lat, lng)
            node_way_ids[node_id].add(osm_id)
            node_classes[node_id].append(highway)
            if name is not None and not pd.isna(name):
                node_names[node_id].add(str(name))

    rows = []
    for node_id, way_ids in node_way_ids.items():
        if len(way_ids) < 2 or node_id not in node_coords:
            continue

        classes = node_classes[node_id]
        class_set = set(classes)
        names = node_names[node_id]
        major_count = sum(1 for cls in classes if cls in MAJOR_ROAD_CLASSES)
        minor_count = sum(1 for cls in classes if cls not in MAJOR_ROAD_CLASSES)
        legacy_major_minor = major_count > 0 and minor_count > 0
        is_class_change = len(class_set) > 1
        is_name_change = len(names) > 1
        is_three_plus_way = len(way_ids) >= 3
        class_delta = road_class_delta(classes)
        is_class_delta = class_delta >= 1
        lat, lng = node_coords[node_id]
        nearest_signal_m = nearest_signal_distance_m(
            lat,
            lng,
            signal_points,
            max_radius_m=TRAFFIC_SIGNAL_RADIUS_M,
        )
        has_signal_nearby = nearest_signal_m is not None
        is_minor_to_major = is_class_delta and not has_signal_nearby

        # Exclude plain OSM segmentation: two ways with the same apparent class/name.
        is_display_intersection = is_three_plus_way or legacy_major_minor or is_class_change or is_name_change
        if not is_display_intersection:
            continue

        rows.append({
            "node_id": node_id,
            "h3": latlng_to_h3_cell(lat, lng),
            "lat": lat,
            "lng": lng,
            "junction_way_count": len(way_ids),
            "junction_degree": len(classes),
            "is_three_plus_way": is_three_plus_way,
            "is_class_change": is_class_change,
            "is_name_change": is_name_change,
            "is_major_junction": major_count > 0,
            "is_minor_to_major_junction": is_minor_to_major,
            "is_legacy_major_minor_junction": legacy_major_minor,
            "class_rank_delta": class_delta,
            "is_class_delta_junction": is_class_delta,
            "has_traffic_signal_nearby": has_signal_nearby,
            "nearest_traffic_signal_m": nearest_signal_m,
            "is_unsignalized_class_delta_junction": is_class_delta and not has_signal_nearby,
            "is_unsignalized_high_delta_junction": class_delta >= 2 and not has_signal_nearby,
            "is_signalized_class_delta_junction": is_class_delta and has_signal_nearby,
            "connected_highway_classes": ";".join(sorted(class_set)),
            "connected_names": ";".join(sorted(names)),
        })

    result = pd.DataFrame(rows).sort_values(["h3", "node_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(result):,}")
    print(f"Unique H3: {result['h3'].nunique():,}" if len(result) else "Unique H3: 0")
    return result


if __name__ == "__main__":
    build_graph_nodes()
