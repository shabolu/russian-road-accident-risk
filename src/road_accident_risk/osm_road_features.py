from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h3
import pandas as pd

from road_accident_risk.paths import (
    H3_MONTH_PANEL_PARQUET,
    H3_MONTH_ROAD_FEATURES_PARQUET,
    H3_ROAD_FEATURES_PARQUET,
    OSM_ROAD_H3_SEGMENTS_PARQUET,
    OSM_ROADS_RAW_PARQUET,
    OSM_ROAD_GRAPH_NODES_PARQUET,
    OSM_ROAD_POINT_NODES_PARQUET,
    OSM_SARATOV_ROADS_JSON,
    OSM_VOLGA_PBF,
)
from road_accident_risk.utils import format_size


BBOX = (51.40, 45.80, 51.70, 46.20)
H3_RESOLUTION = 9
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SAMPLE_STEP_M = 20.0
PBF_PROGRESS_EVERY = 1_000_000
TRAFFIC_SIGNAL_RADIUS_M = 30.0
POINT_HIGHWAY_VALUES = {
    "traffic_signals",
    "crossing",
    "stop",
    "give_way",
    "mini_roundabout",
    "motorway_junction",
    "bus_stop",
}
POINT_RAILWAY_VALUES = {
    "tram_stop",
    "level_crossing",
    "crossing",
}

TARGET = "target_had_accident_next_month"

MOTOR_ROAD_CLASSES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}
MAJOR_ROAD_CLASSES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
}
ROAD_CLASS_RANK = {
    "service": 0,
    "living_street": 1,
    "residential": 2,
    "unclassified": 3,
    "tertiary": 4,
    "tertiary_link": 4,
    "secondary": 5,
    "secondary_link": 5,
    "primary": 6,
    "primary_link": 6,
    "trunk": 7,
    "trunk_link": 7,
    "motorway": 8,
    "motorway_link": 8,
}
LINK_ROAD_CLASSES = {
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}
BAD_SURFACES = {
    "unpaved",
    "gravel",
    "ground",
    "dirt",
    "earth",
    "sand",
    "grass",
    "mud",
    "cobblestone",
    "sett",
}
ASPHALT_SURFACES = {"asphalt", "paved", "concrete", "concrete:plates", "concrete:lanes"}
MAXSPEED_ZONE_DEFAULTS = {
    "ru:urban": 60.0,
    "ru:rural": 90.0,
    "ru:living_street": 20.0,
}
MAXSPEED_IMPUTATION_BY_HIGHWAY = {
    "motorway": 110.0,
    "motorway_link": 90.0,
    "trunk": 90.0,
    "trunk_link": 90.0,
    "primary": 60.0,
    "primary_link": 60.0,
    "secondary": 60.0,
    "secondary_link": 60.0,
    "tertiary": 60.0,
    "tertiary_link": 50.0,
    "unclassified": 60.0,
    "residential": 60.0,
    "living_street": 20.0,
    "service": 20.0,
}


def latlng_to_h3_cell(lat: float, lng: float, resolution: int = H3_RESOLUTION) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    return h3.geo_to_h3(lat, lng, resolution)


def make_overpass_query(bbox: tuple[float, float, float, float] = BBOX) -> str:
    min_lat, min_lng, max_lat, max_lng = bbox
    bbox_text = f"{min_lat},{min_lng},{max_lat},{max_lng}"
    return f"""
[out:json][timeout:180];
(
  way["highway"]({bbox_text});
  node["highway"~"^(traffic_signals|crossing|stop|give_way|mini_roundabout|motorway_junction|bus_stop)$"]({bbox_text});
  node["traffic_calming"]({bbox_text});
  node["railway"~"^(tram_stop|level_crossing|crossing)$"]({bbox_text});
  node["public_transport"~"^(platform|stop_position)$"]({bbox_text});
);
out body;
>;
out skel qt;
"""


def download_overpass_json(
    output_path: Path = OSM_SARATOV_ROADS_JSON,
    bbox: tuple[float, float, float, float] = BBOX,
    overpass_url: str = OVERPASS_URL,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    query = make_overpass_query(bbox)
    encoded = urllib.parse.urlencode({"data": query}).encode("utf-8")

    print(f"Downloading OSM road data from: {overpass_url}")
    print(f"BBOX: {bbox}")
    started_at = time.time()

    request = urllib.request.Request(
        overpass_url,
        data=encoded,
        headers={"User-Agent": "russian-road-accident-risk/0.1"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = response.read()

    output_path.write_bytes(payload)
    elapsed = max(time.time() - started_at, 1e-9)
    print(f"Saved: {output_path}")
    print(f"Size: {format_size(output_path.stat().st_size)}")
    print(f"Elapsed: {elapsed:.1f}s")
    return output_path


def load_overpass(path: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes: dict[int, dict[str, Any]] = {}
    ways: list[dict[str, Any]] = []
    tagged_nodes: list[dict[str, Any]] = []

    for element in data.get("elements", []):
        element_type = element.get("type")
        if element_type == "node":
            node_id = int(element["id"])
            nodes[node_id] = element
            if element.get("tags"):
                tagged_nodes.append(element)
        elif element_type == "way":
            ways.append(element)

    return nodes, ways, tagged_nodes


def point_in_bbox(lat: float, lng: float, bbox: tuple[float, float, float, float] = BBOX) -> bool:
    min_lat, min_lng, max_lat, max_lng = bbox
    return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng


def coords_touch_bbox(
    coords: list[tuple[float, float]],
    bbox: tuple[float, float, float, float] = BBOX,
    padding_deg: float = 0.01,
) -> bool:
    if not coords:
        return False

    min_lat, min_lng, max_lat, max_lng = bbox
    min_lat -= padding_deg
    min_lng -= padding_deg
    max_lat += padding_deg
    max_lng += padding_deg

    for lat, lng in coords:
        if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
            return True
    return False


def print_rate_progress(label: str, processed: int, kept: int, started_at: float) -> None:
    elapsed = max(time.time() - started_at, 1e-9)
    print(
        f"[{label}] processed={processed:,} | kept={kept:,} | "
        f"speed={processed / elapsed:,.0f}/s | elapsed={elapsed:.1f}s",
        flush=True,
    )


class PbfWayCollector:
    def __init__(self, bbox: tuple[float, float, float, float] = BBOX) -> None:
        import osmium

        class Handler(osmium.SimpleHandler):
            def __init__(self, outer: PbfWayCollector) -> None:
                super().__init__()
                self.outer = outer

            def way(self, way) -> None:
                self.outer.processed_ways += 1
                if self.outer.processed_ways % PBF_PROGRESS_EVERY == 0:
                    print_rate_progress(
                        "pbf ways pass 1",
                        self.outer.processed_ways,
                        len(self.outer.ways),
                        self.outer.started_at,
                    )

                tags = dict(way.tags)
                highway = tags.get("highway")
                if highway not in MOTOR_ROAD_CLASSES:
                    return

                node_ids: list[int] = []
                for node_ref in way.nodes:
                    node_ids.append(int(node_ref.ref))

                if len(node_ids) < 2:
                    return

                self.outer.ways.append({
                    "type": "way",
                    "id": int(way.id),
                    "nodes": node_ids,
                    "tags": tags,
                })
                self.outer.required_node_ids.update(node_ids)

        self.bbox = bbox
        self.ways: list[dict[str, Any]] = []
        self.required_node_ids: set[int] = set()
        self.processed_ways = 0
        self.started_at = time.time()
        self.handler = Handler(self)


class PbfNodeCollector:
    def __init__(
        self,
        required_node_ids: set[int],
        bbox: tuple[float, float, float, float] = BBOX,
    ) -> None:
        import osmium

        class Handler(osmium.SimpleHandler):
            def __init__(self, outer: PbfNodeCollector) -> None:
                super().__init__()
                self.outer = outer

            def node(self, node) -> None:
                self.outer.processed_nodes += 1
                if self.outer.processed_nodes % PBF_PROGRESS_EVERY == 0:
                    print_rate_progress(
                        "pbf nodes pass 2",
                        self.outer.processed_nodes,
                        len(self.outer.nodes),
                        self.outer.started_at,
                    )

                if not node.location.valid():
                    return

                node_id = int(node.id)
                lat = float(node.location.lat)
                lng = float(node.location.lon)
                tags = dict(node.tags)

                if node_id in self.outer.required_node_ids:
                    self.outer.nodes[node_id] = {
                        "type": "node",
                        "id": node_id,
                        "lat": lat,
                        "lon": lng,
                    }

                if (
                    point_in_bbox(lat, lng, self.outer.bbox)
                    and is_relevant_point_tags(tags)
                ):
                    self.outer.tagged_nodes.append({
                        "type": "node",
                        "id": node_id,
                        "lat": lat,
                        "lon": lng,
                        "tags": tags,
                    })

        self.required_node_ids = required_node_ids
        self.bbox = bbox
        self.nodes: dict[int, dict[str, Any]] = {}
        self.tagged_nodes: list[dict[str, Any]] = []
        self.processed_nodes = 0
        self.started_at = time.time()
        self.handler = Handler(self)


class PbfTaggedNodeCollector:
    def __init__(self, bbox: tuple[float, float, float, float] = BBOX) -> None:
        import osmium

        class Handler(osmium.SimpleHandler):
            def __init__(self, outer: PbfTaggedNodeCollector) -> None:
                super().__init__()
                self.outer = outer

            def node(self, node) -> None:
                self.outer.processed_nodes += 1
                if self.outer.processed_nodes % PBF_PROGRESS_EVERY == 0:
                    print_rate_progress(
                        "pbf tagged nodes",
                        self.outer.processed_nodes,
                        len(self.outer.tagged_nodes),
                        self.outer.started_at,
                    )

                if not node.location.valid():
                    return

                lat = float(node.location.lat)
                lng = float(node.location.lon)
                if not point_in_bbox(lat, lng, self.outer.bbox):
                    return

                tags = dict(node.tags)
                if is_relevant_point_tags(tags):
                    self.outer.tagged_nodes.append({
                        "type": "node",
                        "id": int(node.id),
                        "lat": lat,
                        "lon": lng,
                        "tags": tags,
                    })

        self.bbox = bbox
        self.tagged_nodes: list[dict[str, Any]] = []
        self.processed_nodes = 0
        self.started_at = time.time()
        self.handler = Handler(self)


def load_pbf_tagged_nodes(
    pbf_path: Path,
    bbox: tuple[float, float, float, float] = BBOX,
) -> list[dict[str, Any]]:
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF not found: {pbf_path}")

    print(f"Reading tagged point nodes from PBF: {pbf_path}")
    print(f"PBF size: {format_size(pbf_path.stat().st_size)}")
    print(f"BBOX: {bbox}")
    collector = PbfTaggedNodeCollector(bbox=bbox)
    collector.handler.apply_file(str(pbf_path), locations=True)
    print_rate_progress(
        "pbf tagged nodes done",
        collector.processed_nodes,
        len(collector.tagged_nodes),
        collector.started_at,
    )
    return collector.tagged_nodes


def load_pbf_roads(
    pbf_path: Path,
    bbox: tuple[float, float, float, float] = BBOX,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF not found: {pbf_path}")

    print(f"Reading PBF: {pbf_path}")
    print(f"PBF size: {format_size(pbf_path.stat().st_size)}")
    print(f"BBOX: {bbox}")

    started_at = time.time()

    way_collector = PbfWayCollector(bbox=bbox)
    print("PBF pass 1/2: collecting motor-road ways and referenced node ids...", flush=True)
    way_collector.handler.apply_file(str(pbf_path), locations=False)
    print_rate_progress(
        "pbf ways pass 1 done",
        way_collector.processed_ways,
        len(way_collector.ways),
        way_collector.started_at,
    )
    print(f"Required node ids: {len(way_collector.required_node_ids):,}", flush=True)

    node_collector = PbfNodeCollector(
        required_node_ids=way_collector.required_node_ids,
        bbox=bbox,
    )
    print("PBF pass 2/2: collecting coordinates for referenced nodes and road point tags...", flush=True)
    node_collector.handler.apply_file(str(pbf_path), locations=True)
    print_rate_progress(
        "pbf nodes pass 2 done",
        node_collector.processed_nodes,
        len(node_collector.nodes),
        node_collector.started_at,
    )

    nodes = node_collector.nodes
    tagged_nodes = node_collector.tagged_nodes

    print("Filtering ways to bbox after coordinates are available...", flush=True)
    ways = []
    missing_coord_ways = 0
    outside_bbox_ways = 0
    for way in way_collector.ways:
        coords = way_coords(way, nodes)
        if len(coords) < 2:
            missing_coord_ways += 1
            continue
        if not coords_touch_bbox(coords, bbox):
            outside_bbox_ways += 1
            continue
        ways.append(way)

    used_node_ids = {int(node_id) for way in ways for node_id in way.get("nodes", [])}
    nodes = {node_id: node for node_id, node in nodes.items() if node_id in used_node_ids}

    elapsed = max(time.time() - started_at, 1e-9)
    print(f"Filtered road nodes: {len(nodes):,}")
    print(f"Filtered road ways: {len(ways):,}")
    print(f"Filtered tagged road nodes: {len(tagged_nodes):,}")
    print(f"Dropped ways without coords: {missing_coord_ways:,}")
    print(f"Dropped ways outside bbox: {outside_bbox_ways:,}")
    print(f"Total elapsed: {elapsed:.1f}s")
    return nodes, ways, tagged_nodes


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None

    value = value.strip().lower().replace(",", ".")
    if not value:
        return None

    first = value.split(";")[0].strip()
    if first in MAXSPEED_ZONE_DEFAULTS:
        return MAXSPEED_ZONE_DEFAULTS[first]
    if first.endswith("mph"):
        number = first.replace("mph", "").strip()
        try:
            return float(number) * 1.609344
        except ValueError:
            return None
    if first in {"none", "signals", "walk", "national"}:
        return None

    digits = "".join(ch if ch.isdigit() or ch == "." else " " for ch in first).split()
    if not digits:
        return None
    try:
        return float(digits[0])
    except ValueError:
        return None


def parse_bool_yes(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"yes", "true", "1"}


def is_relevant_point_tags(tags: dict[str, Any]) -> bool:
    highway = tags.get("highway")
    railway = tags.get("railway")
    public_transport = tags.get("public_transport")
    traffic_calming = tags.get("traffic_calming")
    return (
        highway in POINT_HIGHWAY_VALUES
        or railway in POINT_RAILWAY_VALUES
        or public_transport in {"platform", "stop_position"}
        or bool(traffic_calming)
    )


def way_coords(way: dict[str, Any], nodes: dict[int, dict[str, Any]]) -> list[tuple[float, float]]:
    coords = []
    for node_id in way.get("nodes", []):
        node = nodes.get(int(node_id))
        if node is None:
            continue
        coords.append((float(node["lat"]), float(node["lon"])))
    return coords


def way_length_m(coords: list[tuple[float, float]]) -> float:
    return sum(
        haversine_m(lat1, lng1, lat2, lng2)
        for (lat1, lng1), (lat2, lng2) in zip(coords, coords[1:])
    )


def interpolate(lat1: float, lng1: float, lat2: float, lng2: float, fraction: float) -> tuple[float, float]:
    return lat1 + (lat2 - lat1) * fraction, lng1 + (lng2 - lng1) * fraction


def row_base_for_way(way: dict[str, Any], nodes: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    tags = way.get("tags", {})
    highway = tags.get("highway")
    if highway not in MOTOR_ROAD_CLASSES:
        return None

    coords = way_coords(way, nodes)
    if len(coords) < 2:
        return None

    length_m = way_length_m(coords)
    if length_m <= 0:
        return None

    lanes = parse_number(tags.get("lanes"))
    maxspeed_raw = parse_number(tags.get("maxspeed"))
    maxspeed = maxspeed_raw if maxspeed_raw is not None else MAXSPEED_IMPUTATION_BY_HIGHWAY.get(highway)
    surface = tags.get("surface")

    return {
        "osm_id": int(way["id"]),
        "highway": highway,
        "name": tags.get("name"),
        "ref": tags.get("ref"),
        "length_m": length_m,
        "maxspeed_num_raw": maxspeed_raw,
        "maxspeed_num": maxspeed,
        "maxspeed_is_imputed": maxspeed_raw is None and maxspeed is not None,
        "lanes_num": lanes,
        "oneway": parse_bool_yes(tags.get("oneway")),
        "lit": parse_bool_yes(tags.get("lit")),
        "bridge": parse_bool_yes(tags.get("bridge")),
        "tunnel": parse_bool_yes(tags.get("tunnel")),
        "surface": surface,
        "is_major": highway in MAJOR_ROAD_CLASSES,
        "is_link": highway in LINK_ROAD_CLASSES,
        "is_bad_surface": surface in BAD_SURFACES,
        "is_asphalt_surface": surface in ASPHALT_SURFACES,
        "nodes_json": json.dumps(way.get("nodes", []), ensure_ascii=False),
        "coords_json": json.dumps(coords, ensure_ascii=False),
        "tags_json": json.dumps(tags, ensure_ascii=False),
    }


def build_raw_roads(
    ways: list[dict[str, Any]],
    nodes: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for way in ways:
        row = row_base_for_way(way, nodes)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def allocate_way_to_h3(
    way: dict[str, Any],
    nodes: dict[int, dict[str, Any]],
    h3_resolution: int = H3_RESOLUTION,
    sample_step_m: float = SAMPLE_STEP_M,
) -> list[dict[str, Any]]:
    base = row_base_for_way(way, nodes)
    if base is None:
        return []

    coords = way_coords(way, nodes)
    by_h3: dict[str, float] = defaultdict(float)

    for (lat1, lng1), (lat2, lng2) in zip(coords, coords[1:]):
        segment_len = haversine_m(lat1, lng1, lat2, lng2)
        if segment_len <= 0:
            continue
        pieces = max(1, math.ceil(segment_len / sample_step_m))
        piece_len = segment_len / pieces
        for i in range(pieces):
            fraction = (i + 0.5) / pieces
            lat, lng = interpolate(lat1, lng1, lat2, lng2, fraction)
            by_h3[latlng_to_h3_cell(lat, lng, h3_resolution)] += piece_len

    rows = []
    for cell, length_m in by_h3.items():
        rows.append({
            "h3": cell,
            "osm_id": base["osm_id"],
            "highway": base["highway"],
            "segment_length_m": length_m,
            "maxspeed_num_raw": base["maxspeed_num_raw"],
            "maxspeed_num": base["maxspeed_num"],
            "maxspeed_is_imputed": base["maxspeed_is_imputed"],
            "lanes_num": base["lanes_num"],
            "oneway": base["oneway"],
            "lit": base["lit"],
            "bridge": base["bridge"],
            "tunnel": base["tunnel"],
            "surface": base["surface"],
            "is_major": base["is_major"],
            "is_link": base["is_link"],
            "is_bad_surface": base["is_bad_surface"],
            "is_asphalt_surface": base["is_asphalt_surface"],
        })
    return rows


def build_h3_segments(
    ways: list[dict[str, Any]],
    nodes: dict[int, dict[str, Any]],
    h3_resolution: int = H3_RESOLUTION,
) -> pd.DataFrame:
    rows = []
    for way in ways:
        rows.extend(allocate_way_to_h3(way, nodes, h3_resolution=h3_resolution))
    return pd.DataFrame(rows)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return 0.0
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def road_class_delta(classes: list[str]) -> int:
    ranks = [ROAD_CLASS_RANK[cls] for cls in classes if cls in ROAD_CLASS_RANK]
    if len(ranks) < 2:
        return 0
    return max(ranks) - min(ranks)


def build_point_nodes(
    tagged_nodes: list[dict[str, Any]],
    h3_resolution: int = H3_RESOLUTION,
) -> pd.DataFrame:
    rows = []
    for node in tagged_nodes:
        tags = node.get("tags", {})
        highway = tags.get("highway")
        railway = tags.get("railway")
        public_transport = tags.get("public_transport")
        traffic_calming = tags.get("traffic_calming")
        if not is_relevant_point_tags(tags):
            continue
        lat = float(node["lat"])
        lng = float(node["lon"])
        rows.append({
            "node_id": int(node["id"]),
            "h3": latlng_to_h3_cell(lat, lng, h3_resolution),
            "lat": lat,
            "lng": lng,
            "highway": highway,
            "railway": railway,
            "public_transport": public_transport,
            "crossing": tags.get("crossing"),
            "bus": tags.get("bus"),
            "tram": tags.get("tram"),
            "traffic_calming": traffic_calming,
            "is_traffic_signal": highway == "traffic_signals",
            "tags_json": json.dumps(tags, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def traffic_signal_points(tagged_nodes: list[dict[str, Any]] | None) -> list[tuple[float, float]]:
    if not tagged_nodes:
        return []
    points = []
    for node in tagged_nodes:
        if node.get("tags", {}).get("highway") == "traffic_signals":
            points.append((float(node["lat"]), float(node["lon"])))
    return points


def nearest_signal_distance_m(
    lat: float,
    lng: float,
    signals: list[tuple[float, float]],
    max_radius_m: float = TRAFFIC_SIGNAL_RADIUS_M,
) -> float | None:
    nearest: float | None = None
    for signal_lat, signal_lng in signals:
        distance = haversine_m(lat, lng, signal_lat, signal_lng)
        if distance <= max_radius_m and (nearest is None or distance < nearest):
            nearest = distance
    return nearest


def build_graph_node_stats(
    ways: list[dict[str, Any]],
    nodes: dict[int, dict[str, Any]],
    tagged_nodes: list[dict[str, Any]] | None = None,
    h3_resolution: int = H3_RESOLUTION,
    signal_radius_m: float = TRAFFIC_SIGNAL_RADIUS_M,
) -> pd.DataFrame:
    node_classes: dict[int, list[str]] = defaultdict(list)
    node_way_ids: dict[int, set[int]] = defaultdict(set)
    node_names: dict[int, set[str]] = defaultdict(set)
    signals = traffic_signal_points(tagged_nodes)

    for way in ways:
        tags = way.get("tags", {})
        highway = tags.get("highway")
        if highway not in MOTOR_ROAD_CLASSES:
            continue
        name = tags.get("name")
        for node_id in way.get("nodes", []):
            node_id = int(node_id)
            node_classes[node_id].append(highway)
            node_way_ids[node_id].add(int(way["id"]))
            if name:
                node_names[node_id].add(str(name))

    rows = []
    for node_id, classes in node_classes.items():
        unique_way_count = len(node_way_ids[node_id])
        if unique_way_count < 2:
            continue
        node = nodes.get(node_id)
        if node is None:
            continue
        degree = len(classes)
        class_set = set(classes)
        names = node_names[node_id]
        major_count = sum(1 for cls in classes if cls in MAJOR_ROAD_CLASSES)
        minor_count = sum(1 for cls in classes if cls not in MAJOR_ROAD_CLASSES)
        is_three_plus_way = unique_way_count >= 3
        is_class_change = len(class_set) > 1
        is_name_change = len(names) > 1
        legacy_major_minor = major_count > 0 and minor_count > 0
        class_delta = road_class_delta(classes)
        is_class_delta = class_delta >= 1
        lat = float(node["lat"])
        lng = float(node["lon"])
        nearest_signal_m = nearest_signal_distance_m(
            lat,
            lng,
            signals,
            max_radius_m=signal_radius_m,
        )
        has_signal_nearby = nearest_signal_m is not None
        is_minor_to_major = is_class_delta and not has_signal_nearby

        # Two OSM ways with the same class/name are often just technical street
        # segmentation. Keep only display-level intersections used by the viewer.
        if not (is_three_plus_way or legacy_major_minor or is_class_change or is_name_change):
            continue

        cell = latlng_to_h3_cell(lat, lng, h3_resolution)
        rows.append({
            "h3": cell,
            "node_id": node_id,
            "junction_degree": degree,
            "junction_way_count": unique_way_count,
            "is_three_way": degree == 3,
            "is_four_plus_way": degree >= 4,
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
        })
    return pd.DataFrame(rows)


def build_point_feature_stats(
    tagged_nodes: list[dict[str, Any]],
    h3_resolution: int = H3_RESOLUTION,
) -> pd.DataFrame:
    rows = []
    for node in tagged_nodes:
        tags = node.get("tags", {})
        highway = tags.get("highway")
        railway = tags.get("railway")
        public_transport = tags.get("public_transport")
        traffic_calming = tags.get("traffic_calming")
        if not is_relevant_point_tags(tags):
            continue
        crossing = tags.get("crossing")
        is_crossing = highway == "crossing"
        is_signalized_crossing = is_crossing and crossing in {"traffic_signals", "traffic_signal", "signals"}
        is_unsignalized_crossing = is_crossing and not is_signalized_crossing
        is_bus_stop = highway == "bus_stop" or (
            public_transport in {"platform", "stop_position"} and parse_bool_yes(tags.get("bus"))
        )
        is_tram_stop = railway == "tram_stop" or (
            public_transport in {"platform", "stop_position"} and parse_bool_yes(tags.get("tram"))
        )
        is_public_transport_stop = is_bus_stop or is_tram_stop or public_transport in {"platform", "stop_position"}
        is_railway_crossing = railway in {"level_crossing", "crossing"}
        rows.append({
            "h3": latlng_to_h3_cell(float(node["lat"]), float(node["lon"]), h3_resolution),
            "traffic_signal_count": int(highway == "traffic_signals"),
            "crossing_count": int(is_crossing),
            "crossing_signalized_count": int(is_signalized_crossing),
            "crossing_unsignalized_count": int(is_unsignalized_crossing),
            "stop_count": int(highway == "stop"),
            "give_way_count": int(highway == "give_way"),
            "mini_roundabout_count": int(highway == "mini_roundabout"),
            "motorway_junction_count": int(highway == "motorway_junction"),
            "traffic_calming_count": int(bool(traffic_calming)),
            "public_transport_stop_count": int(is_public_transport_stop),
            "bus_stop_count": int(is_bus_stop),
            "tram_stop_count": int(is_tram_stop),
            "railway_crossing_count": int(is_railway_crossing),
        })
    return pd.DataFrame(rows)


def aggregate_h3_road_features(
    h3_segments: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    point_features: pd.DataFrame,
) -> pd.DataFrame:
    if h3_segments.empty:
        raise RuntimeError("No H3 road segments were produced.")

    grouped_rows = []
    for cell, g in h3_segments.groupby("h3", sort=True):
        total = float(g["segment_length_m"].sum())
        row: dict[str, Any] = {
            "h3": cell,
            "road_length_total_m": total,
            "road_way_count": int(g["osm_id"].nunique()),
            "road_segment_h3_count": int(len(g)),
            "major_road_length_m": float(g.loc[g["is_major"], "segment_length_m"].sum()),
            "link_road_length_m": float(g.loc[g["is_link"], "segment_length_m"].sum()),
            "oneway_length_m": float(g.loc[g["oneway"], "segment_length_m"].sum()),
            "lit_road_length_m": float(g.loc[g["lit"], "segment_length_m"].sum()),
            "bridge_length_m": float(g.loc[g["bridge"], "segment_length_m"].sum()),
            "tunnel_length_m": float(g.loc[g["tunnel"], "segment_length_m"].sum()),
            "bad_surface_length_m": float(g.loc[g["is_bad_surface"], "segment_length_m"].sum()),
            "asphalt_surface_length_m": float(g.loc[g["is_asphalt_surface"], "segment_length_m"].sum()),
            "maxspeed_mean_weighted": weighted_mean(g["maxspeed_num"], g["segment_length_m"]),
            "maxspeed_max": float(g["maxspeed_num"].max()) if g["maxspeed_num"].notna().any() else 0.0,
            "lanes_mean_weighted": weighted_mean(g["lanes_num"], g["segment_length_m"]),
            "lanes_max": float(g["lanes_num"].max()) if g["lanes_num"].notna().any() else 0.0,
            "lane_meters_estimate": float((g["segment_length_m"] * g["lanes_num"].fillna(1)).sum()),
            "maxspeed_tagged_share": float(g.loc[g["maxspeed_num_raw"].notna(), "segment_length_m"].sum() / total),
            "maxspeed_is_imputed_share": float(g.loc[g["maxspeed_is_imputed"], "segment_length_m"].sum() / total),
            "lanes_tagged_share": float(g.loc[g["lanes_num"].notna(), "segment_length_m"].sum() / total),
        }
        class_lengths = Counter()
        for highway, class_group in g.groupby("highway"):
            class_lengths[highway] = float(class_group["segment_length_m"].sum())
        for highway in sorted(MOTOR_ROAD_CLASSES):
            safe_name = highway.replace("_link", "_link")
            length_m = class_lengths.get(highway, 0.0)
            row[f"road_length_{safe_name}_m"] = length_m
            row[f"road_share_{safe_name}"] = length_m / total if total > 0 else 0.0
        grouped_rows.append(row)

    features = pd.DataFrame(grouped_rows)
    features["major_road_share"] = features["major_road_length_m"] / features["road_length_total_m"]
    features["link_road_share"] = features["link_road_length_m"] / features["road_length_total_m"]
    features["oneway_share"] = features["oneway_length_m"] / features["road_length_total_m"]
    features["lit_share"] = features["lit_road_length_m"] / features["road_length_total_m"]
    features["bridge_share"] = features["bridge_length_m"] / features["road_length_total_m"]
    features["tunnel_share"] = features["tunnel_length_m"] / features["road_length_total_m"]
    features["bad_surface_share"] = features["bad_surface_length_m"] / features["road_length_total_m"]
    features["asphalt_surface_share"] = features["asphalt_surface_length_m"] / features["road_length_total_m"]

    if not graph_nodes.empty:
        graph_agg = graph_nodes.groupby("h3").agg(
            junction_count=("node_id", "count"),
            junction_degree_mean=("junction_degree", "mean"),
            junction_degree_max=("junction_degree", "max"),
            three_way_junction_count=("is_three_way", "sum"),
            four_plus_way_junction_count=("is_four_plus_way", "sum"),
            major_junction_count=("is_major_junction", "sum"),
            minor_to_major_junction_count=("is_minor_to_major_junction", "sum"),
            legacy_major_minor_junction_count=("is_legacy_major_minor_junction", "sum"),
            class_delta_junction_count=("is_class_delta_junction", "sum"),
            class_delta_mean=("class_rank_delta", "mean"),
            class_delta_max=("class_rank_delta", "max"),
            class_delta_sum=("class_rank_delta", "sum"),
            signalized_class_delta_junction_count=("is_signalized_class_delta_junction", "sum"),
            unsignalized_class_delta_junction_count=("is_unsignalized_class_delta_junction", "sum"),
            unsignalized_high_delta_junction_count=("is_unsignalized_high_delta_junction", "sum"),
        ).reset_index()
        features = features.merge(graph_agg, on="h3", how="left")

    if not point_features.empty:
        point_agg = point_features.groupby("h3", as_index=False).sum(numeric_only=True)
        features = features.merge(point_agg, on="h3", how="left")

    count_columns = [
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
        "traffic_signal_count",
        "crossing_count",
        "crossing_signalized_count",
        "crossing_unsignalized_count",
        "stop_count",
        "give_way_count",
        "mini_roundabout_count",
        "motorway_junction_count",
        "traffic_calming_count",
        "public_transport_stop_count",
        "bus_stop_count",
        "tram_stop_count",
        "railway_crossing_count",
        "roundabout_count",
    ]
    for column in count_columns:
        if column not in features.columns:
            features[column] = 0
    features[count_columns] = features[count_columns].fillna(0)

    features["junction_density_per_km"] = features["junction_count"] / (features["road_length_total_m"] / 1000)
    features["minor_to_major_junction_density_per_km"] = (
        features["minor_to_major_junction_count"] / (features["road_length_total_m"] / 1000)
    )
    features["class_delta_junction_density_per_km"] = (
        features["class_delta_junction_count"] / (features["road_length_total_m"] / 1000)
    )
    features["unsignalized_class_delta_junction_density_per_km"] = (
        features["unsignalized_class_delta_junction_count"] / (features["road_length_total_m"] / 1000)
    )
    road_length_km = features["road_length_total_m"] / 1000
    features["traffic_signal_density_per_km"] = features["traffic_signal_count"] / road_length_km
    features["crossing_density_per_km"] = features["crossing_count"] / road_length_km
    features["crossing_signalized_density_per_km"] = features["crossing_signalized_count"] / road_length_km
    features["crossing_unsignalized_density_per_km"] = features["crossing_unsignalized_count"] / road_length_km
    features["priority_sign_count"] = features["stop_count"] + features["give_way_count"]
    features["priority_sign_density_per_km"] = features["priority_sign_count"] / road_length_km
    features["stop_density_per_km"] = features["stop_count"] / road_length_km
    features["give_way_density_per_km"] = features["give_way_count"] / road_length_km
    features["stop_share"] = features["stop_count"] / features["priority_sign_count"]
    features["give_way_share"] = features["give_way_count"] / features["priority_sign_count"]
    features["traffic_calming_density_per_km"] = features["traffic_calming_count"] / road_length_km
    features["public_transport_stop_density_per_km"] = features["public_transport_stop_count"] / road_length_km
    features["railway_crossing_density_per_km"] = features["railway_crossing_count"] / road_length_km
    features["mini_roundabout_density_per_km"] = features["mini_roundabout_count"] / road_length_km
    features["roundabout_total_count"] = features["mini_roundabout_count"] + features["roundabout_count"]
    features["roundabout_total_density_per_km"] = features["roundabout_total_count"] / road_length_km
    features["mini_roundabout_share"] = features["mini_roundabout_count"] / features["roundabout_total_count"]
    features["full_roundabout_share"] = features["roundabout_count"] / features["roundabout_total_count"]
    features["roundabout_density_per_km"] = features["roundabout_count"] / road_length_km
    features["motorway_junction_density_per_km"] = features["motorway_junction_count"] / road_length_km
    features["crossing_signalized_share"] = features["crossing_signalized_count"] / features["crossing_count"]
    features["crossing_unsignalized_share"] = features["crossing_unsignalized_count"] / features["crossing_count"]
    features["bus_stop_share"] = features["bus_stop_count"] / features["public_transport_stop_count"]
    features["tram_stop_share"] = features["tram_stop_count"] / features["public_transport_stop_count"]
    share_density_columns = [
        "traffic_signal_density_per_km",
        "crossing_density_per_km",
        "crossing_signalized_density_per_km",
        "crossing_unsignalized_density_per_km",
        "priority_sign_density_per_km",
        "stop_density_per_km",
        "give_way_density_per_km",
        "stop_share",
        "give_way_share",
        "traffic_calming_density_per_km",
        "public_transport_stop_density_per_km",
        "railway_crossing_density_per_km",
        "mini_roundabout_density_per_km",
        "roundabout_total_density_per_km",
        "mini_roundabout_share",
        "full_roundabout_share",
        "roundabout_density_per_km",
        "motorway_junction_density_per_km",
        "crossing_signalized_share",
        "crossing_unsignalized_share",
        "bus_stop_share",
        "tram_stop_share",
    ]
    features[share_density_columns] = (
        features[share_density_columns]
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
    )

    return features.sort_values("h3").reset_index(drop=True)


def build_h3_month_road_features(
    h3_features: pd.DataFrame,
    panel_path: Path = H3_MONTH_PANEL_PARQUET,
    output_path: Path = H3_MONTH_ROAD_FEATURES_PARQUET,
) -> pd.DataFrame:
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")

    panel = pd.read_parquet(panel_path)
    base_columns = ["h3", "month", "year", "month_num", TARGET]
    missing = [column for column in base_columns if column not in panel.columns]
    if missing:
        raise ValueError(f"Missing columns in panel: {missing}")

    result = panel[base_columns].merge(h3_features, on="h3", how="left")
    feature_columns = [column for column in h3_features.columns if column != "h3"]
    result[feature_columns] = result[feature_columns].fillna(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def build_osm_road_features(
    input_json: Path = OSM_SARATOV_ROADS_JSON,
    h3_resolution: int = H3_RESOLUTION,
) -> None:
    if not input_json.exists():
        raise FileNotFoundError(f"OSM JSON not found: {input_json}")

    nodes, ways, tagged_nodes = load_overpass(input_json)
    print(f"OSM nodes: {len(nodes):,}")
    print(f"OSM ways: {len(ways):,}")
    print(f"Tagged road nodes: {len(tagged_nodes):,}")

    raw_roads = build_raw_roads(ways, nodes)
    h3_segments = build_h3_segments(ways, nodes, h3_resolution=h3_resolution)
    graph_nodes = build_graph_node_stats(ways, nodes, tagged_nodes=tagged_nodes, h3_resolution=h3_resolution)
    point_nodes = build_point_nodes(tagged_nodes, h3_resolution=h3_resolution)
    point_features = build_point_feature_stats(tagged_nodes, h3_resolution=h3_resolution)
    h3_features = aggregate_h3_road_features(h3_segments, graph_nodes, point_features)
    h3_month_features = build_h3_month_road_features(h3_features)

    raw_roads.to_parquet(OSM_ROADS_RAW_PARQUET, index=False)
    h3_segments.to_parquet(OSM_ROAD_H3_SEGMENTS_PARQUET, index=False)
    graph_nodes.to_parquet(OSM_ROAD_GRAPH_NODES_PARQUET, index=False)
    point_nodes.to_parquet(OSM_ROAD_POINT_NODES_PARQUET, index=False)
    h3_features.to_parquet(H3_ROAD_FEATURES_PARQUET, index=False)

    print()
    print("Saved parquet files:")
    for path, df in [
        (OSM_ROADS_RAW_PARQUET, raw_roads),
        (OSM_ROAD_H3_SEGMENTS_PARQUET, h3_segments),
        (OSM_ROAD_GRAPH_NODES_PARQUET, graph_nodes),
        (OSM_ROAD_POINT_NODES_PARQUET, point_nodes),
        (H3_ROAD_FEATURES_PARQUET, h3_features),
        (H3_MONTH_ROAD_FEATURES_PARQUET, h3_month_features),
    ]:
        print(f"{path} | rows={len(df):,} | size={format_size(path.stat().st_size)}")


def build_osm_road_features_from_pbf(
    pbf_path: Path = OSM_VOLGA_PBF,
    bbox: tuple[float, float, float, float] = BBOX,
    h3_resolution: int = H3_RESOLUTION,
) -> None:
    nodes, ways, tagged_nodes = load_pbf_roads(pbf_path=pbf_path, bbox=bbox)

    raw_roads = build_raw_roads(ways, nodes)
    h3_segments = build_h3_segments(ways, nodes, h3_resolution=h3_resolution)
    graph_nodes = build_graph_node_stats(ways, nodes, tagged_nodes=tagged_nodes, h3_resolution=h3_resolution)
    point_nodes = build_point_nodes(tagged_nodes, h3_resolution=h3_resolution)
    point_features = build_point_feature_stats(tagged_nodes, h3_resolution=h3_resolution)
    h3_features = aggregate_h3_road_features(h3_segments, graph_nodes, point_features)
    h3_month_features = build_h3_month_road_features(h3_features)

    raw_roads.to_parquet(OSM_ROADS_RAW_PARQUET, index=False)
    h3_segments.to_parquet(OSM_ROAD_H3_SEGMENTS_PARQUET, index=False)
    graph_nodes.to_parquet(OSM_ROAD_GRAPH_NODES_PARQUET, index=False)
    point_nodes.to_parquet(OSM_ROAD_POINT_NODES_PARQUET, index=False)
    h3_features.to_parquet(H3_ROAD_FEATURES_PARQUET, index=False)

    print()
    print("Saved parquet files:")
    for path, df in [
        (OSM_ROADS_RAW_PARQUET, raw_roads),
        (OSM_ROAD_H3_SEGMENTS_PARQUET, h3_segments),
        (OSM_ROAD_GRAPH_NODES_PARQUET, graph_nodes),
        (OSM_ROAD_POINT_NODES_PARQUET, point_nodes),
        (H3_ROAD_FEATURES_PARQUET, h3_features),
        (H3_MONTH_ROAD_FEATURES_PARQUET, h3_month_features),
    ]:
        print(f"{path} | rows={len(df):,} | size={format_size(path.stat().st_size)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and aggregate OSM road structure features to H3.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--output", type=Path, default=OSM_SARATOV_ROADS_JSON)
    download_parser.add_argument("--bbox", default=",".join(map(str, BBOX)))
    download_parser.add_argument("--overpass-url", default=OVERPASS_URL)

    build_parser_ = subparsers.add_parser("build")
    build_parser_.add_argument("--input-json", type=Path, default=OSM_SARATOV_ROADS_JSON)
    build_parser_.add_argument("--h3-resolution", type=int, default=H3_RESOLUTION)

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--input-json", type=Path, default=OSM_SARATOV_ROADS_JSON)
    all_parser.add_argument("--bbox", default=",".join(map(str, BBOX)))
    all_parser.add_argument("--overpass-url", default=OVERPASS_URL)
    all_parser.add_argument("--h3-resolution", type=int, default=H3_RESOLUTION)

    build_pbf_parser = subparsers.add_parser("build-pbf")
    build_pbf_parser.add_argument("--pbf", type=Path, default=OSM_VOLGA_PBF)
    build_pbf_parser.add_argument("--bbox", default=",".join(map(str, BBOX)))
    build_pbf_parser.add_argument("--h3-resolution", type=int, default=H3_RESOLUTION)

    return parser


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be MIN_LAT,MIN_LNG,MAX_LAT,MAX_LNG")
    return parts[0], parts[1], parts[2], parts[3]


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        download_overpass_json(
            output_path=args.output,
            bbox=parse_bbox(args.bbox),
            overpass_url=args.overpass_url,
        )
    elif args.command == "build":
        build_osm_road_features(
            input_json=args.input_json,
            h3_resolution=args.h3_resolution,
        )
    elif args.command == "all":
        download_overpass_json(
            output_path=args.input_json,
            bbox=parse_bbox(args.bbox),
            overpass_url=args.overpass_url,
        )
        build_osm_road_features(
            input_json=args.input_json,
            h3_resolution=args.h3_resolution,
        )
    elif args.command == "build-pbf":
        build_osm_road_features_from_pbf(
            pbf_path=args.pbf,
            bbox=parse_bbox(args.bbox),
            h3_resolution=args.h3_resolution,
        )


if __name__ == "__main__":
    main()
