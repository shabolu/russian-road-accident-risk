"""Build directed car routing graphs from the project's June 2026 OSM snapshot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import osmium
import pandas as pd
from pyproj import Geod
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_accident_risk.route_graph import (  # noqa: E402
    BASE_SPEEDS_KMH,
    SERVICE_SPEEDS_KMH,
    access_rule,
    node_blocked,
    oneway_rule,
    turn_rule,
)

SOURCE = ROOT / "reports" / "multicity_350k"
OUTPUT = ROOT / "reports" / "route_proxy_experiments" / "graph_no_service"
PBF = ROOT / "data" / "raw" / "osm" / "volga-fed-district-260603.osm.pbf"
GEOD = Geod(ellps="WGS84")
LOCAL_SERVICE = {"driveway", "parking_aisle", "drive-through", "drive-thru"}
SERVICE_MULTIPLIERS = {"alley": 1.35, "driveway": 1.5, "parking_aisle": 1.5,
                       "drive-through": 1.5, "drive-thru": 1.5}
BARRIER_KEYS = ("barrier", "access", "vehicle", "motor_vehicle", "motorcar",
                "access:conditional", "vehicle:conditional", "motor_vehicle:conditional",
                "motorcar:conditional")
CONDITIONAL_PREFIXES = ("access:", "vehicle:", "motor_vehicle:", "motorcar:", "oneway:")


def profile(path):
    """Read optional overrides without implicitly changing access rules."""
    overrides = json.loads(path.read_text()) if path else {}
    speeds = BASE_SPEEDS_KMH | overrides.get("highway_speed_kmh", {})
    service = SERVICE_SPEEDS_KMH | overrides.get("service_speed_kmh", {})
    multipliers = SERVICE_MULTIPLIERS | overrides.get("service_multiplier", {})
    if any(not np.isfinite(x) or x <= 0 for x in (*speeds.values(), *service.values(), *multipliers.values())):
        raise ValueError("Profile parameters must be positive and finite")
    return speeds, service, multipliers


def city_edges(city, config, include_service=False):
    speeds, service_speeds, multipliers = config
    table = pd.read_parquet(SOURCE / "cache" / f"{city['slug']}_roads.parquet",
                            columns=["osm_id", "nodes_json", "coords_json", "tags_json"])
    south, west, north, east = city["extract_bbox"]
    data, coords, nodes = [], [], {}
    audit = Counter()
    for row in table.sort_values("osm_id").itertuples(index=False):
        tags = json.loads(row.tags_json)
        highway = tags["highway"]
        if highway not in BASE_SPEEDS_KMH:
            continue
        if highway == "service" and not include_service:
            audit["ways_excluded_service"] += 1
            continue
        service = tags.get("service", "") if highway == "service" else ""
        audit["cache_ways"] += 1
        if any(key.startswith(CONDITIONAL_PREFIXES)
               and key.endswith(":conditional") for key in tags):
            audit["ways_with_unsupported_conditional_access_or_direction"] += 1
        if (service == "emergency_access" or tags.get("smoothness") == "impassable"
                or tags.get("area") == "yes" or tags.get("construction") == "yes"):
            audit["ways_excluded_by_access_or_geometry"] += 1
            continue
        direction = oneway_rule(tags)
        if direction is None:
            audit["ways_excluded_by_reversible_or_alternating_direction"] += 1
            continue
        allowed = [(d, access_rule(tags, d)) for d in ((1, -1) if direction == 0 else (direction,))]
        allowed = [(d, access) for d, access in allowed if access != "blocked"]
        if not allowed:
            audit["ways_excluded_by_access_or_geometry"] += 1
            continue

        ids = json.loads(row.nodes_json)
        xy = np.asarray(json.loads(row.coords_json), dtype=float)[:, ::-1]
        if len(ids) != len(xy) or len(ids) < 2:
            raise ValueError(f"Inconsistent geometry of OSM way {row.osm_id}")
        a, b = xy[:-1], xy[1:]
        lengths = GEOD.inv(a[:, 0], a[:, 1], b[:, 0], b[:, 1])[2]
        cumulative = np.r_[0, np.cumsum(lengths)]
        if cumulative[-1] <= 0:
            audit["zero_length_ways"] += 1
            continue
        fractions = cumulative / cumulative[-1]
        keep = ((np.maximum(a[:, 0], b[:, 0]) >= west) & (np.minimum(a[:, 0], b[:, 0]) <= east)
                & (np.maximum(a[:, 1], b[:, 1]) >= south) & (np.minimum(a[:, 1], b[:, 1]) <= north)
                & (lengths > 0))
        base_speed = service_speeds.get(service, speeds[highway]) if highway == "service" else speeds[highway]
        multiplier = multipliers.get(service, 1.0) if highway == "service" else 1.0
        local_service = service in LOCAL_SERVICE
        for i in np.flatnonzero(keep):
            for d, access in allowed:
                lo, hi = (i, i + 1) if d == 1 else (i + 1, i)
                a_id, b_id = ids[lo], ids[hi]
                ab = xy[[lo, hi]]
                nodes[a_id] = tuple(ab[0])
                nodes[b_id] = tuple(ab[1])
                data.append((row.osm_id, a_id, b_id, lengths[i], highway, service, fractions[lo],
                             fractions[hi], d, access == "local", local_service, base_speed, multiplier,
                             tags.get("name", ""), tags.get("bridge", ""), tags.get("tunnel", "")))
                coords.append(ab)
    columns = ["way_id", "u", "v", "length_m", "highway", "service", "start_frac", "end_frac",
               "direction", "local_access", "local_service", "base_speed_kmh", "service_multiplier",
               "name", "bridge", "tunnel"]
    edges = pd.DataFrame.from_records(data, columns=columns)
    if not len(edges):
        raise ValueError(f"No road edges in {city['slug']}")
    edges.insert(0, "edge_id", np.arange(len(edges), dtype=np.int64))
    edges["is_local"] = edges.local_access | edges.local_service
    edges["base_cost_s"] = 3.6 * edges.length_m / edges.base_speed_kmh * edges.service_multiplier
    edges["geometry"] = shapely.to_wkb(shapely.linestrings(np.asarray(coords)))
    node_table = pd.DataFrame.from_records(
        [(key, *value) for key, value in sorted(nodes.items())], columns=["node_id", "lon", "lat"])
    return edges, node_table, audit


def read_controls(pbf, node_membership, way_membership, city_data):
    entities = osmium.osm.NODE | osmium.osm.RELATION
    proc = (osmium.FileProcessor(str(pbf)).with_filter(osmium.filter.EntityFilter(entities))
            .with_filter(osmium.filter.KeyFilter(*BARRIER_KEYS, "type")))
    for item in proc:
        if item.is_node():
            cities = node_membership.get(item.id)
            if cities:
                tags = dict(item.tags)
                info = dict(node_id=item.id, barrier=tags.get("barrier", ""), node_blocked=node_blocked(tags))
                conditional = any(key.endswith(":conditional") for key in tags.keys() & set(BARRIER_KEYS))
                for slug in cities:
                    _, _, audit, controls, _ = city_data[slug]
                    if info["barrier"] or info["node_blocked"]:
                        controls.append(info)
                    if conditional:
                        audit["nodes_with_unsupported_conditional_access"] += 1
        elif item.tags.get("type") == "restriction" or item.tags.get("type", "").startswith("restriction:"):
            members = [(m.type, m.ref, m.role) for m in item.members]
            relates_to = set()
            for kind, ref, role in members:
                if kind == "w" and role in {"from", "to", "via"}:
                    relates_to.update(way_membership.get(ref, ()))
            if not relates_to:
                continue
            tags = dict(item.tags)
            restrictions, issues = turn_rule(tags, members)
            for slug in relates_to:
                _, _, audit, _, local_restrictions = city_data[slug]
                audit.update(issues)
                for restriction in restrictions:
                    local_restrictions.append(dict(relation_id=item.id, **restriction))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", "--cities", nargs="+", dest="cities", default=["saratov"],
                        help="OSM city slugs, or all. Scans the regional PBF once per invocation.")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-service", action="store_true", help="Reproduce the historical graph")
    args = parser.parse_args()
    if args.include_service and args.output is None:
        parser.error("Use a separate --output to reproduce the historical graph")
    output = args.output or OUTPUT
    cities = {c["slug"]: c for c in json.loads((SOURCE / "cities.json").read_text())}
    selected = sorted(cities) if args.cities == ["all"] else list(dict.fromkeys(args.cities))
    if missing := set(selected) - cities.keys():
        parser.error(f"Unknown city slugs: {sorted(missing)}")
    config = profile(args.profile)
    outputs = {}
    node_membership, way_membership = {}, {}
    for slug in selected:
        edges, nodes, audit = city_edges(cities[slug], config, args.include_service)
        for node_id in nodes.node_id:
            node_membership.setdefault(node_id, []).append(slug)
        for way_id in edges.way_id.unique():
            way_membership.setdefault(way_id, []).append(slug)
        outputs[slug] = (edges, nodes, audit, [], [])
        print(f"{slug}: {len(edges):,} edges and {len(nodes):,} nodes", flush=True)
    print("Reading barriers and turn restrictions from the regional PBF", flush=True)
    read_controls(PBF, node_membership, way_membership, outputs)
    for slug, (edges, nodes, audit, controls, turns) in outputs.items():
        to_edges = set(zip(edges.way_id, edges.u))
        from_edges = set(zip(edges.way_id, edges.v))
        matched = [t for t in turns if (t["from_way"], t["via_node"]) in from_edges and (
            t["kind"].startswith("only_") or (t["to_way"], t["via_node"]) in to_edges
        )]
        audit["only_rules_with_unavailable_departure"] = sum(
            (t["to_way"], t["via_node"]) not in to_edges for t in matched
        )
        audit["via_node_rules_retained"] = len(matched)
        audit["via_node_rules_without_both_routable_ways"] = len(turns) - len(matched)
        metadata = {
            "city": slug,
            "pbf": str(PBF.relative_to(ROOT)),
            "bbox_south_west_north_east": cities[slug]["extract_bbox"],
            "edge_count": len(edges),
            "include_service": args.include_service,
            "node_count": len(nodes),
            "blocked_barrier_nodes": sum(bool(c["node_blocked"]) for c in controls),
            "edges_local_access": int(edges.local_access.sum()),
            "edges_local_service": int(edges.local_service.sum()),
            "base_speed_kmh": config[0],
            "service_speed_kmh": config[1],
            "service_multiplier": config[2],
            "assumptions": [
                "Experimental car profile; speeds are assumed free-flow values.",
                "Source OSM motor-road classes only; no ferry routing or unclassified track access.",
                "Conditional, vehicle-dimension, and via-way controls are not applied.",
                "An untagged gate or lift_gate is treated as passable in the static profile.",
                "Only explicit destination access and selected service subtypes get endpoint-only access.",
                "Private, customer, permit and delivery access require permissions outside this profile.",
                "Via-node turn legality is enforced; no per-maneuver time delay is applied.",
            ],
            "audit": dict(audit),
        }
        target = output / slug
        target.mkdir(parents=True, exist_ok=True)
        edges.to_parquet(target / "edges.parquet", index=False)
        nodes.to_parquet(target / "nodes.parquet", index=False)
        (target / "controls.json").write_text(json.dumps({"nodes": controls, "turns": matched}, ensure_ascii=False))
        (target / "graph_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        print(slug, json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
