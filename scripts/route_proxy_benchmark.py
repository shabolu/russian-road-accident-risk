"""Generate deterministic validation samples and run tagged Saratov routing checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/route_proxy_experiments"
BENCH = REPORT / "benchmark"
COVERAGE = ROOT / "reports/route_proxy_coverage/ways"
CACHE = ROOT / "reports/multicity_350k/cache"
FIELDS = (
    "combined_speed_length_share",
    "combined_surface_coarse_length_share",
    "combined_lanes_length_share",
    "combined_width_length_share",
    "osm_smoothness_length_share",
)
SEED = 20261124


def road_class(highway):
    if highway in {"residential", "unclassified", "service"}:
        return highway
    return "other"


def choose_rows(frame, strata, keys, limit):
    """Keep the same slice when source file row order changes."""
    frame = frame.copy()
    key_text = frame[keys].astype(str).agg("|".join, axis=1)
    frame["sample_key"] = key_text.map(
        lambda value: hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()
    )
    frame["stratum_population"] = frame.groupby(strata)["sample_key"].transform("size")
    return (
        frame.sort_values("sample_key")
        .groupby(strata, sort=True, group_keys=False)
        .head(limit)
        .reset_index(drop=True)
    )


def review_samples(city, out, limit):
    coverage = pd.read_parquet(COVERAGE / f"{city}.parquet")
    roads = pd.read_parquet(CACHE / f"{city}_roads.parquet", columns=["osm_id", "coords_json"])
    missing = coverage.loc[~coverage[list(FIELDS)].ge(1 - 1e-6).any(axis=1)].copy()
    missing["road_group"] = missing.highway.map(road_class)
    missing["coverage_group"] = missing[list(FIELDS)].gt(0).any(axis=1).map(
        {True: "partial", False: "no_observations"}
    )
    sample = choose_rows(missing, ["road_group", "coverage_group"], ["osm_id"], limit)
    sample = sample.merge(roads, on="osm_id", how="left", validate="one_to_one")
    # Provide the full way for inspection. The midpoint can fall beyond the study polygon.
    for column in ("review_result", "review_source", "review_notes"):
        sample[column] = ""
    sample.to_csv(out / "missing_way_review.csv", index=False)

    candidate_path = REPORT / "observations" / city / "review_candidates.parquet"
    result = {"seed": SEED, "per_stratum": limit, "missing_ways": len(missing),
              "sampled_missing_ways": len(sample), "candidate_status": "pending"}
    if candidate_path.exists():
        candidates = pd.read_parquet(candidate_path)
        candidates["road_group"] = candidates.highway.map(road_class)
        candidates = candidates.sort_values(
            ["way_id", "reason", "start_frac", "end_frac", "lon", "lat", "passport_id"],
            kind="stable",
        )
        # Pick the middle candidate interval within each way/reason before sampling ways.
        groups = candidates.groupby(["way_id", "reason"], sort=True).indices
        middle_rows = [rows[len(rows) // 2] for rows in groups.values()]
        ways = candidates.iloc[middle_rows]
        match_sample = choose_rows(ways, ["road_group", "reason"], ["way_id", "reason"], limit)
        for column in ("review_result", "review_source", "review_notes"):
            match_sample[column] = ""
        match_sample.to_csv(out / "match_candidate_review.csv", index=False)
        result.update(candidate_status="ready", candidate_way_reasons=len(ways),
                      sampled_candidates=len(match_sample))
    (out / "review_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_route_case(graph, case, edge_cost_s):
    route = graph.shortest_path(case["source"], case["target"], edge_cost_s=edge_cost_s)
    if route is None:
        return case["reachable"] is not True, []
    rows = graph.edges.iloc[list(route.edge_ids)]
    nodes = [case["source"], *rows.v.astype(int)]
    ways = rows.way_id.astype(int).tolist()
    connected = (
        (nodes[:-1] == rows.u.astype(int).tolist()) and nodes[-1] == case["target"]
    )
    tests = [
        case["reachable"] is not False,
        connected,
        all(node not in nodes[1:-1] for node in case.get("forbidden_internal_nodes", [])),
        not set(ways).intersection(case.get("forbidden_ways", [])),
        set(case.get("required_ways", [])).issubset(ways),
    ]
    for turn in case.get("forbidden_turns", []):
        tests.append(not any(
            left == turn["from"] and node == turn["via"] and right == turn["to"]
            for left, node, right in zip(ways[:-1], nodes[1:-1], ways[1:])
        ))
    return all(tests), ways


def verify_topology_case(graph, case):
    if case["kind"] == "edge":
        direct = graph.edges.loc[
            graph.edges.way_id.eq(case["way_id"]) &
            graph.edges.u.eq(case["source"]) & graph.edges.v.eq(case["target"])
        ]
        return bool(len(direct)) == case["present"]
    if case["kind"] == "separate_ways":
        first, second = (graph.edges[graph.edges.way_id.eq(way)] for way in case["way_ids"])
        return bool(len(first) and len(second)) and not (
            set(first.u).union(first.v) & set(second.u).union(second.v)
        )
    raise ValueError(f"Unknown check kind {case['kind']}")


def run_routing_checks(city, destination, graph_dir):
    sys.path.insert(0, str(ROOT / "src"))
    from road_accident_risk.route_graph import Graph

    graph = Graph.load(graph_dir)
    manifest = json.loads((BENCH / f"{city}_topology_cases.json").read_text())
    # Pure distance deliberately ignores the weights of service roads. Restrictions must still hold.
    profiles = {"class_cost": None, "distance_only": graph.edges.length_m.to_numpy()}
    rows = []
    for case in manifest["cases"]:
        if case["kind"] == "route":
            variants = dict(profiles)
            if case.get("policy_probe"):
                # Make the forbidden cut-through attractive while keeping all costs positive.
                cheap = graph.edges.way_id.isin(case["forbidden_ways"])
                variants["cheap_forbidden_cut_through"] = graph.edges.length_m.where(~cheap, 0.001).to_numpy()
            for profile, cost in variants.items():
                passed, ways = verify_route_case(graph, case, cost)
                rows.append({"case": case["id"], "profile": profile,
                             "passed": passed, "way_ids": ways})
        else:
            passed = verify_topology_case(graph, case)
            rows.append({"case": case["id"], "profile": None, "passed": passed})
    (destination / "routing_check_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="saratov")
    parser.add_argument("--per-stratum", default=8, type=int)
    parser.add_argument("--graph-dir", type=Path)
    args = parser.parse_args()
    if args.per_stratum < 1:
        parser.error("--per-stratum must be positive")
    out = BENCH / args.city
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(review_samples(args.city, out, args.per_stratum)))
    if args.graph_dir:
        checks = run_routing_checks(args.city, out, args.graph_dir)
        passed = sum(row["passed"] for row in checks)
        print(f"Routing and topology checks passed: {passed}/{len(checks)}")
        if passed != len(checks):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
