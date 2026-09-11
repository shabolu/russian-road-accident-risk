from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.paths import (
    H3_MONTH_PANEL_PARQUET,
    H3_MONTH_ROAD_FEATURES_PARQUET,
    H3_ROAD_FEATURES_PARQUET,
    OSM_ROAD_GRAPH_NODES_PARQUET,
)


JUNCTION_COLUMNS = [
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
]


def reconcile_junction_features() -> None:
    features = pd.read_parquet(H3_ROAD_FEATURES_PARQUET)
    graph = pd.read_parquet(OSM_ROAD_GRAPH_NODES_PARQUET)
    defaults = {
        "is_legacy_major_minor_junction": False,
        "is_class_delta_junction": False,
        "class_rank_delta": 0,
        "is_signalized_class_delta_junction": False,
        "is_unsignalized_class_delta_junction": False,
        "is_unsignalized_high_delta_junction": False,
    }
    for column, default in defaults.items():
        if column not in graph.columns:
            graph[column] = default

    graph_agg = (
        graph.assign(
            is_three_way=graph["junction_degree"] == 3,
            is_four_plus_way=graph["junction_degree"] >= 4,
        )
        .groupby("h3")
        .agg(
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
        )
        .reset_index()
    )

    old_sum = features["junction_count"].sum()
    features = features.drop(columns=[col for col in JUNCTION_COLUMNS if col in features.columns])
    features = features.merge(graph_agg, on="h3", how="left")
    features[JUNCTION_COLUMNS] = features[JUNCTION_COLUMNS].fillna(0)
    features["junction_density_per_km"] = (
        features["junction_count"] / (features["road_length_total_m"] / 1000)
    ).replace([float("inf"), -float("inf")], 0).fillna(0)
    features["minor_to_major_junction_density_per_km"] = (
        features["minor_to_major_junction_count"] / (features["road_length_total_m"] / 1000)
    ).replace([float("inf"), -float("inf")], 0).fillna(0)
    features["class_delta_junction_density_per_km"] = (
        features["class_delta_junction_count"] / (features["road_length_total_m"] / 1000)
    ).replace([float("inf"), -float("inf")], 0).fillna(0)
    features["unsignalized_class_delta_junction_density_per_km"] = (
        features["unsignalized_class_delta_junction_count"] / (features["road_length_total_m"] / 1000)
    ).replace([float("inf"), -float("inf")], 0).fillna(0)
    features.to_parquet(H3_ROAD_FEATURES_PARQUET, index=False)

    panel = pd.read_parquet(
        H3_MONTH_PANEL_PARQUET,
        columns=["h3", "month", "year", "month_num", "target_had_accident_next_month"],
    )
    h3_month = panel.merge(features, on="h3", how="left")
    feature_columns = [col for col in features.columns if col != "h3"]
    h3_month[feature_columns] = h3_month[feature_columns].fillna(0)
    h3_month.to_parquet(H3_MONTH_ROAD_FEATURES_PARQUET, index=False)

    new_sum = features["junction_count"].sum()
    print(f"Old junction_count sum: {old_sum:,.0f}")
    print(f"New junction_count sum: {new_sum:,.0f}")
    print(f"Graph node rows: {len(graph):,}")
    print(f"Saved: {H3_ROAD_FEATURES_PARQUET}")
    print(f"Saved: {H3_MONTH_ROAD_FEATURES_PARQUET}")


if __name__ == "__main__":
    reconcile_junction_features()
