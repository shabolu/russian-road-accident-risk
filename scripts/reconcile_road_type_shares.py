from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.osm_road_features import MOTOR_ROAD_CLASSES
from road_accident_risk.paths import (
    H3_MONTH_PANEL_PARQUET,
    H3_MONTH_ROAD_FEATURES_PARQUET,
    H3_ROAD_FEATURES_PARQUET,
    OSM_ROAD_H3_SEGMENTS_PARQUET,
)


def reconcile_road_type_shares() -> None:
    features = pd.read_parquet(H3_ROAD_FEATURES_PARQUET)
    segments = pd.read_parquet(OSM_ROAD_H3_SEGMENTS_PARQUET, columns=["h3", "highway", "segment_length_m"])

    type_lengths = (
        segments.groupby(["h3", "highway"])["segment_length_m"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    share_columns = []
    for highway in sorted(MOTOR_ROAD_CLASSES):
        length_column = f"road_length_{highway}_m"
        share_column = f"road_share_{highway}"
        share_columns.append(share_column)

        if highway not in type_lengths.columns:
            type_lengths[highway] = 0.0
        type_lengths[length_column] = type_lengths[highway]

    keep_columns = ["h3"] + [f"road_length_{highway}_m" for highway in sorted(MOTOR_ROAD_CLASSES)]
    type_lengths = type_lengths[keep_columns]

    drop_columns = [
        column
        for column in features.columns
        if column.startswith("road_share_")
        or (column.startswith("road_length_") and column.endswith("_m") and column != "road_length_total_m")
    ]
    features = features.drop(columns=drop_columns)
    features = features.merge(type_lengths, on="h3", how="left")

    type_length_columns = [column for column in type_lengths.columns if column != "h3"]
    features[type_length_columns] = features[type_length_columns].fillna(0)
    for highway in sorted(MOTOR_ROAD_CLASSES):
        length_column = f"road_length_{highway}_m"
        share_column = f"road_share_{highway}"
        features[share_column] = (
            features[length_column] / features["road_length_total_m"]
        ).replace([float("inf"), -float("inf")], 0).fillna(0)

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
    print("Added/updated road_share columns:")
    for column in share_columns:
        print(f"- {column}")


if __name__ == "__main__":
    reconcile_road_type_shares()
