from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.osm_road_features import parse_number, weighted_mean
from road_accident_risk.paths import (
    H3_MONTH_PANEL_PARQUET,
    H3_MONTH_ROAD_FEATURES_PARQUET,
    H3_ROAD_FEATURES_PARQUET,
    OSM_ROAD_H3_SEGMENTS_PARQUET,
    OSM_ROADS_RAW_PARQUET,
)


MAXSPEED_COLUMNS = [
    "maxspeed_mean_weighted",
    "maxspeed_max",
    "maxspeed_tagged_share",
    "maxspeed_is_imputed_share",
]

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


def maxspeed_from_tags(tags_json: str | None) -> float | None:
    if tags_json is None or pd.isna(tags_json):
        return None
    tags = json.loads(tags_json)
    return parse_number(tags.get("maxspeed"))


def reconcile_maxspeed() -> None:
    raw = pd.read_parquet(OSM_ROADS_RAW_PARQUET)
    segments = pd.read_parquet(OSM_ROAD_H3_SEGMENTS_PARQUET)
    features = pd.read_parquet(H3_ROAD_FEATURES_PARQUET)

    def row_maxspeed(row: pd.Series) -> pd.Series:
        tags = json.loads(row["tags_json"]) if isinstance(row.get("tags_json"), str) else {}
        highway = row.get("highway")
        raw_value = parse_number(tags.get("maxspeed"))
        filled_value = raw_value if raw_value is not None else MAXSPEED_IMPUTATION_BY_HIGHWAY.get(highway)
        return pd.Series({
            "maxspeed_num_raw": raw_value,
            "maxspeed_num": filled_value,
            "maxspeed_is_imputed": raw_value is None and filled_value is not None,
        })

    if "maxspeed_num_raw" in raw.columns:
        old_raw_tagged = int(raw["maxspeed_num_raw"].notna().sum())
    elif "maxspeed_num" in raw.columns:
        old_raw_tagged = int(raw["maxspeed_num"].notna().sum())
    else:
        old_raw_tagged = 0
    raw = raw.drop(columns=["maxspeed_num_raw", "maxspeed_num", "maxspeed_is_imputed"], errors="ignore")
    raw = raw.join(raw.apply(row_maxspeed, axis=1))
    new_raw_tagged = int(raw["maxspeed_num_raw"].notna().sum())

    segments = segments.drop(columns=["maxspeed_num", "maxspeed_num_raw", "maxspeed_is_imputed"], errors="ignore")
    segments = segments.merge(raw[["osm_id", "maxspeed_num_raw", "maxspeed_num", "maxspeed_is_imputed"]], on="osm_id", how="left")

    maxspeed_agg = (
        segments.groupby("h3")
        .apply(lambda g: pd.Series({
            "maxspeed_mean_weighted": weighted_mean(g["maxspeed_num"], g["segment_length_m"]),
            "maxspeed_max": float(g["maxspeed_num"].max()) if g["maxspeed_num"].notna().any() else 0.0,
            "maxspeed_tagged_share": float(
                g.loc[g["maxspeed_num_raw"].notna(), "segment_length_m"].sum() / g["segment_length_m"].sum()
            ),
            "maxspeed_is_imputed_share": float(
                g.loc[g["maxspeed_is_imputed"], "segment_length_m"].sum() / g["segment_length_m"].sum()
            ),
        }))
        .reset_index()
    )

    features = features.drop(columns=[col for col in MAXSPEED_COLUMNS if col in features.columns])
    features = features.merge(maxspeed_agg, on="h3", how="left")
    features[MAXSPEED_COLUMNS] = features[MAXSPEED_COLUMNS].fillna(0)

    raw.to_parquet(OSM_ROADS_RAW_PARQUET, index=False)
    segments.to_parquet(OSM_ROAD_H3_SEGMENTS_PARQUET, index=False)
    features.to_parquet(H3_ROAD_FEATURES_PARQUET, index=False)

    panel = pd.read_parquet(
        H3_MONTH_PANEL_PARQUET,
        columns=["h3", "month", "year", "month_num", "target_had_accident_next_month"],
    )
    h3_month = panel.merge(features, on="h3", how="left")
    feature_columns = [column for column in features.columns if column != "h3"]
    h3_month[feature_columns] = h3_month[feature_columns].fillna(0)
    h3_month.to_parquet(H3_MONTH_ROAD_FEATURES_PARQUET, index=False)

    print(f"Raw ways with parsed maxspeed: {old_raw_tagged:,} -> {new_raw_tagged:,}")
    print(f"Saved: {OSM_ROADS_RAW_PARQUET}")
    print(f"Saved: {OSM_ROAD_H3_SEGMENTS_PARQUET}")
    print(f"Saved: {H3_ROAD_FEATURES_PARQUET}")
    print(f"Saved: {H3_MONTH_ROAD_FEATURES_PARQUET}")


if __name__ == "__main__":
    reconcile_maxspeed()
