from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.osm_road_features import parse_bool_yes
from road_accident_risk.paths import (
    H3_MONTH_PANEL_PARQUET,
    H3_MONTH_ROAD_FEATURES_PARQUET,
    H3_ROAD_FEATURES_PARQUET,
    OSM_ROAD_H3_SEGMENTS_PARQUET,
    OSM_ROAD_POINT_NODES_PARQUET,
    OSM_ROADS_RAW_PARQUET,
)


GROUP2_COLUMNS = [
    "traffic_signal_density_per_km",
    "crossing_signalized_count",
    "crossing_signalized_density_per_km",
    "crossing_unsignalized_count",
    "crossing_density_per_km",
    "crossing_signalized_share",
    "crossing_unsignalized_share",
    "crossing_unsignalized_density_per_km",
    "priority_sign_count",
    "priority_sign_density_per_km",
    "stop_density_per_km",
    "give_way_density_per_km",
    "stop_share",
    "give_way_share",
    "traffic_calming_density_per_km",
    "public_transport_stop_count",
    "public_transport_stop_density_per_km",
    "bus_stop_count",
    "tram_stop_count",
    "bus_stop_share",
    "tram_stop_share",
    "railway_crossing_count",
    "railway_crossing_density_per_km",
    "roundabout_count",
    "mini_roundabout_density_per_km",
    "roundabout_total_count",
    "roundabout_total_density_per_km",
    "mini_roundabout_share",
    "full_roundabout_share",
    "roundabout_density_per_km",
    "motorway_junction_density_per_km",
]


def tags_from_row(row: pd.Series) -> dict[str, object]:
    tags_json = row.get("tags_json")
    if isinstance(tags_json, str) and tags_json:
        return json.loads(tags_json)

    tags: dict[str, object] = {}
    for column in ["highway", "railway", "public_transport", "crossing", "bus", "tram", "traffic_calming"]:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            tags[column] = value
    return tags


def point_flags(row: pd.Series) -> pd.Series:
    tags = tags_from_row(row)
    highway = tags.get("highway")
    railway = tags.get("railway")
    public_transport = tags.get("public_transport")
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

    return pd.Series({
        "crossing_signalized_count": int(is_signalized_crossing),
        "crossing_unsignalized_count": int(is_unsignalized_crossing),
        "public_transport_stop_count": int(is_public_transport_stop),
        "bus_stop_count": int(is_bus_stop),
        "tram_stop_count": int(is_tram_stop),
        "railway_crossing_count": int(is_railway_crossing),
    })


def build_point_aggregation() -> pd.DataFrame:
    if not OSM_ROAD_POINT_NODES_PARQUET.exists():
        return pd.DataFrame(columns=["h3"])

    points = pd.read_parquet(OSM_ROAD_POINT_NODES_PARQUET)
    if points.empty:
        return pd.DataFrame(columns=["h3"])

    flags = points.join(points.apply(point_flags, axis=1))
    return flags.groupby("h3", as_index=False)[[
        "crossing_signalized_count",
        "crossing_unsignalized_count",
        "public_transport_stop_count",
        "bus_stop_count",
        "tram_stop_count",
        "railway_crossing_count",
    ]].sum()


def build_roundabout_aggregation() -> pd.DataFrame:
    if not OSM_ROADS_RAW_PARQUET.exists() or not OSM_ROAD_H3_SEGMENTS_PARQUET.exists():
        return pd.DataFrame(columns=["h3", "roundabout_count"])

    raw = pd.read_parquet(OSM_ROADS_RAW_PARQUET, columns=["osm_id", "tags_json"])
    segments = pd.read_parquet(OSM_ROAD_H3_SEGMENTS_PARQUET, columns=["h3", "osm_id"])

    raw["is_roundabout"] = raw["tags_json"].map(
        lambda value: json.loads(value).get("junction") == "roundabout"
        if isinstance(value, str)
        else False
    )
    roundabout_ids = raw.loc[raw["is_roundabout"], ["osm_id"]]
    if roundabout_ids.empty:
        return pd.DataFrame(columns=["h3", "roundabout_count"])

    roundabout_segments = segments.merge(roundabout_ids, on="osm_id", how="inner")
    return (
        roundabout_segments.groupby("h3")["osm_id"]
        .nunique()
        .rename("roundabout_count")
        .reset_index()
    )


def reconcile_group2_infrastructure_features() -> None:
    features = pd.read_parquet(H3_ROAD_FEATURES_PARQUET)
    old_columns = [column for column in GROUP2_COLUMNS if column in features.columns]
    features = features.drop(columns=old_columns)

    point_agg = build_point_aggregation()
    roundabout_agg = build_roundabout_aggregation()

    features = features.merge(point_agg, on="h3", how="left")
    features = features.merge(roundabout_agg, on="h3", how="left")

    count_columns = [
        "crossing_signalized_count",
        "crossing_unsignalized_count",
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

    density_share_columns = [
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
    features[density_share_columns] = (
        features[density_share_columns]
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
    )

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
    print("Added/updated group 2 infrastructure columns:")
    for column in GROUP2_COLUMNS:
        print(f"- {column}: sum={features[column].sum():,.3f}")


if __name__ == "__main__":
    reconcile_group2_infrastructure_features()
