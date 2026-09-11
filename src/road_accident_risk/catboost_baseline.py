from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score

from road_accident_risk.logreg_baseline import TARGET, TRAIN_END, precision_recall_at_k
from road_accident_risk.osm_road_features import MOTOR_ROAD_CLASSES
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PROJECT_ROOT


FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS = "road_without_type_lengths"
FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR = "road_without_type_lengths_no_calendar"
FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM = "road_without_type_lengths_no_calendar_slim"
FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP = (
    "road_without_type_lengths_no_calendar_slim_dedup"
)
FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP_LOG1P = (
    "road_without_type_lengths_no_calendar_slim_dedup_log1p"
)

SLIM_EXCLUDED_FEATURES = {
    "road_share_motorway",
    "road_share_motorway_link",
    "mini_roundabout_count",
    "mini_roundabout_share",
    "mini_roundabout_density_per_km",
    "roundabout_total_count",
    "roundabout_count",
    "roundabout_density_per_km",
    "roundabout_total_density_per_km",
    "full_roundabout_share",
    "priority_sign_count",
    "give_way_share",
    "motorway_junction_count",
    "stop_count",
    "stop_density_per_km",
    "stop_share",
    "give_way_count",
    "give_way_density_per_km",
    "tunnel_length_m",
    "tunnel_share",
    "link_road_length_m",
    "link_road_share",
    "road_share_primary_link",
    "road_share_secondary_link",
    "road_share_tertiary_link",
    "road_share_trunk_link",
    "crossing_signalized_count",
    "crossing_signalized_share",
    "crossing_signalized_density_per_km",
}

DUPLICATE_EXCLUDED_FEATURES = {
    "road_segment_h3_count",
    "maxspeed_is_imputed_share",
    "unsignalized_class_delta_junction_count",
    "unsignalized_class_delta_junction_density_per_km",
    "traffic_calming_count",
    "bridge_share",
    "tram_stop_share",
    "railway_crossing_count",
    "traffic_signal_density_per_km",
    "crossing_density_per_km",
    "crossing_unsignalized_count",
    "crossing_unsignalized_density_per_km",
    "crossing_unsignalized_share",
    "lit_share",
    "public_transport_stop_density_per_km",
    "bus_stop_count",
    "bus_stop_share",
    "tram_stop_count",
    "oneway_length_m",
    "bad_surface_length_m",
    "major_road_length_m",
    "lanes_max",
    "class_delta_sum",
    "class_delta_junction_count",
    "turn_event_count",
    "short_segment_count",
}

ROAD_ONLY_4_1_LOG1P_FEATURES = {
    "asphalt_surface_length_m",
    "bridge_length_m",
    "class_delta_junction_density_per_km",
    "crossing_count",
    "curvature_deg_per_km",
    "dead_end_count",
    "dead_end_density_per_km",
    "four_plus_way_junction_count",
    "junction_count",
    "junction_density_per_km",
    "lane_meters_estimate",
    "legacy_major_minor_junction_count",
    "lit_road_length_m",
    "major_junction_count",
    "minor_to_major_junction_count",
    "minor_to_major_junction_density_per_km",
    "priority_sign_density_per_km",
    "public_transport_stop_count",
    "railway_crossing_density_per_km",
    "road_length_total_m",
    "road_segment_count",
    "road_segment_density_per_km",
    "road_segment_length_mean",
    "road_segment_length_median",
    "road_way_count",
    "sharp_turn_count",
    "sharp_turn_density_per_km",
    "signalized_class_delta_junction_count",
    "three_way_junction_count",
    "traffic_calming_density_per_km",
    "traffic_signal_count",
    "turn_event_density_per_km",
    "unsignalized_high_delta_junction_count",
}


def build_road_features_without_type_lengths(columns: list[str]) -> list[str]:
    per_type_road_lengths = {f"road_length_{highway}_m" for highway in MOTOR_ROAD_CLASSES}
    excluded = {"h3", "month", TARGET} | per_type_road_lengths
    return [column for column in columns if column not in excluded]


def build_road_features_without_type_lengths_no_calendar(columns: list[str]) -> list[str]:
    excluded = {"year", "month_num"}
    return [
        feature
        for feature in build_road_features_without_type_lengths(columns)
        if feature not in excluded
    ]


def build_road_features_without_type_lengths_no_calendar_slim(columns: list[str]) -> list[str]:
    return [
        feature
        for feature in build_road_features_without_type_lengths_no_calendar(columns)
        if feature not in SLIM_EXCLUDED_FEATURES
    ]


def build_road_features_without_type_lengths_no_calendar_slim_dedup(columns: list[str]) -> list[str]:
    return [
        feature
        for feature in build_road_features_without_type_lengths_no_calendar_slim(columns)
        if feature not in DUPLICATE_EXCLUDED_FEATURES
    ]


def build_road_features_without_type_lengths_no_calendar_slim_dedup_log1p(columns: list[str]) -> list[str]:
    return build_road_features_without_type_lengths_no_calendar_slim_dedup(columns)


def build_features(panel: pd.DataFrame, feature_set: str, features_arg: str | None = None) -> list[str]:
    if features_arg:
        return [feature.strip() for feature in features_arg.split(",") if feature.strip()]

    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS:
        return build_road_features_without_type_lengths(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR:
        return build_road_features_without_type_lengths_no_calendar(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM:
        return build_road_features_without_type_lengths_no_calendar_slim(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP:
        return build_road_features_without_type_lengths_no_calendar_slim_dedup(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP_LOG1P:
        return build_road_features_without_type_lengths_no_calendar_slim_dedup_log1p(list(panel.columns))

    raise ValueError(f"Unknown feature set: {feature_set}")


def default_log1p_features(feature_set: str, features: list[str]) -> list[str]:
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP_LOG1P:
        return [feature for feature in features if feature in ROAD_ONLY_4_1_LOG1P_FEATURES]
    return []


def fit_p99_clip_caps(train: pd.DataFrame, features: list[str]) -> dict[str, float]:
    quantiles = train[features].quantile(0.99, numeric_only=True)
    max_values = train[features].max(numeric_only=True)
    caps: dict[str, float] = {}
    for feature, cap in quantiles.items():
        if not np.isfinite(cap) or cap <= 0:
            continue
        max_value = max_values.get(feature)
        if pd.isna(max_value) or max_value <= cap:
            continue
        caps[feature] = float(cap)
    return caps


def apply_p99_clipping(df: pd.DataFrame, caps: dict[str, float]) -> pd.DataFrame:
    if not caps:
        return df
    result = df.copy()
    for feature, cap in caps.items():
        if feature in result.columns:
            result[feature] = result[feature].clip(upper=cap)
    return result


def apply_log1p_transform(df: pd.DataFrame, log1p_features: list[str]) -> pd.DataFrame:
    if not log1p_features:
        return df
    result = df.copy()
    for feature in log1p_features:
        if feature not in result.columns:
            continue
        min_value = result[feature].min()
        if pd.notna(min_value) and min_value < 0:
            raise ValueError(f"Cannot apply log1p to negative feature: {feature}")
        result[feature] = np.log1p(result[feature])
    return result


def train_and_evaluate(
    panel_path: Path = H3_MONTH_ROAD_FEATURES_PARQUET,
    train_end: str = TRAIN_END,
    feature_set: str = FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS,
    features_arg: str | None = None,
    min_road_length_m: float | None = None,
    clip_p99: bool = False,
    clip_p99_caps_path: Path | None = None,
    log1p_features_path: Path | None = None,
    iterations: int = 800,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_seed: int = 42,
    feature_importance_path: Path = PROJECT_ROOT / "reports" / "catboost_feature_importance_without_type_lengths.csv",
) -> None:
    panel = pd.read_parquet(panel_path)
    if min_road_length_m is not None:
        if "road_length_total_m" not in panel.columns:
            raise ValueError("road_length_total_m is required when --min-road-length-m is used")
        panel = panel[panel["road_length_total_m"] > min_road_length_m].copy()

    features = build_features(panel, feature_set=feature_set, features_arg=features_arg)

    missing = [feature for feature in features if feature not in panel.columns]
    if missing:
        raise ValueError(f"Missing feature columns in panel: {missing}")

    df = panel[["h3", "month", *features, TARGET]].copy()
    df["month"] = pd.to_datetime(df["month"])
    df[TARGET] = df[TARGET].astype(int)

    train = df[df["month"] < train_end].copy()
    test = df[df["month"] >= train_end].copy()
    clip_caps: dict[str, float] = {}
    log1p_features = default_log1p_features(feature_set, features)
    if clip_p99:
        clip_caps = fit_p99_clip_caps(train, features)
        train = apply_p99_clipping(train, clip_caps)
        test = apply_p99_clipping(test, clip_caps)
        if clip_p99_caps_path is not None:
            clip_p99_caps_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"feature": feature, "p99_cap": cap} for feature, cap in clip_caps.items()]
            ).to_csv(clip_p99_caps_path, index=False)
    if log1p_features:
        train = apply_log1p_transform(train, log1p_features)
        test = apply_log1p_transform(test, log1p_features)
        if log1p_features_path is not None:
            log1p_features_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"feature": log1p_features}).to_csv(log1p_features_path, index=False)

    X_train = train[features]
    y_train = train[TARGET]
    X_test = test[features]
    y_test = test[TARGET]

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=random_seed,
        allow_writing_files=False,
        verbose=100,
    )
    train_pool = Pool(X_train, y_train, feature_names=features)
    test_pool = Pool(X_test, y_test, feature_names=features)
    model.fit(train_pool, eval_set=test_pool, use_best_model=False)

    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= 0.5).astype(int)

    print(f"Panel: {panel_path}")
    print(f"Panel shape: {panel.shape}")
    print(f"Feature set: {feature_set}")
    if min_road_length_m is not None:
        print(f"Row filter: road_length_total_m > {min_road_length_m}")
    if clip_p99:
        print(f"P99 clipping: enabled | clipped_features={len(clip_caps)}")
        if clip_p99_caps_path is not None:
            print(f"P99 clipping caps: {clip_p99_caps_path}")
    if log1p_features:
        print(f"Log1p transform: enabled | features={len(log1p_features)}")
        if log1p_features_path is not None:
            print(f"Log1p features: {log1p_features_path}")
    print(f"Feature count: {len(features)}")
    print(f"Features: {', '.join(features)}")
    print(f"Train: {train.shape} | {train['month'].min()} to {train['month'].max()} | rate={y_train.mean():.6f}")
    print(f"Test:  {test.shape} | {test['month'].min()} to {test['month'].max()} | rate={y_test.mean():.6f}")
    print(f"CatBoost: iterations={iterations}, learning_rate={learning_rate}, depth={depth}")
    print(f"ROC-AUC: {roc_auc_score(y_test, test_scores):.6f}")
    print(f"PR-AUC:  {average_precision_score(y_test, test_scores):.6f}")
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_pred))
    print()
    print(classification_report(y_test, test_pred, digits=4))
    print("Top-k metrics:")

    for k_frac in [0.01, 0.03, 0.05, 0.10]:
        metrics = precision_recall_at_k(y_test.to_numpy(), test_scores, k_frac=k_frac)
        print(
            f"Top {k_frac:.0%}: "
            f"k={metrics['k']:,}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"lift={metrics['lift']:.2f}, "
            f"tp={metrics['positives_in_top_k']:,}/{metrics['total_positives']:,}"
        )

    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.get_feature_importance(train_pool, type="FeatureImportance"),
        }
    ).sort_values("importance", ascending=False)
    feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(feature_importance_path, index=False)

    print()
    print(f"Feature importance: {feature_importance_path}")
    print("Top feature importances:")
    for row in feature_importance.head(20).itertuples(index=False):
        print(f"{row.feature}: {row.importance:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a CatBoost baseline.")
    parser.add_argument("--panel", type=Path, default=H3_MONTH_ROAD_FEATURES_PARQUET)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--feature-set", default=FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS)
    parser.add_argument("--features", default=None, help="Comma-separated feature columns to use.")
    parser.add_argument("--min-road-length-m", type=float, default=None)
    parser.add_argument("--clip-p99", action="store_true")
    parser.add_argument("--clip-p99-caps-path", type=Path, default=None)
    parser.add_argument("--log1p-features-path", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--feature-importance-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "catboost_feature_importance_without_type_lengths.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_and_evaluate(
        panel_path=args.panel,
        train_end=args.train_end,
        feature_set=args.feature_set,
        features_arg=args.features,
        min_road_length_m=args.min_road_length_m,
        clip_p99=args.clip_p99,
        clip_p99_caps_path=args.clip_p99_caps_path,
        log1p_features_path=args.log1p_features_path,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=args.random_seed,
        feature_importance_path=args.feature_importance_path,
    )


if __name__ == "__main__":
    main()
