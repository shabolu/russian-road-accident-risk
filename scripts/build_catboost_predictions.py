from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.catboost_baseline import (
    FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS,
    apply_log1p_transform,
    apply_p99_clipping,
    build_features,
    default_log1p_features,
    fit_p99_clip_caps,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PARQUET_DIR


OUTPUT_PATH = PARQUET_DIR / "h3_month_catboost_predictions.parquet"


def build_catboost_predictions(
    panel_path: Path = H3_MONTH_ROAD_FEATURES_PARQUET,
    output_path: Path = OUTPUT_PATH,
    train_end: str = TRAIN_END,
    feature_set: str = FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS,
    score_prefix: str = "catboost",
    min_road_length_m: float | None = None,
    clip_p99: bool = False,
    clip_p99_caps_path: Path | None = None,
    log1p_features_path: Path | None = None,
    iterations: int = 800,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_seed: int = 42,
) -> None:
    panel = pd.read_parquet(panel_path)
    panel["month"] = pd.to_datetime(panel["month"])
    panel[TARGET] = panel[TARGET].astype(int)
    if min_road_length_m is not None:
        if "road_length_total_m" not in panel.columns:
            raise ValueError("road_length_total_m is required when --min-road-length-m is used")
        panel = panel[panel["road_length_total_m"] > min_road_length_m].copy()

    features = build_features(panel, feature_set=feature_set)

    train = panel[panel["month"] < train_end].copy()
    test = panel[panel["month"] >= train_end].copy()
    clip_caps: dict[str, float] = {}
    log1p_features = default_log1p_features(feature_set, features)
    if clip_p99:
        clip_caps = fit_p99_clip_caps(train, features)
        train = apply_p99_clipping(train, clip_caps)
        test = apply_p99_clipping(test, clip_caps)
        panel = apply_p99_clipping(panel, clip_caps)
        if clip_p99_caps_path is not None:
            clip_p99_caps_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"feature": feature, "p99_cap": cap} for feature, cap in clip_caps.items()]
            ).to_csv(clip_p99_caps_path, index=False)
    if log1p_features:
        train = apply_log1p_transform(train, log1p_features)
        test = apply_log1p_transform(test, log1p_features)
        panel = apply_log1p_transform(panel, log1p_features)
        if log1p_features_path is not None:
            log1p_features_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"feature": log1p_features}).to_csv(log1p_features_path, index=False)

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
    model.fit(
        Pool(train[features], train[TARGET], feature_names=features),
        eval_set=Pool(test[features], test[TARGET], feature_names=features),
        use_best_model=False,
    )

    scores = model.predict_proba(panel[features])[:, 1]
    predictions = panel[["h3", "month", TARGET]].copy()
    score_column = f"{score_prefix}_score"
    percentile_column = f"{score_prefix}_score_percentile"
    rank_column = f"{score_prefix}_rank_in_month"
    train_window_column = f"{score_prefix}_train_window"
    predictions[score_column] = scores
    predictions[percentile_column] = predictions.groupby("month")[score_column].rank(pct=True)
    predictions[rank_column] = (
        predictions.groupby("month")[score_column].rank(method="first", ascending=False).astype(int)
    )
    predictions[train_window_column] = predictions["month"] < train_end
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)

    test_scores = predictions.loc[predictions["month"] >= train_end, score_column].to_numpy()
    y_test = test[TARGET].to_numpy()

    print(f"Panel: {panel_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(predictions):,}")
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
    print(f"Feature count: {len(features):,}")
    print(f"Train rows: {len(train):,} | positive rate={train[TARGET].mean():.6f}")
    print(f"Test rows: {len(test):,} | positive rate={test[TARGET].mean():.6f}")
    print(f"CatBoost: iterations={iterations}, learning_rate={learning_rate}, depth={depth}")
    print(f"ROC-AUC: {roc_auc_score(y_test, test_scores):.6f}")
    print(f"PR-AUC: {average_precision_score(y_test, test_scores):.6f}")
    print(f"Latest month: {predictions['month'].max().date()}")
    print(f"Score min/max: {np.min(scores):.6f} / {np.max(scores):.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CatBoost predictions parquet for the viewer.")
    parser.add_argument("--panel", type=Path, default=H3_MONTH_ROAD_FEATURES_PARQUET)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--feature-set", default=FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS)
    parser.add_argument("--score-prefix", default="catboost")
    parser.add_argument("--min-road-length-m", type=float, default=None)
    parser.add_argument("--clip-p99", action="store_true")
    parser.add_argument("--clip-p99-caps-path", type=Path, default=None)
    parser.add_argument("--log1p-features-path", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_catboost_predictions(
        panel_path=args.panel,
        output_path=args.output,
        train_end=args.train_end,
        feature_set=args.feature_set,
        score_prefix=args.score_prefix,
        min_road_length_m=args.min_road_length_m,
        clip_p99=args.clip_p99,
        clip_p99_caps_path=args.clip_p99_caps_path,
        log1p_features_path=args.log1p_features_path,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
