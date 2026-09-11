from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.catboost_baseline import (  # noqa: E402
    FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP,
    apply_p99_clipping,
    build_features,
    fit_p99_clip_caps,
)
from road_accident_risk.logreg_baseline import TRAIN_END, precision_recall_at_k  # noqa: E402
from road_accident_risk.paths import (  # noqa: E402
    H3_MONTH_PANEL_PARQUET,
    H3_MONTH_ROAD_FEATURES_PARQUET,
    PARQUET_DIR,
    PROJECT_ROOT,
)


REPORTS_DIR = PROJECT_ROOT / "reports"
SUMMARY_PATH = REPORTS_DIR / "catboost_road_only_4_0_binary_6m_weight_summary.csv"
TOPK_PATH = REPORTS_DIR / "catboost_road_only_4_0_binary_6m_weight_topk.csv"
CONFUSION_PATH = REPORTS_DIR / "catboost_road_only_4_0_binary_6m_weight_confusion.csv"

TARGET_COLUMN = "target_had_accident_next_6m"
VALID_COLUMN = "target_next_6m_complete"


@dataclass(frozen=True)
class WeightSpec:
    key: str
    label: str
    positive_class_weight: float | None = None
    auto_class_weights: str | None = None

    @property
    def score_prefix(self) -> str:
        return f"catboost_road_only_4_0_binary_next_6m_{self.key}"

    @property
    def output_path(self) -> Path:
        return PARQUET_DIR / f"h3_month_{self.score_prefix}_predictions.parquet"

    @property
    def importance_path(self) -> Path:
        return REPORTS_DIR / f"catboost_feature_importance_road_only_4_0_binary_6m_{self.key}.csv"


WEIGHT_SPECS = [
    WeightSpec(key="weight_none", label="Без весов"),
    WeightSpec(key="weight_2", label="positive class weight = 2", positive_class_weight=2.0),
    WeightSpec(key="weight_4", label="positive class weight = 4", positive_class_weight=4.0),
    WeightSpec(key="weight_6", label="positive class weight = 6", positive_class_weight=6.0),
    WeightSpec(key="weight_balanced", label='auto_class_weights="Balanced"', auto_class_weights="Balanced"),
]


def add_future_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["h3", "month"]).copy()
    next_month = panel.groupby("h3", sort=False)["target_count_next_month"]
    shifted = pd.concat([next_month.shift(-offset) for offset in range(6)], axis=1)
    panel["target_count_next_6m"] = shifted.sum(axis=1, min_count=6)
    panel[TARGET_COLUMN] = panel["target_count_next_6m"] > 0
    panel[VALID_COLUMN] = shifted.notna().all(axis=1)
    return panel


def prepare_panel() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)
    panel["month"] = pd.to_datetime(panel["month"])
    if "target_count_next_month" not in panel.columns:
        target_panel = pd.read_parquet(H3_MONTH_PANEL_PARQUET)
        target_panel["month"] = pd.to_datetime(target_panel["month"])
        panel = panel.merge(
            target_panel[["h3", "month", "accident_count", "target_count_next_month"]],
            on=["h3", "month"],
            how="left",
            validate="one_to_one",
        )
    if panel["target_count_next_month"].isna().any():
        missing_targets = int(panel["target_count_next_month"].isna().sum())
        raise ValueError(f"Missing target_count_next_month after merge: {missing_targets}")

    panel = add_future_targets(panel)
    panel = panel[(panel["road_length_total_m"] > 0) & panel[VALID_COLUMN]].copy()

    features = build_features(
        panel,
        feature_set=FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP,
    )
    features = [
        feature
        for feature in features
        if feature not in {"accident_count", "target_count_next_month"}
        and not feature.startswith("target_count_next_")
        and not feature.startswith("target_had_accident_next_")
        and not feature.startswith("target_next_")
    ]
    missing = [feature for feature in features if feature not in panel.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    panel[TARGET_COLUMN] = panel[TARGET_COLUMN].astype(int)
    return panel, features


def model_kwargs(spec: WeightSpec) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "iterations": 800,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if spec.auto_class_weights is not None:
        kwargs["auto_class_weights"] = spec.auto_class_weights
    if spec.positive_class_weight is not None:
        kwargs["class_weights"] = [1.0, spec.positive_class_weight]
    return kwargs


def build_predictions(
    df: pd.DataFrame,
    features: list[str],
    model: CatBoostClassifier,
    spec: WeightSpec,
) -> pd.DataFrame:
    score_column = f"{spec.score_prefix}_score"
    percentile_column = f"{spec.score_prefix}_score_percentile"
    rank_column = f"{spec.score_prefix}_rank_in_month"
    train_window_column = f"{spec.score_prefix}_train_window"

    predictions = df[["h3", "month", TARGET_COLUMN]].copy()
    predictions[score_column] = model.predict_proba(df[features])[:, 1]
    predictions[percentile_column] = predictions.groupby("month")[score_column].rank(pct=True)
    predictions[rank_column] = (
        predictions.groupby("month")[score_column].rank(method="first", ascending=False).astype(int)
    )
    predictions[train_window_column] = predictions["month"] < TRAIN_END
    return predictions


def score_distribution(scores: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(scores, [0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "score_mean": float(np.mean(scores)),
        "score_p50": float(quantiles[0]),
        "score_p75": float(quantiles[1]),
        "score_p90": float(quantiles[2]),
        "score_p95": float(quantiles[3]),
        "score_p99": float(quantiles[4]),
        "score_max": float(np.max(scores)),
    }


def train_one(
    panel: pd.DataFrame,
    features: list[str],
    spec: WeightSpec,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]], dict[str, int | str]]:
    train = panel[panel["month"] < TRAIN_END].copy()
    test = panel[panel["month"] >= TRAIN_END].copy()

    clip_caps = fit_p99_clip_caps(train, features)
    train = apply_p99_clipping(train, clip_caps)
    test = apply_p99_clipping(test, clip_caps)
    panel_for_predictions = apply_p99_clipping(panel, clip_caps)

    train_pool = Pool(train[features], train[TARGET_COLUMN], feature_names=features)
    test_pool = Pool(test[features], test[TARGET_COLUMN], feature_names=features)
    model = CatBoostClassifier(**model_kwargs(spec))

    print()
    print("=" * 80)
    print(f"{spec.key}: {spec.label}")
    print(f"Rows: train={len(train):,}, test={len(test):,}")
    model.fit(train_pool, eval_set=test_pool, use_best_model=False)

    predictions = build_predictions(panel_for_predictions, features, model, spec)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(spec.output_path, index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.get_feature_importance(train_pool, type="FeatureImportance"),
        }
    ).sort_values("importance", ascending=False)
    spec.importance_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(spec.importance_path, index=False)

    score_column = f"{spec.score_prefix}_score"
    test_predictions = predictions[predictions["month"] >= TRAIN_END].copy()
    y_test = test[TARGET_COLUMN].to_numpy()
    test_scores = test_predictions[score_column].to_numpy()
    pred_binary = (test_scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        pred_binary,
        labels=[1],
        zero_division=0,
    )

    topk_rows: list[dict[str, float | int | str]] = []
    for k_frac in [0.01, 0.03, 0.05, 0.10]:
        metrics = precision_recall_at_k(y_test, test_scores, k_frac=k_frac)
        topk_rows.append({
            "weight_key": spec.key,
            "weight_label": spec.label,
            "k_frac": k_frac,
            "k": metrics["k"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "lift": metrics["lift"],
            "positives_in_top_k": metrics["positives_in_top_k"],
            "total_positives": metrics["total_positives"],
        })

    summary: dict[str, float | int | str] = {
        "weight_key": spec.key,
        "weight_label": spec.label,
        "positive_class_weight": spec.positive_class_weight if spec.positive_class_weight is not None else math.nan,
        "auto_class_weights": spec.auto_class_weights or "",
        "model": "CatBoostClassifier",
        "target": TARGET_COLUMN,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(features),
        "train_positive_rate": float(train[TARGET_COLUMN].mean()),
        "test_positive_rate": float(test[TARGET_COLUMN].mean()),
        "predicted_positive_rate_0_5": float(pred_binary.mean()),
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "pr_auc": float(average_precision_score(y_test, test_scores)),
        "accuracy_0_5": float(accuracy_score(y_test, pred_binary)),
        "precision_0_5": float(precision[0]),
        "recall_0_5": float(recall[0]),
        "f1_0_5": float(f1[0]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "output_path": str(spec.output_path),
        "feature_importance_path": str(spec.importance_path),
    }
    summary.update(score_distribution(test_scores))
    for row in topk_rows:
        k_label = f"top_{int(float(row['k_frac']) * 100)}pct"
        summary[f"{k_label}_precision"] = float(row["precision"])
        summary[f"{k_label}_recall"] = float(row["recall"])
        summary[f"{k_label}_lift"] = float(row["lift"])

    confusion = {
        "weight_key": spec.key,
        "weight_label": spec.label,
        "threshold": "0.5",
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print(f"ROC-AUC: {summary['roc_auc']:.6f}")
    print(f"PR-AUC:  {summary['pr_auc']:.6f}")
    print(
        "Threshold 0.5: "
        f"precision={summary['precision_0_5']:.4f}, "
        f"recall={summary['recall_0_5']:.4f}, "
        f"f1={summary['f1_0_5']:.4f}, "
        f"predicted_positive_rate={summary['predicted_positive_rate_0_5']:.4f}"
    )
    print(
        "Score distribution: "
        f"p50={summary['score_p50']:.4f}, "
        f"p90={summary['score_p90']:.4f}, "
        f"p95={summary['score_p95']:.4f}, "
        f"p99={summary['score_p99']:.4f}"
    )
    print(f"Output: {spec.output_path}")

    return summary, topk_rows, confusion


def main() -> None:
    panel, features = prepare_panel()
    print("CatBoost road-only 4.0 binary 6m weight experiments")
    print(f"Panel: {H3_MONTH_ROAD_FEATURES_PARQUET}")
    print(f"Rows: {len(panel):,}")
    print(f"Feature set: {FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP}")
    print(f"Feature count: {len(features):,}")
    print(f"Train end: {TRAIN_END}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Test positive rate: {panel[panel['month'] >= TRAIN_END][TARGET_COLUMN].mean():.6f}")

    summaries = []
    topk_rows = []
    confusion_rows = []
    for spec in WEIGHT_SPECS:
        summary, topk, confusion = train_one(panel, features, spec)
        summaries.append(summary)
        topk_rows.extend(topk)
        confusion_rows.append(confusion)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(topk_rows).to_csv(TOPK_PATH, index=False)
    pd.DataFrame(confusion_rows).to_csv(CONFUSION_PATH, index=False)

    print()
    print("=" * 80)
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Top-k: {TOPK_PATH}")
    print(f"Confusion: {CONFUSION_PATH}")
    print(pd.DataFrame(summaries)[[
        "weight_key",
        "roc_auc",
        "pr_auc",
        "precision_0_5",
        "recall_0_5",
        "f1_0_5",
        "predicted_positive_rate_0_5",
        "score_p50",
        "score_p90",
        "score_p95",
        "top_1pct_precision",
        "top_5pct_precision",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
