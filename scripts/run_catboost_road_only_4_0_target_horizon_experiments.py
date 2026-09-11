from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_poisson_deviance,
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
SUMMARY_PATH = REPORTS_DIR / "catboost_road_only_4_0_target_horizon_summary.csv"
TOPK_PATH = REPORTS_DIR / "catboost_road_only_4_0_target_horizon_topk.csv"
CONFUSION_PATH = REPORTS_DIR / "catboost_road_only_4_0_target_horizon_confusion.csv"


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    kind: str
    horizon_months: int
    target_column: str
    valid_column: str

    @property
    def score_prefix(self) -> str:
        return f"catboost_road_only_4_0_{self.key}"

    @property
    def output_path(self) -> Path:
        return PARQUET_DIR / f"h3_month_catboost_road_only_4_0_{self.key}_predictions.parquet"

    @property
    def importance_path(self) -> Path:
        return REPORTS_DIR / f"catboost_feature_importance_road_only_4_0_{self.key}.csv"


TARGET_SPECS = [
    TargetSpec(
        key="binary_next_1m",
        label="Будет ли ДТП в следующем месяце",
        kind="binary",
        horizon_months=1,
        target_column="target_had_accident_next_1m",
        valid_column="target_next_1m_complete",
    ),
    TargetSpec(
        key="count_next_1m",
        label="Количество ДТП в следующем месяце",
        kind="count",
        horizon_months=1,
        target_column="target_count_next_1m",
        valid_column="target_next_1m_complete",
    ),
    TargetSpec(
        key="count_next_3m",
        label="Количество ДТП в следующих 3 месяцах",
        kind="count",
        horizon_months=3,
        target_column="target_count_next_3m",
        valid_column="target_next_3m_complete",
    ),
    TargetSpec(
        key="count_next_6m",
        label="Количество ДТП в следующих 6 месяцах",
        kind="count",
        horizon_months=6,
        target_column="target_count_next_6m",
        valid_column="target_next_6m_complete",
    ),
    TargetSpec(
        key="binary_next_3m",
        label="Будет ли ДТП в следующих 3 месяцах",
        kind="binary",
        horizon_months=3,
        target_column="target_had_accident_next_3m",
        valid_column="target_next_3m_complete",
    ),
    TargetSpec(
        key="binary_next_6m",
        label="Будет ли ДТП в следующих 6 месяцах",
        kind="binary",
        horizon_months=6,
        target_column="target_had_accident_next_6m",
        valid_column="target_next_6m_complete",
    ),
]


def add_future_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["h3", "month"]).copy()
    next_month = panel.groupby("h3", sort=False)["target_count_next_month"]

    for horizon in [1, 3, 6]:
        shifted_parts = [next_month.shift(-offset) for offset in range(horizon)]
        shifted = pd.concat(shifted_parts, axis=1)
        valid = shifted.notna().all(axis=1)
        count_column = f"target_count_next_{horizon}m"
        binary_column = f"target_had_accident_next_{horizon}m"
        valid_column = f"target_next_{horizon}m_complete"
        panel[count_column] = shifted.sum(axis=1, min_count=horizon)
        panel[binary_column] = panel[count_column] > 0
        panel[valid_column] = valid

    return panel


def safe_auc(metric_name: str, y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, scores))
    if metric_name == "pr_auc":
        return float(average_precision_score(y_true, scores))
    raise ValueError(metric_name)


def count_topk_metrics(y_count: np.ndarray, scores: np.ndarray, k_frac: float) -> dict[str, float | int]:
    y_count = np.asarray(y_count, dtype=float)
    y_binary = (y_count > 0).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_count)
    k = max(1, int(n * k_frac))
    top_idx = np.argsort(scores)[::-1][:k]
    total_count = float(y_count.sum())
    positives = int(y_binary.sum())
    positives_in_top = int(y_binary[top_idx].sum())
    count_in_top = float(y_count[top_idx].sum())
    positive_rate = float(y_binary.mean())

    return {
        "k": k,
        "precision": positives_in_top / k,
        "event_recall": positives_in_top / positives if positives else math.nan,
        "count_recall": count_in_top / total_count if total_count else math.nan,
        "lift": (positives_in_top / k) / positive_rate if positive_rate else math.nan,
        "positives_in_top_k": positives_in_top,
        "total_positives": positives,
        "accidents_in_top_k": count_in_top,
        "total_accidents": total_count,
    }


def build_predictions(
    df: pd.DataFrame,
    spec: TargetSpec,
    features: list[str],
    model: CatBoostClassifier | CatBoostRegressor,
    train_end: str,
) -> pd.DataFrame:
    score_column = f"{spec.score_prefix}_score"
    percentile_column = f"{spec.score_prefix}_score_percentile"
    rank_column = f"{spec.score_prefix}_rank_in_month"
    train_window_column = f"{spec.score_prefix}_train_window"

    if spec.kind == "binary":
        scores = model.predict_proba(df[features])[:, 1]
    else:
        scores = np.clip(model.predict(df[features]), 0, None)

    predictions = df[["h3", "month", spec.target_column]].copy()
    predictions[score_column] = scores
    predictions[percentile_column] = predictions.groupby("month")[score_column].rank(pct=True)
    predictions[rank_column] = (
        predictions.groupby("month")[score_column].rank(method="first", ascending=False).astype(int)
    )
    predictions[train_window_column] = predictions["month"] < train_end
    return predictions


def train_one_target(
    panel: pd.DataFrame,
    features: list[str],
    spec: TargetSpec,
    train_end: str,
    iterations: int,
    learning_rate: float,
    depth: int,
    random_seed: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]], dict[str, int | str]]:
    df = panel[panel[spec.valid_column]].copy()
    df[spec.target_column] = df[spec.target_column].astype(float if spec.kind == "count" else int)

    train = df[df["month"] < train_end].copy()
    test = df[df["month"] >= train_end].copy()
    clip_caps = fit_p99_clip_caps(train, features)
    train = apply_p99_clipping(train, clip_caps)
    test = apply_p99_clipping(test, clip_caps)
    df_for_predictions = apply_p99_clipping(df, clip_caps)

    train_pool = Pool(train[features], train[spec.target_column], feature_names=features)
    test_pool = Pool(test[features], test[spec.target_column], feature_names=features)

    if spec.kind == "binary":
        model: CatBoostClassifier | CatBoostRegressor = CatBoostClassifier(
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
    else:
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="Poisson",
            eval_metric="Poisson",
            random_seed=random_seed,
            allow_writing_files=False,
            verbose=100,
        )

    print()
    print("=" * 80)
    print(f"{spec.key}: {spec.label}")
    print(f"Rows: train={len(train):,}, test={len(test):,}")
    model.fit(train_pool, eval_set=test_pool, use_best_model=False)

    predictions = build_predictions(df_for_predictions, spec, features, model, train_end)
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

    test_scores = predictions.loc[predictions["month"] >= train_end, f"{spec.score_prefix}_score"].to_numpy()
    y_test = test[spec.target_column].to_numpy()
    y_test_binary = (y_test > 0).astype(int)
    y_train = train[spec.target_column].to_numpy()
    y_train_binary = (y_train > 0).astype(int)

    summary: dict[str, float | int | str] = {
        "target_key": spec.key,
        "target_label": spec.label,
        "target_kind": spec.kind,
        "horizon_months": spec.horizon_months,
        "model": "CatBoostClassifier" if spec.kind == "binary" else "CatBoostRegressorPoisson",
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": len(features),
        "train_positive_rate": float(y_train_binary.mean()),
        "test_positive_rate": float(y_test_binary.mean()),
        "train_mean_count": float(y_train.mean()),
        "test_mean_count": float(y_test.mean()),
        "test_total_count": float(y_test.sum()),
        "roc_auc_event": safe_auc("roc_auc", y_test_binary, test_scores),
        "pr_auc_event": safe_auc("pr_auc", y_test_binary, test_scores),
        "output_path": str(spec.output_path),
        "feature_importance_path": str(spec.importance_path),
    }

    confusion: dict[str, int | str] = {
        "target_key": spec.key,
        "target_label": spec.label,
        "target_kind": spec.kind,
        "threshold": "0.5" if spec.kind == "binary" else "",
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
    }

    if spec.kind == "binary":
        pred_binary = (test_scores >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_binary, pred_binary, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_binary,
            pred_binary,
            labels=[1],
            zero_division=0,
        )
        summary.update({
            "accuracy_0_5": float(accuracy_score(y_test_binary, pred_binary)),
            "precision_0_5": float(precision[0]),
            "recall_0_5": float(recall[0]),
            "f1_0_5": float(f1[0]),
            "mae": math.nan,
            "rmse": math.nan,
            "poisson_deviance": math.nan,
            "spearman": math.nan,
        })
        confusion.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    else:
        y_pred_count = np.clip(test_scores, 1e-9, None)
        summary.update({
            "accuracy_0_5": math.nan,
            "precision_0_5": math.nan,
            "recall_0_5": math.nan,
            "f1_0_5": math.nan,
            "mae": float(mean_absolute_error(y_test, y_pred_count)),
            "rmse": float(np.sqrt(np.mean((y_test - y_pred_count) ** 2))),
            "poisson_deviance": float(mean_poisson_deviance(y_test, y_pred_count)),
            "spearman": float(pd.Series(y_test).corr(pd.Series(y_pred_count), method="spearman")),
        })

    topk_rows: list[dict[str, float | int | str]] = []
    for k_frac in [0.01, 0.03, 0.05, 0.10]:
        if spec.kind == "binary":
            metrics = precision_recall_at_k(y_test_binary, test_scores, k_frac=k_frac)
            row = {
                "target_key": spec.key,
                "target_label": spec.label,
                "target_kind": spec.kind,
                "k_frac": k_frac,
                "k": metrics["k"],
                "precision": metrics["precision"],
                "event_recall": metrics["recall"],
                "count_recall": math.nan,
                "lift": metrics["lift"],
                "positives_in_top_k": metrics["positives_in_top_k"],
                "total_positives": metrics["total_positives"],
                "accidents_in_top_k": math.nan,
                "total_accidents": math.nan,
            }
        else:
            metrics = count_topk_metrics(y_test, test_scores, k_frac=k_frac)
            row = {
                "target_key": spec.key,
                "target_label": spec.label,
                "target_kind": spec.kind,
                "k_frac": k_frac,
                **metrics,
            }
        topk_rows.append(row)

    for row in topk_rows:
        k_label = f"top_{int(float(row['k_frac']) * 100)}pct"
        summary[f"{k_label}_precision"] = float(row["precision"])
        summary[f"{k_label}_event_recall"] = float(row["event_recall"])
        summary[f"{k_label}_count_recall"] = float(row["count_recall"])
        summary[f"{k_label}_lift"] = float(row["lift"])

    print(f"ROC-AUC event: {summary['roc_auc_event']:.6f}")
    print(f"PR-AUC event:  {summary['pr_auc_event']:.6f}")
    if spec.kind == "binary":
        print(
            "Threshold 0.5: "
            f"accuracy={summary['accuracy_0_5']:.4f}, "
            f"precision={summary['precision_0_5']:.4f}, "
            f"recall={summary['recall_0_5']:.4f}, "
            f"f1={summary['f1_0_5']:.4f}"
        )
    else:
        print(
            "Count metrics: "
            f"MAE={summary['mae']:.4f}, "
            f"RMSE={summary['rmse']:.4f}, "
            f"Poisson deviance={summary['poisson_deviance']:.4f}, "
            f"Spearman={summary['spearman']:.4f}"
        )
    print(f"Output: {spec.output_path}")
    print(f"Feature importance: {spec.importance_path}")

    return summary, topk_rows, confusion


def main() -> None:
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
    panel = panel[panel["road_length_total_m"] > 0].copy()

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

    print("CatBoost road-only 4.0 target horizon experiments")
    print(f"Panel: {H3_MONTH_ROAD_FEATURES_PARQUET}")
    print(f"Road-only rows: {len(panel):,}")
    print(f"Feature set: {FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP}")
    print(f"Feature count: {len(features):,}")
    print(f"Train end: {TRAIN_END}")
    print("P99 clipping: enabled per target on train split")

    summaries = []
    topk_rows = []
    confusion_rows = []
    for spec in TARGET_SPECS:
        summary, topk, confusion = train_one_target(
            panel=panel,
            features=features,
            spec=spec,
            train_end=TRAIN_END,
            iterations=800,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
        )
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
        "target_key",
        "target_kind",
        "horizon_months",
        "test_rows",
        "test_positive_rate",
        "roc_auc_event",
        "pr_auc_event",
        "precision_0_5",
        "recall_0_5",
        "mae",
        "rmse",
        "top_1pct_precision",
        "top_5pct_precision",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
