from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from road_accident_risk.paths import H3_MONTH_PANEL_PARQUET


DEFAULT_FEATURES = [
    "year",
    "month_num",
    "accidents_prev_1m",
    "accidents_prev_3m",
    "accidents_prev_6m",
    "accidents_prev_12m",
]
TARGET = "target_had_accident_next_month"
TRAIN_END = "2023-01-01"


def precision_recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_frac: float | None = None,
    k_abs: int | None = None,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    n = len(y_true)

    if k_abs is None:
        if k_frac is None:
            raise ValueError("Pass k_frac or k_abs")
        k_abs = max(1, int(n * k_frac))

    k_abs = min(k_abs, n)
    top_idx = np.argsort(scores)[::-1][:k_abs]
    tp_at_k = int(y_true[top_idx].sum())
    positives = int(y_true.sum())
    positive_rate = float(y_true.mean())

    precision = tp_at_k / k_abs
    recall = tp_at_k / positives if positives > 0 else np.nan
    lift = precision / positive_rate if positive_rate > 0 else np.nan

    return {
        "k": k_abs,
        "precision": precision,
        "recall": recall,
        "lift": lift,
        "positives_in_top_k": tp_at_k,
        "total_positives": positives,
    }


def train_and_evaluate(
    panel_path: Path = H3_MONTH_PANEL_PARQUET,
    train_end: str = TRAIN_END,
    features: list[str] | None = None,
) -> None:
    if features is None:
        features = DEFAULT_FEATURES

    panel = pd.read_parquet(panel_path)
    missing = [feature for feature in features if feature not in panel.columns]
    if missing:
        raise ValueError(f"Missing feature columns in panel: {missing}")

    df = panel[["h3", "month", *features, TARGET]].copy()
    df["month"] = pd.to_datetime(df["month"])
    df[TARGET] = df[TARGET].astype(int)

    train = df[df["month"] < train_end].copy()
    test = df[df["month"] >= train_end].copy()

    X_train = train[features]
    y_train = train[TARGET]
    X_test = test[features]
    y_test = test[TARGET]

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=None,
            ),
        ),
    ])
    model.fit(X_train, y_train)

    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= 0.5).astype(int)

    print(f"Panel: {panel_path}")
    print(f"Panel shape: {panel.shape}")
    print(f"Features: {', '.join(features)}")
    print(f"Train: {train.shape} | {train['month'].min()} to {train['month'].max()} | rate={y_train.mean():.6f}")
    print(f"Test:  {test.shape} | {test['month'].min()} to {test['month'].max()} | rate={y_test.mean():.6f}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the logistic regression baseline.")
    parser.add_argument("--panel", type=Path, default=H3_MONTH_PANEL_PARQUET)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature columns to use.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    features = [feature.strip() for feature in args.features.split(",") if feature.strip()]
    train_and_evaluate(panel_path=args.panel, train_end=args.train_end, features=features)


if __name__ == "__main__":
    main()
