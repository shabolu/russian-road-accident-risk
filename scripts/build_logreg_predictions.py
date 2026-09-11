from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.osm_road_features import MOTOR_ROAD_CLASSES
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PARQUET_DIR


TARGET = "target_had_accident_next_month"
TRAIN_END = "2023-01-01"
OUTPUT_PATH = PARQUET_DIR / "h3_month_logreg_predictions.parquet"


def build_features(columns: list[str]) -> list[str]:
    per_type_road_lengths = {f"road_length_{highway}_m" for highway in MOTOR_ROAD_CLASSES}
    excluded = {"h3", "month", TARGET} | per_type_road_lengths
    return [column for column in columns if column not in excluded]


def build_logreg_predictions(
    panel_path: Path = H3_MONTH_ROAD_FEATURES_PARQUET,
    output_path: Path = OUTPUT_PATH,
    train_end: str = TRAIN_END,
) -> None:
    panel = pd.read_parquet(panel_path)
    panel["month"] = pd.to_datetime(panel["month"])
    panel[TARGET] = panel[TARGET].astype(int)
    features = build_features(list(panel.columns))

    train = panel[panel["month"] < train_end].copy()
    test = panel[panel["month"] >= train_end].copy()

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
    model.fit(train[features], train[TARGET])

    scores = model.predict_proba(panel[features])[:, 1]
    predictions = panel[["h3", "month", TARGET]].copy()
    predictions["logreg_score"] = scores
    predictions["logreg_score_percentile"] = predictions.groupby("month")["logreg_score"].rank(pct=True)
    predictions["logreg_rank_in_month"] = (
        predictions.groupby("month")["logreg_score"].rank(method="first", ascending=False).astype(int)
    )
    predictions["logreg_train_window"] = predictions["month"] < train_end
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)

    test_scores = predictions.loc[predictions["month"] >= train_end, "logreg_score"].to_numpy()
    y_test = test[TARGET].to_numpy()

    print(f"Panel: {panel_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(predictions):,}")
    print(f"Feature count: {len(features):,}")
    print(f"Train rows: {len(train):,} | positive rate={train[TARGET].mean():.6f}")
    print(f"Test rows: {len(test):,} | positive rate={test[TARGET].mean():.6f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, test_scores):.6f}")
    print(f"PR-AUC: {average_precision_score(y_test, test_scores):.6f}")
    print(f"Latest month: {predictions['month'].max().date()}")
    print(f"Score min/max: {np.min(scores):.6f} / {np.max(scores):.6f}")


if __name__ == "__main__":
    build_logreg_predictions()
