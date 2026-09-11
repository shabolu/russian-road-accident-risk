from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from road_accident_risk.catboost_baseline import (  # noqa: E402
    FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP,
    apply_p99_clipping,
    build_features,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END  # noqa: E402
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PARQUET_DIR, PROJECT_ROOT as PACKAGE_ROOT  # noqa: E402


PREDICTIONS_PATH = PARQUET_DIR / "h3_month_catboost_road_only_4_0_predictions.parquet"
CLIP_CAPS_PATH = PACKAGE_ROOT / "reports" / "catboost_road_only_4_0_p99_clip_caps.csv"
IMPORTANCE_PATH = PACKAGE_ROOT / "reports" / "catboost_feature_importance_road_only_4_0_no_calendar_dedup_clip_p99.csv"

REPORTS_DIR = PACKAGE_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "catboost_road_only_4_0_error_analysis"
ROW_OUTPUT_PATH = PARQUET_DIR / "h3_month_catboost_road_only_4_0_error_rows.parquet"
H3_OUTPUT_PATH = PARQUET_DIR / "h3_catboost_road_only_4_0_error_analysis.parquet"
SUMMARY_PATH = REPORTS_DIR / "catboost_road_only_4_0_error_analysis_summary.csv"
FEATURE_GROUP_PATH = REPORTS_DIR / "catboost_road_only_4_0_error_analysis_feature_by_group.csv"
FEATURE_CONTRAST_PATH = REPORTS_DIR / "catboost_road_only_4_0_error_analysis_feature_contrasts.csv"
H3_TOP_PATH = REPORTS_DIR / "catboost_road_only_4_0_error_analysis_top_h3.csv"

SCORE_COLUMN = "catboost_road_only_4_0_score"
THRESHOLD = 0.5
GROUP_ORDER = ["TN", "FP", "FN", "TP"]
GROUP_LABELS_RU = {
    "TN": "TN: верно низкий риск",
    "FP": "FP: высокий риск, ДТП нет",
    "FN": "FN: ДТП есть, риск низкий",
    "TP": "TP: ДТП поймано",
}


def load_caps(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    caps = pd.read_csv(path)
    return {str(row.feature): float(row.p99_cap) for row in caps.itertuples(index=False)}


def classify_error_type(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    actual_bool = actual.astype(bool)
    predicted_bool = predicted.astype(bool)
    result = np.full(len(actual_bool), "TN", dtype=object)
    result[(~actual_bool) & predicted_bool] = "FP"
    result[actual_bool & (~predicted_bool)] = "FN"
    result[actual_bool & predicted_bool] = "TP"
    return pd.Series(result, index=actual.index)


def prepare_error_rows() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    panel = pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)
    panel["month"] = pd.to_datetime(panel["month"])
    panel[TARGET] = panel[TARGET].astype(int)
    panel = panel[panel["road_length_total_m"] > 0].copy()

    features = build_features(
        panel,
        feature_set=FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP,
    )
    caps = load_caps(CLIP_CAPS_PATH)
    clipped_features = apply_p99_clipping(panel[features + ["h3", "month", TARGET]].copy(), caps)

    predictions = pd.read_parquet(PREDICTIONS_PATH)
    predictions["month"] = pd.to_datetime(predictions["month"])

    test = clipped_features[clipped_features["month"] >= TRAIN_END].merge(
        predictions[["h3", "month", SCORE_COLUMN]],
        on=["h3", "month"],
        how="inner",
    )
    test["actual"] = test[TARGET].astype(int)
    test["predicted"] = (test[SCORE_COLUMN] >= THRESHOLD).astype(int)
    test["error_type"] = classify_error_type(test["actual"], test["predicted"])

    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH)
    else:
        importance = pd.DataFrame({"feature": features, "importance": np.nan})

    return test, features, importance


def h3_center(cell: str) -> tuple[float, float]:
    lat, lng = h3.cell_to_latlng(cell)
    return float(lat), float(lng)


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    total = len(rows)
    positives = int(rows["actual"].sum())
    predicted_positives = int(rows["predicted"].sum())
    grouped = (
        rows.groupby("error_type", observed=True)
        .agg(
            rows=("error_type", "size"),
            actual_positives=("actual", "sum"),
            predicted_positives=("predicted", "sum"),
            score_mean=(SCORE_COLUMN, "mean"),
            score_median=(SCORE_COLUMN, "median"),
            score_p95=(SCORE_COLUMN, lambda values: values.quantile(0.95)),
            road_length_mean=("road_length_total_m", "mean"),
            junction_count_mean=("junction_count", "mean"),
            asphalt_length_mean=("asphalt_surface_length_m", "mean"),
        )
        .reindex(GROUP_ORDER)
        .reset_index()
    )
    grouped["share_of_test"] = grouped["rows"] / total
    grouped["share_of_actual_positives"] = grouped["actual_positives"] / max(positives, 1)
    grouped["share_of_predicted_positives"] = grouped["predicted_positives"] / max(predicted_positives, 1)
    return grouped


def build_feature_group_stats(rows: pd.DataFrame, features: list[str], importance: pd.DataFrame) -> pd.DataFrame:
    stats: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(rows[feature], errors="coerce")
        for group in GROUP_ORDER:
            group_values = values[rows["error_type"] == group]
            stats.append(
                {
                    "feature": feature,
                    "error_type": group,
                    "rows": int(group_values.notna().sum()),
                    "mean": group_values.mean(),
                    "median": group_values.median(),
                    "p75": group_values.quantile(0.75),
                    "p95": group_values.quantile(0.95),
                    "zero_share": group_values.eq(0).mean(),
                }
            )
    result = pd.DataFrame(stats).merge(importance, on="feature", how="left")
    return result


def smd(a: pd.Series, b: pd.Series) -> float:
    a_std = a.std()
    b_std = b.std()
    pooled = np.sqrt((a_std**2 + b_std**2) / 2)
    if pd.isna(pooled) or pooled <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def build_feature_contrasts(rows: pd.DataFrame, features: list[str], importance: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("FP_vs_TN", "FP", "TN", "что отличает ложные тревоги от уверенно низкого риска"),
        ("FN_vs_TP", "FN", "TP", "что отличает пропущенные ДТП от пойманных ДТП"),
        ("TP_vs_FP", "TP", "FP", "что отличает реальные ДТП среди high-risk от ложных тревог"),
        ("FN_vs_TN", "FN", "TN", "что отличает пропущенные ДТП от true negative"),
    ]
    rows_out: list[dict[str, Any]] = []
    for contrast, left, right, meaning in pairs:
        for feature in features:
            left_values = pd.to_numeric(rows.loc[rows["error_type"] == left, feature], errors="coerce").dropna()
            right_values = pd.to_numeric(rows.loc[rows["error_type"] == right, feature], errors="coerce").dropna()
            left_mean = left_values.mean()
            right_mean = right_values.mean()
            rows_out.append(
                {
                    "contrast": contrast,
                    "meaning": meaning,
                    "left_group": left,
                    "right_group": right,
                    "feature": feature,
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                    "mean_diff": left_mean - right_mean,
                    "mean_ratio": left_mean / right_mean if abs(right_mean) > 1e-12 else np.nan,
                    "left_median": left_values.median(),
                    "right_median": right_values.median(),
                    "zero_share_left": left_values.eq(0).mean(),
                    "zero_share_right": right_values.eq(0).mean(),
                    "zero_share_diff": left_values.eq(0).mean() - right_values.eq(0).mean(),
                    "smd": smd(left_values, right_values),
                    "abs_smd": abs(smd(left_values, right_values)),
                }
            )
    result = pd.DataFrame(rows_out).merge(importance, on="feature", how="left")
    return result.sort_values(["contrast", "abs_smd", "importance"], ascending=[True, False, False])


def build_h3_aggregation(rows: pd.DataFrame) -> pd.DataFrame:
    pivot = pd.crosstab(rows["h3"], rows["error_type"]).reindex(columns=GROUP_ORDER, fill_value=0)
    agg = (
        rows.groupby("h3", observed=True)
        .agg(
            test_rows=("h3", "size"),
            actual_positives=("actual", "sum"),
            predicted_positives=("predicted", "sum"),
            score_mean=(SCORE_COLUMN, "mean"),
            score_max=(SCORE_COLUMN, "max"),
            road_length_total_m=("road_length_total_m", "first"),
            lane_meters_estimate=("lane_meters_estimate", "first"),
            junction_count=("junction_count", "first"),
            crossing_count=("crossing_count", "first"),
            road_curvature_mean_weighted=("road_curvature_mean_weighted", "first"),
            road_orientation_entropy=("road_orientation_entropy", "first"),
        )
        .join(pivot)
        .reset_index()
    )
    for column in GROUP_ORDER:
        if column not in agg.columns:
            agg[column] = 0
    agg = agg.rename(columns={"TN": "tn_count", "FP": "fp_count", "FN": "fn_count", "TP": "tp_count"})
    agg["fp_share_of_test"] = agg["fp_count"] / agg["test_rows"]
    agg["fn_share_of_test"] = agg["fn_count"] / agg["test_rows"]
    agg["tp_share_of_test"] = agg["tp_count"] / agg["test_rows"]
    agg["actual_positive_rate"] = agg["actual_positives"] / agg["test_rows"]
    agg["predicted_positive_rate"] = agg["predicted_positives"] / agg["test_rows"]
    agg["precision_in_h3"] = agg["tp_count"] / agg["predicted_positives"].replace(0, np.nan)
    agg["recall_in_h3"] = agg["tp_count"] / agg["actual_positives"].replace(0, np.nan)
    centers = agg["h3"].map(h3_center)
    agg["lat"] = [lat for lat, _ in centers]
    agg["lng"] = [lng for _, lng in centers]
    agg["dominant_error_type"] = agg[["fp_count", "fn_count", "tp_count", "tn_count"]].idxmax(axis=1)
    return agg.sort_values(["fp_count", "fn_count", "tp_count"], ascending=False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_summary(summary: pd.DataFrame) -> Path:
    plot_data = summary.copy()
    plot_data["label"] = plot_data["error_type"].map(GROUP_LABELS_RU)
    colors = {"TN": "#64748b", "FP": "#f59e0b", "FN": "#7c3aed", "TP": "#dc2626"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(plot_data["label"], plot_data["rows"], color=[colors[g] for g in plot_data["error_type"]])
    axes[0].set_title("Confusion groups, test rows")
    axes[0].set_ylabel("строки")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(plot_data["label"], plot_data["score_mean"], color=[colors[g] for g in plot_data["error_type"]])
    axes[1].set_title("Средний score по группам")
    axes[1].set_ylabel("score")
    axes[1].tick_params(axis="x", rotation=25)
    path = FIGURES_DIR / "confusion_group_summary.png"
    save_figure(fig, path)
    return path


def plot_score_distribution(rows: pd.DataFrame) -> Path:
    colors = {"TN": "#64748b", "FP": "#f59e0b", "FN": "#7c3aed", "TP": "#dc2626"}
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, 1, 70)
    for group in GROUP_ORDER:
        values = rows.loc[rows["error_type"] == group, SCORE_COLUMN]
        weights = np.ones(len(values)) / max(len(values), 1)
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            alpha=0.45,
            label=GROUP_LABELS_RU[group],
            color=colors[group],
        )
    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.2, label="threshold=0.5")
    ax.set_title("Распределение score по типам ошибок")
    ax.set_xlabel("CatBoost score")
    ax.set_ylabel("доля внутри группы")
    ax.legend()
    path = FIGURES_DIR / "score_distribution_by_error_type.png"
    save_figure(fig, path)
    return path


def plot_feature_heatmap(rows: pd.DataFrame, features: list[str], importance: pd.DataFrame) -> Path:
    top_features = importance["feature"].head(25).tolist()
    top_features = [feature for feature in top_features if feature in features]
    x = rows[top_features].apply(pd.to_numeric, errors="coerce")
    means = x.mean()
    stds = x.std().replace(0, np.nan)
    z = x.sub(means).div(stds)
    heat = z.join(rows["error_type"]).groupby("error_type", observed=True).mean().reindex(GROUP_ORDER)

    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1.2, vmax=1.2)
    ax.set_xticks(range(len(top_features)), top_features, rotation=45, ha="right")
    ax.set_yticks(range(len(GROUP_ORDER)), [GROUP_LABELS_RU[group] for group in GROUP_ORDER])
    ax.set_title("Профиль top features по TP/FP/FN/TN, z-score относительно test")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="средний z-score")
    path = FIGURES_DIR / "feature_profile_heatmap_top25.png"
    save_figure(fig, path)
    return path


def plot_top_contrasts(contrasts: pd.DataFrame) -> Path:
    selected = contrasts[contrasts["contrast"].isin(["FP_vs_TN", "FN_vs_TP", "TP_vs_FP"])].copy()
    panels = [
        ("FP_vs_TN", "FP vs TN: что делает ложные тревоги похожими на high-risk"),
        ("FN_vs_TP", "FN vs TP: чего не хватает у пропущенных ДТП"),
        ("TP_vs_FP", "TP vs FP: что отличает реальные ДТП среди high-risk"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (contrast_name, title) in zip(axes, panels):
        data = selected[selected["contrast"] == contrast_name].head(12).sort_values("smd")
        colors = np.where(data["smd"] >= 0, "#dc2626", "#2563eb")
        ax.barh(data["feature"], data["smd"], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("SMD")
        ax.grid(axis="x", alpha=0.25)
    path = FIGURES_DIR / "top_feature_contrasts.png"
    save_figure(fig, path)
    return path


def plot_h3_error_scatter(h3_agg: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    panels = [
        ("fp_count", "False positives по H3", "#f59e0b"),
        ("fn_count", "False negatives по H3", "#7c3aed"),
        ("tp_count", "True positives по H3", "#dc2626"),
    ]
    for ax, (column, title, color) in zip(axes, panels):
        sizes = 15 + 120 * (h3_agg[column] / max(h3_agg[column].max(), 1))
        ax.scatter(h3_agg["lng"], h3_agg["lat"], s=sizes, c=color, alpha=0.45, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel("lng")
        ax.set_ylabel("lat")
        ax.grid(alpha=0.2)
    path = FIGURES_DIR / "h3_error_scatter.png"
    save_figure(fig, path)
    return path


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    rows, features, importance = prepare_error_rows()
    summary = build_summary(rows)
    feature_group = build_feature_group_stats(rows, features, importance)
    contrasts = build_feature_contrasts(rows, features, importance)
    h3_agg = build_h3_aggregation(rows)

    rows_out = rows[
        [
            "h3",
            "month",
            TARGET,
            "actual",
            "predicted",
            "error_type",
            SCORE_COLUMN,
            "road_length_total_m",
            "lane_meters_estimate",
            "junction_count",
            "crossing_count",
            "road_curvature_mean_weighted",
            "road_orientation_entropy",
        ]
    ].copy()
    rows_out.to_parquet(ROW_OUTPUT_PATH, index=False)
    h3_agg.to_parquet(H3_OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    feature_group.to_csv(FEATURE_GROUP_PATH, index=False)
    contrasts.to_csv(FEATURE_CONTRAST_PATH, index=False)
    h3_agg.head(100).to_csv(H3_TOP_PATH, index=False)

    figures = [
        plot_confusion_summary(summary),
        plot_score_distribution(rows),
        plot_feature_heatmap(rows, features, importance),
        plot_top_contrasts(contrasts),
        plot_h3_error_scatter(h3_agg),
    ]

    payload = {
        "threshold": THRESHOLD,
        "rows": int(len(rows)),
        "positive_rows": int(rows["actual"].sum()),
        "predicted_positive_rows": int(rows["predicted"].sum()),
        "summary_csv": str(SUMMARY_PATH.relative_to(PACKAGE_ROOT)),
        "feature_group_csv": str(FEATURE_GROUP_PATH.relative_to(PACKAGE_ROOT)),
        "feature_contrasts_csv": str(FEATURE_CONTRAST_PATH.relative_to(PACKAGE_ROOT)),
        "row_parquet": str(ROW_OUTPUT_PATH.relative_to(PACKAGE_ROOT)),
        "h3_parquet": str(H3_OUTPUT_PATH.relative_to(PACKAGE_ROOT)),
        "figures": [str(path.relative_to(PACKAGE_ROOT)) for path in figures],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()
    print("Summary:")
    print(summary.to_string(index=False))
    print()
    for contrast_name in ["FP_vs_TN", "FN_vs_TP", "TP_vs_FP", "FN_vs_TN"]:
        print(f"Top contrast: {contrast_name}")
        print(
            contrasts[contrasts["contrast"] == contrast_name][
                ["feature", "importance", "smd", "left_mean", "right_mean", "zero_share_left", "zero_share_right"]
            ]
            .head(12)
            .to_string(index=False)
        )
        print()
    print("Top H3 by FP/FN/TP counts:")
    print(
        h3_agg[
            [
                "h3",
                "fp_count",
                "fn_count",
                "tp_count",
                "tn_count",
                "actual_positive_rate",
                "predicted_positive_rate",
                "score_mean",
                "road_length_total_m",
                "lat",
                "lng",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
