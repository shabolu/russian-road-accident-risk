from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from road_accident_risk.catboost_baseline import (  # noqa: E402
    FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP,
    FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP_LOG1P,
    apply_log1p_transform,
    apply_p99_clipping,
    build_features,
    default_log1p_features,
    fit_p99_clip_caps,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END  # noqa: E402
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET  # noqa: E402


RANDOM_SEED = 42
TEST_SAMPLE_LIMIT = 120_000
PLOT_SAMPLE_NEGATIVE = 80_000
PLOT_SAMPLE_POSITIVE = 50_000


@dataclass(frozen=True)
class AnalysisSpec:
    key: str
    title: str
    feature_set: str
    feature_importance_path: Path
    p99_caps_path: Path
    apply_log1p: bool
    log1p_features_path: Path | None
    output_csv_path: Path
    top_csv_path: Path
    figures_dir: Path
    notebook_path: Path


SPECS = {
    "road_only_4_0": AnalysisSpec(
        key="road_only_4_0",
        title="CatBoost Road-Only 4.0 Positive vs Negative Feature Analysis",
        feature_set=FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP,
        feature_importance_path=PROJECT_ROOT
        / "reports"
        / "catboost_feature_importance_road_only_4_0_no_calendar_dedup_clip_p99.csv",
        p99_caps_path=PROJECT_ROOT / "reports" / "catboost_road_only_4_0_p99_clip_caps.csv",
        apply_log1p=False,
        log1p_features_path=None,
        output_csv_path=PROJECT_ROOT
        / "reports"
        / "catboost_road_only_4_0_positive_negative_feature_comparison.csv",
        top_csv_path=PROJECT_ROOT
        / "reports"
        / "catboost_road_only_4_0_positive_negative_top_features.csv",
        figures_dir=PROJECT_ROOT / "reports" / "figures" / "catboost_road_only_4_0_positive_negative",
        notebook_path=PROJECT_ROOT / "notebooks" / "catboost_road_only_4_0_positive_negative_analysis.ipynb",
    ),
    "road_only_4_1_log": AnalysisSpec(
        key="road_only_4_1_log",
        title="CatBoost Road-Only 4.1 Log Positive vs Negative Feature Analysis",
        feature_set=FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP_LOG1P,
        feature_importance_path=PROJECT_ROOT / "reports" / "catboost_feature_importance_road_only_4_1_log.csv",
        p99_caps_path=PROJECT_ROOT / "reports" / "catboost_road_only_4_1_log_p99_clip_caps.csv",
        apply_log1p=True,
        log1p_features_path=PROJECT_ROOT / "reports" / "catboost_road_only_4_1_log1p_candidates.csv",
        output_csv_path=PROJECT_ROOT
        / "reports"
        / "catboost_road_only_4_1_log_positive_negative_feature_comparison.csv",
        top_csv_path=PROJECT_ROOT
        / "reports"
        / "catboost_road_only_4_1_log_positive_negative_top_features.csv",
        figures_dir=PROJECT_ROOT
        / "reports"
        / "figures"
        / "catboost_road_only_4_1_log_positive_negative",
        notebook_path=PROJECT_ROOT
        / "notebooks"
        / "catboost_road_only_4_1_log_positive_negative_analysis.ipynb",
    ),
}


def get_spec(key: str) -> AnalysisSpec:
    try:
        return SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown analysis spec: {key}") from exc


def load_caps(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    caps = pd.read_csv(path)
    return {str(row.feature): float(row.p99_cap) for row in caps.itertuples(index=False)}


def load_log1p_features(spec: AnalysisSpec, features: list[str]) -> list[str]:
    if not spec.apply_log1p:
        return []
    if spec.log1p_features_path and spec.log1p_features_path.exists():
        values = pd.read_csv(spec.log1p_features_path)["feature"].astype(str).tolist()
        return [feature for feature in features if feature in values]
    return default_log1p_features(spec.feature_set, features)


def prepare_train_frame(spec: AnalysisSpec) -> tuple[pd.DataFrame, list[str], pd.DataFrame, list[str]]:
    panel = pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)
    panel = panel[panel["road_length_total_m"] > 0].copy()
    panel["month"] = pd.to_datetime(panel["month"])
    panel[TARGET] = panel[TARGET].astype(int)

    features = build_features(panel, feature_set=spec.feature_set)
    df = panel[["h3", "month", *features, TARGET]].copy()
    train = df[df["month"] < TRAIN_END].copy()

    caps = load_caps(spec.p99_caps_path)
    if not caps:
        caps = fit_p99_clip_caps(train, features)
    train = apply_p99_clipping(train, caps)

    log1p_features = load_log1p_features(spec, features)
    train = apply_log1p_transform(train, log1p_features)

    if spec.feature_importance_path.exists():
        importance = pd.read_csv(spec.feature_importance_path)
    else:
        importance = pd.DataFrame({"feature": features, "importance": np.nan})

    return train, features, importance, log1p_features


def sample_array(values: pd.Series, rng: np.random.Generator, limit: int = TEST_SAMPLE_LIMIT) -> np.ndarray:
    arr = values.dropna().to_numpy()
    if len(arr) > limit:
        arr = rng.choice(arr, size=limit, replace=False)
    return arr


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(denominator) <= 1e-12:
        return np.nan
    return float(numerator / denominator)


def safe_auc(y_true: pd.Series, values: pd.Series) -> float:
    clean = pd.DataFrame({"y": y_true, "x": values}).dropna()
    if clean["x"].nunique() <= 1 or clean["y"].nunique() <= 1:
        return np.nan
    return float(roc_auc_score(clean["y"], clean["x"]))


def build_target_comparison(
    df: pd.DataFrame,
    features: list[str],
    importance: pd.DataFrame,
    log1p_features: list[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    y = df[TARGET].astype(int)
    log1p_set = set(log1p_features)
    rows: list[dict[str, Any]] = []

    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        pos = values[y == 1]
        neg = values[y == 0]

        pos_mean = pos.mean()
        neg_mean = neg.mean()
        pos_median = pos.median()
        neg_median = neg.median()
        pos_p95 = pos.quantile(0.95)
        neg_p95 = neg.quantile(0.95)
        pos_std = pos.std()
        neg_std = neg.std()
        pooled_std = (
            math.sqrt((pos_std**2 + neg_std**2) / 2)
            if pd.notna(pos_std) and pd.notna(neg_std)
            else np.nan
        )

        pos_sample = sample_array(pos, rng)
        neg_sample = sample_array(neg, rng)
        if len(pos_sample) and len(neg_sample):
            ks_result = ks_2samp(pos_sample, neg_sample)
            if np.unique(pos_sample).size > 1 or np.unique(neg_sample).size > 1:
                mannwhitney_pvalue = float(
                    mannwhitneyu(pos_sample, neg_sample, alternative="two-sided").pvalue
                )
            else:
                mannwhitney_pvalue = np.nan
        else:
            ks_result = None
            mannwhitney_pvalue = np.nan

        auc = safe_auc(y, values)
        rows.append(
            {
                "feature": feature,
                "transformed_log1p": feature in log1p_set,
                "pos_count": int(pos.notna().sum()),
                "neg_count": int(neg.notna().sum()),
                "mean_pos": pos_mean,
                "mean_neg": neg_mean,
                "mean_ratio_pos_to_neg": safe_ratio(pos_mean, neg_mean),
                "mean_diff_pos_minus_neg": pos_mean - neg_mean,
                "median_pos": pos_median,
                "median_neg": neg_median,
                "median_ratio_pos_to_neg": safe_ratio(pos_median, neg_median),
                "median_diff_pos_minus_neg": pos_median - neg_median,
                "p95_pos": pos_p95,
                "p95_neg": neg_p95,
                "p95_ratio_pos_to_neg": safe_ratio(pos_p95, neg_p95),
                "p95_diff_pos_minus_neg": pos_p95 - neg_p95,
                "zero_share_pos": pos.eq(0).mean(),
                "zero_share_neg": neg.eq(0).mean(),
                "zero_share_diff_pos_minus_neg": pos.eq(0).mean() - neg.eq(0).mean(),
                "ks_statistic": float(ks_result.statistic) if ks_result is not None else np.nan,
                "ks_pvalue": float(ks_result.pvalue) if ks_result is not None else np.nan,
                "mannwhitney_pvalue": mannwhitney_pvalue,
                "mannwhitney_neg_log10_p": (
                    -math.log10(max(mannwhitney_pvalue, 1e-300))
                    if pd.notna(mannwhitney_pvalue)
                    else np.nan
                ),
                "single_feature_roc_auc": auc,
                "single_feature_auc_distance": abs(auc - 0.5) if pd.notna(auc) else np.nan,
                "single_feature_oriented_auc": max(auc, 1 - auc) if pd.notna(auc) else np.nan,
                "auc_direction": (
                    "higher_in_positive"
                    if pd.notna(auc) and auc >= 0.5
                    else "higher_in_negative"
                    if pd.notna(auc)
                    else "undefined"
                ),
                "standardized_mean_difference": (
                    (pos_mean - neg_mean) / pooled_std if pd.notna(pooled_std) and pooled_std > 0 else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows)
    result = result.merge(importance, on="feature", how="left")
    return result.sort_values(
        ["single_feature_auc_distance", "ks_statistic", "importance"],
        ascending=[False, False, False],
    )


def save_fig(fig: plt.Figure, path: Path, show_plots: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_barh(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    path: Path,
    show_plots: bool,
    color: str = "#2563eb",
    xline: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(data[y_col], data[x_col], color=color)
    if xline is not None:
        ax.axvline(xline, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, path, show_plots)


def plot_signed_barh(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    path: Path,
    show_plots: bool,
) -> None:
    colors = np.where(data[x_col] >= 0, "#dc2626", "#2563eb")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(data[y_col], data[x_col], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, path, show_plots)


def add_point_labels(ax: plt.Axes, data: pd.DataFrame, label_col: str, limit: int = 12) -> None:
    label_rows = data.head(limit)
    for row in label_rows.itertuples(index=False):
        ax.annotate(
            getattr(row, label_col),
            (getattr(row, "single_feature_auc_distance"), getattr(row, "importance")),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )


def plot_distribution_pages(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    figures_dir: Path,
    show_plots: bool,
) -> list[Path]:
    focus = comparison.head(12)["feature"].tolist()
    paths = []

    for feature in focus:
        values = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        clean = pd.DataFrame({feature: values, TARGET: df[TARGET]}).dropna()
        pos = clean[clean[TARGET] == 1]
        neg = clean[clean[TARGET] == 0]
        if len(pos) > PLOT_SAMPLE_POSITIVE:
            pos = pos.sample(PLOT_SAMPLE_POSITIVE, random_state=RANDOM_SEED)
        if len(neg) > PLOT_SAMPLE_NEGATIVE:
            neg = neg.sample(PLOT_SAMPLE_NEGATIVE, random_state=RANDOM_SEED)
        plot_data = pd.concat([pos, neg], ignore_index=True)

        upper = clean[feature].quantile(0.995)
        lower = clean[feature].quantile(0.005)
        plot_data[feature] = plot_data[feature].clip(lower=lower, upper=upper)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for target_value, label, color in [(0, "target=0", "#64748b"), (1, "target=1", "#dc2626")]:
            subset = plot_data[plot_data[TARGET] == target_value][feature]
            axes[0].hist(subset, bins=45, alpha=0.48, density=True, label=label, color=color)
        axes[0].set_title(f"{feature}: clipped histogram")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("density")
        axes[0].legend()

        bin_data = clean.copy()
        try:
            bin_data["bin"] = pd.qcut(bin_data[feature], q=10, duplicates="drop")
        except ValueError:
            bin_data["bin"] = pd.cut(bin_data[feature], bins=10, duplicates="drop")
        grouped = (
            bin_data.groupby("bin", observed=True)
            .agg(feature_median=(feature, "median"), target_rate=(TARGET, "mean"), rows=(TARGET, "size"))
            .reset_index(drop=True)
        )
        axes[1].plot(grouped["feature_median"], grouped["target_rate"], marker="o", color="#dc2626")
        axes[1].set_title("Target rate by feature decile")
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel("target rate")

        path = figures_dir / f"feature_distribution__{feature}.png"
        save_fig(fig, path, show_plots)
        paths.append(path)

    return paths


def create_visualizations(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    spec: AnalysisSpec,
    show_plots: bool,
) -> list[Path]:
    spec.figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    top_auc = comparison.sort_values("single_feature_auc_distance", ascending=False).head(25)
    path = spec.figures_dir / "top_single_feature_auc.png"
    plot_barh(
        top_auc,
        "single_feature_roc_auc",
        "feature",
        "Top single-feature ROC-AUC",
        "ROC-AUC of one feature",
        path,
        show_plots,
        color="#0f766e",
        xline=0.5,
    )
    paths.append(path)

    top_ks = comparison.sort_values("ks_statistic", ascending=False).head(25)
    path = spec.figures_dir / "top_ks_statistic.png"
    plot_barh(
        top_ks,
        "ks_statistic",
        "feature",
        "Top KS statistic",
        "KS statistic",
        path,
        show_plots,
        color="#7c3aed",
    )
    paths.append(path)

    top_smd = comparison.reindex(
        comparison["standardized_mean_difference"].abs().sort_values(ascending=False).index
    ).head(25)
    path = spec.figures_dir / "top_standardized_mean_difference.png"
    plot_signed_barh(
        top_smd,
        "standardized_mean_difference",
        "feature",
        "Top standardized mean difference",
        "positive mean minus negative mean, in pooled std",
        path,
        show_plots,
    )
    paths.append(path)

    top_zero = comparison.reindex(
        comparison["zero_share_diff_pos_minus_neg"].abs().sort_values(ascending=False).index
    ).head(25)
    path = spec.figures_dir / "top_zero_share_difference.png"
    plot_signed_barh(
        top_zero,
        "zero_share_diff_pos_minus_neg",
        "feature",
        "Top zero-share difference",
        "zero_share_pos - zero_share_neg",
        path,
        show_plots,
    )
    paths.append(path)

    scatter = comparison.dropna(subset=["importance", "single_feature_auc_distance"]).copy()
    scatter = scatter.sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        scatter["single_feature_auc_distance"],
        scatter["importance"],
        s=45,
        alpha=0.72,
        color="#2563eb",
    )
    add_point_labels(ax, scatter, "feature")
    ax.set_title("CatBoost importance vs single-feature separation")
    ax.set_xlabel("|single-feature ROC-AUC - 0.5|")
    ax.set_ylabel("CatBoost feature importance")
    ax.grid(alpha=0.25)
    path = spec.figures_dir / "importance_vs_single_feature_auc_distance.png"
    save_fig(fig, path, show_plots)
    paths.append(path)

    heat = comparison.head(25).copy()
    metrics = [
        "single_feature_oriented_auc",
        "ks_statistic",
        "standardized_mean_difference",
        "zero_share_diff_pos_minus_neg",
    ]
    matrix = heat[metrics].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(metrics)), metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(heat)), heat["feature"])
    ax.set_title("Top feature effect diagnostics")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    path = spec.figures_dir / "effect_diagnostics_heatmap.png"
    save_fig(fig, path, show_plots)
    paths.append(path)

    paths.extend(plot_distribution_pages(df, comparison, spec.figures_dir, show_plots))
    return paths


def run_analysis(spec_key: str, show_plots: bool = False) -> dict[str, Any]:
    spec = get_spec(spec_key)
    train, features, importance, log1p_features = prepare_train_frame(spec)
    comparison = build_target_comparison(train, features, importance, log1p_features)
    comparison["model_variant"] = spec.key

    spec.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(spec.output_csv_path, index=False)
    comparison.head(30).to_csv(spec.top_csv_path, index=False)
    figure_paths = create_visualizations(train, comparison, spec, show_plots=show_plots)

    summary = {
        "spec": spec.key,
        "title": spec.title,
        "rows": int(len(train)),
        "positive_rows": int(train[TARGET].sum()),
        "positive_rate": float(train[TARGET].mean()),
        "feature_count": len(features),
        "log1p_feature_count": len(log1p_features),
        "output_csv": str(spec.output_csv_path.relative_to(PROJECT_ROOT)),
        "top_csv": str(spec.top_csv_path.relative_to(PROJECT_ROOT)),
        "figures_dir": str(spec.figures_dir.relative_to(PROJECT_ROOT)),
        "figure_count": len(figure_paths),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("Top features by single-feature AUC distance:")
    print(
        comparison[
            [
                "feature",
                "importance",
                "single_feature_roc_auc",
                "single_feature_oriented_auc",
                "auc_direction",
                "ks_statistic",
                "standardized_mean_difference",
                "zero_share_pos",
                "zero_share_neg",
                "mean_pos",
                "mean_neg",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    return {
        "spec": spec,
        "train": train,
        "features": features,
        "importance": importance,
        "comparison": comparison,
        "figure_paths": figure_paths,
        "summary": summary,
    }


def markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def notebook_json(spec: AnalysisSpec) -> dict[str, Any]:
    return {
        "cells": [
            markdown_cell(
                f"""
                # {spec.title}

                Notebook сравнивает строки `target=1` и `target=0` для признаков модели `{spec.key}`.

                Здесь считаются:

                - `mean_pos / mean_neg`;
                - `median_pos / median_neg`;
                - `p95_pos / p95_neg`;
                - `zero_share_pos / zero_share_neg`;
                - KS statistic;
                - Mann-Whitney p-value;
                - single-feature ROC-AUC;
                - standardized mean difference.

                Анализ считается на train-части road-only строк:

                ```text
                month < {TRAIN_END}
                road_length_total_m > 0
                ```

                Для 4.0 применяется `p99 clipping`. Для 4.1 применяется `p99 clipping + log1p`
                ровно к тем признакам, которые использовались в модели.
                """
            ),
            markdown_cell(
                """
                ## 1. Запуск анализа

                Вся логика вынесена в `scripts/positive_negative_feature_analysis.py`,
                чтобы notebook и batch-отчет не расходились.
                """
            ),
            code_cell(
                f"""
                from pathlib import Path
                import sys

                PROJECT_ROOT = Path.cwd()
                if not (PROJECT_ROOT / "scripts").exists() and (PROJECT_ROOT.parent / "scripts").exists():
                    PROJECT_ROOT = PROJECT_ROOT.parent

                sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

                from positive_negative_feature_analysis import run_analysis

                result = run_analysis("{spec.key}", show_plots=True)
                comparison = result["comparison"]
                train = result["train"]
                """
            ),
            markdown_cell(
                """
                ## 2. Главная таблица

                Сортировка по `single_feature_auc_distance`: насколько один признак сам по себе
                отделяет `target=1` от `target=0`.
                """
            ),
            code_cell(
                """
                display(
                    comparison[
                        [
                            "feature",
                            "importance",
                            "transformed_log1p",
                            "single_feature_roc_auc",
                            "single_feature_oriented_auc",
                            "auc_direction",
                            "ks_statistic",
                            "mannwhitney_pvalue",
                            "standardized_mean_difference",
                            "mean_pos",
                            "mean_neg",
                            "mean_ratio_pos_to_neg",
                            "median_pos",
                            "median_neg",
                            "p95_pos",
                            "p95_neg",
                            "zero_share_pos",
                            "zero_share_neg",
                        ]
                    ].head(40)
                )
                """
            ),
            markdown_cell(
                """
                ## 3. Неочевидные случаи

                Эти признаки важны для CatBoost, но слабее выглядят как одиночные разделители.
                Значит, они могут работать через взаимодействия с другими признаками.
                """
            ),
            code_cell(
                """
                interaction_like = comparison.dropna(subset=["importance", "single_feature_auc_distance"]).copy()
                interaction_like = interaction_like[
                    (interaction_like["importance"] >= interaction_like["importance"].quantile(0.70))
                    & (interaction_like["single_feature_auc_distance"] <= interaction_like["single_feature_auc_distance"].quantile(0.50))
                ].sort_values("importance", ascending=False)

                display(
                    interaction_like[
                        [
                            "feature",
                            "importance",
                            "single_feature_roc_auc",
                            "ks_statistic",
                            "standardized_mean_difference",
                            "zero_share_pos",
                            "zero_share_neg",
                        ]
                    ].head(30)
                )
                """
            ),
            markdown_cell(
                """
                ## 4. Где лежат сохраненные артефакты
                """
            ),
            code_cell(
                """
                result["summary"]
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(spec: AnalysisSpec) -> None:
    spec.notebook_path.parent.mkdir(parents=True, exist_ok=True)
    spec.notebook_path.write_text(
        json.dumps(notebook_json(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {spec.notebook_path.relative_to(PROJECT_ROOT)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Positive vs negative feature analysis for CatBoost road-only models.")
    parser.add_argument(
        "--spec",
        choices=["all", *SPECS.keys()],
        default="all",
        help="Analysis spec to run.",
    )
    parser.add_argument("--write-notebooks", action="store_true")
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    keys = list(SPECS) if args.spec == "all" else [args.spec]

    if not args.write_notebooks and not args.run_analysis:
        args.write_notebooks = True
        args.run_analysis = True

    for key in keys:
        spec = get_spec(key)
        if args.write_notebooks:
            write_notebook(spec)
        if args.run_analysis:
            run_analysis(key, show_plots=args.show_plots)


if __name__ == "__main__":
    main()
