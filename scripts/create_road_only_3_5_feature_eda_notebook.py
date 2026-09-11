from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "road_only_3_5_feature_eda.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # Road-only 3.5 Feature EDA

        Анализ признаков для финального baseline абсолютного риска `CatBoost road-only 3.5`.

        Цели:

        - проверить распределения всех 86 признаков;
        - найти пропуски, нули, почти константные признаки и выбросы;
        - сравнить `target=1` и `target=0`;
        - посмотреть корреляции и возможные дубли;
        - выделить кандидатов на `log1p`, clipping/winsorization и бинарные флаги.

        Notebook использует тот же feature set, что и модель:

        ```text
        road_without_type_lengths_no_calendar_slim
        ```

        Выходные артефакты сохраняются в `reports/`.
        """
    ),
    markdown(
        """
        ## 1. Imports and Settings

        Здесь задаются пути, стиль графиков и базовые параметры анализа.
        """
    ),
    code(
        """
        from __future__ import annotations

        import math
        import os
        import sys
        import warnings
        from pathlib import Path

        PROJECT_ROOT = Path.cwd()
        if not (PROJECT_ROOT / "src").exists() and (PROJECT_ROOT.parent / "src").exists():
            PROJECT_ROOT = PROJECT_ROOT.parent

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy.stats import ks_2samp, mannwhitneyu
        from sklearn.metrics import roc_auc_score

        from road_accident_risk.catboost_baseline import (
            FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM,
            build_features,
        )
        from road_accident_risk.logreg_baseline import TARGET, TRAIN_END
        from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        pd.set_option("display.max_columns", 120)
        pd.set_option("display.max_rows", 120)
        pd.set_option("display.float_format", lambda value: f"{value:,.6f}")

        plt.style.use("default")
        plt.rcParams.update(
            {
                "figure.figsize": (11, 6),
                "axes.grid": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "font.size": 10,
            }
        )

        REPORTS_DIR = PROJECT_ROOT / "reports"
        FIGURES_DIR = REPORTS_DIR / "figures" / "road_only_3_5_feature_eda"
        REPORTS_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        IMPORTANCE_PATH = REPORTS_DIR / "catboost_feature_importance_road_only_3_5_no_calendar.csv"
        DISTRIBUTION_SUMMARY_PATH = REPORTS_DIR / "road_only_3_5_feature_distribution_summary.csv"
        TARGET_COMPARISON_PATH = REPORTS_DIR / "road_only_3_5_feature_target_comparison.csv"
        CORRELATION_PAIRS_PATH = REPORTS_DIR / "road_only_3_5_feature_high_correlations.csv"
        TRANSFORM_CANDIDATES_PATH = REPORTS_DIR / "road_only_3_5_feature_transform_candidates.csv"

        RANDOM_SEED = 42
        TEST_SAMPLE_LIMIT = 50_000
        PLOT_SAMPLE_NEGATIVE = 80_000
        PLOT_SAMPLE_POSITIVE = 50_000
        RUN_ALL_FEATURE_PLOTS = True

        rng = np.random.default_rng(RANDOM_SEED)
        """
    ),
    markdown(
        """
        ## 2. Load Data

        Загружается полный H3-month panel, после чего оставляются только road-only строки:

        ```text
        road_length_total_m > 0
        ```

        Признаки берутся через `build_features(...)`, чтобы notebook всегда совпадал с текущим кодом модели.
        """
    ),
    code(
        """
        panel = pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)

        if "road_length_total_m" not in panel.columns:
            raise ValueError("road_length_total_m is required for road-only analysis")

        panel = panel[panel["road_length_total_m"] > 0].copy()
        panel["month"] = pd.to_datetime(panel["month"])
        panel[TARGET] = panel[TARGET].astype(int)

        features = build_features(
            panel,
            feature_set=FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM,
        )

        train = panel[panel["month"] < TRAIN_END].copy()
        test = panel[panel["month"] >= TRAIN_END].copy()

        feature_importance = (
            pd.read_csv(IMPORTANCE_PATH)
            if IMPORTANCE_PATH.exists()
            else pd.DataFrame({"feature": features, "importance": np.nan})
        )

        print(f"Panel path: {H3_MONTH_ROAD_FEATURES_PARQUET}")
        print(f"Road-only rows: {len(panel):,}")
        print(f"Feature count: {len(features):,}")
        print(f"Train rows: {len(train):,} | positive rate: {train[TARGET].mean():.4%}")
        print(f"Test rows:  {len(test):,} | positive rate: {test[TARGET].mean():.4%}")
        print(f"Month range: {panel['month'].min().date()} -> {panel['month'].max().date()}")

        display(feature_importance.head(25))
        """
    ),
    markdown(
        """
        ## 3. Distribution Summary for All Features

        Для каждого признака считаются:

        - `missing_share` — доля пропусков;
        - `zero_share` — доля нулей;
        - `unique_count` — количество уникальных значений;
        - `mean`, `std`, `min`, `max`;
        - `p01`, `p05`, `p25`, `median`, `p75`, `p95`, `p99`.

        Дополнительно добавляются диагностические флаги:

        - `is_constant_or_single_value`;
        - `high_zero_share`;
        - `extreme_tail`;
        - `log1p_candidate`.
        """
    ),
    code(
        """
        def numeric_frame(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
            return (
                df[feature_names]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )


        def build_distribution_summary(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
            x = numeric_frame(df, feature_names)
            q = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
            q.columns = ["p01", "p05", "p25", "median", "p75", "p95", "p99"]

            summary = pd.DataFrame(index=feature_names)
            summary["missing_share"] = x.isna().mean()
            summary["zero_share"] = x.eq(0).mean()
            summary["unique_count"] = x.nunique(dropna=True)
            summary["mean"] = x.mean()
            summary["std"] = x.std()
            summary["min"] = x.min()
            summary = summary.join(q)
            summary["max"] = x.max()

            denominator = summary["median"].abs().replace(0, np.nan)
            summary["p99_to_median_abs"] = summary["p99"].abs() / denominator
            summary["max_to_p99_abs"] = summary["max"].abs() / summary["p99"].abs().replace(0, np.nan)
            summary["range_p99_p01"] = summary["p99"] - summary["p01"]

            summary["is_constant_or_single_value"] = (summary["unique_count"] <= 1) | (
                summary["range_p99_p01"].fillna(0) == 0
            )
            summary["high_zero_share"] = summary["zero_share"] >= 0.90
            summary["extreme_tail"] = (summary["max_to_p99_abs"] >= 5) | (summary["p99_to_median_abs"] >= 20)
            summary["non_negative"] = summary["min"].fillna(0) >= 0
            summary["log1p_candidate"] = (
                summary["non_negative"]
                & (~summary["is_constant_or_single_value"])
                & (
                    summary["extreme_tail"]
                    | summary.index.to_series().str.contains(
                        "length|meters|density|count|total", case=False, regex=True
                    )
                )
            )

            summary = summary.reset_index(names="feature")
            summary = summary.merge(feature_importance, on="feature", how="left")
            return summary.sort_values(
                ["importance", "zero_share", "p99_to_median_abs"],
                ascending=[False, False, False],
            )


        distribution_summary = build_distribution_summary(train, features)
        distribution_summary.to_csv(DISTRIBUTION_SUMMARY_PATH, index=False)

        print(f"Saved: {DISTRIBUTION_SUMMARY_PATH}")
        display(distribution_summary.head(30))
        """
    ),
    markdown(
        """
        ### 3.1 Diagnostic Tables

        Эти таблицы помогают быстро увидеть:

        - почти полностью нулевые признаки;
        - признаки с длинным хвостом;
        - признаки-кандидаты на `log1p`.
        """
    ),
    code(
        """
        print("High zero-share features")
        display(
            distribution_summary.sort_values("zero_share", ascending=False)[
                [
                    "feature",
                    "importance",
                    "zero_share",
                    "unique_count",
                    "median",
                    "p95",
                    "p99",
                    "max",
                ]
            ].head(30)
        )

        print("Extreme-tail features")
        display(
            distribution_summary.sort_values(
                ["extreme_tail", "max_to_p99_abs", "p99_to_median_abs"],
                ascending=[False, False, False],
            )[
                [
                    "feature",
                    "importance",
                    "zero_share",
                    "median",
                    "p95",
                    "p99",
                    "max",
                    "p99_to_median_abs",
                    "max_to_p99_abs",
                    "log1p_candidate",
                ]
            ].head(30)
        )

        print("Log1p candidates")
        display(
            distribution_summary[distribution_summary["log1p_candidate"]][
                [
                    "feature",
                    "importance",
                    "zero_share",
                    "median",
                    "p95",
                    "p99",
                    "max",
                    "p99_to_median_abs",
                    "max_to_p99_abs",
                ]
            ].head(50)
        )
        """
    ),
    markdown(
        """
        ### 3.2 Summary Visuals

        Быстрые обзорные графики по всем признакам:

        - доля нулей;
        - важность CatBoost;
        - выбросность через `max / p99`.
        """
    ),
    code(
        """
        def save_current_fig(name: str) -> Path:
            path = FIGURES_DIR / name
            plt.tight_layout()
            plt.savefig(path, dpi=160, bbox_inches="tight")
            return path


        top_zero = distribution_summary.sort_values("zero_share", ascending=False).head(30)
        top_importance = distribution_summary.sort_values("importance", ascending=False).head(30)
        top_tail = distribution_summary.sort_values("max_to_p99_abs", ascending=False).head(30)

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        axes[0].barh(top_zero["feature"], top_zero["zero_share"], color="#6b7280")
        axes[0].invert_yaxis()
        axes[0].set_title("Top zero-share features")
        axes[0].set_xlabel("zero_share")

        axes[1].barh(top_importance["feature"], top_importance["importance"], color="#2563eb")
        axes[1].invert_yaxis()
        axes[1].set_title("Top CatBoost importance")
        axes[1].set_xlabel("importance")

        axes[2].barh(top_tail["feature"], top_tail["max_to_p99_abs"], color="#b45309")
        axes[2].invert_yaxis()
        axes[2].set_title("Top max / p99")
        axes[2].set_xlabel("max_to_p99_abs")

        path = save_current_fig("feature_summary_overview.png")
        print(f"Saved: {path}")
        plt.show()
        """
    ),
    markdown(
        """
        ## 4. Positive vs Negative Rows

        Сравниваем значения признаков между строками:

        ```text
        target = 1: в ячейке будет ДТП в следующем месяце
        target = 0: в ячейке не будет ДТП в следующем месяце
        ```

        Важные поля:

        - `single_feature_auc` — насколько один признак сам по себе отделяет `target=1` от `target=0`;
        - `single_feature_auc_distance` — расстояние от случайного качества `0.5`;
        - `ks_statistic` — насколько различаются распределения;
        - `standardized_mean_diff` — размер сдвига среднего в стандартных отклонениях.
        """
    ),
    code(
        """
        def sample_array(values: pd.Series, limit: int = TEST_SAMPLE_LIMIT) -> np.ndarray:
            arr = values.dropna().to_numpy()
            if len(arr) > limit:
                arr = rng.choice(arr, size=limit, replace=False)
            return arr


        def safe_auc(y_true: pd.Series, values: pd.Series) -> float:
            clean = pd.DataFrame({"y": y_true, "x": values}).dropna()
            if clean["x"].nunique() <= 1 or clean["y"].nunique() <= 1:
                return np.nan
            return roc_auc_score(clean["y"], clean["x"])


        def safe_ks(pos_values: pd.Series, neg_values: pd.Series) -> tuple[float, float]:
            pos_sample = sample_array(pos_values)
            neg_sample = sample_array(neg_values)
            if len(pos_sample) == 0 or len(neg_sample) == 0:
                return np.nan, np.nan
            result = ks_2samp(pos_sample, neg_sample)
            return float(result.statistic), float(result.pvalue)


        def safe_mannwhitney(pos_values: pd.Series, neg_values: pd.Series) -> float:
            pos_sample = sample_array(pos_values)
            neg_sample = sample_array(neg_values)
            if len(pos_sample) == 0 or len(neg_sample) == 0:
                return np.nan
            if np.unique(pos_sample).size <= 1 and np.unique(neg_sample).size <= 1:
                return np.nan
            return float(mannwhitneyu(pos_sample, neg_sample, alternative="two-sided").pvalue)


        def build_target_comparison(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
            x = numeric_frame(df, feature_names)
            y = df[TARGET].astype(int)
            rows = []

            for feature in feature_names:
                values = x[feature]
                pos = values[y == 1]
                neg = values[y == 0]

                pos_mean = pos.mean()
                neg_mean = neg.mean()
                pos_std = pos.std()
                neg_std = neg.std()
                pooled_std = math.sqrt((pos_std**2 + neg_std**2) / 2) if pd.notna(pos_std) and pd.notna(neg_std) else np.nan
                auc = safe_auc(y, values)
                ks_stat, ks_p = safe_ks(pos, neg)

                rows.append(
                    {
                        "feature": feature,
                        "pos_count": pos.notna().sum(),
                        "neg_count": neg.notna().sum(),
                        "pos_mean": pos_mean,
                        "neg_mean": neg_mean,
                        "mean_diff_pos_minus_neg": pos_mean - neg_mean,
                        "mean_ratio_pos_to_neg": pos_mean / neg_mean if abs(neg_mean) > 1e-12 else np.nan,
                        "pos_median": pos.median(),
                        "neg_median": neg.median(),
                        "median_diff_pos_minus_neg": pos.median() - neg.median(),
                        "pos_p95": pos.quantile(0.95),
                        "neg_p95": neg.quantile(0.95),
                        "pos_zero_share": pos.eq(0).mean(),
                        "neg_zero_share": neg.eq(0).mean(),
                        "zero_share_diff_pos_minus_neg": pos.eq(0).mean() - neg.eq(0).mean(),
                        "standardized_mean_diff": (pos_mean - neg_mean) / pooled_std if pooled_std and pooled_std > 0 else np.nan,
                        "single_feature_auc": auc,
                        "single_feature_auc_distance": abs(auc - 0.5) if pd.notna(auc) else np.nan,
                        "ks_statistic": ks_stat,
                        "ks_pvalue": ks_p,
                        "mannwhitney_pvalue": safe_mannwhitney(pos, neg),
                    }
                )

            result = pd.DataFrame(rows)
            result = result.merge(feature_importance, on="feature", how="left")
            return result.sort_values(
                ["single_feature_auc_distance", "ks_statistic", "importance"],
                ascending=[False, False, False],
            )


        target_comparison = build_target_comparison(train, features)
        target_comparison.to_csv(TARGET_COMPARISON_PATH, index=False)

        print(f"Saved: {TARGET_COMPARISON_PATH}")
        display(target_comparison.head(40))
        """
    ),
    markdown(
        """
        ### 4.1 Target Relationship Visuals

        Графики показывают, какие признаки сильнее всего отличаются между `target=1` и `target=0`.
        """
    ),
    code(
        """
        top_auc = target_comparison.sort_values("single_feature_auc_distance", ascending=False).head(30)
        top_ks = target_comparison.sort_values("ks_statistic", ascending=False).head(30)

        fig, axes = plt.subplots(1, 2, figsize=(17, 8))

        axes[0].barh(top_auc["feature"], top_auc["single_feature_auc"], color="#0f766e")
        axes[0].axvline(0.5, color="black", linewidth=1)
        axes[0].invert_yaxis()
        axes[0].set_title("Top single-feature ROC-AUC")
        axes[0].set_xlabel("ROC-AUC of one feature")

        axes[1].barh(top_ks["feature"], top_ks["ks_statistic"], color="#7c3aed")
        axes[1].invert_yaxis()
        axes[1].set_title("Top KS statistic")
        axes[1].set_xlabel("KS statistic")

        path = save_current_fig("target_relationship_overview.png")
        print(f"Saved: {path}")
        plt.show()
        """
    ),
    markdown(
        """
        ## 5. Feature Distribution Plots

        Для ключевых признаков строятся:

        - clipped raw histogram по `target`;
        - `log1p` histogram, если признак неотрицательный;
        - boxplot с ограничением по p99;
        - target rate по бинам признака, посчитанный на полном train без class sampling.
        """
    ),
    code(
        """
        IMPORTANT_FEATURES_FROM_DISCUSSION = [
            "asphalt_surface_length_m",
            "road_share_residential",
            "dead_end_density_per_km",
            "road_curvature_mean_weighted",
            "road_curvature_max",
            "road_orientation_entropy",
            "road_share_unclassified",
            "road_share_service",
            "lane_meters_estimate",
            "junction_density_per_km",
            "road_segment_length_median",
            "curvature_deg_per_km",
            "sharp_turn_density_per_km",
            "class_delta_junction_density_per_km",
        ]

        top_importance_features = feature_importance["feature"].head(20).tolist()
        focus_features = []
        for feature in IMPORTANT_FEATURES_FROM_DISCUSSION + top_importance_features:
            if feature in features and feature not in focus_features:
                focus_features.append(feature)

        print(f"Focus features: {len(focus_features)}")
        display(pd.DataFrame({"feature": focus_features}))
        """
    ),
    code(
        """
        def sampled_for_plot(df: pd.DataFrame, feature: str) -> pd.DataFrame:
            cols = [feature, TARGET]
            pos = df[df[TARGET] == 1][cols]
            neg = df[df[TARGET] == 0][cols]
            if len(pos) > PLOT_SAMPLE_POSITIVE:
                pos = pos.sample(PLOT_SAMPLE_POSITIVE, random_state=RANDOM_SEED)
            if len(neg) > PLOT_SAMPLE_NEGATIVE:
                neg = neg.sample(PLOT_SAMPLE_NEGATIVE, random_state=RANDOM_SEED)
            return pd.concat([pos, neg], ignore_index=True).dropna()


        def clipped_values(values: pd.Series, upper_quantile: float = 0.99) -> pd.Series:
            if values.dropna().empty:
                return values
            upper = values.quantile(upper_quantile)
            lower = values.quantile(1 - upper_quantile)
            return values.clip(lower=lower, upper=upper)


        def add_target_rate_plot(ax: plt.Axes, df: pd.DataFrame, feature: str) -> None:
            clean = df[[feature, TARGET]].dropna().copy()
            if clean[feature].nunique() <= 1:
                ax.text(0.5, 0.5, "single value", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                return

            try:
                clean["bin"] = pd.qcut(clean[feature], q=10, duplicates="drop")
            except ValueError:
                clean["bin"] = pd.cut(clean[feature], bins=10, duplicates="drop")

            grouped = (
                clean.groupby("bin", observed=True)
                .agg(
                    feature_median=(feature, "median"),
                    target_rate=(TARGET, "mean"),
                    rows=(TARGET, "size"),
                )
                .reset_index(drop=True)
            )
            ax.plot(grouped["feature_median"], grouped["target_rate"], marker="o", color="#dc2626")
            ax.set_title("Target rate by feature bins, full train")
            ax.set_xlabel(feature)
            ax.set_ylabel("target rate")


        def plot_feature_distribution(feature: str, df: pd.DataFrame = train) -> Path:
            data = sampled_for_plot(df, feature)
            data[feature] = pd.to_numeric(data[feature], errors="coerce")
            data = data.dropna()

            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle(feature, fontsize=14, fontweight="bold")

            clipped = clipped_values(data[feature])
            clipped_data = data.copy()
            clipped_data[feature] = clipped

            for target_value, color, label in [(0, "#64748b", "target=0"), (1, "#dc2626", "target=1")]:
                values = clipped_data.loc[clipped_data[TARGET] == target_value, feature]
                axes[0, 0].hist(values, bins=40, alpha=0.55, density=True, color=color, label=label)
            axes[0, 0].set_title("Raw distribution clipped to p01-p99")
            axes[0, 0].set_xlabel(feature)
            axes[0, 0].legend()

            if data[feature].min() >= 0:
                log_data = data.copy()
                log_data[f"log1p_{feature}"] = np.log1p(log_data[feature])
                for target_value, color, label in [(0, "#64748b", "target=0"), (1, "#dc2626", "target=1")]:
                    values = log_data.loc[log_data[TARGET] == target_value, f"log1p_{feature}"]
                    axes[0, 1].hist(values, bins=40, alpha=0.55, density=True, color=color, label=label)
                axes[0, 1].set_title("log1p distribution")
                axes[0, 1].set_xlabel(f"log1p({feature})")
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, "log1p skipped: negative values", ha="center", va="center")
                axes[0, 1].set_axis_off()

            box_data = [
                clipped_data.loc[clipped_data[TARGET] == 0, feature],
                clipped_data.loc[clipped_data[TARGET] == 1, feature],
            ]
            axes[1, 0].boxplot(box_data, labels=["target=0", "target=1"], showfliers=False)
            axes[1, 0].set_title("Boxplot clipped to p01-p99")
            axes[1, 0].set_ylabel(feature)

            add_target_rate_plot(axes[1, 1], df, feature)

            path = FIGURES_DIR / f"feature_distribution__{feature}.png"
            plt.tight_layout(rect=(0, 0, 1, 0.96))
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.show()
            return path


        saved_focus_plots = []
        for feature in focus_features:
            saved_focus_plots.append(plot_feature_distribution(feature))

        print(f"Saved {len(saved_focus_plots)} focus feature plots to {FIGURES_DIR}")
        """
    ),
    markdown(
        """
        ## 6. All-Feature Distribution Grids

        Ниже строятся overview-графики по всем признакам страницами по 12 признаков.

        Чтобы не перегружать notebook, можно поставить:

        ```python
        RUN_ALL_FEATURE_PLOTS = False
        ```
        """
    ),
    code(
        """
        def chunks(items: list[str], size: int) -> list[list[str]]:
            return [items[start : start + size] for start in range(0, len(items), size)]


        def plot_feature_grid(feature_names: list[str], page_index: int, df: pd.DataFrame = train) -> Path:
            ncols = 3
            nrows = math.ceil(len(feature_names) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4.2 * nrows))
            axes = np.array(axes).reshape(-1)
            fig.suptitle(f"All feature distributions, page {page_index}", fontsize=15, fontweight="bold")

            for ax, feature in zip(axes, feature_names):
                data = sampled_for_plot(df, feature)
                data[feature] = pd.to_numeric(data[feature], errors="coerce")
                data = data.dropna()
                if data.empty:
                    ax.set_title(f"{feature}\\nempty")
                    ax.set_axis_off()
                    continue

                data[feature] = clipped_values(data[feature])
                for target_value, color, label in [(0, "#64748b", "0"), (1, "#dc2626", "1")]:
                    values = data.loc[data[TARGET] == target_value, feature]
                    ax.hist(values, bins=30, alpha=0.55, density=True, color=color, label=label)
                ax.set_title(feature, fontsize=9)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.legend(title="target", fontsize=8)

            for ax in axes[len(feature_names) :]:
                ax.set_axis_off()

            path = FIGURES_DIR / f"all_feature_distributions_page_{page_index:02d}.png"
            plt.tight_layout(rect=(0, 0, 1, 0.97))
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.show()
            return path


        all_grid_paths = []
        if RUN_ALL_FEATURE_PLOTS:
            for page_index, feature_page in enumerate(chunks(features, 12), start=1):
                all_grid_paths.append(plot_feature_grid(feature_page, page_index))

        print(f"Saved {len(all_grid_paths)} all-feature pages to {FIGURES_DIR}")
        """
    ),
    markdown(
        """
        ## 7. Correlations and Duplicate Signals

        Считаем Spearman correlation, потому что признаки часто нелинейные, с нулями и длинными хвостами.

        Нас интересуют пары с высокой абсолютной корреляцией: это кандидаты на дублирующиеся признаки.
        """
    ),
    code(
        """
        corr_input = numeric_frame(train, features)
        variable_features = [
            feature
            for feature in features
            if corr_input[feature].nunique(dropna=True) > 1 and corr_input[feature].notna().any()
        ]

        spearman_corr = corr_input[variable_features].corr(method="spearman")

        corr_pairs = (
            spearman_corr.where(np.triu(np.ones(spearman_corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        corr_pairs.columns = ["feature_a", "feature_b", "spearman_corr"]
        corr_pairs["abs_spearman_corr"] = corr_pairs["spearman_corr"].abs()
        corr_pairs = corr_pairs.sort_values("abs_spearman_corr", ascending=False)
        high_corr_pairs = corr_pairs[corr_pairs["abs_spearman_corr"] >= 0.85].copy()

        high_corr_pairs.to_csv(CORRELATION_PAIRS_PATH, index=False)

        print(f"Saved: {CORRELATION_PAIRS_PATH}")
        display(high_corr_pairs.head(80))
        """
    ),
    code(
        """
        corr_focus = []
        for feature in top_importance_features + target_comparison["feature"].head(20).tolist():
            if feature in variable_features and feature not in corr_focus:
                corr_focus.append(feature)
        corr_focus = corr_focus[:35]

        corr_matrix = spearman_corr.loc[corr_focus, corr_focus]

        fig, ax = plt.subplots(figsize=(13, 11))
        image = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_focus)))
        ax.set_xticklabels(corr_focus, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr_focus)))
        ax.set_yticklabels(corr_focus, fontsize=8)
        ax.set_title("Spearman correlation among important features")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        path = save_current_fig("spearman_correlation_important_features.png")
        print(f"Saved: {path}")
        plt.show()
        """
    ),
    markdown(
        """
        ## 8. Transform Candidates

        Здесь собираются признаки, для которых стоит проверить преобразования в следующем модельном эксперименте:

        - `log1p(feature)` для длин, counts, densities и признаков с длинным хвостом;
        - clipping/winsorization для признаков, где `max` сильно больше `p99`;
        - binary flag `feature > 0` для признаков с очень большой долей нулей.
        """
    ),
    code(
        """
        transform_candidates = distribution_summary[
            distribution_summary["log1p_candidate"]
            | distribution_summary["high_zero_share"]
            | distribution_summary["extreme_tail"]
        ].copy()

        transform_candidates["suggest_log1p"] = transform_candidates["log1p_candidate"]
        transform_candidates["suggest_clip_p99"] = transform_candidates["max_to_p99_abs"] >= 5
        transform_candidates["suggest_binary_flag"] = transform_candidates["high_zero_share"] & (
            transform_candidates["unique_count"] > 2
        )

        transform_candidates = transform_candidates.merge(
            target_comparison[
                [
                    "feature",
                    "single_feature_auc",
                    "single_feature_auc_distance",
                    "ks_statistic",
                    "standardized_mean_diff",
                ]
            ],
            on="feature",
            how="left",
        )

        transform_candidates = transform_candidates.sort_values(
            ["importance", "single_feature_auc_distance", "ks_statistic"],
            ascending=[False, False, False],
        )

        transform_candidates.to_csv(TRANSFORM_CANDIDATES_PATH, index=False)

        print(f"Saved: {TRANSFORM_CANDIDATES_PATH}")
        display(
            transform_candidates[
                [
                    "feature",
                    "importance",
                    "zero_share",
                    "median",
                    "p95",
                    "p99",
                    "max",
                    "max_to_p99_abs",
                    "single_feature_auc",
                    "ks_statistic",
                    "suggest_log1p",
                    "suggest_clip_p99",
                    "suggest_binary_flag",
                ]
            ].head(80)
        )
        """
    ),
    markdown(
        """
        ## 9. Quick Interpretation Template

        После выполнения notebook этот блок удобно заполнить руками по таблицам выше.

        Рекомендуемые вопросы:

        1. Какие важные признаки имеют очень длинный хвост?
        2. Какие важные признаки почти всегда равны нулю?
        3. Какие признаки почти полностью дублируют друг друга?
        4. Какие признаки отличаются между `target=1` и `target=0` сами по себе?
        5. Какие признаки важны для CatBoost, но слабо отличаются одномерно?
        6. Какие признаки стоит проверить в следующем модельном эксперименте как `log1p`, clipped или binary flag?
        """
    ),
    code(
        """
        print("Artifacts:")
        for path in [
            DISTRIBUTION_SUMMARY_PATH,
            TARGET_COMPARISON_PATH,
            CORRELATION_PAIRS_PATH,
            TRANSFORM_CANDIDATES_PATH,
        ]:
            print(f"- {path.relative_to(PROJECT_ROOT)}")

        print(f"- figures: {FIGURES_DIR.relative_to(PROJECT_ROOT)}")

        print("\\nTop transform candidates by model importance:")
        display(
            transform_candidates[
                [
                    "feature",
                    "importance",
                    "zero_share",
                    "max_to_p99_abs",
                    "single_feature_auc",
                    "suggest_log1p",
                    "suggest_clip_p99",
                    "suggest_binary_flag",
                ]
            ].head(30)
        )

        print("\\nTop one-feature separators:")
        display(
            target_comparison[
                [
                    "feature",
                    "importance",
                    "single_feature_auc",
                    "single_feature_auc_distance",
                    "ks_statistic",
                    "standardized_mean_diff",
                ]
            ].head(30)
        )
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved: {NOTEBOOK_PATH}")
    print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    main()
