from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from textwrap import dedent

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for path in [SCRIPTS_DIR, SRC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from positive_negative_feature_analysis import get_spec, prepare_train_frame  # noqa: E402
from road_accident_risk.logreg_baseline import TARGET  # noqa: E402


FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "positive_negative_axis"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "positive_negative_axis_visualization.ipynb"
SCORE_EXPORT_PATH = PROJECT_ROOT / "reports" / "positive_negative_axis_deciles.csv"
RANDOM_SEED = 42

AXIS_FEATURES = [
    "asphalt_surface_length_m",
    "lane_meters_estimate",
    "road_length_total_m",
    "road_way_count",
    "junction_count",
    "junction_density_per_km",
    "class_delta_max",
    "minor_to_major_junction_count",
    "unsignalized_high_delta_junction_count",
    "public_transport_stop_count",
    "crossing_count",
]


def prepare_axis_data() -> tuple[pd.DataFrame, list[str]]:
    spec = get_spec("road_only_4_0")
    train, features, _, _ = prepare_train_frame(spec)
    axis_features = [feature for feature in AXIS_FEATURES if feature in features]

    x = train[axis_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = x.clip(lower=x.quantile(0.001), upper=x.quantile(0.999), axis=1)

    # log1p makes the explanatory score less dominated by a few very large central cells.
    non_negative = [feature for feature in axis_features if x[feature].min() >= 0]
    x[non_negative] = np.log1p(x[non_negative])

    scaled = StandardScaler().fit_transform(x)
    train = train[["h3", "month", TARGET, *axis_features]].copy()
    train["urban_intensity_axis"] = scaled.mean(axis=1)
    return train, axis_features


def build_deciles(df: pd.DataFrame) -> pd.DataFrame:
    deciles = df[["urban_intensity_axis", TARGET]].copy()
    deciles["decile"] = pd.qcut(deciles["urban_intensity_axis"], q=10, labels=False, duplicates="drop") + 1
    return (
        deciles.groupby("decile", observed=True)
        .agg(
            axis_min=("urban_intensity_axis", "min"),
            axis_median=("urban_intensity_axis", "median"),
            axis_max=("urban_intensity_axis", "max"),
            rows=(TARGET, "size"),
            positives=(TARGET, "sum"),
            target_rate=(TARGET, "mean"),
        )
        .reset_index()
    )


def weighted_hist(values: pd.Series, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(len(values)) / max(len(values), 1)
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, counts


def plot_axis_overview(df: pd.DataFrame, deciles: pd.DataFrame, axis_features: list[str]) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y = df[TARGET].astype(int)
    auc = roc_auc_score(y, df["urban_intensity_axis"])
    pos = df.loc[y == 1, "urban_intensity_axis"]
    neg = df.loc[y == 0, "urban_intensity_axis"]

    lo, hi = df["urban_intensity_axis"].quantile([0.005, 0.995])
    bins = np.linspace(lo, hi, 70)
    neg_x, neg_y = weighted_hist(neg.clip(lo, hi), bins)
    pos_x, pos_y = weighted_hist(pos.clip(lo, hi), bins)

    feature_auc_rows = []
    for feature in axis_features:
        feature_auc_rows.append(
            {
                "feature": feature,
                "auc": roc_auc_score(y, df[feature]),
            }
        )
    feature_auc = pd.DataFrame(feature_auc_rows).sort_values("auc", ascending=True)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], width_ratios=[1.25, 1.0])
    ax_dist = fig.add_subplot(grid[0, :])
    ax_rate = fig.add_subplot(grid[1, 0])
    ax_features = fig.add_subplot(grid[1, 1])

    ax_dist.fill_between(neg_x, neg_y, color="#64748b", alpha=0.38, label="ДТП не будет (target=0)")
    ax_dist.plot(neg_x, neg_y, color="#334155", linewidth=1.6)
    ax_dist.fill_between(pos_x, pos_y, color="#dc2626", alpha=0.32, label="ДТП будет (target=1)")
    ax_dist.plot(pos_x, pos_y, color="#b91c1c", linewidth=1.8)
    ax_dist.axvline(neg.median(), color="#334155", linewidth=1.5, linestyle="--")
    ax_dist.axvline(pos.median(), color="#b91c1c", linewidth=1.5, linestyle="--")
    ax_dist.annotate(
        f"медиана target=0: {neg.median():.2f}",
        xy=(neg.median(), max(neg_y) * 0.92),
        xytext=(-90, 0),
        textcoords="offset points",
        color="#334155",
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#334155"},
    )
    ax_dist.annotate(
        f"медиана target=1: {pos.median():.2f}",
        xy=(pos.median(), max(pos_y) * 0.88),
        xytext=(18, 0),
        textcoords="offset points",
        color="#b91c1c",
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#b91c1c"},
    )
    ax_dist.set_title(
        f"Строки с будущим ДТП сдвинуты к более высокой городской/дорожной насыщенности (AUC={auc:.3f})",
        fontsize=15,
        fontweight="bold",
    )
    ax_dist.set_xlabel("Ось городской насыщенности: средний z-score сильных признаков плотности/экспозиции")
    ax_dist.set_ylabel("доля внутри класса")
    ax_dist.legend(loc="upper left")
    ax_dist.grid(alpha=0.22)

    ax_rate.plot(deciles["decile"], deciles["target_rate"], marker="o", color="#dc2626", linewidth=2)
    baseline = y.mean()
    ax_rate.axhline(baseline, color="#334155", linestyle="--", linewidth=1.2)
    ax_rate.text(
        1,
        baseline,
        f" базовый уровень {baseline:.2%}",
        va="bottom",
        color="#334155",
        fontsize=9,
    )
    ax_rate.set_title("Доля будущих ДТП растет вдоль оси")
    ax_rate.set_xlabel("Дециль городской насыщенности")
    ax_rate.set_ylabel("доля target=1")
    ax_rate.set_xticks(deciles["decile"])
    ax_rate.yaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    ax_rate.grid(alpha=0.22)

    ax_features.barh(feature_auc["feature"], feature_auc["auc"], color="#0f766e", alpha=0.88)
    ax_features.axvline(0.5, color="black", linewidth=1)
    ax_features.set_title("Признаки, из которых собрана ось")
    ax_features.set_xlabel("ROC-AUC одного признака")
    ax_features.grid(axis="x", alpha=0.22)

    fig.tight_layout()
    path = FIGURES_DIR / "urban_intensity_axis_overview.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_shift(df: pd.DataFrame, axis_features: list[str]) -> Path:
    rows = []
    y = df[TARGET].astype(int)
    for feature in axis_features:
        pos = df.loc[y == 1, feature]
        neg = df.loc[y == 0, feature]
        pos_mean = pos.mean()
        neg_mean = neg.mean()
        pooled_std = np.sqrt((pos.std() ** 2 + neg.std() ** 2) / 2)
        rows.append(
            {
                "feature": feature,
                "smd": (pos_mean - neg_mean) / pooled_std if pooled_std > 0 else np.nan,
                "zero_share_pos": (pos == 0).mean(),
                "zero_share_neg": (neg == 0).mean(),
            }
        )
    shifts = pd.DataFrame(rows).sort_values("smd", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    colors = np.where(shifts["smd"] >= 0, "#dc2626", "#2563eb")
    axes[0].barh(shifts["feature"], shifts["smd"], color=colors, alpha=0.88)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Сдвиг признаков: target=1 минус target=0")
    axes[0].set_xlabel("стандартизированная разница средних")
    axes[0].grid(axis="x", alpha=0.22)

    zero_sorted = shifts.assign(diff=shifts["zero_share_pos"] - shifts["zero_share_neg"]).sort_values("diff")
    axes[1].barh(zero_sorted["feature"], zero_sorted["diff"], color="#7c3aed", alpha=0.84)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Сдвиг доли нулей")
    axes[1].set_xlabel("zero_share target=1 минус target=0")
    axes[1].grid(axis="x", alpha=0.22)

    fig.tight_layout()
    path = FIGURES_DIR / "urban_intensity_feature_shifts.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


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


def write_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            markdown(
                """
                # Positive vs Negative: ось городской насыщенности

                Визуализация идеи "оси различия" между строками `target=0` и `target=1`.

                Здесь строится не новая модель, а объяснительная шкала:

                ```text
                urban_intensity_axis = среднее z-score нескольких сильных признаков дорожной/городской насыщенности
                ```

                Если `target=1` заметно сдвинут вправо по этой шкале, значит будущие ДТП-строки в основном отличаются
                именно плотностью и сложностью дорожной среды.
                """
            ),
            code(
                """
                from pathlib import Path
                import sys

                PROJECT_ROOT = Path.cwd()
                if not (PROJECT_ROOT / "scripts").exists() and (PROJECT_ROOT.parent / "scripts").exists():
                    PROJECT_ROOT = PROJECT_ROOT.parent
                sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

                from visualize_positive_negative_axis import main

                main()
                """
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    df, axis_features = prepare_axis_data()
    deciles = build_deciles(df)
    SCORE_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    deciles.to_csv(SCORE_EXPORT_PATH, index=False)

    paths = [
        plot_axis_overview(df, deciles, axis_features),
        plot_feature_shift(df, axis_features),
    ]
    write_notebook()

    print("Признаки оси:")
    for feature in axis_features:
        print(f"- {feature}")
    print()
    print(f"Сохранены децили: {SCORE_EXPORT_PATH.relative_to(PROJECT_ROOT)}")
    for path in paths:
        print(f"Сохранен график: {path.relative_to(PROJECT_ROOT)}")
    print(f"Сохранен notebook: {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")
    print()
    print(deciles.to_string(index=False))


if __name__ == "__main__":
    main()
