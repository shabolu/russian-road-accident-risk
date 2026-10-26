"""Разности, вердикты и фигуры 7–8 этапа 4 (geometry) -> docs/analysis/features/stage4_geometry/figures/.

Входы: reports/road_geo_v2_geometry_seed_runs.csv, reports/catboost_feature_importance_
{road_geo_v2_tags_stage4_rerun, road_geo_v2_geometry}.csv.
Выходы: reports/road_geo_v2_geometry_delta.csv (все сравнения), fig7, fig8 (PNG 300 dpi + CSV).

2σ — соглашение проекта: 2·sqrt((sd_A² + sd_B²)/n), n = 5 сидов. Вердикт по |Δ|/2σ: < 1 — шум,
1–1.5 — на границе, ≥ 1.5 — значимо. Парная оценка: 2·sd разностей по одинаковым сидам.

Запуск: PYTHONPATH=src venv/bin/python scripts/plot_road_geo_v2_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from plot_road_geo_v2_prune import GRID, INK, MUTED, compare  # noqa: E402

from road_accident_risk.catboost_baseline import (  # noqa: E402
    V2_GEOMETRY_BLOCKS_REPLACED, V2_GEOMETRY_CURVATURE_REPLACED, V2_GEOMETRY_OTHER_REPLACED,
)
from road_accident_risk.paths import PROJECT_ROOT  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
FIG = PROJECT_ROOT / "docs" / "analysis" / "features" / "stage4_geometry" / "figures"
A, B1, B2, B3, B = ("road_geo_v2_tags", "geometry_blocks_only", "geometry_curvature_only", "geometry_other_only",
                    "road_geo_v2_geometry")
EXTRA_COLORS = ["#1baf7a", "#4a3aa7", "#c0398f", "#8a6d1f", "#7a7974"]
COLORS = {A: "#52514e", B1: "#eb6834", B2: "#9ec5f0", B3: "#c0398f", B: "#2a78d6"}
NAMES = {A: "A", B1: "B1", B2: "B2", B3: "B3", B: "B"}
_G = [V2_GEOMETRY_BLOCKS_REPLACED, V2_GEOMETRY_CURVATURE_REPLACED, V2_GEOMETRY_OTHER_REPLACED]
OLD = {f for g in _G for f in g}
NEW = {f for g in _G for v in g.values() for f in v}


def save(fig, name: str, data: pd.DataFrame) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  {name}.png / .csv")


def short(v: str) -> str:
    return (v.replace("geometry_plus_old", "B + 9 старых признаков").replace("geometry_plus_", "B + ")
            .replace("geometry_minus_", "B − ").replace("tags_plus_", "A + "))


def main() -> None:
    runs = pd.read_csv(REPORTS / "road_geo_v2_geometry_seed_runs.csv")
    variants = [v for v in [A, B1, B2, B3, B] if v in set(runs.variant)]
    extra = [v for v in runs.variant.unique() if v not in variants]
    for i, v in enumerate(extra):
        COLORS[v] = EXTRA_COLORS[i % len(EXTRA_COLORS)]; NAMES[v] = f"E{i + 1}"
    variants += extra
    deltas = [compare(runs, A, v) for v in variants if v != A]
    for v in extra:
        if (v.startswith("geometry_") or v.endswith("_fillmedian")) and B in variants:
            deltas.append(compare(runs, B, v))
    d = pd.concat(deltas, ignore_index=True)
    d["verdict_paired"] = np.where(d.delta.abs() < d.two_sd_paired, "шум",
                                   np.where(d.delta.abs() < 1.5 * d.two_sd_paired, "на границе", "значимо"))
    d.to_csv(REPORTS / "road_geo_v2_geometry_delta.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(d[["base", "variant", "metric", "mean_base", "mean_variant", "delta", "two_sigma", "ratio",
                 "two_sd_paired", "verdict", "verdict_paired"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # --- 7. метрики по сидам ---
    sel = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision_top_1pct", "precision@1 %"),
           ("recall_top_5pct", "recall@5 %"), ("recall_top_10pct", "recall@10 %")]
    fig, axes = plt.subplots(1, len(sel), figsize=(3.9 * len(sel), 5.2))
    rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axes, sel):
        for x, v in enumerate(variants):
            vals = runs[runs.variant == v][col].to_numpy()
            ax.scatter(x + rng.uniform(-0.09, 0.09, len(vals)), vals, s=26, color=COLORS[v], alpha=0.85,
                       edgecolor="white", lw=0.8, zorder=3)
            ax.errorbar(x + 0.28, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color=INK, ms=4.5,
                        capsize=3.5, lw=1.1, zorder=4)
        lines = [label]
        for v in (B1, B2, B3, B):
            if v in variants:
                r = d[(d.base == A) & (d.variant == v) & (d.column == col)].iloc[0]
                lines.append(f"{NAMES[v]}: Δ {r.delta:+.4f} ({r.verdict})")
        ax.set_title("\n".join(lines), fontsize=8.3, color=INK, loc="left")
        ax.set_xticks(range(len(variants)), [NAMES[v] + f"\n({int(runs[runs.variant == v].n_features.iloc[0])})"
                                             for v in variants], fontsize=8.5)
        ax.set_xlim(-0.5, len(variants) - 0.4); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
    legend = ("A — road_geo_v2_tags (опорный); B1 — сегментация → block_length_median_m; B2 — 4 старых признака кривизны → "
              "curvature_deg_per_km_v2 + min_curve_radius_m;\nB3 — тупики, ориентация, переезды → v2; B — B1 + B2 + B3 = road_geo_v2_geometry")
    if extra:
        legend += "; " + "; ".join(f"{NAMES[v]} — {short(v)}" for v in extra)
    legend += ". В скобках — число признаков."
    fig.suptitle("Рис. 7. Метрики на тесте (2023–2025), 5 сидов: точки — сиды, чёрное — среднее ± ст. откл.; "
                 "Δ — к варианту A, вердикт по 2σ", x=0.01, ha="left", fontsize=10.5, color=INK, y=1.06)
    fig.tight_layout()
    fig.text(0.01, -0.02, legend, fontsize=8.3, color=MUTED, ha="left", va="top")
    save(fig, "fig7_metrics_seeds", runs[runs.variant.isin(variants)])

    # --- 8. важности ---
    ia = pd.read_csv(REPORTS / "catboost_feature_importance_road_geo_v2_tags_stage4_rerun.csv")
    ib = pd.read_csv(REPORTS / "catboost_feature_importance_road_geo_v2_geometry.csv")
    top = 25
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.8))
    cmap = lambda f: "#eb6834" if f in OLD else ("#2a78d6" if f in NEW else "#b4b3ad")  # noqa: E731
    for ax, imp, title in [(axes[0], ia, f"A: road_geo_v2_tags ({len(ia)})"),
                           (axes[1], ib, f"B: road_geo_v2_geometry ({len(ib)})")]:
        dd = imp.head(top).iloc[::-1]
        ax.barh(range(len(dd)), dd.importance_mean5, xerr=dd.importance_sd5, height=0.72,
                color=[cmap(f) for f in dd.feature], error_kw={"lw": 0.7, "ecolor": MUTED})
        ax.set_yticks(range(len(dd)), dd.feature, fontsize=8)
        ax.set_xlabel("важность CatBoost, среднее по 5 сидам ± ст. откл.")
        ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        s_old = imp[imp.feature.isin(OLD)].importance_mean5.sum()
        s_new = imp[imp.feature.isin(NEW)].importance_mean5.sum()
        ax.set_title(f"Топ-{top}: {title}\nсумма важности: заменяемые {s_old:.1f}, новые {s_new:.1f}",
                     loc="left", fontsize=10, color=INK)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#eb6834", "#2a78d6", "#b4b3ad")],
                   labels=["заменяемые признаки геометрии", "новые признаки этапа 4", "прочие признаки"],
                   loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("Рис. 8. Важности признаков до (A) и после (B) замены геометрии рисовки OSM геометрией улиц",
                 x=0.01, ha="left", fontsize=11, color=INK, y=1.01)
    fig.tight_layout()
    data = (ia.assign(rank_A=np.arange(1, len(ia) + 1))[["feature", "importance_mean5", "importance_sd5", "rank_A"]]
            .merge(ib.assign(rank_B=np.arange(1, len(ib) + 1))[["feature", "importance_mean5", "importance_sd5", "rank_B"]],
                   on="feature", how="outer", suffixes=("_A", "_B")))
    data["group"] = np.where(data.feature.isin(OLD), "replaced", np.where(data.feature.isin(NEW), "new", "other"))
    save(fig, "fig8_importance_A_B", data.sort_values("rank_B"))
    with pd.option_context("display.width", 200):
        print(data.sort_values("rank_A").head(15).to_string(index=False))
        print(data[data.group != "other"].to_string(index=False))


if __name__ == "__main__":
    main()
