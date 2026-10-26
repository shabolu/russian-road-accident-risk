"""Разности, вердикты и фигуры 6–7 этапа 2 (junctions) -> docs/analysis/features/stage2_junctions/figures/.

Входы: reports/road_geo_v2_junctions_seed_runs.csv, reports/catboost_feature_importance_
{road_geo_v2_prune_stage2_rerun, road_geo_v2_junctions, road_geo_v2_junctions_C_prune_plus_new}.csv.
Выходы: reports/road_geo_v2_junctions_delta.csv (все сравнения), fig6, fig7 (PNG 300 dpi + CSV).

2σ — соглашение проекта (как в этапе 1): 2·sqrt((sd_A² + sd_B²)/n), n = 5 сидов.
Вердикт по |Δ|/2σ: < 1 — шум, 1–1.5 — на границе, ≥ 1.5 — значимо. Дополнительно — парная оценка:
2·sd разностей по одинаковым сидам (колонка two_sd_paired).

Запуск: PYTHONPATH=src venv/bin/python scripts/plot_road_geo_v2_junctions.py
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

from road_accident_risk.catboost_baseline import V2_JUNCTIONS_REPLACED_FEATURES  # noqa: E402
from road_accident_risk.osm_road_features import JUNCTION_V2_FEATURES  # noqa: E402
from road_accident_risk.paths import PROJECT_ROOT  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
FIG = PROJECT_ROOT / "docs" / "analysis" / "features" / "stage2_junctions" / "figures"
A, B, C = "road_geo_v2_prune", "road_geo_v2_junctions", "prune_plus_junctions"
COLORS = {A: "#52514e", B: "#2a78d6", C: "#eb6834", "junctions_alt_4leg_unsig": "#1baf7a",
          "junctions_no_angle": "#4a3aa7"}
NAMES = {A: "A", B: "B", C: "C", "junctions_alt_4leg_unsig": "B′", "junctions_no_angle": "B″"}
LEGEND = ("A — road_geo_v2_prune (опорный); B — road_geo_v2_junctions (замена старых перекрёстков новыми); "
          "C — prune + новые без удаления старых;\nB′ — B с 4-сторонними без светофора вместо всех 4-сторонних; "
          "B″ — B без отклонения угла. В скобках — число признаков.")
OLD = set(V2_JUNCTIONS_REPLACED_FEATURES)
NEW = set(JUNCTION_V2_FEATURES)


def save(fig, name: str, data: pd.DataFrame) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  {name}.png / .csv")


def main() -> None:
    runs = pd.read_csv(REPORTS / "road_geo_v2_junctions_seed_runs.csv")
    variants = [v for v in [A, B, C, "junctions_alt_4leg_unsig", "junctions_no_angle"] if v in set(runs.variant)]
    variants += [v for v in runs.variant.unique() if v not in variants]
    deltas = [compare(runs, A, v) for v in variants if v != A]
    if B in variants and C in variants:
        deltas.append(compare(runs, B, C))
    for v in variants:
        if v not in (A, B, C) and B in variants:
            deltas.append(compare(runs, B, v))
    d = pd.concat(deltas, ignore_index=True)
    d["verdict_paired"] = np.where(d.delta.abs() < d.two_sd_paired, "шум",
                                   np.where(d.delta.abs() < 1.5 * d.two_sd_paired, "на границе", "значимо"))
    d.to_csv(REPORTS / "road_geo_v2_junctions_delta.csv", index=False)
    with pd.option_context("display.width", 250):
        print(d[["base", "variant", "metric", "mean_base", "mean_variant", "delta", "two_sigma", "ratio",
                 "two_sd_paired", "verdict", "verdict_paired"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # --- 6. метрики по сидам ---
    sel = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision_top_1pct", "precision@1 %"),
           ("recall_top_5pct", "recall@5 %"), ("recall_top_10pct", "recall@10 %")]
    show = [v for v in variants if v in COLORS]
    fig, axes = plt.subplots(1, len(sel), figsize=(3.4 * len(sel), 4.6))
    rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axes, sel):
        for x, v in enumerate(show):
            vals = runs[runs.variant == v][col].to_numpy()
            ax.scatter(x + rng.uniform(-0.09, 0.09, len(vals)), vals, s=26, color=COLORS[v], alpha=0.8,
                       edgecolor="white", lw=0.8, zorder=3)
            ax.errorbar(x + 0.28, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color=INK, ms=4.5,
                        capsize=3.5, lw=1.1, zorder=4)
        lines = [label]
        for v in (B, C):
            if v in show:
                r = d[(d.base == A) & (d.variant == v) & (d.column == col)].iloc[0]
                lines.append(f"{'B' if v == B else 'C'}: Δ {r.delta:+.4f} ({r.verdict})")
        ax.set_title("\n".join(lines), fontsize=8.5, color=INK, loc="left")
        ax.set_xticks(range(len(show)), [NAMES[v] + f"\n({int(runs[runs.variant == v].n_features.iloc[0])})"
                                         for v in show], fontsize=8.5, rotation=0)
        ax.set_xlim(-0.5, len(show) - 0.4); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Рис. 6. Метрики на тесте (2023–2025), 5 сидов: точки — сиды, чёрное — среднее ± ст. откл.; "
                 "Δ — к варианту A, вердикт по 2σ", x=0.01, ha="left", fontsize=10.5, color=INK, y=1.03)
    fig.text(0.01, -0.06, LEGEND, fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    save(fig, "fig6_metrics_seeds", runs[runs.variant.isin(show)])

    # --- 7. важности ---
    ia = pd.read_csv(REPORTS / "catboost_feature_importance_road_geo_v2_prune_stage2_rerun.csv")
    ib = pd.read_csv(REPORTS / "catboost_feature_importance_road_geo_v2_junctions.csv")
    ic = pd.read_csv(REPORTS / "catboost_feature_importance_road_geo_v2_junctions_C_prune_plus_new.csv")
    top = 25
    fig, axes = plt.subplots(1, 3, figsize=(19, 8.8))
    cmap = lambda f: "#eb6834" if f in OLD else ("#2a78d6" if f in NEW else "#b4b3ad")  # noqa: E731
    for ax, imp, title in [(axes[0], ia, f"A: road_geo_v2_prune ({len(ia)})"),
                           (axes[1], ib, f"B: road_geo_v2_junctions ({len(ib)})"),
                           (axes[2], ic, f"C: prune + новые ({len(ic)})")]:
        dd = imp.head(top).iloc[::-1]
        ax.barh(range(len(dd)), dd.importance_mean5, xerr=dd.importance_sd5, height=0.72,
                color=[cmap(f) for f in dd.feature], error_kw={"lw": 0.7, "ecolor": MUTED})
        ax.set_yticks(range(len(dd)), dd.feature, fontsize=7.8)
        ax.set_xlabel("важность CatBoost, среднее по 5 сидам ± ст. откл.")
        ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        s_old = imp[imp.feature.isin(OLD)].importance_mean5.sum()
        s_new = imp[imp.feature.isin(NEW)].importance_mean5.sum()
        ax.set_title(f"Топ-{top}: {title}\nсумма важности: старые перекрёстки {s_old:.1f}, новые {s_new:.1f}",
                     loc="left", fontsize=10, color=INK)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#eb6834"),
                            plt.Rectangle((0, 0), 1, 1, color="#2a78d6"),
                            plt.Rectangle((0, 0), 1, 1, color="#b4b3ad")],
                   labels=["старые признаки перекрёстков", "новые признаки перекрёстков", "прочие признаки"],
                   loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("Рис. 7. Важности признаков: опорный набор, замена старых перекрёстков новыми и добавление "
                 "новых к старым", x=0.01, ha="left", fontsize=11, color=INK, y=1.01)
    fig.tight_layout()
    data = (ia.assign(rank_A=np.arange(1, len(ia) + 1))[["feature", "importance_mean5", "importance_sd5", "rank_A"]]
            .merge(ib.assign(rank_B=np.arange(1, len(ib) + 1))[["feature", "importance_mean5", "importance_sd5", "rank_B"]],
                   on="feature", how="outer", suffixes=("_A", "_B"))
            .merge(ic.assign(rank_C=np.arange(1, len(ic) + 1)).rename(
                columns={"importance_mean5": "importance_mean5_C", "importance_sd5": "importance_sd5_C"})
                [["feature", "importance_mean5_C", "importance_sd5_C", "rank_C"]], on="feature", how="outer"))
    data["group"] = np.where(data.feature.isin(OLD), "old_junction", np.where(data.feature.isin(NEW), "new_junction", "other"))
    save(fig, "fig7_importance_A_B_C", data.sort_values("rank_C"))


if __name__ == "__main__":
    main()
