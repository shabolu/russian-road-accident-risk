"""Разности, вердикты и фигуры 7–8 этапа 5 (СКДФ) -> docs/analysis/features/stage5_skdf/figures/.

Входы: reports/road_geo_v2_skdf_seed_runs.csv, reports/catboost_feature_importance_
{road_geo_v2_geometry_stage5_rerun, road_geo_v2_skdf}.csv.
Выходы: reports/road_geo_v2_skdf_delta.csv (все сравнения), fig7, fig8 (PNG 300 dpi + CSV).

2σ — соглашение проекта: 2·sqrt((sd_A² + sd_B²)/n), n = 5 сидов. Вердикт по |Δ|/2σ: < 1 — шум,
1–1.5 — на границе, ≥ 1.5 — значимо. Парная оценка: 2·sd разностей по одинаковым сидам.

Запуск: PYTHONPATH=src venv/bin/python scripts/plot_road_geo_v2_skdf.py
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

from road_accident_risk.paths import PROJECT_ROOT  # noqa: E402
from road_accident_risk.skdf_v2 import SKDF_V2_ALL_FEATURES  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
FIG = PROJECT_ROOT / "docs" / "analysis" / "features" / "stage5_skdf" / "figures"
A, B = "road_geo_v2_geometry", "road_geo_v2_skdf"
NAMES = {
    A: "A: road_geo_v2_geometry",
    "skdf_B1_width": "B1: + ширина, ширина/полосу",
    "skdf_B2_coating_replace": "B2: одежда СКДФ вместо OSM",
    "skdf_B2b_coating_both": "B2b: + одежда СКДФ (оба)",
    "skdf_B3_hierarchy": "B3: + категория по СП",
    "skdf_B3b_hierarchy_value": "B3b: B3 + значение дороги",
    "skdf_B4a_lanes_replace": "B4a: полосность СКДФ|OSM вместо OSM",
    "skdf_B4b_lanes_both": "B4b: + полосность СКДФ (оба)",
    "skdf_B5_speed_axle": "B5: + скорость, нагрузка",
    "skdf_B5s_speed": "B5s: + скорость",
    "skdf_B6_traffic": "B6: + real_traffic_max",
    "skdf_B6b_vehkm": "B6b: + veh_km_real",
    "skdf_flag": "флаг: + skdf_has_passport",
    "skdf_plus_skdf_roadway_width_mean_w": "A + только ширина",
    "skdf_C_width_hier": "C: ширина + СП",
    "skdf_C_width_hier_value": "C: ширина + СП + значение",
    "skdf_C_width_speed_axle": "C: ширина + скорость + нагрузка",
    "skdf_C_width_hier_speed_axle": "C: ширина + СП + скорость + нагрузка",
    "skdf_C_all_passed": "C36: все прошедшие (B1+B3+B5+B6)",
    "skdf_C_all_passed_fillmedian": "C36, NaN СКДФ → медиана",
    "skdfC36_minus_skdf_axle_load_mean_w": "C36 − нагрузка",
    "skdfC36_minus_skdf_real_traffic_max": "C36 − real_traffic_max",
    "skdfC36_minus_skdf_speed_mean_w": "C36 − скорость",
    "skdfC36_plus_skdf_value_rank_mean_w": "C36 + значение дороги",
    "skdfC36_with_skdf_B2_coating_replace": "C36, одежда СКДФ вместо OSM",
    "skdfC36_with_skdf_B4a_lanes_replace": "C36, полосность СКДФ|OSM",
    B: "B = B1: road_geo_v2_skdf",
    B + "_fillmedian": "B-med: B, NaN СКДФ → медиана",
}
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def save(fig, name: str, data: pd.DataFrame) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  {name}.png / .csv")


def name(v: str) -> str:
    if v in NAMES:
        return NAMES[v]
    return (v.replace("skdfB_minus_", "B − ").replace("skdfB_plus_", "B + ").replace("skdf_plus_", "A + ")
            .replace("_fillmedian", " (NaN → медиана)"))


def main() -> None:
    runs = pd.read_csv(REPORTS / "road_geo_v2_skdf_seed_runs.csv")
    order = [v for v in NAMES if v in set(runs.variant)]
    order += [v for v in runs.variant.unique() if v not in order]
    deltas = [compare(runs, A, v) for v in order if v != A]
    if B in order:
        deltas += [compare(runs, B, v) for v in order if v.startswith("skdfB_") or v == B + "_fillmedian"]
    if "skdf_C_all_passed" in order:
        deltas += [compare(runs, "skdf_C_all_passed", v) for v in order
                   if v.startswith("skdfC36_") or v == "skdf_C_all_passed_fillmedian"]
    d = pd.concat(deltas, ignore_index=True)
    d["verdict_paired"] = np.where(d.delta.abs() < d.two_sd_paired, "шум",
                                   np.where(d.delta.abs() < 1.5 * d.two_sd_paired, "на границе", "значимо"))
    d.to_csv(REPORTS / "road_geo_v2_skdf_delta.csv", index=False)
    sel = ["roc_auc", "pr_auc", "precision_top_1pct", "precision_top_5pct", "recall_top_5pct", "recall_top_10pct"]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(d[d.column.isin(sel)][["base", "variant", "metric", "mean_variant", "delta", "two_sigma", "ratio",
                                     "two_sd_paired", "verdict", "verdict_paired"]]
              .to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # --- 7. Δ к A по сидам ---
    labels = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision_top_1pct", "precision@1 %"),
              ("recall_top_5pct", "recall@5 %"), ("recall_top_10pct", "recall@10 %")]
    var = [v for v in order if v != A]
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 0.42 * len(var) + 2.2), sharey=True)
    a_runs = runs[runs.variant == A].set_index("seed")
    rows = []
    for ax, (col, lab) in zip(axes, labels):
        for i, v in enumerate(var):
            r = runs[runs.variant == v].set_index("seed")
            diff = r[col] - a_runs.loc[r.index, col]
            row = d[(d.base == A) & (d.variant == v) & (d.column == col)].iloc[0]
            c = BLUE if row.verdict == "значимо" and row.delta > 0 else (
                ORANGE if row.verdict == "значимо" and row.delta < 0 else (MUTED if row.verdict == "шум" else "#9ec5f0"))
            ax.scatter(diff, np.full(len(diff), i) + np.linspace(-0.18, 0.18, len(diff)), s=9, color=c, alpha=0.55, lw=0)
            ax.errorbar(row.delta, i, xerr=row.two_sigma, fmt="o", color=c, ms=5, capsize=3, lw=1.3, zorder=3)
            rows.append({"variant": v, "metric": col, "delta_mean": row.delta, "two_sigma": row.two_sigma,
                         "verdict": row.verdict, "verdict_paired": row.verdict_paired,
                         **{f"delta_seed_{s}": x for s, x in diff.items()}})
        ax.axvline(0, color=INK, lw=0.8)
        ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(f"{lab}\nA = {a_runs[col].mean():.4f}", loc="left", fontsize=9.5, color=INK)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
    axes[0].set_yticks(range(len(var)), [name(v) for v in var], fontsize=8.5)
    axes[0].invert_yaxis()
    for i, v in enumerate(var):
        if v == B:
            for ax in axes:
                ax.axhspan(i - 0.5, i + 0.5, color="#dbe9f9", zorder=0)
    fig.suptitle("Рис. 7. Разность метрик с вариантом A на тесте 2023–2025: точки — 5 сидов, кружок — среднее ± 2σ.\n"
                 "Синий — значимый рост, оранжевый — значимое падение (≥ 1.5·2σ), светло-синий — на границе, серый — шум",
                 x=0.01, ha="left", fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "fig7_metrics_seeds", pd.DataFrame(rows))

    # --- 8. важности ---
    pa = REPORTS / "catboost_feature_importance_road_geo_v2_geometry_stage5_rerun.csv"
    pb = REPORTS / "catboost_feature_importance_road_geo_v2_skdf.csv"
    if not pb.exists():
        return
    ia, ib = pd.read_csv(pa), pd.read_csv(pb)
    replaced = set(ia.feature) - set(ib.feature)
    new = set(ib.feature) - set(ia.feature)
    top = 25
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.8))
    cmap = lambda f: ORANGE if f in replaced else (BLUE if f in new else "#b4b3ad")  # noqa: E731
    for ax, imp, title in [(axes[0], ia, f"A: road_geo_v2_geometry ({len(ia)})"),
                           (axes[1], ib, f"B: road_geo_v2_skdf ({len(ib)})")]:
        dd = imp.head(top).iloc[::-1]
        ax.barh(range(len(dd)), dd.importance_mean5, xerr=dd.importance_sd5, height=0.72,
                color=[cmap(f) for f in dd.feature], error_kw={"lw": 0.7, "ecolor": MUTED})
        ax.set_yticks(range(len(dd)), dd.feature, fontsize=8)
        ax.set_xlabel("важность CatBoost, среднее по 5 сидам ± ст. откл.")
        ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        s_new = imp[imp.feature.isin(new)].importance_mean5.sum()
        s_rep = imp[imp.feature.isin(replaced)].importance_mean5.sum()
        rep_txt = f"заменённые OSM {s_rep:.1f}, " if replaced else ""
        ax.set_title(f"Топ-{top}: {title}\nсумма важности: {rep_txt}признаки СКДФ {s_new:.1f}",
                     loc="left", fontsize=10, color=INK)
    leg = [(ORANGE, "признаки OSM, заменённые в B")] if replaced else []
    leg += [(BLUE, "новые признаки СКДФ"), ("#b4b3ad", "признаки набора A")]
    axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in leg], labels=[t for _, t in leg],
                   loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("Рис. 8. Важности признаков до (A) и после (B) добавления паспортных атрибутов СКДФ",
                 x=0.01, ha="left", fontsize=11, color=INK, y=1.01)
    fig.tight_layout()
    data = (ia.assign(rank_A=np.arange(1, len(ia) + 1))[["feature", "importance_mean5", "importance_sd5", "rank_A"]]
            .merge(ib.assign(rank_B=np.arange(1, len(ib) + 1))[["feature", "importance_mean5", "importance_sd5", "rank_B"]],
                   on="feature", how="outer", suffixes=("_A", "_B")))
    data["group"] = np.where(data.feature.isin(replaced), "replaced",
                             np.where(data.feature.isin(new) | data.feature.isin(SKDF_V2_ALL_FEATURES), "new", "other"))
    save(fig, "fig8_importance_A_B", data.sort_values("rank_B"))
    with pd.option_context("display.width", 200):
        print(data.sort_values("rank_B").head(20).to_string(index=False))


if __name__ == "__main__":
    main()
