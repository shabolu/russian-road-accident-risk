"""Этап 6.0: анализ схемы отбора на валидации и фигуры -> docs/analysis/features/stage6_exposure/validation/.

Входы: reports/v2_eval_stage6_0.csv (сплиты valid, test, valid2021), reports/road_geo_v2_skdf_seed_runs.csv и
reports/road_geo_v2_tags_seed_runs.csv (старые тестовые прогоны — для проверки совпадения).
Выходы: reports/v2_eval_stage6_0_delta.csv (все пары × сплиты), reports/v2_eval_stage6_0_selection.csv,
fig1–fig3 (PNG 300 dpi + CSV рядом).

Запуск: PYTHONPATH=src venv/bin/python scripts/plot_v2_eval_stage6_0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.logreg_baseline import TARGET  # noqa: E402
from road_accident_risk.v2_eval import METRICS, REPORTS, SPLITS, compare, compare_folds, load_panel  # noqa: E402

FIG = REPORTS.parent / "docs" / "analysis" / "features" / "stage6_exposure" / "validation" / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SPLIT_COLOR = {"valid2021": "#1baf7a", "valid": "#2a78d6", "test": "#eb6834"}
SPLIT_LABEL = {"valid2021": "валидация 2021 (fit < 2021)", "valid": "валидация 2022 (fit < 2022)",
               "test": "тест 2023–2025 (train < 2023)"}
A = "road_geo_v2_geometry"
PAIRS = [  # (база, вариант, подпись) — решения этапов 4–5, принятые по тесту
    (A, "road_geo_v2_skdf", "этап 5: B1 (= B) − A"),
    (A, "skdf_B3b_hierarchy_value", "этап 5: B3b − A"),
    (A, "skdf_C_all_passed", "этап 5: C36 − A"),
    (A, "skdf_B5_speed_axle", "этап 5: B5 − A"),
    (A, "skdf_C_width_hier_speed_axle", "этап 5: C35 − A"),
    ("road_geo_v2_tags", A, "этап 4: geometry − tags"),
]
SHORT = {A: "A geometry (30)", "road_geo_v2_skdf": "B1 = B skdf (32)", "skdf_B3b_hierarchy_value": "B3b (32)",
         "skdf_C_all_passed": "C36 (36)", "skdf_B5_speed_axle": "B5 (32)", "skdf_C_width_hier_speed_axle": "C35 (35)",
         "road_geo_v2_tags": "tags (33)"}
SEL = ["roc_auc", "pr_auc", "precision_top_1pct", "precision_top_5pct", "recall_top_5pct", "recall_top_10pct"]
LBL = dict(METRICS)
SPLIT_ORDER = ["valid2021", "valid", "test"]


def save(fig, name: str, data: pd.DataFrame) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  {name}.png / .csv")


def style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(REPORTS / "v2_eval_stage6_0.csv")
    splits = [s for s in SPLIT_ORDER if s in set(runs.split)]

    # --- проверка: тест через модуль = старые seed_runs
    old = pd.concat([pd.read_csv(REPORTS / "road_geo_v2_skdf_seed_runs.csv"),
                     pd.read_csv(REPORTS / "road_geo_v2_tags_seed_runs.csv").query("variant == 'road_geo_v2_tags'")])
    t = runs[runs.split == "test"].merge(old, on=["variant", "seed"], suffixes=("", "_old"))
    cols = [c for c, _ in METRICS]
    maxdiff = max((t[c] - t[c + "_old"]).abs().max() for c in cols) if len(t) else np.nan
    print(f"тест через v2_eval против старых seed_runs: {len(t)} прогонов, max |Δ| = {maxdiff}")

    # --- разности по всем парам и сплитам
    d = pd.concat([compare(runs, b, v, split=s).assign(pair=lbl) for s in splits for b, v, lbl in PAIRS
                   if {b, v} <= set(runs[runs.split == s].variant)], ignore_index=True)
    d.to_csv(REPORTS / "v2_eval_stage6_0_delta.csv", index=False)
    piv = d[d.column.isin(SEL)].pivot_table(index=["pair", "metric"], columns="split",
                                            values=["delta", "two_sigma", "two_sd_paired"])
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(piv.to_string(float_format=lambda x: f"{x:+.5f}"))
        v = d[d.column.isin(SEL)][["split", "pair", "metric", "delta", "verdict", "verdict_paired"]]
        print(v.pivot_table(index=["pair", "metric"], columns="split", values=["verdict", "verdict_paired"],
                            aggfunc="first").to_string())

    # совпадение знаков и ранговая связь Δ (валидация против теста), по метрике
    agree = []
    for s in [x for x in splits if x != "test"]:
        for col in SEL:
            a = d[(d.split == s) & (d.column == col)].set_index("pair").delta
            b = d[(d.split == "test") & (d.column == col)].set_index("pair").delta.reindex(a.index)
            agree.append({"split": s, "metric": LBL[col], "pairs": len(a),
                          "same_sign": int((np.sign(a) == np.sign(b)).sum()),
                          "spearman_delta": spearmanr(a, b).statistic if len(a) > 2 else np.nan,
                          "median_ratio_2sigma_vs_test": float(np.median(
                              d[(d.split == s) & (d.column == col)].set_index("pair").two_sigma.values
                              / d[(d.split == "test") & (d.column == col)].set_index("pair").two_sigma
                              .reindex(a.index).values)),
                          "median_ratio_paired_vs_test": float(np.median(
                              d[(d.split == s) & (d.column == col)].set_index("pair").two_sd_paired.values
                              / d[(d.split == "test") & (d.column == col)].set_index("pair").two_sd_paired
                              .reindex(a.index).values))})
    agree = pd.DataFrame(agree)
    agree.to_csv(REPORTS / "v2_eval_stage6_0_agreement.csv", index=False)
    print(agree.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # что выбрал бы отбор: средние по вариантам этапа 5 и правило 5.9 к A
    sel_rows = []
    for s in splits:
        r = runs[runs.split == s]
        for v in [x for x in SHORT if x not in (A, "road_geo_v2_tags")]:
            c = compare(runs, A, v, split=s); c = c[c.column.isin(SEL)]
            worse = ((c.delta < 0) & (c.verdict != "шум")).any()
            better = ((c.delta > 0) & (c.verdict == "значимо")).any()
            m = r[r.variant == v][SEL].mean()
            sel_rows.append({"split": s, "variant": v, **{k: m[k] for k in SEL},
                             "rule59_pass": bool(better and not worse),
                             "significant_plus": ", ".join(LBL[x] for x in c[(c.delta > 0) & (c.verdict == "значимо")].column),
                             "worse_beyond_2sigma": ", ".join(LBL[x] for x in c[(c.delta < 0) & (c.verdict != "шум")].column)})
    sel = pd.DataFrame(sel_rows)
    sel["rank_pr_auc"] = sel.groupby("split").pr_auc.rank(ascending=False).astype(int)
    sel["rank_roc_auc"] = sel.groupby("split").roc_auc.rank(ascending=False).astype(int)
    sel.to_csv(REPORTS / "v2_eval_stage6_0_selection.csv", index=False)
    with pd.option_context("display.width", 250):
        print(sel.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # скользящая схема: два фолда (2021, 2022) × 5 сидов
    if {"valid", "valid2021"} <= set(splits):
        fo = pd.concat([compare_folds(runs, b, v).assign(pair=lbl) for b, v, lbl in PAIRS], ignore_index=True)
        fo.to_csv(REPORTS / "v2_eval_stage6_0_folds_delta.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_rows", 200):
            print(fo[fo.column.isin(SEL)][["pair", "metric", "delta", "two_sigma", "two_sd_paired", "verdict",
                                           "verdict_paired", "delta_valid2021", "delta_valid", "same_sign_all_folds"]]
                  .to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

    # --- fig1: временная схема и доля положительных по месяцам
    panel = load_panel()
    mon = panel.groupby("month")[TARGET].agg(rows="size", positives="sum").reset_index()
    mon["positive_rate"] = mon.positives / mon.rows
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2]))
    lane = {"valid2021": 2, "valid": 1, "test": 0}
    names = {"valid2021": "фолд 2021\n(опц.)", "valid": "валидация", "test": "тест"}
    start, end = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-01-01")
    for s, y in lane.items():
        sp = SPLITS[s]
        te, es = pd.Timestamp(sp.train_end), pd.Timestamp(sp.eval_start)
        ee = pd.Timestamp(sp.eval_end) if sp.eval_end else end
        ax0.barh(y, (te - start).days, left=start, height=0.6, color="#c9c7c1")
        ax0.barh(y, (ee - es).days, left=es, height=0.6, color=SPLIT_COLOR[s])
        ax0.text(start + (te - start) / 2, y, "обучение + каппы p99", ha="center", va="center", fontsize=7.5, color=INK)
        ax0.text(es + (ee - es) / 2, y, "оценка", ha="center", va="center", fontsize=7.5, color="white")
    ax0.set_yticks(list(lane.values())); ax0.set_yticklabels([names[s] for s in lane], fontsize=8)
    ax0.set_xlim(start, end); style(ax0); ax0.grid(False)
    ax0.set_title("Временная схема: модель для отбора учится на 2015–2021 и оценивается на 2022; "
                  "тест 2023–2025 — только протокол", fontsize=9.5, loc="left", color=INK)
    for s in ("valid2021", "valid", "test"):
        sp = SPLITS[s]; es = pd.Timestamp(sp.eval_start); ee = pd.Timestamp(sp.eval_end) if sp.eval_end else end
        ax1.axvspan(es, ee, color=SPLIT_COLOR[s], alpha=0.08, lw=0)
    ax1.plot(mon.month, mon.positive_rate * 100, color=INK, lw=1.2, marker="o", ms=2.5)
    yr = mon.groupby(mon.month.dt.year).agg(m=("month", "min"), r=("positive_rate", "mean"))
    for _, row in yr.iterrows():
        ax1.hlines(row.r * 100, row.m, row.m + pd.DateOffset(months=11), color=MUTED, lw=2, alpha=0.6)
    ax1.set_ylabel("доля ячеек-месяцев с ДТП\nв следующем месяце, %", fontsize=8.5, color=MUTED)
    ax1.set_xlabel("месяц", fontsize=8.5, color=MUTED); style(ax1)
    ax1.text(0.005, 0.03, "серые отрезки — среднее за год; цветной фон — периоды оценки "
             "(зелёный — фолд 2021, синий — валидация 2022, оранжевый — тест)",
             transform=ax1.transAxes, fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, "fig1_split_timeline_positive_rate", mon)

    # --- fig2: Δ валидация против теста для пар, точки сидов, полосы 2σ
    show = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision_top_1pct", "precision@1 %"),
            ("recall_top_10pct", "recall@10 %")]
    pairs = [p for p in PAIRS if p[2] in set(d.pair)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.75 * len(pairs) + 1.8), sharey=True)
    off = {"valid2021": -0.25, "valid": 0.0, "test": 0.25}
    rows = []
    for ax, (col, lab) in zip(axes, show):
        for i, (b, v, lbl) in enumerate(pairs):
            y0 = len(pairs) - 1 - i
            for s in splits:
                r = runs[runs.split == s]
                if not {b, v} <= set(r.variant):
                    continue
                pa = r[r.variant == v].set_index("seed")[col] - r[r.variant == b].set_index("seed")[col]
                c = d[(d.split == s) & (d.pair == lbl) & (d.column == col)].iloc[0]
                y = y0 + off[s]
                ax.errorbar(c.delta, y, xerr=c.two_sigma, fmt="o", ms=6, color=SPLIT_COLOR[s], ecolor=SPLIT_COLOR[s],
                            elinewidth=2, capsize=0, alpha=0.95, zorder=3)
                ax.scatter(pa.values, np.full(len(pa), y), s=9, color="white", edgecolor=SPLIT_COLOR[s], lw=0.8, zorder=4)
                for seed, val in pa.items():
                    rows.append({"pair": lbl, "metric": lab, "split": s, "seed": seed, "delta_seed": val,
                                 "delta_mean": c.delta, "two_sigma": c.two_sigma, "two_sd_paired": c.two_sd_paired,
                                 "verdict": c.verdict, "verdict_paired": c.verdict_paired})
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(lab, fontsize=10, color=INK); ax.set_xlabel("Δ (вариант − база)", fontsize=8, color=MUTED)
        style(ax)
    axes[0].set_yticks(range(len(pairs))); axes[0].set_yticklabels([p[2] for p in pairs][::-1], fontsize=8.5)
    handles = [plt.Line2D([], [], marker="o", color=SPLIT_COLOR[s], lw=2, label=SPLIT_LABEL[s]) for s in splits[::-1]]
    fig.legend(handles=handles, loc="upper center", ncol=len(splits), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Δ на валидации и на тесте для решений этапов 4–5: крупная точка — среднее по 5 сидам, полоса — ±2σ, "
                 "мелкие — разность на каждом сиде", fontsize=9.5, y=-0.01, color=MUTED)
    fig.tight_layout()
    save(fig, "fig2_delta_valid_vs_test", pd.DataFrame(rows))

    # --- fig3: разброс по сидам (ст. откл.) по вариантам и сплитам
    sd = runs.groupby(["split", "variant"])[SEL].std().reset_index()
    sdl = sd.melt(id_vars=["split", "variant"], var_name="column", value_name="sd_across_seeds")
    fig, axes = plt.subplots(1, len(SEL), figsize=(16, 3.8))
    rng = np.random.default_rng(0)
    for ax, col in zip(axes, SEL):
        for j, s in enumerate(splits):
            vals = sdl[(sdl.split == s) & (sdl.column == col)].sd_across_seeds.values
            ax.scatter(j + rng.uniform(-0.12, 0.12, len(vals)), vals, s=22, color=SPLIT_COLOR[s], alpha=0.85,
                       edgecolor="white", lw=0.5, zorder=3)
            ax.hlines(np.median(vals), j - 0.28, j + 0.28, color=INK, lw=1.5, zorder=4)
        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels([{"valid2021": "вал.\n2021", "valid": "вал.\n2022", "test": "тест"}[s] for s in splits],
                           fontsize=8)
        ax.set_title(LBL[col], fontsize=9.5, color=INK); style(ax); ax.set_ylim(bottom=0)
    axes[0].set_ylabel("ст. откл. по 5 сидам", fontsize=8.5, color=MUTED)
    fig.suptitle("Разброс метрик по сидам: точка — вариант (7 наборов), черта — медиана", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "fig3_seed_spread_by_split", sdl)


if __name__ == "__main__":
    main()
