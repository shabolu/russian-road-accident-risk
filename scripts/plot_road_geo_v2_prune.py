"""Разности, вердикты и фигуры этапа 1 (prune) -> docs/analysis/features/stage1_prune/figures/.

Входы: reports/road_geo_v2_prune_seed_runs.csv, reports/road_geo_baseline_spearman_matrix.csv,
reports/road_geo_baseline_feature_stats.csv, reports/catboost_feature_importance_road_geo_{baseline_5seeds,v2_prune}.csv.
Каждая фигура — PNG 300 dpi + CSV с исходными данными рядом.

2σ — соглашение проекта: 2·sqrt((sd_A² + sd_B²)/n), n = 5 сидов (удвоенная ст. ошибка разности средних).
Вердикт по |Δ|/2σ: < 1 — шум, 1–1.5 — на границе, ≥ 1.5 — значимо.

Запуск: venv/bin/python scripts/plot_road_geo_v2_prune.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.catboost_baseline import V2_PRUNE_EXCLUDED_FEATURES  # noqa: E402
from road_accident_risk.paths import PROJECT_ROOT  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
FIG = PROJECT_ROOT / "docs" / "analysis" / "features" / "stage1_prune" / "figures"
BASE, PRUNE = "road_geo_baseline", "road_geo_v2_prune"
STEP_ORDER = [BASE, "prune_step1_trivial", "prune_step2_junctions", "prune_step3_geometry", PRUNE]
STEP_LABELS = {
    BASE: "бейзлайн",
    "prune_step1_trivial": "− нули, композиция,\nполнота OSM, скорость,\nдлина",
    "prune_step2_junctions": "− перекрёстки",
    "prune_step3_geometry": "− кривизна,\nсегментация, тупики",
    PRUNE: "− покрытие\n(итог)",
}
METRICS = [
    ("roc_auc", "ROC-AUC", 6), ("pr_auc", "PR-AUC", 6),
    ("precision_top_1pct", "precision@1 %", 4), ("recall_top_1pct", "recall@1 %", 4),
    ("lift_top_1pct", "lift@1 %", 2),
    ("precision_top_5pct", "precision@5 %", 4), ("recall_top_5pct", "recall@5 %", 4),
    ("lift_top_5pct", "lift@5 %", 2),
    ("precision_top_10pct", "precision@10 %", 4), ("recall_top_10pct", "recall@10 %", 4),
    ("lift_top_10pct", "lift@10 %", 2),
]
KEEP_C, DROP_C, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8f1d1c"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def verdict(delta: float, two_sigma: float) -> str:
    r = abs(delta) / two_sigma if two_sigma > 0 else np.inf
    return "шум" if r < 1 else ("на границе" if r < 1.5 else "значимо")


def compare(runs: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    A = runs[runs.variant == a].set_index("seed"); B = runs[runs.variant == b].set_index("seed")
    rows = []
    for col, label, _ in METRICS:
        n = len(A)
        d = B[col].mean() - A[col].mean()
        ts = 2 * np.sqrt((A[col].var() + B[col].var()) / n)
        paired = (B[col] - A.loc[B.index, col])
        rows.append({"base": a, "variant": b, "n_features_base": int(A.n_features.iloc[0]),
                     "n_features_variant": int(B.n_features.iloc[0]), "metric": label, "column": col,
                     "mean_base": A[col].mean(), "sd_base": A[col].std(),
                     "mean_variant": B[col].mean(), "sd_variant": B[col].std(),
                     "delta": d, "two_sigma": ts, "ratio": abs(d) / ts if ts else np.inf,
                     "two_sd_paired": 2 * paired.std(), "verdict": verdict(d, ts)})
    return pd.DataFrame(rows)


def save(fig, name: str, data: pd.DataFrame) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  {name}.png / .csv")


def cluster_order(corr: pd.DataFrame) -> list[str]:
    dist = 1 - corr.abs().fillna(0).to_numpy()
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    return list(corr.columns[leaves_list(linkage(squareform(dist, checks=False), "average"))])


def heatmap(corr: pd.DataFrame, name: str, title: str, mark_dropped: bool) -> None:
    order = cluster_order(corr)
    c = corr.loc[order, order]
    n = len(order)
    size = 0.19 * n + 3.5
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    im = ax.imshow(c.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    labels = [("× " + f) if (mark_dropped and f in V2_PRUNE_EXCLUDED_FEATURES) else f for f in order]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5); ax.set_yticklabels(labels, fontsize=6.5)
    if mark_dropped:
        for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            if t.get_text().startswith("× "):
                t.set_color(DROP_C); t.set_fontweight("bold")
    for i in range(n):
        for j in range(n):
            if i != j and abs(c.iat[i, j]) >= 0.85:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=0.6))
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("ρ Спирмена (по ячейкам H3)")
    sub = ("рамка — пара с |ρ| ≥ 0.85; оранжевым и «×» — признаки, удалённые на этапе 1"
           if mark_dropped else "рамка — пара с |ρ| ≥ 0.85")
    ax.set_title(f"{title}\n{sub}", fontsize=11, loc="left", color=INK)
    out = c.copy(); out.insert(0, "feature", out.index)
    save(fig, name, out)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(REPORTS / "road_geo_v2_prune_seed_runs.csv")
    variants = list(runs.variant.unique())

    # --- разности ---
    main_delta = compare(runs, BASE, PRUNE)
    main_delta.to_csv(REPORTS / "road_geo_v2_prune_delta.csv", index=False)
    others = [compare(runs, BASE, v) for v in variants if v != BASE]
    steps = [v for v in STEP_ORDER if v in variants]
    others += [compare(runs, a, b) for a, b in zip(steps, steps[1:]) if a != BASE]
    others += [compare(runs, PRUNE, v) for v in variants if v.startswith("prune_plus_")]
    all_delta = pd.concat(others, ignore_index=True).drop_duplicates(["base", "variant", "metric"])
    all_delta.to_csv(REPORTS / "road_geo_v2_prune_all_deltas.csv", index=False)
    print(main_delta[["metric", "mean_base", "sd_base", "mean_variant", "sd_variant", "delta",
                      "two_sigma", "verdict"]].to_string(index=False))

    # --- 1. тепловые карты ---
    corr = pd.read_csv(REPORTS / "road_geo_baseline_spearman_matrix.csv", index_col=0)
    heatmap(corr, "fig1a_spearman_before_61",
            f"Корреляции Спирмена признаков бейзлайна ({len(corr)}), кластерная сортировка", True)
    kept = [f for f in corr.columns if f not in V2_PRUNE_EXCLUDED_FEATURES]
    heatmap(corr.loc[kept, kept], "fig1b_spearman_after_33",
            f"Корреляции Спирмена итогового набора road_geo_v2_prune ({len(kept)})", False)

    # --- 2. доля нулей ---
    st = pd.read_csv(REPORTS / "road_geo_baseline_feature_stats.csv")
    st["dropped"] = st.feature.isin(V2_PRUNE_EXCLUDED_FEATURES)
    st = st.sort_values("zero_share_cells")
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.barh(range(len(st)), st.zero_share_cells * 100, height=0.72,
            color=np.where(st.dropped, DROP_C, KEEP_C))
    ax.set_yticks(range(len(st))); ax.set_yticklabels(st.feature, fontsize=7)
    ax.axvline(95, color=INK, lw=0.8, ls="--")
    ax.text(94.5, len(st) - 0.5, "95 %", color=INK, fontsize=8, va="bottom", ha="right")
    ax.set_xlim(0, 100); ax.set_xlabel("доля ячеек с нулевым значением, %")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=KEEP_C), plt.Rectangle((0, 0), 1, 1, color=DROP_C)],
              labels=["оставлен", "удалён на этапе 1"], loc="center right", frameon=False)
    ax.set_title(f"Доля нулей по признакам бейзлайна (n = {int(st.cells.max())} ячеек с дорогами)",
                 loc="left", fontsize=11, color=INK)
    save(fig, "fig2_zero_share", st[["feature", "zero_share_cells", "nan_share_cells", "single_auc_test",
                                     "importance_seed42", "dropped"]])

    # --- 3. метрики по сидам ---
    sel = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision_top_1pct", "precision@1 %"),
           ("recall_top_1pct", "recall@1 %"), ("recall_top_5pct", "recall@5 %")]
    two = runs[runs.variant.isin([BASE, PRUNE])]
    nb, npr = int(two[two.variant == BASE].n_features.iloc[0]), int(two[two.variant == PRUNE].n_features.iloc[0])
    names = {BASE: f"бейзлайн\n({nb})", PRUNE: f"prune\n({npr})"}
    fig, axes = plt.subplots(1, len(sel), figsize=(15, 3.8))
    rng = np.random.default_rng(0)
    for ax, (col, label) in zip(axes, sel):
        for x, (v, c) in enumerate([(BASE, KEEP_C), (PRUNE, DROP_C)]):
            vals = two[two.variant == v][col].to_numpy()
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(vals)), vals, s=28, color=c, alpha=0.75,
                       edgecolor="white", lw=0.8, zorder=3)
            ax.errorbar(x + 0.25, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color=INK, ms=5,
                        capsize=4, lw=1.2, zorder=4)
        r = main_delta.set_index("column").loc[col]
        ax.set_title(f"{label}\nΔ = {r.delta:+.4f}, 2σ = {r.two_sigma:.4f}\n{r.verdict}", fontsize=9.5, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels([names[BASE], names[PRUNE]])
        ax.set_xlim(-0.5, 1.6); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.suptitle("Метрики на тесте (2023–2025), 5 сидов: точки — сиды, чёрное — среднее ± ст. откл.",
                 x=0.01, ha="left", fontsize=11, color=INK, y=1.08)
    fig.tight_layout()
    save(fig, "fig3_metrics_seeds", two[["variant", "seed", "n_features", *[c for c, _ in sel]]])

    # --- 4. важности бок о бок ---
    ib = pd.read_csv(REPORTS / "catboost_feature_importance_road_geo_baseline_5seeds.csv")
    ip = pd.read_csv(REPORTS / "catboost_feature_importance_road_geo_v2_prune.csv")
    top = 25
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5))
    for ax, imp, title in [(axes[0], ib, f"бейзлайн ({len(ib)} признак)"),
                           (axes[1], ip, f"road_geo_v2_prune ({len(ip)} признака)")]:
        d = imp.head(top).iloc[::-1]
        dropped = d.feature.isin(V2_PRUNE_EXCLUDED_FEATURES)
        ax.barh(range(len(d)), d.importance_mean5, xerr=d.importance_sd5, height=0.72,
                color=np.where(dropped, DROP_C, KEEP_C), error_kw={"lw": 0.7, "ecolor": MUTED})
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([("× " + f) if x else f for f, x in zip(d.feature, dropped)], fontsize=8)
        ax.set_xlabel("важность CatBoost, среднее по 5 сидам ± ст. откл.")
        ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(f"Топ-{top}: {title}", loc="left", fontsize=11, color=INK)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=KEEP_C), plt.Rectangle((0, 0), 1, 1, color=DROP_C)],
                   labels=["остаётся в prune", "удалён на этапе 1 («×»)"], loc="lower right", frameon=False)
    fig.tight_layout()
    rank_b = ib.assign(rank_baseline=np.arange(1, len(ib) + 1))
    rank_p = ip.assign(rank_prune=np.arange(1, len(ip) + 1))
    imp_data = rank_b.merge(rank_p, on="feature", how="outer", suffixes=("_baseline", "_prune"))
    imp_data["dropped"] = imp_data.feature.isin(V2_PRUNE_EXCLUDED_FEATURES)
    save(fig, "fig4_importance_side_by_side", imp_data.sort_values("rank_baseline"))

    # --- 5. шаги удаления групп ---
    if len(steps) > 2:
        cols = [("precision_top_1pct", "precision@1 %"), ("recall_top_1pct", "recall@1 %"),
                ("recall_top_5pct", "recall@5 %"), ("pr_auc", "PR-AUC")]
        fig, axes = plt.subplots(1, len(cols), figsize=(16, 4.6), sharey=False)
        rows = []
        for ax, (col, label) in zip(axes, cols):
            xs, ds, ts, ns = [], [], [], []
            for i, v in enumerate(steps):
                if v == BASE:
                    d, t = 0.0, 0.0
                else:
                    r = compare(runs, BASE, v).set_index("column").loc[col]; d, t = r.delta, r.two_sigma
                xs.append(i); ds.append(d); ts.append(t)
                ns.append(int(runs[runs.variant == v].n_features.iloc[0]))
                rows.append({"metric": label, "variant": v, "n_features": ns[-1], "delta_vs_baseline": d,
                             "two_sigma": t})
            ax.fill_between(xs, -np.array(ts), np.array(ts), color=GRID, alpha=0.8, lw=0, label="±2σ")
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.plot(xs, ds, color=KEEP_C, lw=2, marker="o", ms=7, mec="white", mew=1.5, label="Δ к бейзлайну")
            ax.set_xticks(xs); ax.set_xticklabels([f"{n}" for n in ns])
            ax.set_xlabel("число признаков")
            ax.set_title(f"Δ {label}", loc="left", fontsize=10.5, color=INK)
            ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        axes[0].legend(frameon=False, fontsize=8, loc="lower left")
        steps_txt = " → ".join(STEP_LABELS[s].replace("\n", " ") for s in steps)
        fig.suptitle("Кумулятивное удаление групп признаков: разность среднего по 5 сидам с бейзлайном (61)\n"
                     f"шаги: {steps_txt}", x=0.01, ha="left", fontsize=10.5, color=INK, y=1.06)
        fig.tight_layout()
        save(fig, "fig5_delta_by_step", pd.DataFrame(rows))

    # --- 6. возврат групп (если делался) ---
    plus = [v for v in variants if v.startswith("prune_plus_")]
    if plus:
        d = pd.concat([compare(runs, PRUNE, v) for v in plus])
        d = d[d.column == "precision_top_1pct"].sort_values("delta")
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.5))
        y = np.arange(len(d))
        ax.errorbar(d.delta, y, xerr=d.two_sigma, fmt="o", color=KEEP_C, ecolor=MUTED, capsize=4)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([v.replace("prune_plus_", "+ ") for v in d.variant])
        ax.set_xlabel("Δ precision@1 % к prune (33), ± 2σ")
        ax.set_title("Возврат удалённых групп по одной", loc="left", fontsize=11, color=INK)
        save(fig, "fig6_return_groups", d)


if __name__ == "__main__":
    main()
