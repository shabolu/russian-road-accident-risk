"""Сводка этапов 1–6a плана признаков: метрики по 5 сидам на каждом этапе.

    PYTHONPATH=src venv/bin/python scripts/plot_v2_stages_summary.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/analysis/features/summary"
STAGES = [
    ("Бейзлайн", "road_geo_v2_prune", "road_geo_baseline"),
    ("1\nчистка", "road_geo_v2_prune", "road_geo_v2_prune"),
    ("2\nперекрёстки", "road_geo_v2_junctions", "road_geo_v2_junctions"),
    ("3\nтеги OSM", "road_geo_v2_tags", "road_geo_v2_tags"),
    ("4\nгеометрия", "road_geo_v2_geometry", "road_geo_v2_geometry"),
    ("5\nСКДФ", "road_geo_v2_skdf", "road_geo_v2_skdf"),
    ("6a\nэкспозиция", "v2_eval_stage6a", "road_geo_v2_exposure_a"),  # прогоны модуля v2_eval, split = test
]
METRICS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
           ("precision_top_1pct", "precision@1 %"), ("recall_top_5pct", "recall@5 %")]


def main() -> None:
    rows = []
    for label, file_tag, variant in STAGES:
        if file_tag.startswith("v2_eval"):
            d = pd.read_csv(ROOT / f"reports/{file_tag}.csv")
            d = d[(d.variant == variant) & (d.split == "test")]
        else:
            d = pd.read_csv(ROOT / f"reports/{file_tag}_seed_runs.csv")
            d = d[d.variant == variant]
        assert len(d) == 5, (file_tag, variant, len(d))
        for m, _ in METRICS:
            rows.append({"stage": label.replace("\n", " "), "metric": m, "n_features": int(d.n_features.iloc[0]),
                         "mean": d[m].mean(), "sd": d[m].std(ddof=1), "values": ";".join(f"{v:.6f}" for v in d[m])})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "fig_stages_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    axes = axes.ravel()
    x = np.arange(len(STAGES))
    for ax, (m, name) in zip(axes, METRICS):
        sub = df[df.metric == m]
        for i, (label, file_tag, variant) in enumerate(STAGES):
            vals = np.array(sub.iloc[i]["values"].split(";"), dtype=float)
            ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), vals, s=14, color="#9ecae1", zorder=2)
        ax.errorbar(x, sub["mean"], yerr=sub["sd"], fmt="o-", color="black", capsize=4, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s[0]}\n({n})" for s, n in zip(STAGES, sub.n_features)], fontsize=10)
        ax.set_title(name, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Метрики на тесте 2023–2025 по этапам пересборки признаков (6a — набор road_geo_v2_exposure_a)\n(5 сидов: точки — сиды, "
                 "чёрное — среднее ± ст. откл.; в скобках — число признаков)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig_stages_metrics.png", dpi=300)
    print(df.pivot_table(index="stage", columns="metric", values="mean", sort=False).round(4))


if __name__ == "__main__":
    main()
