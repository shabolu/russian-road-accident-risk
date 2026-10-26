"""Этап 7 шаг 1: таблица сплитов и картинки дымового прогона -> docs/analysis/features/stage7_count12/infra/.

fig1_year_panel_counts.png — ДТП и доля ячеек с ≥ 1 ДТП по годам с разметкой сплитов;
fig2_gain_curves_smoke.png — кривые «доля ячеек → доля ДТП», тест 2023–2025 (сумма), seed 42 + ориентиры.

Запуск (после scripts/run_v2_eval_year.py с --tag stage7_infra_smoke):
    PYTHONPATH=src venv/bin/python scripts/plot_stage7_infra.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk import v2_eval_year as vy  # noqa: E402
from road_accident_risk.paths import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "docs/analysis/features/stage7_count12/infra"


def main() -> None:
    p = pd.read_parquet(vy.YEAR_PANEL_PARQUET)
    ss = vy.split_summary_year(p)
    ss.to_csv(OUT / "split_summary.csv", index=False)
    by = p.groupby("year").agg(accidents=("count_year", "sum"), event_rate=("had_accident_year", "mean"),
                               mean=("count_year", "mean"), var=("count_year", "var"),
                               zero_share=("count_year", lambda s: (s == 0).mean()), max=("count_year", "max"),
                               fatal=("count_fatal_year", "sum"), severity=("severity_weight_year", "sum"))
    by.to_csv(OUT / "year_panel_by_year.csv")
    print(ss.to_string()); print(by.to_string())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    col = ["#9aa5b1" if y <= 2020 else "#e3a33b" if y in (2021, 2022) else "#c0392b" for y in by.index]
    ax.bar(by.index, by.accidents, color=col)
    for y, v in by.accidents.items():
        ax.text(y, v + 15, f"{v}", ha="center", fontsize=8)
    ax.set_ylabel("ДТП с пострадавшими в ячейках с дорогами"); ax.set_xticks(by.index); ax.set_ylim(0, 2300)
    ax2 = ax.twinx(); ax2.plot(by.index, by.event_rate * 100, "k.-"); ax2.set_ylabel("ячеек с ≥ 1 ДТП за год, %")
    ax2.set_ylim(0, 20)
    ax.set_title("Годовая панель: 4 296 ячеек × 2015–2025\nсерый — только обучение, жёлтый — валидационные "
                 "годы (2021, 2022), красный — тест", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig1_year_panel_counts.png", dpi=300); plt.close(fig)

    preds = pd.read_parquet(vy.REPORTS / "v2_eval_year_stage7_infra_smoke_preds.parquet")
    labels = {"road_geo_v2_skdf|count:Poisson": "CatBoost Poisson (32)",
              "road_geo_v2_skdf|binary:Logloss:Balanced": "CatBoost бинарный, Balanced (32)",
              "road_geo_v2_skdf|monthly:mean": "месячная модель → год (32)",
              "baseline:accidents_prev_3y|baseline": "ориентир: ДТП за 3 прошлых года",
              "baseline:accidents_prev_year|baseline": "ориентир: ДТП за прошлый год",
              "baseline:major_road_length_m|baseline": "ориентир: длина магистралей"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    rows = []
    for rid, lab in labels.items():
        if rid not in set(preds.run_id):
            continue
        a = vy.period_frame(preds, rid, "test", "sum")
        g = vy.gain_curve(a.count_year.to_numpy(), a.pred.to_numpy(), n_points=1001)
        for ax in axes:
            ax.plot(g.frac_cells * 100, g.frac_accidents * 100, label=lab, lw=1.3)
        rows.append(g.assign(run_id=rid))
    a = vy.period_frame(preds, "road_geo_v2_skdf|count:Poisson", "test", "sum")
    g = vy.gain_curve(a.count_year.to_numpy(), a.count_year.to_numpy(), n_points=1001)
    for ax in axes:
        ax.plot(g.frac_cells * 100, g.frac_accidents * 100, "k--", lw=1, label="идеал (факт)")
        ax.plot([0, 100], [0, 100], ":", color="grey", lw=1, label="случайный порядок")
        ax.set_xlabel("доля ячеек, отобранных по прогнозу, %"); ax.set_ylabel("доля ДТП 2023–2025, %")
        ax.grid(alpha=.3)
    pd.concat([*rows, g.assign(run_id="oracle")]).to_csv(OUT / "fig2_gain_curves_smoke.csv", index=False)
    axes[1].set_xlim(0, 10); axes[1].set_ylim(0, 75); axes[1].set_title("верхние 10 % ячеек")
    axes[0].set_title("вся кривая"); axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Дымовой прогон (seed 42): тест 2023–2025, сумма трёх лет, 4 296 ячеек", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig2_gain_curves_smoke.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
