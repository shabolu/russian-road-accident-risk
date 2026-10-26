"""Этап 0 плана docs/analysis/features/PLAN.md: отчёт по Spearman-парам внутри набора.

Признаки дорог статичны во времени, поэтому корреляции считаются по ячейкам:
одна строка на ячейку H3 среди ячеек с дорогами (`road_length_total_m > 0`).
Одиночный ROC-AUC каждого признака — на тестовых строках панели (month >= TRAIN_END),
без клиппинга (AUC инвариантен к монотонным преобразованиям, а клиппинг p99
только сливает хвост в одно значение).

Выход (по умолчанию для road_geo_baseline):
    reports/<prefix>_spearman_pairs.csv   — все пары |rho| >= 0.85, по убыванию |rho|
    reports/<prefix>_spearman_matrix.csv  — полная матрица Spearman
    reports/<prefix>_feature_stats.csv    — доля нулей/пропусков, одиночный AUC, важность

Запуск:
    venv/bin/python scripts/feature_dedup_report.py
    venv/bin/python scripts/feature_dedup_report.py --feature-set road_geo_v2_prune --prefix road_geo_v2_prune
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.catboost_baseline import (  # noqa: E402
    FEATURE_SET_ROAD_GEO_BASELINE, attach_road_grade_features, build_features,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END  # noqa: E402
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PROJECT_ROOT  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"


def load_panel(feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)
    panel = attach_road_grade_features(panel)
    panel = panel[panel["road_length_total_m"] > 0].copy()
    features = build_features(panel, feature_set=feature_set)
    panel["month"] = pd.to_datetime(panel["month"])
    return panel, features


def cell_table(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cells = panel.drop_duplicates("h3")[["h3", *features]].set_index("h3")
    # проверка статичности: ни один признак не меняется внутри ячейки
    nun = panel.groupby("h3")[features].nunique(dropna=False).max()
    dynamic = nun[nun > 1]
    if len(dynamic):
        print(f"ВНИМАНИЕ: признаки меняются во времени внутри ячейки: {list(dynamic.index)}")
    return cells


def spearman_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cols = list(corr.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho = corr.loc[a, b]
            if pd.notna(rho) and abs(rho) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "spearman": rho, "abs_spearman": abs(rho)})
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def feature_stats(cells: pd.DataFrame, panel: pd.DataFrame, features: list[str],
                  importance_path: Path | None) -> pd.DataFrame:
    test = panel[panel["month"] >= TRAIN_END]
    y = test[TARGET].astype(int).to_numpy()
    rows = []
    for f in features:
        x = test[f]
        mask = x.notna().to_numpy()
        auc = roc_auc_score(y[mask], x.to_numpy()[mask]) if mask.any() else np.nan
        col = cells[f]
        rows.append({
            "feature": f,
            "cells": int(col.notna().sum()),
            "nan_share_cells": float(col.isna().mean()),
            "zero_share_cells": float((col == 0).mean()),
            "n_unique_cells": int(col.nunique()),
            "median_cells": float(col.median()),
            "p99_cells": float(col.quantile(0.99)),
            "max_cells": float(col.max()),
            "single_auc_test": float(auc),
            "single_auc_abs": float(max(auc, 1 - auc)) if pd.notna(auc) else np.nan,
        })
    out = pd.DataFrame(rows)
    if importance_path is not None and importance_path.exists():
        imp = pd.read_csv(importance_path).rename(columns={"importance": "importance_seed42"})
        imp = imp[["feature", "importance_seed42"]]
        imp["importance_rank"] = imp["importance_seed42"].rank(ascending=False, method="first").astype(int)
        out = out.merge(imp, on="feature", how="left")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-set", default=FEATURE_SET_ROAD_GEO_BASELINE)
    ap.add_argument("--prefix", default="road_geo_baseline")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--importance", type=Path,
                    default=REPORTS / "catboost_feature_importance_road_geo_baseline.csv")
    args = ap.parse_args()

    panel, features = load_panel(args.feature_set)
    cells = cell_table(panel, features)
    print(f"набор {args.feature_set}: {len(features)} признаков; ячеек с дорогами: {len(cells)}; "
          f"тестовых строк: {(panel['month'] >= TRAIN_END).sum()}")

    corr = cells.rank().corr(method="pearson")  # Spearman = Pearson по рангам; NaN попарно
    corr.to_csv(REPORTS / f"{args.prefix}_spearman_matrix.csv")
    pairs = spearman_pairs(corr, args.threshold)
    pairs.to_csv(REPORTS / f"{args.prefix}_spearman_pairs.csv", index=False)
    stats = feature_stats(cells, panel, features, args.importance)
    stats.to_csv(REPORTS / f"{args.prefix}_feature_stats.csv", index=False)

    print(f"пар с |rho| >= {args.threshold}: {len(pairs)}")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(pairs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
