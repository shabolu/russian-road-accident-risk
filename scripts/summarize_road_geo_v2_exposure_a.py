"""Сводные таблицы этапа 6a: блоки (folds + test), финальная проверка, воспроизводимость.

Вход: reports/v2_eval_stage6a.csv, reports/v2_eval_stage6a_repro.csv, reports/road_geo_v2_skdf_seed_runs.csv.
Выход: docs/analysis/features/stage6_exposure/blocks_a/tables/{blocks_summary,final_test,repro}.csv

    PYTHONPATH=src venv/bin/python scripts/summarize_road_geo_v2_exposure_a.py
"""
from __future__ import annotations

import pandas as pd

from road_accident_risk.paths import PROJECT_ROOT
from road_accident_risk.v2_eval import METRICS, compare, compare_folds

R = PROJECT_ROOT / "reports"
TAB = PROJECT_ROOT / "docs/analysis/features/stage6_exposure/blocks_a/tables"
A = "road_geo_v2_skdf"
EX = "road_geo_v2_exposure_a"
KEY = ["roc_auc", "pr_auc", "precision_top_1pct", "recall_top_10pct"]
PAIRS = [(A, v) for v in ["e1_center_two", "e1_center_poi", "e2_cell", "e2_r300", "e2_r500", "e3_build", "e4_old",
                          "e4_new", "e4_routes", EX, "g_E3_E1", "g_E3_E2", "g_E3_E4", "eart", "elevels", "emed"]] + \
        [("e4_old", "e4_routes"), (EX, "g_E3_E1"), (EX, "g_E3_E2"), (EX, "g_E3_E4"), (EX, "eart"), (EX, "elevels"),
         (EX, "emed"), ("road_geo_v2_tags", A), ("road_geo_v2_tags", EX), ("road_geo_v2_tags", "eart")]


def main() -> None:
    runs = pd.read_csv(R / "v2_eval_stage6a.csv")
    rows = []
    for base, v in PAIRS:
        has_folds = not runs[(runs.split == "valid") & (runs.variant == v)].empty and \
            not runs[(runs.split == "valid") & (runs.variant == base)].empty
        fo = compare_folds(runs, base, v).set_index("column") if has_folds else None
        te = compare(runs, base, v, split="test").set_index("column")
        for c in KEY:
            r = {"base": base, "variant": v, "metric": c}
            if fo is not None:
                r.update({"folds_delta": fo.loc[c, "delta"], "folds_2s": fo.loc[c, "two_sigma"],
                          "folds_verdict": fo.loc[c, "verdict"], "d2021": fo.loc[c, "delta_valid2021"],
                          "d2022": fo.loc[c, "delta_valid"], "same_sign": fo.loc[c, "same_sign_all_folds"]})
            r.update({"test_delta": te.loc[c, "delta"], "test_2s": te.loc[c, "two_sigma"], "test_verdict": te.loc[c, "verdict"],
                      "test_2sd_paired": te.loc[c, "two_sd_paired"], "test_verdict_paired": te.loc[c, "verdict_paired"],
                      "test_mean_variant": te.loc[c, "mean_variant"], "test_mean_base": te.loc[c, "mean_base"]})
            rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(TAB / "blocks_summary.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_rows", 500):
        print(s.round(5).to_string(index=False))

    # финал: средние ± sd на тесте
    t = runs[runs.split == "test"]
    fin = t[t.variant.isin([A, EX, "road_geo_v2_tags"])].groupby("variant")[[c for c, _ in METRICS]].agg(["mean", "std"])
    fin.to_csv(TAB / "final_test.csv")
    print(fin.T.round(5).to_string())

    # воспроизводимость: A на тесте против reports/road_geo_v2_skdf_seed_runs.csv
    old = pd.read_csv(R / "road_geo_v2_skdf_seed_runs.csv")
    old = old[old.variant == A].set_index("seed")
    new = t[t.variant == A].set_index("seed")
    cols = [c for c, _ in METRICS]
    diff = (new.loc[old.index, cols] - old[cols]).abs().max().max()
    rep = pd.read_csv(R / "v2_eval_stage6a_repro.csv")
    rep = rep[["variant", "seed", "roc_auc", "pr_auc", "precision_top_1pct"]]
    rep["A_test_5seeds_max_abs_diff_vs_seed_runs"] = diff
    rep.to_csv(TAB / "repro.csv", index=False)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
