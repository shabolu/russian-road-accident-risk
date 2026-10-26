"""Прогон вариантов по схеме этапа 6.0 (отбор на валидации) — см. src/road_accident_risk/v2_eval.py.

Сплиты: valid (fit < 2022, оценка 2022), test (train < 2023, оценка 2023–2025, как на этапах 1–5),
both = valid + test, valid2021 (fit < 2021, оценка 2021; второй фолд скользящей схемы),
folds = valid2021 + valid (при --base дополнительно среднее по фолдам -> reports/v2_eval_<tag>_folds_delta.csv).
Прогоны дописываются в reports/v2_eval_<tag>.csv (строки с тем же split/variant/seed заменяются);
при --base — разности к базе и вердикты (2σ и парно) в reports/v2_eval_<tag>_delta.csv.

Запуск:
    PYTHONPATH=src venv/bin/python scripts/run_v2_eval.py --variants road_geo_v2_skdf,skdf_B3b_hierarchy_value \
        --split valid --base road_geo_v2_skdf --tag stage6_0
    PYTHONPATH=src venv/bin/python scripts/run_v2_eval.py --variants road_geo_baseline --split test --seeds 42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.v2_eval import (  # noqa: E402
    REPORTS, SEEDS, compare, compare_folds, load_panel, run_variant, upsert_runs,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="варианты через запятую")
    ap.add_argument("--split", default="valid",
                    help="valid | test | valid2021 | both (valid+test) | folds (valid2021+valid) | all | список")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--tag", default="runs", help="суффикс файла reports/v2_eval_<tag>.csv")
    ap.add_argument("--base", default=None, help="база для разностей (должна быть прогнана в том же сплите)")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    alias = {"both": ["valid", "test"], "folds": ["valid2021", "valid"], "all": ["valid2021", "valid", "test"]}
    splits = alias.get(args.split) or [s.strip() for s in args.split.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = REPORTS / f"v2_eval_{args.tag}.csv"

    t0 = time.time()
    panel = load_panel()
    print(f"панель {panel.shape} [{time.time() - t0:.0f} с]", flush=True)
    runs = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for split in splits:
        for v in variants:
            t1 = time.time()
            runs = upsert_runs(out, run_variant(panel, v, split, seeds))
            print(f"[{split}] {v}: готово за {time.time() - t1:.0f} с -> {out.name}", flush=True)

    if args.base:
        deltas = [compare(runs, args.base, v, split=s) for s in splits for v in variants if v != args.base]
        if deltas:
            d = pd.concat(deltas, ignore_index=True)
            dpath = REPORTS / f"v2_eval_{args.tag}_delta.csv"
            if dpath.exists():
                old = pd.read_csv(dpath)
                key = ["split", "base", "variant", "column"]
                old = old[~old.set_index(key).index.isin(d.set_index(key).index)]
                d = pd.concat([old, d], ignore_index=True)
            d.to_csv(dpath, index=False)
            sel = d[d.column.isin(["roc_auc", "pr_auc", "precision_top_1pct", "recall_top_10pct"])]
            if {"valid2021", "valid"} <= set(splits):  # скользящая схема: среднее по двум фолдам
                fo = pd.concat([compare_folds(runs, args.base, v) for v in variants if v != args.base])
                fo.to_csv(REPORTS / f"v2_eval_{args.tag}_folds_delta.csv", index=False)
                sel = pd.concat([sel, fo[fo.column.isin(sel.column.unique())]], ignore_index=True)
            with pd.option_context("display.width", 220, "display.max_rows", 400):
                print(sel[["split", "variant", "metric", "delta", "two_sigma", "two_sd_paired", "verdict",
                           "verdict_paired"]].to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
    print(f"всего {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
