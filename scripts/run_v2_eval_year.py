"""Этап 7: прогон вариантов в годовой постановке — см. src/road_accident_risk/v2_eval_year.py.

Сплиты: year_folds = y2021 (обучение 2015–2020) + y2022 (обучение 2015–2021); year_test = test (обучение
2015–2022, оценка 2023, 2024, 2025, их среднее 'mean' и сумма 'sum'); both = все три; или список через запятую.
Прогоны дописываются в reports/v2_eval_year_<tag>.csv (ключ split, run_id, seed, eval_period; run_id =
<вариант>|<модель>, модель = binary:Logloss:Balanced | binary:Logloss:None | count:Poisson | count:Tweedie:…);
прогнозы — в reports/v2_eval_year_<tag>_preds.parquet; при --base — разности в reports/v2_eval_year_<tag>_delta.csv
(по каждому сплиту и периоду + среднее по фолдам, если прогнаны оба фолда).

Запуск:
    PYTHONPATH=src venv/bin/python scripts/run_v2_eval_year.py --variants road_geo_v2_skdf --mode count \
        --loss Poisson --split both --seeds 42 --tag stage7_smoke
    PYTHONPATH=src venv/bin/python scripts/run_v2_eval_year.py --variants road_geo_v2_skdf --mode binary \
        --class-weights Balanced --split year_folds --tag stage7_binary \
        --base 'road_geo_v2_skdf|binary:Logloss:None'
    # ориентиры без модели (seed 0) и месячная модель как годовой скор:
    ... --variants road_geo_v2_skdf --baselines accidents_prev_year,accidents_prev_3y,major_road_length_m
    ... --variants road_geo_v2_skdf --monthly mean
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk import v2_eval_year as vy  # noqa: E402


def upsert_parquet(path: Path, new: pd.DataFrame, key: list[str]) -> None:
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="", help="варианты через запятую (наборы v2_eval, '+lags' — с лагами)")
    ap.add_argument("--mode", default="count", choices=["binary", "count"])
    ap.add_argument("--loss", default=None, help="count: Poisson | Tweedie:variance_power=1.5; binary: Logloss")
    ap.add_argument("--class-weights", default="Balanced", help="binary: Balanced | None")
    ap.add_argument("--split", default="year_folds", help="year_folds | year_test | both | y2021,y2022,test")
    ap.add_argument("--seeds", default=",".join(map(str, vy.SEEDS)))
    ap.add_argument("--params", default=None, help='JSON-переопределение параметров CatBoost, напр. {"depth": 8}')
    ap.add_argument("--baselines", default="", help="колонки панели — ориентиры без модели (через запятую)")
    ap.add_argument("--monthly", default="", help="mean и/или max: месячная модель вариантов как годовой скор")
    ap.add_argument("--tag", default="runs")
    ap.add_argument("--base", default=None, help="run_id базы (variant|model) или имя ориентира 'baseline:<кол>'")
    ap.add_argument("--no-preds", action="store_true", help="не сохранять прогнозы")
    ap.add_argument("--bootstrap", action="store_true",
                    help="бутстреп по ячейкам (B=1000) доли накрытых ДТП на тесте ('sum' и по годам) для прогонов "
                         "этого запуска; при --base — ещё парная разность к базе -> _bootstrap.csv")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    monthly = [a.strip() for a in args.monthly.split(",") if a.strip()]
    splits = vy.resolve_splits(args.split)
    seeds = [int(s) for s in args.seeds.split(",")]
    params = json.loads(args.params) if args.params else None
    cw = None if args.class_weights in ("None", "none", "") else args.class_weights
    out = vy.REPORTS / f"v2_eval_year_{args.tag}.csv"
    pout = vy.REPORTS / f"v2_eval_year_{args.tag}_preds.parquet"

    t0 = time.time()
    panel = vy.load_year_panel()
    print(f"годовая панель {panel.shape} [{time.time() - t0:.0f} с]", flush=True)
    runs = pd.read_csv(out, dtype={"eval_period": str}) if out.exists() else pd.DataFrame()
    timing = []
    for split in splits:
        for v in variants:
            t1 = time.time()
            r, pr = vy.run_year_variant(panel, v, split, mode=args.mode, loss=args.loss, class_weights=cw,
                                        seeds=seeds, params=params, return_predictions=True)
            runs = vy.upsert_year_runs(out, r)
            if not args.no_preds:
                upsert_parquet(pout, pr, ["split", "run_id", "seed", "h3", "year"])
            timing.append((split, r.run_id.iloc[0], time.time() - t1))
            print(f"[{split}] {r.run_id.iloc[0]}: {len(seeds)} сид(ов) за {time.time() - t1:.0f} с -> {out.name}",
                  flush=True)
            if monthly:
                t1 = time.time()
                sc = vy.monthly_model_year_scores(v, split, seeds)
                for agg in monthly:
                    r, pr = vy.run_monthly_as_year(panel, v, split, seeds, agg, scores=sc, return_predictions=True)
                    runs = vy.upsert_year_runs(out, r)
                    if not args.no_preds:
                        upsert_parquet(pout, pr, ["split", "run_id", "seed", "h3", "year"])
                timing.append((split, f"{v}|monthly", time.time() - t1))
        for b in baselines:
            r, pr = vy.run_baseline(panel, b, split, return_predictions=True)
            runs = vy.upsert_year_runs(out, r)
            if not args.no_preds:
                upsert_parquet(pout, pr, ["split", "run_id", "seed", "h3", "year"])

    show = ["split", "run_id", "seed", "eval_period", "coverage_top_1pct", "coverage_top_5pct", "coverage_top_10pct",
            "spearman", "roc_auc", "pr_auc", "poisson_deviance", "calib_ratio"]
    cur = runs[runs.split.isin(splits)]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(cur[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.base:
        ids = sorted(set(cur.run_id) - {args.base})
        base = args.base if "|" in args.base else f"{args.base}|baseline"
        deltas = []
        for s in splits:
            periods = [vy.default_period(s)] if len(vy.YEAR_SPLITS[s].eval_years) == 1 else \
                [*map(str, vy.YEAR_SPLITS[s].eval_years), "mean", "sum"]
            for v in ids:
                for per in periods:
                    deltas.append(vy.compare_year(runs, base, v, split=s, period=per))
        if {"y2021", "y2022"} <= set(splits):
            for v in ids:
                deltas.append(vy.compare_folds_year(runs, base, v).assign(eval_period="folds_mean"))
        if deltas:
            d = pd.concat(deltas, ignore_index=True)
            dpath = vy.REPORTS / f"v2_eval_year_{args.tag}_delta.csv"
            if dpath.exists():
                old = pd.read_csv(dpath, dtype={"eval_period": str})
                key = ["split", "eval_period", "base", "variant", "column"]
                old = old[~old.set_index(key).index.isin(d.set_index(key).index)]
                d = pd.concat([old, d], ignore_index=True)
            d.to_csv(dpath, index=False)
            print(f"разности -> {dpath.name}")
    if args.bootstrap and "test" in splits and pout.exists():
        preds = pd.read_parquet(pout)
        ids = sorted(set(cur[cur.split == "test"].run_id))
        base = None if not args.base else (args.base if "|" in args.base else f"{args.base}|baseline")
        bs = []
        for rid in ids:
            for per in ["sum", *map(str, vy.YEAR_SPLITS["test"].eval_years)]:
                bs.append(vy.bootstrap_from_predictions(preds, rid, "test", per))
                if base and rid != base and base in set(preds.run_id):
                    bs.append(vy.bootstrap_from_predictions(preds, rid, "test", per, base_run_id=base))
        bpath = vy.REPORTS / f"v2_eval_year_{args.tag}_bootstrap.csv"
        b = pd.concat(bs, ignore_index=True)
        if bpath.exists():
            old = pd.read_csv(bpath, dtype={"eval_period": str})
            key = ["run_id", "base_run_id", "eval_period", "target", "frac"]
            old = old[~old.set_index(key).index.isin(b.set_index(key).index)]
            b = pd.concat([old, b], ignore_index=True)
        b.to_csv(bpath, index=False)
        print(f"бутстреп -> {bpath.name}")
    for s, rid, sec in timing:
        print(f"время [{s}] {rid}: {sec:.0f} с ({sec / len(seeds):.0f} с на сид)")
    print(f"всего {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
