"""Этап 7, шаг 3: прогоны — «структура + история», кросс-фит структурной модели, цепочка наборов (7.6), месячные фолды.

Подкоманды (все пишут в reports/, с заменой по ключу; можно перезапускать после сбоя):
    history   Tweedie 1.3 на road_geo_v2_skdf + лаги (2 лага панели; 3 лага с «все прошлые годы»), both, 5 сидов
              -> reports/v2_eval_year_stage7C_history.csv, _preds.parquet
    crossfit  Tweedie 1.3 на road_geo_v2_skdf, прогноз ячейки — от модели, обученной без её группы ячеек
              (5 групп по родителю H3 res 7), both, 5 сидов -> те же файлы stage7C_history
    chain     Tweedie 1.3 на наборах цепочки этапов 0–6, диагностиках и вариантах этапов 5–6 (пары с месячными
              фолдами и тестом), both, 5 сидов -> reports/v2_eval_year_stage7C_chain.csv, _preds.parquet
    month     месячные фолды (valid2021 + valid) для наборов цепочки без месячных фолдов + тест для g_E1_E2
              -> reports/v2_eval_stage7C_month.csv
    repro     месячные эталоны seed 42 (test) -> reports/v2_eval_stage7C_month_check.csv

Утечка. История для года Y — только из лет < Y и только из обучающих лет сплита. На тесте (обучение 2015–2022)
лаги строк 2024 и 2025 гг. заменяются лагами 2023 г. (окна, кончающиеся 2022 г.): прогноз на все три года
делается в конце 2022 г. Лаг «все прошлые годы» (среднее в год за 2015…Y−1) считается здесь; список
признаков набора строится до его добавления, поэтому в «все колонки минус исключения» он не попадает.

    PYTHONPATH=src venv/bin/python scripts/stage7C_run.py history crossfit chain
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road_accident_risk import catboost_baseline as cb  # noqa: E402
from road_accident_risk import exposure_v6 as ev  # noqa: E402
from road_accident_risk import v2_eval as ve  # noqa: E402
from road_accident_risk import v2_eval_year as vy  # noqa: E402

LOSS = "Tweedie:variance_power=1.3"
MODEL = vy.model_label("count", LOSS)
SKDF = cb.FEATURE_SET_ROAD_GEO_V2_SKDF
LAG_ALL = "accidents_prev_all_rate"
REPORTS = vy.REPORTS
HIST_CSV = REPORTS / "v2_eval_year_stage7C_history.csv"
CHAIN_CSV = REPORTS / "v2_eval_year_stage7C_chain.csv"
MONTH_CSV = REPORTS / "v2_eval_stage7C_month.csv"

CHAIN = [cb.FEATURE_SET_ROAD_GEO_BASELINE, cb.FEATURE_SET_ROAD_GEO_V2_PRUNE, cb.FEATURE_SET_ROAD_GEO_V2_JUNCTIONS,
         cb.FEATURE_SET_ROAD_GEO_V2_TAGS, cb.FEATURE_SET_ROAD_GEO_V2_GEOMETRY, SKDF, cb.FEATURE_SET_ROAD_GEO_V2_EXPOSURE_A]
DIAG = ["geometry_plus_old", "g_E1_E2"]
STAGE5 = ["skdf_B3b_hierarchy_value", "skdf_B5_speed_axle", "skdf_C_all_passed", "skdf_C_width_hier_speed_axle"]
STAGE6A = ["e1_center_two", "e1_center_poi", "e2_cell", "e2_r300", "e2_r500", "e3_build", "e4_old", "e4_new",
           "g_E3_E1", "g_E3_E2", "g_E3_E4", "e4_routes", "eart", "elevels", "emed"]


# ---------------------------------------------------------------- варианты этапов 4 и 6

def _load_script(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def register_variants() -> None:
    """geometry_plus_old (этап 4), варианты 6a (CHOSEN как в README 6a) и диагностика g_E1_E2 = skdf + E1(б) + E2(ячейка)."""
    g4 = _load_script("_stage4", "run_road_geo_v2_geometry_experiments.py")
    ve.register_variant("geometry_plus_old", lambda cols: g4.variant_features("geometry_plus_old", cols))
    e6 = _load_script("_stage6a", "run_road_geo_v2_exposure_a_experiments.py")
    e6.CHOSEN.update({"E1": "e1_center_poi", "E2": "e2_cell", "E3": "e3_build", "E4": "e4_old", "E4_base": "e4_old"})
    e6._GREEDY_NAMES.extend(["g_E3_E1", "g_E3_E2", "g_E3_E4", "g_E1_E2"])
    e6.register_all()


def fill_median_year(panel: pd.DataFrame, split: str, feats: list[str]) -> pd.DataFrame:
    """emed: NaN признаков этапа 6 -> медиана по обучающим годам сплита (признаки статичны: = медиана по ячейкам)."""
    new = [f for f in feats if f in ev.EXPOSURE_V6_FEATURES and panel[f].isna().any()]
    tr, _ = vy.split_frames_year(panel, split)
    out = panel.copy()
    for f in new:
        out[f] = out[f].fillna(tr[f].median())
    print(f"[{split}] emed: медианой заполнены {new}", flush=True)
    return out


# ---------------------------------------------------------------- история без утечки

def add_history(panel: pd.DataFrame) -> pd.DataFrame:
    """+ accidents_prev_all_rate (среднее ДТП в год за 2015…Y−1); лаги строк 2024–2025 := лаги 2023 (заморозка теста)."""
    p = panel.sort_values(["h3", "year"]).reset_index(drop=True)
    g = p.groupby("h3", sort=False)[vy.TARGET_COUNT]
    past = g.cumsum() - p[vy.TARGET_COUNT]
    n_past = (p["year"] - vy.YEARS[0]).astype(float)
    p[LAG_ALL] = np.where(n_past > 0, past / n_past.replace(0, np.nan), np.nan)
    lag_cols = [*vy.LAG_COLS, LAG_ALL]
    first_test = min(vy.YEAR_SPLITS["test"].eval_years)
    ref = p.loc[p["year"] == first_test, ["h3", *lag_cols]].set_index("h3")
    late = p["year"] > first_test
    for c in lag_cols:
        p.loc[late, c] = p.loc[late, "h3"].map(ref[c]).to_numpy()
    return p


def check_history(p: pd.DataFrame) -> None:
    """Самопроверка: лаги теста не зависят от фактов ≥ 2023; лаги фолдов — от фактов ≥ года фолда."""
    q = p.copy()
    q.loc[q["year"] >= 2023, vy.TARGET_COUNT] = 999
    # пересчитать лаги панели из «испорченного» факта
    q = vy.add_year_lags(q.drop(columns=vy.LAG_COLS))
    q = add_history(q)
    for c in [*vy.LAG_COLS, LAG_ALL]:
        a = p.loc[p.year >= 2015, c].to_numpy(float); b = q.loc[q.year >= 2015, c].to_numpy(float)
        assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1)), f"утечка в {c}"
    t = p[p.year == 2023].set_index("h3")
    raw = p[p.year <= 2022].groupby("h3")[vy.TARGET_COUNT]
    assert np.allclose(t[LAG_ALL], raw.mean().reindex(t.index)), "prev_all 2023 != среднее 2015–2022"
    assert np.allclose(t["accidents_prev_3y"], p[p.year.between(2020, 2022)].groupby("h3")[vy.TARGET_COUNT].sum()
                       .reindex(t.index)), "prev_3y 2023 != Σ 2020–2022"
    for yr in (2024, 2025):
        u = p[p.year == yr].set_index("h3")
        assert np.allclose(np.nan_to_num(u[LAG_ALL]), np.nan_to_num(t[LAG_ALL].reindex(u.index))), "заморозка"
    print("OK история: лаги теста не зависят от фактов 2023+, prev_all/prev_3y на 2023 = окна 2015–2022 / 2020–2022",
          flush=True)


def cmd_history(seeds, splits) -> None:
    panel = vy.load_year_panel()
    base_feats = vy.year_variant_features(SKDF, panel)       # до добавления новой колонки
    p = add_history(panel)
    check_history(p)
    variants = {f"{SKDF}+lags": [*base_feats, *vy.LAG_COLS], f"{SKDF}+lags_all": [*base_feats, *vy.LAG_COLS, LAG_ALL]}
    allp = []
    for split in splits:
        for name, feats in variants.items():
            t0 = time.time()
            r, pr = vy.run_year_variant(p, name, split, mode="count", loss=LOSS, seeds=seeds, features=feats,
                                        return_predictions=True)
            vy.upsert_year_runs(HIST_CSV, r); allp.append(pr)
            print(f"[{split}] {name}: {time.time() - t0:.0f} с", flush=True)
    _upsert_preds(HIST_CSV, pd.concat(allp, ignore_index=True))


def _upsert_preds(csv: Path, new: pd.DataFrame) -> None:
    path = csv.with_name(csv.stem + "_preds.parquet")
    if path.exists():
        old = pd.read_parquet(path)
        key = set(map(tuple, new[["split", "run_id"]].drop_duplicates().to_numpy()))
        old = old[[(s, r) not in key for s, r in zip(old["split"], old["run_id"])]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(path, index=False)


# ---------------------------------------------------------------- кросс-фит структурной модели

def cell_groups(cells: np.ndarray, n_groups: int = 5, res: int = 7, seed: int = 20261019) -> pd.Series:
    parents = pd.Series([h3.cell_to_parent(c, res) for c in cells], index=cells)
    up = np.sort(parents.unique())
    rng = np.random.default_rng(seed)
    gmap = dict(zip(up, rng.permutation(len(up)) % n_groups))
    return parents.map(gmap)


def cmd_crossfit(seeds, splits) -> None:
    panel = vy.load_year_panel()
    feats = vy.year_variant_features(SKDF, panel)
    grp = cell_groups(np.sort(panel["h3"].unique()))
    print("группы ячеек (родитель res 7):", grp.value_counts().sort_index().to_dict(), flush=True)
    run_id = f"{SKDF}|{MODEL}|crossfit"
    rows, preds = [], []
    for split in splits:
        tr0, ev0 = vy.split_frames_year(panel, split)
        caps = cb.fit_p99_clip_caps(tr0, feats)
        tr = cb.apply_p99_clipping(tr0, caps); evf = cb.apply_p99_clipping(ev0, caps)
        gtr = tr["h3"].map(grp).to_numpy(); gev = evf["h3"].map(grp).to_numpy()
        for seed in seeds:
            t0 = time.time()
            s = np.full(len(evf), np.nan)
            for g in range(5):
                m = vy.make_model("count", LOSS, None, seed)
                m.fit(Pool(tr.loc[gtr != g, feats], tr.loc[gtr != g, vy.TARGET_COUNT].astype(int), feature_names=feats))
                s[gev == g] = m.predict(evf.loc[gev == g, feats], prediction_type="Exponent")
            assert not np.isnan(s).any()
            d = evf[["h3", "year", vy.TARGET_COUNT, vy.SEVERITY]].assign(pred=s)
            for _, r in vy.evaluate_year_predictions(d, "pred", is_count=True).iterrows():
                rows.append({"split": split, "variant": f"{SKDF}|crossfit", "model": MODEL, "run_id": run_id,
                             "seed": seed, "n_features": len(feats), "n_train": len(tr), **r.to_dict(),
                             "fit_seconds": round(time.time() - t0, 1)})
            preds.append(d.assign(split=split, run_id=run_id, seed=seed))
            last = rows[-1]
            print(f"[{split}] crossfit seed={seed} [{last['eval_period']}]: cov@10={last['coverage_top_10pct']:.4f} "
                  f"rho={last['spearman']:.4f} [{time.time() - t0:.0f} с]", flush=True)
    vy.upsert_year_runs(HIST_CSV, pd.DataFrame(rows))
    _upsert_preds(HIST_CSV, pd.concat(preds, ignore_index=True))


# ---------------------------------------------------------------- цепочка наборов (7.6)

def cmd_chain(seeds, splits, variants: list[str] | None = None) -> None:
    register_variants()
    panel = vy.load_year_panel()
    variants = variants or [*CHAIN, *DIAG, *STAGE5, *STAGE6A]
    for v in variants:
        feats = vy.year_variant_features(v, panel)
        assert not set(feats) & set(vy.YEAR_PANEL_COLUMNS), f"{v}: таргет/лаги в признаках"
        allp = []
        for split in splits:
            t0 = time.time()
            p = fill_median_year(panel, split, feats) if v == "emed" else panel
            r, pr = vy.run_year_variant(p, v, split, mode="count", loss=LOSS, seeds=seeds, features=feats,
                                        return_predictions=True, verbose=False)
            vy.upsert_year_runs(CHAIN_CSV, r); allp.append(pr)
            k = r[r.eval_period == vy.default_period(split) if split != "test" else r.eval_period == "sum"]
            print(f"[{split}] {v} ({len(feats)}): cov@5={k.coverage_top_5pct.mean():.4f} "
                  f"cov@10={k.coverage_top_10pct.mean():.4f} rho={k.spearman.mean():.4f} [{time.time() - t0:.0f} с]",
                  flush=True)
        _upsert_preds(CHAIN_CSV, pd.concat(allp, ignore_index=True))


# ---------------------------------------------------------------- месячная схема

def cmd_month(seeds) -> None:
    register_variants()
    panel = ve.load_panel(attach=[cb.attach_exposure_v6_features])
    jobs = [(v, s) for s in ("valid2021", "valid") for v in
            (cb.FEATURE_SET_ROAD_GEO_BASELINE, cb.FEATURE_SET_ROAD_GEO_V2_PRUNE, cb.FEATURE_SET_ROAD_GEO_V2_JUNCTIONS,
             "geometry_plus_old", "g_E1_E2")] + [("g_E1_E2", "test")]
    for v, s in jobs:
        t0 = time.time()
        feats = ve.variant_features(v, list(panel.columns))
        ve.upsert_runs(MONTH_CSV, ve.run_variant(panel, v, s, seeds, features=feats))
        print(f"[month {s}] {v} ({len(feats)}): {time.time() - t0:.0f} с", flush=True)


def cmd_repro() -> None:
    panel = ve.load_panel()
    out = REPORTS / "v2_eval_stage7C_month_check.csv"
    rows = [ve.run_variant(panel, v, "test", [42]) for v in (SKDF, cb.FEATURE_SET_ROAD_GEO_BASELINE)]
    d = pd.concat(rows, ignore_index=True); d.to_csv(out, index=False)
    print(d[["variant", "roc_auc", "pr_auc", "precision_top_1pct"]].to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="+", choices=["history", "crossfit", "chain", "month", "repro"])
    ap.add_argument("--seeds", default=",".join(map(str, vy.SEEDS)))
    ap.add_argument("--split", default="both")
    ap.add_argument("--variants", default="", help="для chain: подмножество вариантов")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    splits = vy.resolve_splits(a.split)
    t0 = time.time()
    for c in a.cmd:
        t1 = time.time()
        if c == "history":
            cmd_history(seeds, splits)
        elif c == "crossfit":
            cmd_crossfit(seeds, splits)
        elif c == "chain":
            cmd_chain(seeds, splits, [v for v in a.variants.split(",") if v] or None)
        elif c == "month":
            cmd_month(seeds)
        elif c == "repro":
            cmd_repro()
        print(f"== {c}: {time.time() - t1:.0f} с", flush=True)
    print(f"всего {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
