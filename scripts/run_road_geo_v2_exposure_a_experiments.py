"""Этап 6.2–6.5 (запуск 6a): блоки E1–E4 поверх A = road_geo_v2_skdf, жадная сборка, диагностики.

Схема — src/road_accident_risk/v2_eval.py: отбор по folds (valid2021 + valid, 5 сидов), тест — протокол.
Прогоны: reports/v2_eval_stage6a.csv; разности: reports/v2_eval_stage6a_delta.csv,
reports/v2_eval_stage6a_folds_delta.csv (к базе --base, по умолчанию A).

Варианты (имена):
    A                      = road_geo_v2_skdf (32)
    e1_center_two / e1_center_poi   — A + центр (а: ближайший из двух; б: пики плотности POI) + major + signal
    e2_cell / e2_r300 / e2_r500     — A + счёты POI в ячейке / в круге 300 / 500 м
    e3_build               — A + площадь основания + доля мелких зданий
    e4_old / e4_new        — A + расстояние до остановки + трамвай; e4_new: старый счёт остановок -> новый
    e4_routes              — выбранный E4 + pt_route_count
    g_<блоки>              — жадная сборка, напр. g_E2_E1 (выбранные варианты блоков, CHOSEN)
    road_geo_v2_exposure_a — итог жадной сборки (набор в catboost_baseline)
    eart / elevels / emed  — диагностики поверх exposure_a

    PYTHONPATH=src venv/bin/python scripts/run_road_geo_v2_exposure_a_experiments.py \
        --variants road_geo_v2_skdf,e1_center_two,e1_center_poi --split folds
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk import catboost_baseline as cb  # noqa: E402
from road_accident_risk import exposure_v6 as ev  # noqa: E402
from road_accident_risk.v2_eval import (  # noqa: E402
    REPORTS, SEEDS, compare, compare_folds, load_panel, register_variant, run_variant, split_frames,
    upsert_runs, variant_features,
)

A = cb.FEATURE_SET_ROAD_GEO_V2_SKDF
OLD_STOPS = "public_transport_stop_count"
# 9 старых признаков рисовки этапа 4 (stage4_geometry/README.md)
ART_FEATURES = ["road_segment_length_median", "short_segment_share", "road_curvature_mean_weighted",
                "road_curvature_max", "curvature_deg_per_km", "sharp_turn_share", "dead_end_count",
                "road_orientation_entropy", "railway_crossing_density_per_km"]


def add(*feats):
    return lambda base: [*base, *[f for f in feats if f not in base]]


def replace_add(old, new, *feats):
    return lambda base: [*[new if f == old else f for f in base], *[f for f in feats if f not in base]]


# Блочные варианты: функция «список признаков -> список признаков».
BLOCK_VARIANTS = {
    "e1_center_two": add("dist_to_center_two_m", *ev.E1_OTHER),
    "e1_center_poi": add("dist_to_center_poi_m", *ev.E1_OTHER),
    "e2_cell": add(*ev.e2_names("cell")),
    "e2_r300": add(*ev.e2_names("r300")),
    # r500: healthcare_count_r500 исключён дедупом (ρ 0.764 с poi_attractor_count_r500, подмножество типов)
    "e2_r500": add("poi_attractor_count_r500", "school_kindergarten_count_r500"),
    # attractor_building_footprint_m2 исключён дедупом (ρ 0.953 с poi_attractor_count, тот же источник)
    "e3_build": add("building_footprint_m2", "small_building_share"),
    "e4_old": add("dist_to_nearest_stop_m", "tram_track_length_m"),
    "e4_new": replace_add(OLD_STOPS, "pt_stop_count_v6", "dist_to_nearest_stop_m", "tram_track_length_m"),
}
# Выбор внутри блоков по folds (заполняется по итогам шага 3, см. LOG): блок -> вариант.
CHOSEN: dict[str, str] = {}
ROUTES = add("pt_route_count")


def block_fn(block: str):
    v = CHOSEN[block]
    if v == "e4_routes":
        return lambda base: ROUTES(BLOCK_VARIANTS[CHOSEN["E4_base"]](base))
    return BLOCK_VARIANTS[v]


def compose(blocks: list[str]):
    def f(cols):
        out = variant_features(A, cols)
        for b in blocks:
            out = block_fn(b)(out)
        assert len(out) == len(set(out)), out
        return out
    return f


def register_all() -> None:
    for name, fn in BLOCK_VARIANTS.items():
        register_variant(name, lambda cols, _fn=fn: _fn(variant_features(A, cols)))
    if "E4_base" in CHOSEN:
        register_variant("e4_routes", lambda cols: ROUTES(BLOCK_VARIANTS[CHOSEN["E4_base"]](variant_features(A, cols))))
    for name in list(_GREEDY_NAMES):
        register_variant(name, compose(name[2:].split("_")))
    if cb_has_exposure_a():
        ex = cb.FEATURE_SET_ROAD_GEO_V2_EXPOSURE_A
        register_variant("eart", lambda cols: [*variant_features(ex, cols), *ART_FEATURES])
        register_variant("elevels", lambda cols: [*variant_features(ex, cols), "levels_tagged_share"])
        register_variant("emed", lambda cols: variant_features(ex, cols))


_GREEDY_NAMES: list[str] = []


def cb_has_exposure_a() -> bool:
    return hasattr(cb, "FEATURE_SET_ROAD_GEO_V2_EXPOSURE_A")


def fill_median(panel: pd.DataFrame, split: str, feats: list[str]) -> pd.DataFrame:
    """E-med: NaN новых признаков -> медиана по обучающей части сплита (по строкам ячейко-месяцев)."""
    new = [f for f in feats if f in ev.EXPOSURE_V6_FEATURES and panel[f].isna().any()]
    tr, _ = split_frames(panel, split)
    out = panel.copy()
    for f in new:
        out[f] = out[f].fillna(tr[f].median())
    print(f"[{split}] emed: заполнены медианой {new}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--split", default="folds")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--tag", default="stage6a")
    ap.add_argument("--base", default=A)
    ap.add_argument("--chosen", default="", help="блок=вариант через запятую, напр. E1=e1_center_poi,E2=e2_cell")
    ap.add_argument("--greedy", default="", help="имена g_<блоки> через запятую для регистрации")
    ap.add_argument("--importance", action="store_true", help="сохранить важности (reports/catboost_feature_importance_*)")
    args = ap.parse_args()
    for kv in filter(None, args.chosen.split(",")):
        k, v = kv.split("="); CHOSEN[k] = v
    _GREEDY_NAMES.extend(filter(None, args.greedy.split(",")))
    register_all()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    alias = {"both": ["valid", "test"], "folds": ["valid2021", "valid"], "all": ["valid2021", "valid", "test"]}
    splits = alias.get(args.split) or [s.strip() for s in args.split.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = REPORTS / f"v2_eval_{args.tag}.csv"

    t0 = time.time()
    panel = load_panel(attach=[cb.attach_exposure_v6_features])
    print(f"панель {panel.shape} [{time.time() - t0:.0f} с]; CHOSEN={CHOSEN}", flush=True)
    runs = pd.read_csv(out) if out.exists() else pd.DataFrame()
    for split in splits:
        for v in variants:
            t1 = time.time()
            feats = variant_features(v, list(panel.columns))
            print(f"[{split}] {v}: {len(feats)} признаков; новые: {[f for f in feats if f in ev.EXPOSURE_V6_FEATURES]}",
                  flush=True)
            p = fill_median(panel, split, feats) if v == "emed" else panel
            imp = (REPORTS / f"catboost_feature_importance_stage6a_{v}_{split}.csv") if args.importance else None
            runs = upsert_runs(out, run_variant(p, v, split, seeds, features=feats, importance_out=imp))
            print(f"[{split}] {v}: готово за {time.time() - t1:.0f} с", flush=True)

    if args.base:
        runs = pd.read_csv(out)
        deltas = [compare(runs, args.base, v, split=s) for s in splits for v in variants
                  if v != args.base and not runs[(runs.split == s) & (runs.variant == args.base)].empty]
        if deltas:
            d = pd.concat(deltas, ignore_index=True)
            dpath = REPORTS / f"v2_eval_{args.tag}_delta.csv"
            if dpath.exists():
                old = pd.read_csv(dpath)
                key = ["split", "base", "variant", "column"]
                old = old[~old.set_index(key).index.isin(d.set_index(key).index)]
                d = pd.concat([old, d], ignore_index=True)
            d.to_csv(dpath, index=False)
            sel = d[d.column.isin(["roc_auc", "pr_auc", "precision_top_1pct", "recall_top_10pct"])
                    & d.variant.isin(variants) & d.split.isin(splits) & (d.base == args.base)]
            if {"valid2021", "valid"} <= set(splits):
                fo = pd.concat([compare_folds(runs, args.base, v) for v in variants if v != args.base])
                fpath = REPORTS / f"v2_eval_{args.tag}_folds_delta.csv"
                if fpath.exists():
                    old = pd.read_csv(fpath)
                    key = ["base", "variant", "column"]
                    old = old[~old.set_index(key).index.isin(fo.set_index(key).index)]
                    fo = pd.concat([old, fo], ignore_index=True)
                fo.to_csv(fpath, index=False)
                sel = pd.concat([sel, fo[fo.column.isin(sel.column.unique()) & fo.variant.isin(variants)
                                         & (fo.base == args.base)]], ignore_index=True)
            with pd.option_context("display.width", 250, "display.max_rows", 400):
                cols = ["split", "variant", "metric", "delta", "two_sigma", "two_sd_paired", "verdict", "verdict_paired"]
                if "same_sign_all_folds" in sel.columns:
                    cols += ["same_sign_all_folds"]
                print(sel[cols].to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
    print(f"всего {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
