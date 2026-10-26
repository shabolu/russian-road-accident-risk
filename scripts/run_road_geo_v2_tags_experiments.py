"""Абляция этапа 3 (tags): теги OSM — «нет тега» = неизвестно (NaN), а не 0; асфальт -> длины.

Рецептура строго как в этапах 1–2: фильтр road_length_total_m > 0, клиппинг p99 с каппами только
по train (свои для каждого варианта; NaN не клиппируются и не заполняются — CatBoost обрабатывает
их сам, Min-режим по умолчанию), auto_class_weights="Balanced", 800/0.05/6, train_end 2023-01-01,
сиды 42/7/123/2024/31337.

Варианты:
    road_geo_v2_junctions   A: опорный набор этапа 2 (32)
    tags_exposure_only      B1: A, asphalt_surface_length_m -> major_road_length_m + street_length_m (33)
    tags_semantics_only     B2: A, bad_surface_share / lanes_mean_weighted / oneway_share -> v2 с NaN (32)
    road_geo_v2_tags        B: B1 + B2 — итоговый набор (33)
    <вариант>_fillmedian    тот же набор, но NaN заполнены медианой train (диагностика сигнала пропуска)
    tags_plus_<признак>     B + один признак (диагностика: asphalt_surface_length_m, surface_tagged_share, ...)
    tags_minus_<признак>    B без одного признака
    junctions_plus_<признак> A + один признак

Сырые прогоны: reports/road_geo_v2_tags_seed_runs.csv (строки того же варианта заменяются).
Запуск:
    PYTHONPATH=src venv/bin/python scripts/run_road_geo_v2_tags_experiments.py \
        --variants road_geo_v2_junctions,tags_exposure_only,tags_semantics_only,road_geo_v2_tags
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.catboost_baseline import (  # noqa: E402
    FEATURE_SET_ROAD_GEO_V2_JUNCTIONS, apply_p99_clipping, attach_junction_v2_features,
    attach_road_grade_features, attach_tags_v2_features, build_features, build_road_geo_v2_tags_features,
    fit_p99_clip_caps,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END, precision_recall_at_k  # noqa: E402
from road_accident_risk.osm_road_features import TAGS_V2_FEATURES  # noqa: E402
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PROJECT_ROOT  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
SEED_RUNS = REPORTS / "road_geo_v2_tags_seed_runs.csv"
SEEDS = (42, 7, 123, 2024, 31337)
IMPORTANCE_OUT = {
    "road_geo_v2_junctions": "road_geo_v2_junctions_stage3_rerun",
    "tags_exposure_only": "road_geo_v2_tags_B1_exposure",
    "tags_semantics_only": "road_geo_v2_tags_B2_semantics",
    "road_geo_v2_tags": "road_geo_v2_tags",
}


FILL_SUFFIX = "_fillmedian"  # диагностика: NaN -> медиана train (убирает сигнал «тег есть/нет»)


def variant_features(name: str, cols: list[str]) -> list[str]:
    if name.endswith(FILL_SUFFIX):
        return variant_features(name[: -len(FILL_SUFFIX)], cols)
    if name == "road_geo_v2_junctions":
        return build_features(pd.DataFrame(columns=cols), FEATURE_SET_ROAD_GEO_V2_JUNCTIONS)
    if name == "tags_exposure_only":
        return build_road_geo_v2_tags_features(cols, replace_semantics=False)
    if name == "tags_semantics_only":
        return build_road_geo_v2_tags_features(cols, replace_exposure=False)
    if name == "road_geo_v2_tags":
        return build_road_geo_v2_tags_features(cols)
    if name.startswith("tags_plus_"):
        return build_road_geo_v2_tags_features(cols, extra=[name[len("tags_plus_"):]])
    if name.startswith("tags_minus_"):
        drop = name[len("tags_minus_"):]
        base = build_road_geo_v2_tags_features(cols)
        assert drop in base, drop
        return [f for f in base if f != drop]
    if name.startswith("junctions_plus_"):
        add = name[len("junctions_plus_"):]
        base = variant_features("road_geo_v2_junctions", cols)
        assert add in cols and add not in base, add
        return [*base, add]
    raise ValueError(f"неизвестный вариант {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="варианты через запятую")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    panel = attach_tags_v2_features(attach_junction_v2_features(
        attach_road_grade_features(pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET))))
    panel = panel[panel["road_length_total_m"] > 0].copy()
    cols = list(panel.columns)
    allf = sorted({f for v in variants for f in variant_features(v, cols)})
    df = panel[["h3", "month", *allf, TARGET]].copy()
    df["month"] = pd.to_datetime(df["month"]); df[TARGET] = df[TARGET].astype(int)
    tr0 = df[df.month < TRAIN_END].copy(); te0 = df[df.month >= TRAIN_END].copy()
    y = te0[TARGET].to_numpy()
    print(f"панель {df.shape}; train {tr0.shape}, test {te0.shape}", flush=True)

    for name in variants:
        feats = variant_features(name, cols)
        caps = fit_p99_clip_caps(tr0, feats)
        if name == "road_geo_v2_tags":
            pd.DataFrame([{"feature": f, "p99_cap": c} for f, c in caps.items()]) \
                .to_csv(REPORTS / "road_geo_v2_tags_p99_caps.csv", index=False)
        tr = apply_p99_clipping(tr0, caps); te = apply_p99_clipping(te0, caps)
        if name.endswith(FILL_SUFFIX):
            med = tr[feats].median()
            # только признаки этапа 3 с NaN-семантикой; NaN уклона и угла (этапы 0–2) не трогаем
            nan_cols = [f for f in feats if f in TAGS_V2_FEATURES and (tr[f].isna().any() or te[f].isna().any())]
            tr = tr.copy(); te = te.copy()
            tr[nan_cols] = tr[nan_cols].fillna(med[nan_cols]); te[nan_cols] = te[nan_cols].fillna(med[nan_cols])
            print(f"{name}: NaN -> медиана train в {nan_cols}", flush=True)
        nan_tr = int(tr[feats].isna().sum().sum()); nan_tr0 = int(tr0[feats].isna().sum().sum())
        assert nan_tr == nan_tr0 or name.endswith(FILL_SUFFIX), "клиппинг изменил число NaN"
        rows, imps = [], []
        t0 = time.time()
        for seed in SEEDS:
            m = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, loss_function="Logloss",
                                   eval_metric="PRAUC", auto_class_weights="Balanced", random_seed=seed,
                                   allow_writing_files=False, verbose=0)
            m.fit(Pool(tr[feats], tr[TARGET], feature_names=feats))
            s = m.predict_proba(te[feats])[:, 1]
            r = {"variant": name, "seed": seed, "n_features": len(feats),
                 "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s)}
            for frac in (0.01, 0.05, 0.10):
                k = precision_recall_at_k(y, s, frac); p = int(frac * 100)
                r[f"precision_top_{p}pct"] = k["precision"]; r[f"recall_top_{p}pct"] = k["recall"]
                r[f"lift_top_{p}pct"] = k["lift"]
            rows.append(r)
            imps.append(pd.Series(m.get_feature_importance(), index=feats, name=seed))
            print(f"{name} ({len(feats)}) seed={seed}: roc={r['roc_auc']:.6f} pr={r['pr_auc']:.6f} "
                  f"p@1={r['precision_top_1pct']:.4f} [{time.time() - t0:.0f} с]", flush=True)

        imp = pd.concat(imps, axis=1)
        out_imp = pd.DataFrame({"feature": feats, "importance_seed42": imp[42].values,
                                "importance_mean5": imp.mean(axis=1).values,
                                "importance_sd5": imp.std(axis=1).values}) \
            .sort_values("importance_mean5", ascending=False)
        out_imp.to_csv(REPORTS / f"catboost_feature_importance_{IMPORTANCE_OUT.get(name, 'road_geo_v2_tags_' + name)}.csv",
                       index=False)

        new = pd.DataFrame(rows)
        if SEED_RUNS.exists():
            old = pd.read_csv(SEED_RUNS)
            new = pd.concat([old[old.variant != name], new], ignore_index=True)
        new.to_csv(SEED_RUNS, index=False)
        print(f"{name}: готово за {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
