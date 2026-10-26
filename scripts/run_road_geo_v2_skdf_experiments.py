"""Абляция этапа 5 (skdf): паспортные атрибуты ФГИС СКДФ поверх road_geo_v2_geometry.

Рецептура строго как в этапах 1–4: фильтр road_length_total_m > 0, клиппинг p99 с каппами только
по train (свои для каждого варианта; NaN не клиппируются и не заполняются — CatBoost обрабатывает
их сам), auto_class_weights="Balanced", 800/0.05/6, train_end 2023-01-01, сиды 42/7/123/2024/31337.

Варианты (блоки — build_road_geo_v2_skdf_features в catboost_baseline.py):
    road_geo_v2_geometry      A: опорный набор этапа 4 (30)
    skdf_B1_width             A + ширина проезжей части и ширина на полосу (32)
    skdf_B2_coating_replace   A, bad_surface_share_among_tagged -> skdf_low_coating_share (30)
    skdf_B2b_coating_both     A + skdf_low_coating_share (31)
    skdf_B3_hierarchy         A + ранг категории улицы по СП (31)
    skdf_B3b_hierarchy_value  B3 + ранг значения дороги (фед./рег./мест.) (32)
    skdf_B4a_lanes_replace    A, lanes_mean_weighted_v2 -> lanes_skdf_or_osm (30)
    skdf_B4b_lanes_both       A + skdf_lanes_mean_w (31)
    skdf_B5_speed_axle        A + скорость и осевая нагрузка (32)
    skdf_B5s_speed            A + скорость (31)
    skdf_B6_traffic           A + skdf_real_traffic_max (31)
    skdf_B6b_vehkm            A + skdf_veh_km_real (31; дубль B6, ρ 0.91 — только для решения)
    skdf_flag                 A + skdf_has_passport (31; диагностика «дорога на балансе»)
    skdf_C_*                  комбинации блоков (ширина + иерархия / + значение / + скорость и нагрузка)
    road_geo_v2_skdf          B: итоговая комбинация (ROAD_GEO_V2_SKDF_FINAL)
    road_geo_v2_skdf_fillmedian  B-med: B, NaN признаков СКДФ -> медиана train
    <любой>_fillmedian        тот же вариант с заполнением NaN признаков СКДФ медианой train
    skdf_C_all_passed         B1 + B3 + B5 + B6 — все блоки, прошедшие правило поодиночке (36);
                              skdfC36_{minus,plus,with}_… — операции от него
    skdfB_with_<вариант>      B, дополненный блоком варианта (B2/B4a — замены);
    skdf_plus_<признак>       A + признак;  skdfB_minus_<признак> / skdfB_plus_<признак> — B без/с признаком

Сырые прогоны: reports/road_geo_v2_skdf_seed_runs.csv (строки того же варианта заменяются).
Запуск:
    PYTHONPATH=src venv/bin/python scripts/run_road_geo_v2_skdf_experiments.py --variants road_geo_v2_geometry,skdf_B1_width
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
    ROAD_GEO_V2_SKDF_ALL_PASSED, ROAD_GEO_V2_SKDF_FINAL, apply_p99_clipping, attach_geometry_v2_features, attach_junction_v2_features,
    attach_road_grade_features, attach_skdf_v2_features, attach_tags_v2_features, build_road_geo_v2_geometry_features,
    build_road_geo_v2_skdf_features, fit_p99_clip_caps,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END, precision_recall_at_k  # noqa: E402
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PROJECT_ROOT  # noqa: E402
from road_accident_risk.skdf_v2 import SKDF_V2_ALL_FEATURES  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
SEED_RUNS = REPORTS / "road_geo_v2_skdf_seed_runs.csv"
SEEDS = (42, 7, 123, 2024, 31337)
FILL_SUFFIX = "_fillmedian"
VARIANTS: dict[str, dict] = {
    "skdf_B1_width": dict(width=True),
    "skdf_B2_coating_replace": dict(coating="replace"),
    "skdf_B2b_coating_both": dict(coating="both"),
    "skdf_B3_hierarchy": dict(hierarchy=True),
    "skdf_B3b_hierarchy_value": dict(hierarchy=True, extra=["skdf_value_rank_mean_w"]),
    "skdf_B4a_lanes_replace": dict(lanes="replace"),
    "skdf_B4b_lanes_both": dict(lanes="both"),
    "skdf_B5_speed_axle": dict(speed=True, axle=True),
    "skdf_B5s_speed": dict(speed=True),
    "skdf_B6_traffic": dict(traffic=True),
    "skdf_B6b_vehkm": dict(extra=["skdf_veh_km_real"]),
    "skdf_flag": dict(extra=["skdf_has_passport"]),
    # комбинации блоков, прошедших правило 5.9 поодиночке (поиск лучшей комбинации B)
    "skdf_C_width_hier": dict(width=True, hierarchy=True),
    "skdf_C_width_hier_value": dict(width=True, hierarchy=True, extra=["skdf_value_rank_mean_w"]),
    "skdf_C_width_speed_axle": dict(width=True, speed=True, axle=True),
    "skdf_C_width_hier_speed_axle": dict(width=True, hierarchy=True, speed=True, axle=True),
    "skdf_C_all_passed": ROAD_GEO_V2_SKDF_ALL_PASSED,
}
IMPORTANCE_OUT = {"road_geo_v2_geometry": "road_geo_v2_geometry_stage5_rerun", "road_geo_v2_skdf": "road_geo_v2_skdf"}


def variant_features(name: str, cols: list[str]) -> list[str]:
    if name.endswith(FILL_SUFFIX):
        return variant_features(name[: -len(FILL_SUFFIX)], cols)
    if name == "road_geo_v2_geometry":
        return build_road_geo_v2_geometry_features(cols)
    if name == "road_geo_v2_skdf":
        return build_road_geo_v2_skdf_features(cols, **ROAD_GEO_V2_SKDF_FINAL)
    if name in VARIANTS:
        return build_road_geo_v2_skdf_features(cols, **VARIANTS[name])
    if name.startswith("skdf_plus_"):
        return build_road_geo_v2_skdf_features(cols, extra=[name[len("skdf_plus_"):]])
    if name.startswith("skdfC36_"):  # те же операции, что skdfB_*, но от skdf_C_all_passed (36)
        rest = name[len("skdfC36_"):]
        kw = dict(ROAD_GEO_V2_SKDF_ALL_PASSED)
        if rest.startswith("with_"):
            return build_road_geo_v2_skdf_features(cols, **{**kw, **VARIANTS[rest[len("with_"):]]})
        if rest.startswith("plus_"):
            return build_road_geo_v2_skdf_features(cols, **kw, extra=[rest[len("plus_"):]])
        if rest.startswith("minus_"):
            base = build_road_geo_v2_skdf_features(cols, **kw)
            assert rest[len("minus_"):] in base
            return [f for f in base if f != rest[len("minus_"):]]
    if name.startswith("skdfB_with_"):  # B + блок-вариант (замены/добавления), напр. skdfB_with_skdf_B2_coating_replace
        kw = {**ROAD_GEO_V2_SKDF_FINAL, **VARIANTS[name[len("skdfB_with_"):]]}
        return build_road_geo_v2_skdf_features(cols, **kw)
    if name.startswith("skdfB_plus_"):
        kw = dict(ROAD_GEO_V2_SKDF_FINAL); kw["extra"] = [*kw.get("extra", ()), name[len("skdfB_plus_"):]]
        return build_road_geo_v2_skdf_features(cols, **kw)
    if name.startswith("skdfB_minus_"):
        drop = name[len("skdfB_minus_"):]
        base = build_road_geo_v2_skdf_features(cols, **ROAD_GEO_V2_SKDF_FINAL)
        assert drop in base, drop
        return [f for f in base if f != drop]
    raise ValueError(f"неизвестный вариант {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="варианты через запятую")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    panel = attach_skdf_v2_features(attach_geometry_v2_features(attach_tags_v2_features(attach_junction_v2_features(
        attach_road_grade_features(pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET))))))
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
        if name == "road_geo_v2_skdf":
            pd.DataFrame([{"feature": f, "p99_cap": c} for f, c in caps.items()]) \
                .to_csv(REPORTS / "road_geo_v2_skdf_p99_caps.csv", index=False)
        tr = apply_p99_clipping(tr0, caps); te = apply_p99_clipping(te0, caps)
        if name.endswith(FILL_SUFFIX):
            med = tr[feats].median()
            # только признаки этапа 5 с NaN; NaN этапов 0–4 не трогаем. lanes_skdf_or_osm: заполняется
            # и там, где нет ни паспорта, ни тега OSM (смешанный признак) — оговорено в README.
            nan_cols = [f for f in feats if f in SKDF_V2_ALL_FEATURES and (tr[f].isna().any() or te[f].isna().any())]
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
        out_imp.to_csv(REPORTS / f"catboost_feature_importance_{IMPORTANCE_OUT.get(name, 'road_geo_v2_skdf_' + name)}.csv",
                       index=False)

        new = pd.DataFrame(rows)
        if SEED_RUNS.exists():
            old = pd.read_csv(SEED_RUNS)
            new = pd.concat([old[old.variant != name], new], ignore_index=True)
        new.to_csv(SEED_RUNS, index=False)
        print(f"{name}: готово за {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
