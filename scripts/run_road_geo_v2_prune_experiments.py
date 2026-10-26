"""Абляция этапа 1 (prune): road_geo_baseline (61) против road_geo_v2_prune (33) и промежуточных шагов.

Рецептура бейзлайна: фильтр road_length_total_m > 0, клиппинг p99 с каппами только по train
(свои каппы для каждого варианта), auto_class_weights="Balanced", 800/0.05/6,
train_end 2023-01-01, сиды 42/7/123/2024/31337. Каркас — scripts/run_road_only_4_0_grade_experiments.py.

Варианты:
    road_geo_baseline       61 признак
    road_geo_v2_prune       33 признака (V2_PRUNE_EXCLUDED_FEATURES)
    prune_step1_trivial     минус мёртвый груз, композиция, полнота OSM, скорость, длина (49)
    prune_step2_junctions   + минус перекрёстки (41)
    prune_step3_geometry    + минус кривизна, сегментация, тупики (34); финал = + минус покрытие (33)
    prune_plus_<группа>     33 + возврат одной группы (поиск виновника при просадке)

Сырые прогоны дописываются в reports/road_geo_v2_prune_seed_runs.csv (строки того же варианта
заменяются). Запуск:
    venv/bin/python scripts/run_road_geo_v2_prune_experiments.py --variants road_geo_baseline,road_geo_v2_prune
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.catboost_baseline import (  # noqa: E402
    FEATURE_SET_ROAD_GEO_BASELINE, V2_PRUNE_EXCLUDED_GROUPS, apply_p99_clipping,
    attach_road_grade_features, build_features, build_road_geo_v2_prune_features, fit_p99_clip_caps,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END, precision_recall_at_k  # noqa: E402
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PROJECT_ROOT  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
SEED_RUNS = REPORTS / "road_geo_v2_prune_seed_runs.csv"
SEEDS = (42, 7, 123, 2024, 31337)

# Шаги кумулятивного удаления (группы V2_PRUNE_EXCLUDED_GROUPS)
STEPS = {
    "prune_step1_trivial": ["dead_weight", "composition", "osm_tag_completeness", "speed", "length"],
    "prune_step2_junctions": ["junctions"],
    "prune_step3_geometry": ["curvature", "segmentation", "dead_ends"],
    "road_geo_v2_prune": ["surface"],
}
# Макрогруппы для возврата по одной (поиск виновника)
RETURN_GROUPS = {
    "junctions": ["junctions"],
    "curvature": ["curvature"],
    "segmentation": ["segmentation", "dead_ends"],
    "surface": ["surface"],
    "speed_lanes": ["osm_tag_completeness", "speed", "length"],
    "dead_weight": ["dead_weight", "composition"],
}


def group_features(groups: list[str]) -> set[str]:
    return {f for g in groups for f in V2_PRUNE_EXCLUDED_GROUPS[g]}


def variant_features(name: str, columns: list[str], baseline: list[str]) -> list[str]:
    if name == "road_geo_baseline":
        return baseline
    if name in STEPS:
        removed: set[str] = set()
        for step, groups in STEPS.items():
            removed |= group_features(groups)
            if step == name:
                break
        return [f for f in baseline if f not in removed]
    if name.startswith("prune_plus_"):
        return build_road_geo_v2_prune_features(columns, keep_excluded=group_features(RETURN_GROUPS[name[11:]]))
    raise ValueError(f"неизвестный вариант {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="варианты через запятую")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    panel = attach_road_grade_features(pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET))
    panel = panel[panel["road_length_total_m"] > 0].copy()
    baseline = build_features(panel, feature_set=FEATURE_SET_ROAD_GEO_BASELINE)
    df = panel[["h3", "month", *baseline, TARGET]].copy()
    df["month"] = pd.to_datetime(df["month"]); df[TARGET] = df[TARGET].astype(int)
    tr0 = df[df.month < TRAIN_END].copy(); te0 = df[df.month >= TRAIN_END].copy()
    y = te0[TARGET].to_numpy()
    print(f"панель {df.shape}; train {tr0.shape}, test {te0.shape}", flush=True)

    for name in variants:
        feats = variant_features(name, list(panel.columns), baseline)
        caps = fit_p99_clip_caps(tr0, feats)
        if name == "road_geo_v2_prune":
            pd.DataFrame([{"feature": f, "p99_cap": c} for f, c in caps.items()]) \
                .to_csv(REPORTS / "road_geo_v2_prune_p99_caps.csv", index=False)
        tr = apply_p99_clipping(tr0, caps); te = apply_p99_clipping(te0, caps)
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
        if name in ("road_geo_baseline", "road_geo_v2_prune"):
            suffix = "road_geo_v2_prune" if name == "road_geo_v2_prune" else "road_geo_baseline_5seeds"
            out_imp.to_csv(REPORTS / f"catboost_feature_importance_{suffix}.csv", index=False)

        new = pd.DataFrame(rows)
        if SEED_RUNS.exists():
            old = pd.read_csv(SEED_RUNS)
            new = pd.concat([old[old.variant != name], new], ignore_index=True)
        new.to_csv(SEED_RUNS, index=False)
        print(f"{name}: готово за {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
