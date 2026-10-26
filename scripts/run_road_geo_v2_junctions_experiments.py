"""Абляция этапа 2 (junctions): перекрёстки с честной топологией.

Рецептура строго как в этапе 1 (scripts/run_road_geo_v2_prune_experiments.py): фильтр
road_length_total_m > 0, клиппинг p99 с каппами только по train (свои для каждого варианта),
auto_class_weights="Balanced", 800/0.05/6, train_end 2023-01-01, сиды 42/7/123/2024/31337.

Варианты:
    road_geo_v2_prune              A: опорный набор этапа 1 (33)
    road_geo_v2_junctions          B: prune − 8 старых признаков перекрёстков + 7 новых (32)
    prune_plus_junctions           C: prune + 7 новых без удаления старых (40)
    junctions_alt_4leg_unsig       B, но street_junction_4leg_unsignalized_count вместо street_junction_4leg_count
    junctions_no_angle             B без junction_angle_dev_mean
    junctions_plus_<старый>        B + один старый признак (поиск недостающего)

Сырые прогоны: reports/road_geo_v2_junctions_seed_runs.csv (строки того же варианта заменяются).
Запуск:
    PYTHONPATH=src venv/bin/python scripts/run_road_geo_v2_junctions_experiments.py \
        --variants road_geo_v2_prune,road_geo_v2_junctions,prune_plus_junctions
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
    FEATURE_SET_ROAD_GEO_V2_PRUNE, V2_JUNCTIONS_NEW_FEATURES, V2_JUNCTIONS_REPLACED_FEATURES,
    apply_p99_clipping, attach_junction_v2_features, attach_road_grade_features, build_features,
    build_road_geo_v2_junctions_features, fit_p99_clip_caps,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END, precision_recall_at_k  # noqa: E402
from road_accident_risk.osm_road_features import JUNCTION_V2_FEATURES  # noqa: E402
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PROJECT_ROOT  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
SEED_RUNS = REPORTS / "road_geo_v2_junctions_seed_runs.csv"
SEEDS = (42, 7, 123, 2024, 31337)
IMPORTANCE_OUT = {
    "road_geo_v2_prune": "road_geo_v2_prune_stage2_rerun",
    "road_geo_v2_junctions": "road_geo_v2_junctions",
    "prune_plus_junctions": "road_geo_v2_junctions_C_prune_plus_new",
}


def variant_features(name: str, cols: list[str]) -> list[str]:
    if name == "road_geo_v2_prune":
        return build_features(pd.DataFrame(columns=cols), FEATURE_SET_ROAD_GEO_V2_PRUNE)
    if name == "road_geo_v2_junctions":
        return build_road_geo_v2_junctions_features(cols)
    if name == "prune_plus_junctions":
        return build_road_geo_v2_junctions_features(cols, keep_replaced=set(V2_JUNCTIONS_REPLACED_FEATURES))
    if name == "junctions_alt_4leg_unsig":
        new = [("street_junction_4leg_unsignalized_count" if f == "street_junction_4leg_count" else f)
               for f in V2_JUNCTIONS_NEW_FEATURES]
        return build_road_geo_v2_junctions_features(cols, new_features=new)
    if name == "junctions_no_angle":
        return build_road_geo_v2_junctions_features(
            cols, new_features=[f for f in V2_JUNCTIONS_NEW_FEATURES if f != "junction_angle_dev_mean"])
    if name.startswith("junctions_plus_"):
        old = name[len("junctions_plus_"):]
        assert old in V2_JUNCTIONS_REPLACED_FEATURES, old
        return build_road_geo_v2_junctions_features(cols, keep_replaced={old})
    raise ValueError(f"неизвестный вариант {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="варианты через запятую")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    panel = attach_junction_v2_features(attach_road_grade_features(pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)))
    panel = panel[panel["road_length_total_m"] > 0].copy()
    cols = list(panel.columns)
    allf = sorted({f for v in variants for f in variant_features(v, cols)} | set(JUNCTION_V2_FEATURES))
    df = panel[["h3", "month", *allf, TARGET]].copy()
    df["month"] = pd.to_datetime(df["month"]); df[TARGET] = df[TARGET].astype(int)
    tr0 = df[df.month < TRAIN_END].copy(); te0 = df[df.month >= TRAIN_END].copy()
    y = te0[TARGET].to_numpy()
    print(f"панель {df.shape}; train {tr0.shape}, test {te0.shape}", flush=True)

    for name in variants:
        feats = variant_features(name, cols)
        caps = fit_p99_clip_caps(tr0, feats)
        if name == "road_geo_v2_junctions":
            pd.DataFrame([{"feature": f, "p99_cap": c} for f, c in caps.items()]) \
                .to_csv(REPORTS / "road_geo_v2_junctions_p99_caps.csv", index=False)
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
        if name in IMPORTANCE_OUT:
            out_imp.to_csv(REPORTS / f"catboost_feature_importance_{IMPORTANCE_OUT[name]}.csv", index=False)

        new = pd.DataFrame(rows)
        if SEED_RUNS.exists():
            old = pd.read_csv(SEED_RUNS)
            new = pd.concat([old[old.variant != name], new], ignore_index=True)
        new.to_csv(SEED_RUNS, index=False)
        print(f"{name}: готово за {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
