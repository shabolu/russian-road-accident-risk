"""Ablation: road-only 4.0 (59 признаков) против road-only 4.0 + 6 признаков уклона.

Тот же сплит, гиперпараметры и клиппинг p99, что в базовом рецепте 4.0.
Пять сидов на вариант. Дополнительно сохраняются средние тестовые скоры по ячейкам
для обоих вариантов — для карты «где модель изменила мнение».

Запуск:
    venv/bin/python scripts/run_road_only_4_0_grade_experiments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.catboost_baseline import (  # noqa: E402
    FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP as FEATURE_SET,
    apply_p99_clipping, build_features, fit_p99_clip_caps,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END, precision_recall_at_k  # noqa: E402
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PARQUET_DIR, PROJECT_ROOT  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
GRADE_FEATURES = [
    "road_grade_abs_mean_weighted", "road_grade_abs_p90", "steep_road_share",
    "major_road_grade_abs_mean", "junction_on_grade_count", "grade_change_density_per_km",
]
SEEDS = (42, 7, 123, 2024, 31337)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=",".join(GRADE_FEATURES),
                    help="признаки уклона через запятую (по умолчанию все шесть)")
    ap.add_argument("--tag", default="grade", help="суффикс выходных файлов")
    args = ap.parse_args()
    feats_grade = [f.strip() for f in args.features.split(",") if f.strip()]
    tag = args.tag
    panel = pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)
    panel = panel[panel["road_length_total_m"] > 0].copy()
    base = build_features(panel, feature_set=FEATURE_SET)
    grade = pd.read_parquet(PARQUET_DIR / "h3_road_grade_features.parquet")
    df = panel[["h3", "month", *base, TARGET]].merge(grade[["h3", *feats_grade]], on="h3", how="left")
    df["month"] = pd.to_datetime(df["month"]); df[TARGET] = df[TARGET].astype(int)
    print(f"панель {df.shape}; пропусков уклона: {df[feats_grade[0]].isna().mean():.2%}", flush=True)
    tr0 = df[df.month < TRAIN_END].copy(); te0 = df[df.month >= TRAIN_END].copy()

    rows, scores = [], {}
    for name, feats in (("road_only_4_0", base), (f"road_only_4_0_{tag}", [*base, *feats_grade])):
        caps = fit_p99_clip_caps(tr0, feats)
        tr = apply_p99_clipping(tr0, caps); te = apply_p99_clipping(te0, caps)
        acc = np.zeros(len(te))
        for seed in SEEDS:
            m = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, loss_function="Logloss",
                                   eval_metric="PRAUC", auto_class_weights="Balanced", random_seed=seed,
                                   allow_writing_files=False, verbose=0)
            m.fit(Pool(tr[feats], tr[TARGET], feature_names=feats))
            s = m.predict_proba(te[feats])[:, 1]; y = te[TARGET].to_numpy(); acc += s
            r = {"variant": name, "seed": seed, "n_features": len(feats),
                 "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s)}
            for frac in (0.01, 0.05, 0.10):
                k = precision_recall_at_k(y, s, frac); p = int(frac * 100)
                r[f"precision_top_{p}pct"] = k["precision"]; r[f"recall_top_{p}pct"] = k["recall"]; r[f"lift_top_{p}pct"] = k["lift"]
            rows.append(r)
            if seed == SEEDS[0]:
                pd.DataFrame({"feature": feats, "importance": m.get_feature_importance()}) \
                    .sort_values("importance", ascending=False) \
                    .to_csv(REPORTS / f"catboost_feature_importance_{name}_seed42.csv", index=False)
            print(f"{name} seed={seed} ok", flush=True)
        scores[name] = acc / len(SEEDS)

    pd.DataFrame(rows).to_csv(REPORTS / f"road_only_4_0_{tag}_seed_runs.csv", index=False)
    out = te0[["h3", "month", TARGET]].copy()
    out["score_base"] = scores["road_only_4_0"]; out["score_grade"] = scores[f"road_only_4_0_{tag}"]
    out.to_parquet(PARQUET_DIR / f"h3_month_road_only_4_0_{tag}_scores.parquet", index=False)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
