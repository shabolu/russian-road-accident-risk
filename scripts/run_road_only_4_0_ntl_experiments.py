"""Эксперимент: даёт ли ночная освещённость (VIIRS) прирост к road-only 4.0.

Три варианта на одном и том же сплите и одних гиперпараметрах:

  1. road_only_4_0                  — базлайн, 60 признаков
  2. road_only_4_0 + ntl_alltime    — + одно статическое значение на ячейку
                                      (среднее годовых композитов 2015-2025)
  3. road_only_4_0 + ntl_year       — + значение композита за год строки

Освещённость подаётся в исходных единицах VIIRS — nW/cm²/sr, без бинаризации
и без логарифма. Клиппинг p99 применяется ко всем признакам одинаково, как
и в базовом рецепте 4.0 (каппы считаются только на train).

Запуск:
    venv/bin/python scripts/run_road_only_4_0_ntl_experiments.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.catboost_baseline import (  # noqa: E402
    FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP as FEATURE_SET,
    apply_p99_clipping,
    build_features,
    fit_p99_clip_caps,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END, precision_recall_at_k  # noqa: E402
from road_accident_risk.paths import (  # noqa: E402
    H3_MONTH_ROAD_FEATURES_PARQUET,
    PARQUET_DIR,
    PROJECT_ROOT,
)

REPORTS_DIR = PROJECT_ROOT / "reports"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-ekolugov-Documents-Playground-russian-road-accident-risk"
    "/cbe47b33-3eb7-441f-8bbe-fe725dbd5191/scratchpad"
)
YEARS = list(range(2015, 2026))
TOPK_FRACTIONS = (0.01, 0.05, 0.10)


def build_ntl_table(cells: list[str]) -> pd.DataFrame:
    """Средняя радиояркость VIIRS в ячейке за каждый год, nW/cm²/sr.

    Берётся среднее по пикселям, чьи центры попали в шестиугольник;
    если таких нет (ячейка меньше пикселя) — ближайший пиксель.
    """
    meta = json.loads((SCRATCH / "win_meta.json").read_text())
    res, i0, j0 = meta["res"], meta["i0"], meta["j0"]
    lat_top, lon_left = meta["lat_top"], meta["lon_left"]

    grids = {y: np.load(SCRATCH / f"wide_{y}.npy") for y in YEARS}
    rows, cols = next(iter(grids.values())).shape

    # индекс пикселей -> список ячеек (по центру пикселя)
    pix_cell = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        lat = lat_top - (i0 + r + 0.5) * res
        for c in range(cols):
            lon = lon_left + (j0 + c + 0.5) * res
            pix_cell[r, c] = h3.latlng_to_cell(lat, lon, 9)

    from collections import defaultdict

    members: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            members[pix_cell[r, c]].append((r, c))

    out = {"h3": cells}
    for y in YEARS:
        g = grids[y]
        vals = []
        for cell in cells:
            idx = members.get(cell)
            if idx:
                vals.append(float(np.mean([g[r, c] for r, c in idx])))
            else:  # ячейка меньше пикселя — берём ближайший
                lat, lon = h3.cell_to_latlng(cell)
                r = int(round((lat_top - lat) / res)) - i0
                c = int(round((lon - lon_left) / res)) - j0
                r = min(max(r, 0), rows - 1)
                c = min(max(c, 0), cols - 1)
                vals.append(float(g[r, c]))
        out[f"ntl_{y}"] = vals
    return pd.DataFrame(out)


def evaluate(name: str, features: list[str], train: pd.DataFrame, test: pd.DataFrame) -> dict:
    caps = fit_p99_clip_caps(train, features)
    tr = apply_p99_clipping(train, caps)
    te = apply_p99_clipping(test, caps)
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=42,
        allow_writing_files=False,
        verbose=0,
    )
    model.fit(Pool(tr[features], tr[TARGET], feature_names=features))
    scores = model.predict_proba(te[features])[:, 1]
    y = te[TARGET].to_numpy()

    row = {
        "variant": name,
        "n_features": len(features),
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
    }
    for frac in TOPK_FRACTIONS:
        k = precision_recall_at_k(y, scores, frac)
        pct = int(frac * 100)
        row[f"precision_top_{pct}pct"] = k["precision"]
        row[f"recall_top_{pct}pct"] = k["recall"]
        row[f"lift_top_{pct}pct"] = k["lift"]

    imp = pd.DataFrame(
        {"feature": features, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    imp.to_csv(REPORTS_DIR / f"catboost_feature_importance_{name}.csv", index=False)

    ntl_feats = [f for f in features if f.startswith("ntl")]
    if ntl_feats:
        sub = imp[imp.feature.isin(ntl_feats)]
        row["ntl_importance"] = float(sub.importance.sum())
        row["ntl_rank"] = int(imp.reset_index(drop=True).index[imp.reset_index(drop=True).feature.isin(ntl_feats)][0]) + 1
    return row


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)
    panel = panel[panel["road_length_total_m"] > 0].copy()
    base_features = build_features(panel, feature_set=FEATURE_SET)

    cells = sorted(panel["h3"].unique())
    ntl_path = PARQUET_DIR / "h3_ntl_viirs_annual_2015_2025.parquet"
    if ntl_path.exists():
        ntl = pd.read_parquet(ntl_path)
    else:
        ntl = build_ntl_table(cells)
        ntl.to_parquet(ntl_path, index=False)
    year_cols = [f"ntl_{y}" for y in YEARS]
    ntl["ntl_alltime"] = ntl[year_cols].mean(axis=1)

    df = panel[["h3", "month", *base_features, TARGET]].copy()
    df["month"] = pd.to_datetime(df["month"])
    df[TARGET] = df[TARGET].astype(int)
    df["year"] = df["month"].dt.year.clip(YEARS[0], YEARS[-1])

    df = df.merge(ntl[["h3", "ntl_alltime", *year_cols]], on="h3", how="left")
    long = ntl.melt(id_vars="h3", value_vars=year_cols, var_name="yc", value_name="ntl_year")
    long["year"] = long["yc"].str.slice(4).astype(int)
    df = df.merge(long[["h3", "year", "ntl_year"]], on=["h3", "year"], how="left")

    print(f"панель: {df.shape} | ячеек {df.h3.nunique():,}")
    print(f"ntl_alltime: пропусков {df.ntl_alltime.isna().sum()}, "
          f"медиана {df.ntl_alltime.median():.2f} nW/cm²/sr")
    print(f"ntl_year:    пропусков {df.ntl_year.isna().sum()}, "
          f"медиана {df.ntl_year.median():.2f} nW/cm²/sr")

    train = df[df["month"] < TRAIN_END].copy()
    test = df[df["month"] >= TRAIN_END].copy()
    print(f"train {train.shape} | test {test.shape} | base rate test {test[TARGET].mean():.6f}\n")

    variants = [
        ("road_only_4_0", base_features),
        ("road_only_4_0_ntl_alltime", [*base_features, "ntl_alltime"]),
        ("road_only_4_0_ntl_year", [*base_features, "ntl_year"]),
    ]
    rows = []
    for name, feats in variants:
        print(f"обучаю {name} ({len(feats)} признаков)...", flush=True)
        rows.append(evaluate(name, feats, train, test))

    summary = pd.DataFrame(rows)
    out = REPORTS_DIR / "road_only_4_0_ntl_summary.csv"
    summary.to_csv(out, index=False)
    print()
    print(summary.to_string(index=False))
    print(f"\nсохранено: {out}")


if __name__ == "__main__":
    main()
