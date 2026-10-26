"""Схема оценки этапа 6 (Road+Geo v2): отбор вариантов на валидации, тест — только для протокола.

Сплиты (панель ячейка H3 res 9 × месяц, фильтр road_length_total_m > 0):
    valid      fit = месяцы < 2022-01-01 (2015-01…2021-12), оценка = 2022-01…2022-12;
    test       train = месяцы < 2023-01-01 (2015-01…2022-12), оценка = ≥ 2023-01-01 (как на этапах 1–5);
    valid2021  fit = < 2021-01-01, оценка = 2021-01…2021-12 (второй фолд скользящей схемы, опционально).
Каппы p99 считаются по обучающей части своего сплита (для valid — только по fit).

Рецептура этапов 1–5: Balanced, 800 итераций, lr 0.05, depth 6, сиды 42/7/123/2024/31337, NaN не
заполняются. Метрики: ROC-AUC, PR-AUC, precision/recall/lift @1/5/10 %.

Соглашение о значимости (README этапов 1, 2, 5; scripts/plot_road_geo_v2_prune.py::verdict):
    2σ = 2·sqrt((sd_A² + sd_B²)/n), n — число сидов; вердикт по |Δ|/2σ: < 1 — шум, 1–1.5 — на границе,
    ≥ 1.5 — значимо. Парная оценка: 2·sd разностей по одинаковым сидам, те же пороги.

Варианты: имена наборов из catboost_baseline (road_geo_baseline, road_geo_v2_prune, …, road_geo_v2_skdf),
варианты абляции этапа 5 (как в scripts/run_road_geo_v2_skdf_experiments.py) и любые варианты,
зарегистрированные через register_variant(name, builder) — для подпунктов 6.2–6.9.

Пример:
    from road_accident_risk.v2_eval import load_panel, run_variant, compare
    panel = load_panel()
    runs = pd.concat([run_variant(panel, v, "valid") for v in ("road_geo_v2_skdf", "my_variant")])
    print(compare(runs, "road_geo_v2_skdf", "my_variant", split="valid"))
"""
from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score

from road_accident_risk import catboost_baseline as cb
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END, precision_recall_at_k
from road_accident_risk.paths import H3_MONTH_ROAD_FEATURES_PARQUET, PROJECT_ROOT

SEEDS: tuple[int, ...] = (42, 7, 123, 2024, 31337)
CATBOOST_PARAMS = dict(iterations=800, learning_rate=0.05, depth=6, loss_function="Logloss", eval_metric="PRAUC",
                       auto_class_weights="Balanced", allow_writing_files=False, verbose=0)
FIT_END = "2022-01-01"
REPORTS = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class Split:
    name: str
    train_end: str          # обучение: month < train_end (и каппы p99 — по этой части)
    eval_start: str         # оценка: eval_start <= month < eval_end
    eval_end: str | None    # None — до конца панели


SPLITS: dict[str, Split] = {
    "valid": Split("valid", FIT_END, FIT_END, TRAIN_END),
    "test": Split("test", TRAIN_END, TRAIN_END, None),
    "valid2021": Split("valid2021", "2021-01-01", "2021-01-01", FIT_END),
}

METRICS: list[tuple[str, str]] = [
    ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
    ("precision_top_1pct", "precision@1 %"), ("recall_top_1pct", "recall@1 %"), ("lift_top_1pct", "lift@1 %"),
    ("precision_top_5pct", "precision@5 %"), ("recall_top_5pct", "recall@5 %"), ("lift_top_5pct", "lift@5 %"),
    ("precision_top_10pct", "precision@10 %"), ("recall_top_10pct", "recall@10 %"), ("lift_top_10pct", "lift@10 %"),
]

# ---------------------------------------------------------------- варианты

_SET_BUILDERS: dict[str, Callable[[list[str]], list[str]]] = {
    cb.FEATURE_SET_ROAD_GEO_BASELINE: cb.build_road_geo_baseline_features,
    cb.FEATURE_SET_ROAD_GEO_V2_PRUNE: cb.build_road_geo_v2_prune_features,
    cb.FEATURE_SET_ROAD_GEO_V2_JUNCTIONS: cb.build_road_geo_v2_junctions_features,
    cb.FEATURE_SET_ROAD_GEO_V2_TAGS: cb.build_road_geo_v2_tags_features,
    cb.FEATURE_SET_ROAD_GEO_V2_GEOMETRY: cb.build_road_geo_v2_geometry_features,
    cb.FEATURE_SET_ROAD_GEO_V2_SKDF: lambda cols: cb.build_road_geo_v2_skdf_features(cols, **cb.ROAD_GEO_V2_SKDF_FINAL),
    cb.FEATURE_SET_ROAD_GEO_V2_EXPOSURE_A: cb.build_road_geo_v2_exposure_a_features,  # этап 6a: нужна attach_exposure_v6_features
}
_REGISTRY: dict[str, Callable[[list[str]], list[str]]] = {}
_stage5_variant_features: Callable[[str, list[str]], list[str]] | None = None


def register_variant(name: str, builder: Callable[[list[str]], list[str]] | Iterable[str]) -> None:
    """Регистрирует вариант: функция (колонки панели -> список признаков) или готовый список признаков."""
    if not callable(builder):
        feats = list(builder)
        builder = lambda cols, _f=feats: list(_f)  # noqa: E731
    _REGISTRY[name] = builder


def _stage5(name: str, cols: list[str]) -> list[str]:
    """Варианты абляции этапа 5 — из scripts/run_road_geo_v2_skdf_experiments.py (одна точка истины)."""
    global _stage5_variant_features
    if _stage5_variant_features is None:
        path = PROJECT_ROOT / "scripts" / "run_road_geo_v2_skdf_experiments.py"
        spec = importlib.util.spec_from_file_location("_run_road_geo_v2_skdf_experiments", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _stage5_variant_features = mod.variant_features
    return _stage5_variant_features(name, cols)


def variant_features(name: str, cols: list[str]) -> list[str]:
    if name.endswith("_fillmedian"):
        raise NotImplementedError("заполнение NaN медианой в v2_eval не реализовано — зарегистрируйте вариант явно")
    if name in _REGISTRY:
        return _REGISTRY[name](cols)
    if name in _SET_BUILDERS:
        return _SET_BUILDERS[name](cols)
    return _stage5(name, cols)


# ---------------------------------------------------------------- данные

def load_panel(path: Path = H3_MONTH_ROAD_FEATURES_PARQUET,
               attach: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = ()) -> pd.DataFrame:
    """Панель с признаками этапов 0–5 (как в скрипте этапа 5) и фильтром road_length_total_m > 0.

    attach — дополнительные функции подтягивания признаков (этап 6), применяются после штатных, до фильтра.
    """
    panel = pd.read_parquet(path)
    for f in (cb.attach_road_grade_features, cb.attach_junction_v2_features, cb.attach_tags_v2_features,
              cb.attach_geometry_v2_features, cb.attach_skdf_v2_features, *attach):
        panel = f(panel)
    panel = panel[panel["road_length_total_m"] > 0].copy()
    panel["month"] = pd.to_datetime(panel["month"])
    panel[TARGET] = panel[TARGET].astype(int)
    return panel


def split_frames(panel: pd.DataFrame, split: str | Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = SPLITS[split] if isinstance(split, str) else split
    tr = panel[panel["month"] < s.train_end]
    m = panel["month"] >= s.eval_start
    if s.eval_end is not None:
        m &= panel["month"] < s.eval_end
    return tr, panel[m]


def split_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, s in SPLITS.items():
        tr, ev = split_frames(panel, s)
        for part, d in (("train", tr), ("eval", ev)):
            rows.append({"split": name, "part": part, "month_min": d.month.min().date(), "month_max": d.month.max().date(),
                         "months": d.month.nunique(), "rows": len(d), "positives": int(d[TARGET].sum()),
                         "positive_rate": d[TARGET].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- обучение

def metrics_row(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    r = {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s)}
    for frac in (0.01, 0.05, 0.10):
        k = precision_recall_at_k(y, s, frac); p = int(frac * 100)
        r[f"precision_top_{p}pct"] = k["precision"]; r[f"recall_top_{p}pct"] = k["recall"]
        r[f"lift_top_{p}pct"] = k["lift"]
    return r


def run_variant(panel: pd.DataFrame, name: str, split: str = "valid", seeds: Iterable[int] = SEEDS,
                features: list[str] | None = None, verbose: bool = True,
                importance_out: Path | None = None) -> pd.DataFrame:
    """Обучает вариант на обучающей части сплита для каждого сида и оценивает на его оценочной части.

    Каппы p99 — по обучающей части сплита (valid: только fit < 2022). Возвращает строки
    split, variant, seed, n_features, n_train, n_eval, метрики, fit_seconds.
    """
    feats = features if features is not None else variant_features(name, list(panel.columns))
    tr0, ev0 = split_frames(panel, split)
    tr0 = tr0[["h3", "month", *feats, TARGET]]; ev0 = ev0[["h3", "month", *feats, TARGET]]
    caps = cb.fit_p99_clip_caps(tr0, feats)
    tr = cb.apply_p99_clipping(tr0, caps); ev = cb.apply_p99_clipping(ev0, caps)
    assert int(tr[feats].isna().sum().sum()) == int(tr0[feats].isna().sum().sum()), "клиппинг изменил число NaN"
    y = ev[TARGET].to_numpy()
    rows, imps = [], []
    for seed in seeds:
        t0 = time.time()
        m = CatBoostClassifier(**CATBOOST_PARAMS, random_seed=seed)
        m.fit(Pool(tr[feats], tr[TARGET], feature_names=feats))
        s = m.predict_proba(ev[feats])[:, 1]
        r = {"split": split if isinstance(split, str) else split.name, "variant": name, "seed": seed,
             "n_features": len(feats), "n_train": len(tr), "n_eval": len(ev), **metrics_row(y, s),
             "fit_seconds": round(time.time() - t0, 1)}
        rows.append(r)
        imps.append(pd.Series(m.get_feature_importance(), index=feats, name=seed))
        if verbose:
            print(f"[{r['split']}] {name} ({len(feats)}) seed={seed}: roc={r['roc_auc']:.6f} pr={r['pr_auc']:.6f} "
                  f"p@1={r['precision_top_1pct']:.4f} [{r['fit_seconds']:.0f} с]", flush=True)
    if importance_out is not None:
        imp = pd.concat(imps, axis=1)
        pd.DataFrame({"feature": feats, "importance_mean": imp.mean(axis=1).values,
                      "importance_sd": imp.std(axis=1).values}) \
            .sort_values("importance_mean", ascending=False).to_csv(importance_out, index=False)
    return pd.DataFrame(rows)


def upsert_runs(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Дописывает прогоны в CSV, заменяя строки с тем же (split, variant, seed)."""
    if path.exists():
        old = pd.read_csv(path)
        key = ["split", "variant", "seed"]
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


# ---------------------------------------------------------------- сравнение

def verdict(delta: float, two_sigma: float) -> str:
    """Вердикт по соглашению проекта: |Δ|/2σ < 1 — шум, 1–1.5 — на границе, ≥ 1.5 — значимо."""
    r = abs(delta) / two_sigma if two_sigma > 0 else np.inf
    return "шум" if r < 1 else ("на границе" if r < 1.5 else "значимо")


def compare(runs: pd.DataFrame, base: str, variant: str, split: str | None = None) -> pd.DataFrame:
    """Δ = variant − base по каждой метрике: 2σ (непарная), парная 2·sd разностей, вердикты обеих."""
    if split is not None and "split" in runs.columns:
        runs = runs[runs["split"] == split]
    A = runs[runs.variant == base].set_index("seed"); B = runs[runs.variant == variant].set_index("seed")
    if A.empty or B.empty:
        raise ValueError(f"нет прогонов {base if A.empty else variant} (split={split})")
    common = A.index.intersection(B.index)
    rows = []
    for col, label in METRICS:
        n = len(common)
        d = B.loc[common, col].mean() - A.loc[common, col].mean()
        ts = 2 * np.sqrt((A.loc[common, col].var() + B.loc[common, col].var()) / n)
        tp = 2 * (B.loc[common, col] - A.loc[common, col]).std()
        rows.append({"split": split, "base": base, "variant": variant, "metric": label, "column": col, "n_seeds": n,
                     "mean_base": A.loc[common, col].mean(), "sd_base": A.loc[common, col].std(),
                     "mean_variant": B.loc[common, col].mean(), "sd_variant": B.loc[common, col].std(),
                     "delta": d, "two_sigma": ts, "ratio": abs(d) / ts if ts else np.inf,
                     "two_sd_paired": tp, "verdict": verdict(d, ts), "verdict_paired": verdict(d, tp)})
    return pd.DataFrame(rows)


def compare_folds(runs: pd.DataFrame, base: str, variant: str,
                  folds: Iterable[str] = ("valid2021", "valid")) -> pd.DataFrame:
    """Скользящая схема: Δ = среднее по фолдам Δ_f (каждая Δ_f — по 5 сидам этого фолда).

    two_sigma — соглашение проекта для среднего: 2·sqrt(Σ_f (sd_A,f² + sd_B,f²)/n_f) / F — только сидовый
    шум, уровень метрики между годами не смешивается; two_sd_paired — 2·sd парных разностей (одинаковые
    фолд и сид) после вычитания средней разности фолда, т. е. тоже без межгодового сдвига. Межгодовую
    неустойчивость показывают delta_<фолд> и same_sign_all_folds (знак Δ одинаков в каждом фолде).
    """
    folds = list(folds)
    per = {f: compare(runs, base, variant, split=f).set_index("column") for f in folds}
    r = runs[runs["split"].isin(folds)]
    A = r[r.variant == base].set_index(["split", "seed"]); B = r[r.variant == variant].set_index(["split", "seed"])
    common = A.index.intersection(B.index)
    rows = []
    for col, label in METRICS:
        fd = {f"delta_{f}": per[f].loc[col, "delta"] for f in folds}
        d = float(np.mean(list(fd.values())))
        ts = 2 * np.sqrt(sum((per[f].loc[col, "sd_base"] ** 2 + per[f].loc[col, "sd_variant"] ** 2)
                             / per[f].loc[col, "n_seeds"] for f in folds)) / len(folds)
        diff = (B.loc[common, col] - A.loc[common, col])
        diff_c = diff - diff.groupby(level="split").transform("mean")
        tp = 2 * np.sqrt((diff_c ** 2).sum() / (len(diff_c) - len(folds)))
        rows.append({"split": "+".join(folds), "base": base, "variant": variant, "metric": label, "column": col,
                     "n_runs": len(common), "delta": d, "two_sigma": ts, "two_sd_paired": tp,
                     "verdict": verdict(d, ts), "verdict_paired": verdict(d, tp), **fd,
                     "same_sign_all_folds": bool(len({np.sign(v) for v in fd.values()}) == 1)})
    return pd.DataFrame(rows)
