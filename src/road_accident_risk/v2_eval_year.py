"""Этап 7 (v2_count12): годовая постановка — панель «ячейка × год», сплиты, обучение, метрики 7.4.

Отдельный модуль, месячная схема (`v2_eval`) не меняется; отсюда переиспользуются её наборы признаков
(`variant_features`, `register_variant`), рецептура (`CATBOOST_PARAMS`, `SEEDS`), загрузка месячной панели,
клиппинг p99 (`catboost_baseline`) и вердикты (`verdict`).

Панель `parquet_output/h3_year_panel.parquet` (build_year_panel): одна строка на ячейку H3 res 9 с дорогами
(`road_length_total_m > 0`, те же 4 296 ячеек, что в месячной схеме) и календарный год 2015–2025. ДТП — те же,
что в месячной панели (`crashes_slim`, bbox Саратова, привязка `latlng_to_cell`); признаки к файлу не
приклеены — `load_year_panel` подтягивает их по h3 из месячной панели (все наборы v2 статичны по ячейке).

Сплиты (YEAR_SPLITS): y2021 — обучение 2015–2020, оценка 2021; y2022 — обучение 2015–2021, оценка 2022
(вместе — `year_folds`); test — обучение 2015–2022, оценка 2023, 2024, 2025 (`year_test`). Каппы p99 — по
обучающей части сплита. На тесте метрики считаются по каждому году (eval_period = '2023' …), их среднему
('mean') и по сумме трёх лет ('sum': факт — сумма по годам, скор — 3 × прогноз первого года, т. е. прогноз,
сделанный до 2023 г.; см. evaluate_year_predictions).

Пример:
    from road_accident_risk import v2_eval_year as vy
    panel = vy.load_year_panel("road_geo_v2_skdf")
    runs, preds = vy.run_year_variant(panel, "road_geo_v2_skdf", "test", mode="count", loss="Poisson",
                                      seeds=[42], return_predictions=True)
    base = vy.run_baseline(panel, "accidents_prev_3y", "test")
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_poisson_deviance, roc_auc_score

from road_accident_risk import catboost_baseline as cb
from road_accident_risk import v2_eval as ve
from road_accident_risk.h3_month_panel import BBOX, H3_RESOLUTION, apply_bbox_filter, latlng_to_h3_cell
from road_accident_risk.logreg_baseline import TARGET as MONTH_TARGET
from road_accident_risk.paths import (CRASHES_SLIM_PARQUET, H3_MONTH_PANEL_PARQUET, H3_MONTH_ROAD_FEATURES_PARQUET,
                                      PARQUET_DIR)

YEAR_PANEL_PARQUET = PARQUET_DIR / "h3_year_panel.parquet"
YEARS: tuple[int, ...] = tuple(range(2015, 2026))
SEEDS = ve.SEEDS
REPORTS = ve.REPORTS
FROZEN_FEATURE_SET = cb.FEATURE_SET_ROAD_GEO_V2_SKDF  # 7.0: основной набор этапа 7 (32 признака)

# Виды ДТП (accident_type) с долей ≥ 1 % в bbox; остальное — other (≈ 0.7 %).
ACCIDENT_TYPES: dict[str, str] = {
    "Столкновение": "collision",
    "Наезд на пешехода": "pedestrian",
    "Наезд на препятствие": "obstacle",
    "Падение пассажира": "passenger_fall",
    "Наезд на велосипедиста": "cyclist",
    "Наезд на стоящее ТС": "parked_vehicle",
    "Опрокидывание": "rollover",
    "Съезд с дороги": "run_off_road",
}
TYPE_COLS = [f"count_type_{v}" for v in ACCIDENT_TYPES.values()] + ["count_type_other"]
TARGET_COUNT = "count_year"
TARGET_BINARY = "had_accident_year"
SEVERITY = "severity_weight_year"
LAG_COLS = ["accidents_prev_year", "accidents_prev_3y"]
YEAR_PANEL_COLUMNS = ["h3", "year", TARGET_COUNT, TARGET_BINARY, "count_fatal_year", "count_dead_persons_year",
                      "count_injured_persons_year", SEVERITY, *TYPE_COLS, *LAG_COLS]


# ================================================================ панель

def _road_cells(path: Path = H3_MONTH_ROAD_FEATURES_PARQUET) -> np.ndarray:
    """Ячейки с дорогами — как в месячной схеме (фильтр road_length_total_m > 0; признаки статичны)."""
    first = pd.read_parquet(path, columns=["h3", "month", "road_length_total_m"],
                            filters=[("month", "==", pd.Timestamp("2015-01-01"))])
    return np.sort(first.loc[first["road_length_total_m"] > 0, "h3"].unique())


def load_crashes_in_bbox(path: Path = CRASHES_SLIM_PARQUET) -> pd.DataFrame:
    """ДТП в bbox с ячейкой H3 и годом — та же выборка и привязка, что в h3_month_panel."""
    c = pd.read_parquet(path, columns=["datetime", "lat", "lng", "accident_type", "dead_count", "injured_count",
                                       "has_fatal", "severity_weight"])
    c["datetime"] = pd.to_datetime(c["datetime"], errors="coerce")
    c["lat"] = pd.to_numeric(c["lat"], errors="coerce"); c["lng"] = pd.to_numeric(c["lng"], errors="coerce")
    c = apply_bbox_filter(c.dropna(subset=["datetime", "lat", "lng"]), BBOX)
    c["h3"] = [latlng_to_h3_cell(a, b, H3_RESOLUTION) for a, b in zip(c["lat"], c["lng"])]
    c["year"] = c["datetime"].dt.year.astype(int)
    return c


def add_year_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Лаги только из прошлых лет: prev_year = count(Y−1), prev_3y = Σ count(Y−3…Y−1); NaN без полной истории."""
    p = panel.sort_values(["h3", "year"]).reset_index(drop=True)
    g = p.groupby("h3", sort=False)[TARGET_COUNT]
    p["accidents_prev_year"] = g.shift(1).astype(float)
    p["accidents_prev_3y"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=3).sum()).astype(float)
    return p


def build_year_panel(write: bool = True, path: Path = YEAR_PANEL_PARQUET, verbose: bool = True) -> pd.DataFrame:
    """Строит панель ячейка × год (2015–2025) по ячейкам с дорогами; при write — пишет parquet."""
    c = load_crashes_in_bbox()
    c = c[c["year"].isin(YEARS)]
    cells = _road_cells()
    t = c["accident_type"].map(ACCIDENT_TYPES).fillna("other")
    agg = pd.DataFrame({
        "h3": c["h3"], "year": c["year"], TARGET_COUNT: 1, "count_fatal_year": c["has_fatal"].astype(int),
        "count_dead_persons_year": c["dead_count"].astype(int),
        "count_injured_persons_year": c["injured_count"].astype(int), SEVERITY: c["severity_weight"].astype(int),
        **{f"count_type_{k}": (t == k).astype(int) for k in [*ACCIDENT_TYPES.values(), "other"]},
    }).groupby(["h3", "year"], as_index=False).sum()
    grid = pd.MultiIndex.from_product([cells, list(YEARS)], names=["h3", "year"]).to_frame(index=False)
    p = grid.merge(agg, on=["h3", "year"], how="left")
    num = [col for col in agg.columns if col not in ("h3", "year")]
    p[num] = p[num].fillna(0).astype("int32")
    p[TARGET_BINARY] = (p[TARGET_COUNT] > 0).astype("int8")
    p = add_year_lags(p)
    p["year"] = p["year"].astype("int16")
    p = p[YEAR_PANEL_COLUMNS]
    if verbose:
        n_out = int((~c["h3"].isin(cells)).sum())
        print(f"панель {p.shape}: {p.h3.nunique()} ячеек × {p.year.nunique()} лет; ДТП в ячейках с дорогами "
              f"{int(p[TARGET_COUNT].sum())}, вне их {n_out}", flush=True)
    if write:
        p.to_parquet(path, index=False)
    return p


@lru_cache(maxsize=4)
def _cell_features(attach_key: tuple[Callable, ...]) -> pd.DataFrame:
    """Признаки ячейки: один месяц месячной панели v2_eval.load_panel (порядок колонок сохраняется)."""
    m = ve.load_panel(attach=attach_key)
    m = m[m["month"] == m["month"].min()]
    return m.drop(columns=["month", "year", "month_num", MONTH_TARGET]).reset_index(drop=True)


DEFAULT_ATTACH: tuple[Callable, ...] = (cb.attach_exposure_v6_features,)


def load_year_panel(feature_set: str | None = None, path: Path = YEAR_PANEL_PARQUET,
                    attach: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = DEFAULT_ATTACH) -> pd.DataFrame:
    """Годовая панель + все признаки ячейки (этапы 0–6a) по h3. feature_set — только проверка, что набор собирается.

    Признаки берутся из `v2_eval.load_panel(attach=...)` (кэшируется), поэтому любой набор v2_eval доступен.
    Колонки признаков — всё, что не входит в YEAR_PANEL_COLUMNS (см. feature_columns).
    """
    p = pd.read_parquet(path)
    feats = _cell_features(tuple(attach))
    out = p.merge(feats, on="h3", how="left", validate="many_to_one")
    assert out["road_length_total_m"].notna().all(), "у части ячеек годовой панели нет признаков"
    if feature_set is not None:
        year_variant_features(feature_set, out)
    return out


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Колонки признаков ячейки (в порядке месячной панели) — без таргетов, лагов и года."""
    return [c for c in panel.columns if c not in YEAR_PANEL_COLUMNS]


def year_variant_features(name: str, panel_or_cols: pd.DataFrame | list[str]) -> list[str]:
    """Признаки варианта для годовой панели.

    Имя — любой вариант v2_eval (наборы, абляции этапа 5, register_variant). Билдерам передаются только
    колонки признаков ячейки, чтобы таргет и лаги не попали в «все колонки минус исключения».
    Суффикс '+lags' добавляет accidents_prev_year и accidents_prev_3y (ориентир «структура + история»).
    """
    cols = list(panel_or_cols.columns) if isinstance(panel_or_cols, pd.DataFrame) else list(panel_or_cols)
    base_cols = [c for c in cols if c not in YEAR_PANEL_COLUMNS]
    if name.endswith("+lags"):
        return [*year_variant_features(name[: -len("+lags")], cols), *LAG_COLS]
    return ve.variant_features(name, base_cols)


# ================================================================ сплиты

@dataclass(frozen=True)
class YearSplit:
    name: str
    train_years: tuple[int, ...]   # обучение и каппы p99
    eval_years: tuple[int, ...]


YEAR_SPLITS: dict[str, YearSplit] = {
    "y2021": YearSplit("y2021", tuple(range(2015, 2021)), (2021,)),
    "y2022": YearSplit("y2022", tuple(range(2015, 2022)), (2022,)),
    "test": YearSplit("test", tuple(range(2015, 2023)), (2023, 2024, 2025)),
}
SPLIT_GROUPS: dict[str, list[str]] = {"year_folds": ["y2021", "y2022"], "year_test": ["test"],
                                      "both": ["y2021", "y2022", "test"]}


def resolve_splits(split: str) -> list[str]:
    if split in SPLIT_GROUPS:
        return SPLIT_GROUPS[split]
    return [s.strip() for s in split.split(",")]


def split_frames_year(panel: pd.DataFrame, split: str | YearSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = YEAR_SPLITS[split] if isinstance(split, str) else split
    tr = panel[panel["year"].isin(s.train_years)]
    ev = panel[panel["year"].isin(s.eval_years)]
    assert tr["year"].max() < ev["year"].min(), "обучение должно предшествовать оценке"
    return tr, ev


def split_summary_year(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, s in YEAR_SPLITS.items():
        tr, ev = split_frames_year(panel, s)
        for part, d in (("train", tr), ("eval", ev)):
            rows.append({"split": name, "part": part, "years": f"{d.year.min()}–{d.year.max()}",
                         "n_years": d.year.nunique(), "rows": len(d), "cells": d.h3.nunique(),
                         "accidents": int(d[TARGET_COUNT].sum()), "event_rows": int(d[TARGET_BINARY].sum()),
                         "event_rate": d[TARGET_BINARY].mean(), "mean_count": d[TARGET_COUNT].mean(),
                         "var_count": d[TARGET_COUNT].var()})
    return pd.DataFrame(rows)


# ================================================================ метрики 7.4

def gain_curve(y: np.ndarray, s: np.ndarray, n_points: int | None = 101) -> pd.DataFrame:
    """Кривая «доля ячеек → доля ДТП» при ранжировании по убыванию s.

    Связи (одинаковые скоры) учитываются честно: внутри группы равных скоров кривая линейна — это
    матожидание при случайном порядке внутри группы. n_points=None — узлы групп без интерполяции.
    """
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    if np.isnan(s).any():
        raise ValueError("в скорах есть NaN — заполните или исключите строки")
    n = len(y); total = y.sum()
    order = np.argsort(-s, kind="stable")
    ss, ys = s[order], y[order]
    ends = np.r_[np.nonzero(np.diff(ss))[0], n - 1]          # последний индекс каждой группы равных скоров
    cx = np.r_[0.0, (ends + 1) / n]
    cy = np.r_[0.0, np.cumsum(ys)[ends] / total] if total > 0 else np.r_[0.0, np.full(len(ends), np.nan)]
    if n_points is None:
        return pd.DataFrame({"frac_cells": cx, "frac_accidents": cy})
    grid = np.linspace(0, 1, n_points)
    return pd.DataFrame({"frac_cells": grid, "frac_accidents": np.interp(grid, cx, cy)})


def coverage_at(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    """Доля суммы y в верхних frac ячеек по s; k = max(1, int(n·frac)) (как precision_recall_at_k); связи — честно."""
    y = np.asarray(y, dtype=float); n = len(y)
    if y.sum() <= 0:
        return np.nan
    k = max(1, int(n * frac))
    g = gain_curve(y, s, n_points=None)
    return float(np.interp(k / n, g["frac_cells"], g["frac_accidents"]))


def coverages(y: np.ndarray, s: np.ndarray, fracs: Iterable[float] = (0.01, 0.05, 0.10)) -> list[float]:
    """coverage_at для нескольких долей за одну сортировку."""
    y = np.asarray(y, dtype=float); n = len(y)
    if y.sum() <= 0:
        return [np.nan for _ in fracs]
    g = gain_curve(y, s, n_points=None)
    return [float(np.interp(max(1, int(n * f)) / n, g["frac_cells"], g["frac_accidents"])) for f in fracs]


def calibration_table(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Калибровка по децилям прогноза: сумма прогноза против суммы факта (связи — по порядку строк)."""
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    r = pd.Series(pred).rank(method="first").to_numpy()
    b = np.ceil(r / len(r) * n_bins).astype(int)
    t = pd.DataFrame({"bin": b, "y": y, "pred": pred}).groupby("bin").agg(
        n=("y", "size"), sum_pred=("pred", "sum"), sum_fact=("y", "sum"), mean_pred=("pred", "mean"),
        mean_fact=("y", "mean")).reset_index()
    t["ratio"] = t["sum_pred"] / t["sum_fact"].replace(0, np.nan)
    return t


FRACS = (0.01, 0.05, 0.10)


def metrics_year(y: np.ndarray, s: np.ndarray, severity: np.ndarray | None = None,
                 is_count: bool = False) -> dict[str, float]:
    """Метрики 7.4 для одного периода: y — число ДТП ячейки, s — скор (прогноз), severity — сумма весов тяжести.

    is_count=True — s в единицах числа ДТП (Poisson/Tweedie, ориентир-счёт): считаются отклонение Пуассона
    и калибровка (сумма прогноза / сумма факта); для бинарных и ранговых скоров они NaN.
    """
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    ev = (y >= 1).astype(int)
    r: dict[str, float] = {"n_cells": len(y), "n_accidents": float(y.sum()), "event_rate": ev.mean()}
    for f in FRACS:
        p = int(round(f * 100))
        r[f"coverage_top_{p}pct"] = coverage_at(y, s, f)
        r[f"coverage_oracle_top_{p}pct"] = coverage_at(y, y, f)
        if severity is not None:
            r[f"sev_coverage_top_{p}pct"] = coverage_at(severity, s, f)
    r["spearman"] = float(spearmanr(s, y).statistic) if np.ptp(s) > 0 else np.nan
    r["roc_auc"] = roc_auc_score(ev, s) if 0 < ev.sum() < len(ev) else np.nan
    r["pr_auc"] = average_precision_score(ev, s) if ev.sum() > 0 else np.nan
    if is_count:
        r["poisson_deviance"] = mean_poisson_deviance(y, np.clip(s, 1e-9, None))
        r["calib_ratio"] = s.sum() / y.sum() if y.sum() > 0 else np.nan
        ct = calibration_table(y, s)
        r["calib_top_decile_ratio"] = float(ct["ratio"].iloc[-1])
    else:
        r["poisson_deviance"] = r["calib_ratio"] = r["calib_top_decile_ratio"] = np.nan
    return r


def bootstrap_coverage(y: np.ndarray, s: np.ndarray, fracs: Iterable[float] = FRACS, B: int = 1000,
                       seed: int = 20261019, s_base: np.ndarray | None = None, ci: float = 0.95) -> pd.DataFrame:
    """Бутстреп по ячейкам: доверительный интервал доли накрытых ДТП (перцентильный).

    s_base задан — парный бутстреп разности coverage(s) − coverage(s_base) на одних и тех же выборках ячеек.
    """
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    sb = None if s_base is None else np.asarray(s_base, dtype=float)
    rng = np.random.default_rng(seed); n = len(y); fracs = list(fracs)

    def stat(idx: np.ndarray | None) -> list[float]:
        yy = y if idx is None else y[idx]; ss = s if idx is None else s[idx]
        out = coverages(yy, ss, fracs)
        if sb is not None:
            out = [o - c for o, c in zip(out, coverages(yy, sb if idx is None else sb[idx], fracs))]
        return out

    point = stat(None)
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    a = (1 - ci) / 2
    return pd.DataFrame({"frac": fracs, "point": point, "boot_mean": boots.mean(0), "boot_sd": boots.std(0, ddof=1),
                         "ci_low": np.quantile(boots, a, axis=0), "ci_high": np.quantile(boots, 1 - a, axis=0),
                         "B": B, "seed": seed, "paired_diff": sb is not None})


def period_frame(preds: pd.DataFrame, run_id: str, split: str = "test", period: str = "sum") -> pd.DataFrame:
    """Факт и скор ячейки на периоде из таблицы прогнозов (среднее скора по сидам): h3, count_year,
    severity_weight_year, pred. period — год ('2023') или 'sum' (скор = n_лет × скор первого года)."""
    d = preds[(preds["run_id"] == run_id) & (preds["split"] == split)]
    if d.empty:
        raise ValueError(f"нет прогнозов {run_id} (split={split})")
    d = d.groupby(["h3", "year"], as_index=False).agg(**{TARGET_COUNT: (TARGET_COUNT, "first"),
                                                         SEVERITY: (SEVERITY, "first"), "pred": ("pred", "mean")})
    if period != "sum":
        return d[d["year"] == int(period)].drop(columns="year").reset_index(drop=True)
    years = sorted(d["year"].unique())
    a = d.groupby("h3")[[TARGET_COUNT, SEVERITY]].sum()
    a["pred"] = len(years) * d[d["year"] == years[0]].set_index("h3")["pred"].reindex(a.index)
    return a.reset_index()


def bootstrap_from_predictions(preds: pd.DataFrame, run_id: str, split: str = "test", period: str = "sum",
                               base_run_id: str | None = None, B: int = 1000, seed: int = 20261019,
                               target: str = TARGET_COUNT) -> pd.DataFrame:
    """Бутстреп по ячейкам доли накрытых ДТП (target = count_year или severity_weight_year) для прогона из
    таблицы прогнозов (скор — среднее по сидам); base_run_id — парная разность к базе на тех же выборках."""
    a = period_frame(preds, run_id, split, period)
    sb = None
    if base_run_id is not None:
        b = period_frame(preds, base_run_id, split, period).set_index("h3").reindex(a["h3"])
        sb = b["pred"].to_numpy()
    r = bootstrap_coverage(a[target].to_numpy(), a["pred"].to_numpy(), B=B, seed=seed, s_base=sb)
    return r.assign(split=split, eval_period=period, run_id=run_id, base_run_id=base_run_id or "",
                    target=target)


# ---------------------------------------------------------------- оценка прогнозов по периодам

def evaluate_year_predictions(df: pd.DataFrame, score_col: str = "pred", is_count: bool = False) -> pd.DataFrame:
    """df: h3, year, count_year, severity_weight_year, score_col (строки одного прогона: годы оценки).

    Возвращает строки eval_period = '<год>' для каждого года; при нескольких годах ещё 'mean' (среднее метрик
    по годам) и 'sum' (факт — сумма по годам; метрики по сумме).

    Скор периода 'sum' — прогноз, сделанный до первого года оценки: n_лет × скор первого года. Сумма погодовых
    скоров недопустима для скоров с лагами: прогноз 2024 г. использует факт 2023 г., который входит в
    суммарный таргет (утечка). Для моделей на статичных признаках скор одинаков во все годы, и это то же, что сумма.
    """
    rows = []
    for yr, d in df.groupby("year", sort=True):
        rows.append({"eval_period": str(yr), **metrics_year(d[TARGET_COUNT].to_numpy(), d[score_col].to_numpy(),
                                                            d[SEVERITY].to_numpy(), is_count)})
    if len(rows) > 1:
        m = pd.DataFrame(rows).drop(columns="eval_period").mean(numeric_only=True).to_dict()
        rows.append({"eval_period": "mean", **m})
        years = sorted(df["year"].unique())
        a = df.groupby("h3")[[TARGET_COUNT, SEVERITY]].sum()
        first = df[df["year"] == years[0]].set_index("h3")[score_col].reindex(a.index)
        rows.append({"eval_period": "sum", **metrics_year(a[TARGET_COUNT].to_numpy(),
                                                          len(years) * first.to_numpy(dtype=float),
                                                          a[SEVERITY].to_numpy(), is_count)})
    return pd.DataFrame(rows)


# ================================================================ обучение

def make_model(mode: str = "count", loss: str | None = None, class_weights: str | None = "Balanced",
               seed: int = 42, params: dict | None = None):
    """CatBoost по рецептуре этапов 1–6 (800 / 0.05 / 6). mode='binary': Logloss, auto_class_weights =
    class_weights ('Balanced' | None); mode='count': loss = 'Poisson' | 'Tweedie:variance_power=p'. params
    переопределяют любые параметры."""
    base = dict(iterations=800, learning_rate=0.05, depth=6, allow_writing_files=False, verbose=0, random_seed=seed)
    if mode == "binary":
        base.update(loss_function=loss or "Logloss", eval_metric="PRAUC")
        if class_weights not in (None, "None", "none"):
            base["auto_class_weights"] = class_weights
        base.update(params or {})
        return CatBoostClassifier(**base)
    if mode == "count":
        base.update(loss_function=loss or "Poisson")
        base.update(params or {})
        return CatBoostRegressor(**base)
    raise ValueError(f"mode: binary | count, получено {mode}")


def model_label(mode: str, loss: str | None = None, class_weights: str | None = "Balanced") -> str:
    if mode == "binary":
        return f"binary:{loss or 'Logloss'}:{class_weights or 'None'}"
    return f"count:{loss or 'Poisson'}"


def run_year_variant(panel: pd.DataFrame, name: str, split: str = "y2022", mode: str = "count",
                     loss: str | None = None, class_weights: str | None = "Balanced",
                     seeds: Iterable[int] = SEEDS, features: list[str] | None = None, params: dict | None = None,
                     verbose: bool = True, return_predictions: bool = False,
                     importance_out: Path | None = None) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Обучение на годах обучения сплита (каппы p99 — по ним же) и оценка на годах оценки, по каждому сиду.

    Строки: split, variant, model, run_id (= variant|model), seed, eval_period, n_features, n_train, метрики 7.4,
    fit_seconds. return_predictions — ещё таблица прогнозов (split, run_id, seed, h3, year, pred, факт).
    """
    feats = features if features is not None else year_variant_features(name, panel)
    target = TARGET_BINARY if mode == "binary" else TARGET_COUNT
    tr0, ev0 = split_frames_year(panel, split)
    keep = ["h3", "year", TARGET_COUNT, TARGET_BINARY, SEVERITY]
    tr0 = tr0[[*keep, *[f for f in feats if f not in keep]]]; ev0 = ev0[[*keep, *[f for f in feats if f not in keep]]]
    caps = cb.fit_p99_clip_caps(tr0, feats)
    tr = cb.apply_p99_clipping(tr0, caps); ev = cb.apply_p99_clipping(ev0, caps)
    assert int(tr[feats].isna().sum().sum()) == int(tr0[feats].isna().sum().sum()), "клиппинг изменил число NaN"
    label = model_label(mode, loss, class_weights)
    sname = split if isinstance(split, str) else split.name
    rows, preds, imps = [], [], []
    for seed in seeds:
        t0 = time.time()
        m = make_model(mode, loss, class_weights, seed, params)
        m.fit(Pool(tr[feats], tr[target].astype(int), feature_names=feats))
        s = m.predict_proba(ev[feats])[:, 1] if mode == "binary" else m.predict(ev[feats], prediction_type="Exponent")
        fit_s = round(time.time() - t0, 1)
        d = ev[["h3", "year", TARGET_COUNT, SEVERITY]].assign(pred=s)
        met = evaluate_year_predictions(d, "pred", is_count=(mode == "count"))
        for _, r in met.iterrows():
            rows.append({"split": sname, "variant": name, "model": label, "run_id": f"{name}|{label}", "seed": seed,
                         "n_features": len(feats), "n_train": len(tr), **r.to_dict(), "fit_seconds": fit_s})
        if return_predictions:
            preds.append(d.assign(split=sname, run_id=f"{name}|{label}", seed=seed))
        imps.append(pd.Series(m.get_feature_importance(), index=feats, name=seed))
        if verbose:
            key = met.set_index("eval_period").iloc[-1]
            print(f"[{sname}] {name} {label} ({len(feats)}) seed={seed} [{key.name}]: "
                  f"cov@1={key['coverage_top_1pct']:.4f} cov@10={key['coverage_top_10pct']:.4f} "
                  f"rho={key['spearman']:.4f} roc={key['roc_auc']:.4f} pr={key['pr_auc']:.4f} [{fit_s:.0f} с]",
                  flush=True)
    if importance_out is not None:
        imp = pd.concat(imps, axis=1)
        pd.DataFrame({"feature": feats, "importance_mean": imp.mean(axis=1).values,
                      "importance_sd": imp.std(axis=1).values}) \
            .sort_values("importance_mean", ascending=False).to_csv(importance_out, index=False)
    out = pd.DataFrame(rows)
    return (out, pd.concat(preds, ignore_index=True)) if return_predictions else out


# ================================================================ ориентиры без модели (7.5)

def run_baseline(panel: pd.DataFrame, score: str | Callable[[pd.DataFrame], pd.Series], split: str = "test",
                 name: str | None = None, is_count: bool = False, fillna: float | None = None,
                 return_predictions: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Ранжирование без модели: скор = колонка панели (accidents_prev_year, accidents_prev_3y,
    major_road_length_m, …) или функция panel -> Series. Те же метрики и тот же формат строк (seed = 0,
    model = 'baseline'). is_count=True — скор в единицах числа ДТП за год (например prev_3y/3), тогда
    считаются отклонение Пуассона и калибровка. NaN в скоре — ошибка, если не задан fillna.
    """
    _, ev = split_frames_year(panel, split)
    s = ev[score] if isinstance(score, str) else score(ev)
    if fillna is not None:
        s = s.fillna(fillna)
    name = name or (f"baseline:{score}" if isinstance(score, str) else "baseline:custom")
    sname = split if isinstance(split, str) else split.name
    d = ev[["h3", "year", TARGET_COUNT, SEVERITY]].assign(pred=np.asarray(s, dtype=float))
    met = evaluate_year_predictions(d, "pred", is_count=is_count)
    out = pd.DataFrame([{"split": sname, "variant": name, "model": "baseline", "run_id": f"{name}|baseline",
                         "seed": 0, "n_features": 0, "n_train": 0, **r.to_dict(), "fit_seconds": 0.0}
                        for _, r in met.iterrows()])
    if return_predictions:
        return out, d.assign(split=sname, run_id=f"{name}|baseline", seed=0)
    return out


# ================================================================ месячная модель -> годовой скор (7.2)

def monthly_split_for(split: str | YearSplit, strict: bool = True) -> ve.Split:
    """Месячный сплит, соответствующий годовому. strict=True: в обучение идут только строки, чей таргет
    (ДТП в следующем месяце) лежит в годах обучения, т. е. month < 1 декабря последнего года обучения;
    strict=False — как в v2_eval (month < 1 января первого года оценки; метка декабря = январь года оценки)."""
    s = YEAR_SPLITS[split] if isinstance(split, str) else split
    first_eval = pd.Timestamp(f"{min(s.eval_years)}-01-01")
    train_end = first_eval - pd.DateOffset(months=1) if strict else first_eval
    eval_start = first_eval - pd.DateOffset(months=1)               # строка декабря Y−1 предсказывает январь Y
    eval_end = pd.Timestamp(f"{max(s.eval_years)}-12-01")           # строка ноября последнего года — декабрь
    return ve.Split(f"month_for_{s.name}", str(train_end.date()), str(eval_start.date()), str(eval_end.date()))


def monthly_model_year_scores(name: str, split: str = "test", seeds: Iterable[int] = SEEDS,
                              month_panel: pd.DataFrame | None = None, strict: bool = True,
                              params: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    """Обучает месячную модель v2_eval (Balanced, 800/0.05/6, таргет «ДТП в следующем месяце») и сворачивает
    её прогнозы в годовой скор ячейки: score_mean / score_max по 12 месячным прогнозам, чьи целевые месяцы
    лежат в году Y. Возвращает h3, year (целевой), seed, score_mean, score_max.

    Замечание: у наборов v2 все признаки статичны, поэтому 12 прогнозов ячейки одинаковы и mean = max.
    """
    mp = month_panel if month_panel is not None else ve.load_panel(attach=DEFAULT_ATTACH)
    ms = monthly_split_for(split, strict)
    feats = ve.variant_features(name, list(mp.columns))
    tr0, ev0 = ve.split_frames(mp, ms)
    tr0 = tr0[["h3", "month", *feats, MONTH_TARGET]]; ev0 = ev0[["h3", "month", *feats]]
    caps = cb.fit_p99_clip_caps(tr0, feats)
    tr = cb.apply_p99_clipping(tr0, caps); ev = cb.apply_p99_clipping(ev0, caps)
    target_year = (ev["month"] + pd.DateOffset(months=1)).dt.year
    out = []
    for seed in seeds:
        t0 = time.time()
        p = dict(ve.CATBOOST_PARAMS); p.update(params or {})
        m = CatBoostClassifier(**p, random_seed=seed)
        m.fit(Pool(tr[feats], tr[MONTH_TARGET], feature_names=feats))
        s = m.predict_proba(ev[feats])[:, 1]
        g = pd.DataFrame({"h3": ev["h3"].to_numpy(), "year": target_year.to_numpy(), "s": s}) \
            .groupby(["h3", "year"])["s"].agg(score_mean="mean", score_max="max", n_months="size").reset_index()
        assert (g["n_months"] == 12).all(), "у ячейки-года не 12 месячных прогнозов"
        out.append(g.assign(seed=seed))
        if verbose:
            print(f"[monthly→{ms.name}] {name} seed={seed}: train<{ms.train_end}, {len(tr)} строк "
                  f"[{time.time() - t0:.0f} с]", flush=True)
    return pd.concat(out, ignore_index=True)


def run_monthly_as_year(year_panel: pd.DataFrame, name: str, split: str = "test", seeds: Iterable[int] = SEEDS,
                        agg: str = "mean", month_panel: pd.DataFrame | None = None, strict: bool = True,
                        scores: pd.DataFrame | None = None,
                        return_predictions: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Годовые метрики месячной модели (скор = agg ∈ {mean, max} месячных прогнозов). scores — готовый
    результат monthly_model_year_scores (чтобы не обучать повторно для mean и max)."""
    sc = scores if scores is not None else monthly_model_year_scores(name, split, seeds, month_panel, strict)
    _, ev = split_frames_year(year_panel, split)
    sname = split if isinstance(split, str) else split.name
    label = f"monthly:{agg}" + ("" if strict else ":nonstrict")
    rows, preds = [], []
    for seed, g in sc.groupby("seed"):
        d = ev[["h3", "year", TARGET_COUNT, SEVERITY]].merge(g[["h3", "year", f"score_{agg}"]], on=["h3", "year"],
                                                              how="left", validate="one_to_one")
        assert d[f"score_{agg}"].notna().all()
        if return_predictions:
            preds.append(d.rename(columns={f"score_{agg}": "pred"}).assign(split=sname, run_id=f"{name}|{label}",
                                                                           seed=seed))
        for _, r in evaluate_year_predictions(d, f"score_{agg}").iterrows():
            rows.append({"split": sname, "variant": name, "model": label, "run_id": f"{name}|{label}", "seed": seed,
                         "n_features": len(ve.variant_features(name, feature_columns(year_panel))),
                         "n_train": np.nan, **r.to_dict(), "fit_seconds": np.nan})
    out = pd.DataFrame(rows)
    return (out, pd.concat(preds, ignore_index=True)) if return_predictions else out


# ================================================================ сравнение

METRICS_YEAR: list[tuple[str, str, bool]] = [  # (колонка, подпись, больше = лучше)
    ("coverage_top_1pct", "доля ДТП в топ-1 %", True), ("coverage_top_5pct", "доля ДТП в топ-5 %", True),
    ("coverage_top_10pct", "доля ДТП в топ-10 %", True),
    ("sev_coverage_top_1pct", "доля тяжести в топ-1 %", True), ("sev_coverage_top_5pct", "доля тяжести в топ-5 %", True),
    ("sev_coverage_top_10pct", "доля тяжести в топ-10 %", True),
    ("spearman", "Spearman", True), ("roc_auc", "ROC-AUC (≥1 ДТП)", True), ("pr_auc", "PR-AUC (≥1 ДТП)", True),
    ("poisson_deviance", "отклонение Пуассона", False), ("calib_ratio", "Σпрогноз/Σфакт", False),
]


def upsert_year_runs(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Дописывает прогоны в CSV, заменяя строки с тем же (split, run_id, seed, eval_period)."""
    if path.exists():
        old = pd.read_csv(path, dtype={"eval_period": str})
        key = ["split", "run_id", "seed", "eval_period"]
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def _pick(runs: pd.DataFrame, run_id: str, split: str, period: str) -> pd.DataFrame:
    r = runs[(runs["run_id"] == run_id) & (runs["split"] == split) & (runs["eval_period"].astype(str) == str(period))]
    return r.set_index("seed")


def _verdict(delta: float, two_sigma: float) -> str:
    """v2_eval.verdict, но без оценки шума (1 сид, ориентир против ориентира) — «нет 2σ», а не «значимо»."""
    if not (np.isfinite(two_sigma) and two_sigma > 0):
        return "нет 2σ"
    return ve.verdict(delta, two_sigma)


def default_period(split: str) -> str:
    s = YEAR_SPLITS[split]
    return str(s.eval_years[0]) if len(s.eval_years) == 1 else "mean"


def compare_year(runs: pd.DataFrame, base: str, variant: str, split: str = "y2022",
                 period: str | None = None) -> pd.DataFrame:
    """Δ = variant − base (run_id) по метрикам METRICS_YEAR на периоде оценки: 2σ = 2·√((sd_A²+sd_B²)/n),
    парно — 2·sd разностей по общим сидам; вердикты — v2_eval.verdict. Ориентир (seed 0) против модели:
    сравнение по среднему, sd ориентира = 0. better — улучшение ли Δ с учётом направления метрики."""
    period = period or default_period(split)
    A = _pick(runs, base, split, period); B = _pick(runs, variant, split, period)
    if A.empty or B.empty:
        raise ValueError(f"нет прогонов {base if A.empty else variant} (split={split}, period={period})")
    common = A.index.intersection(B.index)
    if len(common) == 0:  # разные сиды (ориентир seed 0 против модели) — непарное сравнение
        a_idx, b_idx = A.index, B.index
    else:
        a_idx = b_idx = common
    rows = []
    for col, label, hib in METRICS_YEAR:
        a = A.loc[a_idx, col].astype(float); b = B.loc[b_idx, col].astype(float)
        if a.isna().all() or b.isna().all():
            continue
        n = max(len(a_idx), len(b_idx))
        va = a.var() if len(a) > 1 else 0.0; vb = b.var() if len(b) > 1 else 0.0
        d = b.mean() - a.mean()
        ts = 2 * np.sqrt((va + vb) / n)
        tp = 2 * (b - a).std() if len(common) > 1 else np.nan
        rows.append({"split": split, "eval_period": period, "base": base, "variant": variant, "metric": label,
                     "column": col, "higher_is_better": hib, "n_seeds": n, "mean_base": a.mean(), "sd_base": a.std(),
                     "mean_variant": b.mean(), "sd_variant": b.std(), "delta": d,
                     "better": bool((d > 0) == hib) if d != 0 else False, "two_sigma": ts,
                     "ratio": abs(d) / ts if ts else np.inf, "two_sd_paired": tp,
                     "verdict": _verdict(d, ts), "verdict_paired": _verdict(d, tp)})
    return pd.DataFrame(rows)


def compare_folds_year(runs: pd.DataFrame, base: str, variant: str,
                       folds: Iterable[str] = ("y2021", "y2022")) -> pd.DataFrame:
    """Скользящая схема (как v2_eval.compare_folds): Δ — среднее по фолдам; 2σ среднего — сидовый;
    парная — 2·sd разностей после вычитания средней разности фолда; same_sign_all_folds — знак Δ одинаков."""
    folds = list(folds)
    per = {f: compare_year(runs, base, variant, split=f).set_index("column") for f in folds}
    rows = []
    for col, label, hib in METRICS_YEAR:
        if not all(col in per[f].index for f in folds):
            continue
        fd = {f"delta_{f}": per[f].loc[col, "delta"] for f in folds}
        d = float(np.mean(list(fd.values())))
        ts = 2 * np.sqrt(sum((np.nan_to_num(per[f].loc[col, "sd_base"]) ** 2
                              + np.nan_to_num(per[f].loc[col, "sd_variant"]) ** 2) / per[f].loc[col, "n_seeds"]
                             for f in folds)) / len(folds)
        diffs = []
        for f in folds:
            A = _pick(runs, base, f, default_period(f)); B = _pick(runs, variant, f, default_period(f))
            c = A.index.intersection(B.index)
            if len(c):
                dd = (B.loc[c, col] - A.loc[c, col]).astype(float)
                diffs.append(dd - dd.mean())
        dc = pd.concat(diffs) if diffs else pd.Series(dtype=float)
        tp = 2 * np.sqrt((dc ** 2).sum() / (len(dc) - len(folds))) if len(dc) > len(folds) else np.nan
        rows.append({"split": "+".join(folds), "base": base, "variant": variant, "metric": label, "column": col,
                     "higher_is_better": hib, "delta": d, "better": bool((d > 0) == hib) if d != 0 else False,
                     "two_sigma": ts, "two_sd_paired": tp, "verdict": _verdict(d, ts),
                     "verdict_paired": _verdict(d, tp), **fd,
                     "same_sign_all_folds": bool(len({np.sign(v) for v in fd.values()}) == 1)})
    return pd.DataFrame(rows)
