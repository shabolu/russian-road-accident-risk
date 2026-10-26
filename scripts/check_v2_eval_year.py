"""Самопроверки этапа 7 шаг 1 (src/road_accident_risk/v2_eval_year.py).

1) метрики на синтетике с известным ответом; 2) сумма ДТП годовой панели = сумма месячного accident_count;
3) нет утечки: лаги только из прошлых лет, каппы p99 только по обучающим годам, обучение раньше оценки;
4) наборы признаков на годовой панели совпадают с месячными (билдеры не подхватили таргет/лаги).

Запуск:
    PYTHONPATH=src venv/bin/python scripts/check_v2_eval_year.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk import catboost_baseline as cb  # noqa: E402
from road_accident_risk import v2_eval as ve  # noqa: E402
from road_accident_risk import v2_eval_year as vy  # noqa: E402
from road_accident_risk.paths import H3_MONTH_PANEL_PARQUET  # noqa: E402

ok = lambda msg: print(f"OK  {msg}", flush=True)  # noqa: E731


def check_synthetic() -> None:
    # 100 ячеек, ДТП только в 10; идеальный скор = y
    y = np.zeros(100); y[:10] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert np.isclose(vy.coverage_at(y, y, 0.01), 10 / 55)              # k=1: лучшая ячейка — 10 из 55
    assert np.isclose(vy.coverage_at(y, y, 0.05), 40 / 55)              # k=5: 10+9+8+7+6
    assert np.isclose(vy.coverage_at(y, y, 0.10), 1.0)
    assert np.isclose(vy.coverage_at(y, -y, 0.10), 0.0)                 # обратный скор: ни одного ДТП
    # постоянный скор: связи -> матожидание случайного порядка = k/n
    assert np.isclose(vy.coverage_at(y, np.ones(100), 0.05), 0.05)
    # связи на границе: 2 ячейки со скором 1 (y=10 и y=0), k=1 -> половина от 10
    assert np.isclose(vy.coverage_at(np.array([10., 0, 0, 0]), np.array([1., 1, 0, 0]), 0.25), 0.5)
    g = vy.gain_curve(y, y)
    assert g.frac_cells.iloc[0] == 0 and g.frac_accidents.iloc[0] == 0 and np.isclose(g.frac_accidents.iloc[-1], 1)
    assert (np.diff(g.frac_accidents) >= -1e-12).all()
    assert np.allclose(vy.coverages(y, np.ones(100) + y * 0.1, (0.01, 0.05, 0.1)),
                       [vy.coverage_at(y, np.ones(100) + y * 0.1, f) for f in (0.01, 0.05, 0.1)])
    ok("доля накрытых ДТП и кривая gain на синтетике (вкл. связи)")
    m = vy.metrics_year(y, y, severity=2 * y, is_count=True)
    assert np.isclose(m["spearman"], 1) and np.isclose(m["roc_auc"], 1) and np.isclose(m["pr_auc"], 1)
    assert np.isclose(m["sev_coverage_top_5pct"], 40 / 55) and np.isclose(m["coverage_oracle_top_5pct"], 40 / 55)
    assert np.isclose(m["calib_ratio"], 1)
    assert np.isclose(vy.metrics_year(y, y + 0.0, is_count=True)["poisson_deviance"], 0, atol=1e-6)
    assert np.isnan(vy.metrics_year(y, y, is_count=False)["poisson_deviance"])
    rng = np.random.default_rng(0); lam = rng.gamma(1.0, 1.0, 5000); yy = rng.poisson(lam).astype(float)
    dev_true = vy.metrics_year(yy, lam, is_count=True)["poisson_deviance"]
    dev_flat = vy.metrics_year(yy, np.full(5000, yy.mean()), is_count=True)["poisson_deviance"]
    assert dev_true < dev_flat
    ct = vy.calibration_table(yy, lam)
    assert len(ct) == 10 and np.isclose(ct.sum_fact.sum(), yy.sum()) and np.isclose(ct.sum_pred.sum(), lam.sum())
    assert ct.mean_fact.is_monotonic_increasing or ct.mean_fact.diff().min() > -0.2
    ok(f"Spearman/ROC/PR/калибровка/отклонение Пуассона (истинный λ {dev_true:.3f} < константа {dev_flat:.3f})")
    b1 = vy.bootstrap_coverage(yy, lam, B=200, seed=1); b2 = vy.bootstrap_coverage(yy, lam, B=200, seed=1)
    assert b1.equals(b2) and ((b1.ci_low <= b1.point) & (b1.point <= b1.ci_high)).all()
    bd = vy.bootstrap_coverage(yy, lam, B=200, seed=1, s_base=lam)
    assert np.allclose(bd[["point", "ci_low", "ci_high"]], 0)
    ok("бутстреп: воспроизводим при одном сиде, ДИ содержит точку, парная разность скора с собой = 0")
    # оценка по периодам: 'mean' = среднее годов, 'sum' — по сумме
    df = pd.DataFrame({"h3": np.tile(np.arange(100), 2), "year": np.repeat([2023, 2024], 100),
                       "count_year": np.r_[y, y[::-1]], "severity_weight_year": 1.0, "pred": np.r_[y, y[::-1]]})
    e = vy.evaluate_year_predictions(df, is_count=True).set_index("eval_period")
    assert list(e.index) == ["2023", "2024", "mean", "sum"]
    assert np.isclose(e.loc["mean", "coverage_top_5pct"], e.loc[["2023", "2024"], "coverage_top_5pct"].mean())
    df2 = df.copy(); df2.loc[df2.year == 2024, "pred"] = df2.loc[df2.year == 2024, "count_year"] * 100  # «утечка» в 2024
    e2 = vy.evaluate_year_predictions(df2, is_count=False).set_index("eval_period")
    e1 = vy.evaluate_year_predictions(df, is_count=False).set_index("eval_period")
    assert np.allclose(e1.loc["sum"].astype(float), e2.loc["sum"].astype(float), equal_nan=True)
    ok("evaluate_year_predictions: периоды по годам, mean, sum; скор 'sum' = прогноз первого года (2024 не влияет)")


def check_panel(p: pd.DataFrame) -> None:
    assert p.h3.nunique() == 4296 and p.year.nunique() == 11 and len(p) == 4296 * 11
    assert not p.duplicated(["h3", "year"]).any()
    mp = pd.read_parquet(H3_MONTH_PANEL_PARQUET, columns=["h3", "month", "accident_count"])
    mp = mp[mp.h3.isin(p.h3.unique())]
    m = mp.groupby(mp.month.dt.year).accident_count.sum()
    yv = p.groupby("year").count_year.sum()
    assert (m.reindex(yv.index).to_numpy() == yv.to_numpy()).all(), pd.concat([m, yv], axis=1)
    cm = mp.assign(year=mp.month.dt.year).groupby(["h3", "year"]).accident_count.sum()
    cy = p.set_index(["h3", "year"]).count_year
    assert (cm.reindex(cy.index).fillna(0).to_numpy() == cy.to_numpy()).all()
    ok(f"ДТП годовой панели = месячный accident_count по каждой ячейке-году ({int(yv.sum())} за 2015–2025)")
    assert (p[vy.TYPE_COLS].sum(axis=1) == p.count_year).all()
    assert (p.count_fatal_year <= p.count_year).all() and (p.severity_weight_year >= p.count_year).all()
    assert (p.had_accident_year == (p.count_year > 0)).all()
    ok("виды ДТП в сумме = count_year; fatal ≤ count; тяжесть ≥ count; had = count>0")


def check_leakage(p: pd.DataFrame) -> None:
    w = p.pivot(index="h3", columns="year", values="count_year")
    py = p.pivot(index="h3", columns="year", values="accidents_prev_year")
    p3 = p.pivot(index="h3", columns="year", values="accidents_prev_3y")
    assert py[2015].isna().all() and p3[[2015, 2016, 2017]].isna().all().all()
    for y in range(2016, 2026):
        assert (py[y] == w[y - 1]).all()
    for y in range(2018, 2026):
        assert (p3[y] == w[[y - 1, y - 2, y - 3]].sum(axis=1)).all()
    ok("лаги: prev_year = count(Y−1), prev_3y = Σ count(Y−3…Y−1); NaN без полной истории (2015 / 2015–2017)")
    # лаги не зависят от текущего и будущих лет: переписываем count 2023+ и пересчитываем
    q = p[["h3", "year", "count_year"]].copy(); q.loc[q.year >= 2023, "count_year"] = 999
    q = vy.add_year_lags(q).set_index(["h3", "year"]); r = p.set_index(["h3", "year"])
    for col in vy.LAG_COLS:
        assert q.loc[q.index.get_level_values(1) <= 2023, col].fillna(-1).equals(
            r.loc[r.index.get_level_values(1) <= 2023, col].fillna(-1))
    ok("лаги до 2023 включительно не меняются при подмене счётов 2023+")


def check_caps_and_splits(panel: pd.DataFrame) -> None:
    feats = vy.year_variant_features("road_geo_v2_skdf", panel)
    for name, s in vy.YEAR_SPLITS.items():
        tr, ev = vy.split_frames_year(panel, name)
        assert set(tr.year) == set(s.train_years) and set(ev.year) == set(s.eval_years) and tr.year.max() < ev.year.min()
        c1 = cb.fit_p99_clip_caps(tr, feats)
        # подмена признаков в годах оценки не меняет капп
        pert = panel.copy(); pert.loc[pert.year.isin(s.eval_years), feats] = 1e12
        c2 = cb.fit_p99_clip_caps(vy.split_frames_year(pert, name)[0], feats)
        assert c1 == c2
    ok("сплиты: обучение раньше оценки; каппы p99 зависят только от обучающих лет")
    mp = ve.load_panel(attach=vy.DEFAULT_ATTACH)
    sets = ["road_geo_baseline", "road_geo_v2_prune", "road_geo_v2_junctions", "road_geo_v2_tags",
            "road_geo_v2_geometry", "road_geo_v2_skdf", "road_geo_v2_exposure_a"]
    for s in sets:
        a = vy.year_variant_features(s, panel); b = ve.variant_features(s, list(mp.columns))
        assert a == b, (s, set(a) ^ set(b))
        assert not set(a) & set(vy.YEAR_PANEL_COLUMNS)
    assert vy.year_variant_features("road_geo_v2_skdf+lags", panel)[-2:] == vy.LAG_COLS
    ok(f"наборы на годовой панели = месячные (7 наборов, тот же порядок); '+lags' добавляет {vy.LAG_COLS}")
    # признаки ячейки годовой панели совпадают со значениями в любой месяц месячной панели
    f = vy.feature_columns(panel)
    one = panel[panel.year == 2020].set_index("h3")[f].sort_index()
    mm = mp[mp.month == "2020-06-01"].set_index("h3")[f].sort_index()
    pd.testing.assert_frame_equal(one, mm, check_dtype=False)
    ok(f"{len(f)} колонок признаков ячейки = значения месячной панели (2020-06)")
    ms = vy.monthly_split_for("test")
    assert (ms.train_end, ms.eval_start, ms.eval_end) == ("2022-12-01", "2022-12-01", "2025-12-01")
    ok("месячный сплит для годового теста (strict): train < 2022-12, оценка 2022-12…2025-11 (таргет 2023-01…2025-12)")


if __name__ == "__main__":
    check_synthetic()
    p = pd.read_parquet(vy.YEAR_PANEL_PARQUET)
    check_panel(p)
    check_leakage(p)
    check_caps_and_splits(vy.load_year_panel())
    print("все проверки пройдены")
