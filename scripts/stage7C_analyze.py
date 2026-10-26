"""Этап 7, шаг 3: анализ — модель против истории, «структура + история», EB, очаги ОДМ, цепочка наборов, выбросы.

Подкоманды (результаты — CSV в docs/analysis/features/stage7_count12/history_eb/, рисунки — в figures/):
    compare   метрики всех вариантов (модель, история, структура + история, кросс-фит, EB), выбор k по фолдам,
              бутстреп разностей к модели, очаги ОДМ (7.7)
    chain     цепочка наборов на годовом таргете (7.6) и согласие знаков folds/test против месячного таргета
    outliers  топ-15 ячеек по факту и прогнозу, недо- и переоценка (тест 2023–2025)
    plots     рисунки 1–7 (+ копии в summary/)

Нужны прогоны scripts/stage7C_run.py (history, crossfit, chain, month) и reports/v2_eval_year_stage7A_models_preds.parquet.

    PYTHONPATH=src venv/bin/python scripts/stage7C_analyze.py compare chain outliers plots
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from road_accident_risk import v2_eval as ve  # noqa: E402
from road_accident_risk import v2_eval_year as vy  # noqa: E402
import stage7C_run as R  # noqa: E402
from stage7B_descriptive import bootstrap_odm_recall, odm_flag, odm_recall  # noqa: E402

OUT = ROOT / "docs/analysis/features/stage7_count12/history_eb"
FIG = OUT / "figures"
SUMMARY = ROOT / "docs/analysis/features/summary"
FRACS = (0.01, 0.05, 0.10)
FOLDS = ("y2021", "y2022")
B = 1000
BOOT_SEED = 20261019

M_RUN = f"{R.SKDF}|{R.MODEL}"
RUNS = {  # вариант -> run_id в таблицах прогнозов
    "model": M_RUN,
    "sh2": f"{R.SKDF}+lags|{R.MODEL}",
    "sh3": f"{R.SKDF}+lags_all|{R.MODEL}",
    "cf": f"{R.SKDF}|{R.MODEL}|crossfit",
}
K_MAIN = (0.1, 0.15, 0.2)
K_GRID = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf)
LABEL = {
    "model": "модель (Tweedie 1.3, структура)",
    "h1": "история: прошлый год", "h3": "история: 3 года", "hall": "история: все прошлые годы",
    "sh2": "структура + история (2 лага)", "sh3": "структура + история (3 лага)",
    "cf": "модель, кросс-фит по ячейкам",
}


# ================================================================ данные

def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = R.add_history(vy.load_year_panel())
    panel["odm_hot"] = odm_flag(panel).to_numpy()
    a = pd.read_parquet(vy.REPORTS / "v2_eval_year_stage7A_models_preds.parquet")
    a = a[a.run_id == M_RUN]
    h = pd.read_parquet(R.HIST_CSV.with_name(R.HIST_CSV.stem + "_preds.parquet"))
    return panel, pd.concat([a, h], ignore_index=True)


def eval_frame(panel: pd.DataFrame, split: str) -> pd.DataFrame:
    _, ev = vy.split_frames_year(panel, split)
    return ev.sort_values(["year", "h3"]).reset_index(drop=True)


def n_train_years(split: str) -> int:
    return len(vy.YEAR_SPLITS[split].train_years)


def seed_scores(preds: pd.DataFrame, run_id: str, split: str, ef: pd.DataFrame) -> pd.DataFrame:
    """Прогнозы прогона по сидам, выровненные по строкам ef (колонки — сиды)."""
    d = preds[(preds.run_id == run_id) & (preds.split == split)]
    w = d.pivot_table(index=["h3", "year"], columns="seed", values="pred")
    w = w.reindex(pd.MultiIndex.from_frame(ef[["h3", "year"]]))
    assert w.notna().all().all(), f"нет прогнозов {run_id} {split}"
    return w.reset_index(drop=True)


def history_score(ef: pd.DataFrame, which: str, split: str) -> np.ndarray:
    """История в единицах ДТП в год (заморожена на конец обучения сплита)."""
    if which == "h1":
        return ef["accidents_prev_year"].to_numpy(float)
    if which == "h3":
        return ef["accidents_prev_3y"].to_numpy(float) / 3
    if which == "hall":
        return ef[R.LAG_ALL].to_numpy(float)
    raise ValueError(which)


def eb_score(n_pred: np.ndarray, ef: pd.DataFrame, window: str, k: float, split: str) -> np.ndarray:
    """N_EB = w·N_pred + (1−w)·N_obs/T, w = 1/(1 + k·N_pred·T); всё в ДТП в год."""
    T = 3 if window == "3y" else n_train_years(split)
    rate = history_score(ef, "h3" if window == "3y" else "hall", split)
    if np.isinf(k):
        return rate.copy()
    w = 1.0 / (1.0 + k * n_pred * T)
    return w * n_pred + (1 - w) * rate


def eb_name(window: str, k: float, base: str = "model") -> str:
    kk = "inf" if np.isinf(k) else f"{k:g}"
    return f"eb{'cf' if base == 'cf' else ''}_{window}_k{kk}"


def eb_label(name: str) -> str:
    _, w, k = name.split("_")
    cf = ", кросс-фит" if name.startswith("ebcf") else ""
    return f"EB ({'3 года' if w == '3y' else 'все годы'}, k = {k[1:]}{cf})"


def label(v: str) -> str:
    return LABEL.get(v) or (eb_label(v) if v.startswith("eb") else v)


def variant_scores(panel, preds, split, ks=K_MAIN) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """ef и словарь вариант -> DataFrame скоров по сидам (у ориентиров один столбец seed 0)."""
    ef = eval_frame(panel, split)
    sc: dict[str, pd.DataFrame] = {}
    for v, rid in RUNS.items():
        sc[v] = seed_scores(preds, rid, split, ef)
    for h in ("h1", "h3", "hall"):
        sc[h] = pd.DataFrame({0: history_score(ef, h, split)})
    for base in ("model", "cf"):
        for window in ("3y", "all"):
            for k in ks:
                sc[eb_name(window, k, base)] = sc[base].apply(lambda c: pd.Series(eb_score(c.to_numpy(), ef, window, k,
                                                                                            split)))
    for v, s in sc.items():
        assert not s.isna().any().any(), f"NaN в скоре {v} ({split})"
    return ef, sc


def metrics_runs(ef: pd.DataFrame, sc: dict[str, pd.DataFrame], split: str) -> pd.DataFrame:
    rows = []
    base = ef[["h3", "year", vy.TARGET_COUNT, vy.SEVERITY]]
    for v, s in sc.items():
        for seed in s.columns:
            d = base.assign(pred=s[seed].to_numpy())
            for _, r in vy.evaluate_year_predictions(d, "pred", is_count=True).iterrows():
                rows.append({"split": split, "run_id": v, "variant": v, "seed": seed, **r.to_dict()})
    return pd.DataFrame(rows)


# ================================================================ бутстреп (по ячейкам, общая выборка для всех периодов)

def period_arrays(ef: pd.DataFrame, s: np.ndarray, period: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(h3, y, s) периода; 'sum' — факт за все годы оценки, скор — n × скор первого года."""
    d = ef[["h3", "year", vy.TARGET_COUNT]].assign(s=s)
    if period != "sum":
        d = d[d.year == int(period)].sort_values("h3")
        return d.h3.to_numpy(), d[vy.TARGET_COUNT].to_numpy(float), d.s.to_numpy(float)
    yrs = sorted(d.year.unique())
    a = d.groupby("h3").agg(y=(vy.TARGET_COUNT, "sum")).sort_index()
    first = d[d.year == yrs[0]].set_index("h3")["s"].reindex(a.index)
    return a.index.to_numpy(), a.y.to_numpy(float), len(yrs) * first.to_numpy(float)


def boot_paired(items: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]], fracs=FRACS, B=B,
                seed=BOOT_SEED, target_items: list[np.ndarray] | None = None) -> pd.DataFrame:
    """items — список (y, s, s_base) по периодам с одинаковым порядком ячеек; статистика — среднее по периодам
    coverage(s) [− coverage(s_base)]. Одна выборка ячеек на все периоды (как bootstrap_odm_recall)."""
    rng = np.random.default_rng(seed); n = len(items[0][0])

    def stat(idx):
        out = []
        for y, s, sb in items:
            yy, ss = (y, s) if idx is None else (y[idx], s[idx])
            c = np.array(vy.coverages(yy, ss, fracs))
            if sb is not None:
                c = c - np.array(vy.coverages(yy, sb if idx is None else sb[idx], fracs))
            out.append(c)
        return np.mean(out, axis=0)
    point = stat(None)
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    return pd.DataFrame({"frac": fracs, "point": point, "ci_low": np.quantile(boots, 0.025, 0),
                         "ci_high": np.quantile(boots, 0.975, 0), "boot_sd": boots.std(0, ddof=1)})


# ================================================================ compare

def cmd_compare() -> None:
    t0 = time.time()
    panel, preds = load_all()
    efs, scs, runs = {}, {}, []
    for split in (*FOLDS, "test"):
        ef, sc = variant_scores(panel, preds, split, ks=K_GRID)
        efs[split], scs[split] = ef, sc
        runs.append(metrics_runs(ef, sc, split))
        print(f"[{split}] метрики {len(sc)} вариантов [{time.time() - t0:.0f} с]", flush=True)
    runs = pd.concat(runs, ignore_index=True)
    runs.to_csv(OUT / "runs_all_variants.csv", index=False)

    # проверка: модель = шаг 2A
    a = pd.read_csv(vy.REPORTS / "v2_eval_year_stage7A_models.csv", dtype={"eval_period": str})
    a = a[(a.run_id == M_RUN)].set_index(["split", "seed", "eval_period"])["coverage_top_10pct"]
    m = runs[runs.variant == "model"].set_index(["split", "seed", "eval_period"])["coverage_top_10pct"]
    print(f"модель против 7A: max |Δ cov@10| = {np.abs(m - a.reindex(m.index)).max():.2e}", flush=True)

    # ---- сводка: среднее по сидам; фолды — по году фолда, тест — sum и по годам
    key = ["coverage_top_1pct", "coverage_top_5pct", "coverage_top_10pct", "spearman", "roc_auc", "pr_auc",
           "poisson_deviance", "calib_ratio"]
    sm = runs.groupby(["variant", "split", "eval_period"])[key].mean().reset_index()
    sm.to_csv(OUT / "metrics_summary.csv", index=False)

    # ---- выбор k по фолдам (среднее топ-5 и топ-10 по двум фолдам; основной набор 0.1 / 0.15 / 0.2)
    fold_rows = sm[sm.split.isin(FOLDS) & (sm.eval_period == sm.split.str[1:])]
    kt = []
    for base in ("model", "cf"):
        for window in ("3y", "all"):
            for k in K_GRID:
                v = eb_name(window, k, base)
                f = fold_rows[fold_rows.variant == v]
                t = sm[(sm.variant == v) & (sm.split == "test") & (sm.eval_period == "sum")].iloc[0]
                kt.append({"base": base, "window": window, "k": k, "variant": v,
                           "folds_top1": f.coverage_top_1pct.mean(), "folds_top5": f.coverage_top_5pct.mean(),
                           "folds_top10": f.coverage_top_10pct.mean(), "folds_spearman": f.spearman.mean(),
                           "folds_deviance": f.poisson_deviance.mean(),
                           "folds_score": (f.coverage_top_5pct.mean() + f.coverage_top_10pct.mean()) / 2,
                           "test_top1": t.coverage_top_1pct, "test_top5": t.coverage_top_5pct,
                           "test_top10": t.coverage_top_10pct, "test_spearman": t.spearman,
                           "test_deviance_mean_years": sm[(sm.variant == v) & (sm.split == "test")
                                                          & (sm.eval_period == "mean")].poisson_deviance.iloc[0]})
    kt = pd.DataFrame(kt); kt.to_csv(OUT / "eb_k_grid.csv", index=False)
    chosen = {}
    for (base, window), g in kt[kt.k.isin(K_MAIN)].groupby(["base", "window"]):
        chosen[(base, window)] = float(g.sort_values(["folds_score", "folds_top10"], ascending=False).k.iloc[0])
    print("выбор k по фолдам:", chosen, flush=True)
    pd.DataFrame([{"base": b, "window": w, "k": k} for (b, w), k in chosen.items()]).to_csv(OUT / "eb_k_chosen.csv",
                                                                                            index=False)

    # ---- варианты главной таблицы
    main = ["model", "h1", "h3", "hall", "sh2", "sh3", "cf",
            *[eb_name(w, k) for w in ("3y", "all") for k in K_MAIN],
            *[eb_name(w, chosen[("cf", w)], "cf") for w in ("3y", "all")]]
    main = list(dict.fromkeys(main))

    # ---- сидовые сравнения к модели (folds: среднее Δ, 2σ, знак в обоих фолдах; тест sum)
    cmp = []
    for v in main[1:]:
        f = vy.compare_folds_year(runs, "model", v)
        t = vy.compare_year(runs, "model", v, split="test", period="sum")
        for _, r in f.iterrows():
            tt = t[t.column == r.column]
            cmp.append({"variant": v, "label": label(v), "metric": r.metric, "column": r.column,
                        "folds_delta": r.delta, "folds_2sigma": r.two_sigma, "folds_verdict": r.verdict,
                        "delta_y2021": r.delta_y2021, "delta_y2022": r.delta_y2022,
                        "same_sign_folds": r.same_sign_all_folds,
                        "test_delta": tt.delta.iloc[0] if len(tt) else np.nan,
                        "test_2sigma": tt.two_sigma.iloc[0] if len(tt) else np.nan,
                        "test_verdict": tt.verdict.iloc[0] if len(tt) else ""})
    pd.DataFrame(cmp).to_csv(OUT / "cmp_to_model_seeds.csv", index=False)

    # ---- бутстреп: уровни и парные разности к модели (скор — среднее по сидам)
    mean_s = {sp: {v: scs[sp][v].mean(axis=1).to_numpy() for v in main} for sp in scs}
    bt = []
    for v in main:
        for per, splits, pers in (("folds", FOLDS, [s[1:] for s in FOLDS]), ("test_sum", ("test",), ["sum"]),
                                  *[(f"test_{y}", ("test",), [str(y)]) for y in (2023, 2024, 2025)],
                                  *[(sp, (sp,), [sp[1:]]) for sp in FOLDS]):
            lv, df_ = [], []
            for sp, pr in zip(splits, pers):
                _, y, s = period_arrays(efs[sp], mean_s[sp][v], pr)
                _, _, sb = period_arrays(efs[sp], mean_s[sp]["model"], pr)
                lv.append((y, s, None)); df_.append((y, s, sb))
            r1 = boot_paired(lv).assign(kind="level")
            parts = [r1]
            if v != "model":
                parts.append(boot_paired(df_).assign(kind="diff_to_model"))
            for r in parts:
                bt.append(r.assign(variant=v, label=label(v), period=per))
        print(f"бутстреп {v} [{time.time() - t0:.0f} с]", flush=True)
    # потолок
    for per, splits, pers in (("folds", FOLDS, [s[1:] for s in FOLDS]), ("test_sum", ("test",), ["sum"])):
        items = []
        for sp, pr in zip(splits, pers):
            _, y, _ = period_arrays(efs[sp], mean_s[sp]["model"], pr); items.append((y, y, None))
        bt.append(boot_paired(items).assign(kind="level", variant="oracle", label="потолок (факт)", period=per))
    bt = pd.concat(bt, ignore_index=True); bt.to_csv(OUT / "bootstrap.csv", index=False)

    # ---- очаги ОДМ (7.7): recall в топ-q %, folds (2021, 2022) и тест (2023–2025, скор на конец 2022 г.)
    odm_rows, odm_boot = [], []
    hot_tab = panel[["h3", "year", "odm_hot"]]
    for v in [*main, "oracle"]:
        for per, splits in (("folds", FOLDS), ("test", ("test",))):
            hb, sb_, mb = {}, {}, {}
            for sp in splits:
                ef = efs[sp]
                for yr in sorted(ef.year.unique()):
                    m_ = (ef.year == yr).to_numpy()
                    order = np.argsort(ef.h3.to_numpy()[m_])
                    h = ef.odm_hot.to_numpy()[m_][order].astype(float)
                    s = (ef[vy.TARGET_COUNT].to_numpy()[m_] if v == "oracle" else mean_s[sp][v][m_])[order]
                    hb[yr], sb_[yr], mb[yr] = h, s.astype(float), mean_s[sp]["model"][m_][order]
                    odm_rows.append({"variant": v, "period": per, "year": yr, "n_hot": int(h.sum()),
                                     **{f"recall_top_{int(f * 100)}pct": c for f, c in zip(FRACS, vy.coverages(h, s, FRACS))}})
            r = bootstrap_odm_recall(hb, sb_, FRACS, B=B).assign(kind="level")
            odm_boot.append(r.assign(variant=v, period=per))
            if v not in ("model", "oracle"):
                odm_boot.append(_odm_paired(hb, sb_, mb).assign(kind="diff_to_model", variant=v, period=per))
    od = pd.DataFrame(odm_rows); od.to_csv(OUT / "odm_recall_by_year.csv", index=False)
    pd.concat(odm_boot, ignore_index=True).assign(label=lambda d: d.variant.map(
        lambda x: "потолок (факт года)" if x == "oracle" else label(x))).to_csv(OUT / "odm_recall_pooled.csv", index=False)
    # сверка с функцией агента B
    ef = efs["test"]; d23 = ef[ef.year == 2023]
    chk = odm_recall(pd.Series(mean_s["test"]["model"][(ef.year == 2023).to_numpy()], index=d23.h3.to_numpy()), 2023)
    mine = od[(od.variant == "model") & (od.year == 2023)].iloc[0]
    assert abs(chk["recall_top_5pct"] - mine.recall_top_5pct) < 1e-12, (chk, mine)
    print(f"ОДМ: сверка с odm_recall (2023, модель) OK; [{time.time() - t0:.0f} с]", flush=True)


def _odm_paired(hb, sb, mb, fracs=FRACS, B=B, seed=BOOT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed); years = list(hb); n = len(hb[years[0]])

    def stat(idx):
        num = np.zeros(len(fracs)); den = 0.0
        for yr in years:
            h = hb[yr] if idx is None else hb[yr][idx]
            if h.sum() == 0:
                continue
            s = sb[yr] if idx is None else sb[yr][idx]; m = mb[yr] if idx is None else mb[yr][idx]
            num += (np.array(vy.coverages(h, s, fracs)) - np.array(vy.coverages(h, m, fracs))) * h.sum()
            den += h.sum()
        return num / den
    point = stat(None); boots = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    return pd.DataFrame({"frac": fracs, "point": point, "ci_low": np.quantile(boots, 0.025, 0),
                         "ci_high": np.quantile(boots, 0.975, 0), "B": B})


# ================================================================ parts: сочетание против каждой части

PARTS_PAIRS = [("hall", "model"), ("hall", "sh3"), ("hall", "ebcf_all_k0.2"), ("cf", "ebcf_all_k0.2"),
               ("sh3", "ebcf_all_k0.2"), ("hall", "cf"), ("h3", "ebcf_3y_k0.2"), ("cf", "ebcf_3y_k0.2")]


def cmd_parts() -> None:
    """Парный бутстреп (B = 1000) для пар без модели в роли базы: EB против истории и против структуры."""
    panel, preds = load_all()
    efs, ms = {}, {}
    for sp in (*FOLDS, "test"):
        ef, sc = variant_scores(panel, preds, sp)
        efs[sp] = ef; ms[sp] = {v: s.mean(axis=1).to_numpy() for v, s in sc.items()}
    rows = []
    for base, var in PARTS_PAIRS:
        for per, splits, pers in (("folds", FOLDS, [s[1:] for s in FOLDS]), ("test_sum", ("test",), ["sum"])):
            items = []
            for sp, pr in zip(splits, pers):
                _, y, s = period_arrays(efs[sp], ms[sp][var], pr); _, _, sb = period_arrays(efs[sp], ms[sp][base], pr)
                items.append((y, s, sb))
            rows.append(boot_paired(items).assign(base=base, variant=var, base_label=label(base), label=label(var),
                                                  period=per))
    d = pd.concat(rows, ignore_index=True); d.to_csv(OUT / "bootstrap_parts.csv", index=False)
    d["v"] = d.point.round(3).astype(str) + " [" + d.ci_low.round(3).astype(str) + "; " + d.ci_high.round(3).astype(str) + "]"
    print(d.pivot_table(index=["variant", "base"], columns=["period", "frac"], values="v", aggfunc="first").to_string())


# ================================================================ chain (7.6)

SET_LABEL = {
    "road_geo_baseline": "Бейзлайн (61)", "road_geo_v2_prune": "1 чистка (33)", "road_geo_v2_junctions": "2 перекрёстки (32)",
    "road_geo_v2_tags": "3 теги OSM (33)", "road_geo_v2_geometry": "4 геометрия (30)", "road_geo_v2_skdf": "5 СКДФ (32)",
    "road_geo_v2_exposure_a": "6a экспозиция (34)",
}
CHAIN_PAIRS = list(zip(R.CHAIN[:-1], R.CHAIN[1:]))
DIAG_PAIRS = [("road_geo_v2_geometry", "geometry_plus_old"), (R.SKDF, "g_E1_E2")]
STAGE5_PAIRS = [("road_geo_v2_geometry", v) for v in R.STAGE5]
STAGE6A_PAIRS = [*[(R.SKDF, v) for v in ["e1_center_two", "e1_center_poi", "e2_cell", "e2_r300", "e2_r500", "e3_build",
                                         "e4_old", "e4_new"]],
                 *[("e3_build", v) for v in ["g_E3_E1", "g_E3_E2", "g_E3_E4", "e4_routes"]],
                 *[("road_geo_v2_exposure_a", v) for v in ["eart", "elevels", "emed"]]]
YEAR_METRICS = ["coverage_top_1pct", "coverage_top_5pct", "coverage_top_10pct", "spearman", "roc_auc", "pr_auc",
                "poisson_deviance"]
MONTH_METRICS = ["roc_auc", "pr_auc", "precision_top_1pct", "recall_top_5pct", "recall_top_10pct"]
METRIC_RU = {"coverage_top_1pct": "топ-1 %", "coverage_top_5pct": "топ-5 %", "coverage_top_10pct": "топ-10 %",
             "spearman": "Spearman", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "poisson_deviance": "отклонение",
             "precision_top_1pct": "precision@1 %", "recall_top_5pct": "recall@5 %", "recall_top_10pct": "recall@10 %"}


def pair_group(base: str, var: str) -> str:
    if (base, var) in CHAIN_PAIRS:
        return "цепочка"
    if (base, var) in DIAG_PAIRS:
        return "диагностика"
    return "этап 5" if (base, var) in STAGE5_PAIRS else "этап 6a"


def month_runs() -> pd.DataFrame:
    """Месячные прогоны (5 сидов): фолды valid2021 / valid и тест из всех доступных отчётов."""
    parts = []
    for f in ("v2_eval_stage6_0.csv", "v2_eval_stage6a.csv", "v2_eval_stage7C_month.csv"):
        if (vy.REPORTS / f).exists():
            parts.append(pd.read_csv(vy.REPORTS / f))
    for f in ("road_geo_v2_prune", "road_geo_v2_junctions", "road_geo_v2_tags", "road_geo_v2_geometry", "road_geo_v2_skdf"):
        d = pd.read_csv(vy.REPORTS / f"{f}_seed_runs.csv"); parts.append(d.assign(split="test"))
    m = pd.concat(parts, ignore_index=True)
    return m.drop_duplicates(["split", "variant", "seed"], keep="first")


def month_delta(m: pd.DataFrame, base: str, var: str, split: str, col: str) -> tuple[float, float]:
    a = m[(m.split == split) & (m.variant == base)].set_index("seed")[col]
    b = m[(m.split == split) & (m.variant == var)].set_index("seed")[col]
    if a.empty or b.empty:
        return np.nan, np.nan
    return float(b.mean() - a.mean()), float(2 * np.sqrt((a.var() + b.var()) / max(len(a), len(b))))


def cmd_chain() -> None:
    runs = pd.read_csv(R.CHAIN_CSV, dtype={"eval_period": str})
    preds = pd.read_parquet(R.CHAIN_CSV.with_name(R.CHAIN_CSV.stem + "_preds.parquet"))
    rid = lambda v: f"{v}|{R.MODEL}"  # noqa: E731
    runs["variant_name"] = runs["variant"]

    # ---- таблица наборов
    tab = []
    for v in runs.variant.unique():
        r = runs[runs.variant == v]
        row = {"variant": v, "label": SET_LABEL.get(v, v), "n_features": int(r.n_features.iloc[0])}
        for sp in (*FOLDS, "test"):
            per = "sum" if sp == "test" else sp[1:]
            q = r[(r.split == sp) & (r.eval_period == per)]
            for c in YEAR_METRICS:
                row[f"{sp}_{c}"] = q[c].mean(); row[f"{sp}_{c}_sd"] = q[c].std()
        for c in YEAR_METRICS:
            row[f"folds_{c}"] = np.mean([row[f"{sp}_{c}"] for sp in FOLDS])
        tab.append(row)
    tab = pd.DataFrame(tab); tab.to_csv(OUT / "chain_table.csv", index=False)

    # проверка: skdf в цепочке = модель шага 2A
    a = pd.read_csv(vy.REPORTS / "v2_eval_year_stage7A_models.csv", dtype={"eval_period": str})
    a = a[a.run_id == M_RUN].set_index(["split", "seed", "eval_period"])["coverage_top_10pct"]
    b = runs[runs.run_id == rid(R.SKDF)].set_index(["split", "seed", "eval_period"])["coverage_top_10pct"]
    print(f"skdf в цепочке против 7A: max |Δ cov@10| = {np.abs(b - a.reindex(b.index)).max():.2e}", flush=True)

    # ---- пары: годовые Δ (сиды) + бутстреп по ячейкам на тесте и фолдах, месячные Δ
    mon = month_runs()
    rows = []
    t0 = time.time()
    for base, var in [*CHAIN_PAIRS, *DIAG_PAIRS, *STAGE5_PAIRS, *STAGE6A_PAIRS]:
        f = vy.compare_folds_year(runs, rid(base), rid(var)).set_index("column")
        t = vy.compare_year(runs, rid(base), rid(var), split="test", period="sum").set_index("column")
        bt = {}
        for per, splits, pers in (("folds", FOLDS, [s[1:] for s in FOLDS]), ("test", ("test",), ["sum"])):
            items = []
            for sp, pr in zip(splits, pers):
                fa = vy.period_frame(preds, rid(var), sp, pr).sort_values("h3")
                fb = vy.period_frame(preds, rid(base), sp, pr).set_index("h3").reindex(fa.h3)
                items.append((fa[vy.TARGET_COUNT].to_numpy(float), fa.pred.to_numpy(), fb.pred.to_numpy()))
            bt[per] = boot_paired(items, B=500).set_index("frac")
        for c in YEAR_METRICS:
            row = {"group": pair_group(base, var), "base": base, "variant": var, "scale": "год", "metric": c,
                   "folds_delta": f.loc[c, "delta"], "folds_2sigma": f.loc[c, "two_sigma"],
                   "folds_verdict": f.loc[c, "verdict"], "delta_y2021": f.loc[c, "delta_y2021"],
                   "delta_y2022": f.loc[c, "delta_y2022"], "same_sign_folds": f.loc[c, "same_sign_all_folds"],
                   "test_delta": t.loc[c, "delta"], "test_2sigma": t.loc[c, "two_sigma"],
                   "test_verdict": t.loc[c, "verdict"], "higher_is_better": c != "poisson_deviance"}
            if c.startswith("coverage"):
                fr = {"coverage_top_1pct": 0.01, "coverage_top_5pct": 0.05, "coverage_top_10pct": 0.10}[c]
                for per in ("folds", "test"):
                    row[f"{per}_boot_lo"] = bt[per].loc[fr, "ci_low"]; row[f"{per}_boot_hi"] = bt[per].loc[fr, "ci_high"]
            rows.append(row)
        for c in MONTH_METRICS:
            d21, _ = month_delta(mon, base, var, "valid2021", c); d22, _ = month_delta(mon, base, var, "valid", c)
            dt, ts = month_delta(mon, base, var, "test", c)
            rows.append({"group": pair_group(base, var), "base": base, "variant": var, "scale": "месяц", "metric": c,
                         "folds_delta": np.nanmean([d21, d22]) if np.isfinite([d21, d22]).all() else np.nan,
                         "delta_y2021": d21, "delta_y2022": d22, "test_delta": dt, "test_2sigma": ts,
                         "test_verdict": ve.verdict(dt, ts) if np.isfinite(dt) and ts > 0 else "",
                         "higher_is_better": True})
        print(f"пара {base} -> {var} [{time.time() - t0:.0f} с]", flush=True)
    pr = pd.DataFrame(rows)
    pr["same_sign_folds_test"] = np.where(pr.folds_delta.notna() & pr.test_delta.notna(),
                                          np.sign(pr.folds_delta) == np.sign(pr.test_delta), np.nan)
    pr.to_csv(OUT / "chain_pairs.csv", index=False)

    # ---- согласие знаков: одинаковый набор пар у месяца и года
    ag = []
    for scale in ("год", "месяц"):
        q = pr[(pr.scale == scale) & pr.same_sign_folds_test.notna()]
        for grp, g0 in (("этапы 5–6a (21 пара, как в 6.0 и 6a)", q[q.group.isin(["этап 5", "этап 6a"])
                                                                       | ((q.base == "road_geo_v2_geometry") & (q.variant == R.SKDF))
                                                                       | ((q.base == "road_geo_v2_tags") & (q.variant == "road_geo_v2_geometry"))]),
                        ("все пары с месячными фолдами", q)):
            for c, g in g0.groupby("metric"):
                ag.append({"scale": scale, "set": grp, "metric": c, "metric_ru": METRIC_RU[c], "pairs": len(g),
                           "same_sign": int(g.same_sign_folds_test.astype(bool).sum()),
                           "spearman_folds_test": g[["folds_delta", "test_delta"]].corr("spearman").iloc[0, 1]})
    ag = pd.DataFrame(ag); ag.to_csv(OUT / "sign_agreement.csv", index=False)
    print(ag.to_string(index=False), flush=True)


# ================================================================ outliers

def cmd_outliers(n: int = 15) -> None:
    panel, preds = load_all()
    ef = eval_frame(panel, "test")
    s = seed_scores(preds, M_RUN, "test", ef).mean(axis=1).to_numpy()
    _, y, mu = period_arrays(ef, s, "sum")
    cells = np.sort(ef.h3.unique())
    feats = panel[panel.year == 2023].set_index("h3").reindex(cells)
    hist = panel[panel.year <= 2022].groupby("h3")[vy.TARGET_COUNT].sum().reindex(cells)
    hot = panel[panel.year >= 2023].groupby("h3")["odm_hot"].sum().reindex(cells)
    k = 0.15
    d = pd.DataFrame({"h3": cells, "fact_2023_2025": y, "pred_2023_2025": mu, "resid": y - mu,
                      "z": (y - mu) / np.sqrt(mu + k * mu ** 2), "hist_2015_2022": hist.to_numpy(),
                      "hist_rate_x3": hist.to_numpy() / 8 * 3, "odm_years_2023_2025": hot.to_numpy()})
    for c in ("major_road_length_m", "link_road_length_m", "bridge_length_m", "tunnel_length_m", "motorway_junction_count",
              "traffic_signal_count", "junction_count", "road_length_total_m", "lanes_max", "crossing_signalized_count",
              "crossing_unsignalized_count"):
        d[c] = feats[c].to_numpy()
    d["rank_fact"] = d.fact_2023_2025.rank(ascending=False, method="min").astype(int)
    d["rank_pred"] = d.pred_2023_2025.rank(ascending=False, method="min").astype(int)
    ll = [R.h3.cell_to_latlng(c) for c in cells]
    d["lat"] = [a for a, _ in ll]; d["lng"] = [b for _, b in ll]
    names = pd.read_parquet(ROOT / "docs/analysis/features/stage7_count12/descriptive/_edge_length_by_cell_name.parquet")
    nm = names[names.name != "(без названия)"].sort_values("len_m", ascending=False)
    top = nm.groupby("h3").apply(lambda g: ", ".join(g.groupby("name").len_m.sum().sort_values(ascending=False)
                                                     .index[:3]), include_groups=False)
    hw = names.groupby("h3").apply(lambda g: g.groupby("highway").len_m.sum().idxmax(), include_groups=False)
    d["streets"] = d.h3.map(top).fillna("—"); d["main_highway"] = d.h3.map(hw).fillna("—")
    d["interchange"] = (d.link_road_length_m > 0) | (d.motorway_junction_count > 0)
    d["bridge"] = d.bridge_length_m > 0
    d.to_csv(OUT / "outliers_all_cells.csv", index=False)
    sel = []
    for kind, q in (("топ по факту", d.nlargest(n, "fact_2023_2025")), ("топ по прогнозу", d.nlargest(n, "pred_2023_2025")),
                    ("недооценка (факт − прогноз)", d.nlargest(n, "resid")),
                    ("переоценка (прогноз − факт)", d.nsmallest(n, "resid"))):
        sel.append(q.assign(kind=kind))
    sel = pd.concat(sel, ignore_index=True); sel.to_csv(OUT / "outliers_top15.csv", index=False)
    # характеристика групп
    grp = []
    for kind, q in sel.groupby("kind", sort=False):
        grp.append({"kind": kind, "n": len(q), "fact_mean": q.fact_2023_2025.mean(), "pred_mean": q.pred_2023_2025.mean(),
                    "hist_x3_mean": q.hist_rate_x3.mean(), "interchange_share": q.interchange.mean(),
                    "bridge_share": q.bridge.mean(), "signals_mean": q.traffic_signal_count.mean(),
                    "major_m_mean": q.major_road_length_m.mean(), "odm_years_mean": q.odm_years_2023_2025.mean()})
    grp.append({"kind": "все ячейки", "n": len(d), "fact_mean": d.fact_2023_2025.mean(), "pred_mean": d.pred_2023_2025.mean(),
                "hist_x3_mean": d.hist_rate_x3.mean(), "interchange_share": d.interchange.mean(),
                "bridge_share": d.bridge.mean(), "signals_mean": d.traffic_signal_count.mean(),
                "major_m_mean": d.major_road_length_m.mean(), "odm_years_mean": d.odm_years_2023_2025.mean()})
    pd.DataFrame(grp).to_csv(OUT / "outliers_groups.csv", index=False)
    print(pd.DataFrame(grp).round(2).to_string(index=False), flush=True)
    print(f"пересечение топ-{n} по факту и по прогнозу: "
          f"{len(set(d.nlargest(n, 'fact_2023_2025').h3) & set(d.nlargest(n, 'pred_2023_2025').h3))}", flush=True)


# ================================================================ plots

C = {"model": "#1f3a93", "hall": "#eb6834", "h3": "#eda100", "h1": "#c9a26b", "sh2": "#7fb3a2", "sh3": "#1baf7a",
     "cf": "#8a8985", "eb": "#a33ea1", "ebcf": "#e34948", "oracle": "#111111"}


def _save(fig, name: str, data: pd.DataFrame | None = None, summary_name: str | None = None) -> None:
    import shutil
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    if data is not None:
        data.to_csv(FIG / f"{name}.csv", index=False)
    if summary_name:
        shutil.copy(FIG / f"{name}.png", SUMMARY / f"{summary_name}.png")
        if data is not None:
            shutil.copy(FIG / f"{name}.csv", SUMMARY / f"{summary_name}.csv")


def _color(v: str) -> str:
    if v.startswith("ebcf"):
        return C["ebcf"]
    if v.startswith("eb"):
        return C["eb"]
    return C.get(v, "#555555")


def cmd_plots() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    kch = pd.read_csv(OUT / "eb_k_chosen.csv")
    k_cf = kch[(kch.base == "cf") & (kch.window == "all")].k.iloc[0]; EBCF = eb_name("all", k_cf, "cf")
    k_m = kch[(kch.base == "model") & (kch.window == "all")].k.iloc[0]; EBM = eb_name("all", k_m)

    # ---- 1. кривые gain на тесте
    panel, preds = load_all()
    ef, sc = variant_scores(panel, preds, "test")
    show = ["model", "hall", "sh3", EBCF, "cf", "h3"]
    rows = []
    for v in [*show, "oracle"]:
        s = (ef[vy.TARGET_COUNT].to_numpy(float) if v == "oracle" else sc[v].mean(axis=1).to_numpy())
        if v == "oracle":
            _, y, _ = period_arrays(ef, s, "sum"); ss = y
        else:
            _, y, ss = period_arrays(ef, s, "sum")
        g = vy.gain_curve(y, ss, n_points=201)
        rows.append(g.assign(variant=v, label="потолок (факт 2023–2025)" if v == "oracle" else label(v)))
    gc = pd.concat(rows, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, xmax in zip(axes, (1.0, 0.2)):
        for v in [*show, "oracle"]:
            g = gc[gc.variant == v]
            ax.plot(g.frac_cells * 100, g.frac_accidents * 100, color=_color(v), lw=2.2 if v in ("model", EBCF) else 1.4,
                    ls="--" if v in ("oracle", "cf", "h3") else "-", label=g.label.iloc[0])
        ax.set_xlim(0, xmax * 100); ax.set_xlabel("доля ячеек, отобранных по прогнозу, %")
        ax.set_ylabel("доля ДТП 2023–2025, %"); ax.grid(alpha=0.3)
        if xmax < 1:
            ax.set_ylim(0, 92)
            for q in (1, 5, 10):
                ax.axvline(q, color="#bbbbbb", lw=0.8, zorder=0)
    axes[0].plot([0, 100], [0, 100], color="#cccccc", lw=0.8)
    axes[1].legend(fontsize=7.5, loc="lower right")
    axes[0].set_title("вся сеть"); axes[1].set_title("верхние 20 % ячеек")
    fig.suptitle("Тест 2023–2025: прогноз и история на конец 2022 г. (3 979 ДТП, 4 296 ячеек; модели — среднее 5 сидов)")
    _save(fig, "fig1_gain_curves_test", gc, "stage7_gain_curves_test"); plt.close(fig)

    # ---- 2. Δ к модели с бутстреп-ДИ
    bt = pd.read_csv(OUT / "bootstrap.csv")
    order = ["h1", "h3", "hall", "sh2", "sh3", "cf", eb_name("3y", 0.1), EBM, eb_name("3y", 0.2, "cf"), EBCF]
    d2 = bt[(bt.kind == "diff_to_model") & bt.variant.isin(order) & bt.period.isin(["folds", "test_sum"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    yy = {v: i for i, v in enumerate(order[::-1])}
    for ax, fr in zip(axes, FRACS):
        for per, off, mk, col in (("folds", 0.17, "s", "#888888"), ("test_sum", -0.17, "o", None)):
            q = d2[(d2.frac == fr) & (d2.period == per)]
            for _, r in q.iterrows():
                c = col or _color(r.variant)
                ax.errorbar(r.point * 100, yy[r.variant] + off, xerr=[[(r.point - r.ci_low) * 100], [(r.ci_high - r.point) * 100]],
                            fmt=mk, color=c, ms=4.5, capsize=2.5, lw=1.2)
        ax.axvline(0, color="black", lw=0.8); ax.set_title(f"топ-{int(fr * 100)} %"); ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("Δ доли ДТП к модели, п.п.")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([label(v) for v in order[::-1]], fontsize=8)
    axes[0].plot([], [], "s", color="#888888", label="фолды 2021–2022 (среднее)"); axes[0].plot([], [], "o", color="#333333", label="тест 2023–2025")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Разность доли накрытых ДТП к структурной модели (Tweedie 1.3): точка и 95 % ДИ, бутстреп по ячейкам (B = 1000)")
    _save(fig, "fig2_delta_to_model", d2, "stage7_delta_to_model"); plt.close(fig)

    # ---- 3. очаги ОДМ
    od = pd.read_csv(OUT / "odm_recall_pooled.csv")
    vs = ["h1", "h3", "hall", "cf", "model", "sh3", EBM, EBCF]
    d3 = od[(od.kind == "level") & od.variant.isin(vs)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    w = 0.1
    for ax, per, ttl in zip(axes, ("folds", "test"), ("фолды: очаги 2021 и 2022 (228)", "тест: очаги 2023–2025 (235), скор на конец 2022 г.")):
        for j, v in enumerate(vs):
            q = d3[(d3.period == per) & (d3.variant == v)].sort_values("frac")
            x = np.arange(3) + (j - (len(vs) - 1) / 2) * w
            ax.bar(x, q.point * 100, w, color=_color(v), label=label(v))
            ax.errorbar(x, q.point * 100, yerr=[(q.point - q.ci_low) * 100, (q.ci_high - q.point) * 100], fmt="none",
                        ecolor="#333333", lw=0.8, capsize=1.5)
        orc = od[(od.kind == "level") & (od.variant == "oracle") & (od.period == per)].sort_values("frac")
        ax.scatter(np.arange(3), orc.point * 100, marker="_", s=900, color="black", zorder=4,
                   label="потолок (факт года)")
        ax.set_xticks(range(3)); ax.set_xticklabels(["топ-1 %", "топ-5 %", "топ-10 %"]); ax.set_title(ttl, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("доля очагов ОДМ в верхних q % ячеек, %")
    fig.legend(*axes[1].get_legend_handles_labels(), fontsize=7.5, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Очаги ОДМ (≥ 3 ДТП одного вида или ≥ 5 за год): recall в верхних q %, 95 % ДИ — бутстреп по ячейкам")
    _save(fig, "fig3_odm_recall", d3, "stage7_odm_recall"); plt.close(fig)

    # ---- 4. цепочка наборов: год против месяца
    runs = pd.read_csv(R.CHAIN_CSV, dtype={"eval_period": str})
    mon = month_runs()
    chain = R.CHAIN
    labs = ["Бейзлайн\n(61)", "1\nчистка\n(33)", "2\nперекр.\n(32)", "3\nтеги\n(33)", "4\nгеом.\n(30)",
            "5\nСКДФ\n(32)", "6a\nэксп.\n(34)"]
    panels = [("месяц", "test", "roc_auc", "месячный таргет, тест: ROC-AUC"), ("месяц", "test", "pr_auc", "месячный таргет, тест: PR-AUC"),
              ("год", "folds", "coverage_top_10pct", "годовой, фолды 2021+2022: доля ДТП в топ-10 %"),
              ("год", "test", "coverage_top_5pct", "годовой, тест: доля ДТП в топ-5 %"),
              ("год", "test", "coverage_top_10pct", "годовой, тест: доля ДТП в топ-10 %"),
              ("год", "test", "spearman", "годовой, тест: Spearman")]
    rows4 = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5)); axes = axes.ravel()
    cp = pd.read_csv(OUT / "chain_pairs.csv")
    for ax, (scale, per, met, ttl) in zip(axes, panels):
        means = []
        for i, v in enumerate(chain):
            if scale == "месяц":
                vals = mon[(mon.split == "test") & (mon.variant == v)].sort_values("seed")[met].to_numpy()
            elif per == "folds":
                q = runs[(runs.variant == v) & runs.split.isin(FOLDS) & (runs.eval_period == runs.split.str[1:])]
                vals = q.groupby("seed")[met].mean().to_numpy()
            else:
                vals = runs[(runs.variant == v) & (runs.split == "test") & (runs.eval_period == "sum")][met].to_numpy()
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=12, color="#9ecae1", zorder=2)
            means.append((vals.mean(), vals.std(ddof=1)))
            rows4.extend({"scale": scale, "period": per, "metric": met, "variant": v, "seed_idx": j, "value": x}
                         for j, x in enumerate(vals))
        m_, s_ = np.array(means).T
        ax.errorbar(range(len(chain)), m_, yerr=s_, fmt="o-", color="black", capsize=3, zorder=3)
        if scale == "год" and met.startswith("coverage"):
            pp = cp[(cp.scale == "год") & (cp.group == "цепочка") & (cp.metric == met)]
            pf = "folds" if per == "folds" else "test"
            half = float(((pp[f"{pf}_boot_hi"] - pp[f"{pf}_boot_lo"]) / 2).median())
            c0 = m_[chain.index(R.SKDF)]
            ax.axhspan(c0 - half, c0 + half, color="#f2d0a4", alpha=0.45, zorder=0,
                       )
            ax.text(0.02, 0.03, f"полоса ±{half * 100:.1f} п.п.", transform=ax.transAxes, fontsize=7.5, color="#9a6a2a")
        ax.set_xticks(range(len(chain))); ax.set_xticklabels(labs, fontsize=7.5); ax.set_title(ttl, fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Цепочка наборов признаков: месячный бинарный таргет (Balanced) против годового счётного (Tweedie 1.3)\n"
                 "точки — 5 сидов, чёрное — среднее ± ст. откл.; полоса вокруг СКДФ — медианная полуширина 95 % ДИ парной разности соседних наборов (бутстреп по ячейкам): она в несколько раз шире сидового разброса",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig4_chain_year_vs_month", pd.DataFrame(rows4), "stage7_chain_year_vs_month"); plt.close(fig)

    # ---- 5. согласие знаков folds/test
    ag = pd.read_csv(OUT / "sign_agreement.csv")
    ag = ag[ag.set.str.startswith("этапы")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    mm = [("месяц", m) for m in MONTH_METRICS] + [("год", m) for m in YEAR_METRICS]
    xs, hs, cs, tl = [], [], [], []
    for i, (sc_, m) in enumerate(mm):
        r = ag[(ag.scale == sc_) & (ag.metric == m)]
        if r.empty:
            continue
        r = r.iloc[0]; x = i + (0.6 if sc_ == "год" else 0)
        ax.bar(x, r.same_sign / r.pairs * 100, 0.7, color="#9ecae1" if sc_ == "месяц" else "#1f3a93")
        ax.text(x, r.same_sign / r.pairs * 100 + 1.5, f"{r.same_sign}/{r.pairs}", ha="center", fontsize=7.5)
        xs.append(x); tl.append(METRIC_RU[m])
    ax.axhline(50, color="#e34948", ls="--", lw=1); ax.text(xs[4] + 0.55, 46, "уровень\nслучайности", color="#e34948", fontsize=7, ha="center", va="top")
    ax.set_xticks(xs); ax.set_xticklabels(tl, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("пары с одинаковым знаком Δ, %"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#9ecae1", label="месячный таргет (Balanced)"),
                       Patch(color="#1f3a93", label="годовой таргет (Tweedie 1.3)")], fontsize=8, loc="upper left")
    ax.set_title("Знак Δ на фолдах 2021–2022 и на тесте 2023–2025: 21 пара наборов этапов 4–6a", fontsize=10)
    _save(fig, "fig5_sign_agreement", ag); plt.close(fig)

    # ---- 6. карта выбросов
    d = pd.read_csv(OUT / "outliers_all_cells.csv"); top = pd.read_csv(OUT / "outliers_top15.csv")
    polys = [[(lng, lat) for lat, lng in R.h3.cell_to_boundary(c)] for c in d.h3]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    lim = 8
    for ax, col, ttl, cmap, vmin, vmax in ((axes[0], "resid", "факт − прогноз, ДТП за 2023–2025", "RdBu_r", -lim, lim),
                                           (axes[1], "fact_2023_2025", "факт 2023–2025 (контуры — топ-15 по прогнозу)", "magma_r", 0, 25)):
        pc = PolyCollection(polys, array=d[col].clip(vmin, vmax).to_numpy(), cmap=cmap, edgecolors="none")
        pc.set_clim(vmin, vmax); ax.add_collection(pc)
        ax.autoscale_view(); ax.set_aspect(1 / np.cos(np.radians(51.55))); ax.set_title(ttl, fontsize=9.5)
        fig.colorbar(pc, ax=ax, shrink=0.7)
        ax.set_xlabel("долгота"); ax.set_ylabel("широта")
    for kind, mk, c in (("недооценка (факт − прогноз)", "^", "#b2182b"), ("переоценка (прогноз − факт)", "v", "#2166ac")):
        q = top[top.kind == kind]
        axes[0].scatter(q.lng, q.lat, marker=mk, s=40, facecolors="none", edgecolors=c, lw=1.2, label=f"{kind}: топ-15")
    axes[0].legend(fontsize=7, loc="lower right")
    q = top[top.kind == "топ по прогнозу"]
    for c_ in q.h3:
        b = [(lng, lat) for lat, lng in R.h3.cell_to_boundary(c_)]
        axes[1].add_patch(plt.Polygon(b, fill=False, edgecolor="#1baf7a", lw=1.2))
    q = top[top.kind == "топ по факту"]
    axes[1].scatter(q.lng, q.lat, s=8, color="#1f3a93", label="топ-15 по факту (точки)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Тест 2023–2025: где модель (обучена на 2015–2022) недо- и переоценивает; шкала Δ обрезана ±8")
    _save(fig, "fig6_outliers_map", top); plt.close(fig)

    # ---- 7. EB: метрика против k
    kt = pd.read_csv(OUT / "eb_k_grid.csv")
    kx = {k: i for i, k in enumerate(K_GRID)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, (col, ttl) in zip(axes, (("folds_top10", "фолды 2021–2022: доля ДТП в топ-10 %"), ("test_top10", "тест 2023–2025: доля ДТП в топ-10 %"))):
        for (base, window), g in kt.groupby(["base", "window"]):
            g = g.sort_values("k")
            ax.plot([kx[k] for k in g.k], g[col] * 100, marker="o", ms=3.5,
                    color=C["ebcf"] if base == "cf" else C["eb"], ls="-" if window == "all" else ":",
                    label=f"EB на {'кросс-фите' if base == 'cf' else 'обычной модели'}, окно {'все годы' if window == 'all' else '3 года'}")
        ax.axvspan(kx[0.1] - 0.3, kx[0.2] + 0.3, color="#eeeeee", zorder=0)
        ax.set_xticks(range(len(K_GRID))); ax.set_xticklabels(["0\n(модель)", *[f"{k:g}" for k in K_GRID[1:-1]], "∞\n(история)"], fontsize=7.5)
        ax.set_xlabel("k (сверхдисперсия); серая полоса — сетка выбора 0.1–0.2"); ax.set_title(ttl, fontsize=9.5); ax.grid(alpha=0.3)
    axes[0].set_ylabel("%"); axes[1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Эмпирический байесовский ориентир: w = 1/(1 + k·N_pred·T), N_EB = w·N_pred + (1 − w)·N_obs/T")
    _save(fig, "fig7_eb_k_curve", kt); plt.close(fig)
    print("рисунки готовы", flush=True)


# ================================================================ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="+", choices=["compare", "parts", "chain", "outliers", "plots"])
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for c in a.cmd:
        t0 = time.time()
        globals()[f"cmd_{c}"]()
        print(f"== {c}: {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
