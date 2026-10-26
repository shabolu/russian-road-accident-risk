"""Этап 7, шаг D: достижимый предел предсказуемости (noise ceiling) — симуляция оракула.

Оракул знает истинный годовой риск λ_i каждой ячейки, но не случайность конкретных лет. Ранжируя ячейки по λ,
он накрывает в среднем столько ДТП в топ-q %, сколько позволяет пуассоновский шум. Это предел для любого
честного прогноза (в отличие от «потолка по факту», который знает и случайность).

Шаги:
1. Гамма-пуассоновская (NB) иерархия по суммам ячеек за окно лет: одна гамма (MLE и метод моментов), смеси
   2 и 3 гамм (MLE, выбор по BIC). Апостериор ячейки — смесь Gamma(α_j + S_i, β_j + n) с весами компонент.
2. Симуляция: λ_i ~ апостериор (основная), λ_i = S_i / n (граница сверху), λ_i = E[λ_i | S_i] (граница снизу);
   ДТП по видам — независимые Пуассоны λ_i·c_t·p_ik (p_i ~ Dirichlet(счёты видов + A_TYPES·доли города)), сумма
   по видам — Poisson(λ_i·c_t); вариант со сверхдисперсией — множитель Gamma(k, k) на ячейко-год, k = 53 (φ ≈ 1.08).
3. Оракул: ранжирование по λ → vy.coverages (связи честно), Spearman(λ, Y), recall очагов ОДМ (полный критерий,
   pooled по годам теста); «реализованный потолок» — ранжирование по самим Y.
4. Нестационарность: λ по 2015–2022 с поправкой уровня / без; λ по всем годам с поправкой; λ только по 2023–2025.
5. Самопроверка на фолдах 2021 и 2022.

Запуск: PYTHONPATH=src venv/bin/python scripts/stage7D_noise_ceiling.py [--nsim 1000] [--plots-only]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize, special  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk import v2_eval_year as vy  # noqa: E402
from road_accident_risk.paths import PROJECT_ROOT  # noqa: E402

SEED = 20261026
OUT = PROJECT_ROOT / "docs/analysis/features/stage7_count12/noise_ceiling"
FIG = OUT / "figures"
HEB = PROJECT_ROOT / "docs/analysis/features/stage7_count12/history_eb"
SUMMARY_FIG = PROJECT_ROOT / "docs/analysis/features/summary/stage7_noise_ceiling.png"
YEARS = list(vy.YEARS)
TEST_YEARS = [2023, 2024, 2025]
FRACS = (0.01, 0.05, 0.10)
TYPE_COLS = vy.TYPE_COLS                    # 8 видов + other
ODM_TYPES = len(TYPE_COLS) - 1              # критерий «≥ 3 одного вида» — без other
A_TYPES = 5.0                               # сила априора долей видов (в «ДТП»)
K_WITHIN = 53.0                             # NB-параметр внутри ячейки (descriptive: φ = 1.08, k ≈ 53)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8985"; INK = "#3d3d3a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})

VARIANTS = [  # (ключ в history_eb, подпись, цвет)
    ("ebcf_all_k0.2", "EB кросс-фит", C[0]),
    ("model", "Tweedie 1.3", C[1]),
    ("hall", "история «все годы»", C[2]),
    ("cf", "кросс-фит по ячейкам", C[3]),
]


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  ->", FIG / name, flush=True)


# ================================================================ данные

def load() -> tuple[np.ndarray, np.ndarray, list[str]]:
    p = pd.read_parquet(vy.YEAR_PANEL_PARQUET).sort_values(["h3", "year"])
    cells = p["h3"].unique().tolist()
    Y = p.pivot(index="h3", columns="year", values=vy.TARGET_COUNT).loc[cells, YEARS].to_numpy(float)
    T = np.stack([p.pivot(index="h3", columns="year", values=c).loc[cells, YEARS].to_numpy(float)
                  for c in TYPE_COLS], axis=2)                       # ячейка × год × вид
    assert np.allclose(T.sum(2), Y)
    return Y, T, cells


def odm_hot(types: np.ndarray) -> np.ndarray:
    """types: (..., 9) → флаг очага ОДМ: ≥ 3 одного вида (без other) или ≥ 5 любых."""
    return ((types[..., :ODM_TYPES].max(-1) >= 3) | (types.sum(-1) >= 5)).astype(float)


# ================================================================ гамма-пуассон

def nb_logpmf(S: np.ndarray, a: float, b: float, n: float) -> np.ndarray:
    """log P(S) при λ ~ Gamma(a, rate b), S | λ ~ Poisson(n·λ)."""
    return (special.gammaln(a + S) - special.gammaln(a) - special.gammaln(S + 1)
            + a * np.log(b / (b + n)) + S * np.log(n / (b + n)))


LOG_BOUND = 7.0          # |log a|, |log b| ≤ 7: без этого смесь 3 гамм вырождается в точку λ = 0 с «бесконечным» правдоподобием
ZERO_A, ZERO_B = 1e-8, 1.0   # «нулевая» компонента (λ ≡ 0) для модели с нулевым раздуванием


def mix_unpack(theta: np.ndarray, m: int, zero: bool = False):
    """m гамма-компонент (+ нулевая при zero=True): theta = [логиты весов (m − 1 + zero), log a (m), log b (m)]."""
    nw = m - 1 + int(zero)
    w = np.exp(np.r_[0.0, theta[:nw]]); w /= w.sum()
    th = np.clip(theta[nw:], -LOG_BOUND, LOG_BOUND)
    a = np.exp(th[:m]); b = np.exp(th[m:])
    if zero:
        a = np.r_[ZERO_A, a]; b = np.r_[ZERO_B, b]
    return w, a, b


def mix_loglik_cells(S, w, a, b, n) -> np.ndarray:
    comp = np.stack([np.log(wj) + nb_logpmf(S, aj, bj, n) for wj, aj, bj in zip(w, a, b)], 1)
    return special.logsumexp(comp, axis=1)


def fit_mixture(S: np.ndarray, n: int, m: int, rng: np.random.Generator, n_starts: int = 8,
                zero: bool = False) -> dict:
    mean = S.mean() / n
    best = None
    nw = m - 1 + int(zero)
    f = lambda th: -mix_loglik_cells(S, *mix_unpack(th, m, zero), n).sum()      # noqa: E731
    for st in range(n_starts):
        if m == 1 and st == 0:
            th0 = np.r_[np.zeros(nw), np.log(0.2), np.log(0.2 / mean)]
        else:
            means = np.sort(mean * np.exp(rng.normal(0, 1.5, m)))
            shapes = np.exp(rng.normal(np.log(0.5), 0.8, m))
            th0 = np.r_[rng.normal(0, 1, nw), np.log(shapes), np.log(shapes / means)]
        res = optimize.minimize(f, th0, method="Nelder-Mead",
                                options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-7, "fatol": 1e-7})
        res = optimize.minimize(f, res.x, method="BFGS")
        if best is None or res.fun < best.fun:
            best = res
        if m == 1 and not zero and st >= 1:
            break
    w, a, b = mix_unpack(best.x, m, zero)
    order = np.argsort(a / b); w, a, b = w[order], a[order], b[order]
    ll = -best.fun; k = 3 * m - 1 + int(zero)
    return {"m": m, "zero": zero, "w": w, "a": a, "b": b, "loglik": ll, "aic": 2 * k - 2 * ll,
            "bic": k * np.log(len(S)) - 2 * ll, "n_par": k}


FIT_SPECS = {"gamma_mle": (1, False), "gamma_mix2": (2, False), "gamma_mix3": (3, False),
             "zi_gamma": (1, True), "zi_gamma_mix2": (2, True)}


def fit_mom(S: np.ndarray, n: int) -> dict:
    m = S.mean() / n
    v_lam = (S.var() - S.mean()) / n ** 2
    a = m ** 2 / v_lam; b = m / v_lam
    ll = nb_logpmf(S, a, b, n).sum()
    return {"m": 1, "w": np.array([1.0]), "a": np.array([a]), "b": np.array([b]), "loglik": ll,
            "aic": np.nan, "bic": np.nan, "n_par": 2}


def predicted_pmf(fit: dict, n: int, smax: int) -> np.ndarray:
    s = np.arange(smax + 1, dtype=float)
    return np.exp(mix_loglik_cells(s, fit["w"], fit["a"], fit["b"], n))


class Posterior:
    """Апостериор λ ячейки под смесью гамм; сэмплирование и среднее."""

    def __init__(self, fit: dict, S: np.ndarray, n: int):
        self.S, self.n = S, n
        self.a, self.b = fit["a"], fit["b"]
        lw = np.stack([np.log(wj) + nb_logpmf(S, aj, bj, n) for wj, aj, bj in
                       zip(fit["w"], fit["a"], fit["b"])], 1)
        self.r = np.exp(lw - special.logsumexp(lw, 1, keepdims=True))    # веса компонент у ячейки
        self.cum = np.cumsum(self.r, 1)

    def mean(self) -> np.ndarray:
        return (self.r * (self.a[None] + self.S[:, None]) / (self.b[None] + self.n)).sum(1)

    def sample(self, rng) -> np.ndarray:
        u = rng.random(len(self.S))
        j = np.minimum((u[:, None] > self.cum).sum(1), len(self.a) - 1)
        return rng.gamma(self.a[j] + self.S) / (self.b[j] + self.n)


# ================================================================ симуляция

Q_GROUPS = (5, 10, 20, 50, 200)   # оракул «с группами»: λ, огрублённая до K групп по квантилям (для Spearman)


def quantile_groups(lam: np.ndarray, K: int) -> np.ndarray:
    r = np.argsort(np.argsort(lam, kind="stable"), kind="stable")
    return np.floor(r * K / len(lam))


def simulate(name: str, lam_fn, shares_fn, c_t: list[float], k_within: float | None, nsim: int,
             rng: np.random.Generator, keep_draws: bool = False, n_hist: int = 0) -> tuple[dict, pd.DataFrame | None]:
    """lam_fn(rng) → λ (ячейка, ДТП/год), shares_fn(rng) → доли видов (ячейка × 9); c_t — множители уровня по годам.

    n_hist > 0 — апостериорная прогностическая проверка: в том же симулированном мире генерируется n_hist лет
    «истории» (Poisson(λ)), и честный ориентир «сумма истории» оценивается так же, как на факте."""
    rows = []
    c_t = np.asarray(c_t, float)
    for _ in range(nsim):
        lam = lam_fn(rng); P = shares_fn(rng)
        mu = lam[:, None] * c_t[None, :]                            # ячейка × год
        if k_within is not None:
            mu = mu * rng.gamma(k_within, 1.0 / k_within, size=mu.shape)
        types = rng.poisson(mu[:, :, None] * P[:, None, :])          # ячейка × год × вид
        Yt = types.sum(2); Ys = Yt.sum(1).astype(float)
        hot = odm_hot(types)
        r = {}
        for f, cv, cr in zip(FRACS, vy.coverages(Ys, lam, FRACS), vy.coverages(Ys, Ys, FRACS)):
            p = int(round(f * 100))
            r[f"cov_{p}"] = cv; r[f"real_{p}"] = cr
        r["spearman"] = float(spearmanr(lam, Ys).statistic)
        # оракул «с формой Y»: ранги λ переносятся на распределение независимой копии Y (повторяет связи-нули Y);
        # Spearman — корреляция рангов, и строгий порядок по λ его не максимизирует при 70–85 % нулей в Y
        Yind = rng.poisson(lam[:, None] * c_t[None, :]).sum(1)
        s_match = np.sort(Yind)[np.argsort(np.argsort(lam, kind="stable"), kind="stable")]
        r["spearman_matched"] = float(spearmanr(s_match, Ys).statistic)
        for K in Q_GROUPS:
            r[f"spearman_q{K}"] = float(spearmanr(quantile_groups(lam, K), Ys).statistic)
        r["n_acc"] = Ys.sum(); r["zero_share"] = (Ys == 0).mean()
        if n_hist:
            h = rng.poisson(lam * n_hist).astype(float)
            for f, cv in zip(FRACS, vy.coverages(Ys, h, FRACS)):
                r[f"hist_{int(round(f * 100))}"] = cv
            r["hist_spearman"] = float(spearmanr(h, Ys).statistic)
        nh = hot.sum(0); num = np.zeros(len(FRACS)); numr = np.zeros(len(FRACS))
        for t in range(len(c_t)):
            if nh[t] > 0:
                num += np.array(vy.coverages(hot[:, t], lam, FRACS)) * nh[t]
                numr += np.array(vy.coverages(hot[:, t], Yt[:, t].astype(float), FRACS)) * nh[t]
        for i, f in enumerate(FRACS):
            p = int(round(f * 100))
            r[f"odm_{p}"] = num[i] / nh.sum(); r[f"odm_real_{p}"] = numr[i] / nh.sum()
        r["n_hot"] = nh.sum()
        rows.append(r)
    d = pd.DataFrame(rows)
    summ = {"scenario": name}
    for c in d.columns:
        summ[f"{c}_mean"] = d[c].mean()
        summ[f"{c}_lo"] = d[c].quantile(0.025); summ[f"{c}_hi"] = d[c].quantile(0.975)
    return summ, (d if keep_draws else None)


def run(nsim: int) -> None:
    t0 = time.time()
    Y, T, cells = load()
    rng = np.random.default_rng(SEED)
    city_share = T.sum((0, 1)) / T.sum()
    print(f"ячеек {Y.shape[0]}, лет {Y.shape[1]}, ДТП {Y.sum():.0f}; доли видов города {np.round(city_share, 3)}")

    def window(years):
        idx = [YEARS.index(y) for y in years]
        return Y[:, idx].sum(1), T[:, idx, :].sum(1), len(idx)

    # ---------------- 1. подгонка на всех 11 годах
    S, TS, n = window(YEARS)
    fits = {"gamma_mom": fit_mom(S, n)}
    for key, (m, z) in FIT_SPECS.items():
        fits[key] = fit_mixture(S, n, m, rng, zero=z)
        print(f"  подгонка {key}: logL = {fits[key]['loglik']:.2f}, BIC = {fits[key]['bic']:.2f}", flush=True)
    smax = int(S.max())
    rows = []
    for k, f in fits.items():
        pm = predicted_pmf(f, n, 3000)
        sv = lambda thr: pm[thr:].sum()                                     # noqa: E731
        rows.append({"model": k, "n_par": f["n_par"], "loglik": f["loglik"], "aic": f["aic"], "bic": f["bic"],
                     "weights": ";".join(f"{x:.4f}" for x in f["w"]), "shape_a": ";".join(f"{x:.4f}" for x in f["a"]),
                     "rate_b": ";".join(f"{x:.4f}" for x in f["b"]),
                     "mean_lambda": ";".join(f"{x:.4f}" for x in f["a"] / f["b"]),
                     "zero_share_pred": pm[0], "share_ge10_pred": sv(10), "share_ge30_pred": sv(30),
                     "share_ge60_pred": sv(60), "share_ge100_pred": sv(100),
                     "exp_max_cells_ge_obsmax": sv(smax) * len(S)})
    obs = {"model": "наблюдение", "zero_share_pred": (S == 0).mean(), "share_ge10_pred": (S >= 10).mean(),
           "share_ge30_pred": (S >= 30).mean(), "share_ge60_pred": (S >= 60).mean(),
           "share_ge100_pred": (S >= 100).mean(), "exp_max_cells_ge_obsmax": 1}
    fit_tab = pd.DataFrame(rows + [obs])
    fit_tab.to_csv(OUT / "gamma_fit.csv", index=False)
    print(fit_tab.to_string())
    # частоты для рис. 1
    sgrid = np.arange(smax + 1)
    fig1 = pd.DataFrame({"s": sgrid, "observed_cells": np.bincount(S.astype(int), minlength=smax + 1)})
    for k in fits:
        fig1[f"pred_{k}"] = predicted_pmf(fits[k], n, smax) * len(S)
    fig1.to_csv(FIG / "fig1_gamma_fit.csv", index=False)

    best_key = min(FIT_SPECS, key=lambda k: fits[k]["bic"])
    print("лучшая по BIC:", best_key, flush=True)
    (OUT / "chosen_model.txt").write_text(best_key + "\n")

    # ---------------- общие конструкторы сценариев
    def make(years_fit, fit_key=None):
        S_, TS_, n_ = window(years_fit)
        m_, z_ = FIT_SPECS[fit_key or best_key]
        f = fit_mixture(S_, n_, m_, rng, zero=z_) if years_fit != YEARS else fits[fit_key or best_key]
        post = Posterior(f, S_, n_)
        alpha = TS_ + A_TYPES * city_share[None, :]
        P_mean = alpha / alpha.sum(1, keepdims=True)
        lam_post_mean = post.mean(); lam_raw = S_ / n_
        lam_fns = {"posterior": post.sample, "raw_mean": lambda r: lam_raw, "post_mean": lambda r: lam_post_mean}
        shares = {"posterior": lambda r: (g := r.gamma(alpha)) / g.sum(1, keepdims=True),
                  "raw_mean": lambda r: P_mean, "post_mean": lambda r: P_mean}
        base_rate = S_.sum() / n_                                    # ДТП в год по городу в окне подгонки
        return lam_fns, shares, base_rate, f

    yearly_tot = dict(zip(YEARS, Y.sum(0)))
    summaries, draws = [], {}

    def scen(tag, split, years_fit, eval_years, correct, kinds=("posterior", "raw_mean", "post_mean"),
             k_within=None, fit_key=None):
        lam_fns, shares, base_rate, f = make(years_fit, fit_key)
        c_t = [yearly_tot[y] / base_rate if correct else 1.0 for y in eval_years]
        for kind in kinds:
            name = f"{tag}|{kind}"
            t1 = time.time()
            s, d = simulate(name, lam_fns[kind], shares[kind], c_t, k_within, nsim, rng,
                            keep_draws=(split == "test"), n_hist=len([y for y in YEARS if y < eval_years[0]]))
            s.update({"split": split, "fit_years": f"{years_fit[0]}–{years_fit[-1]}", "level_correction": correct,
                      "lambda": kind, "within_k": k_within or np.inf, "c_t": ";".join(f"{c:.3f}" for c in c_t),
                      "fit_model": fit_key or best_key})
            summaries.append(s)
            if d is not None:
                draws[name] = d
            print(f"  {name}: топ-1/5/10 = {s['cov_1_mean']:.3f}/{s['cov_5_mean']:.3f}/{s['cov_10_mean']:.3f}, "
                  f"Spearman {s['spearman_mean']:.3f}/{s['spearman_matched_mean']:.3f} (группы: "
                  + "/".join(f"{s[f'spearman_q{K}_mean']:.3f}" for K in Q_GROUPS) + f"), "
                  f"история {s['hist_10_mean']:.3f}/{s['hist_spearman_mean']:.3f}, нули {s['zero_share_mean']:.3f}, ODM10 {s['odm_10_mean']:.3f}, реализ. "
                  f"{s['real_1_mean']:.3f}/{s['real_5_mean']:.3f}/{s['real_10_mean']:.3f} ({time.time() - t1:.0f} с)",
                  flush=True)
            pd.DataFrame(summaries).to_csv(OUT / "ceiling_scenarios.csv", index=False)

    pre_test = [y for y in YEARS if y < 2023]
    # ---------------- 2–3. тест: основная + границы
    scen("test_main", "test", YEARS, TEST_YEARS, False)
    scen("test_main_nb53", "test", YEARS, TEST_YEARS, False, kinds=("posterior",), k_within=K_WITHIN)
    scen("test_main_gamma1", "test", YEARS, TEST_YEARS, False, kinds=("posterior",), fit_key="gamma_mle")
    # ---------------- 4. нестационарность
    scen("test_pre_corr", "test", pre_test, TEST_YEARS, True)
    scen("test_pre_nocorr", "test", pre_test, TEST_YEARS, False, kinds=("posterior",))
    scen("test_all_corr", "test", YEARS, TEST_YEARS, True, kinds=("posterior",))
    # оракул, знающий риск именно тестового периода: λ только по 2023–2025 (проверка «реализованного потолка»)
    scen("test_testonly", "test", TEST_YEARS, TEST_YEARS, False)
    # ---------------- 5. фолды
    for fy in (2021, 2022):
        pre = [y for y in YEARS if y < fy]
        scen(f"fold{fy}_pre_corr", f"y{fy}", pre, [fy], True)
        scen(f"fold{fy}_pre_nocorr", f"y{fy}", pre, [fy], False, kinds=("posterior",))
        scen(f"fold{fy}_all", f"y{fy}", YEARS, [fy], False)
        scen(f"fold{fy}_all_corr", f"y{fy}", YEARS, [fy], True, kinds=("posterior",))

    pd.concat({k: v for k, v in draws.items()}, names=["scenario", "sim"]).reset_index().to_parquet(
        OUT / "sim_draws_test.parquet", index=False)
    # факт: реализованный потолок на тесте и фолдах
    real = []
    for split, yrs in (("test", TEST_YEARS), ("y2021", [2021]), ("y2022", [2022])):
        idx = [YEARS.index(y) for y in yrs]
        ys = Y[:, idx].sum(1)
        r = {"split": split, "n_acc": ys.sum(), "zero_share": (ys == 0).mean()}
        for f, c in zip(FRACS, vy.coverages(ys, ys, FRACS)):
            r[f"real_{int(round(f * 100))}"] = c
        hot = odm_hot(T[:, idx, :]); nh = hot.sum(0)
        num = sum(np.array(vy.coverages(hot[:, t], Y[:, idx[t]], FRACS)) * nh[t] for t in range(len(idx)))
        for i, f in enumerate(FRACS):
            r[f"odm_real_{int(round(f * 100))}"] = num[i] / nh.sum()
        r["n_hot"] = nh.sum()
        real.append(r)
    pd.DataFrame(real).to_csv(OUT / "observed_realized_ceiling.csv", index=False)
    print(pd.DataFrame(real).to_string())
    print(f"готово за {time.time() - t0:.0f} с")


# ================================================================ сравнение и рисунки

def fact_table() -> pd.DataFrame:
    """Факт вариантов: доля накрытых (eb_top1_5_10.csv), Spearman (metrics_summary.csv), ОДМ (odm_recall_pooled)."""
    eb = pd.read_csv(HEB / "eb_top1_5_10.csv")
    ms = pd.read_csv(HEB / "metrics_summary.csv")
    odm = pd.read_csv(HEB / "odm_recall_pooled.csv")
    rows = []
    for key, lab, _ in VARIANTS + [("oracle", "потолок по факту", None)]:
        for split, per_eb, per_odm in (("test", "test_sum", "test"), ("folds", "folds", "folds")):
            r = {"variant": key, "label": lab, "split": split}
            for f in FRACS:
                p = int(round(f * 100))
                q = eb[(eb.kind == "level") & (eb.variant == key) & (eb.period == per_eb) & np.isclose(eb.frac, f)]
                r[f"cov_{p}"] = q.point.iloc[0] if len(q) else np.nan
                q = odm[(odm.kind == "level") & (odm.variant == key) & (odm.period == per_odm) & np.isclose(odm.frac, f)]
                r[f"odm_{p}"] = q.point.iloc[0] if len(q) else np.nan
            if split == "test":
                q = ms[(ms.variant == key) & (ms.split == "test") & (ms.eval_period.astype(str) == "sum")]
            else:
                q = ms[(ms.variant == key) & (ms.split.isin(["y2021", "y2022"]))
                       & (ms.eval_period.astype(str).isin(["2021", "2022"]))]
            r["spearman"] = q.spearman.mean() if len(q) else np.nan
            rows.append(r)
    # по отдельным фолдам — для самопроверки
    for key, lab, _ in VARIANTS:
        for fy in (2021, 2022):
            q = ms[(ms.variant == key) & (ms.split == f"y{fy}") & (ms.eval_period.astype(str) == str(fy))]
            r = {"variant": key, "label": lab, "split": f"y{fy}", "spearman": q.spearman.iloc[0]}
            for f in FRACS:
                p = int(round(f * 100))
                r[f"cov_{p}"] = q[f"coverage_top_{p}pct"].iloc[0]
            rows.append(r)
    return pd.DataFrame(rows)


def analyze() -> None:
    sc = pd.read_csv(OUT / "ceiling_scenarios.csv").set_index("scenario")
    fact = fact_table(); fact.to_csv(OUT / "variants_fact.csv", index=False)
    real_obs = pd.read_csv(OUT / "observed_realized_ceiling.csv").set_index("split")
    metrics = [("cov_1", "топ-1 %"), ("cov_5", "топ-5 %"), ("cov_10", "топ-10 %"), ("spearman", "Spearman"),
               ("odm_1", "ОДМ топ-1 %"), ("odm_5", "ОДМ топ-5 %"), ("odm_10", "ОДМ топ-10 %")]
    main, lo_b, hi_b = sc.loc["test_main|posterior"], sc.loc["test_main|post_mean"], sc.loc["test_main|raw_mean"]
    rows = []
    for key, lab, _ in VARIANTS:
        fr = fact[(fact.variant == key) & (fact.split == "test")].iloc[0]
        for m, mlab in metrics:
            if m == "spearman":   # предел Spearman — лучший из двух оракулов (строгий порядок λ / «с формой Y»)
                pick = lambda x: max(x["spearman_mean"], x["spearman_matched_mean"])   # noqa: E731
                lim, lim_lo, lim_hi = pick(main), pick(lo_b), pick(hi_b)
                ci = (main["spearman_matched_lo"], main["spearman_matched_hi"]) \
                    if main["spearman_matched_mean"] >= main["spearman_mean"] else (main["spearman_lo"], main["spearman_hi"])
            else:
                lim = main[f"{m}_mean"]; lim_lo = lo_b[f"{m}_mean"]; lim_hi = hi_b[f"{m}_mean"]
                ci = (main[f"{m}_lo"], main[f"{m}_hi"])
            ceil_fact = fact[(fact.variant == "oracle") & (fact.split == "test")].iloc[0].get(m, np.nan)
            rows.append({"variant": key, "label": lab, "metric": m, "metric_label": mlab, "fact": fr[m],
                         "limit": lim, "limit_ci_lo": ci[0], "limit_ci_hi": ci[1],
                         "limit_post_mean": lim_lo, "limit_raw_mean": lim_hi,
                         "share_of_limit": fr[m] / lim,
                         "share_of_limit_min": fr[m] / max(lim, lim_lo, lim_hi),
                         "share_of_limit_max": fr[m] / min(lim, lim_lo, lim_hi),
                         "fact_ceiling": ceil_fact if m.startswith(("cov", "odm")) else np.nan,
                         "share_of_fact_ceiling": fr[m] / ceil_fact if m.startswith(("cov", "odm")) else np.nan})
    cmp_ = pd.DataFrame(rows); cmp_.to_csv(OUT / "comparison.csv", index=False)
    print(cmp_.round(3).to_string())

    # самопроверка на фолдах
    rows = []
    for fy in (2021, 2022):
        for scn in (f"fold{fy}_pre_corr|posterior", f"fold{fy}_pre_corr|post_mean", f"fold{fy}_pre_corr|raw_mean",
                    f"fold{fy}_pre_nocorr|posterior", f"fold{fy}_all|posterior", f"fold{fy}_all|post_mean",
                    f"fold{fy}_all|raw_mean", f"fold{fy}_all_corr|posterior"):
            s = sc.loc[scn]
            r = {"fold": fy, "scenario": scn}
            for m in ("cov_1", "cov_5", "cov_10", "spearman", "spearman_matched", "real_1", "real_5", "real_10",
                      "hist_1", "hist_5", "hist_10", "hist_spearman", "zero_share"):
                r[m] = s[f"{m}_mean"]; r[f"{m}_lo"] = s[f"{m}_lo"]
            r["spearman_best"] = max(r["spearman"], r["spearman_matched"])
            for key, lab, _ in VARIANTS:
                fr = fact[(fact.variant == key) & (fact.split == f"y{fy}")].iloc[0]
                for m in ("cov_1", "cov_5", "cov_10", "spearman"):
                    r[f"{key}_{m}"] = fr[m]
            for m in ("real_1", "real_5", "real_10"):
                r[f"observed_{m}"] = real_obs.loc[f"y{fy}", m]
            best = {m: max(fact[(fact.variant == k) & (fact.split == f"y{fy}")].iloc[0][m] for k, _, _ in VARIANTS)
                    for m in ("cov_1", "cov_5", "cov_10", "spearman")}
            r["limit_ge_best_model_all"] = all(r["spearman_best" if m == "spearman" else m] >= best[m] for m in best)
            r["limit_ge_best_model_coverage"] = all(r[m] >= best[m] for m in best if m != "spearman")
            r["best_model"] = ";".join(f"{m}={v:.3f}" for m, v in best.items())
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "fold_selfcheck.csv", index=False)
    print(pd.DataFrame(rows)[["scenario", "cov_1", "cov_5", "cov_10", "spearman", "spearman_matched", "hist_10",
                              "real_10", "observed_real_10", "limit_ge_best_model_all", "best_model"]].round(3).to_string())


def plots() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(OUT / "ceiling_scenarios.csv").set_index("scenario")
    fact = pd.read_csv(OUT / "variants_fact.csv")
    best_key = (OUT / "chosen_model.txt").read_text().strip()
    # ---- рис. 1: подгонка
    f1 = pd.read_csv(FIG / "fig1_gamma_fit.csv")
    labs = {"gamma_mom": "одна гамма (метод моментов)", "gamma_mle": "одна гамма (ML)",
            "gamma_mix2": "смесь 2 гамм (ML)", "gamma_mix3": "смесь 3 гамм (ML)",
            "zi_gamma": "гамма + нулевой риск (ML)", "zi_gamma_mix2": "2 гаммы + нулевой риск (ML)"}
    cols = {"gamma_mom": C[3], "gamma_mle": C[1], "gamma_mix2": C[0], "gamma_mix3": C[6], "zi_gamma": C[4],
            "zi_gamma_mix2": C[5]}
    shown = list(dict.fromkeys(["gamma_mom", "gamma_mle", "gamma_mix2", best_key]))
    labs = {k: labs[k] for k in shown}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    edges = np.r_[[e for e in (0, 1, 2, 3, 5, 8, 12, 18, 26, 40, 60, 90, 140) if e <= f1.s.max()], f1.s.max() + 1]
    b = pd.cut(f1.s, edges, right=False)
    g = f1.groupby(b, observed=False).sum(numeric_only=True)
    xl = [f"{int(l)}" if r - l == 1 else f"{int(l)}–{int(r) - 1}" for l, r in zip(edges[:-1], edges[1:])]
    x = np.arange(len(g))
    ax[0].bar(x, g.observed_cells, color="#d9d8d2", label="наблюдение", width=0.8)
    for k in labs:
        ax[0].plot(x, g[f"pred_{k}"], marker="o", ms=3.5, lw=1.5, color=cols[k], label=labs[k])
    ax[0].set_yscale("log"); ax[0].set_xticks(x, xl, rotation=45, ha="right")
    ax[0].set_xlabel("ДТП в ячейке за 2015–2025 (интервал)"); ax[0].set_ylabel("число ячеек (лог. шкала)")
    ax[0].set_title("Распределение сумм ДТП по ячейкам")
    ax[0].legend(fontsize=7, frameon=False)
    s = f1.s.to_numpy()
    surv_obs = f1.observed_cells[::-1].cumsum()[::-1] / f1.observed_cells.sum()
    ax[1].step(s[1:], surv_obs[1:], where="post", color=INK, lw=1.6, label="наблюдение")
    for k in labs:
        pk = f1[f"pred_{k}"]; tot = f1.observed_cells.sum()
        tail_beyond = 1 - pk.sum() / tot                         # масса за пределами наблюдаемого максимума
        surv = (pk[::-1].cumsum()[::-1] / tot + tail_beyond)
        ax[1].plot(s[1:], surv[1:], color=cols[k], lw=1.4, label=labs[k])
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("порог s, ДТП за 11 лет"); ax[1].set_ylabel("доля ячеек с суммой ≥ s")
    ax[1].set_title("Хвост: доля ячеек с суммой ≥ s")
    fig.suptitle(f"Подгонка распределения риска ячеек (выбрана: {labs[best_key]})", y=1.02)
    save(fig, "fig1_gamma_fit.png")

    # ---- рис. 2–3: распределения предела
    d = pd.read_parquet(OUT / "sim_draws_test.parquet")
    dm = d[d.scenario == "test_main|posterior"]
    fact_t = fact[fact.split == "test"].set_index("variant")

    def panel(ax, p, legend=True, compact=False):
        v = dm[f"cov_{p}"]; vr = dm[f"real_{p}"]
        bins = np.linspace(min(dm[f"hist_{p}"].min(), fact_t.loc["cf", f"cov_{p}"]) - 0.01,
                           max(vr.max(), fact_t.loc["oracle", f"cov_{p}"]) + 0.01, 70)
        ax.hist(v, bins=bins, color="#9dbfe8", label="предел (оракул знает λ)", alpha=0.9)
        ax.hist(vr, bins=bins, color="#d9d8d2", label="«реализованный потолок» в симуляции", alpha=0.9)
        ax.hist(dm[f"hist_{p}"], bins=bins, histtype="step", color=C[6], lw=1.1,
                label="симулированная «сумма истории» (стационарный мир)")
        lo, hi = sc.loc["test_main|post_mean", f"cov_{p}_mean"], sc.loc["test_main|raw_mean", f"cov_{p}_mean"]
        ax.axvspan(min(lo, hi), max(lo, hi), color="#2a78d6", alpha=0.08, label="вилка предела (λ: сглаж. … сырое)")
        ax.axvline(v.mean(), color="#2a78d6", lw=1.2, ls=":")
        for key, lab, col in VARIANTS:
            ax.axvline(fact_t.loc[key, f"cov_{p}"], color=col, lw=1.6, label=f"{lab}: {fact_t.loc[key, f'cov_{p}']:.3f}")
        ax.axvline(fact_t.loc["oracle", f"cov_{p}"], color=INK, lw=1.6, ls="--",
                   label=f"потолок по факту: {fact_t.loc['oracle', f'cov_{p}']:.3f}")
        ax.set_xlabel(f"доля ДТП 2023–2025 в топ-{p} % ячеек"); ax.set_ylabel("число симуляций")
        ax.set_title(f"Топ-{p} %: предел {v.mean():.3f} [{v.quantile(.025):.3f}; {v.quantile(.975):.3f}]",
                     fontsize=9 if compact else 10)
        if legend:
            ax.legend(fontsize=7 if compact else 7.5, frameon=False, loc="upper left")
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    panel(ax, 10)
    fig.suptitle("Достижимый предел для топ-10 % (1000 симуляций теста 2023–2025)", y=1.0)
    save(fig, "fig2_ceiling_top10.png")
    SUMMARY_FIG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIG / "fig2_ceiling_top10.png", SUMMARY_FIG)
    dm[["sim", "cov_1", "cov_5", "cov_10", "real_1", "real_5", "real_10", "hist_1", "hist_5", "hist_10", "spearman",
        "spearman_matched"]].to_csv(
        FIG / "fig2_fig3_draws_main.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    panel(ax[0], 1, legend=False, compact=True); panel(ax[1], 5, legend=True, compact=True)
    ax[1].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Достижимый предел для топ-1 % и топ-5 % (1000 симуляций)", y=1.02)
    save(fig, "fig3_ceiling_top1_top5.png")

    # ---- рис. 4: доля от предела
    cmp_ = pd.read_csv(OUT / "comparison.csv")
    mets = [("cov_1", "топ-1 %"), ("cov_5", "топ-5 %"), ("cov_10", "топ-10 %"), ("spearman", "Spearman"),
            ("odm_10", "ОДМ топ-10 %")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.19; x = np.arange(len(mets))
    for i, (key, lab, col) in enumerate(VARIANTS):
        q = cmp_[cmp_.variant == key].set_index("metric").loc[[m for m, _ in mets]]
        pos = x + (i - 1.5) * w
        ax.bar(pos, q.share_of_limit * 100, width=w - 0.02, color=col, label=lab)
        ax.errorbar(pos, q.share_of_limit * 100, yerr=[(q.share_of_limit - q.share_of_limit_min) * 100,
                                                       (q.share_of_limit_max - q.share_of_limit) * 100],
                    fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    alt = [("test_testonly|posterior", "D", "предел по λ только 2023–2025"),
           ("test_pre_corr|posterior", "x", "предел по λ 2015–2022 + поправка уровня")]
    for j, (scn, mk, lab_alt) in enumerate(alt):
        for i, (key, lab, col) in enumerate(VARIANTS):
            fr = fact[(fact.variant == key) & (fact.split == "test")].iloc[0]
            ys = []
            for m, _ in mets:
                lim = max(sc.loc[scn, "spearman_mean"], sc.loc[scn, "spearman_matched_mean"]) if m == "spearman" \
                    else sc.loc[scn, f"{m}_mean"]
                ys.append(fr[m] / lim * 100)
            ax.scatter(x + (i - 1.5) * w, ys, marker=mk, s=14, color=INK, zorder=5, linewidths=0.9,
                       label=lab_alt if i == 0 else None)
    for xi, (m, _) in zip(x, mets):
        q = cmp_[cmp_.metric == m]
        ax.text(xi, 103, f"предел\n{q.limit.iloc[0]:.3f}", ha="center", fontsize=7, color=INK)
    ax.axhline(100, color=INK, lw=0.8)
    ax.set_xticks(x, [l for _, l in mets]); ax.set_ylabel("% от достижимого предела")
    ax.set_ylim(0, 118)
    ax.set_title("Факт на тесте 2023–2025 как доля от предела\n(столбец — основная оценка, усы — вилка λ сырое … сглаженное, значки — нестационарность)", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.36))
    cmp_[cmp_.metric.isin([m for m, _ in mets])].to_csv(FIG / "fig4_share_of_limit.csv", index=False)
    save(fig, "fig4_share_of_limit.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=1000)
    ap.add_argument("--plots-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    if not a.plots_only:
        run(a.nsim)
    analyze()
    plots()
