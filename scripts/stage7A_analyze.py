"""Этап 7, шаг 2A: анализ годовых моделей (7.2, 7.3) -> docs/analysis/features/stage7_count12/models/.

Читает reports/v2_eval_year_stage7A_models*.csv|parquet (scripts/stage7A_run_models.sh) и строит:
- tab_folds.csv / tab_test.csv — метрики вариантов (среднее ± sd по сидам) по фолдам, их среднему и тесту;
- cmp_folds.csv / cmp_test.csv — Δ пар вариантов (compare_folds_year / compare_year на тесте sum);
- bootstrap.csv — бутстреп доли накрытых ДТП у выбранной модели (тест sum и по годам) + парные разности;
- calibration_by_year.csv, calibration_deciles.csv — калибровка и поправка C по HSM;
- overdispersion_*.csv — параметр k NB (метод моментов по группам прогноза, поячеечный ММ, NB-подгонка);
- figures/fig1..fig5 (+ CSV).

Запуск: PYTHONPATH=src venv/bin/python scripts/stage7A_analyze.py [--selected 'road_geo_v2_skdf|count:Poisson']
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import Pool  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import gammaln  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk import catboost_baseline as cb  # noqa: E402
from road_accident_risk import v2_eval_year as vy  # noqa: E402
from road_accident_risk.paths import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "docs/analysis/features/stage7_count12/models"
FIG = OUT / "figures"
TAG = "stage7A_models"
V = "road_geo_v2_skdf"
RID = {"bin_bal": f"{V}|binary:Logloss:Balanced", "bin_none": f"{V}|binary:Logloss:None",
       "poisson": f"{V}|count:Poisson", "tw13": f"{V}|count:Tweedie:variance_power=1.3",
       "tw16": f"{V}|count:Tweedie:variance_power=1.6", "month_mean": f"{V}|monthly:mean",
       "month_max": f"{V}|monthly:max", "prev3": "baseline:accidents_prev_3y|baseline",
       "prev1": "baseline:accidents_prev_year|baseline", "major": "baseline:major_road_length_m|baseline"}
LAB = {RID["bin_bal"]: "бинарная, Balanced", RID["bin_none"]: "бинарная, без весов",
       RID["poisson"]: "Poisson", RID["tw13"]: "Tweedie 1.3", RID["tw16"]: "Tweedie 1.6",
       RID["month_mean"]: "месячная → год (среднее)", RID["month_max"]: "месячная → год (максимум)",
       RID["prev3"]: "ориентир: ДТП за 3 прошлых года", RID["prev1"]: "ориентир: ДТП за прошлый год",
       RID["major"]: "ориентир: длина магистралей"}
MAIN = ["coverage_top_1pct", "coverage_top_5pct", "coverage_top_10pct", "spearman", "roc_auc", "pr_auc",
        "poisson_deviance", "calib_ratio"]
MLAB = {"coverage_top_1pct": "доля ДТП в топ-1 %", "coverage_top_5pct": "доля ДТП в топ-5 %",
        "coverage_top_10pct": "доля ДТП в топ-10 %", "spearman": "Spearman"}
COL = {RID["poisson"]: "#1f6fb4", RID["bin_bal"]: "#e3a33b", RID["bin_none"]: "#b07d2b", RID["tw13"]: "#5aa9e6",
       RID["tw16"]: "#8e6bbf", RID["month_mean"]: "#2e9e5b", RID["month_max"]: "#7fc79b", RID["prev3"]: "#c0392b",
       RID["prev1"]: "#e58e84", RID["major"]: "#7f8c8d", "oracle": "#222222"}


def agg_table(runs: pd.DataFrame, splits: list[str], periods: list[str]) -> pd.DataFrame:
    r = runs[runs.split.isin(splits) & runs.eval_period.isin(periods)]
    g = r.groupby(["run_id", "split", "eval_period"])[MAIN + ["coverage_oracle_top_1pct", "coverage_oracle_top_5pct",
                                                              "coverage_oracle_top_10pct"]]
    m = g.mean().add_suffix("_mean"); s = g.std().add_suffix("_sd"); n = g.size().rename("n_seeds")
    return pd.concat([m, s, n], axis=1).reset_index()


def fmt(m: float, s: float) -> str:
    if not np.isfinite(m):
        return "—"
    return f"{m:.4f}" if not np.isfinite(s) or s == 0 else f"{m:.4f} ± {s:.4f}"


# ------------------------------------------------------------------ сверхдисперсия

def nb_loglik(y: np.ndarray, mu: np.ndarray, k: float) -> float:
    th = 1.0 / k
    return float(np.sum(gammaln(y + th) - gammaln(th) - gammaln(y + 1) + th * np.log(th / (th + mu))
                        + y * np.log(mu / (th + mu))))


def k_estimates(y: np.ndarray, mu: np.ndarray, n_bins: int = 20) -> dict:
    y = np.asarray(y, float); mu = np.clip(np.asarray(mu, float), 1e-9, None)
    k_cell = float(np.sum((y - mu) ** 2 - mu) / np.sum(mu ** 2))            # поячеечный ММ: E[(y−μ)²] = μ + kμ²
    b = binned(y, mu, n_bins)
    w = b["n"].to_numpy()
    k_bin = float(np.sum(w * (b["msr"] - b["mu"]) * b["mu"] ** 2) / np.sum(w * b["mu"] ** 4))  # МНК через 0
    res = minimize_scalar(lambda lk: -nb_loglik(y, mu, np.exp(lk)), bounds=(np.log(1e-3), np.log(50)),
                          method="bounded")
    k_ml = float(np.exp(res.x))
    ll_p = float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
    return {"n": len(y), "sum_y": y.sum(), "sum_mu": mu.sum(), "k_mm_cell": k_cell, "k_mm_binned": k_bin,
            "k_nb_ml": k_ml, "loglik_poisson": ll_p, "loglik_nb": -res.fun, "lr_stat": 2 * (-res.fun - ll_p)}


def binned(y: np.ndarray, mu: np.ndarray, n_bins: int = 20) -> pd.DataFrame:
    r = pd.Series(mu).rank(method="first").to_numpy()
    b = np.ceil(r / len(r) * n_bins).astype(int)
    d = pd.DataFrame({"b": b, "y": y, "mu": mu, "sr": (y - mu) ** 2})
    return d.groupby("b").agg(n=("y", "size"), mu=("mu", "mean"), mean_y=("y", "mean"), var_y=("y", "var"),
                              msr=("sr", "mean")).reset_index()


def insample_preds(panel: pd.DataFrame, loss: str, seeds) -> pd.DataFrame:
    """Прогноз счётной модели теста (обучение 2015–2022) на годах обучения — для k «на обучении»."""
    feats = vy.year_variant_features(V, panel)
    tr0, _ = vy.split_frames_year(panel, "test")
    caps = cb.fit_p99_clip_caps(tr0, feats); tr = cb.apply_p99_clipping(tr0, caps)
    ps = []
    for seed in seeds:
        m = vy.make_model("count", loss, None, seed)
        m.fit(Pool(tr[feats], tr[vy.TARGET_COUNT].astype(int), feature_names=feats))
        ps.append(m.predict(tr[feats], prediction_type="Exponent"))
    return tr[["h3", "year", vy.TARGET_COUNT]].assign(pred=np.mean(ps, axis=0)).reset_index(drop=True)


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default=RID["poisson"])
    ap.add_argument("--binary", default=RID["bin_bal"], help="выбранная бинарная (7.2)")
    ap.add_argument("--skip-insample", action="store_true")
    args = ap.parse_args()
    SEL, BIN = args.selected, args.binary
    COL[SEL] = "#0b4f8a"  # выбранная модель — тёмно-синяя, рисуется поверх остальных
    FIG.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(vy.REPORTS / f"v2_eval_year_{TAG}.csv", dtype={"eval_period": str})
    preds = pd.read_parquet(vy.REPORTS / f"v2_eval_year_{TAG}_preds.parquet")
    panel_y = pd.read_parquet(vy.YEAR_PANEL_PARQUET)

    # ---------------- таблицы
    tf = agg_table(runs, ["y2021", "y2022"], ["2021", "2022"])
    fm = tf.groupby("run_id")[[c for c in tf.columns if c.endswith("_mean")]].mean().reset_index()
    fm["split"] = "folds_mean"; fm["eval_period"] = "folds_mean"
    tf = pd.concat([tf, fm], ignore_index=True)
    tf.to_csv(OUT / "tab_folds.csv", index=False)
    tt = agg_table(runs, ["test"], ["2023", "2024", "2025", "mean", "sum"])
    tt.to_csv(OUT / "tab_test.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        cols = ["run_id", "split", "eval_period", *[f"{c}_mean" for c in MAIN[:4]], "calib_ratio_mean"]
        print(tf[cols].sort_values(["split", "run_id"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(tt[cols].sort_values(["eval_period", "run_id"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------- сравнения
    pairs = [(RID["bin_none"], RID["bin_bal"]), (BIN, RID["month_mean"]), (BIN, RID["month_max"]),
             (RID["poisson"], RID["tw13"]), (RID["poisson"], RID["tw16"]), (RID["tw13"], RID["tw16"]),
             (BIN, RID["poisson"]), (BIN, RID["tw13"]), (BIN, RID["tw16"]), (RID["month_mean"], RID["poisson"]),
             (RID["prev3"], SEL), (RID["prev1"], SEL), (RID["major"], SEL), (RID["month_mean"], SEL)]
    pairs = list(dict.fromkeys(p for p in pairs if p[0] != p[1]))
    cf, ct = [], []
    for a, b in pairs:
        cf.append(vy.compare_folds_year(runs, a, b))
        for per in ["sum", "2023", "2024", "2025", "mean"]:
            ct.append(vy.compare_year(runs, a, b, split="test", period=per))
    cf = pd.concat(cf, ignore_index=True); ct = pd.concat(ct, ignore_index=True)
    cf.to_csv(OUT / "cmp_folds.csv", index=False); ct.to_csv(OUT / "cmp_test.csv", index=False)
    show = ["base", "variant", "column", "delta", "two_sigma", "verdict", "verdict_paired", "delta_y2021",
            "delta_y2022", "same_sign_all_folds"]
    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.max_colwidth", 45):
        print(cf[cf.column.isin(MAIN)][show].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        print(ct[(ct.eval_period == "sum") & ct.column.isin(MAIN)][
            ["base", "variant", "column", "delta", "two_sigma", "verdict", "verdict_paired"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---------------- бутстреп выбранной модели
    bs = []
    for per in ["sum", "2023", "2024", "2025"]:
        bs.append(vy.bootstrap_from_predictions(preds, SEL, "test", per))
        for base in [RID["prev3"], RID["prev1"], BIN, RID["month_mean"], RID["major"]]:
            if base != SEL:
                bs.append(vy.bootstrap_from_predictions(preds, SEL, "test", per, base_run_id=base))
    bs = pd.concat(bs, ignore_index=True); bs.to_csv(OUT / "bootstrap.csv", index=False)
    print(bs[bs.eval_period == "sum"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---------------- калибровка
    sums = panel_y.groupby("year")[vy.TARGET_COUNT].sum()
    cal = []
    for rid in [RID["poisson"], RID["tw13"], RID["tw16"]]:
        d = preds[preds.run_id == rid]
        for split, g in d.groupby("split"):
            s = vy.YEAR_SPLITS[split]
            train_mean = sums.loc[list(s.train_years)].mean(); last = max(s.train_years)
            pm = g.groupby(["seed", "year"])["pred"].sum().groupby("year").mean()
            # признаки статичны: прогноз ячейки одинаков во все годы, значит Σпрогноза на последнем году
            # обучения равна Σпрогноза года оценки; C (HSM) = факт / прогноз на последнем году обучения
            c_hsm = sums.loc[last] / pm.iloc[0]
            c_3y = sums.loc[[last - 2, last - 1, last]].mean() / pm.iloc[0]
            for yr, p in pm.items():
                cal.append({"run_id": rid, "split": split, "year": yr, "sum_pred": p, "sum_fact": sums.loc[yr],
                            "ratio": p / sums.loc[yr], "train_mean_per_year": train_mean,
                            "train_mean_over_pred": train_mean / p, "last_train_year": last,
                            "fact_last_train_year": sums.loc[last], "C_hsm": c_hsm, "ratio_after_C": c_hsm * p / sums.loc[yr],
                            "C_3y": c_3y, "ratio_after_C3y": c_3y * p / sums.loc[yr]})
    cal = pd.DataFrame(cal); cal.to_csv(OUT / "calibration_by_year.csv", index=False)
    print(cal.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    dec = []
    for per in ["2023", "2024", "2025", "sum"]:
        f = vy.period_frame(preds, SEL, "test", per)
        t = vy.calibration_table(f[vy.TARGET_COUNT].to_numpy(), f["pred"].to_numpy())
        dec.append(t.assign(eval_period=per))
    for split in ["y2021", "y2022"]:
        f = vy.period_frame(preds, SEL, split, split[1:])
        dec.append(vy.calibration_table(f[vy.TARGET_COUNT].to_numpy(), f["pred"].to_numpy()).assign(eval_period=split[1:]))
    dec = pd.concat(dec, ignore_index=True); dec.to_csv(OUT / "calibration_deciles.csv", index=False)
    # инвариантность ранжирования к множителю C
    f = vy.period_frame(preds, SEL, "test", "sum")
    c0 = vy.coverages(f[vy.TARGET_COUNT].to_numpy(), f["pred"].to_numpy())
    c1 = vy.coverages(f[vy.TARGET_COUNT].to_numpy(), 0.9 * f["pred"].to_numpy())
    print("покрытие до/после множителя C:", c0, c1)

    # ---------------- сверхдисперсия
    sel_loss = SEL.split("|count:")[1] if "|count:" in SEL else "Poisson"
    od_rows, od_bins = [], []

    def add(name: str, y: np.ndarray, mu: np.ndarray, years: np.ndarray | None = None):
        od_rows.append({"sample": name, "rescaled": False, **k_estimates(y, mu)})
        od_bins.append(binned(y, mu).assign(sample=name, rescaled=False))
        if years is not None:  # поправка на уровень года: μ × Σфакт/Σпрогноз внутри года
            mu2 = mu.copy()
            for yr in np.unique(years):
                i = years == yr; mu2[i] = mu[i] * y[i].sum() / mu[i].sum()
            od_rows.append({"sample": name, "rescaled": True, **k_estimates(y, mu2)})
            od_bins.append(binned(y, mu2).assign(sample=name, rescaled=True))

    for split, yr in (("y2021", 2021), ("y2022", 2022)):
        f = vy.period_frame(preds, SEL, split, str(yr))
        add(f"fold {yr}", f[vy.TARGET_COUNT].to_numpy(float), f["pred"].to_numpy(float), np.full(len(f), yr))
    ft = preds[(preds.run_id == SEL) & (preds.split == "test")].groupby(["h3", "year"], as_index=False) \
        .agg(y=(vy.TARGET_COUNT, "first"), mu=("pred", "mean"))
    for yr in (2023, 2024, 2025):
        g = ft[ft.year == yr]; add(f"тест {yr}", g.y.to_numpy(float), g.mu.to_numpy(float), np.full(len(g), yr))
    add("тест 2023–2025", ft.y.to_numpy(float), ft.mu.to_numpy(float), ft.year.to_numpy())
    if not args.skip_insample:
        ins = insample_preds(vy.load_year_panel(), sel_loss, vy.SEEDS)
        ins.to_parquet(OUT / "insample_preds_test_model.parquet", index=False)
        add("обучение 2015–2022 (in-sample)", ins[vy.TARGET_COUNT].to_numpy(float), ins.pred.to_numpy(float),
            ins.year.to_numpy())
    # без модели — как в METHODOLOGY.md (по ячейкам с историей)
    p = panel_y[panel_y.year.between(2015, 2025)]
    for nm, d in (("без модели: все ячейки-годы", p),
                  ("без модели: ячейки с ДТП в прошлом году", p[p.accidents_prev_year > 0])):
        m, v = d[vy.TARGET_COUNT].mean(), d[vy.TARGET_COUNT].var()
        od_rows.append({"sample": nm, "rescaled": False, "n": len(d), "sum_y": d[vy.TARGET_COUNT].sum(),
                        "k_mm_cell": (v - m) / m ** 2, "mean": m, "var": v})
    od = pd.DataFrame(od_rows); od.to_csv(OUT / "overdispersion_k.csv", index=False)
    odb = pd.concat(od_bins, ignore_index=True); odb.to_csv(OUT / "overdispersion_bins.csv", index=False)
    print(od.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ================================================================ рисунки
    plt.rcParams.update({"font.size": 9})
    # fig1: кривые gain на тесте (sum)
    curves = [(SEL, f"{LAB[SEL]} (выбранная)"), (BIN, LAB[BIN]), (RID["month_mean"], LAB[RID["month_mean"]]),
              (RID["prev3"], LAB[RID["prev3"]]), (RID["major"], LAB[RID["major"]])]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for rid, lab in curves:
        f = vy.period_frame(preds, rid, "test", "sum")
        g = vy.gain_curve(f[vy.TARGET_COUNT].to_numpy(), f["pred"].to_numpy(), n_points=1001)
        rows.append(g.assign(curve=lab))
        for ax in axes:
            ax.plot(g.frac_cells * 100, g.frac_accidents * 100, color=COL[rid], lw=2.0 if rid == SEL else 1.1,
                    label=lab, zorder=4 if rid == SEL else 2)
    g = vy.gain_curve(f[vy.TARGET_COUNT].to_numpy(), f[vy.TARGET_COUNT].to_numpy(), n_points=1001)
    rows.append(g.assign(curve="потолок (ранжирование по факту)"))
    for ax in axes:
        ax.plot(g.frac_cells * 100, g.frac_accidents * 100, color=COL["oracle"], ls=":", lw=1.4,
                label="потолок (ранжирование по факту)")
        ax.plot([0, 100], [0, 100], color="#bbbbbb", lw=0.8, ls="--", label="случайный порядок")
        ax.set_xlabel("доля отобранных ячеек, %"); ax.set_ylabel("доля накрытых ДТП 2023–2025, %")
        ax.grid(alpha=0.3)
    axes[0].set_title("Вся кривая"); axes[1].set_xlim(0, 20); axes[1].set_ylim(0, 90)
    axes[1].set_title("Верхние 20 % ячеек")
    for q in (1, 5, 10):
        axes[1].axvline(q, color="#999999", lw=0.6)
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Тест 2023–2025 (3 979 ДТП, 4 296 ячеек): прогноз до 2023 г., среднее по 5 сидам", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig1_gain_curves_test.png", dpi=300); plt.close(fig)
    pd.concat(rows).to_csv(FIG / "fig1_gain_curves_test.csv", index=False)

    # fig2: метрики вариантов по сидам
    models = [RID["bin_bal"], RID["bin_none"], RID["month_mean"], RID["poisson"], RID["tw13"], RID["tw16"]]
    sp = [("y2021", "2021", "фолд 2021"), ("y2022", "2022", "фолд 2022"), ("test", "sum", "тест 2023–2025 (сумма)")]
    mets = ["coverage_top_1pct", "coverage_top_5pct", "coverage_top_10pct", "spearman"]
    fig, axes = plt.subplots(len(mets), 3, figsize=(12, 11))
    rows = []
    for i, met in enumerate(mets):
        for j, (s, per, title) in enumerate(sp):
            ax = axes[i, j]
            for x, rid in enumerate(models):
                v = runs[(runs.run_id == rid) & (runs.split == s) & (runs.eval_period == per)][["seed", met]]
                rows.append(v.assign(run_id=rid, split=s, metric=met))
                ax.scatter(np.full(len(v), x) + np.linspace(-0.15, 0.15, len(v)), v[met], s=14, color=COL[rid])
                ax.hlines(v[met].mean(), x - 0.25, x + 0.25, color=COL[rid], lw=1.5)
            for rid, ls in ((RID["prev3"], "--"),):
                b = runs[(runs.run_id == rid) & (runs.split == s) & (runs.eval_period == per)][met]
                if len(b):
                    ax.axhline(b.mean(), color=COL[rid], ls=ls, lw=1, label="ориентир: ДТП за 3 прошлых года")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([LAB[m].replace(", ", ",\n").replace(" → год (среднее)", "\n→ год").replace("Tweedie ", "Tweedie\n") for m in models]
                               if i == len(mets) - 1 else [], fontsize=7)
            ax.grid(alpha=0.3, axis="y")
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(MLAB[met])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", fontsize=7.5)
    fig.suptitle("Годовые модели на road_geo_v2_skdf (32): точки — 5 сидов, черта — среднее", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig2_metrics_by_seed.png", dpi=300); plt.close(fig)
    pd.concat(rows).to_csv(FIG / "fig2_metrics_by_seed.csv", index=False)

    # fig3: калибровка
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    d = dec[dec.eval_period.isin(["2023", "2024", "2025"])]
    for per, mk in zip(["2023", "2024", "2025"], ["o", "s", "^"]):
        q = d[d.eval_period == per]
        axes[0].plot(q["mean_pred"], q["mean_fact"], marker=mk, lw=1, label=f"{per}")
    lim = max(d["mean_pred"].max(), d["mean_fact"].max()) * 1.1
    axes[0].plot([0, lim], [0, lim], color="#999", ls="--", lw=0.8, label="идеальная калибровка")
    axes[0].set_xscale("symlog", linthresh=0.05); axes[0].set_yscale("symlog", linthresh=0.05)
    axes[0].set_xlabel("средний прогноз в дециле, ДТП/ячейку/год"); axes[0].set_ylabel("среднее фактическое число ДТП")
    axes[0].set_title(f"{LAB[SEL]}: калибровка по децилям прогноза (тест)"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    c = cal[cal.run_id == SEL].sort_values("year")
    ax = axes[1]
    ax.plot(sums.index, sums.values, "o-", color="#7f8c8d", label="факт: ДТП за год (все годы панели)")
    ax.plot(c.year, c.sum_pred, "o-", color=COL[SEL], lw=2, label="Σ прогноза модели")
    ax.plot(c.year, c.sum_pred * c.C_hsm, "s--", color="#2e9e5b", label="Σ прогноза × C (HSM)")
    for _, r in c.iterrows():
        ax.text(r.year, max(r.sum_pred, r.sum_pred * r.C_hsm) + 25, f"{r.ratio:.2f}", ha="center", fontsize=8,
                color=COL[SEL])
    ax.axvspan(2014.5, 2020.5, color="#eeeeee", zorder=0)
    ax.text(2017.5, 1730, "годы обучения фолда 2021", ha="center", fontsize=7.5, color="#666666")
    ax.set_ylim(1200, 1800); ax.set_xticks(sums.index); ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("ДТП за год (ось с 1200)"); ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("Σ прогноза и Σ факта по годам; подписи — Σпрогноз/Σфакт\n(2021, 2022 — фолды; 2023–2025 — тест)")
    fig.tight_layout(); fig.savefig(FIG / "fig3_calibration.png", dpi=300); plt.close(fig)
    c.to_csv(FIG / "fig3_calibration_sums.csv", index=False); dec.to_csv(FIG / "fig3_calibration_deciles.csv", index=False)

    # fig4: сверхдисперсия
    show_s = ["тест 2023–2025", "обучение 2015–2022 (in-sample)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, sname in zip(axes, show_s):
        b = odb[(odb["sample"] == sname) & odb.rescaled]
        if b.empty:
            continue
        k = od[(od["sample"] == sname) & od.rescaled]["k_mm_binned"].iloc[0]
        kml = od[(od["sample"] == sname) & od.rescaled]["k_nb_ml"].iloc[0]
        ax.scatter(b.mu, b.msr, color=COL[SEL], s=22, zorder=3, label="группы прогноза (20 квантилей)")
        x = np.linspace(b.mu.min() * 0.8, b.mu.max() * 1.1, 200)
        ax.plot(x, x, color="#999", ls="--", label="Пуассон: var = μ")
        ax.plot(x, x + k * x ** 2, color="#c0392b", label=f"var = μ + kμ², k = {k:.2f} (ММ)")
        ax.plot(x, x + kml * x ** 2, color="#2e9e5b", ls="-.", label=f"NB-подгонка, k = {kml:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("средний прогноз μ в группе, ДТП/ячейку/год")
        ax.set_title(sname); ax.legend(fontsize=7.5)
    axes[0].set_ylabel("средний квадрат остатка (y − μ)²")
    fig.suptitle(f"Сверхдисперсия годового счёта относительно прогноза ({LAB[SEL]}, μ приведён к Σфакта года)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig4_overdispersion.png", dpi=300); plt.close(fig)
    odb.to_csv(FIG / "fig4_overdispersion_bins.csv", index=False)

    # fig5: карта прогноза и факта 2025
    f = vy.period_frame(preds, SEL, "test", "2025")
    polys = [np.array([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in f.h3]
    vmax = float(np.quantile(f[vy.TARGET_COUNT], 0.999))
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax, col, title in ((axes[0], "pred", f"Прогноз на 2025 г. ({LAB[SEL]}, обучение 2015–2022)"),
                           (axes[1], vy.TARGET_COUNT, "Факт 2025 г.")):
        pc = PolyCollection(polys, array=f[col].to_numpy(float), cmap="magma_r", norm=norm, edgecolors="none")
        ax.add_collection(pc); ax.autoscale_view(); ax.set_aspect(1 / np.cos(np.deg2rad(51.55)))
        ax.set_title(title, fontsize=9); ax.set_xlabel("долгота"); ax.set_facecolor("#f4f4f4")
    axes[0].set_ylabel("широта")
    cbar = fig.colorbar(pc, ax=axes, shrink=0.8, extend="max"); cbar.set_label("ДТП в ячейке за год")
    fig.suptitle(f"Саратов, ячейки H3 res 9 (4 296): Σ прогноза {f.pred.sum():.0f}, Σ факта "
                 f"{f[vy.TARGET_COUNT].sum():.0f}", fontsize=10)
    fig.savefig(FIG / "fig5_map_2025.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    f.to_csv(FIG / "fig5_map_2025.csv", index=False)
    print("готово")


if __name__ == "__main__":
    main()
