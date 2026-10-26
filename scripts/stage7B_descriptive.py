"""Этап 7, шаг 2B: описательная часть (7.1), ориентиры без модели (7.5), очаги ОДМ (7.7).

Модели не обучаются. Всё пишется в docs/analysis/features/stage7_count12/descriptive/ (CSV + figures/*.png)
и в reports/v2_eval_year_stage7B_baselines*.{csv,parquet}.

Запуск (подкоманды независимы, можно продолжать с любой):
    PYTHONPATH=src venv/bin/python scripts/stage7B_descriptive.py describe   # 7.1: распределение, k, Spearman, тренд
    PYTHONPATH=src venv/bin/python scripts/stage7B_descriptive.py baselines  # 7.5: ориентиры + бутстреп
    PYTHONPATH=src venv/bin/python scripts/stage7B_descriptive.py odm        # 7.7: очаги, recall, «на стыке», карта
    PYTHONPATH=src venv/bin/python scripts/stage7B_descriptive.py all

Функция для шага 3 (recall очагов ОДМ в верхних q % скора):
    from stage7B_descriptive import odm_recall, odm_recall_from_preds   (sys.path += scripts/)
    odm_recall(scores, year)                       # scores: Series h3 -> скор (или DataFrame h3, score)
    odm_recall_from_preds(preds, run_id, split)    # таблица _preds.parquet агента A; по годам + pooled
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk import v2_eval_year as vy  # noqa: E402
from road_accident_risk.paths import PARQUET_DIR, PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "docs/analysis/features/stage7_count12/descriptive"
FIG = OUT / "figures"
TAG = "stage7B_baselines"
YEARS = list(vy.YEARS)
ODM_TYPE_COLS = vy.TYPE_COLS[:-1]          # без count_type_other (смесь видов)
FRACS = (0.01, 0.05, 0.10)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8985"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  ->", FIG / name, flush=True)


# ================================================================ общая подготовка

def odm_flag(df: pd.DataFrame) -> pd.Series:
    """Очаг ОДМ 218.6.015-2015 для строки (ячейка × год): ≥ 3 ДТП одного вида (без 'other') или ≥ 5 любых."""
    return ((df[ODM_TYPE_COLS].max(axis=1) >= 3) | (df[vy.TARGET_COUNT] >= 5)).astype(int)


def load_panel() -> pd.DataFrame:
    """Годовая панель + производные: накопленная история, флаг очага, очаг прошлого года."""
    p = vy.load_year_panel().sort_values(["h3", "year"]).reset_index(drop=True)
    g = p.groupby("h3", sort=False)
    past = g[vy.TARGET_COUNT].cumsum() - p[vy.TARGET_COUNT]
    n_past = p["year"] - YEARS[0]
    p["cum_prev_rate"] = np.where(n_past > 0, past / n_past.replace(0, np.nan), np.nan)
    p["prev3y_rate"] = p["accidents_prev_3y"] / 3
    p["odm_hot"] = odm_flag(p)
    p["odm_hot_prev_year"] = g["odm_hot"].shift(1)
    hp = g["odm_hot"].cumsum() - p["odm_hot"]
    p["odm_hot_past_years"] = np.where(n_past > 0, hp, np.nan)
    return p


def cell_centers(cells) -> pd.DataFrame:
    ll = [h3.cell_to_latlng(c) for c in cells]
    return pd.DataFrame({"h3": list(cells), "lat": [a for a, _ in ll], "lng": [b for _, b in ll]})


def load_edges() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_DIR / "osm_street_graph_consolidated_edges.parquet",
                           columns=["edge_id", "highway", "name", "coords_json"])


def edge_length_by_cell(edges: pd.DataFrame, step_m: float = 20.0) -> pd.DataFrame:
    """Длина рёбер графа этапа 4 по ячейкам и названиям улиц: отрезки режутся на куски ≤ step_m, кусок
    относится к ячейке своей середины."""
    rows = []
    for name, hw, cj in zip(edges["name"], edges["highway"], edges["coords_json"]):
        pts = np.asarray(json.loads(cj), dtype=float)        # [[lat, lng], ...]
        for (la1, lo1), (la2, lo2) in zip(pts[:-1], pts[1:]):
            dy = (la2 - la1) * 111_320; dx = (lo2 - lo1) * 111_320 * np.cos(np.radians(la1))
            L = float(np.hypot(dx, dy)); n = max(1, int(np.ceil(L / step_m)))
            for t in (np.arange(n) + 0.5) / n:
                rows.append((h3.latlng_to_cell(la1 + t * (la2 - la1), lo1 + t * (lo2 - lo1), 9),
                             name if isinstance(name, str) and name else "(без названия)", hw, L / n))
    d = pd.DataFrame(rows, columns=["h3", "name", "highway", "len_m"])
    return d.groupby(["h3", "name", "highway"], as_index=False)["len_m"].sum()


# ================================================================ 7.1 описательная часть

def expected_topk_membership(x: np.ndarray, frac: float) -> np.ndarray:
    """Вероятность попасть в верхние k = max(1, int(n·frac)) ячеек при случайном разрешении связей."""
    n = len(x); k = max(1, int(n * frac))
    t = np.sort(x)[::-1][k - 1]
    above = x > t; tie = x == t
    p = above.astype(float)
    p[tie] = (k - above.sum()) / tie.sum()
    return p


def describe(p: pd.DataFrame) -> None:
    print("== 7.1 описательная часть", flush=True)
    y = p[vy.TARGET_COUNT]
    # --- распределение ячейко-лет
    bins = [0, 1, 2, 3, 4, 5, 10, 1000]; labs = ["0", "1", "2", "3", "4", "5–9", "≥10"]
    p["_b"] = pd.cut(y, bins=bins, right=False, labels=labs)
    dist = pd.crosstab(p["year"], p["_b"]).reindex(columns=labs)
    dist.loc["2015–2025"] = dist.sum()
    dist_share = dist.div(dist.sum(axis=1), axis=0)
    dist.to_csv(OUT / "dist_count_year_by_year.csv"); dist_share.round(5).to_csv(OUT / "dist_count_year_by_year_share.csv")
    vc = y.value_counts().sort_index().rename_axis("count_year").rename("n_cell_years").to_frame()
    vc["share"] = vc["n_cell_years"] / len(y)
    vc["accidents_share"] = vc.index * vc["n_cell_years"] / y.sum()
    vc.to_csv(OUT / "dist_count_year_values.csv")
    p.drop(columns="_b", inplace=True)

    tot = p.groupby("h3")[vy.TARGET_COUNT].sum().sort_values(ascending=False)
    n = len(tot)
    conc = {f"top{int(f*100)}pct_cells_share_accidents": tot.iloc[:max(1, int(n * f))].sum() / tot.sum()
            for f in (0.01, 0.05, 0.10, 0.20)}
    cell_sum = {"n_cells": n, "cells_zero_all_years": float((tot == 0).mean()),
                "cells_ge1_any_year": float((tot > 0).mean()),
                **{f"q{q}": float(tot.quantile(q / 100)) for q in (50, 75, 90, 95, 99)},
                "max_total": int(tot.max()), **conc,
                "cell_years_zero_share": float((y == 0).mean()), "cell_year_max": int(y.max()),
                "cell_year_p99": float(y.quantile(0.99))}
    pd.Series(cell_sum).to_csv(OUT / "dist_cell_totals_summary.csv", header=["value"])
    print(pd.Series(cell_sum).round(4).to_string(), flush=True)

    # --- топ-ячейки: координаты и улицы
    top = tot.head(20).rename("total_2015_2025").to_frame()
    pv = p.pivot(index="h3", columns="year", values=vy.TARGET_COUNT)
    top["mean_per_year"] = top["total_2015_2025"] / len(YEARS)
    top["min_year"] = pv.loc[top.index].min(axis=1); top["max_year"] = pv.loc[top.index].max(axis=1)
    top["years_odm_hot"] = p.groupby("h3")["odm_hot"].sum().reindex(top.index)
    sh = p.groupby("h3")[["count_type_collision", "count_type_pedestrian"]].sum().reindex(top.index)
    top["share_collision"] = sh["count_type_collision"] / top["total_2015_2025"]
    top["share_pedestrian"] = sh["count_type_pedestrian"] / top["total_2015_2025"]
    cc = cell_centers(top.index).set_index("h3"); top = top.join(cc)
    el = edge_length_by_cell(load_edges())
    el.to_parquet(OUT / "_edge_length_by_cell_name.parquet", index=False)
    named = el[el["name"] != "(без названия)"]
    st = (named.groupby(["h3", "name"])["len_m"].sum().reset_index().sort_values(["h3", "len_m"], ascending=[True, False])
          .groupby("h3").head(3).groupby("h3").apply(lambda d: "; ".join(f"{a} ({b:.0f} м)" for a, b in zip(d["name"], d["len_m"])),
                                                     include_groups=False))
    top["streets_top3_by_length"] = st.reindex(top.index)
    hw = el.groupby(["h3", "highway"])["len_m"].sum().reset_index().sort_values("len_m", ascending=False).groupby("h3").head(1)
    top["main_highway_class"] = hw.set_index("h3")["highway"].reindex(top.index)
    top["rank"] = np.arange(1, len(top) + 1)
    top.reset_index().to_csv(OUT / "top20_cells.csv", index=False)
    print(top[["total_2015_2025", "min_year", "max_year", "years_odm_hot", "lat", "lng", "streets_top3_by_length"]]
          .to_string(max_colwidth=70), flush=True)

    # --- сверхдисперсия
    od = []
    for yr, d in p.groupby("year"):
        m, v = d[vy.TARGET_COUNT].mean(), d[vy.TARGET_COUNT].var()
        od.append({"period": str(yr), "mean": m, "var": v, "var_over_mean": v / m, "k_mom": m * m / (v - m)})
    m, v = y.mean(), y.var()
    od.append({"period": "все ячейко-годы", "mean": m, "var": v, "var_over_mean": v / m, "k_mom": m * m / (v - m)})
    # внутри ячейки (во времени): ожидание = эффект ячейки × эффект года (фиксированные эффекты, без модели)
    cm = p.groupby("h3")[vy.TARGET_COUNT].transform("mean")
    yf = p.groupby("year")[vy.TARGET_COUNT].transform("sum") / (y.sum() / len(YEARS))
    mu = cm * yf
    ok = mu > 0
    resid2 = (y - mu) ** 2
    phi = float((resid2[ok] / mu[ok]).sum() / (ok.sum() - p.loc[ok, "h3"].nunique() - (len(YEARS) - 1)))
    k_within = float((mu[ok] ** 2).sum() / (resid2[ok] - mu[ok]).sum())
    od.append({"period": "внутри ячейки (ячейка × год)", "mean": float(mu.mean()), "var": np.nan,
               "var_over_mean": phi, "k_mom": k_within})
    od = pd.DataFrame(od)
    od.to_csv(OUT / "overdispersion.csv", index=False)
    print(od.round(4).to_string(), flush=True)
    cs = p.groupby("h3")[vy.TARGET_COUNT].agg(["mean", "var"]).reset_index()
    cs.to_csv(OUT / "cell_mean_var_2015_2025.csv", index=False)

    # --- устойчивость рангов
    pv = pv[YEARS]
    rho = pd.DataFrame(spearmanr(pv.to_numpy()).statistic, index=YEARS, columns=YEARS)
    rho.to_csv(OUT / "spearman_year_by_year.csv")
    memb = {yr: expected_topk_membership(pv[yr].to_numpy(float), 0.05) for yr in YEARS}
    k5 = max(1, int(len(pv) * 0.05))
    ov = pd.DataFrame([[float((memb[a] * memb[b]).sum() / k5) for b in YEARS] for a in YEARS], index=YEARS, columns=YEARS)
    ov.to_csv(OUT / "top5pct_overlap_year_by_year.csv")
    cov = pd.DataFrame([[vy.coverage_at(pv[b].to_numpy(float), pv[a].to_numpy(float), 0.05) for b in YEARS] for a in YEARS],
                       index=YEARS, columns=YEARS)
    cov.to_csv(OUT / "top5pct_coverage_year_a_to_b.csv")
    lag = []
    for L in range(1, len(YEARS)):
        pairs = [(a, a + L) for a in YEARS if a + L in YEARS]
        lag.append({"lag": L, "n_pairs": len(pairs),
                    "spearman_mean": np.mean([rho.loc[a, b] for a, b in pairs]),
                    "spearman_min": np.min([rho.loc[a, b] for a, b in pairs]),
                    "spearman_max": np.max([rho.loc[a, b] for a, b in pairs]),
                    "top5_overlap_mean": np.mean([ov.loc[a, b] for a, b in pairs]),
                    "top5_coverage_next_mean": np.mean([cov.loc[a, b] for a, b in pairs])})
    lag = pd.DataFrame(lag); lag.to_csv(OUT / "spearman_by_lag.csv", index=False)
    print(lag.round(3).to_string(), flush=True)
    blocks = []
    for a, b in [((2016, 2018), (2019, 2021)), ((2019, 2021), (2022, 2024)), ((2017, 2019), (2023, 2025)),
                 ((2015, 2022), (2023, 2025)), ((2020, 2022), (2023, 2025))]:
        sa = pv.loc[:, a[0]:a[1]].sum(axis=1); sb = pv.loc[:, b[0]:b[1]].sum(axis=1)
        blocks.append({"block_a": f"{a[0]}–{a[1]}", "block_b": f"{b[0]}–{b[1]}",
                       "spearman": spearmanr(sa, sb).statistic,
                       "top5_coverage": vy.coverage_at(sb.to_numpy(float), sa.to_numpy(float), 0.05),
                       "top10_coverage": vy.coverage_at(sb.to_numpy(float), sa.to_numpy(float), 0.10)})
    blocks = pd.DataFrame(blocks); blocks.to_csv(OUT / "spearman_blocks.csv", index=False)
    print(blocks.round(3).to_string(), flush=True)

    # --- тренд
    tr = p.groupby("year").agg(accidents=(vy.TARGET_COUNT, "sum"), cells_ge1=("had_accident_year", "sum"),
                               collision=("count_type_collision", "sum"), pedestrian=("count_type_pedestrian", "sum"),
                               fatal=("count_fatal_year", "sum"), severity=(vy.SEVERITY, "sum"),
                               odm_hot_cells=("odm_hot", "sum"))
    tr["other_types"] = tr["accidents"] - tr["collision"] - tr["pedestrian"]
    base = tr.loc[2015:2022].mean()
    tr["ci95_low"] = tr["accidents"] - 1.96 * np.sqrt(tr["accidents"])
    tr["ci95_high"] = tr["accidents"] + 1.96 * np.sqrt(tr["accidents"])
    for c in ("accidents", "collision", "pedestrian", "other_types", "cells_ge1"):
        tr[f"{c}_ratio_to_2015_2022"] = tr[c] / base[c]
    tr.to_csv(OUT / "trend_by_year.csv")
    print(tr[["accidents", "accidents_ratio_to_2015_2022", "collision_ratio_to_2015_2022",
              "pedestrian_ratio_to_2015_2022", "cells_ge1"]].round(3).to_string(), flush=True)
    # где упало: группы ячеек по истории 2015–2022 и по наличию магистралей
    h = p[p.year <= 2022].groupby("h3")[vy.TARGET_COUNT].sum()
    grp = pd.cut(h, [-1, 0, 2, 5, 10, 20, 1000], labels=["0", "1–2", "3–5", "6–10", "11–20", ">20"])
    major = p.groupby("h3")["major_road_length_m"].first() > 0
    rows = []
    for gname, key in [("история 2015–2022", grp), ("есть магистраль", major.map({True: "да", False: "нет"}))]:
        for gv, cells in key.groupby(key, observed=True).groups.items():
            d = p[p.h3.isin(cells)].groupby("year")[vy.TARGET_COUNT].sum()
            rows.append({"grouping": gname, "group": gv, "n_cells": len(cells),
                         "mean_2015_2022": d.loc[2015:2022].mean(), "mean_2021_2022": d.loc[2021:2022].mean(),
                         **{f"y{yr}": d.loc[yr] for yr in (2023, 2024, 2025)},
                         "ratio_2023_2025_to_2015_2022": d.loc[2023:2025].mean() / d.loc[2015:2022].mean() if d.loc[2015:2022].sum() > 0 else np.nan})
    tg = pd.DataFrame(rows); tg.to_csv(OUT / "trend_by_group.csv", index=False)
    print(tg.round(3).to_string(), flush=True)
    plot_describe(p, vc, cs, od, rho, lag, tr)


def plot_describe(p, vc, cs, od, rho, lag, tr) -> None:
    # fig1: распределение и сверхдисперсия
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    a = ax[0]
    a.bar(vc.index, vc["n_cell_years"], color=C[0], width=0.8)
    a.set_yscale("log"); a.set_xlabel("ДТП в ячейке за год"); a.set_ylabel("число ячейко-лет (лог. шкала)")
    a.set_title(f"а) Распределение count_year, 47 256 ячейко-лет\nнулей {vc.loc[0, 'share']*100:.1f} %, "
                f"максимум {int(vc.index.max())}")
    a = ax[1]
    d = cs[cs["mean"] > 0]
    a.scatter(d["mean"], d["var"].clip(lower=1e-3), s=6, alpha=0.4, color=C[0], edgecolors="none", label="ячейка (11 лет)")
    xx = np.logspace(np.log10(d["mean"].min()), np.log10(d["mean"].max()), 50)
    a.plot(xx, xx, color=GRAY, lw=1.5, ls="--", label="Пуассон: дисперсия = среднему")
    kw = float(od.loc[od.period.str.startswith("внутри"), "k_mom"].iloc[0])
    a.plot(xx, xx + xx ** 2 / kw, color=C[1], lw=2, label=f"отр. биномиальное, k = {kw:.1f}")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("среднее ДТП ячейки за год"); a.set_ylabel("дисперсия по годам")
    a.set_title("б) Внутри ячейки: дисперсия против среднего\n(разброс по 11 годам)")
    a.legend(fontsize=7, loc="upper left")
    a = ax[2]
    oy = od[od.period.str.isdigit()]
    a.plot(oy.period.astype(int), oy.var_over_mean, "o-", color=C[0], lw=2, ms=6, label="между ячейками (за год)")
    phi = float(od.loc[od.period.str.startswith("внутри"), "var_over_mean"].iloc[0])
    a.axhline(phi, color=C[1], lw=2, label=f"внутри ячейки по годам: {phi:.2f}")
    a.axhline(1, color=GRAY, ls="--", lw=1.2, label="Пуассон = 1")
    a.set_ylim(0, 4.3); a.set_xticks(YEARS); a.tick_params(axis="x", rotation=45)
    a.set_ylabel("дисперсия / среднее"); a.set_title("в) Сверхдисперсия по годам")
    a.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); save(fig, "fig1_distribution_overdispersion.png")

    # fig2: Spearman год × год и спад по лагу
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    a = ax[0]
    im = a.imshow(rho.to_numpy(), cmap="Blues", vmin=0.3, vmax=1)
    for i in range(len(YEARS)):
        for j in range(len(YEARS)):
            v = rho.iloc[i, j]
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.75 else "#0b0b0b")
    a.set_xticks(range(len(YEARS)), YEARS, rotation=45); a.set_yticks(range(len(YEARS)), YEARS); a.grid(False)
    a.set_title("а) Spearman числа ДТП ячеек между годами (4 296 ячеек)")
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    a = ax[1]
    a.fill_between(lag.lag, lag.spearman_min, lag.spearman_max, color=C[0], alpha=0.15, lw=0)
    a.plot(lag.lag, lag.spearman_mean, "o-", color=C[0], lw=2, ms=6, label="Spearman (среднее по парам; полоса — мин–макс)")
    a.plot(lag.lag, lag.top5_overlap_mean, "s-", color=C[1], lw=2, ms=6, label="пересечение топ-5 % ячеек (связи — случайно)")
    a.plot(lag.lag, lag.top5_coverage_next_mean, "^-", color=C[2], lw=2, ms=6,
           label="доля ДТП года Y+лаг в топ-5 % ячеек года Y")
    a.axhline(0.05, color=GRAY, ls=":", lw=1); a.text(10, 0.065, "случайно: 0.05", ha="right", fontsize=7, color=GRAY)
    a.set_ylim(0, 1); a.set_xticks(lag.lag); a.set_xlabel("разница лет"); a.set_title("б) Спад устойчивости с лагом")
    a.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); save(fig, "fig2_rank_stability.png")

    # fig3: тренд
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    a = ax[0]
    col = [GRAY if yr <= 2020 else C[3] if yr in (2021, 2022) else C[7] for yr in tr.index]
    a.bar(tr.index, tr.accidents, color=col, width=0.75)
    a.errorbar(tr.index, tr.accidents, yerr=1.96 * np.sqrt(tr.accidents), fmt="none", ecolor="#0b0b0b", lw=1, capsize=3)
    base = tr.loc[2015:2022, "accidents"].mean()
    a.axhline(base, color="#0b0b0b", ls="--", lw=1)
    a.text(2025.4, base + 230, f"среднее 2015–2022: {base:.0f}", fontsize=7, ha="right")
    for yr, v in tr.accidents.items():
        a.text(yr, 60, f"{v}", ha="center", fontsize=7, color="white", rotation=90, va="bottom")
    a.set_ylim(0, 1950); a.set_xticks(tr.index); a.tick_params(axis="x", rotation=45)
    a.set_ylabel("ДТП с пострадавшими (ячейки с дорогами)")
    a.set_title("а) ДТП по годам (±1.96√N); серый — обучение,\nжёлтый — валидация, красный — тест")
    a = ax[1]
    for c, lab, cc in (("accidents", "все ДТП", "#0b0b0b"), ("collision", "столкновения", C[0]),
                       ("pedestrian", "наезды на пешеходов", C[1]), ("other_types", "прочие виды", C[2])):
        a.plot(tr.index, tr[f"{c}_ratio_to_2015_2022"], "o-", color=cc, lw=2, ms=5, label=lab)
    a.axhline(1, color=GRAY, ls="--", lw=1); a.set_xticks(tr.index); a.tick_params(axis="x", rotation=45)
    a.set_ylabel("отношение к среднему 2015–2022"); a.set_title("б) Тренд по видам ДТП")
    a.legend(fontsize=7)
    fig.tight_layout(); save(fig, "fig3_trend_by_year.png")


# ================================================================ 7.5 ориентиры без модели

BASELINES = [  # (имя, скор, is_count, fillna, подпись)
    ("baseline:accidents_prev_year", "accidents_prev_year", True, None, "ДТП за прошлый год"),
    ("baseline:prev3y_rate", "prev3y_rate", True, None, "ДТП за 3 прошлых года (/3)"),
    ("baseline:cum_prev_rate", "cum_prev_rate", True, None, "ДТП за все прошлые годы (на год)"),
    ("baseline:major_road_length_m", "major_road_length_m", False, None, "длина магистралей"),
    ("baseline:road_length_total_m", "road_length_total_m", False, None, "длина уличной сети"),
    ("baseline:skdf_roadway_width_mean_w", "skdf_roadway_width_mean_w", False, 0.0, "ширина проезжей части СКДФ"),
]
LABEL = {f"{n}|baseline": lab for n, _, _, _, lab in BASELINES}


def baselines(p: pd.DataFrame) -> None:
    print("== 7.5 ориентиры", flush=True)
    runs, preds = [], []
    for split in ("y2021", "y2022", "test"):
        for name, sc, is_count, fill, _ in BASELINES:
            r, d = vy.run_baseline(p, sc, split, name=name, is_count=is_count, fillna=fill, return_predictions=True)
            runs.append(r); preds.append(d)
    runs = pd.concat(runs, ignore_index=True); preds = pd.concat(preds, ignore_index=True)
    runs.to_csv(vy.REPORTS / f"v2_eval_year_{TAG}.csv", index=False)
    preds.to_parquet(vy.REPORTS / f"v2_eval_year_{TAG}_preds.parquet", index=False)
    cols = ["split", "run_id", "eval_period", "coverage_top_1pct", "coverage_top_5pct", "coverage_top_10pct",
            "spearman", "roc_auc", "pr_auc", "poisson_deviance", "calib_ratio",
            "coverage_oracle_top_1pct", "coverage_oracle_top_5pct", "coverage_oracle_top_10pct"]
    runs[cols].to_csv(OUT / "baselines_metrics.csv", index=False)
    print(runs[cols].round(3).to_string(), flush=True)

    # бутстреп: доля ДТП в топ-1/5/10 % (B = 1000, по ячейкам)
    bs = []
    periods = {"y2021": ["2021"], "y2022": ["2022"], "test": ["2023", "2024", "2025", "sum"]}
    for split, pers in periods.items():
        for per in pers:
            for name, *_ in BASELINES:
                bs.append(vy.bootstrap_from_predictions(preds, f"{name}|baseline", split, per))
            a = vy.period_frame(preds, f"{BASELINES[0][0]}|baseline", split, per)
            y = a[vy.TARGET_COUNT].to_numpy(float)
            bs.append(vy.bootstrap_coverage(y, y).assign(split=split, eval_period=per, run_id="oracle", base_run_id="",
                                                         target=vy.TARGET_COUNT))
            for v, b in (("baseline:cum_prev_rate", "baseline:prev3y_rate"),
                         ("baseline:prev3y_rate", "baseline:accidents_prev_year"),
                         ("baseline:prev3y_rate", "baseline:major_road_length_m")):
                bs.append(vy.bootstrap_from_predictions(preds, f"{v}|baseline", split, per, base_run_id=f"{b}|baseline"))
        print("  бутстреп", split, flush=True)
    bs = pd.concat(bs, ignore_index=True)
    bs.to_csv(vy.REPORTS / f"v2_eval_year_{TAG}_bootstrap.csv", index=False)
    bs.to_csv(OUT / "baselines_bootstrap.csv", index=False)

    # устойчивость ориентиров по всем годам (без сплитов: скор известен до года Y)
    rows = []
    for yr in YEARS[1:]:
        d = p[p.year == yr]; yv = d[vy.TARGET_COUNT].to_numpy(float)
        for sc, lab in (("accidents_prev_year", "ДТП за прошлый год"), ("prev3y_rate", "ДТП за 3 прошлых года"),
                        ("cum_prev_rate", "ДТП за все прошлые годы"), ("major_road_length_m", "длина магистралей"),
                        (vy.TARGET_COUNT, "потолок (факт)")):
            s = d[sc].to_numpy(float)
            if np.isnan(s).any():
                continue
            c1, c5, c10 = vy.coverages(yv, s)
            rows.append({"year": yr, "score": sc, "label": lab, "coverage_top_1pct": c1, "coverage_top_5pct": c5,
                         "coverage_top_10pct": c10, "spearman": spearmanr(s, yv).statistic})
    by = pd.DataFrame(rows); by.to_csv(OUT / "baselines_by_year_2016_2025.csv", index=False)
    print(by.pivot(index="year", columns="score", values="coverage_top_10pct").round(3).to_string(), flush=True)
    plot_baselines(preds, by)


def plot_baselines(preds: pd.DataFrame, by: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    a = ax[0]; rows = []
    order = ["baseline:cum_prev_rate|baseline", "baseline:prev3y_rate|baseline", "baseline:accidents_prev_year|baseline",
             "baseline:major_road_length_m|baseline", "baseline:road_length_total_m|baseline",
             "baseline:skdf_roadway_width_mean_w|baseline"]
    for i, rid in enumerate(order):
        f = vy.period_frame(preds, rid, "test", "sum")
        g = vy.gain_curve(f[vy.TARGET_COUNT].to_numpy(float), f["pred"].to_numpy(float), n_points=201)
        a.plot(g.frac_cells * 100, g.frac_accidents * 100, color=C[[0, 1, 2, 3, 6, 4][i]], lw=2, label=LABEL[rid])
        rows.append(g.assign(run_id=rid))
    y = f[vy.TARGET_COUNT].to_numpy(float)
    g = vy.gain_curve(y, y, n_points=201)
    a.plot(g.frac_cells * 100, g.frac_accidents * 100, color="#0b0b0b", lw=2, ls="--", label="потолок (ранжирование по факту)")
    rows.append(g.assign(run_id="oracle"))
    a.plot([0, 100], [0, 100], color=GRAY, lw=1, ls=":", label="случайно")
    for q in (1, 5, 10):
        a.axvline(q, color=GRAY, lw=0.6, ls=":")
    a.set_xlim(0, 40); a.set_ylim(0, 100)
    a.set_xlabel("доля отобранных ячеек, %"); a.set_ylabel("доля накрытых ДТП 2023–2025, %")
    a.set_title("а) Тест 2023–2025 (сумма): ориентиры без модели\n(скор известен до 2023 г.)")
    a.legend(fontsize=7, loc="lower right")
    pd.concat(rows).to_csv(FIG / "fig4_gain_curves_baselines_test.csv", index=False)
    a = ax[1]
    for (sc, lab), cc in zip((("cum_prev_rate", "ДТП за все прошлые годы"), ("prev3y_rate", "ДТП за 3 прошлых года"),
                             ("accidents_prev_year", "ДТП за прошлый год"), ("major_road_length_m", "длина магистралей")),
                             (C[0], C[1], C[2], C[3])):
        d = by[by.score == sc]
        a.plot(d.year, d.coverage_top_10pct * 100, "o-", color=cc, lw=2, ms=5, label=lab)
    d = by[by.score == vy.TARGET_COUNT]
    a.plot(d.year, d.coverage_top_10pct * 100, "o--", color="#0b0b0b", lw=1.5, ms=4, label="потолок")
    a.axvspan(2022.5, 2025.5, color=C[7], alpha=0.07, lw=0); a.text(2024, 33, "тест", ha="center", fontsize=8, color=C[7])
    a.set_ylim(30, 90); a.set_xticks(YEARS[1:]); a.set_xlabel("год Y")
    a.set_ylabel("доля ДТП года Y в топ-10 % ячеек, %"); a.set_title("б) Устойчивость ориентиров по годам")
    a.legend(fontsize=7, loc="center", bbox_to_anchor=(0.45, 0.31), ncol=3)
    by.to_csv(FIG / "fig4b_baselines_by_year.csv", index=False)
    fig.tight_layout(); save(fig, "fig4_gain_curves_baselines.png")


# ================================================================ 7.7 очаги ОДМ

def _hot_table(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    if panel is None:
        panel = pd.read_parquet(vy.YEAR_PANEL_PARQUET)
    return panel[["h3", "year"]].assign(odm_hot=odm_flag(panel).to_numpy())


def odm_recall(scores: pd.Series | pd.DataFrame, year: int, fracs=FRACS, panel: pd.DataFrame | None = None,
               score_col: str = "score") -> dict[str, float]:
    """Recall очагов ОДМ года `year` в верхних q % ячеек по скору (функция для шага 3).

    scores — Series h3 -> скор (прогноз, сделанный до года `year`) или DataFrame с колонками h3 и score_col;
    должны быть заданы для всех 4 296 ячеек панели (иначе ValueError). Очаг = ≥ 3 ДТП одного вида (без 'other')
    или ≥ 5 любых в ячейке за календарный год. k = max(1, int(n·q)), связи скоров — честно (как vy.coverage_at:
    это доля очагов среди верхних k, матожидание при случайном порядке внутри групп равных скоров).
    Возвращает {'n_hot', 'n_cells', 'recall_top_1pct', ...}."""
    hot = _hot_table(panel)
    hot = hot[hot["year"] == year].set_index("h3")["odm_hot"]
    s = scores.set_index("h3")[score_col] if isinstance(scores, pd.DataFrame) else scores
    s = s.reindex(hot.index)
    if s.isna().any():
        raise ValueError(f"нет скора для {int(s.isna().sum())} ячеек года {year}")
    h = hot.to_numpy(float); sv = s.to_numpy(float)
    out = {"year": year, "n_cells": len(h), "n_hot": int(h.sum())}
    for f, c in zip(fracs, vy.coverages(h, sv, fracs)):
        out[f"recall_top_{int(round(f * 100))}pct"] = c
    return out


def odm_recall_from_preds(preds: pd.DataFrame, run_id: str, split: str = "test", fracs=FRACS,
                          panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recall очагов ОДМ для прогона из таблицы прогнозов агента A (_preds.parquet: split, run_id, seed, h3, year,
    pred): скор — среднее по сидам; строки по годам оценки + 'pooled' (Σ попавших / Σ очагов по годам)."""
    d = preds[(preds["run_id"] == run_id) & (preds["split"] == split)]
    d = d.groupby(["h3", "year"], as_index=False)["pred"].mean()
    rows = [odm_recall(g.set_index("h3")["pred"], int(yr), fracs, panel) for yr, g in d.groupby("year")]
    r = pd.DataFrame(rows)
    if len(r) > 1:
        pooled = {"year": "pooled", "n_cells": int(r.n_cells.sum()), "n_hot": int(r.n_hot.sum())}
        for f in fracs:
            c = f"recall_top_{int(round(f * 100))}pct"
            pooled[c] = float((r[c] * r.n_hot).sum() / r.n_hot.sum())
        r = pd.concat([r, pd.DataFrame([pooled])], ignore_index=True)
    return r.assign(run_id=run_id, split=split)


def bootstrap_odm_recall(hot_by_year: dict[int, np.ndarray], s_by_year: dict[int, np.ndarray], fracs=FRACS,
                         B: int = 1000, seed: int = 20261019) -> pd.DataFrame:
    """Бутстреп по ячейкам (одна выборка ячеек на все годы) pooled-recall очагов в верхних q %."""
    rng = np.random.default_rng(seed); years = list(hot_by_year); n = len(hot_by_year[years[0]])

    def stat(idx):
        num = np.zeros(len(fracs)); den = 0.0
        for yr in years:
            h = hot_by_year[yr] if idx is None else hot_by_year[yr][idx]
            s = s_by_year[yr] if idx is None else s_by_year[yr][idx]
            if h.sum() == 0:
                continue
            num += np.array(vy.coverages(h, s, fracs)) * h.sum(); den += h.sum()
        return num / den
    point = stat(None); boots = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    return pd.DataFrame({"frac": fracs, "point": point, "ci_low": np.quantile(boots, 0.025, 0),
                         "ci_high": np.quantile(boots, 0.975, 0), "B": B})


ODM_SCORES = [  # (колонка/функция, подпись)
    ("odm_hot_prev_year", "был очагом в прошлом году"),
    ("hot_prev_then_3y", "был очагом в прошлом году, затем ДТП за 3 года"),
    ("odm_hot_past_years_tb", "число лет-очагов в прошлом, затем ДТП за все годы"),
    ("accidents_prev_3y", "ДТП за 3 прошлых года"),
    ("accidents_prev_year", "ДТП за прошлый год"),
    ("cum_prev_rate", "ДТП за все прошлые годы"),
    ("major_road_length_m", "длина магистралей"),
    ("count_year", "потолок: факт года Y"),
]


def rolling_odm(crashes: pd.DataFrame, cells: np.ndarray) -> pd.DataFrame:
    """Критерий ОДМ на скользящих окнах 12 мес. (как в METHODOLOGY 5.1.1): строка (h3, месяц t) — очаг в
    месяцах t+1…t+12. Возвращает h3, t (начало окна = t+1), hot, rule."""
    c = crashes[crashes.h3.isin(cells)].copy()
    c["m"] = (c["year"] - 2015) * 12 + c["datetime"].dt.month - 1
    c["t"] = c["accident_type"].map(vy.ACCIDENT_TYPES).fillna("other")
    nm = int(c["m"].max()) + 1
    idx = {h: i for i, h in enumerate(cells)}
    types = list(vy.ACCIDENT_TYPES.values())
    arr = np.zeros((len(cells), nm, len(types) + 1), dtype=np.int32)
    ti = c["t"].map({t: i for i, t in enumerate(types)}).fillna(len(types)).astype(int).to_numpy()
    np.add.at(arr, (c["h3"].map(idx).to_numpy(), c["m"].to_numpy(), ti), 1)
    cs = np.concatenate([np.zeros((len(cells), 1, arr.shape[2]), np.int32), arr.cumsum(1)], axis=1)
    starts = np.arange(1, nm - 11)            # окно m = start…start+11 полностью в данных; строка t = start−1
    w = cs[:, starts + 12, :] - cs[:, starts, :]
    total = w.sum(2); same = w[:, :, :-1].max(2)
    r3 = same >= 3; r5 = total >= 5
    return pd.DataFrame({"h3": np.repeat(cells, len(starts)), "start": np.tile(starts, len(cells)),
                         "hot": (r3 | r5).ravel(), "r3_only": (r3 & ~r5).ravel(), "r5_only": (r5 & ~r3).ravel(),
                         "both": (r3 & r5).ravel()})


def odm(p: pd.DataFrame) -> None:
    print("== 7.7 очаги ОДМ", flush=True)
    p = p.copy()
    p["hot_prev_then_3y"] = p["odm_hot_prev_year"] * 1000 + p["accidents_prev_3y"]
    p["odm_hot_past_years_tb"] = p["odm_hot_past_years"] * 1000 + p["cum_prev_rate"]
    r3 = p[ODM_TYPE_COLS].max(axis=1) >= 3; r5 = p[vy.TARGET_COUNT] >= 5
    p["rule"] = np.select([r3 & r5, r3, r5], ["оба", "только ≥3 одного вида", "только ≥5 любых"], "нет")
    trig = p[ODM_TYPE_COLS].idxmax(axis=1).str.replace("count_type_", "")
    # --- доли по годам
    rows = []
    for yr, d in p.groupby("year"):
        h = d.odm_hot == 1
        prev = d.odm_hot_prev_year
        rows.append({"year": yr, "n_hot": int(h.sum()), "share_cells": h.mean(),
                     "accidents_in_hot": int(d.loc[h, vy.TARGET_COUNT].sum()),
                     "share_accidents_in_hot": d.loc[h, vy.TARGET_COUNT].sum() / d[vy.TARGET_COUNT].sum(),
                     "rule_both": int((d.rule == "оба").sum()), "rule_r3_only": int((d.rule == "только ≥3 одного вида").sum()),
                     "rule_r5_only": int((d.rule == "только ≥5 любых").sum()),
                     "r3_by_collision": int((r3[d.index] & (trig[d.index] == "collision")).sum()),
                     "r3_by_pedestrian": int((r3[d.index] & (trig[d.index] == "pedestrian")).sum()),
                     "p_hot_given_hot_prev": float(h[prev == 1].mean()) if (prev == 1).any() else np.nan,
                     "p_hot_given_not_hot_prev": float(h[prev == 0].mean()) if (prev == 0).any() else np.nan,
                     "share_hot_new": float((prev[h] == 0).mean()) if yr > 2015 else np.nan})
    by = pd.DataFrame(rows); by.to_csv(OUT / "odm_by_year.csv", index=False)
    print(by.round(3).to_string(), flush=True)
    yrs_hot = p.groupby("h3")["odm_hot"].sum()
    summ = {"cell_years_hot": int(p.odm_hot.sum()), "cell_years": len(p), "share_cell_years": p.odm_hot.mean(),
            "cells_ever_hot": int((yrs_hot > 0).sum()), "share_cells_ever_hot": (yrs_hot > 0).mean(),
            "cells_hot_ge3_years": int((yrs_hot >= 3).sum()), "cells_hot_ge6_years": int((yrs_hot >= 6).sum()),
            "cells_hot_all_11_years": int((yrs_hot == 11).sum()),
            "accidents_in_hot_share_2015_2025": p.loc[p.odm_hot == 1, vy.TARGET_COUNT].sum() / p[vy.TARGET_COUNT].sum(),
            "persistence_P_hot_Y_given_hot_Y-1": float(p.loc[p.odm_hot_prev_year == 1, "odm_hot"].mean()),
            "P_hot_Y_given_not_hot_Y-1": float(p.loc[p.odm_hot_prev_year == 0, "odm_hot"].mean())}

    # --- сверка с METHODOLOGY 5.1.1 (скользящие окна 12 мес., вся сетка bbox и ячейки с дорогами)
    crashes = vy.load_crashes_in_bbox()
    bbox_cells = np.sort(pd.read_parquet(vy.H3_MONTH_ROAD_FEATURES_PARQUET, columns=["h3", "month"],
                                         filters=[("month", "==", pd.Timestamp("2015-01-01"))])["h3"].unique())
    rw = rolling_odm(crashes, bbox_cells)
    road = set(p.h3.unique())
    rw_road = rw[rw.h3.isin(road)]
    rr_cells = set(rw_road.loc[rw_road.hot, "h3"]); cal_cells = set(yrs_hot[yrs_hot > 0].index)
    summ.update({"rolling_bbox_cells": len(bbox_cells), "rolling_rows": len(rw), "rolling_hot_rows": int(rw.hot.sum()),
                 "rolling_r3_only": int(rw.r3_only.sum()), "rolling_r5_only": int(rw.r5_only.sum()),
                 "rolling_both": int(rw.both.sum()), "rolling_base_rate": rw.hot.mean(),
                 "rolling_unique_cells_bbox": int(rw.loc[rw.hot, "h3"].nunique()),
                 "rolling_unique_cells_road": len(rr_cells),
                 "calendar_cells_also_rolling": len(cal_cells & rr_cells),
                 "rolling_cells_never_calendar_hot": len(rr_cells - cal_cells),
                 "calendar_rule_r3_share": float(((p.rule == "оба") | (p.rule == "только ≥3 одного вида")).sum() / p.odm_hot.sum())})
    # персистентность «месяц к месяцу» на скользящих окнах (как в 5.1.1)
    rw_s = rw.sort_values(["h3", "start"])
    nxt = rw_s.groupby("h3")["hot"].shift(-1)
    summ["rolling_persistence_month"] = float(nxt[rw_s.hot & nxt.notna()].mean())
    # календарный год ⊂ скользящих окон: окно, начинающееся в январе — это календарный год
    jan = rw_road[(rw_road.start % 12 == 0)].assign(year=lambda d: 2015 + d.start // 12)
    cal = p[p.year.isin(jan.year.unique())]
    summ["check_jan_window_equals_calendar"] = bool(int(jan.hot.sum()) == int(cal.odm_hot.sum()))
    summ["jan_windows_hot"] = int(jan.hot.sum())
    # сколько ячеек-лет «очаг хотя бы в одном 12-мес. окне, начавшемся в году Y», но не очаг календарного года
    rw_road = rw_road.assign(year=2015 + rw_road.start // 12)
    any_hot = rw_road.groupby(["h3", "year"])["hot"].any()
    cal_hot = p.set_index(["h3", "year"])["odm_hot"].reindex(any_hot.index)
    summ["cell_years_rolling_any_not_calendar"] = int((any_hot & (cal_hot == 0)).sum())
    summ["cell_years_rolling_any"] = int(any_hot.sum())
    s = pd.Series(summ); s.to_csv(OUT / "odm_summary.csv", header=["value"])
    print(s.to_string(), flush=True)

    # --- recall ориентиров в топ-q %
    rec = []
    curves = []
    qgrid = np.linspace(0, 0.20, 81)
    for yr in range(2016, 2026):
        d = p[p.year == yr]; h = d.odm_hot.to_numpy(float)
        for sc, lab in ODM_SCORES:
            s_ = d[sc].to_numpy(float)
            if np.isnan(s_).any():
                continue
            r = {"year": yr, "score": sc, "label": lab, "n_hot": int(h.sum())}
            for f, c in zip(FRACS, vy.coverages(h, s_, FRACS)):
                r[f"recall_top_{int(round(f * 100))}pct"] = c
            rec.append(r)
            if yr >= 2023:
                g = vy.gain_curve(h, s_, n_points=None)
                curves.append(pd.DataFrame({"q": qgrid, "recall": np.interp(qgrid, g.frac_cells, g.frac_accidents),
                                            "year": yr, "score": sc, "n_hot": h.sum()}))
    rec = pd.DataFrame(rec); rec.to_csv(OUT / "odm_recall_by_year.csv", index=False)
    pooled = []
    for (grp, yrs) in (("folds 2021–2022", (2021, 2022)), ("test 2023–2025", (2023, 2024, 2025))):
        for sc, lab in ODM_SCORES:
            d = rec[(rec.score == sc) & rec.year.isin(yrs)]
            hb = {yr: p.loc[p.year == yr, "odm_hot"].to_numpy(float) for yr in yrs}
            sb = {yr: p.loc[p.year == yr, sc].to_numpy(float) for yr in yrs}
            bt = bootstrap_odm_recall(hb, sb)
            r = {"period": grp, "score": sc, "label": lab, "n_hot": int(d.n_hot.sum())}
            for i, f in enumerate(FRACS):
                c = f"recall_top_{int(round(f * 100))}pct"
                r[c] = float((d[c] * d.n_hot).sum() / d.n_hot.sum())
                r[c + "_ci_low"] = bt.ci_low.iloc[i]; r[c + "_ci_high"] = bt.ci_high.iloc[i]
            pooled.append(r)
    pooled = pd.DataFrame(pooled); pooled.to_csv(OUT / "odm_recall_pooled.csv", index=False)
    print(pooled[["period", "label", "n_hot", "recall_top_1pct", "recall_top_5pct", "recall_top_5pct_ci_low",
                  "recall_top_5pct_ci_high", "recall_top_10pct"]].round(3).to_string(), flush=True)
    curves = pd.concat(curves)
    cp = curves.groupby(["score", "q"]).apply(lambda d: (d.recall * d.n_hot).sum() / d.n_hot.sum(),
                                               include_groups=False).rename("recall").reset_index()
    cp.to_csv(FIG / "fig6_odm_recall_curves_test.csv", index=False)
    # самопроверка функции для шага 3 на ориентире
    chk = odm_recall(p[p.year == 2024].set_index("h3")["accidents_prev_3y"], 2024)
    ref = rec[(rec.year == 2024) & (rec.score == "accidents_prev_3y")].iloc[0]
    assert abs(chk["recall_top_5pct"] - ref["recall_top_5pct"]) < 1e-12, (chk, ref)
    print("  odm_recall: самопроверка OK", chk, flush=True)

    # --- «на стыке»: пары соседних ячеек и точечный критерий 200 м
    boundary(p, crashes)
    plot_odm(p, cp, pooled)


def boundary(p: pd.DataFrame, crashes: pd.DataFrame) -> None:
    from sklearn.neighbors import BallTree
    cells = p.h3.unique(); cs = set(cells)
    nb = {c: [n for n in h3.grid_disk(c, 1) if n != c and n in cs] for c in cells}
    rows = []
    vec_cols = ODM_TYPE_COLS + [vy.TARGET_COUNT]
    for yr, d in p.groupby("year"):
        v = d.set_index("h3")[vec_cols]; hot = d.set_index("h3")["odm_hot"]
        active = v.index[v[vy.TARGET_COUNT] > 0]
        pairs = set()
        for a in active:
            if hot[a]:
                continue
            for b in nb[a]:
                if hot[b] or v.at[b, vy.TARGET_COUNT] == 0:
                    continue
                s = v.loc[a].to_numpy() + v.loc[b].to_numpy()
                if s[:-1].max() >= 3 or s[-1] >= 5:
                    pairs.add(tuple(sorted((a, b))))
        pc = {c for pr in pairs for c in pr}
        ring = 0
        for a in v.index[(hot == 0)]:
            s = v.loc[[a, *nb[a]]].to_numpy().sum(0)
            if s[:-1].max() >= 3 or s[-1] >= 5:
                ring += 1
        rows.append({"year": yr, "n_hot": int(hot.sum()), "boundary_pairs": len(pairs), "boundary_cells": len(pc),
                     "nonhot_cells_ring1_hot": ring})
    # точечный критерий: диск диаметром 200 м (радиус 100 м) вокруг ДТП, тот же год
    c = crashes[crashes.h3.isin(cs) & crashes.year.between(2015, 2025)].reset_index(drop=True)
    c["t"] = c["accident_type"].map(vy.ACCIDENT_TYPES).fillna("other")
    hot_set = set(map(tuple, p.loc[p.odm_hot == 1, ["h3", "year"]].to_numpy()))
    R = 6_371_000.0
    for i, (yr, d) in enumerate(c.groupby("year")):
        X = np.radians(d[["lat", "lng"]].to_numpy())
        tree = BallTree(X, metric="haversine")
        ind = tree.query_radius(X, r=100 / R)
        types = d["t"].to_numpy(); h3s = d["h3"].to_numpy()
        center = np.zeros(len(d), bool)
        for j, nbh in enumerate(ind):
            tt = types[nbh]; tt = tt[tt != "other"]
            same = pd.Series(tt).value_counts().max() if len(tt) else 0
            center[j] = len(nbh) >= 5 or same >= 3
        # кластеры: центры, чьи диски пересекаются (≤ 200 м), — union-find
        cidx = np.nonzero(center)[0]
        parent = {j: j for j in cidx}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        if len(cidx):
            ind2 = tree.query_radius(X[cidx], r=200 / R)
            for j, nbh in zip(cidx, ind2):
                for k in nbh:
                    if k in parent:
                        parent[find(j)] = find(k)
        clusters: dict[int, set] = {}
        for j in cidx:
            clusters.setdefault(find(j), set()).update(ind[j])
        n_cl = len(clusters); captured = 0; multi_cell = 0
        for members in clusters.values():
            mc = {h3s[m] for m in members}
            multi_cell += len(mc) > 1
            captured += any((h, yr) in hot_set for h in mc)
        hot_cells_y = {h for (h, y_) in hot_set if y_ == yr}
        center_cells = {h3s[m] for members in clusters.values() for m in members}
        rows[i].update({"point_clusters": n_cl, "point_clusters_multi_cell": multi_cell,
                        "point_clusters_captured_by_cell_hot": captured,
                        "point_clusters_cut": n_cl - captured,
                        "cell_hot_without_point_cluster": len(hot_cells_y - center_cells)})
    b = pd.DataFrame(rows)
    tot = b.drop(columns="year").sum(); tot["year"] = "2015–2025"
    b = pd.concat([b, tot.to_frame().T], ignore_index=True)
    b.to_csv(OUT / "odm_boundary.csv", index=False)
    print(b.to_string(), flush=True)


def plot_odm(p: pd.DataFrame, cp: pd.DataFrame, pooled: pd.DataFrame) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cells = p.h3.unique()
    poly = {c: [(lng, lat) for lat, lng in h3.cell_to_boundary(c)] for c in cells}
    yrs_hot = p.groupby("h3")["odm_hot"].sum()
    d25 = p[p.year == 2025].set_index("h3"); d24 = p[p.year == 2024].set_index("h3")
    edges = load_edges()
    major = edges[edges.highway.isin(["trunk", "primary", "secondary", "motorway"])]
    lines = [[(lo, la) for la, lo in json.loads(cj)] for cj in major.coords_json]
    lat0 = np.mean([v[0][1] for v in poly.values()])
    fig, ax = plt.subplots(1, 2, figsize=(13, 7.2))
    cmap = ListedColormap(["#fde0c5", "#f6a77a", "#e9663a", "#c0391b", "#7a1c0c"])
    norm = BoundaryNorm([1, 2, 4, 7, 10, 12], cmap.N)
    for a in ax:
        a.add_collection(PolyCollection(list(poly.values()), facecolor="#efeeea", edgecolor="#dcdbd5", lw=0.2))
        a.add_collection(LineCollection(lines, colors="#9a9994", linewidths=0.4))
        a.set_aspect(1 / np.cos(np.radians(lat0))); a.grid(False)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    hc = yrs_hot[yrs_hot > 0]
    pc = PolyCollection([poly[c] for c in hc.index], array=hc.to_numpy(float), cmap=cmap, norm=norm,
                        edgecolor="#4d4c48", lw=0.3)
    ax[0].add_collection(pc)
    cb = fig.colorbar(pc, ax=ax[0], fraction=0.03, pad=0.01, ticks=[1.5, 3, 5.5, 8.5, 11])
    cb.ax.set_yticklabels(["1", "2–3", "4–6", "7–9", "10–11"]); cb.set_label("лет, когда ячейка — очаг")
    ax[0].set_title(f"а) Очаги ОДМ 2015–2025: {len(hc)} ячеек хотя бы раз\n(≥ 3 ДТП одного вида или ≥ 5 любых за год)")
    h25 = d25.index[d25.odm_hot == 1]
    was = [c for c in h25 if d24.at[c, "odm_hot"] == 1]
    new = [c for c in h25 if d24.at[c, "odm_hot"] == 0]
    lost = [c for c in d24.index[d24.odm_hot == 1] if d25.at[c, "odm_hot"] == 0]
    ax[1].add_collection(PolyCollection([poly[c] for c in was], facecolor=C[7], edgecolor="#4d4c48", lw=0.4))
    ax[1].add_collection(PolyCollection([poly[c] for c in new], facecolor=C[3], edgecolor="#4d4c48", lw=0.4))
    ax[1].add_collection(PolyCollection([poly[c] for c in lost], facecolor="none", edgecolor=C[0], lw=1.0))
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(facecolor=C[7], label=f"очаг 2025 и 2024 ({len(was)})"),
                          Patch(facecolor=C[3], label=f"очаг 2025, новый ({len(new)})"),
                          Patch(facecolor="none", edgecolor=C[0], label=f"очаг 2024, в 2025 нет ({len(lost)})")],
                 loc="lower right", fontsize=8, frameon=True)
    ax[1].set_title(f"б) Очаги ОДМ 2025 г.: {len(h25)} ячеек")
    xs = np.concatenate([np.array(v)[:, 0] for v in poly.values()]); ys = np.concatenate([np.array(v)[:, 1] for v in poly.values()])
    for a in ax:
        a.set_xlim(xs.min() - 0.005, xs.max() + 0.005); a.set_ylim(ys.min() - 0.005, ys.max() + 0.005)
    fig.text(0.5, 0.02, "серые шестиугольники — ячейки H3 res 9 с дорогами; линии — магистрали OSM (trunk/primary/secondary)",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, "fig5_odm_map.png")
    out = pd.DataFrame({"h3": cells}).assign(years_hot=lambda d: d.h3.map(yrs_hot),
                                             hot_2025=lambda d: d.h3.map(d25.odm_hot), hot_2024=lambda d: d.h3.map(d24.odm_hot))
    out = out.merge(cell_centers(cells), on="h3"); out.to_csv(FIG / "fig5_odm_map.csv", index=False)

    # fig6: recall очагов
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    show = [("count_year", "потолок: факт года Y", "#0b0b0b", "--"),
            ("hot_prev_then_3y", "был очагом в прошлом году → ДТП за 3 года", C[0], "-"),
            ("odm_hot_past_years_tb", "лет-очагов в прошлом → ДТП за все годы", C[6], "-"),
            ("cum_prev_rate", "ДТП за все прошлые годы", C[7], "-"),
            ("accidents_prev_3y", "ДТП за 3 прошлых года", C[1], "-"),
            ("odm_hot_prev_year", "был очагом в прошлом году (только флаг)", C[2], "-"),
            ("accidents_prev_year", "ДТП за прошлый год", C[3], "-"),
            ("major_road_length_m", "длина магистралей", C[4], "-")]
    for sc, lab, cc, ls in show:
        d = cp[cp.score == sc]
        a.plot(d.q * 100, d.recall * 100, color=cc, lw=2, ls=ls, label=lab)
    for q in (1, 5, 10):
        a.axvline(q, color=GRAY, lw=0.6, ls=":")
    a.set_xlim(0, 20); a.set_ylim(0, 102); a.set_xlabel("доля отобранных ячеек (верх по скору), %")
    a.set_ylabel("доля очагов года Y в отборе (recall), %")
    a.set_title("а) Recall очагов ОДМ, тест 2023–2025 (по годам вместе)")
    a.legend(fontsize=7, loc="lower right")
    a = ax[1]
    t = pooled[pooled.period == "test 2023–2025"].set_index("score")
    keys = [s for s, *_ in show]
    x = np.arange(3); w = 0.105
    for i, (sc, lab, cc, _) in enumerate(show):
        vals = [t.at[sc, f"recall_top_{q}pct"] * 100 for q in (1, 5, 10)]
        lo = [t.at[sc, f"recall_top_{q}pct_ci_low"] * 100 for q in (1, 5, 10)]
        hi = [t.at[sc, f"recall_top_{q}pct_ci_high"] * 100 for q in (1, 5, 10)]
        a.bar(x + (i - 3.5) * w, vals, w * 0.9, color=cc, label=lab)
        a.errorbar(x + (i - 3.5) * w, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)], fmt="none",
                   ecolor="#0b0b0b", lw=0.8, capsize=1.5)
    a.set_xticks(x, ["топ-1 %", "топ-5 %", "топ-10 %"]); a.set_ylim(0, 105)
    a.set_ylabel("recall очагов, %"); a.set_title(f"б) То же в точках 1/5/10 % (95 % ДИ, бутстреп по ячейкам)\n"
                                                  f"очагов в тесте: {int(t.n_hot.iloc[0])}; цвета — как на рис. а")
    _ = keys
    fig.tight_layout(); save(fig, "fig6_odm_recall.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    p = load_panel()
    if cmd in ("describe", "all"):
        describe(p)
    if cmd in ("baselines", "all"):
        baselines(p)
    if cmd in ("odm", "all"):
        odm(p)


if __name__ == "__main__":
    main()
