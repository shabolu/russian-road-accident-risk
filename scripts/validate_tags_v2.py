"""Валидация этапа 3 (теги OSM: NaN вместо 0) до обучения моделей.

1. Покрытие тегов surface / lanes / oneway по классам дорог (доля длины, весь город) -> рис. 1.
2. Старые и новые признаки: доля нулей / NaN, Spearman старое ↔ новое, одиночный AUC на тесте -> рис. 4.
3. «Артефакт карты»: насколько asphalt_surface_length_m объясняется major_road_length_m,
   street_length_m и полнотой разметки (Spearman, частные корреляции, OLS на рангах) -> рис. 2, 3.
4. Чувствительность к порогу размеченной длины (0/50/100/200 м).
5. Ручная проверка ячеек с крупными расхождениями старого и нового.

Запуск: PYTHONPATH=src venv/bin/python scripts/validate_tags_v2.py
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
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.catboost_baseline import (  # noqa: E402
    attach_junction_v2_features, attach_road_grade_features, attach_tags_v2_features,
    build_road_geo_v2_tags_features, build_features, FEATURE_SET_ROAD_GEO_V2_JUNCTIONS,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END  # noqa: E402
from road_accident_risk.osm_road_features import (  # noqa: E402
    aggregate_h3_tags_v2_features, build_h3_tag_v2_segments,
)
from road_accident_risk.paths import (  # noqa: E402
    H3_MONTH_ROAD_FEATURES_PARQUET, OSM_ROAD_H3_SEGMENTS_PARQUET, OSM_ROADS_RAW_PARQUET, PROJECT_ROOT,
)

REPORTS = PROJECT_ROOT / "reports"
FIG = PROJECT_ROOT / "docs" / "analysis" / "features" / "stage3_tags" / "figures"
C_BLUE, C_ORANGE, C_AQUA, C_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b4b3ad"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9ec5f0", "#2a78d6", "#0d366b"])
SEQ_O = LinearSegmentedColormap.from_list("oranges", ["#fdf3ec", "#f5b48a", "#eb6834", "#8a3110"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
CLASS_GROUPS = [("trunk", {"trunk", "trunk_link", "motorway", "motorway_link"}),
                ("primary", {"primary", "primary_link"}), ("secondary", {"secondary", "secondary_link"}),
                ("tertiary", {"tertiary", "tertiary_link"}), ("unclassified", {"unclassified"}),
                ("residential", {"residential"}), ("living_street", {"living_street"}), ("service", {"service"})]
PAIRS = [  # старый -> новый
    ("asphalt_surface_length_m", "asphalt_share_among_tagged"),
    ("bad_surface_share", "bad_surface_share_among_tagged"),
    ("lanes_mean_weighted", "lanes_mean_weighted_v2"),
    ("oneway_share", "oneway_share_v2"),
]
LABELS = {
    "asphalt_surface_length_m": "asphalt_surface_length_m (стар.)",
    "bad_surface_share": "bad_surface_share (стар.)",
    "lanes_mean_weighted": "lanes_mean_weighted (стар.)",
    "oneway_share": "oneway_share (стар.)",
    "asphalt_share_among_tagged": "asphalt_share_among_tagged (нов., NaN)",
    "bad_surface_share_among_tagged": "bad_surface_share_among_tagged (нов., NaN)",
    "lanes_mean_weighted_v2": "lanes_mean_weighted_v2 (нов., NaN)",
    "oneway_share_v2": "oneway_share_v2 (нов.)",
    "major_road_length_m": "major_road_length_m (нов. в наборе)",
    "street_length_m": "street_length_m (нов.)",
    "surface_tagged_share": "surface_tagged_share (диагностика)",
    "road_length_total_m": "road_length_total_m (в наборе)",
}


def save(fig, name: str, data: pd.DataFrame | None = None, index: bool = False) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if data is not None:
        data.to_csv(FIG / f"{name}.csv", index=index)
    print(f"  {name}.png")


# ---------------------------------------------------------------- 1. покрытие тегов
def coverage(st: pd.DataFrame, roads: pd.DataFrame) -> pd.DataFrame:
    tags = roads.set_index("osm_id")["tags_json"].map(json.loads)
    has = pd.DataFrame({"has_oneway": tags.map(lambda d: "oneway" in d),
                        "has_junction_ring": tags.map(lambda d: d.get("junction") in ("roundabout", "circular"))})
    s = st.join(has, on="osm_id")
    L = s.segment_length_m
    rows = []
    for name, cls in [*CLASS_GROUPS, ("все классы", None), ("уличная сеть (без service)", "street")]:
        if cls is None:
            m = np.ones(len(s), bool)
        elif cls == "street":
            m = (s.highway != "service").to_numpy()
        else:
            m = s.highway.isin(cls).to_numpy()
        tot = L[m].sum()
        rows.append({"class": name, "length_km": tot / 1000, "ways": s.osm_id[m].nunique(),
                     "surface_tagged": L[m & s.surface_tagged.to_numpy()].sum() / tot,
                     "lanes_tagged": L[m & s.lanes_v2.notna().to_numpy()].sum() / tot,
                     "oneway_tagged": L[m & s.has_oneway.to_numpy()].sum() / tot,
                     "oneway_v2_share": L[m & s.oneway_v2.astype(bool).to_numpy()].sum() / tot,
                     "asphalt_among_tagged": L[m & (s.is_asphalt_v2 == True).to_numpy()].sum()  # noqa: E712
                     / max(L[m & s.surface_tagged.to_numpy()].sum(), 1e-9),
                     "bad_among_tagged": L[m & (s.is_bad_surface_v2 == True).to_numpy()].sum()  # noqa: E712
                     / max(L[m & s.surface_tagged.to_numpy()].sum(), 1e-9),
                     "share_of_city_length": tot / L.sum()})
    return pd.DataFrame(rows)


def fig1(cov: pd.DataFrame) -> None:
    d = cov[cov["class"].isin([c for c, _ in CLASS_GROUPS])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(d)); w = 0.26
    for i, (col, lab, c) in enumerate([("surface_tagged", "surface", C_BLUE), ("lanes_tagged", "lanes", C_ORANGE),
                                        ("oneway_tagged", "oneway (тег явно)", C_AQUA)]):
        b = ax.bar(x + (i - 1) * w, d[col] * 100, w, color=c, label=lab)
        for xi, v in zip(x + (i - 1) * w, d[col] * 100):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.2, color=INK)
    ax.set_xticks(x, [f"{c}\n{km:,.0f} км".replace(",", " ") for c, km in zip(d["class"], d.length_km)], fontsize=8.5)
    ax.set_ylabel("доля длины класса с тегом, %"); ax.set_ylim(0, 112)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9, ncol=3)
    ax.set_title("Рис. 1. Полнота тегов OSM по классам дорог, Саратов (доля длины в ячейках H3 с дорогами).\n"
                 "Магистрали размечены почти полностью, внутриквартальная сеть — на 7–30 %: «нет тега» ≠ «нет свойства»",
                 loc="left", fontsize=10, color=INK)
    fig.text(0.01, -0.04, "trunk включает motorway и *_link; остальные классы — вместе со своими *_link. "
             "Для oneway отсутствие тега по правилам OSM означает двустороннее движение, поэтому NaN там не вводится.",
             fontsize=8, color=MUTED, ha="left")
    save(fig, "fig1_tag_coverage_by_class", cov)


# ---------------------------------------------------------------- 2–5. статистики
def partial_spearman(c: pd.DataFrame, x: str, y: str, z: list[str]) -> float:
    r = c[[x, y, *z]].rank()
    Z = np.column_stack([np.ones(len(r)), r[z].to_numpy()])
    rx = r[x] - Z @ np.linalg.lstsq(Z, r[x], rcond=None)[0]
    ry = r[y] - Z @ np.linalg.lstsq(Z, r[y], rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def ols_r2(c: pd.DataFrame, y: str, xs: list[str], transform: str) -> float:
    d = c[[y, *xs]].copy()
    d = d.rank() if transform == "rank" else np.log1p(d.clip(lower=0)) if transform == "log1p" else d
    X = np.column_stack([np.ones(len(d)), d[xs].to_numpy()])
    beta = np.linalg.lstsq(X, d[y].to_numpy(), rcond=None)[0]
    res = d[y].to_numpy() - X @ beta
    return float(1 - res.var() / d[y].var(ddof=0))


def single_auc(test: pd.DataFrame, f: str, mask_from: str | None = None) -> tuple[float, int]:
    x = test[f].to_numpy(float); y = test[TARGET].to_numpy(int)
    mk = ~np.isnan(x)
    if mask_from is not None:
        mk &= ~np.isnan(test[mask_from].to_numpy(float))
    return float(roc_auc_score(y[mk], x[mk])), int(mk.sum())


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(OSM_ROAD_H3_SEGMENTS_PARQUET)
    roads = pd.read_parquet(OSM_ROADS_RAW_PARQUET, columns=["osm_id", "highway", "name", "tags_json", "length_m"])
    st = build_h3_tag_v2_segments(seg, roads)

    cov = coverage(st, roads)
    cov.to_csv(REPORTS / "road_geo_v2_tags_coverage_by_class.csv", index=False)
    with pd.option_context("display.width", 250):
        print(cov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fig1(cov)

    panel = attach_tags_v2_features(attach_junction_v2_features(
        attach_road_grade_features(pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET))))
    panel = panel[panel.road_length_total_m > 0].copy()
    panel["month"] = pd.to_datetime(panel["month"])
    cells = panel.drop_duplicates("h3").set_index("h3")
    test = panel[panel.month >= TRAIN_END]
    print(f"ячеек {len(cells)}, тестовых строк {len(test)}")
    J = build_features(panel.head(1), FEATURE_SET_ROAD_GEO_V2_JUNCTIONS)
    B = build_road_geo_v2_tags_features(list(panel.columns))

    # --- статистики старых/новых признаков
    feats = ["asphalt_surface_length_m", "bad_surface_share", "lanes_mean_weighted", "oneway_share",
             "asphalt_share_among_tagged", "bad_surface_share_among_tagged", "lanes_mean_weighted_v2",
             "oneway_share_v2", "major_road_length_m", "street_length_m", "surface_tagged_share",
             "road_length_total_m"]
    new_of = {o: n for o, n in PAIRS}
    corr = cells[sorted(set(feats) | set(J) | set(B))].rank().corr()  # Spearman, попарно по не-NaN
    rows = []
    for f in feats:
        auc, n = single_auc(test, f)
        row = {"feature": f, "zero_share_cells": float((cells[f] == 0).mean()),
               "nan_share_cells": float(cells[f].isna().mean()), "single_auc_test": auc,
               "single_auc_abs": max(auc, 1 - auc), "auc_rows": n}
        if f in new_of:
            row["paired_new"] = new_of[f]; row["rho_old_new"] = float(corr.loc[f, new_of[f]])
            auc_same, _ = single_auc(test, f, mask_from=new_of[f])
            row["single_auc_old_on_new_rows"] = auc_same
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats.to_csv(REPORTS / "road_geo_v2_tags_feature_stats.csv", index=False)
    with pd.option_context("display.width", 250):
        print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- пары |rho| >= 0.85 с участием новых признаков набора B
    newB = [f for f in B if f not in J]
    pairs = [{"feature_a": a, "feature_b": b, "spearman": float(corr.loc[a, b])}
             for a in newB for b in sorted(set(B) | set(J) | {"surface_tagged_share", "asphalt_share_among_tagged"})
             if a != b and abs(corr.loc[a, b]) >= 0.85]
    pd.DataFrame(pairs).to_csv(REPORTS / "road_geo_v2_tags_spearman_pairs.csv", index=False)
    corr.to_csv(REPORTS / "road_geo_v2_tags_spearman_matrix.csv")
    print("пары |rho| >= 0.85 с новыми признаками B:\n", pd.DataFrame(pairs).to_string())
    for f in newB:
        s = corr.loc[f, [g for g in B if g != f]].abs()
        print(f"  {f}: max |rho| внутри B = {s.max():.3f} ({s.idxmax()})")
    print(f"  rho street_length_m ~ road_length_total_m = {corr.loc['street_length_m', 'road_length_total_m']:.3f}")

    # --- артефакт асфальта
    a = "asphalt_surface_length_m"
    c = cells.copy()
    c["surface_tagged_length_m"] = c["surface_tagged_share"] * c["road_length_total_m"]
    c["nonmajor_length_m"] = c["road_length_total_m"] - c["major_road_length_m"]
    c["nonmajor_tagged_length_m"] = (c["surface_tagged_length_m"] - c["major_road_length_m"]).clip(lower=0)
    art = []
    for x in ["major_road_length_m", "street_length_m", "road_length_total_m", "surface_tagged_share",
              "surface_tagged_length_m"]:
        art.append({"kind": "spearman", "what": f"{a} ~ {x}", "value": float(c[[a, x]].rank().corr().iloc[0, 1])})
    art += [
        {"kind": "partial_spearman", "what": f"{a} ~ surface_tagged_share | major, street",
         "value": partial_spearman(c, a, "surface_tagged_share", ["major_road_length_m", "street_length_m"])},
        {"kind": "partial_spearman", "what": f"{a} ~ major | surface_tagged_share, street",
         "value": partial_spearman(c, a, "major_road_length_m", ["surface_tagged_share", "street_length_m"])},
        {"kind": "partial_spearman", "what": f"{a} ~ street | major, surface_tagged_share",
         "value": partial_spearman(c, a, "street_length_m", ["major_road_length_m", "surface_tagged_share"])},
    ]
    for xs in (["major_road_length_m"], ["major_road_length_m", "street_length_m"],
               ["major_road_length_m", "surface_tagged_share"],
               ["major_road_length_m", "street_length_m", "surface_tagged_share"],
               ["road_length_total_m", "surface_tagged_share"], ["surface_tagged_length_m"]):
        for tr in ("rank", "log1p"):
            if tr == "log1p" and "surface_tagged_share" in xs:
                d = c.copy(); d["surface_tagged_share"] = d["surface_tagged_share"] * 1.0
            art.append({"kind": f"ols_r2_{tr}", "what": f"{a} ~ " + " + ".join(xs), "value": ols_r2(c, a, xs, tr)})
    # доля асфальта, приходящаяся на магистрали, и на размеченный residential/service
    asph_by = st[st.is_asphalt_v2 == True].groupby("highway").segment_length_m.sum()  # noqa: E712
    tot_asph = asph_by.sum()
    maj = asph_by[asph_by.index.isin({"trunk", "trunk_link", "primary", "primary_link", "secondary",
                                      "secondary_link", "motorway", "motorway_link"})].sum()
    art += [{"kind": "share_of_asphalt_length", "what": "major classes", "value": maj / tot_asph},
            {"kind": "share_of_asphalt_length", "what": "tertiary", "value": asph_by.get("tertiary", 0) / tot_asph},
            {"kind": "share_of_asphalt_length", "what": "residential", "value": asph_by.get("residential", 0) / tot_asph},
            {"kind": "share_of_asphalt_length", "what": "service", "value": asph_by.get("service", 0) / tot_asph}]
    # одиночные AUC для сопоставления
    for f in ["asphalt_surface_length_m", "major_road_length_m", "street_length_m", "surface_tagged_share",
              "surface_tagged_length_m", "road_length_total_m"]:
        art.append({"kind": "single_auc_test", "what": f, "value": single_auc(test, f)[0]})
    # AUC остатка: асфальт за вычетом линейного (по рангам) прогноза из major+street+surface_tagged_share
    r = c[[a, "major_road_length_m", "street_length_m", "surface_tagged_share"]].rank()
    X = np.column_stack([np.ones(len(r)), r.iloc[:, 1:].to_numpy()])
    resid = r[a] - X @ np.linalg.lstsq(X, r[a], rcond=None)[0]
    tt = test[["h3", TARGET]].merge(resid.rename("resid").reset_index(), on="h3")
    art.append({"kind": "single_auc_test", "what": "остаток rank(asphalt) | major, street, surface_tagged_share",
                "value": float(roc_auc_score(tt[TARGET].astype(int), tt.resid))})
    art = pd.DataFrame(art)
    art.to_csv(REPORTS / "road_geo_v2_tags_asphalt_artifact.csv", index=False)
    print(art.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- чувствительность порога
    sens = []
    for thr in (0.0, 50.0, 100.0, 200.0):
        g = aggregate_h3_tags_v2_features(st, min_tagged_m=thr).set_index("h3").reindex(cells.index)
        if thr == 0.0:  # «нет размеченных» — строго ноль
            g["asphalt_share_among_tagged"] = g["asphalt_share_among_tagged"].where(g.surface_tagged_length_m > 0)
            g["bad_surface_share_among_tagged"] = g["bad_surface_share_among_tagged"].where(g.surface_tagged_length_m > 0)
        tt = test[["h3", TARGET, "bad_surface_share", "lanes_mean_weighted"]].join(
            g[["asphalt_share_among_tagged", "bad_surface_share_among_tagged", "lanes_mean_weighted_v2"]], on="h3")
        for f, old in [("bad_surface_share_among_tagged", "bad_surface_share"),
                       ("asphalt_share_among_tagged", None), ("lanes_mean_weighted_v2", "lanes_mean_weighted")]:
            col = g.loc[cells.index, f]
            row = {"threshold_m": thr, "feature": f, "nan_share_cells": float(col.isna().mean()),
                   "single_auc_test": single_auc(tt, f)[0]}
            if old:
                row["rho_with_old"] = float(pd.concat([col, cells[old]], axis=1).rank().corr().iloc[0, 1])
            row["rho_with_major_road_length_m"] = float(
                pd.concat([col, cells["major_road_length_m"]], axis=1).rank().corr().iloc[0, 1])
            sens.append(row)
    sens = pd.DataFrame(sens)
    sens.to_csv(REPORTS / "road_geo_v2_tags_threshold_sensitivity.csv", index=False)
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig2(c); fig3(c); fig4(stats)
    manual(c, st, roads)


def fig2(c: pd.DataFrame) -> None:
    a = "asphalt_surface_length_m"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    norm = Normalize(0, 1)
    for ax, x, lab in [(axes[0], "major_road_length_m", "major_road_length_m, м (trunk + primary + secondary и *_link)"),
                       (axes[1], "street_length_m", "street_length_m, м (уличная сеть без service)")]:
        o = c.sort_values("surface_tagged_share")
        sc = ax.scatter(o[x], o[a], c=o.surface_tagged_share, cmap=SEQ_O, norm=norm, s=7, alpha=0.85, lw=0)
        mx = max(o[x].max(), o[a].max())
        ax.plot([0, mx], [0, mx], color=MUTED, lw=0.8, ls="--")
        rho = c[[a, x]].rank().corr().iloc[0, 1]
        ax.set_xlabel(lab); ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(f"ρ Спирмена = {rho:.2f}", loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("asphalt_surface_length_m (старый признак), м")
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("surface_tagged_share — доля длины ячейки с тегом surface")
    fig.suptitle("Рис. 2. Старый признак «длина асфальта» против длины магистралей и длины улиц; цвет — полнота тега surface.\n"
                 "Пунктир — y = x: всё, что выше него на панели (а), — размеченные немагистральные дороги, и их количество "
                 "растёт с полнотой разметки", x=0.01, ha="left", fontsize=10.5, color=INK, y=1.02)
    save(fig, "fig2_asphalt_vs_major_street", c[[a, "major_road_length_m", "street_length_m",
                                                  "surface_tagged_share", "road_length_total_m"]].reset_index())


def _hexmap(ax, c: pd.DataFrame, col: str, cmap, norm, title: str):
    polys = [[(lng, lat) for lat, lng in h3.cell_to_boundary(h)] for h in c.index]
    pc = PolyCollection(polys, array=c[col].to_numpy(), cmap=cmap, norm=norm, edgecolor="none")
    ax.add_collection(pc)
    lats = [p[1] for poly in polys for p in poly]; lngs = [p[0] for poly in polys for p in poly]
    ax.set_xlim(min(lngs), max(lngs)); ax.set_ylim(min(lats), max(lats))
    ax.set_aspect(1 / np.cos(np.radians(51.55)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    return pc


def fig3(c: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.2))
    specs = [("surface_tagged_share", SEQ_O, Normalize(0, 1), "а) surface_tagged_share — полнота тега surface"),
             ("asphalt_surface_length_m", SEQ, PowerNorm(0.5, 0, c.asphalt_surface_length_m.quantile(0.99)),
              "б) asphalt_surface_length_m — старый признак, м"),
             ("major_road_length_m", SEQ, PowerNorm(0.5, 0, c.major_road_length_m.quantile(0.99)),
              "в) major_road_length_m — длина магистралей, м")]
    for ax, (col, cmap, norm, title) in zip(axes, specs):
        pc = _hexmap(ax, c, col, cmap, norm, title)
        fig.colorbar(pc, ax=ax, shrink=0.6, pad=0.01, orientation="horizontal")
    rho_ab = c[["surface_tagged_share", "asphalt_surface_length_m"]].rank().corr().iloc[0, 1]
    rho_bc = c[["major_road_length_m", "asphalt_surface_length_m"]].rank().corr().iloc[0, 1]
    fig.suptitle("Рис. 3. Ячейки H3 (res 9) Саратова с дорогами: старый признак асфальта (б) повторяет карту полноты разметки (а), "
                 f"а не только сеть магистралей (в).\nρ Спирмена по ячейкам: (а)–(б) {rho_ab:.2f}, (в)–(б) {rho_bc:.2f}. "
                 "Шкалы (б), (в) — квадратный корень, обрезка по 99-му перцентилю.",
                 x=0.01, ha="left", fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "fig3_map_surface_tagged_vs_asphalt",
         c[["surface_tagged_share", "asphalt_surface_length_m", "major_road_length_m"]].reset_index())


def fig4(stats: pd.DataFrame) -> None:
    order = ["asphalt_surface_length_m", "major_road_length_m", "street_length_m", "surface_tagged_share",
             "road_length_total_m", "bad_surface_share", "bad_surface_share_among_tagged",
             "asphalt_share_among_tagged", "lanes_mean_weighted", "lanes_mean_weighted_v2",
             "oneway_share", "oneway_share_v2"]
    d = stats.set_index("feature").loc[order].reset_index().iloc[::-1]
    col = [C_ORANGE if f in {o for o, _ in PAIRS} else C_GREY if f in ("surface_tagged_share", "road_length_total_m")
           else C_BLUE for f in d.feature]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(d))
    a1.barh(y, d.single_auc_abs - 0.5, left=0.5, color=col, height=0.7)
    for yi, v, raw in zip(y, d.single_auc_abs, d.single_auc_test):
        a1.text(v + 0.004, yi, f"{v:.3f}" + (" (обр.)" if raw < 0.5 else ""), va="center", fontsize=8)
    a1.set_yticks(y, [LABELS[f] for f in d.feature], fontsize=8.5)
    a1.set_xlim(0.5, 0.92); a1.set_xlabel("одиночный ROC-AUC на тесте 2023–2025, max(AUC, 1 − AUC); NaN-строки исключены")
    a1.xaxis.grid(True, color=GRID, lw=0.6); a1.set_axisbelow(True)
    a2.barh(y + 0.18, d.zero_share_cells * 100, height=0.34, color=C_GREY, label="нули")
    a2.barh(y - 0.18, d.nan_share_cells * 100, height=0.34, color=C_ORANGE, label="NaN («тег неизвестен»)")
    for yi, z, n in zip(y, d.zero_share_cells * 100, d.nan_share_cells * 100):
        a2.text(z + 1, yi + 0.18, f"{z:.0f} %", va="center", fontsize=7.5)
        if n > 0:
            a2.text(n + 1, yi - 0.18, f"{n:.0f} %", va="center", fontsize=7.5, color=C_ORANGE)
    a2.set_xlim(0, 100); a2.set_xlabel("доля ячеек, %"); a2.legend(frameon=False, loc="center right", fontsize=8.5)
    a2.xaxis.grid(True, color=GRID, lw=0.6); a2.set_axisbelow(True)
    fig.suptitle("Рис. 4. Одиночный AUC и доля нулей / NaN старых (оранжевые) и новых (синие) признаков, 4 296 ячеек.\n"
                 "Нули старых признаков «нет тега» превращаются в NaN; у новых AUC считается только на ячейках с разметкой",
                 x=0.01, ha="left", fontsize=10.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_single_auc_zero_nan", stats)


def manual(c: pd.DataFrame, st: pd.DataFrame, roads: pd.DataFrame) -> None:
    """Ячейки с крупнейшими расхождениями старого и нового; что там за дороги и теги."""
    rng = np.random.default_rng(42)
    picks = []
    def take(mask, key, label, n=2, asc=False):
        d = c[mask].sort_values(key, ascending=asc).head(n)
        for h in d.index:
            if h not in [p[0] for p in picks]:
                picks.append((h, label))
    # 1) асфальта много, магистралей нет: размеченный residential/service
    c["_excess"] = c.asphalt_surface_length_m - c.major_road_length_m
    take(c.major_road_length_m == 0, "_excess", "асфальт без магистралей: размеченные внутриквартальные")
    # 2) много улиц, асфальт 0 (не размечено)
    take(c.asphalt_surface_length_m == 0, "street_length_m", "асфальт 0 при длинной уличной сети (нет тегов)")
    # 3) старый bad_surface_share мал, новый велик (мало размеченных, почти все грунт)
    c["_bad_gap"] = c.bad_surface_share_among_tagged - c.bad_surface_share
    take(c._bad_gap.notna(), "_bad_gap", "bad: старая доля мала, среди размеченных — велика")
    # 4) старый bad = 0, новый NaN при длинной сети
    take((c.bad_surface_share == 0) & c.bad_surface_share_among_tagged.isna(), "road_length_total_m",
         "bad: старое 0, новое NaN (нет тегов)", n=1)
    # 5) lanes старое 0 -> NaN при наличии магистралей
    take((c.lanes_mean_weighted == 0) & c.lanes_mean_weighted_v2.isna(), "major_road_length_m",
         "lanes: старое 0, новое NaN при магистралях", n=1)
    # 6) oneway: расхождение из-за -1 / кольца
    c["_ow_gap"] = c.oneway_share_v2 - c.oneway_share
    take(c._ow_gap > 0, "_ow_gap", "oneway: кольцо или oneway=-1", n=2)
    tags = roads.set_index("osm_id")
    rows = []
    for h, label in picks:
        r = c.loc[h]
        s = st[st.h3 == h].groupby(["osm_id", "highway"], as_index=False).segment_length_m.sum()
        s = s.sort_values("segment_length_m", ascending=False)
        desc = []
        for o in s.itertuples():
            t = json.loads(tags.loc[o.osm_id, "tags_json"])
            desc.append(f"{o.highway}{' «' + t['name'] + '»' if t.get('name') else ''} {o.segment_length_m:.0f} м "
                        f"[surface={t.get('surface', '—')}, lanes={t.get('lanes', '—')}, oneway={t.get('oneway', '—')}"
                        f"{', junction=' + t['junction'] if t.get('junction') else ''}]")
        lat, lng = h3.cell_to_latlng(h)
        rows.append({"case": label, "h3": h, "lat": round(lat, 5), "lng": round(lng, 5),
                     "road_length_total_m": r.road_length_total_m, "street_length_m": r.street_length_m,
                     "major_road_length_m": r.major_road_length_m, "surface_tagged_share": r.surface_tagged_share,
                     "asphalt_surface_length_m": r.asphalt_surface_length_m,
                     "bad_surface_share": r.bad_surface_share, "bad_surface_share_among_tagged": r.bad_surface_share_among_tagged,
                     "lanes_mean_weighted": r.lanes_mean_weighted, "lanes_mean_weighted_v2": r.lanes_mean_weighted_v2,
                     "oneway_share": r.oneway_share, "oneway_share_v2": r.oneway_share_v2,
                     "n_ways": len(s), "ways_top": " | ".join(desc[:8]),
                     "osm_url": f"https://www.openstreetmap.org/#map=18/{lat:.5f}/{lng:.5f}"})
    m = pd.DataFrame(rows)
    m.to_csv(FIG / "manual_check_cells.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_colwidth", 400):
        for r in m.itertuples():
            print(f"\n[{r.case}] {r.h3} ({r.lat}, {r.lng}) total {r.road_length_total_m:.0f} street {r.street_length_m:.0f} "
                  f"major {r.major_road_length_m:.0f} tagged {r.surface_tagged_share:.2f} asph {r.asphalt_surface_length_m:.0f} "
                  f"bad {r.bad_surface_share:.2f}->{r.bad_surface_share_among_tagged:.2f} lanes {r.lanes_mean_weighted:.2f}->"
                  f"{r.lanes_mean_weighted_v2:.2f} oneway {r.oneway_share:.2f}->{r.oneway_share_v2:.2f}\n   {r.ways_top}")


if __name__ == "__main__":
    main()
