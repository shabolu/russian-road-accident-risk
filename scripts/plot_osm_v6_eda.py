"""Этап 6.1: EDA новых слоёв OSM — таблицы (CSV) и рисунки 1–11 (PNG 300 dpi).

Вход: parquet_output/osm_v6_*.parquet (scripts/build_osm_v6_layers.py),
      docs/analysis/features/stage6_exposure/osm_eda/cells_osm_v6_eda.csv (scripts/eda_osm_v6_cells.py).
Выход: docs/analysis/features/stage6_exposure/osm_eda/{figures/*.png,*.csv, tables/*.csv}

    PYTHONPATH=src venv/bin/python scripts/plot_osm_v6_eda.py [--only fig3,fig6]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.path import Path as MplPath
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_osm_v6_cells import (APART_TYPES, COMMERCIAL_TYPES, ENGELS_CENTER, HOUSE_TYPES,  # noqa: E402
                              SARATOV_CENTER, dedup_points, dist_m, poi_classes)
from plot_blocks_satellite import lonlat_to_px, mosaic, Z  # noqa: E402

from road_accident_risk.paths import PARQUET_DIR, PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "docs/analysis/features/stage6_exposure/osm_eda"
FIG = OUT / "figures"
TAB = OUT / "tables"
BBOX = (51.40, 45.80, 51.70, 46.20)
BUFFER = (51.31, 45.655, 51.79, 46.345)
SAMARA_CENTER = (53.1956255, 50.1014927)  # node 27503945 place=city «Самара»

# палитра (dataviz reference instance, светлая тема): категориальные слоты в фиксированном порядке
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#9a9994"
INK = "#0b0b0b"
INK2 = "#52514e"
DISTRICT_ORDER = ["центр Саратова", "многоэтажная / крупная застройка", "смешанная застройка",
                  "частный сектор", "Энгельс", "малозастроенные окраины"]
DISTRICT_COLORS = dict(zip(DISTRICT_ORDER, C[:6]))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name: str, df: pd.DataFrame | None = None) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if df is not None:
        df.to_csv(FIG / f"{name}.csv", index=False)
    print("saved", name, flush=True)


def caption(fig, text: str, y: float = 0.0) -> None:
    fig.text(0.01, y, text, ha="left", va="top", fontsize=8.5, color=INK2, wrap=True)


def hexes(cells: pd.Series) -> list[np.ndarray]:
    return [np.array([(lo, la) for la, lo in h3.cell_to_boundary(c)]) for c in cells]


def map_axes(ax, bbox=BBOX, pad=0.0) -> None:
    ax.set_xlim(bbox[1] - pad, bbox[3] + pad)
    ax.set_ylim(bbox[0] - pad, bbox[2] + pad)
    ax.set_aspect(1 / math.cos(math.radians((bbox[0] + bbox[2]) / 2)))
    ax.set_xlabel("долгота, °")
    ax.set_ylabel("широта, °")
    ax.tick_params(labelsize=8)


def center_marks(ax) -> None:
    for (la, lo), t in [(SARATOV_CENTER, "центр Саратова\n(Театральная пл., временно)"),
                        (ENGELS_CENTER, "центр Энгельса")]:
        ax.plot(lo, la, marker="*", ms=12, mec="white", mfc=INK, mew=0.8, zorder=10)
        ax.annotate(t, (lo, la), xytext=(6, 6), textcoords="offset points", fontsize=7.5, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8), zorder=11)


def load():
    cells = pd.read_csv(OUT / "cells_osm_v6_eda.csv")
    b = pd.read_parquet(PARQUET_DIR / "osm_v6_buildings.parquet")
    p = poi_classes(pd.read_parquet(PARQUET_DIR / "osm_v6_poi.parquet"))
    s = pd.read_parquet(PARQUET_DIR / "osm_v6_pt_stops.parquet")
    r = pd.read_parquet(PARQUET_DIR / "osm_v6_pt_routes.parquet")
    st = pd.read_parquet(PARQUET_DIR / "osm_v6_streets.parquet")
    tr = pd.read_parquet(PARQUET_DIR / "osm_v6_tram_tracks.parquet")
    lu = pd.read_parquet(PARQUET_DIR / "osm_v6_landuse.parquet")
    return cells, b, p, s, r, st, tr, lu


def dist_band(d_m: pd.Series) -> pd.Series:
    return pd.cut(d_m / 1000, [0, 2.5, 5, 10, 1e9], labels=["0–2.5 км", "2.5–5 км", "5–10 км", "> 10 км"],
                  include_lowest=True)


# ---------------------------------------------------------------- 1. полнота зданий
def fig1(cells, b):
    sb = b[b.in_bbox].copy()
    sb["h3"] = [h3.latlng_to_cell(a, c, 9) for a, c in zip(sb.lat, sb.lon)]
    sb = sb.merge(cells[["h3", "district"]], on="h3", how="left")
    sb["district"] = sb.district.fillna("вне ячеек панели")
    sb["has_levels"] = sb.levels.notna()
    sb["typed"] = sb.building != "yes"
    sb["big"] = sb.footprint_m2 >= 500
    rows = []
    for d in DISTRICT_ORDER + ["вне ячеек панели", "весь bbox"]:
        x = sb if d == "весь bbox" else sb[sb.district == d]
        rows.append({"район": d, "зданий": len(x), "площадь_осн_км2": x.footprint_m2.sum() / 1e6,
                     "доля_с_этажностью": x.has_levels.mean(),
                     "доля_с_этажностью_по_площади": (x.footprint_m2 * x.has_levels).sum() / x.footprint_m2.sum(),
                     "доля_с_типом": x.typed.mean(),
                     "доля_с_этажностью_среди_≥500м2": x[x.big].has_levels.mean(),
                     "доля_с_типом_среди_≥500м2": x[x.big].typed.mean(),
                     "доля_house_detached": x.building.isin(HOUSE_TYPES).mean()})
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    labs = DISTRICT_ORDER + ["весь bbox"]
    tt = t.set_index("район").loc[labs]
    metrics = [("доля_с_этажностью", "с этажностью (доля зданий)"),
               ("доля_с_этажностью_по_площади", "с этажностью (доля площади основания)"),
               ("доля_с_этажностью_среди_≥500м2", "с этажностью среди зданий ≥ 500 м²"),
               ("доля_с_типом", "с типом (building ≠ yes)")]
    y = np.arange(len(labs))
    h = 0.2
    for i, (m, lab) in enumerate(metrics):
        ax.barh(y + (i - 1.5) * h, tt[m] * 100, height=h * 0.9, color=C[i], label=lab)
        for yy, v in zip(y + (i - 1.5) * h, tt[m] * 100):
            ax.text(v + 0.8, yy, f"{v:.0f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [f"{l}\n({int(tt.loc[l, 'зданий']):,} зд.)".replace(",", " ") for l in labs], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("%")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("а) Полнота атрибутов зданий по районам (bbox Саратова)", loc="left")
    ax = axes[1]
    types = ["apartments", "residential", "house", "yes", "retail", "commercial", "school", "kindergarten",
             "industrial", "garages"]
    data, lab2 = [], []
    for ty in types:
        x = sb[sb.building == ty]
        v = x.levels.dropna()
        if len(v):
            data.append(v.clip(upper=30))
            lab2.append(f"{ty}\n{len(v):,} из {len(x):,} ({len(v) / len(x) * 100:.0f} %)".replace(",", " "))
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", ms=2, mfc=GRAY, mec=GRAY))
    for patch in bp["boxes"]:
        patch.set_facecolor("#cde2fb")
        patch.set_edgecolor(C[0])
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_yticks(range(1, len(lab2) + 1), lab2, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("building:levels (обрезано на 30)")
    ax.set_title("б) Этажность по типу здания: размечено из всего", loc="left")
    caption(fig, "Рис. 1. Полнота зданий OSM (снимок 2026-06-03). Районы — по правилу из ячеек H3 (см. README): "
                 "центр — ≤ 2.5 км от Театральной пл.; «многоэтажная» — медиана площади основания ≥ 400 м²; "
                 "«частный сектор» — ≥ 70 % зданий < 200 м²; «Энгельс» — левый берег; окраины — < 5 зданий или > 12 км.",
            y=-0.02)
    save(fig, "fig1_buildings_completeness", t)
    t.to_csv(TAB / "buildings_completeness_by_district.csv", index=False)
    lv = sb.groupby("building").agg(n=("lat", "size"), with_levels=("has_levels", "sum"),
                                    levels_median=("levels", "median"), levels_mean=("levels", "mean"),
                                    footprint_median_m2=("footprint_m2", "median"))
    lv.sort_values("n", ascending=False).head(40).to_csv(TAB / "buildings_by_type_saratov.csv")


# ---------------------------------------------------------------- 2. карта доли с этажностью
def fig2(cells):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    cmap = plt.get_cmap("Blues")
    for ax, col, title in [(axes[0], "bld_levels_area_share", "а) доля площади основания зданий с building:levels"),
                           (axes[1], "bld_typed_share", "б) доля зданий с типом (building ≠ yes)")]:
        has = cells[cells.bld_n >= 3]
        no = cells[cells.bld_n < 3]
        pc0 = PolyCollection(hexes(no.h3), facecolors="#e8e7e3", edgecolors="none")
        ax.add_collection(pc0)
        v = has[col].values
        pc = PolyCollection(hexes(has.h3), array=v, cmap=cmap, edgecolors="none")
        pc.set_clim(0, 1)
        ax.add_collection(pc)
        center_marks(ax)
        map_axes(ax)
        ax.set_title(title, loc="left")
        cb = fig.colorbar(pc, ax=ax, shrink=0.7, pad=0.01)
        cb.set_label("доля")
    caption(fig, "Рис. 2. Полнота атрибутов зданий по ячейкам H3 res 9 панели (ячейки с дорогами). Серым — ячейки, где "
                 "в OSM < 3 зданий. Этажность размечена пятнами (отдельные микрорайоны и центр), а не по типу застройки.",
            y=0.06)
    save(fig, "fig2_map_levels_share", cells[["h3", "lat", "lon", "district", "bld_n", "bld_levels_share",
                                               "bld_levels_area_share", "bld_typed_share"]])


# ---------------------------------------------------------------- 3. здания на спутнике
def pick_microdistrict_window(cells):
    c = cells[(cells.district == "многоэтажная / крупная застройка") & (cells.bld_n >= 15)]
    c = c.sort_values(["bld_levels_share", "bld_fp_m2"], ascending=[True, False]).iloc[0]
    la, lo = c.lat, c.lon
    return la - 0.0042, lo - 0.0068, la + 0.0042, lo + 0.0068


def fig3(cells, b):
    w3 = pick_microdistrict_window(cells)
    windows = [
        ("а) центр (регулярные кварталы)", (51.5420, 46.0050, 51.5505, 46.0185)),
        ("б) крупная застройка с наименьшей долей этажности", w3),
        ("в) частный сектор", (51.5095, 45.9505, 51.5180, 45.9640)),
    ]
    sb = b[b.in_bbox]
    fig, axes = plt.subplots(1, 3, figsize=(21, 8.2))
    rows = []
    cols = {"этажность есть": C[2], "тип есть, этажности нет": C[3], "building=yes, без этажности": C[7]}
    for ax, (title, (lat1, lon1, lat2, lon2)) in zip(axes, windows):
        img, (ox, oy) = mosaic(lat1, lon1, lat2, lon2)
        ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
        x = sb[sb.lat.between(lat1 - 0.001, lat2 + 0.001) & sb.lon.between(lon1 - 0.0015, lon2 + 0.0015)]
        polys, fcs = [], []
        cnt = {k: 0 for k in cols}
        for rr in x.itertuples():
            cat = ("этажность есть" if not np.isnan(rr.levels) else
                   ("тип есть, этажности нет" if rr.building != "yes" else "building=yes, без этажности"))
            if lat1 <= rr.lat <= lat2 and lon1 <= rr.lon <= lon2:
                cnt[cat] += 1
            for sign, ring in json.loads(rr.rings_json):
                if sign != 1:
                    continue
                ring = np.array(ring)
                px, py = lonlat_to_px(ring[:, 1], ring[:, 0])
                polys.append(np.column_stack([px - ox, py - oy]))
                fcs.append(cols[cat])
        ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors=fcs, linewidths=1.3))
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        n = sum(cnt.values())
        ax.set_title(f"{title}\nзданий в окне: {n}; с этажностью {cnt['этажность есть']}, "
                     f"building=yes {cnt['building=yes, без этажности']}", loc="left", fontsize=10.5)
        m_per_px = 156543.03392 * math.cos(math.radians((lat1 + lat2) / 2)) / 2 ** Z
        L = 200 / m_per_px
        x0b, y0b = img.shape[1] * 0.05, img.shape[0] * 0.95
        ax.plot([x0b, x0b + L], [y0b, y0b], color="white", lw=4)
        ax.text(x0b + L / 2, y0b - 12, "200 м", color="white", ha="center", fontsize=10, weight="bold")
        rows.append({"окно": title, "lat1": lat1, "lon1": lon1, "lat2": lat2, "lon2": lon2, **cnt})
    handles = [plt.Line2D([0], [0], color=c, lw=3) for c in cols.values()]
    fig.legend(handles, list(cols), loc="lower center", ncol=3, frameon=False, fontsize=11,
               title="контуры зданий OSM")
    fig.suptitle("Рис. 3. Здания OSM поверх спутникового снимка (Esri World Imagery, z17; только для визуальной проверки)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, "fig3_buildings_satellite", pd.DataFrame(rows))


# ---------------------------------------------------------------- 4. POI по типам и карта
def poi_group(p: pd.DataFrame) -> pd.Series:
    am = p.amenity.fillna("")
    g = np.full(len(p), "прочее", dtype=object)
    g[p.leisure.notna().values] = "досуг/спорт (leisure)"
    g[p.tourism.notna().values] = "туризм (tourism)"
    g[am.isin({"parking", "parking_space", "bicycle_parking", "motorcycle_parking"}).values] = "парковки"
    g[am.isin({"bench", "waste_basket", "waste_disposal", "recycling", "drinking_water", "shelter", "toilets",
               "vending_machine", "post_box", "telephone", "clock", "fountain"}).values] = "уличная мебель / быт. точки"
    g[am.isin({"school", "kindergarten", "college", "university"}).values] = "образование"
    g[(p.healthcare.notna() | am.isin({"hospital", "clinic", "doctors", "dentist", "pharmacy"})).values] = "медицина и аптеки"
    g[am.isin({"restaurant", "cafe", "fast_food", "bar", "pub", "nightclub", "food_court"}).values] = "общепит и бары"
    g[am.isin({"bank", "atm", "post_office", "marketplace", "fuel", "car_wash"}).values] = "услуги (банк, почта, АЗС)"
    g[p.office.notna().values] = "офисы"
    g[p.shop.notna().values] = "магазины (shop)"
    return pd.Series(g, index=p.index)


def fig4(cells, p):
    sp = p[p.in_bbox].copy()
    sm = p[p.region == "samara"].copy()
    sp["group"] = poi_group(sp)
    sm["group"] = poi_group(sm)
    t = pd.DataFrame({"Саратов (bbox)": sp.group.value_counts(), "Самара (центр)": sm.group.value_counts()}).fillna(0)
    t["named_share_saratov"] = sp.groupby("group").name.apply(lambda x: x.notna().mean())
    t = t.sort_values("Саратов (bbox)", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["Саратов (bbox)"], height=0.38, color=C[0], label="Саратов + Энгельс (bbox проекта)")
    ax.barh(y - 0.2, t["Самара (центр)"], height=0.38, color=C[1], label="Самара (bbox этапа 5)")
    for yy, v, nm in zip(y, t["Саратов (bbox)"], t["named_share_saratov"]):
        ax.text(v + 30, yy + 0.2, f"{int(v):,}".replace(",", " ") + f"  (с назв. {nm * 100:.0f} %)",
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, t.index, fontsize=9)
    ax.set_xlabel("число объектов (узлы + центроиды полигонов)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("а) POI по группам", loc="left")
    ax = axes[1]
    has = cells[cells.poi_attr_n > 0]
    no = cells[cells.poi_attr_n == 0]
    ax.add_collection(PolyCollection(hexes(no.h3), facecolors="#eeede9", edgecolors="none"))
    bins = [1, 2, 4, 8, 16, 32, 100]
    cm = plt.get_cmap("Blues")
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(bins, 256)
    pc = PolyCollection(hexes(has.h3), array=has.poi_attr_n.clip(upper=99).values, cmap=cm, norm=norm,
                        edgecolors="none")
    ax.add_collection(pc)
    center_marks(ax)
    map_axes(ax)
    cb = fig.colorbar(pc, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label("POI-аттракторов в ячейке")
    ax.set_title("б) Плотность POI-аттракторов (shop, office, общепит, аптеки, банки, клиники)", loc="left")
    caption(fig, "Рис. 4. POI OSM: узлы и центроиды полигонов с shop/amenity/office/leisure/tourism/healthcare. "
                 "Аттракторы — набор из PLAN 6.3. Светло-серые ячейки — нет аттракторов.", y=0.02)
    out = t.reset_index().rename(columns={"index": "group"})
    save(fig, "fig4_poi_types_density", out)
    rows = []
    for reg, d in [("saratov_bbox", sp), ("samara", sm)]:
        for k in ["shop", "amenity", "office", "leisure", "tourism", "healthcare"]:
            vc = d[k].value_counts()
            for v, n in vc.head(25).items():
                rows.append({"region": reg, "key": k, "value": v, "n": n,
                             "named_share": d.loc[d[k] == v, "name"].notna().mean()})
    pd.DataFrame(rows).to_csv(TAB / "poi_by_tag_value.csv", index=False)


# ---------------------------------------------------------------- 5. полнота POI по районам
def fig5(cells, p):
    sp = p[p.in_bbox].copy()
    sp["h3"] = [h3.latlng_to_cell(a, c, 9) for a, c in zip(sp.lat, sp.lon)]
    sp["group"] = poi_group(sp)
    sp = sp.merge(cells[["h3", "district"]], on="h3", how="left")
    g = cells.groupby("district").agg(cells_n=("h3", "size"), street_km=("street_length_m", "sum"),
                                      bld_n=("bld_n", "sum"), attr=("poi_attr_n", "sum"), poi=("poi_n", "sum"),
                                      named=("poi_named_n", "sum"))
    g["street_km"] /= 1000
    furn = sp[sp.group == "уличная мебель / быт. точки"].groupby("district").size()
    g["furniture"] = furn
    g = g.fillna(0).loc[DISTRICT_ORDER]
    g["attr_per_km"] = g.attr / g.street_km
    g["attr_per_1000_bld"] = g.attr / g.bld_n * 1000
    g["furniture_per_km"] = g.furniture / g.street_km
    g["named_share"] = g.named / g.poi
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharey=True)
    specs = [("attr_per_km", "аттракторов на 1 км улиц"), ("attr_per_1000_bld", "аттракторов на 1000 зданий"),
             ("furniture_per_km", "«уличной мебели» на 1 км улиц\n(скамьи, урны, контейнеры)"),
             ("named_share", "доля POI с названием")]
    y = np.arange(len(g))
    for ax, (col, lab) in zip(axes, specs):
        ax.barh(y, g[col], color=[DISTRICT_COLORS[d] for d in g.index], height=0.65)
        for yy, v in zip(y, g[col]):
            ax.text(v, yy, f" {v:.2f}" if v < 10 else f" {v:.0f}", va="center", fontsize=8, color=INK2)
        ax.set_title(lab, loc="left", fontsize=10)
        ax.margins(x=0.18)
    axes[0].set_yticks(y, g.index)
    axes[0].invert_yaxis()
    caption(fig, "Рис. 5. Полнота POI по районам Саратова. «Уличная мебель» не связана с активностью: её плотность "
                 "показывает, где работали картографы-энтузиасты.", y=0.0)
    save(fig, "fig5_poi_by_district", g.reset_index())


# ---------------------------------------------------------------- 6. остановки и трамвай
def stop_dedup(s: pd.DataFrame) -> pd.DataFrame:
    s = s.reset_index(drop=True).copy()
    s["cluster"] = dedup_points(s.lat.values, s.lon.values)
    return s


def fig6(s, r, tr):
    out_rows = []
    res = {}
    for reg in ["saratov_buffer", "samara"]:
        x = stop_dedup(s[s.region == reg])
        if reg == "saratov_buffer":
            x["in"] = x.in_bbox
        else:
            x["in"] = True
        cl = x.groupby("cluster").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size"),
                                      inn=("in", "max"), schemes=("scheme", lambda v: ",".join(sorted(set(v)))))
        xi = x[x["in"]]
        cli = cl[cl.inn]
        res[reg] = (xi, cli)
        out_rows.append({"region": reg, "objects_before": len(xi), "after_dedup_30m": len(cli),
                         "ptv2_objects_share": (xi.scheme == "ptv2").mean(),
                         "clusters_only_legacy": (cli.schemes == "legacy").mean(),
                         "clusters_mixed": (cli.schemes == "legacy,ptv2").mean(),
                         "clusters_size_1": (cli.n == 1).mean(), "clusters_size_ge4": (cli.n >= 4).mean(),
                         "nodes_platform": (xi.public_transport == "platform").sum(),
                         "nodes_stop_position": (xi.public_transport == "stop_position").sum(),
                         "legacy_bus_stop_only": ((xi.highway == "bus_stop") & xi.public_transport.isna()).sum(),
                         "tram_stop": (xi.railway == "tram_stop").sum(),
                         "tram_track_km": tr[(tr.region == reg) & (tr.touches_bbox if reg == "saratov_buffer" else True)].length_m.sum() / 1000})
    t = pd.DataFrame(out_rows)
    xi, cli = res["saratov_buffer"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    st = pd.read_parquet(PARQUET_DIR / "osm_v6_streets.parquet", columns=["highway", "coords_json", "touches_bbox"])
    segs = [np.array(json.loads(c))[:, ::-1] for c in st[st.touches_bbox].coords_json]
    ax.add_collection(LineCollection(segs, colors="#d6d5d0", linewidths=0.35))
    tsegs = [np.array(json.loads(c))[:, ::-1] for c in tr[tr.region == "saratov_buffer"].coords_json]
    ax.add_collection(LineCollection(tsegs, colors=C[7], linewidths=1.4, label="трамвайные пути (railway=tram)"))
    ax.scatter(cli.lon, cli.lat, s=4, c=C[0], lw=0, label=f"остановки после дедупа 30 м ({len(cli):,})".replace(",", " "))
    center_marks(ax)
    map_axes(ax)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, markerscale=3)
    ax.set_title("а) Остановки ОТ и трамвайные пути, bbox Саратова", loc="left")
    # окно до/после дедупа
    ax = axes[1]
    lat1, lon1, lat2, lon2 = 51.527, 46.010, 51.541, 46.040
    ax.add_collection(LineCollection(segs, colors="#bdbcb7", linewidths=0.8))
    w = xi[xi.lat.between(lat1, lat2) & xi.lon.between(lon1, lon2)]
    for sch, col, mk in [("ptv2", C[0], "o"), ("legacy", C[1], "s")]:
        ww = w[w.scheme == sch]
        ax.scatter(ww.lon, ww.lat, s=26, c=col, marker=mk, lw=0, alpha=0.85,
                   label=f"{'PTv2 (platform / stop_position)' if sch == 'ptv2' else 'старая схема (только highway=bus_stop)'}: {len(ww)}")
    cw = cli[cli.lat.between(lat1, lat2) & cli.lon.between(lon1, lon2)]
    ax.scatter(cw.lon, cw.lat, s=140, facecolors="none", edgecolors=INK, lw=1.0,
               label=f"кластер после дедупа 30 м: {len(cw)}")
    map_axes(ax, (lat1, lon1, lat2, lon2))
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.set_title("б) Центр: объекты разметки до и после дедупа", loc="left")
    caption(fig, f"Рис. 6. Остановки ОТ OSM. bbox Саратова: {len(xi):,} объектов → {len(cli):,} остановок после "
                 f"склейки в радиусе 30 м (single linkage). PTv2 даёт 2–3 объекта на остановку (платформа + точка остановки).".replace(",", " "),
            y=0.03)
    save(fig, "fig6_stops_tram", t)
    t.to_csv(TAB / "pt_stops_summary.csv", index=False)
    rr = r[r.region.notna()].copy()
    rr["has_stops"] = (rr.n_stop_members + rr.n_platform_members) > 0
    rs = rr.groupby(["region", "route"]).agg(relations=("rel_id", "size"), refs=("ref", "nunique"),
                                             with_stops=("has_stops", "mean"), ptv2=("ptv", lambda v: (v == "2").mean()),
                                             ways_median=("n_way_members", "median"),
                                             stops_median=("n_stop_members", "median"),
                                             platforms_median=("n_platform_members", "median"),
                                             length_km_median=("length_km", "median")).reset_index()
    rs.to_csv(TAB / "pt_routes_summary.csv", index=False)
    print(t.to_string())
    print(rs.to_string())


# ---------------------------------------------------------------- 7. уличная сеть буфера
def street_components(st: pd.DataFrame) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    ids = {}
    rows, cols = [], []
    for k, nodes in enumerate(st.node_ids):
        for n in nodes:
            j = ids.setdefault(int(n), len(ids))
            rows.append(k)
            cols.append(j)
    m = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(st), len(ids))).tocsr()
    adj = m @ m.T
    return connected_components(adj, directed=False)[1]


def fig7(st):
    st = st.copy()
    st["comp"] = street_components(st)
    main = st.groupby("comp").length_m.sum().idxmax()
    st["main"] = st.comp == main
    st["in_bbox"] = st.mid_lat.between(BBOX[0], BBOX[2]) & st.mid_lon.between(BBOX[1], BBOX[3])
    cls = {"trunk": "trunk", "trunk_link": "trunk", "motorway": "trunk", "motorway_link": "trunk",
           "primary": "primary", "primary_link": "primary", "secondary": "secondary", "secondary_link": "secondary",
           "tertiary": "tertiary", "tertiary_link": "tertiary", "residential": "residential / living",
           "living_street": "residential / living", "unclassified": "unclassified"}
    st["cls"] = st.highway.map(cls)
    order = ["trunk", "primary", "secondary", "tertiary", "unclassified", "residential / living"]
    colors = dict(zip(order, [C[7], C[1], C[3], C[0], C[2], "#b8b7b2"]))
    widths = dict(zip(order, [1.6, 1.2, 1.0, 0.8, 0.45, 0.3]))
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    for c in order[::-1]:
        x = st[st.cls == c]
        segs = [np.array(json.loads(v))[:, ::-1] for v in x.coords_json]
        ax.add_collection(LineCollection(segs, colors=colors[c], linewidths=widths[c], label=c))
    nm = st[~st.main]
    ax.add_collection(LineCollection([np.array(json.loads(v))[:, ::-1] for v in nm.coords_json], colors=INK,
                                     linewidths=1.2, label="вне главной компоненты"))
    ax.plot([BBOX[1], BBOX[3], BBOX[3], BBOX[1], BBOX[1]], [BBOX[0], BBOX[0], BBOX[2], BBOX[2], BBOX[0]],
            color=INK, lw=1.4, ls="--", label="bbox проекта")
    center_marks(ax)
    map_axes(ax, BUFFER)
    ax.legend(loc="lower left", fontsize=8, frameon=True, ncol=2)
    ax.set_title("а) Уличная сеть буферной области (bbox ± 10 км), без service", loc="left")
    ax = axes[1]
    t = st.groupby(["cls", "in_bbox"]).length_m.sum().unstack().fillna(0) / 1000
    t = t.loc[order]
    t.columns = ["буфер (вне bbox)", "внутри bbox"]
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["внутри bbox"], height=0.38, color=C[0], label="внутри bbox")
    ax.barh(y - 0.2, t["буфер (вне bbox)"], height=0.38, color=C[1], label="в буфере (вне bbox)")
    for yy, a, b_ in zip(y, t["внутри bbox"], t["буфер (вне bbox)"]):
        ax.text(a + 10, yy + 0.2, f"{a:.0f}", va="center", fontsize=8, color=INK2)
        ax.text(b_ + 10, yy - 0.2, f"{b_:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("протяжённость, км (по середине way)")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    share_main = st[st.main].length_m.sum() / st.length_m.sum()
    share_main_bbox = st[st.main & st.in_bbox].length_m.sum() / st[st.in_bbox].length_m.sum()
    ax.set_title(f"б) Протяжённость по классам; главная компонента — {share_main * 100:.1f} % длины\n"
                 f"(внутри bbox — {share_main_bbox * 100:.1f} %), компонент всего {st.comp.nunique():,}".replace(",", " "),
                 loc="left")
    caption(fig, "Рис. 7. Уличная сеть OSM в буферной области для центральности (6.6). Связность — по общим узлам ways "
                 "(без учёта односторонности). Буфер добавляет в основном трассы (СКАД, Саратов – Озинки, «Каспий») и пригороды, а не городскую сеть.",
            y=0.04)
    tt = t.reset_index().rename(columns={"index": "class"})
    tt.loc[len(tt)] = ["главная компонента, доля длины (вся область | bbox)", share_main, share_main_bbox]
    save(fig, "fig7_street_network_buffer", tt)
    comp = st.groupby("comp").agg(km=("length_m", "sum"), ways=("way_id", "size"),
                                  in_bbox_km=("length_m", lambda v: v[st.loc[v.index, "in_bbox"]].sum()))
    comp["km"] /= 1000
    comp["in_bbox_km"] /= 1000
    comp.sort_values("km", ascending=False).head(30).to_csv(TAB / "street_components_top30.csv")
    print("components", st.comp.nunique(), "main share", share_main, share_main_bbox)
    trunk_names = st[(st.cls == "trunk") & ~st.in_bbox].groupby("name").length_m.sum().sort_values(ascending=False) / 1000
    trunk_names.head(15).to_csv(TAB / "buffer_trunk_by_name.csv")
    print(trunk_names.head(10))


# ---------------------------------------------------------------- 8. ДТП по децилям
def bin_with_zero(v: pd.Series, q: int = 10) -> pd.Series:
    out = pd.Series(index=v.index, dtype=object)
    out[v.isna()] = "нет данных"
    out[v == 0] = "0"
    pos = v[v > 0]
    if len(pos):
        k = pd.qcut(pos.rank(method="first"), min(q, max(len(pos) // 50, 1)), labels=False)
        lo = pos.groupby(k).min()
        hi = pos.groupby(k).max()
        lab = {i: (f"{lo[i]:.3g}–{hi[i]:.3g}") for i in lo.index}
        out[pos.index] = k.map(lab)
    return out


INT_EDGES = {"poi_attr_n": [0, 1, 2, 3, 5, 8, 16, 1e9], "bld_levels_median": [0, 1, 2, 3, 5, 6, 9, 12, 1e9]}


def int_bins(v: pd.Series, edges: list[float]) -> pd.Series:
    out = pd.Series("нет данных", index=v.index, dtype=object)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v < hi) if lo > 0 else (v <= 0)
        lab = "0" if lo == 0 else (f"{lo:g}" if hi - lo == 1 else (f"≥ {lo:g}" if hi > 1e8 else f"{lo:g}–{hi - 1:g}"))
        if lo == 0:
            m = v == 0
        out[m] = lab
    return out


def fig8(cells):
    specs = [("poi_attr_n", "POI-аттракторы в ячейке", True),
             ("comm_fp_m2", "коммерческая площадь основания, м²\n(building ∈ commercial/retail/office/…)", True),
             ("bld_levels_median", "медиана этажности зданий ячейки\n(где этажность есть)", True),
             ("bld_fp_m2", "суммарная площадь основания зданий, м²", True),
             ("dist_center_m", "расстояние до Театральной пл., км (временно)", False),
             ("dist_stop_m", "расстояние до ближайшей остановки, м", False)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    rows = []
    base = cells.dtp_share.mean()
    for ax, (col, lab, zero) in zip(axes.flat, specs):
        v = cells[col] / (1000 if col == "dist_center_m" else 1)
        if col in INT_EDGES:
            vv = v.round() if col == "bld_levels_median" else v
            bins = int_bins(vv, INT_EDGES[col])
        elif zero:
            bins = bin_with_zero(v)
        else:
            k = pd.qcut(v, 10, labels=False)
            lo, hi = v.groupby(k).min(), v.groupby(k).max()
            bins = k.map({i: f"{lo[i]:.3g}–{hi[i]:.3g}" for i in lo.index})
        g = cells.assign(bin=bins).groupby("bin", sort=False).agg(dtp=("dtp_share", "mean"), n=("h3", "size"),
                                                                  vmin=(col, "min"))
        g = g.sort_values("vmin", na_position="first") if zero else g.sort_values("vmin")
        if "нет данных" in g.index:
            g = pd.concat([g.loc[["нет данных"]], g.drop("нет данных")])
        x = np.arange(len(g))
        colors = [GRAY if i in ("нет данных", "0") else C[0] for i in g.index]
        ax.bar(x, g.dtp * 100, color=colors, width=0.75)
        ax.axhline(base * 100, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(x, [f"{i}\n(n={n})" for i, n in zip(g.index, g.n)], rotation=60, ha="right", fontsize=6.5)
        ax.set_ylabel("доля ячейко-месяцев с ДТП, %")
        ax.set_title(lab, loc="left", fontsize=10)
        rho = spearmanr(cells[col], cells.dtp_share, nan_policy="omit")[0]
        ax.text(0.02, 0.95, f"ρ Спирмена по ячейкам = {rho:.2f}", transform=ax.transAxes, fontsize=8.5, color=INK,
                va="top")
        for i, row in g.iterrows():
            rows.append({"variable": col, "bin": i, "cells": row.n, "dtp_share": row.dtp, "rho": rho})
    caption(fig, "Рис. 8. Доля ячейко-месяцев с ДТП в следующем месяце (вся панель 2015–2025, 4 296 ячеек × 132 мес.) "
                 "по децилям новых величин. Серым — «0» и «нет данных», пунктир — средняя доля. Описательно, без моделей и "
                 "без контроля длины дорог.", y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig8_dtp_by_deciles", pd.DataFrame(rows))


# ---------------------------------------------------------------- 9. артефакт разметки
def partial_spearman(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> float:
    d = df[[x, y] + controls].dropna()
    R = d.rank()
    Z = np.column_stack([np.ones(len(R))] + [R[c].values for c in controls])
    rx = R[x].values - Z @ np.linalg.lstsq(Z, R[x].values, rcond=None)[0]
    ry = R[y].values - Z @ np.linalg.lstsq(Z, R[y].values, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def fig9(cells):
    c = cells[cells.bld_n >= 5].copy()
    c["dist_km"] = c.dist_center_m / 1000
    metrics = [("bld_levels_area_share", "доля площади зданий\nс этажностью"),
               ("bld_typed_share", "доля зданий с типом\n(building ≠ yes)"),
               ("poi_per_km_street", "POI на 1 км улиц")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    rows = []
    c["surf_bin"] = pd.cut(c.surface_tagged_share, [-0.01, 0.0001, 0.2, 0.4, 0.6, 0.8, 1.0],
                           labels=["0", "0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"])
    c["dist_bin"] = pd.cut(c.dist_km, [0, 2, 4, 6, 8, 10, 12, 15, 25],
                           labels=["0–2", "2–4", "4–6", "6–8", "8–10", "10–12", "12–15", "> 15"])
    for j, (m, lab) in enumerate(metrics):
        for i, (xb, xl, xvar) in enumerate([("surf_bin", "полнота тега surface на улицах ячейки (этап 3)",
                                             "surface_tagged_share"),
                                            ("dist_bin", "расстояние до центра Саратова, км", "dist_km")]):
            ax = axes[i, j]
            g = c.groupby(xb, observed=True)[m].agg(["mean", "median", "size",
                                                     lambda v: v.quantile(0.25), lambda v: v.quantile(0.75)])
            g.columns = ["mean", "median", "n", "q25", "q75"]
            x = np.arange(len(g))
            ax.fill_between(x, g.q25, g.q75, color="#cde2fb", lw=0, label="межквартильный размах")
            ax.plot(x, g["mean"], color=C[0], lw=2, marker="o", ms=5, label="среднее")
            ax.plot(x, g["median"], color=C[1], lw=1.5, ls="--", label="медиана")
            ax.set_xticks(x, [f"{a}\n(n={n})" for a, n in zip(g.index, g.n)], fontsize=7.5)
            ax.set_xlabel(xl, fontsize=9)
            ax.set_ylabel(lab.replace("\n", " "), fontsize=9)
            rho = spearmanr(c[m], c[xvar], nan_policy="omit")[0]
            ax.set_title(f"{lab.replace(chr(10), ' ')}: ρ = {rho:.2f}", loc="left", fontsize=10)
            if m == "poi_per_km_street":
                ax.set_ylim(0, None)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False)
            for (a, row) in g.iterrows():
                rows.append({"metric": m, "by": xvar, "bin": a, **row.to_dict(), "rho": rho})
    caption(fig, "Рис. 9. Полнота новых слоёв против полноты тегов улиц (surface_tagged_share, этап 3) и удалённости "
                 "от центра. Ячейки панели, где в OSM ≥ 5 зданий. Рост полноты к центру и там, где размечено покрытие "
                 "улиц, — признак «прилежания картографов».", y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig9_completeness_vs_surface_dist", pd.DataFrame(rows))

    # 9b: частные ρ с долей ДТП при контроле «физики»
    cc = cells[cells.bld_n >= 5].copy()
    controls = ["street_length_m", "major_road_length_m", "bld_fp_m2", "dist_center_m"]
    cand = [("bld_levels_area_share", "полнота: доля площади с этажностью"),
            ("bld_typed_share", "полнота: доля зданий с типом"),
            ("surface_tagged_share", "полнота: surface на улицах (этап 3)"),
            ("poi_named_share", "полнота: доля POI с названием"),
            ("poi_attr_n", "POI-аттракторы"),
            ("poi_n", "POI все"),
            ("poi_school_n", "школы и детсады"),
            ("comm_fp_m2", "коммерческая площадь основания"),
            ("bld_floor_known_m2", "площадь этажей (только где этажность есть)"),
            ("stop_n", "остановки (после дедупа)"),
            ("dist_stop_m", "расстояние до остановки")]
    rows = []
    for col, lab in cand:
        rows.append({"variable": col, "label": lab,
                     "rho_raw": spearmanr(cc[col], cc.dtp_share, nan_policy="omit")[0],
                     "rho_partial": partial_spearman(cc, col, "dtp_share", controls),
                     "rho_partial_plus_surface": partial_spearman(cc, col, "dtp_share", controls + ["surface_tagged_share"])
                     if col != "surface_tagged_share" else np.nan,
                     "rho_with_surface": spearmanr(cc[col], cc.surface_tagged_share, nan_policy="omit")[0],
                     "n": cc[col].notna().sum()})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(t))
    ax.barh(y + 0.2, t.rho_raw, height=0.38, color=C[0], label="ρ без контроля")
    ax.barh(y - 0.2, t.rho_partial, height=0.38, color=C[1],
            label="частный ρ | длина улиц, длина магистралей, площадь застройки, расстояние до центра")
    for yy, a, b_ in zip(y, t.rho_raw, t.rho_partial):
        ax.text(a + 0.01, yy + 0.2, f"{a:.2f}", va="center", fontsize=8, color=INK2)
        ax.text(b_ + 0.01, yy - 0.2, f"{b_:.2f}", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, t.label)
    ax.invert_yaxis()
    ax.set_xlabel("ρ Спирмена с долей ячейко-месяцев с ДТП (ячейки с ≥ 5 зданиями)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1)
    ax.set_title("Рис. 9б. Полнота слоёв предсказывает ДТП и после контроля «физики» ячейки", loc="left")
    save(fig, "fig9b_partial_rho_dtp", t)
    print(t.round(3).to_string())


# ---------------------------------------------------------------- 10. Саратов и Самара
def fig10(b, p, s):
    rows = []
    sar_b = b[b.in_bbox].copy()
    sam_b = b[b.region == "samara"].copy()
    sar_b["band"] = dist_band(pd.Series(dist_m(sar_b.lat.values, sar_b.lon.values, SARATOV_CENTER), index=sar_b.index))
    sam_b["band"] = dist_band(pd.Series(dist_m(sam_b.lat.values, sam_b.lon.values, SAMARA_CENTER), index=sam_b.index))
    for city, d in [("Саратов", sar_b), ("Самара", sam_b)]:
        for band in ["все"] + list(d.band.cat.categories):
            x = d if band == "все" else d[d.band == band]
            big = x[x.footprint_m2 >= 500]
            rows.append({"город": city, "кольцо": band, "зданий": len(x),
                         "с этажностью, % зданий": x.levels.notna().mean() * 100,
                         "с этажностью, % площади": (x.footprint_m2 * x.levels.notna()).sum() / x.footprint_m2.sum() * 100,
                         "с этажностью среди ≥ 500 м², %": big.levels.notna().mean() * 100,
                         "с типом (≠ yes), %": (x.building != "yes").mean() * 100})
    t = pd.DataFrame(rows)
    sp, sm = p[p.in_bbox], p[p.region == "samara"]
    # население — ориентир по памяти (перепись 2021, округлённо): Саратов 0.90 млн + Энгельс 0.22 млн; Самара 1.17 млн
    pop = {"Саратов": 1.12, "Самара": 1.17}
    prow = []
    for city, d in [("Саратов", sp), ("Самара", sm)]:
        am = d.amenity.fillna("")
        prow.append({"город": city, "POI всего": len(d), "shop": d.shop.notna().sum(), "office": d.office.notna().sum(),
                     "аптеки (amenity|healthcare=pharmacy)": (am.eq("pharmacy") | d.healthcare.eq("pharmacy")).sum(),
                     "школы": am.eq("school").sum(), "детсады": am.eq("kindergarten").sum(),
                     "общепит": am.isin({"restaurant", "cafe", "fast_food", "bar", "pub"}).sum(),
                     "доля POI с названием, %": d.name.notna().mean() * 100, "население, млн (ориентир)": pop[city]})
    pt = pd.DataFrame(prow)
    for col in ["аптеки (amenity|healthcare=pharmacy)", "школы", "детсады", "shop"]:
        pt[col + " на 100 тыс."] = pt[col] / (pt["население, млн (ориентир)"] * 10)
    srow = []
    for city, reg in [("Саратов", "saratov_buffer"), ("Самара", "samara")]:
        x = s[s.region == reg]
        if reg == "saratov_buffer":
            x = x[x.in_bbox]
        x = stop_dedup(x)
        srow.append({"город": city, "объектов-остановок": len(x), "остановок после дедупа": x.cluster.nunique(),
                     "доля объектов PTv2, %": (x.scheme == "ptv2").mean() * 100})
    stt = pd.DataFrame(srow)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axes[0]
    bands = ["0–2.5 км", "2.5–5 км", "5–10 км", "> 10 км", "все"]
    x = np.arange(len(bands))
    for k, (city, col) in enumerate([("Саратов", C[0]), ("Самара", C[1])]):
        tt = t[t["город"] == city].set_index("кольцо").reindex(bands)
        ax.bar(x + (k - 0.5) * 0.38, tt["с этажностью, % площади"], width=0.36, color=col, label=city)
        for xx, v in zip(x + (k - 0.5) * 0.38, tt["с этажностью, % площади"]):
            ax.text(xx, v + 1, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{b}" for b in bands])
    ax.set_xlabel("кольцо удалённости от центра города")
    ax.set_ylabel("% площади основания с building:levels")
    ax.set_title("а) Этажность зданий по кольцам", loc="left")
    ax.legend(frameon=False)
    ax = axes[1]
    labs = ["с этажностью, % зданий", "с этажностью среди ≥ 500 м², %", "с типом (≠ yes), %"]
    tv = t[t["кольцо"] == "все"].set_index("город")
    x = np.arange(len(labs))
    for k, (city, col) in enumerate([("Саратов", C[0]), ("Самара", C[1])]):
        vals = tv.loc[city, labs].values.astype(float)
        ax.bar(x + (k - 0.5) * 0.38, vals, width=0.36, color=col, label=city)
        for xx, v in zip(x + (k - 0.5) * 0.38, vals):
            ax.text(xx, v + 1, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, ["с этажностью,\n% зданий", "с этажностью\nсреди ≥ 500 м²", "с типом\n(≠ yes)"])
    ax.set_ylabel("%")
    ax.set_title("б) Полнота зданий, весь bbox", loc="left")
    ax = axes[2]
    labs2 = ["аптеки (amenity|healthcare=pharmacy) на 100 тыс.", "школы на 100 тыс.", "детсады на 100 тыс.",
             "shop на 100 тыс."]
    x = np.arange(len(labs2))
    for k, (city, col) in enumerate([("Саратов", C[0]), ("Самара", C[1])]):
        vals = pt.set_index("город").loc[city, labs2].values.astype(float)
        vals_plot = vals.copy()
        vals_plot[3] = vals[3] / 10
        ax.bar(x + (k - 0.5) * 0.38, vals_plot, width=0.36, color=col, label=city)
        for xx, v, vp in zip(x + (k - 0.5) * 0.38, vals, vals_plot):
            ax.text(xx, vp + 0.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, ["аптеки", "школы", "детсады", "магазины (shop)\n÷ 10"])
    ax.set_ylabel("объектов на 100 тыс. жителей (ориентир)")
    ax.set_title("в) POI на 100 тыс. жителей", loc="left")
    caption(fig, "Рис. 10. Переносимость: полнота OSM в Саратове (bbox проекта, с Энгельсом) и в центральной части Самары "
                 "(bbox этапа 5), тот же снимок PBF. Население — округлённые данные переписи 2021 по памяти, только ориентир.",
            y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig10_saratov_vs_samara", t)
    pt.to_csv(TAB / "poi_per_capita_saratov_samara.csv", index=False)
    stt.to_csv(TAB / "stops_saratov_samara.csv", index=False)
    print(t.round(1).to_string())
    print(pt.round(1).T.to_string())
    print(stt.to_string())


# ---------------------------------------------------------------- 11. landuse
def fig11(lu):
    x = lu[lu.region == "saratov_buffer"]
    step = 0.0006  # ~ 67 × 42 м
    la = np.arange(BBOX[0] + step / 2, BBOX[2], step)
    lo = np.arange(BBOX[1] + step / 2, BBOX[3], step * 1.6)
    LO, LA = np.meshgrid(lo, la)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    lab = np.full(len(pts), "", dtype=object)
    order = x.sort_values("area_m2", ascending=False)  # мелкие полигоны поверх крупных
    for rr in order.itertuples():
        rings = json.loads(rr.rings_json)
        outer = [np.array(r)[:, ::-1] for s_, r in rings if s_ == 1]
        inner = [np.array(r)[:, ::-1] for s_, r in rings if s_ == -1]
        allr = np.vstack(outer)
        m = ((pts[:, 0] >= allr[:, 0].min()) & (pts[:, 0] <= allr[:, 0].max())
             & (pts[:, 1] >= allr[:, 1].min()) & (pts[:, 1] <= allr[:, 1].max()))
        if not m.any():
            continue
        idx = np.where(m)[0]
        inside = np.zeros(len(idx), bool)
        for o in outer:
            inside |= MplPath(o).contains_points(pts[idx])
        for i_ in inner:
            inside &= ~MplPath(i_).contains_points(pts[idx])
        lab[idx[inside]] = rr.landuse
    groups = {"residential": "жилая (residential)", "industrial": "промышленная", "commercial": "коммерческая",
              "retail": "коммерческая", "farmland": "с/х и сады", "allotments": "с/х и сады", "orchard": "с/х и сады",
              "farmyard": "с/х и сады", "meadow": "трава, луга, лес", "grass": "трава, луга, лес",
              "forest": "трава, луга, лес", "garages": "гаражи", "railway": "прочее", "cemetery": "прочее",
              "military": "прочее", "construction": "прочее", "landfill": "прочее"}
    g = pd.Series(lab).map(lambda v: "нет landuse" if v == "" else groups.get(v, "прочее"))
    order_g = ["жилая (residential)", "коммерческая", "промышленная", "гаражи", "с/х и сады", "трава, луга, лес",
               "прочее", "нет landuse"]
    cols = dict(zip(order_g, [C[0], C[1], C[6], C[4], C[3], C[2], "#8a8984", "#f1f0ec"]))
    share = g.value_counts(normalize=True).reindex(order_g).fillna(0)
    # покрытие ячеек панели
    cells = pd.read_csv(OUT / "cells_osm_v6_eda.csv")
    hh = [h3.latlng_to_cell(a, b_, 9) for b_, a in pts]
    cov = pd.DataFrame({"h3": hh, "has": g.values != "нет landuse"}).groupby("h3").has.mean()
    cells["landuse_cover"] = cells.h3.map(cov)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    from matplotlib.colors import ListedColormap
    codes = pd.Categorical(g, categories=order_g).codes.reshape(LA.shape)
    ax.imshow(codes, origin="lower", extent=(BBOX[1], BBOX[3], BBOX[0], BBOX[2]),
              cmap=ListedColormap([cols[k] for k in order_g]), vmin=-0.5, vmax=len(order_g) - 0.5,
              interpolation="nearest", aspect="auto")
    center_marks(ax)
    map_axes(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[k]) for k in order_g]
    ax.legend(handles, [f"{k}: {share[k] * 100:.0f} %" for k in order_g], loc="lower left", fontsize=8, frameon=True)
    ax.set_title("а) landuse в bbox Саратова (растр ≈ 67 × 42 м)", loc="left")
    ax = axes[1]
    gg = cells.groupby("district").landuse_cover.mean().reindex(DISTRICT_ORDER)
    ax.barh(np.arange(len(gg)), gg * 100, color=[DISTRICT_COLORS[d] for d in gg.index], height=0.6)
    for yy, v in enumerate(gg * 100):
        ax.text(v + 1, yy, f"{v:.0f} %", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(np.arange(len(gg)), gg.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("средняя доля площади ячейки под каким-либо landuse, %")
    ax.set_title("б) Покрытие landuse ячеек панели по районам", loc="left")
    fig.tight_layout(w_pad=4)
    caption(fig, "Рис. 11. landuse OSM (полигоны и мультиполигоны). Используется только для контроля полноты.", y=0.0)
    out = pd.DataFrame({"group": order_g, "share_bbox": share.values})
    save(fig, "fig11_landuse", out)
    cells[["h3", "district", "landuse_cover"]].to_csv(TAB / "landuse_cover_by_cell.csv", index=False)
    print(out)
    print(gg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    TAB.mkdir(parents=True, exist_ok=True)
    cells, b, p, s, r, st, tr, lu = load()
    jobs = {"fig1": lambda: fig1(cells, b), "fig2": lambda: fig2(cells), "fig3": lambda: fig3(cells, b),
            "fig4": lambda: fig4(cells, p), "fig5": lambda: fig5(cells, p), "fig6": lambda: fig6(s, r, tr),
            "fig7": lambda: fig7(st), "fig8": lambda: fig8(cells), "fig9": lambda: fig9(cells),
            "fig10": lambda: fig10(b, p, s), "fig11": lambda: fig11(lu)}
    for k, f in jobs.items():
        if only is None or k in only:
            f()


if __name__ == "__main__":
    main()
