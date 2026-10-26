"""Фигуры этапа 6.2–6.5 (запуск 6a) -> docs/analysis/features/stage6_exposure/blocks_a/figures/ (PNG 300 dpi + CSV).

    --part descr    рис. 1–6 (описательные: карты, спутник, остановки, децили, Spearman)
    --part results  рис. 7–10 (Δ по блокам, жадная сборка, exposure_a по сидам, важности)

    PYTHONPATH=src venv/bin/python scripts/plot_road_geo_v2_exposure_a.py --part descr
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import h3  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.ticker  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

from road_accident_risk import catboost_baseline as cb  # noqa: E402
from road_accident_risk import exposure_v6 as ev  # noqa: E402
from road_accident_risk.paths import PARQUET_DIR, PROJECT_ROOT  # noqa: E402

DOC = PROJECT_ROOT / "docs/analysis/features/stage6_exposure/blocks_a"
FIG = DOC / "figures"
TAB = DOC / "tables"
REPORTS = PROJECT_ROOT / "reports"
TARGET = "target_had_accident_next_month"
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
BLUE, ORANGE, GREEN, RED, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#d1242f", "#8250df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.titlesize": 11})
LAT0 = 51.55
RU = {
    "dist_to_center_two_m": "расстояние до центра (а: ближайший из двух), м",
    "dist_to_center_poi_m": "расстояние до центра (б: пики плотности POI), м",
    "dist_to_nearest_major_m": "до ближайшей trunk/primary, м",
    "dist_to_nearest_signal_m": "до регулируемого перекрёстка, м",
    "poi_attractor_count": "POI-аттракторы в ячейке",
    "poi_attractor_count_r300": "POI-аттракторы в круге 300 м",
    "poi_attractor_count_r500": "POI-аттракторы в круге 500 м",
    "school_kindergarten_count": "школы и детсады в ячейке",
    "school_kindergarten_count_r300": "школы и детсады, 300 м",
    "school_kindergarten_count_r500": "школы и детсады, 500 м",
    "healthcare_count": "медицина в ячейке",
    "healthcare_count_r300": "медицина, 300 м",
    "healthcare_count_r500": "медицина, 500 м",
    "building_footprint_m2": "площадь основания зданий, м²",
    "small_building_share": "доля площади мелких зданий (< 200 м²)",
    "attractor_building_footprint_m2": "площадь зданий с POI-аттрактором, м²",
    "pt_stop_count_v6": "остановки (дедуп 30 м)",
    "dist_to_nearest_stop_m": "до ближайшей остановки, м",
    "tram_track_length_m": "длина трамвайных путей, м",
    "pt_route_count": "маршруты автобуса/троллейбуса",
    "levels_tagged_share": "доля площади зданий с этажностью (диагн.)",
    "public_transport_stop_count": "остановки (старый счёт)",
}


def save(fig, name: str, data: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  {name}.png / .csv", flush=True)


def load_cells() -> pd.DataFrame:
    panel = pd.read_parquet(PROJECT_ROOT / "parquet_output/h3_month_road_features.parquet",
                            columns=["h3", "month", TARGET, "road_length_total_m", "public_transport_stop_count"])
    panel = panel[panel.road_length_total_m > 0]
    cells = panel.groupby("h3").agg(dtp_share=(TARGET, "mean"), n_months=("month", "size"),
                                    public_transport_stop_count=("public_transport_stop_count", "first")).reset_index()
    ex = pd.read_parquet(PARQUET_DIR / "h3_exposure_v6_features.parquet")
    return cells.merge(ex, on="h3", how="left")


def hex_polys(h3s) -> list[np.ndarray]:
    return [np.array([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in h3s]


def hex_map(ax, cells: pd.DataFrame, col: str, norm, cmap="viridis", label: str = ""):
    pc = PolyCollection(hex_polys(cells.h3), array=cells[col].to_numpy(), cmap=cmap, norm=norm,
                        edgecolors="none", linewidths=0)
    ax.add_collection(pc)
    ax.set_xlim(45.80, 46.20); ax.set_ylim(51.40, 51.70)
    ax.set_aspect(1 / math.cos(math.radians(LAT0)))
    ax.set_xticks([]); ax.set_yticks([])
    cb_ = plt.colorbar(pc, ax=ax, shrink=0.75, pad=0.01)
    cb_.set_label(label)
    return pc


def deg_bar(ax, km=2.0):
    dlon = km * 1000 / (111320 * math.cos(math.radians(LAT0)))
    x0, y0 = 45.82, 51.415
    ax.add_patch(plt.Rectangle((x0 - 0.008, y0 - 0.006), dlon + 0.016, 0.019, fc="white", ec=GRID, zorder=6))
    ax.plot([x0, x0 + dlon], [y0, y0], color=INK, lw=3, zorder=7)
    ax.text(x0 + dlon / 2, y0 + 0.004, f"{km:.0f} км", ha="center", fontsize=9, zorder=7)


# ---------------------------------------------------------------- рис. 1–6
def fig1_center(cells: pd.DataFrame) -> None:
    peaks = pd.read_csv(TAB / "poi_peaks.csv")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    rows = []
    for ax, col, title in ((axes[0], "dist_to_center_two_m", "а) до ближайшего из двух центров (узлы OSM)"),
                           (axes[1], "dist_to_center_poi_m", "б) до ближайшего пика плотности POI-аттракторов")):
        hex_map(ax, cells, col, Normalize(0, 15000), "magma_r", "расстояние, м (шкала обрезана на 15 км)")
        if col == "dist_to_center_two_m":
            for (lat, lon), nm in ((ev.SARATOV_CENTER, "Саратов: Театральная пл."), (ev.ENGELS_CENTER, "Энгельс: place=city")):
                ax.scatter(lon, lat, marker="*", s=260, c="white", edgecolors=INK, zorder=5)
                ax.annotate(nm, (lon, lat), xytext=(-10, 10), textcoords="offset points", fontsize=9, ha="right",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID))
                rows.append({"method": "a", "name": nm, "lat": lat, "lon": lon})
        else:
            for i, r in peaks.iterrows():
                ax.scatter(r.lon, r.lat, marker="*", s=260, c="white", edgecolors=INK, zorder=5)
                ax.annotate(f"пик {i + 1}: {r.rel_density:.2f} от макс.", (r.lon, r.lat), xytext=(-10, 10), ha="right",
                            textcoords="offset points", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID))
                rows.append({"method": "b", "name": f"peak{i + 1}", "lat": r.lat, "lon": r.lon, "rel_density": r.rel_density})
        ax.text(45.975, 51.468, "р. Волга", color=MUTED, fontsize=10, style="italic", rotation=40)
        deg_bar(ax)
        ax.set_title(title, loc="left")
    fig.suptitle("Рис. 1. Расстояние от центроида ячейки до центра: два способа учёта Энгельса "
                 f"(ячейки панели, n = {len(cells)})", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig1_dist_to_center_map", pd.DataFrame(rows))


def fig2_poi_buildings(cells: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    c1 = cells.copy()
    c1["poi_plot"] = c1.poi_attractor_count.where(c1.poi_attractor_count > 0)
    c1["fp_plot"] = c1.building_footprint_m2.where(c1.building_footprint_m2 > 0)
    for ax, col, lab, title in ((axes[0], "poi_plot", "число POI-аттракторов (лог. шкала)", "а) POI-аттракторы в ячейке"),
                                (axes[1], "fp_plot", "площадь основания, м² (лог. шкала)", "б) площадь основания зданий в ячейке")):
        empty = c1[c1[col].isna()]
        ax.add_collection(PolyCollection(hex_polys(empty.h3), facecolors="#eceff1", edgecolors="none"))
        sub = c1[c1[col].notna()]
        hex_map(ax, sub, col, LogNorm(max(1, sub[col].min()), sub[col].max()), "viridis", lab)
        deg_bar(ax)
        ax.set_title(f"{title} (серым — ноль: {len(empty)} из {len(c1)} ячеек)", loc="left")
    fig.suptitle("Рис. 2. POI-аттракторы и площадь основания зданий по ячейкам H3 res 9 (OSM, 03.06.2026)",
                 x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig2_poi_buildings_map", cells[["h3", "lat", "lon", "poi_attractor_count", "building_footprint_m2"]])


def fig3_satellite(cells: pd.DataFrame) -> None:
    import plot_blocks_satellite as sat
    sat.Z = 17
    b = pd.read_parquet(PARQUET_DIR / "osm_v6_buildings.parquet", columns=["region", "lat", "lon", "footprint_m2", "rings_json"])
    b = b[b.region == "saratov_buffer"]
    # окно 2 — ячейка с наибольшим разбросом доли мелких зданий среди соседей (переход частный сектор / многоэтажки)
    c = cells.set_index("h3")
    best, best_sd = None, -1
    for h in c.index[c.small_building_share.notna() & (c.building_count >= 10)]:
        nb = [x for x in h3.grid_disk(h, 1) if x in c.index and c.loc[x, "building_count"] >= 10]
        if len(nb) >= 6:
            sd = c.loc[nb, "small_building_share"].std()
            if sd > best_sd:
                best, best_sd = h, sd
    lat_c, lon_c = h3.cell_to_latlng(best)
    windows = [("а) частный сектор (окно рис. 9 этапа 4)", (51.5095, 45.9505, 51.5180, 45.9640)),
               (f"б) переход застройки (макс. разброс доли у соседей, σ = {best_sd:.2f})",
                (lat_c - 0.0045, lon_c - 0.0072, lat_c + 0.0045, lon_c + 0.0072))]
    fig, axes = plt.subplots(1, 2, figsize=(18, 9.6))
    rows = []
    cmap = plt.get_cmap("RdYlBu_r")
    for ax, (title, (lat1, lon1, lat2, lon2)) in zip(axes, windows):
        img, (ox, oy) = sat.mosaic(lat1, lon1, lat2, lon2)
        ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))
        bb = b[b.lat.between(lat1 - 0.001, lat2 + 0.001) & b.lon.between(lon1 - 0.0015, lon2 + 0.0015)]
        segs, cols = [], []
        for r in bb.itertuples():
            for s, ring in json.loads(r.rings_json):
                ring = np.array(ring)
                px, py = sat.lonlat_to_px(ring[:, 1], ring[:, 0])
                segs.append(np.column_stack([px - ox, py - oy]))
                cols.append("#ffd400" if r.footprint_m2 < ev.SMALL_BUILDING_M2 else "#00e5ff")
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.8))
        hs = {h3.latlng_to_cell(la, lo, 9) for la in np.linspace(lat1, lat2, 25) for lo in np.linspace(lon1, lon2, 25)}
        for h in hs:
            bd = np.array(h3.cell_to_boundary(h) + (h3.cell_to_boundary(h)[0],))
            px, py = sat.lonlat_to_px(bd[:, 1], bd[:, 0])
            v = c.loc[h, "small_building_share"] if h in c.index else np.nan
            col = cmap(v) if np.isfinite(v) else (1, 1, 1, 1)
            ax.plot(px - ox, py - oy, color=col, lw=2.6)
            la, lo = h3.cell_to_latlng(h)
            cx, cy = sat.lonlat_to_px(lo, la)
            if 0 < cx - ox < img.shape[1] and 0 < cy - oy < img.shape[0]:
                txt = f"{v:.2f}" if np.isfinite(v) else ("нет зданий" if h in c.index else "вне панели")
                ax.text(cx - ox, cy - oy, txt, color="white", ha="center", va="center", fontsize=11, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc=col if np.isfinite(v) else (0.3, 0.3, 0.3, 0.7), ec="none", alpha=0.85))
                rows.append({"window": title, "h3": h, "small_building_share": v,
                             "building_footprint_m2": c.loc[h, "building_footprint_m2"] if h in c.index else np.nan})
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.set_xticks([]); ax.set_yticks([])
        m_per_px = 156543.03392 * math.cos(math.radians((lat1 + lat2) / 2)) / 2 ** sat.Z
        L = 200 / m_per_px
        x0b, y0b = img.shape[1] * 0.05, img.shape[0] * 0.95
        ax.plot([x0b, x0b + L], [y0b, y0b], color="white", lw=4)
        ax.text(x0b + L / 2, y0b - 12, "200 м", color="white", ha="center", fontsize=11, weight="bold")
        ax.set_title(title, loc="left", fontsize=12)
    handles = [plt.Line2D([0], [0], color="#ffd400", lw=3), plt.Line2D([0], [0], color="#00e5ff", lw=3)]
    fig.legend(handles, [f"контур здания < {ev.SMALL_BUILDING_M2:.0f} м²", f"контур ≥ {ev.SMALL_BUILDING_M2:.0f} м²"],
               loc="lower center", ncol=2, fontsize=11, frameon=False,
               title="граница и число в ячейке — small_building_share (синий 0 … красный 1)")
    fig.suptitle("Рис. 3. Доля площади мелких зданий (прокси частного сектора) поверх спутникового снимка. "
                 "Подложка: Esri World Imagery", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    save(fig, "fig3_small_building_share_satellite", pd.DataFrame(rows))


def fig4_stops() -> None:
    s = pd.read_parquet(PARQUET_DIR / "osm_v6_pt_stops.parquet")
    s = s[s.region == "saratov_buffer"]
    st = pd.read_csv(TAB / "stops_dedup.csv")
    lat1, lon1, lat2, lon2 = 51.526, 46.010, 51.540, 46.040
    streets = pd.read_parquet(PARQUET_DIR / "osm_v6_streets.parquet", columns=["highway", "coords_json", "mid_lat", "mid_lon"])
    streets = streets[streets.mid_lat.between(lat1 - 0.01, lat2 + 0.01) & streets.mid_lon.between(lon1 - 0.015, lon2 + 0.015)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), sharex=True, sharey=True)
    segs = [np.array(json.loads(c))[:, ::-1] for c in streets.coords_json]
    w = [2.2 if h in ("trunk", "primary", "secondary") else 0.8 for h in streets.highway]
    kind = np.where(s.public_transport.eq("platform"), "platform (PTv2)",
                    np.where(s.public_transport.eq("stop_position"), "stop_position (PTv2)", "bus_stop / tram_stop (старая схема)"))
    s = s.assign(kind=kind)
    sw = s[s.lat.between(lat1, lat2) & s.lon.between(lon1, lon2)]
    cw = st[st.lat.between(lat1, lat2) & st.lon.between(lon1, lon2)]
    for ax in axes:
        ax.add_collection(LineCollection(segs, colors="#9aa5b1", linewidths=w, zorder=1))
        ax.set_xlim(lon1, lon2); ax.set_ylim(lat1, lat2); ax.set_aspect(1 / math.cos(math.radians(LAT0)))
        ax.set_xticks([]); ax.set_yticks([])
    for k, col, m in (("platform (PTv2)", BLUE, "s"), ("stop_position (PTv2)", ORANGE, "o"),
                      ("bus_stop / tram_stop (старая схема)", PURPLE, "^")):
        d = sw[sw.kind == k]
        axes[0].scatter(d.lon, d.lat, s=22, c=col, marker=m, label=f"{k}: {len(d)}", zorder=3, edgecolors="white", linewidths=0.4)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    axes[0].set_title(f"а) объекты OSM до дедупа: {len(sw)}", loc="left")
    r_deg = ev.STOP_DEDUP_M / 111320 / math.cos(math.radians(LAT0))
    for r in cw.itertuples():
        axes[1].add_patch(matplotlib.patches.Ellipse((r.lon, r.lat), 2 * r_deg, 2 * r_deg * math.cos(math.radians(LAT0)),
                                                     color=GREEN, alpha=0.25, zorder=2))
    axes[1].scatter(sw.lon, sw.lat, s=6, c=MUTED, zorder=3)
    sc = axes[1].scatter(cw.lon, cw.lat, s=20 + 12 * cw.n_objects, c=GREEN, edgecolors=INK, linewidths=0.6, zorder=4)
    axes[1].set_title(f"б) после склейки в {ev.STOP_DEDUP_M:.0f} м: {len(cw)} остановок "
                      f"(размер — число объектов в кластере)", loc="left")
    dlon = 200 / (111320 * math.cos(math.radians(LAT0)))
    for ax in axes:
        ax.plot([lon1 + 0.001, lon1 + 0.001 + dlon], [lat1 + 0.0008] * 2, color=INK, lw=3)
        ax.text(lon1 + 0.001 + dlon / 2, lat1 + 0.0013, "200 м", ha="center", fontsize=9)
    fig.suptitle("Рис. 4. Остановки общественного транспорта до и после дедупликации (фрагмент центра Саратова)",
                 x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig4_stops_dedup", cw.assign(window="center"))


def fig5_deciles(cells: pd.DataFrame, feats: list[str]) -> None:
    n = len(feats); ncol = 4; nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), squeeze=False)
    rows = []
    for ax, f in zip(axes.flat, feats):
        x = cells[[f, "dtp_share", "n_months"]].dropna()
        # нули (если их > 10 %) — отдельный бин «0», остальное — децили (или меньше при ничьих)
        zero = x[f].eq(0)
        if zero.mean() > 0.10:
            q = pd.Series(0, index=x.index)
            q[~zero] = 1 + pd.qcut(x.loc[~zero, f], 9, labels=False, duplicates="drop")
        else:
            q = pd.qcut(x[f], 10, labels=False, duplicates="drop")
        g = x.groupby(q).apply(lambda d: pd.Series({"dtp": np.average(d.dtp_share, weights=d.n_months),
                                                     "lo": d[f].min(), "hi": d[f].max(), "n": len(d)}))
        ax.bar(range(len(g)), g.dtp * 100, color=BLUE, width=0.75)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([f"{v:.0f}" if abs(v) >= 10 or v == 0 else f"{v:.2f}" for v in g.hi], rotation=60, fontsize=7)
        nan_txt = f" (NaN {cells[f].isna().mean() * 100:.0f} % ячеек)" if cells[f].isna().any() else ""
        ax.set_title(RU.get(f, f) + nan_txt, fontsize=9, loc="left")
        ax.set_ylabel("доля с ДТП, %", fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for i, r in g.reset_index().iterrows():
            rows.append({"feature": f, "bin": i, "lo": r.lo, "hi": r.hi, "n_cells": r.n, "dtp_share": r.dtp})
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Рис. 5. Доля ячейко-месяцев с ДТП в следующем месяце по децилям новых признаков "
                 "(описательно, 2015–2025; подпись — верхняя граница бина; при > 10 % нулей первый бин — «0», остальные — квантили ненулевых)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig5_dtp_by_deciles", pd.DataFrame(rows))


def fig6_spearman(feats: list[str]) -> None:
    r = pd.read_csv(TAB / "spearman_new_vs_a.csv", index_col=0)
    a = [c for c in r.columns if c not in ev.EXPOSURE_V6_FEATURES]
    cols = [*feats, *a]
    m = r.loc[feats, cols]
    fig, ax = plt.subplots(figsize=(17, 0.42 * len(feats) + 3))
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=7)
    ax.axvline(len(feats) - 0.5, color=INK, lw=1.5)
    for i in range(len(feats)):
        for j in range(len(cols)):
            v = m.values[i, j]
            if abs(v) >= 0.5 and not (j < len(feats) and feats[j] == feats[i]):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) >= 0.75 else INK, weight="bold" if abs(v) >= 0.85 else None)
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.01, label="ρ Спирмена")
    ax.set_title("Рис. 6. Spearman новых признаков (строки) между собой (слева от черты) и с набором A "
                 f"road_geo_v2_skdf (32, справа); подписаны |ρ| ≥ 0.5, жирным ≥ 0.85; ячеек панели 4 296",
                 loc="left", fontsize=10)
    fig.tight_layout()
    out = m.reset_index().rename(columns={"index": "feature"})
    save(fig, "fig6_spearman_new_vs_a", out)

# ---------------------------------------------------------------- рис. 7–10
KEY_METRICS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision_top_1pct", "precision@1 %"),
               ("recall_top_10pct", "recall@10 %")]
A_NAME = "road_geo_v2_skdf"
LABELS = {
    "e1_center_two": "E1 (центр а: два узла)", "e1_center_poi": "E1 (центр б: пики POI)",
    "e2_cell": "E2 (POI в ячейке)", "e2_r300": "E2 (POI, круг 300 м)", "e2_r500": "E2 (POI, круг 500 м)",
    "e3_build": "E3 (застройка)", "e4_old": "E4 (старый счёт остановок)", "e4_new": "E4 (новый счёт остановок)",
    "e4_routes": "E4 + маршруты", "road_geo_v2_exposure_a": "exposure_a", "eart": "E-art: + 9 рисовки",
    "elevels": "E-levels: + этажность", "emed": "E-med: NaN → медиана", "road_geo_v2_tags": "road_geo_v2_tags",
}


def label(v: str) -> str:
    if v in LABELS:
        return LABELS[v]
    if v.startswith("g_"):
        return "A + " + " + ".join(v[2:].split("_"))
    return v


def fig7_blocks(variants: list[str]) -> None:
    from road_accident_risk.v2_eval import compare, compare_folds
    runs = pd.read_csv(REPORTS / "v2_eval_stage6a.csv")
    rows = []
    for v in variants:
        fo = compare_folds(runs, A_NAME, v).set_index("column")
        te = compare(runs, A_NAME, v, split="test").set_index("column") if \
            not runs[(runs.split == "test") & (runs.variant == v)].empty else None
        for f in ("valid2021", "valid"):
            cf = compare(runs, A_NAME, v, split=f).set_index("column")
            for c, _ in KEY_METRICS:
                rows.append({"variant": v, "kind": f, "metric": c, "delta": cf.loc[c, "delta"], "two_sigma": cf.loc[c, "two_sigma"],
                             "verdict": cf.loc[c, "verdict"]})
        for c, _ in KEY_METRICS:
            rows.append({"variant": v, "kind": "folds_mean", "metric": c, "delta": fo.loc[c, "delta"],
                         "two_sigma": fo.loc[c, "two_sigma"], "verdict": fo.loc[c, "verdict"],
                         "same_sign": fo.loc[c, "same_sign_all_folds"]})
            if te is not None:
                rows.append({"variant": v, "kind": "test", "metric": c, "delta": te.loc[c, "delta"],
                             "two_sigma": te.loc[c, "two_sigma"], "verdict": te.loc[c, "verdict"]})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(18, 0.62 * len(variants) + 2.4), sharey=True)
    style = {"valid2021": ("фолд 2021", "#9ecae1", "v", 0.27), "valid": ("фолд 2022", "#3182bd", "^", 0.09),
             "folds_mean": ("среднее фолдов ± 2σ (отбор)", INK, "o", -0.09), "test": ("тест 2023–25 ± 2σ (протокол)", ORANGE, "s", -0.27)}
    for ax, (c, nm) in zip(axes, KEY_METRICS):
        for k, (lab, col, mk, off) in style.items():
            sub = d[(d.metric == c) & (d.kind == k)].set_index("variant")
            y = [len(variants) - 1 - variants.index(v) + off for v in sub.index]
            if k in ("folds_mean", "test"):
                ax.errorbar(sub.delta, y, xerr=sub.two_sigma, fmt=mk, color=col, ms=6, capsize=3, label=lab, lw=1.2)
                if k == "folds_mean" and c in ("roc_auc", "pr_auc"):
                    for yy, (vv, r) in zip(y, sub.iterrows()):
                        if r.verdict == "значимо" and r.get("same_sign", False) and r.delta > 0:
                            ax.text(r.delta + r.two_sigma, yy, " ✓", va="center", color=GREEN, fontsize=11, weight="bold")
            else:
                ax.scatter(sub.delta, y, marker=mk, color=col, s=30, label=lab, zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        for sep in ("e4_routes", "g_E3_E4"):
            if sep in variants:
                ax.axhline(len(variants) - 1 - variants.index(sep) - 0.5, color=MUTED, lw=0.8, ls="--")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(nm)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_xlabel("Δ к A (road_geo_v2_skdf)")
    axes[0].set_yticks(range(len(variants)))
    axes[0].set_yticklabels([label(v) for v in variants[::-1]])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Рис. 7. Варианты поверх A: Δ по фолдам 2021 и 2022, среднее фолдов (критерий отбора) и тест (протокол); "
                 "5 сидов, полосы — 2σ\n✓ — рост ROC/PR-AUC значим по среднему и знак одинаков в обоих годах; "
                 "пунктир делит выбор в блоках / жадную сборку / диагностики", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, "fig7_blocks_delta", d)


def fig8_greedy() -> None:
    from road_accident_risk.v2_eval import compare, compare_folds
    path = json.loads((TAB / "greedy_path.json").read_text())
    runs = pd.read_csv(REPORTS / "v2_eval_stage6a.csv")
    rows = []
    for i, st in enumerate(path):
        v = st["variant"]
        for c, nm in KEY_METRICS[:2]:
            if v == A_NAME:
                rows.append({"step": i, "variant": v, "metric": c, "folds_delta": 0.0, "folds_2s": 0.0, "test_delta": 0.0,
                             "test_2s": 0.0, "accepted": True, "label": st["label"]})
                continue
            fo = compare_folds(runs, A_NAME, v).set_index("column")
            te = compare(runs, A_NAME, v, split="test").set_index("column") if \
                not runs[(runs.split == "test") & (runs.variant == v)].empty else None
            rows.append({"step": i, "variant": v, "metric": c, "folds_delta": fo.loc[c, "delta"], "folds_2s": fo.loc[c, "two_sigma"],
                         "test_delta": te.loc[c, "delta"] if te is not None else np.nan,
                         "test_2s": te.loc[c, "two_sigma"] if te is not None else np.nan,
                         "accepted": st["accepted"], "label": st["label"]})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, (c, nm) in zip(axes, KEY_METRICS[:2]):
        sub = d[d.metric == c]
        x = sub.step.to_numpy()
        ax.errorbar(x - 0.08, sub.folds_delta, yerr=sub.folds_2s, fmt="o-", color=INK, capsize=4, label="среднее фолдов ± 2σ (отбор)")
        ax.errorbar(x + 0.08, sub.test_delta, yerr=sub.test_2s, fmt="s--", color=ORANGE, capsize=4, label="тест ± 2σ (протокол)", alpha=0.85)
        for xx, (_, r) in zip(x, sub.iterrows()):
            ax.annotate("принят" if r.accepted else "отклонён", (xx, r.folds_delta), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=8, color=GREEN if r.accepted else RED)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x); ax.set_xticklabels(sub.label, fontsize=8, rotation=15, ha="right")
        ax.set_title(f"{nm}: Δ к A")
        ax.grid(axis="y", color=GRID, lw=0.5)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Рис. 8. Путь жадной сборки road_geo_v2_exposure_a: Δ к A; решение на шаге 2 — по Δ к A + E3 "
                 "(среднее двух фолдов), тест — только протокол", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig8_greedy_path", d)


def fig9_final() -> None:
    runs = pd.read_csv(REPORTS / "v2_eval_stage6a.csv")
    vs = [("road_geo_v2_tags", "road_geo_v2_tags\n(33, до потери эт. 4)", PURPLE),
          (A_NAME, "A = road_geo_v2_skdf\n(32)", BLUE), ("road_geo_v2_exposure_a", "exposure_a\n(34)", GREEN)]
    mets = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("precision_top_1pct", "precision@1 %"), ("recall_top_10pct", "recall@10 %")]
    t = runs[runs.split == "test"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    rows = []
    for ax, (c, nm) in zip(axes, mets):
        for i, (v, lab, col) in enumerate(vs):
            d = t[t.variant == v].sort_values("seed")
            ax.scatter(np.full(len(d), i) + np.linspace(-0.12, 0.12, len(d)), d[c], color=col, s=26, alpha=0.7)
            ax.errorbar(i + 0.25, d[c].mean(), yerr=d[c].std(), fmt="o", color=INK, capsize=4)
            for _, r in d.iterrows():
                rows.append({"variant": v, "seed": r.seed, "metric": c, "value": r[c], "n_features": r.n_features})
        ax.set_xticks(range(len(vs))); ax.set_xticklabels([lab for _, lab, _ in vs], fontsize=8)
        ax.set_title(nm); ax.grid(axis="y", color=GRID, lw=0.5)
    fig.suptitle("Рис. 9. Тест 2023–2025: road_geo_v2_exposure_a против A и road_geo_v2_tags по 5 сидам "
                 "(точки — сиды, чёрное — среднее ± ст. откл.)", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig9_final_test_by_seed", pd.DataFrame(rows))


def fig10_importance() -> None:
    a = pd.read_csv(REPORTS / "catboost_feature_importance_stage6a_road_geo_v2_skdf_test.csv").set_index("feature")
    b = pd.read_csv(REPORTS / "catboost_feature_importance_stage6a_road_geo_v2_exposure_a_test.csv").set_index("feature")
    top = b.importance_mean.sort_values(ascending=False).index[:25]
    d = pd.DataFrame({"A": a.importance_mean.reindex(top), "exposure_a": b.importance_mean.reindex(top),
                      "exposure_a_sd": b.importance_sd.reindex(top)}).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 9.5))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d.A.fillna(0), height=0.4, color=BLUE, label="A = road_geo_v2_skdf")
    ax.barh(y - 0.2, d.exposure_a, height=0.4, color=GREEN, xerr=d.exposure_a_sd, label="road_geo_v2_exposure_a", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{f}  ★" if f in ev.EXPOSURE_V6_FEATURES else f for f in d.index], fontsize=8)
    ax.set_xlabel("важность CatBoost (PredictionValuesChange), среднее по 5 сидам, тест-модель")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_title("Рис. 10. Важности: топ-25 road_geo_v2_exposure_a против A (★ — признаки этапа 6a)", loc="left")
    fig.tight_layout()
    save(fig, "fig10_importance_top25", d.reset_index().rename(columns={"index": "feature"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="descr")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    if args.part == "descr":
        cells = load_cells()
        feats = [f for f in ev.EXPOSURE_V6_FEATURES if f not in ("building_count", "pt_stop_raw_count")]
        main_feats = ["dist_to_center_two_m", "dist_to_center_poi_m", "dist_to_nearest_major_m", "dist_to_nearest_signal_m",
                      "poi_attractor_count", "school_kindergarten_count", "healthcare_count", "poi_attractor_count_r300",
                      "building_footprint_m2", "small_building_share", "attractor_building_footprint_m2",
                      "pt_stop_count_v6", "dist_to_nearest_stop_m", "tram_track_length_m", "pt_route_count",
                      "levels_tagged_share"]
        jobs = {"1": lambda: fig1_center(cells), "2": lambda: fig2_poi_buildings(cells), "3": lambda: fig3_satellite(cells),
                "4": fig4_stops, "5": lambda: fig5_deciles(cells, main_feats), "6": lambda: fig6_spearman(feats)}
        for k, fn in jobs.items():
            if not only or k in only:
                fn()
    else:
        cfg = json.loads((TAB / "results_config.json").read_text())
        jobs = {"7": lambda: fig7_blocks(cfg["blocks"]), "8": fig8_greedy, "9": fig9_final, "10": fig10_importance}
        for k, fn in jobs.items():
            if not only or k in only:
                fn()


if __name__ == "__main__":
    main()
