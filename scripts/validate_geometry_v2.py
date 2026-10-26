"""Валидация этапа 4 (геометрия улиц без service) до/после обучения; фигуры 1–6.

1. Склейка дублей: фрагменты до/после, число перекрёстков и доля коротких кварталов от tol -> рис. 1.
2. Длины рёбер по классам, карта block_length_median_m -> рис. 2.
3. 6 улиц: сырая / упрощённая линия, учтённые вершины с радиусами -> рис. 3.
4. Старая против новой кривизны по ячейкам, цвет — доля service -> рис. 4 (+ ячейки с расхождением).
5. Тупики до/после на фрагменте -> рис. 5.
6. Одиночный AUC и доля нулей/NaN старых и новых признаков -> рис. 6; Spearman, пары |rho| >= 0.85.

Запуск: PYTHONPATH=src venv/bin/python scripts/validate_geometry_v2.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.catboost_baseline import (  # noqa: E402
    V2_GEOMETRY_BLOCKS_REPLACED, V2_GEOMETRY_CURVATURE_REPLACED, V2_GEOMETRY_OTHER_REPLACED,
    attach_geometry_v2_features, attach_junction_v2_features, attach_road_grade_features, attach_tags_v2_features,
    build_road_geo_v2_geometry_features, build_road_geo_v2_tags_features,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END  # noqa: E402
from road_accident_risk.osm_road_features import MAJOR_ROAD_CLASSES, MOTOR_ROAD_CLASSES  # noqa: E402
from road_accident_risk.paths import (  # noqa: E402
    H3_MONTH_ROAD_FEATURES_PARQUET, OSM_ROADS_RAW_PARQUET, OSM_STREET_BLOCKS_PARQUET, OSM_STREET_CURVE_VERTICES_PARQUET,
    OSM_STREET_GRAPH_CONSOLIDATED_EDGES_PARQUET, OSM_STREET_GRAPH_CONSOLIDATED_NODES_PARQUET,
    OSM_STREET_GRAPH_EDGES_PARQUET, OSM_STREET_GRAPH_NODES_PARQUET, PROJECT_ROOT,
)
from road_accident_risk.street_graph_v2 import (  # noqa: E402
    DP_TOL_M, GEOMETRY_V2_FEATURES, JUNCTION_CLEARANCE_M, build_street_blocks, douglas_peucker, utm38n_forward,
)

REPORTS = PROJECT_ROOT / "reports"
FIG = PROJECT_ROOT / "docs" / "analysis" / "features" / "stage4_geometry" / "figures"
C_BLUE, C_ORANGE, C_AQUA, C_GREY, C_VIOLET, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#b4b3ad", "#4a3aa7", "#c0392b"
INK, MUTED, GRID, LIGHT = "#0b0b0b", "#52514e", "#e4e3df", "#dcdbd6"
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
                ("residential", {"residential"}), ("living_street", {"living_street"})]
OLD_NEW = [  # старый -> новый (для таблицы и рис. 6)
    ("road_segment_length_median", "block_length_median_m"),
    ("short_segment_share", "block_length_median_m"),
    ("curvature_deg_per_km", "curvature_deg_per_km_v2"),
    ("road_curvature_mean_weighted", "curvature_deg_per_km_v2"),
    ("road_curvature_max", "min_curve_radius_m"),
    ("sharp_turn_share", "min_curve_radius_m"),
    ("road_orientation_entropy", "road_orientation_entropy_v2"),
    ("dead_end_count", "street_dead_end_count"),
    ("railway_crossing_density_per_km", "level_crossing_count"),
]
OLD = [o for o, _ in OLD_NEW]
NEW = ["block_length_median_m", "curvature_deg_per_km_v2", "min_curve_radius_m", "road_orientation_entropy_v2",
       "street_dead_end_count", "level_crossing_count", "share_length_radius_lt_200m"]


def save(fig, name: str, data: pd.DataFrame | None = None) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    if data is not None:
        data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  {name}.png / .csv", flush=True)


def cls_group(hw: str) -> str:
    for g, s in CLASS_GROUPS:
        if hw in s:
            return g
    return hw


def xy_lines(coords_json: pd.Series) -> list[np.ndarray]:
    out = []
    for c in coords_json:
        a = np.array(json.loads(c), float)
        x, y = utm38n_forward(a[:, 0], a[:, 1])
        out.append(np.c_[x, y])
    return out


def window_axes(ax, cx: float, cy: float, w: float, title: str) -> None:
    ax.set_xlim(cx - w, cx + w); ax.set_ylim(cy - w, cy + w); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(LIGHT)
    ax.set_title(title, loc="left", fontsize=9, color=INK)


def scalebar(ax, cx: float, cy: float, w: float, length: float = 50) -> None:
    x0, y0 = cx - w + 0.06 * 2 * w, cy - w + 0.05 * 2 * w
    ax.plot([x0, x0 + length], [y0, y0], color=INK, lw=2)
    ax.text(x0 + length / 2, y0 + 0.02 * 2 * w, f"{length:.0f} м", ha="center", fontsize=7.5, color=INK)


def single_auc(test: pd.DataFrame, f: str) -> tuple[float, float]:
    x = test[f].to_numpy(float); y = test[TARGET].to_numpy(int)
    mk = ~np.isnan(x)
    return float(roc_auc_score(y[mk], x[mk])), float(mk.mean())


def hexmap(ax, c: pd.DataFrame, col: str, cmap, norm, title: str):
    polys = [[(lng, lat) for lat, lng in h3.cell_to_boundary(h)] for h in c.index]
    pc = PolyCollection(polys, array=c[col].to_numpy(), cmap=cmap, norm=norm, edgecolor="none")
    ax.add_collection(pc)
    lats = [p[1] for poly in polys for p in poly]; lngs = [p[0] for poly in polys for p in poly]
    ax.set_xlim(min(lngs), max(lngs)); ax.set_ylim(min(lats), max(lats))
    ax.set_aspect(1 / np.cos(np.radians(51.55))); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    return pc


# --- рис. 1 -----------------------------------------------------------------------------------
def fig1(n: pd.DataFrame, e: pd.DataFrame, cn: pd.DataFrame, ce: pd.DataFrame, svc_lines: list[np.ndarray]) -> None:
    sweep = pd.read_csv(REPORTS / "road_geo_v2_geometry_tol_sweep.csv")
    multi = cn[cn.n_raw_nodes > 1]
    # окна: место с наибольшим числом кластеров на магистралях в радиусе 200 м и крупное кольцо
    from scipy.spatial import cKDTree
    major_nodes = set(ce.loc[ce.highway.isin(MAJOR_ROAD_CLASSES), "u"]) | set(ce.loc[ce.highway.isin(MAJOR_ROAD_CLASSES), "v"])
    multi_all = multi
    multi = multi[multi.node_id.isin(major_nodes) & ~multi.at_edge]
    t = cKDTree(multi[["x_utm", "y_utm"]].to_numpy())
    cnt = np.array([len(x) for x in t.query_ball_point(multi[["x_utm", "y_utm"]].to_numpy(), 200)])
    w1 = multi.iloc[int(np.argmax(cnt))]
    ring_ids = set(e.loc[e.is_roundabout, "u"]) | set(e.loc[e.is_roundabout, "v"])
    rings = multi_all[[bool(set(json.loads(r)) & ring_ids) for r in multi_all.raw_node_ids]]
    w2 = rings.sort_values("n_raw_nodes", ascending=False).iloc[0]
    raw_lines = xy_lines(e.coords_json); cons_lines = xy_lines(ce.coords_json)
    raw_major = e.highway.isin(MAJOR_ROAD_CLASSES).to_numpy(); cons_major = ce.highway.isin(MAJOR_ROAD_CLASSES).to_numpy()
    fig = plt.figure(figsize=(16, 10.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], hspace=0.2, wspace=0.22)
    windows = [(w1, 170, "раздельные магистрали"), (w2, 110, "кольцо")]
    rows = []
    for r, (wc, W, label) in enumerate(windows):
        cx, cy = wc.x_utm, wc.y_utm
        for k, (lines, mj, nodes, kind) in enumerate([(raw_lines, raw_major, n, "до склейки"),
                                                       (cons_lines, cons_major, cn, "после склейки")]):
            ax = fig.add_subplot(gs[r, k])
            near = lambda L: (np.abs(L[:, 0] - cx) < W + 50).any() and (np.abs(L[:, 1] - cy) < W + 50).any()  # noqa: E731
            ax.add_collection(LineCollection([L for L in svc_lines if near(L)], colors=LIGHT, lw=0.8, zorder=1))
            ax.add_collection(LineCollection([L for L, m in zip(lines, mj) if near(L) and not m], colors=MUTED, lw=1.3, zorder=2))
            ax.add_collection(LineCollection([L for L, m in zip(lines, mj) if near(L) and m], colors=C_BLUE, lw=1.5, zorder=2))
            nn = nodes[(abs(nodes.x_utm - cx) < W) & (abs(nodes.y_utm - cy) < W)]
            j = nn[nn.node_type == "junction"]
            if "n_raw_nodes" in nn:
                jm = j[j.n_raw_nodes > 1]; js = j[j.n_raw_nodes == 1]
                ax.scatter(js.x_utm, js.y_utm, s=26, color="white", edgecolor=INK, lw=0.9, zorder=4)
                ax.scatter(jm.x_utm, jm.y_utm, s=70, color=C_ORANGE, edgecolor="white", lw=1.0, zorder=5)
                for q in jm.itertuples():
                    ax.annotate(str(q.n_raw_nodes), (q.x_utm, q.y_utm), xytext=(5, 4), textcoords="offset points",
                                fontsize=7.5, color=INK)
            else:
                ax.scatter(j.x_utm, j.y_utm, s=26, color="white", edgecolor=INK, lw=0.9, zorder=4)
            d = nn[nn.node_type == "dead_end"]
            ax.scatter(d.x_utm, d.y_utm, s=16, marker="s", color=C_BLUE, zorder=4)
            window_axes(ax, cx, cy, W, f"{'аб'[k] if r == 0 else 'гд'[k]}) {label}, {kind}:\n"
                                      f"перекрёстков в окне {len(j)}")
            scalebar(ax, cx, cy, W, 50)
            rows.append({"panel": f"{label}: {kind}", "center_lat": wc.lat, "center_lng": wc.lng,
                         "half_width_m": W, "junctions_in_window": len(j)})
    ax = fig.add_subplot(gs[0, 2])
    for rb, col, ls, lab in [(False, C_GREY, "--", "только по расстоянию"), (True, C_BLUE, "-", "+ правило колец")]:
        sw = sweep[sweep.roundabouts == rb]
        ax.plot(sw.tol_m, sw.junctions, ls=ls, color=col, lw=2, marker="o", ms=6, label=lab)
    ax.axhline(8102, color=MUTED, lw=0.8, ls=":")
    ax.text(30, 8102 - 60, "сырой граф: 8 102", fontsize=8, color=MUTED, ha="right")
    ax.axvline(20, color=C_ORANGE, lw=1, alpha=0.6)
    ax.set_ylabel("перекрёстков (степень ≥ 3)"); ax.set_xticks([10, 15, 20, 25, 30])
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax.set_title("в) число перекрёстков от tol", loc="left", fontsize=9)
    ax2 = fig.add_subplot(gs[1, 2])
    sw = sweep[sweep.roundabouts]
    ax2.plot(sw.tol_m, 100 * sw.block_share_lt_25m, color=C_ORANGE, marker="o", ms=5, lw=2, label="кварталы < 25 м, %")
    ax2.plot(sw.tol_m, 100 * sw.block_share_lt_40m, color=C_VIOLET, marker="o", ms=5, lw=2, label="кварталы < 40 м, %")
    ax2.plot(sw.tol_m, sw.clusters_ge5, color=C_GREY, marker="s", ms=4, lw=1.4, label="кластеров ≥ 5 узлов, шт.")
    ax2.axvline(20, color=C_ORANGE, lw=1, alpha=0.6)
    ax2.set_xlabel("радиус склейки tol, м"); ax2.set_xticks([10, 15, 20, 25, 30])
    ax2.yaxis.grid(True, color=GRID, lw=0.6); ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.set_title("е) короткие кварталы и крупные кластеры (с правилом колец);\nсырой граф: < 25 м — 7.4 %, < 40 м — 13.4 %",
                  loc="left", fontsize=9)
    fig.suptitle("Рис. 1. Склейка узлов раздельных дорог и колец (tol 20 м): фрагменты до и после, выбор радиуса склейки\n"
                 "○ — перекрёсток, ● — узел-кластер (цифра — число склеенных узлов), ■ — тупик; синее — магистрали, "
                 "тёмно-серое — прочие улицы, светлое — service", x=0.01, ha="left", fontsize=10.5, y=0.995)
    save(fig, "fig1_consolidation", pd.concat([sweep, pd.DataFrame(rows)], ignore_index=True))


# --- рис. 2 -----------------------------------------------------------------------------------
def fig2(ce: pd.DataFrame, blocks: pd.DataFrame, cells: pd.DataFrame) -> None:
    jb = blocks[blocks.is_junction_block & ~blocks.at_edge].copy()
    jb["grp"] = jb.highway.map(cls_group)
    order = [g for g, _ in CLASS_GROUPS if g in set(jb.grp)]
    stats = jb.groupby("grp").length_m.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).reindex(order)
    stats["share_lt_25m"] = jb.groupby("grp").apply(lambda g: (g.length_m < 25).mean(), include_groups=False)
    ce2 = ce.assign(grp=ce.highway.map(cls_group))
    stats["edges_n"] = ce2.groupby("grp").size(); stats["edges_median_m"] = ce2.groupby("grp").length_m.median()
    stats.to_csv(REPORTS / "road_geo_v2_geometry_block_length_by_class.csv")
    print(stats.round(1).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.6), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.45})
    ax = axes[0]
    data = [np.log10(jb.loc[jb.grp == g, "length_m"]) for g in order]
    parts = ax.violinplot(data, vert=False, showextrema=False, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor(C_BLUE); b.set_alpha(0.25); b.set_edgecolor("none")
    for i, g in enumerate(order, start=1):
        q = np.log10(jb.loc[jb.grp == g, "length_m"].quantile([0.25, 0.5, 0.75]))
        ax.plot([q.iloc[0], q.iloc[2]], [i, i], color=C_BLUE, lw=3, solid_capstyle="round")
        ax.scatter(q.iloc[1], i, s=36, color="white", edgecolor=INK, zorder=3)
        ax.text(4.08, i, f"медиана {10 ** q.iloc[1]:.0f} м · n = {int((jb.grp == g).sum()):,}".replace(",", " "),
                va="center", fontsize=8, color=MUTED, clip_on=False)
    ax.set_yticks(range(1, len(order) + 1), order)
    ticks = [10, 25, 50, 100, 200, 500, 1000, 3000]
    ax.set_xticks(np.log10(ticks), [str(t) for t in ticks]); ax.set_xlim(0.8, 4.02)
    ax.set_xlabel("длина квартала (перекрёсток → перекрёсток), м, лог. шкала")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"а) кварталы consolidated-графа по классу (n = {len(jb):,}".replace(",", " ") + "; без края)",
                 loc="left", fontsize=10)
    c = cells[cells.block_length_median_m.notna()]
    pc = hexmap(axes[1], c, "block_length_median_m", SEQ_O.reversed(), LogNorm(vmin=40, vmax=2000),
                f"б) block_length_median_m по ячейкам (NaN — {cells.block_length_median_m.isna().mean():.0%} ячеек не показаны)")
    cb = fig.colorbar(pc, ax=axes[1], shrink=0.75, pad=0.01)
    cb.set_label("взвешенная медиана длины квартала, м (лог.)")
    fig.suptitle("Рис. 2. Длина квартала: распределение по классам улиц и карта по ячейкам H3 (тёмное — короткие кварталы)",
                 x=0.01, ha="left", fontsize=11, y=1.0)
    save(fig, "fig2_block_length", stats.reset_index().rename(columns={"grp": "class_group"}))


# --- рис. 3 -----------------------------------------------------------------------------------
def fig3(n: pd.DataFrame, cv: pd.DataFrame) -> None:
    e = pd.read_parquet(OSM_STREET_GRAPH_EDGES_PARQUET)
    raw_blocks, _ = build_street_blocks(n, e)
    c = cv[cv.counted]
    agg = c.groupby("raw_block_id").agg(turn=("turn_deg", "sum"), rmin=("radius_m", "min"))
    b = raw_blocks.set_index("block_id").join(agg)
    b["turn_per_km"] = b.turn / (b.length_m / 1000)
    rng = np.random.default_rng(0)
    picks = []
    for cls, lo, hi in [("trunk", 300, 3000), ("primary", 300, 2000), ("secondary", 250, 2000), ("tertiary", 200, 2000),
                        ("unclassified", 250, 1500), ("residential", 150, 800)]:
        cand = b[(b.highway.map(cls_group) == cls) & b.length_m.between(lo, hi) & (b.rmin < 300) & ~b.at_edge]
        cand = cand.sort_values("turn_per_km", ascending=False).head(15)
        picks.append(cand.index[rng.integers(len(cand))])
    jx = n[n.node_type == "junction"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    rows = []
    rcol = lambda r: C_RED if r < 50 else C_ORANGE if r < 200 else C_BLUE if r < 1000 else "#9ec5f0"  # noqa: E731
    for ax, bid in zip(axes.flat, picks):
        r = b.loc[bid]
        cs = np.array(json.loads(r.coords_json)); x, y = utm38n_forward(cs[:, 0], cs[:, 1]); xy = np.c_[x, y]
        k3 = douglas_peucker(xy, DP_TOL_M)
        ax.plot(xy[:, 0], xy[:, 1], color=C_GREY, lw=5, solid_capstyle="round", alpha=0.6, zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=5, color=MUTED, zorder=2)
        ax.plot(xy[k3, 0], xy[k3, 1], color=INK, lw=1.0, zorder=3)
        vv = cv[cv.raw_block_id == bid]
        for q in vv.itertuples():
            if not q.counted:
                ax.scatter(q.x_utm, q.y_utm, s=34, marker="x", color=MUTED, zorder=5)
                continue
            if np.isnan(q.radius_m):
                ax.scatter(q.x_utm, q.y_utm, s=26, color="white", edgecolor=MUTED, zorder=5)
                continue
            ax.scatter(q.x_utm, q.y_utm, s=34, color=rcol(q.radius_m), edgecolor="white", lw=0.8, zorder=6)
            if q.radius_m < 1000:
                ax.annotate(f"R {q.radius_m:.0f}", (q.x_utm, q.y_utm), fontsize=7, xytext=(4, 3),
                            textcoords="offset points", color=INK)
            rows.append({"raw_block_id": bid, "highway": r.highway, "lat": q.lat, "lng": q.lng,
                         "turn_deg": q.turn_deg, "radius_m": q.radius_m})
        pad = 45
        xmin, xmax, ymin, ymax = xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad
        jj = jx[jx.x_utm.between(xmin, xmax) & jx.y_utm.between(ymin, ymax)]
        ax.scatter(jj.x_utm, jj.y_utm, s=40, facecolor="white", edgecolor=INK, zorder=7)
        for xx, yy in jj[["x_utm", "y_utm"]].to_numpy():
            ax.add_patch(plt.Circle((xx, yy), JUNCTION_CLEARANCE_M, fill=False, ls=":", color=INK, lw=0.7))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(LIGHT)
        lat, lng = cs.mean(0)
        ax.set_title(f"{r.highway}, {r.length_m:.0f} м: Σ поворота {r.turn:.0f}°, мин. R {r.rmin:.0f} м\n"
                     f"({lat:.5f}, {lng:.5f})", fontsize=9, loc="left")
    handles = [plt.Line2D([], [], color=C_GREY, lw=5, alpha=0.6), plt.Line2D([], [], color=INK, lw=1),
               plt.Line2D([], [], ls="", marker="o", color=C_RED), plt.Line2D([], [], ls="", marker="o", color=C_ORANGE),
               plt.Line2D([], [], ls="", marker="o", color=C_BLUE), plt.Line2D([], [], ls="", marker="o", color="#9ec5f0"),
               plt.Line2D([], [], ls="", marker="x", color=MUTED),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK)]
    labels = ["сырая линия OSM (точки — вершины)", f"упрощённая, Douglas–Peucker {DP_TOL_M:.0f} м",
              "R < 50 м", "R 50–200 м", "R 200–1000 м", "R ≥ 1000 м (прямая)",
              f"вершина не учтена (ближе {JUNCTION_CLEARANCE_M:.0f} м к перекрёстку)",
              f"перекрёсток и зона {JUNCTION_CLEARANCE_M:.0f} м"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Рис. 3. Кривизна и радиусы на шести улицах разных классов: сырая и упрощённая линия, радиус окружности "
                 "через три соседние вершины (звенья ≥ 10 м)", x=0.01, ha="left", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig3_curves_radii", pd.DataFrame(rows))
    # гистограмма радиусов по классам — в CSV для README
    cc = c[c.radius_m.notna()].assign(grp=lambda d: d.highway.map(cls_group))
    h = cc.groupby("grp").radius_m.describe(percentiles=[0.05, 0.25, 0.5]).reindex([g for g, _ in CLASS_GROUPS])
    h["share_lt_50"] = cc.groupby("grp").radius_m.apply(lambda s: (s < 50).mean())
    h["share_lt_200"] = cc.groupby("grp").radius_m.apply(lambda s: (s < 200).mean())
    h.to_csv(REPORTS / "road_geo_v2_geometry_radius_by_class.csv")
    print(h.round(3).to_string())


# --- рис. 4 -----------------------------------------------------------------------------------
def fig4(cells: pd.DataFrame) -> pd.DataFrame:
    c = cells[cells.curvature_deg_per_km_v2.notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, (xo, yn, xl, yl, logx, logy) in zip(axes, [
        ("curvature_deg_per_km", "curvature_deg_per_km_v2", "старая curvature_deg_per_km, °/км (все классы, все вершины)",
         "новая curvature_deg_per_km_v2, °/км (улицы, DP 3 м, без зоны перекрёстков)", True, False),
        ("road_curvature_max", "min_curve_radius_m", "старая road_curvature_max, ° (макс. угол в вершине)",
         "новая min_curve_radius_m, м (cap 1000)", False, True)]):
        cmap4 = LinearSegmentedColormap.from_list("o4", ["#f7cfb4", "#eb6834", "#8a3110", "#3d1607"])
        sc = ax.scatter(c[xo] + (1 if logx else 0), c[yn], c=c.road_share_service, s=7,
                        cmap=cmap4, vmin=0, vmax=1, alpha=0.8, lw=0)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        rk = c[[xo, yn, "road_share_service"]].rank().corr()
        rho = rk.loc[xo, yn]
        ax.set_xlabel(xl + (" + 1, лог." if logx else ""), fontsize=9); ax.set_ylabel(yl, fontsize=9)
        ax.set_title(f"{'аб'[list(axes).index(ax)]}) ρ(старый ↔ новый) = {rho:.2f}; ρ с долей service — "
                     f"старый {rk.loc[xo, 'road_share_service']:+.2f} / новый {rk.loc[yn, 'road_share_service']:+.2f}\n"
                     f"(n = {len(c):,} ячеек с улицами ≥ 200 м)".replace(",", " ").replace("  ", " "),
                     loc="left", fontsize=9.5)
        ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.01); cb.set_label("доля service в длине дорог ячейки")
    fig.suptitle("Рис. 4. Старая и новая кривизна по ячейкам (цвет — доля service); вертикальная полоса на б) — "
                 "прямые углы ≈ 90° парковочных проездов",
                 x=0.01, ha="left", fontsize=11, y=1.0)
    save(fig, "fig4_old_vs_new_curvature", c[["curvature_deg_per_km", "curvature_deg_per_km_v2", "road_curvature_max",
                                              "min_curve_radius_m", "road_share_service"]].reset_index())
    # ячейки с крупным расхождением рангов
    c["r_old"] = c.curvature_deg_per_km.rank(pct=True); c["r_new"] = c.curvature_deg_per_km_v2.rank(pct=True)
    c["gap"] = c.r_old - c.r_new
    out = pd.concat([c.nlargest(4, "gap").assign(case="старая ≫ новой"), c.nsmallest(4, "gap").assign(case="новая ≫ старой")])
    out["lat"] = [h3.cell_to_latlng(h)[0] for h in out.index]; out["lng"] = [h3.cell_to_latlng(h)[1] for h in out.index]
    out["osm_url"] = [f"https://www.openstreetmap.org/#map=17/{a:.5f}/{b:.5f}" for a, b in zip(out.lat, out.lng)]
    out = out.reset_index()[["case", "h3", "lat", "lng", "curvature_deg_per_km", "curvature_deg_per_km_v2",
                             "road_curvature_max", "min_curve_radius_m", "road_share_service", "street_length_m", "osm_url"]]
    out.to_csv(FIG / "manual_check_curvature_cells.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out


# --- рис. 5 -----------------------------------------------------------------------------------
def old_dead_ends(roads: pd.DataFrame) -> pd.DataFrame:
    """Старое определение (reconcile_group3_geometry_features.py): вершина с одним соседом, все motor-классы."""
    adj: dict[int, set[int]] = defaultdict(set); coord: dict[int, tuple[float, float]] = {}
    for ids, cs in zip(roads.nodes_json, roads.coords_json):
        ids = [int(x) for x in json.loads(ids)]; cs = json.loads(cs)
        if len(ids) != len(cs) or len(ids) < 2:
            continue
        for i, c in zip(ids, cs):
            coord[i] = (c[0], c[1])
        for a, b in zip(ids, ids[1:]):
            if a != b:
                adj[a].add(b); adj[b].add(a)
    rows = [(k, *coord[k]) for k, v in adj.items() if len(v) == 1]
    d = pd.DataFrame(rows, columns=["node_id", "lat", "lng"])
    d["x_utm"], d["y_utm"] = utm38n_forward(d.lat.to_numpy(), d.lng.to_numpy())
    return d


def fig5(roads: pd.DataFrame, cn: pd.DataFrame, ce: pd.DataFrame, svc_lines: list[np.ndarray]) -> dict:
    od = old_dead_ends(roads)
    svc_nodes = set()
    for hw, ids in zip(roads.highway, roads.nodes_json):
        if hw == "service":
            svc_nodes.update(int(x) for x in json.loads(ids))
    street_nodes = set()
    for hw, ids in zip(roads.highway, roads.nodes_json):
        if hw != "service":
            street_nodes.update(int(x) for x in json.loads(ids))
    od["is_service_end"] = od.node_id.isin(svc_nodes) & ~od.node_id.isin(street_nodes)
    nd = cn[cn.node_type == "dead_end"]
    stats = {"old_dead_ends": len(od), "old_service_ends": int(od.is_service_end.sum()),
             "old_service_share": float(od.is_service_end.mean()), "new_dead_ends": len(nd),
             "new_dead_ends_at_edge": int(nd.at_edge.sum()),
             "new_with_service_continuation": int((nd.degree_service > 0).sum())}
    print(stats)
    # окно: много старых тупиков и несколько новых
    from scipy.spatial import cKDTree
    W = 260
    t_old = cKDTree(od[["x_utm", "y_utm"]].to_numpy()); t_new = cKDTree(nd[["x_utm", "y_utm"]].to_numpy())
    cand = nd[~nd.at_edge][["x_utm", "y_utm"]].to_numpy()
    n_old = np.array([len(x) for x in t_old.query_ball_point(cand, W)])
    n_new = np.array([len(x) for x in t_new.query_ball_point(cand, W)])
    score = np.where(n_new >= 4, n_old, 0)
    cx, cy = cand[int(np.argmax(score))]
    street_lines = xy_lines(ce.coords_json)
    near = lambda L: (np.abs(L[:, 0] - cx) < W + 50).any() and (np.abs(L[:, 1] - cy) < W + 50).any()  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2))
    for k, ax in enumerate(axes):
        ax.add_collection(LineCollection([L for L in svc_lines if near(L)], colors=C_GREY, lw=0.9, zorder=1))
        ax.add_collection(LineCollection([L for L in street_lines if near(L)], colors=INK, lw=1.6, zorder=2))
        if k == 0:
            o = od[(abs(od.x_utm - cx) < W) & (abs(od.y_utm - cy) < W)]
            ax.scatter(o[o.is_service_end].x_utm, o[o.is_service_end].y_utm, s=22, color=C_ORANGE, zorder=4,
                       label=f"конец service ({int(o.is_service_end.sum())})")
            ax.scatter(o[~o.is_service_end].x_utm, o[~o.is_service_end].y_utm, s=46, color=C_BLUE, zorder=5,
                       edgecolor="white", label=f"конец улицы ({int((~o.is_service_end).sum())})")
            title = f"а) старый dead_end_count: {len(o)} тупиков в окне (все классы)"
        else:
            o = nd[(abs(nd.x_utm - cx) < W) & (abs(nd.y_utm - cy) < W)]
            ax.scatter(o.x_utm, o.y_utm, s=46, color=C_BLUE, zorder=5, edgecolor="white",
                       label=f"тупик уличного графа ({len(o)})")
            title = f"б) новый street_dead_end_count: {len(o)} тупиков в окне (уличная сеть)"
        window_axes(ax, cx, cy, W, title); scalebar(ax, cx, cy, W, 100)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.9)
    lat = cn.loc[(cn.x_utm - cx).abs().idxmin()] if False else None  # noqa: F841
    fig.suptitle("Рис. 5. Тупики до и после: старый признак считает концы проездов service (парковочные ряды, дворы), "
                 "новый — только тупиковые улицы\n(в т. ч. улицы, переходящие в проезд service: для транзита это тупик; "
                 "чёрное — улицы, серое — service; окно 520 × 520 м)",
                 x=0.01, ha="left", fontsize=10.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_dead_ends_before_after", pd.DataFrame([{**stats, "window_x_utm": cx, "window_y_utm": cy}]))
    return stats


# --- рис. 6 -----------------------------------------------------------------------------------
def fig6(stats: pd.DataFrame) -> None:
    order = []
    for o, nw in OLD_NEW:
        if o not in order:
            order.append(o)
        if nw not in order:
            order.append(nw)
    order.append("share_length_radius_lt_200m")
    d = stats.set_index("feature").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(d))[::-1]
    col = [C_ORANGE if f in OLD else C_BLUE for f in d.feature]
    axes[0].barh(y, d.single_auc_abs - 0.5, left=0.5, color=col, height=0.66)
    for yi, v, raw in zip(y, d.single_auc_abs, d.single_auc_test):
        axes[0].text(v + 0.004, yi, f"{v:.3f}{' (обр.)' if raw < 0.5 else ''}", va="center", fontsize=8, color=INK)
    axes[0].set_yticks(y, d.feature, fontsize=8.5); axes[0].set_xlim(0.5, 0.78)
    axes[0].set_xlabel("одиночный ROC-AUC на тесте 2023–2025, max(AUC, 1 − AUC), по строкам без NaN")
    axes[0].xaxis.grid(True, color=GRID, lw=0.6); axes[0].set_axisbelow(True)
    axes[0].set_title("а) одиночный AUC", loc="left", fontsize=10)
    axes[1].barh(y + 0.17, 100 * d.zero_share_cells, height=0.32, color=C_GREY, label="нули")
    axes[1].barh(y - 0.17, 100 * d.nan_share_cells, height=0.32, color=C_VIOLET, label="NaN")
    for yi, z, nn in zip(y, d.zero_share_cells, d.nan_share_cells):
        axes[1].text(100 * z + 1, yi + 0.17, f"{z:.0%}", va="center", fontsize=7.5, color=MUTED)
        if nn > 0:
            axes[1].text(100 * nn + 1, yi - 0.17, f"{nn:.0%}", va="center", fontsize=7.5, color=MUTED)
    axes[1].set_xlim(0, 110); axes[1].set_xlabel("доля ячеек панели, %")
    axes[1].xaxis.grid(True, color=GRID, lw=0.6); axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="lower right"); axes[1].set_title("б) доля нулей и NaN", loc="left", fontsize=10)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_ORANGE), plt.Rectangle((0, 0), 1, 1, color=C_BLUE)],
                   labels=["старый признак (заменяется)", "новый признак этапа 4"], loc="lower right", frameon=False,
                   fontsize=8.5)
    fig.suptitle("Рис. 6. Старые и новые признаки геометрии: одиночный AUC и доля нулей / NaN (4 296 ячеек панели)",
                 x=0.01, ha="left", fontsize=11, y=1.0)
    fig.tight_layout()
    save(fig, "fig6_single_auc_zero_nan", d)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    roads = pd.read_parquet(OSM_ROADS_RAW_PARQUET, columns=["highway", "nodes_json", "coords_json"])
    n = pd.read_parquet(OSM_STREET_GRAPH_NODES_PARQUET); e = pd.read_parquet(OSM_STREET_GRAPH_EDGES_PARQUET)
    cn = pd.read_parquet(OSM_STREET_GRAPH_CONSOLIDATED_NODES_PARQUET)
    ce = pd.read_parquet(OSM_STREET_GRAPH_CONSOLIDATED_EDGES_PARQUET)
    blocks = pd.read_parquet(OSM_STREET_BLOCKS_PARQUET); cv = pd.read_parquet(OSM_STREET_CURVE_VERTICES_PARQUET)
    svc_lines = xy_lines(roads.loc[roads.highway == "service", "coords_json"])

    panel = attach_geometry_v2_features(attach_tags_v2_features(attach_junction_v2_features(
        attach_road_grade_features(pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)))))
    panel = panel[panel.road_length_total_m > 0].copy()
    panel["month"] = pd.to_datetime(panel["month"])
    cells = panel.drop_duplicates("h3").set_index("h3")
    test = panel[panel.month >= TRAIN_END]
    A = build_road_geo_v2_tags_features(list(panel.columns)); B = build_road_geo_v2_geometry_features(list(panel.columns))
    print(f"ячеек {len(cells)}, тестовых строк {len(test)}; A {len(A)}, B {len(B)}", flush=True)

    # --- статистики, Spearman, пары
    corr = cells[sorted(set(A) | set(B) | set(GEOMETRY_V2_FEATURES))].rank().corr()
    rows = []
    for f in [*OLD, *[x for x in NEW if x not in OLD]]:
        auc, frac = single_auc(test, f)
        rows.append({"feature": f, "group": "old" if f in OLD else "new", "zero_share_cells": float((cells[f] == 0).mean()),
                     "nan_share_cells": float(cells[f].isna().mean()), "single_auc_test": auc,
                     "single_auc_abs": max(auc, 1 - auc), "auc_rows_share": frac,
                     "rho_with_replacement": float(corr.loc[f, dict(OLD_NEW)[f]]) if f in dict(OLD_NEW) else np.nan,
                     "max_abs_rho_in_B": float(corr.loc[f, [x for x in B if x != f]].abs().max()) if f in B else np.nan})
    stats = pd.DataFrame(rows).drop_duplicates("feature")
    stats.to_csv(REPORTS / "road_geo_v2_geometry_feature_stats.csv", index=False)
    with pd.option_context("display.width", 250):
        print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    newB = [f for f in B if f not in A]
    pairs = [{"feature_a": a, "feature_b": b, "spearman": float(corr.loc[a, b])}
             for a in [*newB, "share_length_radius_lt_200m"] for b in sorted(set(B) | set(A) | {"share_length_radius_lt_200m"})
             if a != b and abs(corr.loc[a, b]) >= 0.85]
    pd.DataFrame(pairs).to_csv(REPORTS / "road_geo_v2_geometry_spearman_pairs.csv", index=False)
    corr.to_csv(REPORTS / "road_geo_v2_geometry_spearman_matrix.csv")
    print("пары |rho| >= 0.85 с новыми признаками:\n", pd.DataFrame(pairs).to_string())
    top = corr.loc[newB, [x for x in B]].copy()
    for f in newB:
        s = top.loc[f].drop(f).abs().sort_values(ascending=False).head(3)
        print(f"  {f}: " + ", ".join(f"{k} {corr.loc[f, k]:+.3f}" for k in s.index))
    j = ["street_junction_3leg_count", "street_junction_4leg_count"]
    print("block_length_median_m ~", {k: round(float(corr.loc['block_length_median_m', k]), 3) for k in j})
    print("dead_end_count ~ driveway_count:", round(float(corr.loc["dead_end_count", "driveway_count"]), 3)
          if "dead_end_count" in corr and "driveway_count" in corr else cells[["dead_end_count", "driveway_count"]].rank().corr().iloc[0, 1])

    only = set(sys.argv[1:])  # например: fig1 fig3 — перерисовать только эти
    run = lambda k: not only or k in only  # noqa: E731
    if run("fig1"):
        fig1(n, e, cn, ce, svc_lines)
    if run("fig2"):
        fig2(ce, blocks, cells)
    if run("fig3"):
        fig3(n, cv)
    if run("fig4"):
        fig4(cells)
    if run("fig5"):
        fig5(roads, cn, ce, svc_lines)
    if run("fig6"):
        fig6(stats)


if __name__ == "__main__":
    main()
