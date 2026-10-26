"""Валидация этапа 2 (перекрёстки с честной топологией) до обучения моделей.

Сравнивает старую классификацию узлов (osm_road_graph_nodes.parquet, junction_degree = число
вхождений узла в ways) с новой (osm_junction_v2_nodes.parquet, степень = число подходов),
считает Spearman, одиночный AUC и долю нулей новых признаков, готовит выборку узлов для
ручной проверки и рисует фигуры 1–5 и 8 в docs/analysis/features/stage2_junctions/figures/.

Запуск: PYTHONPATH=src venv/bin/python scripts/validate_junction_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.catboost_baseline import (  # noqa: E402
    FEATURE_SET_ROAD_GEO_V2_PRUNE, V2_JUNCTIONS_REPLACED_FEATURES, attach_junction_v2_features,
    attach_road_grade_features, build_features,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END  # noqa: E402
from road_accident_risk.osm_road_features import JUNCTION_V2_FEATURES, STREET_ROAD_CLASSES  # noqa: E402
from road_accident_risk.paths import (  # noqa: E402
    H3_MONTH_ROAD_FEATURES_PARQUET, OSM_JUNCTION_V2_NODES_PARQUET, OSM_ROAD_GRAPH_NODES_PARQUET,
    OSM_ROADS_RAW_PARQUET, PROJECT_ROOT,
)

REPORTS = PROJECT_ROOT / "reports"
FIG = PROJECT_ROOT / "docs" / "analysis" / "features" / "stage2_junctions" / "figures"
OLD_JUNCTION = list(V2_JUNCTIONS_REPLACED_FEATURES)
# окно карты: фрагмент центра Саратова с наибольшим числом регулируемых перекрёстков (~0.9 x 0.9 км)
MAP_BOX = (51.5420, 46.0050, 51.5500, 46.0180)  # lat_min, lng_min, lat_max, lng_max

C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9ec5f0", "#2a78d6", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8f1d1c"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name: str, data: pd.DataFrame | None = None, index: bool = False) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if data is not None:
        data.to_csv(FIG / f"{name}.csv", index=index)


def bucket(v: float) -> str:
    return "5+" if v >= 5 else str(int(v))


# ---------------------------------------------------------------- 1. переходы степеней
def transitions(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    m = new.merge(old[["node_id", "junction_degree"]], on="node_id", how="outer")
    assert m["degree_full"].notna().all(), "узел старого графа не найден в новой таблице"
    m["old"] = m["junction_degree"].map(lambda v: "нет в старом\nграфе" if pd.isna(v) else bucket(v))
    m["new_full"] = m["degree_full"].map(bucket)
    m["street_type"] = np.select(
        [m.degree_street >= 4, m.degree_street == 3, m.is_driveway],
        ["улица: 4+", "улица: 3", "въезд service"], "не уличный\nперекрёсток")
    return m


def fig1(m: pd.DataFrame) -> None:
    rows = ["2", "3", "4", "5+", "нет в старом\nграфе"]
    t1 = pd.crosstab(m["old"], m["new_full"]).reindex(index=rows, columns=["2", "3", "4", "5+"], fill_value=0)
    cols2 = ["улица: 3", "улица: 4+", "въезд service", "не уличный\nперекрёсток"]
    t2 = pd.crosstab(m["old"], m["street_type"]).reindex(index=rows, columns=cols2, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [4, 4]})
    for ax, t, title, xl in [
        (axes[0], t1, "а) старая степень × новая степень (полная сеть)", "новая степень = число подходов"),
        (axes[1], t2, "б) старая степень × новый тип узла (уличная сеть)", "новый тип узла"),
    ]:
        v = t.to_numpy().astype(float)
        ax.imshow(np.log10(v + 1), cmap=SEQ, aspect="auto", vmin=0, vmax=np.log10(v.max() + 1))
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                val = int(v[i, j])
                dark = np.log10(val + 1) > 0.6 * np.log10(v.max() + 1)
                ax.text(j, i, f"{val:,}".replace(",", " "), ha="center", va="center", fontsize=9,
                        color="white" if dark else INK)
        ax.set_xticks(range(v.shape[1]), t.columns, fontsize=9)
        ax.set_yticks(range(v.shape[0]), t.index, fontsize=9)
        ax.set_xlabel(xl); ax.set_title(title, fontsize=10.5, loc="left")
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
    axes[0].set_ylabel("старая «степень» = число вхождений узла в ways")
    fig.suptitle("Рис. 1. Переходы между старой и новой классификацией узлов OSM-графа Саратова "
                 "(цвет — log10 числа узлов)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out = pd.concat({"old_x_new_full_degree": t1, "old_x_new_street_type": t2}, axis=1)
    out.index = [i.replace("\n", " ") for i in out.index]
    out.columns = [f"{a}|{b.replace(chr(10), ' ')}" for a, b in out.columns]
    save(fig, "fig1_degree_transition", out, index=True)


# ---------------------------------------------------------------- 2. типы узлов до/после
def fig2(old: pd.DataFrame, new: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    old_m = m[m.junction_degree.notna()]
    rows = [
        ("3-сторонние", int((old.junction_degree == 3).sum()), int(new.is_street_3leg.sum())),
        ("4-сторонние и более", int((old.junction_degree >= 4).sum()), int(new.is_street_4leg.sum())),
        ("с участием service", int(old.connected_highway_classes.str.contains("service").sum()),
         int(new.is_driveway.sum())),
        ("не перекрёсток\n(смена имени/класса)", int((old_m.degree_full == 2).sum()), 0),
    ]
    d = pd.DataFrame(rows, columns=["тип", "старая классификация", "новая классификация"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(d)); w = 0.38
    for k, (col, c) in enumerate([("старая классификация", C_ORANGE), ("новая классификация", C_BLUE)]):
        bars = ax.bar(x + (k - 0.5) * w, d[col], w - 0.03, color=c, label=col)
        for b, v in zip(bars, d[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 150, f"{v:,}".replace(",", " "),
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, d["тип"]); ax.set_ylabel("число узлов")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Рис. 2. Число узлов по типам: старая классификация (степень = число ways, все классы)\n"
                 "и новая (степень = число подходов, уличная сеть; service — отдельно как въезды)",
                 fontsize=10.5, loc="left")
    ax.text(0.0, -0.2, "Старое «с участием service» — узлы графа, где среди классов есть service (66.5 %);\n"
            "новое — въезды: service примыкает к проходящей улице (degree_street ≥ 2, degree_service ≥ 1).",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    d["тип"] = d["тип"].str.replace("\n", " ")
    save(fig, "fig2_node_types_before_after", d)
    return d


# ---------------------------------------------------------------- 3. карта центра
def fig3(roads: pd.DataFrame, old: pd.DataFrame, new: pd.DataFrame) -> None:
    la0, lo0, la1, lo1 = MAP_BOX
    pad = 0.0015
    kx = np.cos(np.radians((la0 + la1) / 2))
    edges = []
    for r in roads.itertuples(index=False):
        c = np.array(json.loads(r.coords_json))
        if not ((c[:, 0] > la0 - pad) & (c[:, 0] < la1 + pad) & (c[:, 1] > lo0 - pad) & (c[:, 1] < lo1 + pad)).any():
            continue
        edges.append((r.highway, c))
    inb = lambda d, la="lat", lo="lng": d[d[la].between(la0, la1) & d[lo].between(lo0, lo1)]  # noqa: E731
    o, n = inb(old), inb(new)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    for ax in axes:
        for hw, c in edges:
            street = hw in STREET_ROAD_CLASSES
            ax.plot(c[:, 1], c[:, 0], color="#9b9a95" if street else "#d9d8d3",
                    lw=1.3 if street else 0.6, zorder=1, solid_capstyle="round")
        ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1); ax.set_aspect(1 / kx)
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
    # старая: цвет по старой степени, у всех узлов
    old_style = [(2, "#c3c2b7", "o", 14, "«степень 2»"), (3, C_BLUE, "o", 22, "«степень 3»"),
                 (4, C_ORANGE, "s", 26, "«степень 4+»")]
    a = axes[0]
    for deg, col, mk, sz, lab in old_style:
        d = o[o.junction_degree == deg] if deg < 4 else o[o.junction_degree >= 4]
        a.scatter(d.lng, d.lat, s=sz, c=col, marker=mk, edgecolors="white", linewidths=0.5, zorder=3,
                  label=f"{lab}: {len(d)}")
    a.legend(loc="lower left", frameon=True, fontsize=8, title="старая степень = число ways",
             title_fontsize=8)
    a.set_title(f"а) старая классификация: {len(o)} узлов «перекрёстков»", fontsize=10.5, loc="left")
    b = axes[1]
    dw = n[n.is_driveway & ~n.is_street_junction]
    b.scatter(dw.lng, dw.lat, s=10, c=C_AQUA, marker="v", edgecolors="none", zorder=2,
              label=f"въезд service → улица: {len(dw)}")
    t3, t4 = n[n.is_street_3leg], n[n.is_street_4leg]
    b.scatter(t3.lng, t3.lat, s=24, c=C_BLUE, marker="o", edgecolors="white", linewidths=0.5, zorder=3,
              label=f"3-сторонний: {len(t3)}")
    b.scatter(t4.lng, t4.lat, s=30, c=C_ORANGE, marker="s", edgecolors="white", linewidths=0.5, zorder=3,
              label=f"4-сторонний: {len(t4)}")
    sg = n[n.is_signalized_junction]
    b.scatter(sg.lng, sg.lat, s=90, facecolors="none", edgecolors=C_VIOLET, linewidths=1.2, zorder=4,
              label=f"со светофором ≤ 30 м: {len(sg)}")
    b.legend(loc="lower left", frameon=True, fontsize=8, title="новая степень = число подходов",
             title_fontsize=8)
    b.set_title(f"б) новая классификация: {len(t3) + len(t4)} уличных перекрёстков + {len(dw)} въездов",
                fontsize=10.5, loc="left")
    # масштабная линейка 200 м
    for ax in axes:
        x0 = lo1 - 0.0045; y0 = la0 + 0.0005; dx = 200 / (111_320 * kx)
        ax.plot([x0, x0 + dx], [y0, y0], color=INK, lw=2); ax.text(x0 + dx / 2, y0 + 0.00015, "200 м",
                                                                     ha="center", fontsize=8)
    fig.suptitle("Рис. 3. Фрагмент центра Саратова: улицы — тёмно-серые, проезды service — светло-серые. "
                 f"Окно {la0}–{la1} с. ш., {lo0}–{lo1} в. д.", fontsize=10.5, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    cols = ["node_id", "lat", "lng"]
    out = pd.concat([
        o[cols + ["junction_degree", "connected_highway_classes"]].assign(classification="old"),
        n[cols + ["degree_full", "degree_street", "degree_service", "is_street_3leg", "is_street_4leg",
                  "is_driveway", "is_signalized_junction", "street_classes"]].assign(classification="new"),
    ])
    save(fig, "fig3_map_center_old_vs_new", out)


# ---------------------------------------------------------------- 8. ручная проверка
def manual_sample(new: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    mm = m.set_index("node_id")
    in_old = mm.junction_degree.notna()
    picks = [
        ("старая 2 → Т (улица)", (mm.junction_degree == 2) & (mm.degree_street == 3), 2),
        ("старая 2 → крест (улица)", (mm.junction_degree == 2) & (mm.degree_street == 4), 2),
        ("старый узел, степень 2: смена класса", in_old & (mm.degree_full == 2) & (mm.all_classes.str.contains(";")), 1),
        ("старый узел, степень 2: смена имени", in_old & (mm.degree_full == 2) & ~mm.all_classes.str.contains(";"), 1),
        ("нет в старом → уличный перекрёсток", ~in_old & mm.is_street_junction, 2),
        ("крест со светофором", mm.is_signalized_junction & (mm.degree_street == 4), 1),
        ("магистраль × улица без светофора", mm.is_major_x_street_unsignalized, 2),
        ("въезд service → улица", mm.is_driveway & ~mm.is_street_junction, 2),
        ("уличный перекрёсток, угол > 40°", mm.is_street_junction & (mm.junction_angle_dev > 40), 1),
        ("только service, степень ≥ 3", (mm.degree_street == 0) & (mm.degree_full >= 3), 1),
    ]
    rows = []
    for label, mask, k in picks:
        ids = mm.index[mask.fillna(False).to_numpy()]
        for nid in rng.choice(ids, size=min(k, len(ids)), replace=False):
            r = mm.loc[nid]
            rows.append({"case": label, "node_id": int(nid), "lat": round(r.lat, 6), "lng": round(r.lng, 6),
                         "old_degree": None if pd.isna(r.junction_degree) else int(r.junction_degree),
                         "degree_full": int(r.degree_full), "degree_street": int(r.degree_street),
                         "degree_service": int(r.degree_service), "way_count": int(r.way_count),
                         "all_classes": r.all_classes, "names": r.names,
                         "signal_m": None if pd.isna(r.nearest_traffic_signal_m) else round(r.nearest_traffic_signal_m, 1),
                         "street_class_delta": int(r.street_class_delta),
                         "angle_dev": None if pd.isna(r.junction_angle_dev) else round(r.junction_angle_dev, 1),
                         "osm_url": f"https://www.openstreetmap.org/node/{int(nid)}"})
    return pd.DataFrame(rows)


def fig8(sample: pd.DataFrame, roads: pd.DataFrame) -> None:
    geoms = [(r.highway, np.array(json.loads(r.coords_json))) for r in roads.itertuples(index=False)]
    bbs = np.array([[c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()] for _, c in geoms])
    n = len(sample); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow))
    half = 60.0
    for ax, r in zip(axes.ravel(), sample.itertuples(index=False)):
        kx = np.cos(np.radians(r.lat)); dla = half / 110_540; dlo = half / (111_320 * kx)
        sel = np.where((bbs[:, 1] > r.lat - dla) & (bbs[:, 0] < r.lat + dla)
                       & (bbs[:, 3] > r.lng - dlo) & (bbs[:, 2] < r.lng + dlo))[0]
        for i in sel:
            hw, c = geoms[i]
            street = hw in STREET_ROAD_CLASSES
            ax.plot(c[:, 1], c[:, 0], color=C_BLUE if street else "#b4b3ad", lw=1.8 if street else 0.9, zorder=1)
            ax.plot(c[:, 1], c[:, 0], "o", ms=1.5, color=MUTED, zorder=2)
        ax.scatter([r.lng], [r.lat], s=70, facecolors="none", edgecolors=C_ORANGE, linewidths=1.8, zorder=3)
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((r.lng, r.lat), 40 / (111_320 * kx), 40 / 110_540, fill=False, ls=":",
                             color=C_ORANGE, lw=0.8))
        ax.set_xlim(r.lng - dlo, r.lng + dlo); ax.set_ylim(r.lat - dla, r.lat + dla); ax.set_aspect(1 / kx)
        ax.set_xticks([]); ax.set_yticks([])
        od = "—" if pd.isna(r.old_degree) else int(r.old_degree)
        ang = "" if pd.isna(r.angle_dev) else f", угол {r.angle_dev:.0f}°"
        ax.set_title(f"{r.case}\nстар. {od} | полн. {r.degree_full} | ул. {r.degree_street} | "
                     f"serv. {r.degree_service}{ang}", fontsize=7.5, loc="left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Рис. 8. Ручная проверка узлов (окно 120 × 120 м; синие — улицы, серые — service; "
                 "оранжевый круг — узел, пунктир — 20 м для азимута)", fontsize=10, x=0.01, ha="left", y=1.005)
    fig.tight_layout()
    save(fig, "fig8_manual_check_nodes", sample)


# ---------------------------------------------------------------- 4–5. AUC, Spearman
def feature_stats(features_prune: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = attach_junction_v2_features(
        attach_road_grade_features(pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET)))
    panel = panel[panel["road_length_total_m"] > 0].copy()
    panel["month"] = pd.to_datetime(panel["month"])
    cells = panel.drop_duplicates("h3").set_index("h3")
    allf = [*JUNCTION_V2_FEATURES, *features_prune]
    corr = cells[allf].rank().corr()
    corr.to_csv(REPORTS / "road_geo_v2_junctions_spearman_matrix.csv")
    test = panel[panel.month >= TRAIN_END]
    y = test[TARGET].astype(int).to_numpy()
    rows = []
    for f in [*JUNCTION_V2_FEATURES, *OLD_JUNCTION]:
        x = test[f].to_numpy(); mk = ~np.isnan(x)
        auc = roc_auc_score(y[mk], x[mk])
        col = cells[f]
        others = [g for g in features_prune if g != f and g not in OLD_JUNCTION] if f in JUNCTION_V2_FEATURES else \
            [g for g in features_prune if g != f]
        rr = corr.loc[f, others].abs()
        rows.append({"feature": f, "group": "new" if f in JUNCTION_V2_FEATURES else "old",
                     "single_auc_test": auc, "single_auc_abs": max(auc, 1 - auc),
                     "auc_rows_nonnan": int(mk.sum()),
                     "zero_share_cells": float((col == 0).mean()), "nan_share_cells": float(col.isna().mean()),
                     "max_abs_rho_other": float(rr.max()), "argmax_rho_other": rr.idxmax()})
    stats = pd.DataFrame(rows)
    stats.to_csv(REPORTS / "road_geo_v2_junctions_feature_stats.csv", index=False)
    pairs = []
    for i, a in enumerate(allf):
        for b in allf[i + 1:]:
            if a in JUNCTION_V2_FEATURES or b in JUNCTION_V2_FEATURES:
                if abs(corr.loc[a, b]) >= 0.85:
                    pairs.append({"feature_a": a, "feature_b": b, "spearman": corr.loc[a, b]})
    pd.DataFrame(pairs).to_csv(REPORTS / "road_geo_v2_junctions_spearman_pairs.csv", index=False)
    print(f"ячеек {len(cells)}, тестовых строк {len(test)}")
    print("пары |rho| >= 0.85 с участием новых признаков:\n", pd.DataFrame(pairs).to_string())
    return stats, corr


LABELS = {
    "street_junction_3leg_count": "3-сторонние (улицы)",
    "street_junction_4leg_count": "4-сторонние (улицы)",
    "street_junction_4leg_unsignalized_count": "4-сторонние без светофора",
    "major_x_street_unsignalized_count": "магистраль × улица без светофора",
    "signalized_junction_count": "перекрёстки со светофором",
    "driveway_count": "въезды service → улица",
    "street_class_delta_max": "макс. разница рангов (улицы)",
    "junction_angle_dev_mean": "отклонение угла от 90° (сред.)",
    "junction_count": "junction_count (стар.)",
    "junction_degree_mean": "junction_degree_mean (стар.)",
    "three_way_junction_count": "three_way_junction_count (стар.)",
    "four_plus_way_junction_count": "four_plus_way_junction_count (стар.)",
    "major_junction_count": "major_junction_count (стар.)",
    "class_delta_max": "class_delta_max (стар.)",
    "unsignalized_high_delta_junction_count": "unsignalized_high_delta (стар.)",
    "traffic_signal_count": "traffic_signal_count (стар.)",
}


def fig4(stats: pd.DataFrame) -> None:
    d = stats.sort_values("single_auc_abs")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = [C_BLUE if g == "new" else C_ORANGE for g in d.group]
    ax.barh(range(len(d)), d.single_auc_abs - 0.5, left=0.5, color=colors, height=0.7)
    for i, (v, z, raw) in enumerate(zip(d.single_auc_abs, d.zero_share_cells, d.single_auc_test)):
        sign = "" if raw >= 0.5 else " (−)"
        ax.text(v + 0.004, i, f"{v:.3f}{sign}  нулей {z:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(d)), [LABELS[f] for f in d.feature], fontsize=8.5)
    ax.set_xlim(0.5, 0.92); ax.set_xlabel("одиночный ROC-AUC на тесте (2023–2025), max(AUC, 1 − AUC)")
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], color=C_BLUE, lw=6, label="новые признаки"),
                       Line2D([], [], color=C_ORANGE, lw=6, label="старые признаки (prune)")],
              frameon=False, loc="lower right")
    ax.set_title("Рис. 4. Одиночный AUC признаков перекрёстков: новые и заменяемые старые\n"
                 "(«(−)» — связь обратная; угол — только по ячейкам с уличными перекрёстками)",
                 fontsize=10.5, loc="left")
    save(fig, "fig4_single_auc_old_new", stats)


def _heat(ax, sub: pd.DataFrame, ylabels: bool) -> None:
    ax.imshow(sub.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else INK, fontweight="bold" if abs(v) >= 0.85 else "normal")
            if abs(v) >= 0.85 and sub.index[i] != sub.columns[j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    ax.set_yticks(range(sub.shape[0]), [LABELS[f] for f in sub.index] if ylabels else [], fontsize=8.5)
    ax.set_xticks(range(sub.shape[1]), [LABELS.get(c, c) if c in JUNCTION_V2_FEATURES else c for c in sub.columns],
                  rotation=40, ha="right", fontsize=8)
    ax.spines[:].set_visible(False); ax.tick_params(length=0)


def fig5(corr: pd.DataFrame, features_prune: list[str]) -> None:
    key = [*OLD_JUNCTION, "road_length_total_m", "asphalt_surface_length_m", "road_share_service",
           "road_share_residential", "road_share_primary", "road_share_secondary", "crossing_count",
           "dead_end_count", "road_orientation_entropy", "road_segment_length_median"]
    key = [k for k in key if k in features_prune]
    inner = corr.loc[JUNCTION_V2_FEATURES, JUNCTION_V2_FEATURES]
    sub = corr.loc[JUNCTION_V2_FEATURES, key]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [len(inner), len(key)]})
    _heat(axes[0], inner, True); _heat(axes[1], sub, False)
    axes[1].axvline(len(OLD_JUNCTION) - 0.5, color=INK, lw=1)
    axes[0].set_title("а) новые признаки между собой", fontsize=10, loc="left")
    axes[1].set_title("б) с заменяемыми старыми (слева от черты) и ключевыми признаками набора prune",
                      fontsize=10, loc="left")
    fig.suptitle("Рис. 5. Spearman новых признаков перекрёстков; 4 296 ячеек с дорогами, рамка — |ρ| ≥ 0.85",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig5_spearman_new_vs_set", pd.concat([inner, sub], axis=1), index=True)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(OSM_ROAD_GRAPH_NODES_PARQUET)
    new = pd.read_parquet(OSM_JUNCTION_V2_NODES_PARQUET)
    roads = pd.read_parquet(OSM_ROADS_RAW_PARQUET, columns=["osm_id", "highway", "coords_json"])
    m = transitions(old, new)
    fig1(m)
    d2 = fig2(old, new, m)
    print(d2.to_string(index=False))
    fig3(roads, old, new)
    sample = manual_sample(new, m)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        print(sample.drop(columns="osm_url").to_string(index=False))
    fig8(sample, roads)
    panel = attach_road_grade_features(pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET))
    features_prune = build_features(panel[panel.road_length_total_m > 0], FEATURE_SET_ROAD_GEO_V2_PRUNE)
    stats, corr = feature_stats(features_prune)
    with pd.option_context("display.width", 250):
        print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fig4(stats)
    fig5(corr, features_prune)


if __name__ == "__main__":
    main()
