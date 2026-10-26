"""Валидация этапа 5 (СКДФ) и фигуры 1–6, 9 -> docs/analysis/features/stage5_skdf/figures/.

1. покрытие СКДФ по ячейкам и доля длины OSM по классам, покрытая паспортом;
2. ширина проезжей части и ширина на полосу по категориям СП и классам OSM;
3. полосность OSM против СКДФ: рассеяние по участкам + 3 разобранных примера (+ CSV 28 худших);
4. покрытие (дорожная одежда) СКДФ против surface OSM: матрица согласия по длине и по ячейкам;
5. категория улицы по СП против highway=*: сопряжённость по длине;
6. одиночный AUC и доля NaN новых признаков;
9. Spearman внутри кандидатов СКДФ и с признаками набора (дедупликация).
Таблицы: reports/road_geo_v2_skdf_{single_auc,spearman_pairs,lanes_by_type}.csv.

Сопоставление СКДФ ↔ OSM: точки через 10 м, ближайший параллельный (±25°) way не дальше 20 м
(skdf_v2.match_points_to_osm). Запуск: PYTHONPATH=src venv/bin/python scripts/validate_skdf_v2.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import h3  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from plot_road_geo_v2_prune import DIVERGING, GRID, INK, MUTED  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from road_accident_risk.catboost_baseline import (  # noqa: E402
    attach_geometry_v2_features, attach_junction_v2_features, attach_road_grade_features, attach_skdf_v2_features,
    attach_tags_v2_features, build_road_geo_v2_geometry_features,
)
from road_accident_risk.logreg_baseline import TARGET, TRAIN_END  # noqa: E402
from road_accident_risk.osm_road_features import ASPHALT_SURFACES, BAD_SURFACES, MOTOR_ROAD_CLASSES, road_tags_v2_for_way  # noqa: E402
from road_accident_risk.paths import (  # noqa: E402
    H3_MONTH_ROAD_FEATURES_PARQUET, OSM_ROADS_RAW_PARQUET, PROJECT_ROOT, SKDF_TRAFFIC_GEOJSON,
)
from road_accident_risk.skdf_v2 import (  # noqa: E402
    SKDF_V2_CANDIDATE_FEATURES, SKDF_V2_DERIVED_FEATURES, SP_NAME, load_skdf_segments, match_points_to_osm,
    sample_lines,
)

REPORTS = PROJECT_ROOT / "reports"
FIG = PROJECT_ROOT / "docs" / "analysis" / "features" / "stage5_skdf" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#d9d8d3"
SEQ = LinearSegmentedColormap.from_list("blues", ["#dbe9f9", "#9ec5f0", "#2a78d6", "#0d366b"])
BBOX = (51.40, 45.80, 51.70, 46.20)
HW_ORDER = ["trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "service", "link"]
SP_ORDER = ["мгд", "муоз", "мурз", "местн", "проезд", "нет категории"]
SP_LABEL = {**{k: f"{k}: {v}" for k, v in SP_NAME.items()}, "нет категории": "нет категории СП (загородные рег./фед.)"}
COAT_ORDER = ["капитальная", "облегчённая", "переходная", "низшая"]
OSM_SURF_ORDER = ["асфальт/бетон", "прочее твёрдое", "плохое (грунт, гравий, брусчатка)", "нет тега surface"]


def save(fig, name: str, data: pd.DataFrame) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    data.to_csv(FIG / f"{name}.csv", index=False)
    plt.close(fig)
    print(f"  {name}.png / .csv", flush=True)


def hw_group(h: str) -> str:
    return "link" if h.endswith("_link") else h


def osm_surface_class(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return OSM_SURF_ORDER[3]
    if s in ASPHALT_SURFACES:
        return OSM_SURF_ORDER[0]
    if s in BAD_SURFACES:
        return OSM_SURF_ORDER[2]
    return OSM_SURF_ORDER[1]


def hexmap(ax, cells: list[str], values: np.ndarray, cmap, norm):
    polys = [[(lng, lat) for lat, lng in h3.cell_to_boundary(c)] for c in cells]
    pc = PolyCollection(polys, array=values, cmap=cmap, norm=norm, edgecolor="none")
    ax.add_collection(pc)
    lats = [p[1] for poly in polys for p in poly]; lngs = [p[0] for poly in polys for p in poly]
    ax.set_xlim(min(lngs), max(lngs)); ax.set_ylim(min(lats), max(lats))
    ax.set_aspect(1 / np.cos(np.radians(51.55))); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return pc


def load_panel():
    p = attach_skdf_v2_features(attach_geometry_v2_features(attach_tags_v2_features(attach_junction_v2_features(
        attach_road_grade_features(pd.read_parquet(H3_MONTH_ROAD_FEATURES_PARQUET))))))
    p = p[p["road_length_total_m"] > 0].copy()
    base = build_road_geo_v2_geometry_features(list(p.columns))
    p["month"] = pd.to_datetime(p["month"])
    return p, base


def main() -> None:
    t0 = time.time()
    panel, base = load_panel()
    cell = panel.drop_duplicates("h3").set_index("h3")
    te = panel[panel.month >= TRAIN_END]
    y = te[TARGET].astype(int).to_numpy()
    seg = load_skdf_segments(SKDF_TRAFFIC_GEOJSON)
    segi = seg.set_index("skdf_id")
    roads = pd.read_parquet(OSM_ROADS_RAW_PARQUET, columns=["osm_id", "highway", "coords_json", "tags_json", "name"])
    roads = roads[roads.highway.isin(MOTOR_ROAD_CLASSES)].reset_index(drop=True)
    roads = pd.concat([roads, pd.DataFrame([road_tags_v2_for_way(json.loads(t)) for t in roads.tags_json])], axis=1)
    roads["coords"] = [json.loads(c) for c in roads.coords_json]
    sk = sample_lines(list(zip(seg.skdf_id, seg.coords)))
    inb = (sk.lat > BBOX[0]) & (sk.lat < BBOX[2]) & (sk.lng > BBOX[1]) & (sk.lng < BBOX[3])
    sk = sk[inb].reset_index(drop=True)
    op = sample_lines([(i, [c]) for i, c in enumerate(roads.coords)])
    m = match_points_to_osm(sk, op)
    op_in = op[(op.lat > BBOX[0]) & (op.lat < BBOX[2]) & (op.lng > BBOX[1]) & (op.lng < BBOX[3])].reset_index(drop=True)
    rev = match_points_to_osm(op_in, sk)  # точки OSM -> ближайший участок СКДФ
    print(f"точек СКДФ {len(sk):,} (сопоставлено {np.mean(m.osm_idx >= 0):.1%}); точек OSM {len(op_in):,} "
          f"(покрыто СКДФ {np.mean(rev.osm_idx >= 0):.1%}); {time.time() - t0:.0f} с", flush=True)

    m = m[m.osm_idx >= 0].copy()
    oi = m.osm_idx.to_numpy()
    m["hw"] = [hw_group(h) for h in roads.highway.to_numpy()[oi]]
    m["osm_lanes"] = roads.lanes_v2.to_numpy()[oi]; m["osm_oneway"] = roads.oneway_v2.to_numpy()[oi]
    m["osm_surf"] = [osm_surface_class(s) for s in roads.surface_v2.to_numpy()[oi]]
    nonserv = roads.highway.to_numpy() != "service"; ow = roads.oneway_v2.to_numpy().astype(bool)
    m["dual"] = [sum(1 for i in p if nonserv[i] and ow[i]) >= 2 for p in m.parallel_ids]
    for c in ["sp_class", "coating_class", "lanes", "roadway_width", "width_per_lane"]:
        m[c] = segi[c].reindex(m.id).to_numpy()
    m["sp_class"] = m["sp_class"].fillna("нет категории")

    # ---------------- 1. покрытие ----------------
    rev["hw"] = [hw_group(h) for h in roads.highway.to_numpy()[rev.id.to_numpy()]]
    rev["covered"] = rev.osm_idx >= 0
    cov = rev.groupby("hw").apply(lambda d: pd.Series({"km": d.w.sum() / 1000,
                                                     "covered_share": d.w[d.covered].sum() / d.w.sum()}),
                                  include_groups=False).reindex(HW_ORDER).dropna()
    cells = list(cell.index)
    has = cell.skdf_len_m.notna().to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.45, 1]})
    hexmap(a1, [c for c, h in zip(cells, has) if not h], np.zeros((~has).sum()),
           LinearSegmentedColormap.from_list("g", [LIGHT, LIGHT]), Normalize(0, 1))
    cs = cell.skdf_coverage_share[has].clip(upper=1.5).fillna(1.5).to_numpy()
    pc = hexmap(a1, [c for c, h in zip(cells, has) if h], cs, SEQ, Normalize(0, 1.5))
    hexmap(a1, cells, np.zeros(len(cells)), SEQ, Normalize(0, 1)).set_alpha(0)  # общий охват осей
    cb = fig.colorbar(pc, ax=a1, shrink=0.65, pad=0.01)
    cb.set_label("длина СКДФ / уличная длина OSM (обрезано на 1.5)", fontsize=8.5)
    a1.set_title(f"Ячейки панели: с паспортом СКДФ {has.sum():,} из {len(has):,} ({has.mean():.0%});\n"
                 f"серые — паспортизованных дорог нет (NaN признаков СКДФ)", loc="left", fontsize=10, color=INK)
    lab = {"link": "съезды (*_link)", "living_street": "living_street"}
    yy = np.arange(len(cov))
    a2.barh(yy, cov.covered_share, color=BLUE, height=0.62)
    a2.set_yticks(yy, [f"{lab.get(h, h)}  ({cov.km[h]:.0f} км)" for h in cov.index], fontsize=9)
    a2.invert_yaxis(); a2.set_xlim(0, 1.05); a2.xaxis.grid(True, color=GRID, lw=0.6); a2.set_axisbelow(True)
    for i, v in enumerate(cov.covered_share):
        a2.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8.5, color=INK)
    a2.set_xlabel("доля длины OSM, у которой в 20 м есть параллельный участок СКДФ")
    a2.set_title("Покрытие сети OSM паспортом по классам highway", loc="left", fontsize=10, color=INK)
    fig.suptitle("Рис. 1. Покрытие ФГИС СКДФ: паспортизована магистральная и уличная сеть, "
                 "внутриквартальные проезды (service) — почти нет", x=0.01, ha="left", fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "fig1_skdf_coverage", cov.reset_index().assign(cells_with_skdf=int(has.sum()), cells_total=len(has)))

    # ---------------- 2. ширина ----------------
    segw = m.groupby("id").agg(hw=("hw", lambda s: s.mode().iat[0]), L=("w", "sum")).join(
        segi[["sp_class", "roadway_width", "width_per_lane", "lanes"]])
    segw["sp_class"] = segw["sp_class"].fillna("нет категории")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharey="row")
    rows2 = []
    for r_, (col, ylab) in enumerate([("roadway_width", "ширина проезжей части, м"),
                                      ("width_per_lane", "ширина на полосу, м")]):
        for c_, (grp, order, labf) in enumerate([("sp_class", SP_ORDER, lambda k: k),
                                                 ("hw", [h for h in HW_ORDER if h != "living_street"], lambda k: k)]):
            ax = axes[r_, c_]
            data = [segw.loc[(segw[grp] == k) & segw[col].notna(), col].to_numpy() for k in order]
            keep = [i for i, d in enumerate(data) if len(d) >= 5]
            bp = ax.boxplot([data[i] for i in keep], widths=0.55, patch_artist=True, showfliers=False,
                            medianprops={"color": INK, "lw": 1.4}, whiskerprops={"color": MUTED},
                            capprops={"color": MUTED}, boxprops={"edgecolor": MUTED})
            for b in bp["boxes"]:
                b.set_facecolor("#9ec5f0" if grp == "sp_class" else "#d9d8d3")
            ax.set_xticks(range(1, len(keep) + 1), [f"{labf(order[i])}\nn={len(data[i])}" for i in keep], fontsize=8)
            ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
            if c_ == 0:
                ax.set_ylabel(ylab)
            if col == "width_per_lane":
                ax.axhspan(3.0, 3.75, color=GREEN, alpha=0.12, lw=0)
                ax.text(0.6, 3.38, "норма 3.0–3.75 м", fontsize=7.5, color=GREEN, va="center")
            ax.set_title(("по категории улицы по СП" if grp == "sp_class" else "по классу OSM highway (сопоставленный way)")
                         + f" — {ylab.split(',')[0]}", loc="left", fontsize=9.5, color=INK)
            for i in keep:
                d = data[i]
                rows2.append({"metric": col, "group_by": grp, "group": order[i], "n": len(d),
                              "median": np.median(d), "p25": np.percentile(d, 25), "p75": np.percentile(d, 75)})
    fig.suptitle("Рис. 2. Ширина проезжей части (K₂) и ширина на полосу по паспорту СКДФ (участки, ящики — p25–p75, "
                 "усы — 1.5 IQR)\nмгд — магистр. городские дороги, муоз — магистр. улицы общегородского, "
                 "мурз — районного значения, местн — улицы местного значения, проезд — коды 20439xx",
                 x=0.01, ha="left", fontsize=10.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_width_by_category", pd.DataFrame(rows2))

    # ---------------- 3. полосность ----------------
    g = m.groupby("id").agg(L=("w", "sum"), osm_lanes=("osm_lanes", "median"), oneway=("osm_oneway", "mean"),
                            dual=("dual", "mean"), osm_way=("osm_idx", lambda s: s.mode().iat[0]),
                            lat=("lat", lambda s: s.iloc[len(s) // 2]), lng=("lng", lambda s: s.iloc[len(s) // 2]))
    g = g.join(segi[["lanes", "roadway_width", "road_name"]])
    g["hw"] = roads.highway.to_numpy()[g.osm_way]; g["osm_id"] = roads.osm_id.to_numpy()[g.osm_way]
    q = g[g.osm_lanes.notna() & g.lanes.notna() & (g.L >= 50)].copy()
    q["type"] = np.where(q.dual > 0.5, "две проезжие части в OSM", np.where(q.oneway > 0.5, "oneway, одна линия",
                                                                          "двусторонний way"))
    q["ratio"] = q.lanes / q.osm_lanes
    q["osm_harm"] = np.where(q.dual > 0.5, 2 * q.osm_lanes, q.osm_lanes)
    bytype = q.groupby("type").agg(segments=("L", "size"), km=("L", lambda s: s.sum() / 1000),
                                   agree=("ratio", lambda s: (s == 1).mean()), ratio_median=("ratio", "median"),
                                   skdf_eq_2x_osm=("ratio", lambda s: (s == 2).mean()))
    bytype.loc["все"] = [len(q), q.L.sum() / 1000, (q.ratio == 1).mean(), q.ratio.median(), (q.ratio == 2).mean()]
    bytype["agree_harmonized"] = list(q.groupby("type").apply(lambda d: (d.lanes == d.osm_harm).mean(),
                                                              include_groups=False)) + [(q.lanes == q.osm_harm).mean()]
    rho_raw = spearmanr(q.lanes, q.osm_lanes)[0]; rho_h = spearmanr(q.lanes, q.osm_harm)[0]
    both = cell.lanes_mean_weighted_v2.notna() & cell.skdf_lanes_mean_w.notna()
    rho_cell = spearmanr(cell.lanes_mean_weighted_v2[both], cell.skdf_lanes_mean_w[both])[0]
    bytype.reset_index().to_csv(REPORTS / "road_geo_v2_skdf_lanes_by_type.csv", index=False)
    print(bytype.round(3).to_string()); print(f"ρ участки: сырой {rho_raw:.3f}, гармонизированный {rho_h:.3f}; "
                                              f"ρ ячейки ({int(both.sum())}): {rho_cell:.3f}")
    q["absdiff"] = (q.lanes - q.osm_lanes).abs()
    q.sort_values(["absdiff", "L"], ascending=False).head(28)[
        ["road_name", "hw", "type", "lanes", "osm_lanes", "roadway_width", "L", "osm_id", "lat", "lng"]].reset_index() \
        .to_csv(FIG / "fig3_lanes_osm_vs_skdf_top28.csv", index=False)

    tw = q[q.type == "двусторонний way"]; du = q[q.type == "две проезжие части в OSM"]
    examples = [
        ("две проезжие части", du[(du.ratio == 2)].sort_values("L", ascending=False).index[0]),
        ("узкая местная улица", tw[(tw.lanes == 1) & (tw.osm_lanes == 2)].sort_values("L", ascending=False).index[0]),
        ("широкая улица, СКДФ занижает", tw[(tw.lanes == 2) & (tw.osm_lanes == 4) & (tw.roadway_width >= 12)].sort_values("L", ascending=False).index[0]),
    ]
    fig = plt.figure(figsize=(16, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :2])
    rng = np.random.default_rng(0)
    for t, col in [("двусторонний way", MUTED), ("oneway, одна линия", GREEN), ("две проезжие части в OSM", ORANGE)]:
        d = q[q.type == t]
        ax.scatter(d.osm_lanes + rng.uniform(-0.22, 0.22, len(d)), d.lanes + rng.uniform(-0.22, 0.22, len(d)),
                   s=np.clip(d.L / 40, 6, 90), color=col, alpha=0.55, edgecolor="white", lw=0.4,
                   label=f"{t}: {len(d)} уч., совпадение {(d.ratio == 1).mean():.0%}")
    xs = np.array([0.5, 6.5])
    ax.plot(xs, xs, color=INK, lw=1, ls="--"); ax.text(6.0, 6.25, "СКДФ = OSM", fontsize=8, color=INK)
    ax.plot(xs, 2 * xs, color=ORANGE, lw=1, ls=":"); ax.text(2.6, 6.1, "СКДФ = 2 × OSM", fontsize=8, color=ORANGE)
    ax.set_xlim(0.5, 6.5); ax.set_ylim(0.5, 6.6)
    ax.set_xlabel("полосность OSM (`lanes` сопоставленного way)"); ax.set_ylabel("полосность СКДФ (паспорт участка)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", title="тип сечения в OSM; размер — длина", title_fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(f"Участки СКДФ с полосностью в обоих источниках ({len(q)}, {q.L.sum() / 1000:.0f} км); "
                 f"ρ = {rho_raw:.2f}, после удвоения OSM на двух проезжих частях {rho_h:.2f}; "
                 f"по ячейкам ρ = {rho_cell:.2f}", loc="left", fontsize=9.5, color=INK)
    axt = fig.add_subplot(gs[0, 2]); axt.axis("off")
    tb = bytype.copy()
    txt = "\n".join([f"{i}:\n  {int(r.segments)} уч., {r.km:.0f} км; совпадение {r.agree:.0%},\n"
                     f"  СКДФ/OSM медиана {r.ratio_median:.1f}, доля «×2» {r.skdf_eq_2x_osm:.0%}"
                     for i, r in tb.iterrows()])
    axt.text(0, 1, "Почему не совпадает\n\n" + txt + "\n\nВывод: OSM ставит lanes на каждую проезжую часть,\n"
             "СКДФ — суммарно на дорогу; на узких улицах СКДФ\nсчитает 1 полосу по ширине, OSM — 2 по направлениям.",
             va="top", fontsize=8.6, color=INK, family="DejaVu Sans")
    for k, (title, sid) in enumerate(examples):
        ax = fig.add_subplot(gs[1, k])
        r0 = q.loc[sid]; lat0, lng0 = r0.lat, r0.lng
        half = 0.0028
        win = lambda la, ln: (abs(la - lat0) < half) & (abs(ln - lng0) < half * 1.6)  # noqa: E731
        for i, cs in enumerate(roads.coords):
            a = np.asarray(cs)
            if not win(a[:, 0], a[:, 1]).any():
                continue
            hwi = roads.highway.iat[i]
            ax.plot(a[:, 1], a[:, 0], color="#b4b3ad" if hwi == "service" else MUTED,
                    lw=0.6 if hwi == "service" else 1.4, zorder=2)
            if hwi != "service" and roads.lanes_v2.iat[i] == roads.lanes_v2.iat[i]:
                inw = np.nonzero(win(a[:, 0], a[:, 1]))[0]
                j = inw[len(inw) // 2]
                ax.text(a[j, 1], a[j, 0], f"OSM {int(roads.lanes_v2.iat[i])}" + (" →" if ow[i] else ""),
                        fontsize=7.5, color=INK, zorder=5,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": MUTED, "lw": 0.5})
        for line in segi.loc[sid, "coords"]:
            a = np.asarray(line)
            ax.plot(a[:, 1], a[:, 0], color=BLUE, lw=5, alpha=0.35, zorder=1, solid_capstyle="round")
        w_ = r0.roadway_width
        ax.text(0.02, 0.03, f"СКДФ: {int(r0.lanes)} пол., ширина {w_:.1f} м" if w_ == w_ else f"СКДФ: {int(r0.lanes)} пол.",
                transform=ax.transAxes, fontsize=8.5, color=BLUE, bbox={"fc": "white", "ec": BLUE, "lw": 0.6})
        ax.set_xlim(lng0 - half * 1.6, lng0 + half * 1.6); ax.set_ylim(lat0 - half, lat0 + half)
        ax.set_aspect(1 / np.cos(np.radians(lat0))); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{chr(1072 + k)}) {title}:\n{str(r0.road_name)[:55]}", loc="left", fontsize=9, color=INK)
    fig.suptitle("Рис. 3. Полосность OSM против СКДФ: синяя полоса — осевая участка СКДФ, серые линии — ways OSM "
                 "(тонкие — service; «→» — oneway)", x=0.01, ha="left", fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "fig3_lanes_osm_vs_skdf", q.reset_index()[["id", "road_name", "hw", "type", "lanes", "osm_lanes",
                                                          "osm_harm", "roadway_width", "L", "lat", "lng"]])

    # ---------------- 4. покрытие (одежда) ----------------
    cm = m[m.coating_class.notna()].pivot_table(index="coating_class", columns="osm_surf", values="w", aggfunc="sum",
                                                fill_value=0).reindex(index=COAT_ORDER, columns=OSM_SURF_ORDER).fillna(0) / 1000
    tagged = cm[OSM_SURF_ORDER[:3]]
    both = cell.skdf_low_coating_share.notna() & cell.bad_surface_share_among_tagged.notna()
    cs_ = cell[both]
    rho_c = spearmanr(cs_.skdf_low_coating_share, cs_.bad_surface_share_among_tagged)[0]
    bins = lambda s: pd.cut(s, [-0.01, 0.0001, 0.5, 1.01], labels=["0", "(0; 0.5]", "> 0.5"])  # noqa: E731
    cellmat = pd.crosstab(bins(cs_.skdf_low_coating_share), bins(cs_.bad_surface_share_among_tagged))
    only_sk = int((cell.skdf_low_coating_share.notna() & cell.bad_surface_share_among_tagged.isna()).sum())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    rowp = cm.div(cm.sum(axis=1), axis=0)
    a1.imshow(rowp.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = rowp.iat[i, j]
            a1.text(j, i, f"{cm.iat[i, j]:.0f} км\n{v:.0%}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.55 else INK)
    a1.set_xticks(range(4), [s.replace(" (", "\n(") for s in OSM_SURF_ORDER], fontsize=8.5)
    a1.set_yticks(range(4), [f"{c} ({cm.loc[c].sum():.0f} км)" for c in COAT_ORDER], fontsize=9)
    a1.set_xlabel("тег surface сопоставленного way OSM"); a1.set_ylabel("тип дорожной одежды по паспорту СКДФ")
    low_tagged = tagged.loc[["переходная", "низшая"]].sum()
    a1.set_title(f"Участки (по длине, доля по строке). Тег surface есть у {tagged.to_numpy().sum() / cm.to_numpy().sum():.0%} "
                 f"паспортизованной длины;\nиз низшей/переходной с тегом OSM «плохое» — "
                 f"{low_tagged.iloc[2] / low_tagged.sum():.0%}, «асфальт» — {low_tagged.iloc[0] / low_tagged.sum():.0%}",
                 loc="left", fontsize=9.5, color=INK)
    a2.imshow(cellmat.to_numpy(), cmap=SEQ, aspect="auto")
    for i in range(cellmat.shape[0]):
        for j in range(cellmat.shape[1]):
            v = cellmat.iat[i, j]
            a2.text(j, i, f"{v}", ha="center", va="center", fontsize=10,
                    color="white" if v > cellmat.to_numpy().max() * 0.55 else INK)
    a2.set_xticks(range(cellmat.shape[1]), list(cellmat.columns), fontsize=9)
    a2.set_yticks(range(cellmat.shape[0]), list(cellmat.index), fontsize=9)
    a2.set_xlabel("bad_surface_share_among_tagged (OSM)"); a2.set_ylabel("skdf_low_coating_share (СКДФ)")
    a2.set_title(f"Ячейки, где определены оба ({int(both.sum())}); ρ = {rho_c:.2f}.\n"
                 f"Ещё {only_sk} ячеек: СКДФ есть, OSM-доля — NaN (< 100 м размечено)", loc="left", fontsize=9.5, color=INK)
    fig.suptitle("Рис. 4. Покрытие: паспортный тип дорожной одежды СКДФ против тега surface OSM",
                 x=0.01, ha="left", fontsize=11, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_coating_agreement", cm.reset_index().melt(id_vars="coating_class", var_name="osm_surface", value_name="km"))

    # ---------------- 5. СП против highway ----------------
    ct = m.pivot_table(index="sp_class", columns="hw", values="w", aggfunc="sum", fill_value=0) / 1000
    ct = ct.reindex(index=SP_ORDER, columns=HW_ORDER).fillna(0)
    rowp = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.imshow(rowp.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = rowp.iat[i, j]
            if ct.iat[i, j] >= 0.5:
                ax.text(j, i, f"{ct.iat[i, j]:.0f}\n{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(HW_ORDER)), [h.replace("link", "*_link") for h in HW_ORDER], fontsize=9)
    ax.set_yticks(range(len(SP_ORDER)), [f"{SP_LABEL[s]} ({ct.loc[s].sum():.0f} км)" for s in SP_ORDER], fontsize=9)
    ax.set_xlabel("класс OSM highway сопоставленного way")
    # согласие: доля длины СП-магистралей (мгд/муоз/мурз) на trunk/primary/secondary
    mag = ct.loc[["мгд", "муоз", "мурз"]]; maj = mag[["trunk", "primary", "secondary"]].to_numpy().sum() / mag.to_numpy().sum()
    loc_ = ct.loc[["местн", "проезд"]]; minor = loc_[["tertiary", "unclassified", "residential", "living_street", "service"]].to_numpy().sum() / loc_.to_numpy().sum()
    ax.set_title(f"Длина, км, и доля по строке. Магистральные по СП лежат на trunk/primary/secondary на {maj:.0%} длины; "
                 f"местные и проезды — на tertiary и ниже на {minor:.0%}", loc="left", fontsize=9.5, color=INK)
    fig.suptitle("Рис. 5. Категория городской улицы по СП 42.13330 против класса highway OSM (сопряжённость по длине)",
                 x=0.01, ha="left", fontsize=11, color=INK, y=1.01)
    fig.tight_layout()
    save(fig, "fig5_sp_vs_highway", ct.reset_index().melt(id_vars="sp_class", var_name="highway", value_name="km"))

    # ---------------- 6. одиночный AUC и NaN ----------------
    feats6 = [*SKDF_V2_CANDIDATE_FEATURES, "lanes_skdf_or_osm", "skdf_has_passport", "skdf_len_m",
              "lanes_mean_weighted_v2", "bad_surface_share_among_tagged", "major_road_length_m", "road_length_total_m"]
    rows6 = []
    for f in feats6:
        x = te[f].to_numpy(float); mk = ~np.isnan(x)
        a_nn = roc_auc_score(y[mk], x[mk])
        rows6.append({"feature": f, "source": "OSM" if f in base else ("СКДФ" if f.startswith("skdf") else "смешанный"),
                      "nan_share_cells": float(cell[f].isna().mean()), "nan_share_test_rows": float(1 - mk.mean()),
                      "auc_non_nan": a_nn, "auc_non_nan_oriented": max(a_nn, 1 - a_nn), "sign": "+" if a_nn >= 0.5 else "−",
                      "auc_nan_as_min": roc_auc_score(y, np.nan_to_num(x, nan=-1e12))})
    sa = pd.DataFrame(rows6)
    sa.to_csv(REPORTS / "road_geo_v2_skdf_single_auc.csv", index=False)
    print(sa.round(3).to_string(index=False))
    sa = sa.sort_values("auc_non_nan_oriented")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 8.4), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
    yy = np.arange(len(sa))
    col = [MUTED if s == "OSM" else (PURPLE if s == "смешанный" else BLUE) for s in sa.source]
    a1.barh(yy - 0.18, sa.auc_non_nan_oriented - 0.5, left=0.5, height=0.36, color=col, label="только ячейки без NaN")
    a1.barh(yy + 0.18, sa.auc_nan_as_min - 0.5, left=0.5, height=0.36, color=col, alpha=0.35, label="NaN как минимум")
    for i, r in enumerate(sa.itertuples()):
        a1.text(max(r.auc_non_nan_oriented, r.auc_nan_as_min) + 0.004, i, f"{r.auc_non_nan_oriented:.3f} ({r.sign})",
                va="center", fontsize=7.8, color=INK)
    a1.set_yticks(yy, sa.feature, fontsize=8.3); a1.set_xlim(0.5, 0.86)
    a1.xaxis.grid(True, color=GRID, lw=0.6); a1.set_axisbelow(True)
    a1.set_xlabel("ROC-AUC одного признака на тесте 2023–2025 (ориентированный; «−» — связь обратная)")
    a1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.35),
                       plt.Rectangle((0, 0), 1, 1, color=MUTED), plt.Rectangle((0, 0), 1, 1, color=PURPLE)],
              labels=["СКДФ, только ячейки без NaN", "СКДФ, NaN как минимум (несёт «нет паспорта»)",
                      "OSM (набор A), для сравнения", "СКДФ или OSM (lanes_skdf_or_osm)"],
              frameon=False, fontsize=8.3, loc="upper left", bbox_to_anchor=(0, -0.07), ncol=2)
    a1.set_title("Одиночный AUC", loc="left", fontsize=10, color=INK)
    a2.barh(yy, sa.nan_share_cells, color=col, height=0.6)
    for i, v in enumerate(sa.nan_share_cells):
        a2.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=7.8, color=INK)
    a2.set_xlim(0, 1); a2.xaxis.grid(True, color=GRID, lw=0.6); a2.set_axisbelow(True)
    a2.set_xlabel("доля NaN по ячейкам панели (4 296)")
    a2.set_title("Пропуски", loc="left", fontsize=10, color=INK)
    fig.suptitle("Рис. 6. Разделяющая сила и заполненность признаков СКДФ (37 % ячеек без паспорта — NaN)",
                 x=0.01, ha="left", fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "fig6_single_auc_nan", sa)

    # ---------------- 9. Spearman ----------------
    ref = ["road_length_total_m", "major_road_length_m", "lanes_mean_weighted_v2", "oneway_share_v2",
           "bad_surface_share_among_tagged", "road_share_trunk", "road_share_primary", "road_share_secondary",
           "road_share_residential", "road_share_service", "crossing_count"]
    sk_feats = [*SKDF_V2_CANDIDATE_FEATURES, "lanes_skdf_or_osm"]
    C = cell[[*sk_feats, *base]].corr("spearman")
    pairs = []
    for i, a in enumerate(sk_feats):
        for b in [*sk_feats[i + 1:], *base]:
            if b == a:
                continue
            r = C.loc[a, b]
            if abs(r) >= 0.7:
                pairs.append({"a": a, "b": b, "rho": r, "b_in_set_A": b in base,
                              "class": "дубль (≥ 0.85)" if abs(r) >= 0.85 else "пограничный (0.7–0.85)"})
    pd.DataFrame(pairs).sort_values("rho", key=abs, ascending=False).to_csv(REPORTS / "road_geo_v2_skdf_spearman_pairs.csv", index=False)
    mx = C.loc[sk_feats, base].abs().max(axis=1)
    print("max |ρ| с набором A:\n" + mx.round(3).to_string())
    show = [f for f in sk_feats if not f.startswith("skdf_cat_sp_share_m") or f == "skdf_cat_sp_share_magistral"]
    M = C.loc[show, [*show, *ref]]
    fig, ax = plt.subplots(figsize=(17, 8.8))
    ax.imshow(M.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if i == j and j < len(show):
                continue
            ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=6.6,
                    color="white" if abs(v) > 0.6 else INK, fontweight="bold" if abs(v) >= 0.85 else "normal")
    ax.axvline(len(show) - 0.5, color=INK, lw=1.2)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=60, ha="right", fontsize=7.6)
    ax.set_yticks(range(M.shape[0]), M.index, fontsize=8)
    ax.text(len(show) + len(ref) / 2 - 0.5, -1.0, "признаки набора A (OSM)", ha="center", fontsize=9, color=INK)
    ax.text(len(show) / 2 - 0.5, -1.0, "кандидаты СКДФ", ha="center", fontsize=9, color=INK)
    fig.suptitle("Рис. 9. Spearman ρ по ячейкам (2 716 с паспортом): кандидаты СКДФ между собой и с набором A; "
                 "жирным — |ρ| ≥ 0.85", x=0.01, ha="left", fontsize=11, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "fig9_spearman_dedup", M.reset_index().rename(columns={"index": "feature"}))
    print(f"{time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
