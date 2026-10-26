"""Road+Geo v2, этап 4 (2026-10-16): уличный граф «перекрёсток → перекрёсток» и геометрия без service.

Новый модуль (старые функции osm_road_features.py и scripts/reconcile_group3_geometry_features.py
не меняются). Итог и определения: docs/analysis/features/stage4_geometry/README.md.

Содержимое:
- utm38n_forward — проекция WGS84 -> UTM 38N (EPSG:32638), ряды Крюгера (Karney 2011);
- build_street_graph_raw — граф уличной сети (motor-классы без service), разрезанный в узлах
  со степенью != 2 (перекрёстки этапа 2, тупики) и в стыках ways (псевдоузлы степени 2);
- consolidate_street_graph — склейка узлов-перекрёстков в радиусе tol (раздельные дороги);
- build_street_blocks — цепочки рёбер через узлы степени 2 (квартал = перекрёсток -> перекрёсток);
- curve_vertices_for_blocks — вершины упрощённой (Douglas–Peucker) линии: угол поворота, радиус;
- aggregate_h3_geometry_v2_features — признаки ячейки H3.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Iterable

import h3
import numpy as np
import pandas as pd

from road_accident_risk.osm_road_features import (
    BBOX,
    H3_RESOLUTION,
    MAJOR_ROAD_CLASSES,
    ROAD_CLASS_RANK,
    SAMPLE_STEP_M,
    STREET_ROAD_CLASSES,
    haversine_m,
    parse_oneway_v2,
)

# --- параметры (все пороги — здесь) ---------------------------------------------------------
# Ways выгружены, если хотя бы одна вершина внутри BBOX ± 0.01° (coords_touch_bbox), и целиком.
# Сеть полна внутри этого прямоугольника; узлы вне него или ближе EDGE_BUFFER_M к его границе
# помечаются at_edge (возможны недостающие подходы и ложные тупики).
DATA_BBOX_PADDING_DEG = 0.01
EDGE_BUFFER_M = 50.0
CONSOLIDATION_TOL_M = 20.0          # выбранный радиус склейки (обоснование — README этапа 4)
CONSOLIDATION_PATH_FACTOR = 2.0     # склеиваются только узлы, связанные по улицам путём <= factor·tol
ROUNDABOUT_JUNCTIONS = {"roundabout", "circular"}
ROUNDABOUT_MAX_RING_M = 400.0       # кольцо длиннее — не склеивается целиком (не «один перекрёсток»)
DP_TOL_M = 3.0                      # Douglas–Peucker
JUNCTION_CLEARANCE_M = 25.0         # вершины ближе к перекрёстку — не кривизна (план: 15 м; см. LOG §4)
MIN_LINK_M = 10.0                   # радиус — только при обоих звеньях >= 10 м
RADIUS_CAP_M = 1000.0               # R >= 1000 м — «прямая»
SHARP_RADIUS_M = 200.0              # share_length_radius_lt_200m
MIN_STREET_LENGTH_M = 200.0         # ниже — энтропия, кривизна, радиус = NaN
ORIENTATION_BINS = 8

UTM38_LON0 = 45.0

GEOMETRY_V2_MODEL_FEATURES = [
    "block_length_median_m",
    "curvature_deg_per_km_v2",
    "min_curve_radius_m",
    "road_orientation_entropy_v2",
    "street_dead_end_count",
    "level_crossing_count",
]
# Строятся, но в набор по умолчанию не входят (альтернатива / диагностика).
GEOMETRY_V2_DIAGNOSTIC_FEATURES = [
    "share_length_radius_lt_200m",
    "street_dead_end_count_raw",
    "block_count",
    "street_length_geom_m",
]
GEOMETRY_V2_FEATURES = [*GEOMETRY_V2_MODEL_FEATURES, *GEOMETRY_V2_DIAGNOSTIC_FEATURES]


# --- проекция ---------------------------------------------------------------------------------
def utm38n_forward(lat: Any, lon: Any) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 (градусы) -> UTM зона 38N (EPSG:32638), метры. Ряды Крюгера 6-го порядка (Karney 2011)."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lam = np.radians(np.asarray(lon, dtype=float) - UTM38_LON0)
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64 + n**6 / 256)
    alpha = [
        n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16 + 41 * n**4 / 180 - 127 * n**5 / 288 + 7891 * n**6 / 37800,
        13 * n**2 / 48 - 3 * n**3 / 5 + 557 * n**4 / 1440 + 281 * n**5 / 630 - 1983433 * n**6 / 1935360,
        61 * n**3 / 240 - 103 * n**4 / 140 + 15061 * n**5 / 26880 + 167603 * n**6 / 181440,
        49561 * n**4 / 161280 - 179 * n**5 / 168 + 6601661 * n**6 / 7257600,
        34729 * n**5 / 80640 - 3418889 * n**6 / 1995840,
        212378941 * n**6 / 319334400,
    ]
    e = 2 * math.sqrt(n) / (1 + n)
    t = np.sinh(np.arctanh(np.sin(lat)) - e * np.arctanh(e * np.sin(lat)))
    xi_p = np.arctan2(t, np.cos(lam))
    eta_p = np.arctanh(np.sin(lam) / np.sqrt(1 + t**2))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, aj in enumerate(alpha, start=1):
        xi = xi + aj * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta = eta + aj * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    return 500000.0 + k0 * A * eta, k0 * A * xi


def data_edge_flags(lat: np.ndarray, lng: np.ndarray, buffer_m: float = EDGE_BUFFER_M,
                    bbox: tuple[float, float, float, float] = BBOX,
                    padding_deg: float = DATA_BBOX_PADDING_DEG) -> np.ndarray:
    """True — узел вне BBOX ± padding или ближе buffer_m к его границе (сеть там может быть неполной)."""
    lat = np.asarray(lat, float); lng = np.asarray(lng, float)
    min_lat, min_lng, max_lat, max_lng = bbox
    min_lat -= padding_deg; min_lng -= padding_deg; max_lat += padding_deg; max_lng += padding_deg
    ky = 110_540.0
    kx = 111_320.0 * np.cos(np.radians(lat))
    d = np.minimum.reduce([(lat - min_lat) * ky, (max_lat - lat) * ky, (lng - min_lng) * kx, (max_lng - lng) * kx])
    return d <= buffer_m


# --- вспомогательная геометрия ----------------------------------------------------------------
def polyline_length_m(coords: list[tuple[float, float]]) -> float:
    return float(sum(haversine_m(a[0], a[1], b[0], b[1]) for a, b in zip(coords, coords[1:])))


def coords_to_wkt(coords: list[tuple[float, float]]) -> str:
    return "LINESTRING (" + ", ".join(f"{lng:.7f} {lat:.7f}" for lat, lng in coords) + ")"


def allocate_coords_to_h3(coords: list[tuple[float, float]], step_m: float = SAMPLE_STEP_M,
                          resolution: int = H3_RESOLUTION) -> dict[str, float]:
    """Длина линии по ячейкам H3 через сэмплы ~20 м (как allocate_way_to_h3 этапов 0–3)."""
    out: dict[str, float] = defaultdict(float)
    for (lat1, lng1), (lat2, lng2) in zip(coords, coords[1:]):
        seg = haversine_m(lat1, lng1, lat2, lng2)
        if seg <= 0:
            continue
        pieces = max(1, math.ceil(seg / step_m))
        piece = seg / pieces
        for i in range(pieces):
            fr = (i + 0.5) / pieces
            out[h3.latlng_to_cell(lat1 + (lat2 - lat1) * fr, lng1 + (lng2 - lng1) * fr, resolution)] += piece
    return out


def douglas_peucker(xy: np.ndarray, tol: float) -> np.ndarray:
    """Индексы вершин, оставшихся после упрощения Douglas–Peucker (концы всегда остаются)."""
    n = len(xy)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        seg = b - a
        L = math.hypot(seg[0], seg[1])
        p = xy[i + 1:j] - a
        if L == 0:
            d = np.hypot(p[:, 0], p[:, 1])
        else:
            d = np.abs(seg[0] * p[:, 1] - seg[1] * p[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m)); stack.append((m, j))
    return np.flatnonzero(keep)


# --- 4.0. сырой уличный граф ------------------------------------------------------------------
def _oneway_dir(tags: dict[str, Any]) -> int:
    """1 — движение по направлению way (u -> v), -1 — против, 0 — двустороннее."""
    if not parse_oneway_v2(tags):
        return 0
    v = str(tags.get("oneway", "")).strip().lower()
    return -1 if v in {"-1", "reverse"} else 1


def _clean_way(node_ids: list[int], coords: list[tuple[float, float]]) -> tuple[list[int], list[tuple[float, float]]]:
    """Убирает подряд идущие повторы одной вершины (нулевые отрезки)."""
    ids, cs = [], []
    for nid, c in zip(node_ids, coords):
        if ids and ids[-1] == nid:
            continue
        ids.append(nid); cs.append(c)
    return ids, cs


def build_street_graph_raw(roads: pd.DataFrame, h3_resolution: int = H3_RESOLUTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Граф уличной сети из osm_roads_raw: узлы и рёбра (кусок одного way между двумя узлами графа).

    Узел графа — вершина OSM, которая является концом street-way, общей для >= 2 street-ways или
    повторяется в одном way. Степень = число подходов по уличной сети (как degree_street этапа 2):
    junction (>= 3), dead_end (1), pseudo (2: стык ways, смена имени/класса — не перекрёсток).
    """
    ways = []
    service_deg: dict[int, int] = defaultdict(int)
    for r in roads[["osm_id", "highway", "name", "nodes_json", "coords_json", "tags_json"]].itertuples(index=False):
        hw = str(r.highway)
        ids = [int(x) for x in json.loads(r.nodes_json)]
        cs = [(float(c[0]), float(c[1])) for c in json.loads(r.coords_json)]
        if len(ids) != len(cs):
            continue
        ids, cs = _clean_way(ids, cs)
        if len(ids) < 2:
            continue
        if hw == "service":
            for i, nid in enumerate(ids):
                service_deg[nid] += int(i > 0) + int(i < len(ids) - 1)
            continue
        if hw not in STREET_ROAD_CLASSES:
            continue
        tags = json.loads(r.tags_json) if isinstance(r.tags_json, str) else {}
        rb = str(tags.get("junction", "")).strip().lower() in ROUNDABOUT_JUNCTIONS
        ways.append((int(r.osm_id), hw, r.name if isinstance(r.name, str) else None, _oneway_dir(tags), rb, ids, cs))

    degree: dict[int, int] = defaultdict(int)
    usage: dict[int, int] = defaultdict(int)
    coord: dict[int, tuple[float, float]] = {}
    is_end: set[int] = set()
    for _, _, _, _, _, ids, cs in ways:
        seen: set[int] = set()
        for i, nid in enumerate(ids):
            degree[nid] += int(i > 0) + int(i < len(ids) - 1)
            coord[nid] = cs[i]
            if nid in seen:
                usage[nid] += 1  # повтор в одном way (кольцо, петля)
            seen.add(nid)
        for nid in seen:
            usage[nid] += 1
        is_end.add(ids[0]); is_end.add(ids[-1])
    vertex = {nid for nid in degree if nid in is_end or usage[nid] >= 2}

    edges = []
    for osm_id, hw, name, odir, rb, ids, cs in ways:
        cut = [i for i, nid in enumerate(ids) if nid in vertex]
        for a, b in zip(cut, cut[1:]):
            seg = cs[a:b + 1]
            edges.append({
                "u": ids[a], "v": ids[b], "osm_id": osm_id, "highway": hw, "name": name,
                "oneway": odir != 0, "oneway_dir": odir, "is_roundabout": rb, "length_m": polyline_length_m(seg),
                "n_vertices": len(seg), "coords_json": json.dumps(seg),
            })
    e = pd.DataFrame(edges)
    e.insert(0, "edge_id", np.arange(len(e), dtype=np.int64))

    nid = np.array(sorted(vertex), dtype=np.int64)
    lat = np.array([coord[i][0] for i in nid]); lng = np.array([coord[i][1] for i in nid])
    x, y = utm38n_forward(lat, lng)
    deg = np.array([degree[i] for i in nid])
    n = pd.DataFrame({
        "node_id": nid, "lat": lat, "lng": lng, "x_utm": x, "y_utm": y, "degree": deg,
        "node_type": np.select([deg >= 3, deg == 1], ["junction", "dead_end"], "pseudo"),
        "degree_service": [service_deg.get(i, 0) for i in nid],
        "at_edge": data_edge_flags(lat, lng),
        "h3": [h3.latlng_to_cell(a, b, h3_resolution) for a, b in zip(lat, lng)],
    })
    # степень по рёбрам должна совпасть со счётом подходов
    ends = pd.concat([e["u"], e["v"]]).value_counts()
    chk = n.set_index("node_id")["degree"]
    bad = (ends.reindex(chk.index).fillna(0).astype(int) != chk).sum()
    if bad:
        raise RuntimeError(f"build_street_graph_raw: у {bad} узлов степень != числу концов рёбер")
    e["at_edge"] = e["u"].map(n.set_index("node_id")["at_edge"]) | e["v"].map(n.set_index("node_id")["at_edge"])
    e["wkt"] = [coords_to_wkt(json.loads(c)) for c in e["coords_json"]]
    return n, e


# --- 4.0. склейка дублей ---------------------------------------------------------------------
class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def junction_clusters(nodes: pd.DataFrame, edges: pd.DataFrame, tol_m: float,
                      path_factor: float = CONSOLIDATION_PATH_FACTOR, roundabouts: bool = True) -> pd.Series:
    """Кластер для каждого узла-перекрёстка: пары ближе tol_m (UTM), связанные по уличной сети
    путём <= path_factor·tol_m, объединяются (single linkage). roundabouts=True — дополнительно все
    узлы одного кольца (junction=roundabout/circular, длина кольца <= 400 м) — один кластер. Возвращает Series node_id -> cluster
    (cluster = минимальный node_id кластера; у прочих узлов — свой node_id)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    idx = pd.Series(np.arange(len(nodes)), index=nodes["node_id"].to_numpy())
    jmask = (nodes["node_type"] == "junction").to_numpy()
    jpos = np.flatnonzero(jmask)
    xy = nodes[["x_utm", "y_utm"]].to_numpy()[jpos]
    pairs = cKDTree(xy).query_pairs(tol_m, output_type="ndarray")
    dsu = _DSU(len(nodes))
    if len(pairs):
        u = idx.loc[edges["u"]].to_numpy(); v = idx.loc[edges["v"]].to_numpy()
        w = edges["length_m"].to_numpy() + 1e-6
        g = coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(len(nodes),) * 2).tocsr()
        src = np.unique(jpos[pairs[:, 0]])
        limit = path_factor * tol_m
        dist_rows: dict[int, np.ndarray] = {}
        for k in range(0, len(src), 300):
            batch = src[k:k + 300]
            dm = dijkstra(g, directed=False, indices=batch, limit=limit)
            for s, row in zip(batch, dm):
                dist_rows[int(s)] = row
        for a, b in pairs:
            pa, pb = int(jpos[a]), int(jpos[b])
            if dist_rows[pa][pb] <= limit:
                dsu.union(pa, pb)
    if roundabouts and "is_roundabout" in edges.columns:
        ring = edges[edges["is_roundabout"]]
        rdsu = _DSU(len(nodes))
        for a, b in zip(idx.loc[ring["u"]].to_numpy(), idx.loc[ring["v"]].to_numpy()):
            rdsu.union(int(a), int(b))
        rroot = np.array([rdsu.find(int(i)) for i in idx.loc[ring["u"]].to_numpy()])
        ring_len = pd.Series(ring["length_m"].to_numpy()).groupby(rroot).sum()
        for a, b in zip(idx.loc[ring["u"]].to_numpy(), idx.loc[ring["v"]].to_numpy()):
            if ring_len[rdsu.find(int(a))] <= ROUNDABOUT_MAX_RING_M:
                dsu.union(int(a), int(b))
    root = np.array([dsu.find(i) for i in range(len(nodes))])
    return pd.Series(nodes["node_id"].to_numpy()[root], index=nodes["node_id"].to_numpy(), name="cluster")


def consolidate_street_graph(nodes: pd.DataFrame, edges: pd.DataFrame, tol_m: float = CONSOLIDATION_TOL_M,
                             path_factor: float = CONSOLIDATION_PATH_FACTOR, roundabouts: bool = True,
                             h3_resolution: int = H3_RESOLUTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Склейка узлов-перекрёстков раздельных дорог: кластер -> один узел в центроиде.

    Рёбра внутри кластера длиной <= (path_factor + 1)·tol (перемычки между проезжими частями) и дуги
    склеенного кольца удаляются; остальные переподключаются к центроиду (центроид дописывается в геометрию концом).
    Узлы-кластеры получают отрицательные id: −(минимальный node_id кластера).
    """
    cl = junction_clusters(nodes, edges, tol_m, path_factor, roundabouts)
    n = nodes.copy()
    n["cluster"] = n["node_id"].map(cl)
    size = n.groupby("cluster")["node_id"].transform("size")
    n["new_id"] = np.where(size > 1, -n["cluster"], n["node_id"])

    g = n.groupby("new_id")
    out_n = g.agg(lat=("lat", "mean"), lng=("lng", "mean"), x_utm=("x_utm", "mean"), y_utm=("y_utm", "mean"),
                  degree_service=("degree_service", "sum"), at_edge=("at_edge", "max"),
                  n_raw_nodes=("node_id", "size")).reset_index().rename(columns={"new_id": "node_id"})
    raw_ids = g["node_id"].apply(lambda s: json.dumps(sorted(int(v) for v in s)))
    out_n["raw_node_ids"] = out_n["node_id"].map(raw_ids)
    # диаметр кластера (макс. расстояние до центроида ×2) — контроль «цепочек»
    cxy = n[["new_id", "x_utm", "y_utm"]].merge(out_n[["node_id", "x_utm", "y_utm"]], left_on="new_id",
                                                 right_on="node_id", suffixes=("", "_c"))
    cxy["r"] = np.hypot(cxy.x_utm - cxy.x_utm_c, cxy.y_utm - cxy.y_utm_c)
    out_n["cluster_radius_m"] = out_n["node_id"].map(cxy.groupby("new_id")["r"].max())

    newid = n.set_index("node_id")["new_id"]
    cpos = out_n.set_index("node_id")[["lat", "lng"]]
    e = edges.copy()
    e["raw_edge_id"] = e["edge_id"]; e["raw_u"] = e["u"]; e["raw_v"] = e["v"]
    e["u"] = e["raw_u"].map(newid); e["v"] = e["raw_v"].map(newid)
    multi = set(out_n.loc[out_n.n_raw_nodes > 1, "node_id"])
    rb = e["is_roundabout"] if "is_roundabout" in e.columns else False
    internal = (e["u"] == e["v"]) & e["u"].isin(multi) & ((e["length_m"] <= (path_factor + 1) * tol_m) | rb)
    e = e[~internal].copy()
    coords_new, lens = [], []
    for c, u, v in zip(e["coords_json"], e["u"], e["v"]):
        cs = [tuple(p) for p in json.loads(c)]
        if u in multi:
            cs = [tuple(cpos.loc[u])] + cs
        if v in multi:
            cs = cs + [tuple(cpos.loc[v])]
        coords_new.append(cs); lens.append(polyline_length_m(cs))
    e["coords_json"] = [json.dumps(c) for c in coords_new]
    e["length_m"] = lens
    e["n_vertices"] = [len(c) for c in coords_new]
    e["wkt"] = [coords_to_wkt(c) for c in coords_new]
    e = e.reset_index(drop=True)
    e["edge_id"] = np.arange(len(e), dtype=np.int64)
    key = [tuple(sorted(p)) for p in zip(e["u"], e["v"])]
    e["parallel_count"] = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy()

    deg = pd.concat([e["u"], e["v"]]).value_counts()
    out_n["degree"] = out_n["node_id"].map(deg).fillna(0).astype(int)
    out_n = out_n[out_n["degree"] > 0].copy()
    d = out_n["degree"].to_numpy()
    out_n["node_type"] = np.select([d >= 3, d == 1], ["junction", "dead_end"], "pseudo")
    out_n["h3"] = [h3.latlng_to_cell(a, b, h3_resolution) for a, b in zip(out_n.lat, out_n.lng)]
    ae = out_n.set_index("node_id")["at_edge"]
    e["at_edge"] = e["u"].map(ae) | e["v"].map(ae)
    cols = ["node_id", "lat", "lng", "x_utm", "y_utm", "degree", "node_type", "degree_service", "at_edge", "h3",
            "n_raw_nodes", "cluster_radius_m", "raw_node_ids"]
    return out_n[cols].sort_values("node_id").reset_index(drop=True), e


# --- кварталы: цепочки рёбер через узлы степени 2 ------------------------------------------------
def build_street_blocks(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Квартал (block) = максимальная цепочка рёбер, все внутренние узлы которой имеют степень 2.

    Возвращает таблицу кварталов и Series edge_id -> block_id. Концы квартала — узлы степени != 2
    (перекрёсток или тупик); цепочки только из узлов степени 2 — кольца (is_ring).
    """
    deg = nodes.set_index("node_id")["degree"]
    ntype = nodes.set_index("node_id")["node_type"]
    at_edge = nodes.set_index("node_id")["at_edge"]
    inc: dict[int, list[int]] = defaultdict(list)
    eu = edges["u"].to_numpy(); ev = edges["v"].to_numpy(); eid = edges["edge_id"].to_numpy()
    pos = {int(k): i for i, k in enumerate(eid)}
    for i, (u, v) in enumerate(zip(eu, ev)):
        inc[int(u)].append(i); inc[int(v)].append(i)
    coords = [json.loads(c) for c in edges["coords_json"]]
    lengths = edges["length_m"].to_numpy()
    hw = edges["highway"].to_numpy()
    visited = np.zeros(len(edges), bool)
    block_of = np.full(len(edges), -1, np.int64)
    blocks = []

    def walk(start_node: int, first_edge: int) -> tuple[list[int], list[list[float]], int]:
        chain, cs = [], []
        node, ei = start_node, first_edge
        while True:
            visited[ei] = True; chain.append(ei)
            c = coords[ei] if eu[ei] == node else coords[ei][::-1]
            cs = c if not cs else cs + c[1:]
            node = int(ev[ei]) if eu[ei] == node else int(eu[ei])
            if deg.get(node, 0) != 2 or node == start_node:
                return chain, cs, node
            nxt = [k for k in inc[node] if not visited[k]]
            if not nxt:
                return chain, cs, node
            ei = nxt[0]

    def add(chain: list[int], cs: list[list[float]], a: int, b: int, ring: bool) -> None:
        bid = len(blocks)
        block_of[chain] = bid
        L = float(lengths[chain].sum())
        by_cls: dict[str, float] = defaultdict(float)
        for k in chain:
            by_cls[hw[k]] += lengths[k]
        main_cls = max(by_cls, key=by_cls.get)
        blocks.append({
            "block_id": bid, "node_a": a, "node_b": b, "length_m": L, "n_edges": len(chain),
            "type_a": ntype.get(a), "type_b": ntype.get(b), "is_ring": ring,
            "at_edge": bool(at_edge.get(a, False) or at_edge.get(b, False)),
            "highway": main_cls, "is_major": main_cls in MAJOR_ROAD_CLASSES,
            "class_rank": ROAD_CLASS_RANK.get(main_cls, -1), "coords_json": json.dumps(cs),
        })

    for node in nodes.loc[nodes["degree"] != 2, "node_id"].astype(int):
        for ei in inc[node]:
            if visited[ei]:
                continue
            chain, cs, end = walk(node, ei)
            add(chain, cs, node, end, False)
    for ei in range(len(edges)):  # кольца без узлов степени != 2
        if not visited[ei]:
            node = int(eu[ei])
            chain, cs, end = walk(node, ei)
            add(chain, cs, node, end, True)
    b = pd.DataFrame(blocks)
    b["is_junction_block"] = (b["type_a"] == "junction") & (b["type_b"] == "junction") & ~b["is_ring"]
    return b, pd.Series(block_of, index=eid, name="block_id")


# --- 4.2. кривизна и радиусы ------------------------------------------------------------------
def turn_and_radius(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Для внутренних вершин полилинии: угол поворота (°, без знака), радиус окружности через три
    вершины (м), длины звеньев до и после."""
    a = xy[1:-1] - xy[:-2]; b = xy[2:] - xy[1:-1]
    l1 = np.hypot(a[:, 0], a[:, 1]); l2 = np.hypot(b[:, 0], b[:, 1])
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = (a * b).sum(1)
    turn = np.degrees(np.abs(np.arctan2(cross, dot)))
    c = xy[2:] - xy[:-2]
    l3 = np.hypot(c[:, 0], c[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        R = l1 * l2 * l3 / (2 * np.abs(cross))
    R[np.abs(cross) < 1e-9] = np.inf
    return turn, R, l1, l2


def curve_vertices_for_blocks(blocks: pd.DataFrame, junction_xy: np.ndarray, dp_tol_m: float = DP_TOL_M,
                              clearance_m: float = JUNCTION_CLEARANCE_M, min_link_m: float = MIN_LINK_M,
                              h3_resolution: int = H3_RESOLUTION) -> pd.DataFrame:
    """Вершины упрощённых линий кварталов (сырой граф) с углом поворота и радиусом.

    Учитываются только внутренние вершины упрощённой линии, отстоящие >= clearance_m по линии от
    конца-перекрёстка и >= clearance_m по прямой от любого узла-перекрёстка сырого графа.
    radius_m — только при обоих звеньях >= min_link_m (иначе NaN).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(junction_xy)
    rows = []
    for r in blocks[["block_id", "coords_json", "type_a", "type_b", "highway", "is_ring"]].itertuples(index=False):
        cs = np.array(json.loads(r.coords_json), float)
        if len(cs) < 3:
            continue
        x, y = utm38n_forward(cs[:, 0], cs[:, 1])
        xy = np.c_[x, y]
        keep = douglas_peucker(xy, dp_tol_m)
        if len(keep) < 3:
            continue
        s = xy[keep]
        turn, R, l1, l2 = turn_and_radius(s)
        seg = np.hypot(*np.diff(s, axis=0).T)
        arc = np.r_[0, np.cumsum(seg)]
        total = arc[-1]
        inner_arc = arc[1:-1]
        ok = np.ones(len(turn), bool)
        if r.type_a == "junction":
            ok &= inner_arc >= clearance_m
        if r.type_b == "junction":
            ok &= (total - inner_arc) >= clearance_m
        dj, _ = tree.query(s[1:-1])
        ok &= dj >= clearance_m
        radius = np.where((l1 >= min_link_m) & (l2 >= min_link_m), R, np.nan)
        ll = cs[keep][1:-1]
        for k in range(len(turn)):
            rows.append((r.block_id, r.highway, ll[k, 0], ll[k, 1], s[k + 1, 0], s[k + 1, 1], turn[k],
                         radius[k], l1[k], l2[k], bool(ok[k])))
    v = pd.DataFrame(rows, columns=["block_id", "highway", "lat", "lng", "x_utm", "y_utm", "turn_deg", "radius_m",
                                    "link_before_m", "link_after_m", "counted"])
    v["h3"] = [h3.latlng_to_cell(a, b, h3_resolution) for a, b in zip(v.lat, v.lng)]
    # длина «на кривой»: дуга R·δ, но не больше полусуммы звеньев
    v["curve_length_m"] = np.minimum(np.nan_to_num(v.radius_m, nan=0) * np.radians(v.turn_deg),
                                     (v.link_before_m + v.link_after_m) / 2)
    return v


# --- 4.3. ориентация ---------------------------------------------------------------------------
def street_orientation_bins(roads: pd.DataFrame, step_m: float = SAMPLE_STEP_M, bins: int = ORIENTATION_BINS,
                            h3_resolution: int = H3_RESOLUTION) -> pd.DataFrame:
    """Длина уличной сети по ячейкам и бинам направления (0..180°, bins бинов), сэмплы ~20 м."""
    acc: dict[tuple[str, int], float] = defaultdict(float)
    width = 180.0 / bins
    for r in roads[["highway", "coords_json"]].itertuples(index=False):
        if str(r.highway) not in STREET_ROAD_CLASSES:
            continue
        cs = json.loads(r.coords_json)
        for (lat1, lng1), (lat2, lng2) in zip(cs, cs[1:]):
            seg = haversine_m(lat1, lng1, lat2, lng2)
            if seg <= 0:
                continue
            dx = (lng2 - lng1) * math.cos(math.radians((lat1 + lat2) / 2)); dy = lat2 - lat1
            b = min(int((math.degrees(math.atan2(dx, dy)) % 180.0) / width), bins - 1)
            pieces = max(1, math.ceil(seg / step_m)); piece = seg / pieces
            for i in range(pieces):
                fr = (i + 0.5) / pieces
                acc[(h3.latlng_to_cell(lat1 + (lat2 - lat1) * fr, lng1 + (lng2 - lng1) * fr, h3_resolution), b)] += piece
    s = pd.Series(acc)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["h3", "bin"])
    return s.unstack("bin", fill_value=0.0).reindex(columns=range(bins), fill_value=0.0)


def entropy_bits(w: np.ndarray) -> np.ndarray:
    tot = w.sum(1, keepdims=True)
    p = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(1)
    return h


# --- агрегирование в ячейки -----------------------------------------------------------------------
def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def aggregate_h3_geometry_v2_features(
    cells: Iterable[str],
    blocks: pd.DataFrame,
    curve_vertices: pd.DataFrame,
    orientation: pd.DataFrame,
    cons_nodes: pd.DataFrame,
    raw_nodes: pd.DataFrame,
    point_nodes: pd.DataFrame,
    min_street_m: float = MIN_STREET_LENGTH_M,
    radius_cap_m: float = RADIUS_CAP_M,
    sharp_radius_m: float = SHARP_RADIUS_M,
) -> pd.DataFrame:
    """Признаки ячейки этапа 4 (определения — README этапа 4).

    block_length_median_m      — медиана длин кварталов (перекрёсток -> перекрёсток, consolidated, без
                                 at_edge), взвешенная длиной квартала внутри ячейки; NaN без кварталов;
    curvature_deg_per_km_v2    — Σ|угол поворота| учтённых вершин / уличная длина, км; NaN при < 200 м;
    min_curve_radius_m         — мин. радиус учтённых вершин, cap 1000 м; NaN при уличной длине < 200 м;
    share_length_radius_lt_200m — длина на кривых R < 200 м / уличная длина (альтернатива, диагностика);
    road_orientation_entropy_v2 — энтропия (бит) 8 бинов направления уличной сети; NaN при < 200 м;
    street_dead_end_count      — узлы степени 1 consolidated-графа, кроме at_edge;
    street_dead_end_count_raw  — то же без исключения at_edge (диагностика);
    level_crossing_count       — railway=level_crossing (автомобильные переезды);
    block_count, street_length_geom_m — диагностика.
    """
    idx = pd.Index(sorted(set(cells)), name="h3")
    out = pd.DataFrame(index=idx)

    street_len = orientation.sum(1).reindex(idx).fillna(0.0)
    out["street_length_geom_m"] = street_len
    ok_len = street_len >= min_street_m

    # длина квартала
    rows = []
    for r in blocks[blocks.is_junction_block & ~blocks.at_edge][["block_id", "length_m", "coords_json"]].itertuples(index=False):
        for cell, L in allocate_coords_to_h3(json.loads(r.coords_json)).items():
            rows.append((cell, r.block_id, r.length_m, L))
    bc = pd.DataFrame(rows, columns=["h3", "block_id", "block_length_m", "len_in_cell"])
    med = bc.groupby("h3").apply(lambda g: weighted_median(g.block_length_m.to_numpy(), g.len_in_cell.to_numpy()),
                                 include_groups=False)
    out["block_length_median_m"] = med.reindex(idx)
    out["block_count"] = bc.groupby("h3")["block_id"].nunique().reindex(idx).fillna(0).astype(int)

    # кривизна
    cv = curve_vertices[curve_vertices.counted]
    out["curvature_deg_per_km_v2"] = (cv.groupby("h3")["turn_deg"].sum().reindex(idx).fillna(0.0)
                                      / (street_len / 1000)).where(ok_len)
    rmin = cv.groupby("h3")["radius_m"].min().reindex(idx)
    out["min_curve_radius_m"] = rmin.clip(upper=radius_cap_m).fillna(radius_cap_m).where(ok_len)
    sharp = cv[cv.radius_m < sharp_radius_m].groupby("h3")["curve_length_m"].sum().reindex(idx).fillna(0.0)
    out["share_length_radius_lt_200m"] = (sharp / street_len).clip(upper=1.0).where(ok_len)

    # ориентация
    w = orientation.reindex(idx).fillna(0.0).to_numpy()
    out["road_orientation_entropy_v2"] = pd.Series(entropy_bits(w), index=idx).where(ok_len)

    # тупики
    de = cons_nodes[cons_nodes.node_type == "dead_end"]
    out["street_dead_end_count"] = de[~de.at_edge].groupby("h3").size().reindex(idx).fillna(0).astype(int)
    out["street_dead_end_count_raw"] = de.groupby("h3").size().reindex(idx).fillna(0).astype(int)

    # переезды
    lc = point_nodes[point_nodes["railway"] == "level_crossing"]
    out["level_crossing_count"] = lc.groupby("h3").size().reindex(idx).fillna(0).astype(int)
    return out.reset_index()
