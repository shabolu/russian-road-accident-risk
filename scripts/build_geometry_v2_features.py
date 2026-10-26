"""Этап 4 плана docs/analysis/features/PLAN.md: геометрия улиц без service.

Строит из osm_roads_raw.parquet (+ переезды из osm_road_point_nodes.parquet):
    parquet_output/osm_street_graph_{nodes,edges}.parquet               — сырой уличный граф
    parquet_output/osm_street_graph_consolidated_{nodes,edges}.parquet  — после склейки (tol 20 м + кольца)
    parquet_output/osm_street_blocks.parquet          — кварталы consolidated-графа (перекрёсток -> перекрёсток)
    parquet_output/osm_street_curve_vertices.parquet  — вершины DP-линий кварталов сырого графа (угол, радиус)
    parquet_output/h3_geometry_v2_features.parquet    — признаки ячейки (GEOMETRY_V2_FEATURES)
    reports/road_geo_v2_geometry_tol_sweep.csv        — перебор tol склейки
Старые parquet не трогаются. Схема графа — docs/analysis/features/stage4_geometry/README.md.

Запуск: PYTHONPATH=src venv/bin/python scripts/build_geometry_v2_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from road_accident_risk.paths import (  # noqa: E402
    H3_GEOMETRY_V2_FEATURES_PARQUET, H3_ROAD_FEATURES_PARQUET, H3_TAGS_V2_FEATURES_PARQUET,
    OSM_ROAD_POINT_NODES_PARQUET, OSM_ROADS_RAW_PARQUET, OSM_STREET_BLOCKS_PARQUET,
    OSM_STREET_CURVE_VERTICES_PARQUET, OSM_STREET_GRAPH_CONSOLIDATED_EDGES_PARQUET,
    OSM_STREET_GRAPH_CONSOLIDATED_NODES_PARQUET, OSM_STREET_GRAPH_EDGES_PARQUET, OSM_STREET_GRAPH_NODES_PARQUET,
    PROJECT_ROOT,
)
from road_accident_risk.street_graph_v2 import (  # noqa: E402
    CONSOLIDATION_TOL_M, DP_TOL_M, GEOMETRY_V2_FEATURES, aggregate_h3_geometry_v2_features, build_street_blocks,
    build_street_graph_raw, consolidate_street_graph, curve_vertices_for_blocks, street_orientation_bins,
)

SWEEP_TOLS = (10, 15, 20, 25, 30)


def components(n: pd.DataFrame, e: pd.DataFrame) -> tuple[int, float]:
    idx = pd.Series(np.arange(len(n)), index=n.node_id)
    u = idx.loc[e.u].to_numpy(); v = idx.loc[e.v].to_numpy()
    k, lab = connected_components(coo_matrix((np.ones(len(e)), (u, v)), shape=(len(n),) * 2), directed=False)
    L = pd.Series(e.length_m.to_numpy()).groupby(lab[u]).sum()
    return k, float(L.max() / L.sum())


def sweep(n: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rb in (False, True):
        for tol in SWEEP_TOLS:
            cn, ce = consolidate_street_graph(n, e, tol_m=tol, roundabouts=rb)
            multi = cn[cn.n_raw_nodes > 1]
            bl, _ = build_street_blocks(cn, ce)
            jb = bl[bl.is_junction_block & ~bl.at_edge]
            rows.append({"roundabouts": rb, "tol_m": tol, "nodes": len(cn), "edges": len(ce),
                         "junctions": int((cn.node_type == "junction").sum()), "clusters": len(multi),
                         "raw_nodes_in_clusters": int(multi.n_raw_nodes.sum()),
                         "max_cluster_size": int(multi.n_raw_nodes.max()),
                         "clusters_ge5": int((multi.n_raw_nodes >= 5).sum()),
                         "max_cluster_radius_m": float(multi.cluster_radius_m.max()),
                         "p99_cluster_radius_m": float(multi.cluster_radius_m.quantile(0.99)),
                         "junction_blocks": len(jb), "block_median_m": float(jb.length_m.median()),
                         "block_share_lt_25m": float((jb.length_m < 25).mean()),
                         "block_share_lt_40m": float((jb.length_m < 40).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    roads = pd.read_parquet(OSM_ROADS_RAW_PARQUET)
    points = pd.read_parquet(OSM_ROAD_POINT_NODES_PARQUET)
    cells = pd.read_parquet(H3_ROAD_FEATURES_PARQUET, columns=["h3"])["h3"]

    n, e = build_street_graph_raw(roads)
    k, share = components(n, e)
    print(f"сырой граф: узлов {len(n):,} ({n.node_type.value_counts().to_dict()}), рёбер {len(e):,}, "
          f"{e.length_m.sum() / 1000:.1f} км; компонент {k}, главная {share:.1%} длины; at_edge {int(n.at_edge.sum())} "
          f"({n[n.at_edge].node_type.value_counts().to_dict()})", flush=True)

    sw = sweep(n, e)
    sw.to_csv(PROJECT_ROOT / "reports" / "road_geo_v2_geometry_tol_sweep.csv", index=False)
    print(sw.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)

    cn, ce = consolidate_street_graph(n, e, tol_m=CONSOLIDATION_TOL_M, roundabouts=True)
    k2, share2 = components(cn, ce)
    multi = cn[cn.n_raw_nodes > 1]
    print(f"consolidated (tol {CONSOLIDATION_TOL_M:.0f} м + кольца): узлов {len(cn):,} "
          f"({cn.node_type.value_counts().to_dict()}), рёбер {len(ce):,}, {ce.length_m.sum() / 1000:.1f} км; "
          f"кластеров {len(multi)} из {int(multi.n_raw_nodes.sum())} сырых узлов; компонент {k2}, главная {share2:.1%}; "
          f"at_edge {int(cn.at_edge.sum())}; удалено внутренних рёбер {len(e) - len(ce)}", flush=True)

    blocks, block_of = build_street_blocks(cn, ce)
    ce["block_id"] = ce["edge_id"].map(block_of)
    raw_blocks, raw_block_of = build_street_blocks(n, e)
    e["raw_block_id"] = e["edge_id"].map(raw_block_of)
    jxy = n.loc[n.node_type == "junction", ["x_utm", "y_utm"]].to_numpy()
    cv = curve_vertices_for_blocks(raw_blocks, jxy, dp_tol_m=DP_TOL_M).rename(columns={"block_id": "raw_block_id"})

    n.to_parquet(OSM_STREET_GRAPH_NODES_PARQUET, index=False)
    e.to_parquet(OSM_STREET_GRAPH_EDGES_PARQUET, index=False)
    cn.to_parquet(OSM_STREET_GRAPH_CONSOLIDATED_NODES_PARQUET, index=False)
    ce.to_parquet(OSM_STREET_GRAPH_CONSOLIDATED_EDGES_PARQUET, index=False)
    blocks.to_parquet(OSM_STREET_BLOCKS_PARQUET, index=False)
    cv.to_parquet(OSM_STREET_CURVE_VERTICES_PARQUET, index=False)
    jb = blocks[blocks.is_junction_block & ~blocks.at_edge]
    print(f"кварталов: {len(blocks):,}, перекрёсток->перекрёсток без края: {len(jb):,}, медиана {jb.length_m.median():.1f} м",
          flush=True)
    print(f"вершин DP {DP_TOL_M:.0f} м: {len(cv):,}, учтено {int(cv.counted.sum()):,}", flush=True)

    orient = street_orientation_bins(roads)
    cv_ = cv.rename(columns={"raw_block_id": "block_id"})
    feats = aggregate_h3_geometry_v2_features(cells, blocks, cv_, orient, cn, n, points)

    tags = pd.read_parquet(H3_TAGS_V2_FEATURES_PARQUET, columns=["h3", "street_length_m"]).set_index("h3")
    chk = feats.set_index("h3").join(tags)
    diff = (chk.street_length_geom_m - chk.street_length_m).abs().max()
    print(f"сверка уличной длины с этапом 3: max |Δ| = {diff:.3e} м")
    if diff > 1e-3:
        raise SystemExit("расхождение уличной длины с этапом 3")
    extra = set(orient.index) - set(cells)
    if extra:
        raise SystemExit(f"{len(extra)} ячеек с улицами вне h3_road_features")

    feats.to_parquet(H3_GEOMETRY_V2_FEATURES_PARQUET, index=False)
    d = feats[GEOMETRY_V2_FEATURES]
    print(pd.DataFrame({"nan_share": d.isna().mean(), "zero_share": (d == 0).mean(), "median": d.median(),
                        "p99": d.quantile(0.99), "max": d.max()}).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"ячеек: {len(feats):,} -> {H3_GEOMETRY_V2_FEATURES_PARQUET}")
    print(f"{time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
