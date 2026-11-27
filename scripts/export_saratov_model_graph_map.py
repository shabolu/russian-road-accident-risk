"""Export the saved stage-4 model graph, without recomputing its topology."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "parquet_output"


def js_safe(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def graph(mode: str) -> dict:
    suffix = "consolidated_" if mode == "merged" else ""
    node_path = SOURCE / f"osm_street_graph_{suffix}nodes.parquet"
    edge_path = SOURCE / f"osm_street_graph_{suffix}edges.parquet"
    nodes = pd.read_parquet(node_path)
    edges = pd.read_parquet(edge_path)
    assert nodes.node_id.is_unique and edges.edge_id.is_unique
    ids = set(nodes.node_id)
    assert set(edges.u) | set(edges.v) == ids
    degree = pd.concat([edges.u, edges.v]).value_counts()
    assert degree.reindex(nodes.node_id).to_numpy().tolist() == nodes.degree.tolist()
    assert not edges.highway.eq("service").any()
    lookup = {int(r.node_id): (float(r.lat), float(r.lng)) for r in nodes.itertuples()}
    encoded_edges = []
    for r in edges.itertuples():
        coords = json.loads(r.coords_json)
        # Use the stored post-consolidation geometry, including its centroid connectors.
        assert np.allclose(coords[0], lookup[int(r.u)], rtol=0, atol=2e-7)
        assert np.allclose(coords[-1], lookup[int(r.v)], rtol=0, atol=2e-7)
        encoded_edges.append([
            int(r.edge_id), str(r.u), str(r.v), str(r.osm_id), str(r.highway),
            r.name if isinstance(r.name, str) else "", round(float(r.length_m), 2),
            int(r.oneway_dir), coords, bool(r.at_edge),
        ])
    encoded_nodes = []
    for r in nodes.itertuples():
        members = json.loads(r.raw_node_ids) if mode == "merged" else [int(r.node_id)]
        encoded_nodes.append([
            str(r.node_id), float(r.lat), float(r.lng), int(r.degree),
            bool(r.at_edge), [str(x) for x in members],
            float(r.cluster_radius_m) if mode == "merged" else 0,
        ])
    stats = {
        "nodes": len(nodes), "edges": len(edges),
        "junctions": int(nodes.degree.ge(3).sum()),
        "three_leg": int(nodes.degree.eq(3).sum()),
        "four_plus": int(nodes.degree.ge(4).sum()),
        "dead_ends": int(nodes.degree.eq(1).sum()),
        "pseudo": int(nodes.degree.eq(2).sum()),
        "boundary_nodes": int(nodes.at_edge.sum()),
        "merged_clusters": int((nodes.n_raw_nodes > 1).sum()) if mode == "merged" else 0,
        "length_km": round(float(edges.length_m.sum() / 1000), 3),
        "sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (node_path, edge_path)
        },
    }
    return {"nodes": encoded_nodes, "edges": encoded_edges, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.home() / "Downloads" / "Граф_дорог_Саратова_из_модели")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {"merged": graph("merged"), "raw": graph("raw")}
    junctions = pd.read_parquet(SOURCE / "osm_junction_v2_nodes.parquet")
    features = junctions[junctions.is_street_junction]
    assert set(features.node_id.astype(str)) == {
        n[0] for n in data["raw"]["nodes"] if n[3] >= 3
    }
    data["feature_nodes"] = [
        [str(r.node_id), float(r.lat), float(r.lng), int(r.degree_street),
         bool(r.is_signalized_junction), str(r.names) if isinstance(r.names, str) else ""]
        for r in features.itertuples()
    ]
    data["bbox"] = [[51.40, 45.80], [51.70, 46.20]]
    assets = Path(tempfile.gettempdir()) / "dtp-map-assets"
    assets.mkdir(exist_ok=True)
    for filename in ("leaflet.css", "leaflet.js"):
        path = assets / filename
        if not path.exists():
            with urllib.request.urlopen(f"https://unpkg.com/leaflet@1.9.4/dist/{filename}", timeout=40) as response:
                path.write_bytes(response.read())
    template = (ROOT / "scripts/saratov_model_graph_map.html").read_text()
    html = template.replace("__LEAFLET_CSS__", (assets / "leaflet.css").read_text())
    html = html.replace("__LEAFLET_JS__", (assets / "leaflet.js").read_text()).replace("__GRAPH_DATA__", js_safe(data))
    target = args.output / "Карта_графа_Саратова.html"
    target.write_text(html)
    stats = {k: data[k]["stats"] for k in ("merged", "raw")}
    stats["feature_junctions_stage2"] = len(features)
    (args.output / "Параметры_графа.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    (args.output / "Прочитай_меня.txt").write_text(
        "Карта сохранённой уличной сети модели, а не новый маршрутный граф.\n"
        "По умолчанию: результат этапа 4, tol=20 м, связь по улицам <=40 м, отдельное правило колец <=400 м.\n"
        "Узлы и геометрия рёбер прочитаны непосредственно из parquet_output/osm_street_graph_consolidated_*.parquet.\n"
        "Можно переключить граф до склейки и включить исходные точки склеенных перекрёстков.\n"
        "Дополнительный слой этапа 2 содержит именно точки признаков street_junction_3leg_count / street_junction_4leg_count.\n"
        "Этап 2 исправил число подходов; этап 4 консолидировал граф для геометрических признаков.\n"
        "Подсчёты перекрёстков этапа 2 не подменены количеством консолидированных узлов.\n"
        "Уличная сеть исключает service, как в этапах 2/4. Степень 2 — стык, а не перекрёсток.\n"
        "Область сохранённого саратовского эксперимента включает часть Энгельса и граничные дороги.\n"
        "Узлы at_edge отмечены отдельно, ложные граничные тупики не выдаются за обычные.\n"
        "Это исторический снимок OSM проекта; топология и признаки в источниках не изменялись.\n"
        "Рёбра после склейки уже содержат короткие соединения с центроидом кластера — показана сохранённая геометрия.\n"
        "Карта открывается обычным браузером. Геометрия и Leaflet встроены в HTML; подложке нужен интернет.\n"
        "Нажмите узел: подсветятся подходы, появятся степень и состав склейки. Нажмите ребро: название и параметры.\n"
    )
    report = ROOT / "docs/analysis/model_graph_map"
    report.mkdir(exist_ok=True)
    (report / "statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(target)
    print(json.dumps({k: {x: v for x, v in s.items() if x != "sources"} if isinstance(s, dict) else s for k, s in stats.items()}, ensure_ascii=False))
    print(f"HTML: {target.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
