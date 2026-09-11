from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.osm_road_features import BBOX, H3_RESOLUTION, build_point_nodes, load_pbf_tagged_nodes
from road_accident_risk.paths import OSM_ROAD_POINT_NODES_PARQUET, OSM_VOLGA_PBF
from road_accident_risk.utils import format_size


def main() -> None:
    tagged_nodes = load_pbf_tagged_nodes(OSM_VOLGA_PBF, bbox=BBOX)
    point_nodes = build_point_nodes(tagged_nodes, h3_resolution=H3_RESOLUTION)
    OSM_ROAD_POINT_NODES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    point_nodes.to_parquet(OSM_ROAD_POINT_NODES_PARQUET, index=False)

    print(f"Saved: {OSM_ROAD_POINT_NODES_PARQUET}")
    print(f"Rows: {len(point_nodes):,}")
    print(f"Traffic signals: {int(point_nodes['is_traffic_signal'].sum()) if len(point_nodes) else 0:,}")
    print(f"Size: {format_size(OSM_ROAD_POINT_NODES_PARQUET.stat().st_size)}")


if __name__ == "__main__":
    main()
