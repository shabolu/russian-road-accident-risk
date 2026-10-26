"""Этап 6.1: региональная выборка из PBF Приволжского ФО (Саратов с буфером + центр Самары).

Без osmium-tool: pyosmium IdTracker.
  проход 1 — узлы внутри областей; ways, у которых хоть один узел внутри; relations,
             у которых хоть один член (узел/way) внутри;
  проход 2 — дополнение обратных ссылок (все узлы выбранных ways, члены relations);
  проход 3 — запись выбранных объектов в новый PBF.

    PYTHONPATH=src venv/bin/python scripts/extract_osm_v6_region.py
"""
from __future__ import annotations

import time
from pathlib import Path

import osmium

from road_accident_risk.paths import OSM_VOLGA_PBF, RAW_OSM_DIR

OUT = RAW_OSM_DIR / "v6_extract_saratov_buffer_samara.osm.pbf"

# (lat_min, lon_min, lat_max, lon_max)
SARATOV_BBOX = (51.40, 45.80, 51.70, 46.20)
BUF_LAT, BUF_LON = 0.09, 0.145  # ~10 км: 0.09° широты = 10.0 км, 0.145° долготы на 51.5° = 10.0 км
SARATOV_BUFFER = (SARATOV_BBOX[0] - BUF_LAT, SARATOV_BBOX[1] - BUF_LON,
                  SARATOV_BBOX[2] + BUF_LAT, SARATOV_BBOX[3] + BUF_LON)
SAMARA_BBOX = (53.10, 49.95, 53.35, 50.40)
REGIONS = [SARATOV_BUFFER, SAMARA_BBOX]


def inside(lat: float, lon: float) -> bool:
    for a, b, c, d in REGIONS:
        if a <= lat <= c and b <= lon <= d:
            return True
    return False


def main() -> None:
    t0 = time.time()
    tr = osmium.IdTracker()
    n_nodes = n_in = n_ways = n_rel = 0
    for obj in osmium.FileProcessor(str(OSM_VOLGA_PBF)):
        if obj.is_node():
            n_nodes += 1
            loc = obj.location
            if loc.valid() and inside(loc.lat, loc.lon):
                tr.add_node(obj.id)
                n_in += 1
        elif obj.is_way():
            if tr.contains_any_references(obj):
                tr.add_way(obj.id)
                n_ways += 1
        elif obj.is_relation():
            if tr.contains_any_references(obj):
                tr.add_relation(obj.id)
                n_rel += 1
    t1 = time.time()
    print(f"pass1: nodes total {n_nodes:,}, in regions {n_in:,}; ways {n_ways:,}; relations {n_rel:,}; {t1 - t0:.0f}s", flush=True)
    tr.complete_backward_references(str(OSM_VOLGA_PBF), relation_depth=1)
    t2 = time.time()
    print(f"pass2 backward refs: {t2 - t1:.0f}s", flush=True)
    if OUT.exists():
        OUT.unlink()
    with osmium.SimpleWriter(str(OUT)) as w:
        for obj in osmium.FileProcessor(str(OSM_VOLGA_PBF)).with_filter(tr.id_filter()):
            w.add(obj)
    t3 = time.time()
    print(f"pass3 write: {t3 - t2:.0f}s; out {OUT} {OUT.stat().st_size / 1e6:.1f} MB; total {t3 - t0:.0f}s", flush=True)
    print("SARATOV_BUFFER", SARATOV_BUFFER)


if __name__ == "__main__":
    main()
