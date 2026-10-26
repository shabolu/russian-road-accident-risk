"""Этап 6.1: слои OSM из региональной выборки → parquet_output/osm_v6_*.parquet.

Вход — data/raw/osm/v6_extract_saratov_buffer_samara.osm.pbf (scripts/extract_osm_v6_region.py).
Один проход pyosmium с кэшем координат и сборкой площадей (closed ways + multipolygon relations).

Слои (region ∈ {saratov_buffer, samara}; in_bbox — внутри bbox проекта Саратова):
  osm_v6_streets       — уличная сеть буферной области (классы графа этапа 4, без service)
  osm_v6_tram_tracks   — railway=tram
  osm_v6_poi           — узлы и площади (центроиды) с shop/amenity/office/leisure/tourism/healthcare
  osm_v6_buildings     — полигоны building=* (этажность, квартиры, площадь основания UTM 38N, центроид)
  osm_v6_pt_stops      — highway=bus_stop, public_transport=platform|stop_position, railway=tram_stop
  osm_v6_pt_routes     — relations route=bus|trolleybus|tram|share_taxi (состав ways/stops)
  osm_v6_landuse       — полигоны landuse=*

    PYTHONPATH=src venv/bin/python scripts/build_osm_v6_layers.py
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import Counter

import numpy as np
import osmium
import pandas as pd

from road_accident_risk.osm_road_features import STREET_ROAD_CLASSES
from road_accident_risk.paths import PARQUET_DIR, RAW_OSM_DIR
from road_accident_risk.street_graph_v2 import utm38n_forward

EXTRACT = RAW_OSM_DIR / "v6_extract_saratov_buffer_samara.osm.pbf"
SARATOV_BBOX = (51.40, 45.80, 51.70, 46.20)
SARATOV_BUFFER = (51.31, 45.655, 51.79, 46.345)
SAMARA_BBOX = (53.10, 49.95, 53.35, 50.40)

POI_KEYS = ("shop", "amenity", "office", "leisure", "tourism", "healthcare")
ROUTE_TYPES = {"bus", "trolleybus", "tram", "share_taxi"}
BUILDING_KEEP_TAGS = ("name", "addr:street", "addr:housenumber", "height", "roof:levels",
                      "building:levels:underground", "building:min_level", "start_date")


def region_of(lat: float, lon: float) -> str | None:
    a, b, c, d = SARATOV_BUFFER
    if a <= lat <= c and b <= lon <= d:
        return "saratov_buffer"
    a, b, c, d = SAMARA_BBOX
    if a <= lat <= c and b <= lon <= d:
        return "samara"
    return None


def in_bbox(lat: float, lon: float, bb=SARATOV_BBOX) -> bool:
    return bb[0] <= lat <= bb[2] and bb[1] <= lon <= bb[3]


def parse_levels(v: str | None, cap: float = 100) -> float:
    """building:levels → число; '5;9' / '5-9' → максимум; мусор → NaN."""
    if v is None:
        return np.nan
    nums = re.findall(r"\d+(?:[.,]\d+)?", str(v))
    if not nums:
        return np.nan
    x = max(float(n.replace(",", ".")) for n in nums)
    return x if 0 < x <= cap else np.nan


def ring_xy(ring) -> np.ndarray:
    pts = [(n.lat, n.lon) for n in ring if n.location.valid()]
    return np.array(pts, dtype=float) if pts else np.zeros((0, 2))


def poly_area_centroid(xy: np.ndarray) -> tuple[float, float, float]:
    """Подписанная площадь и центроид многоугольника (x, y в метрах)."""
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-9:
        return 0.0, float(x.mean()), float(y.mean())
    cx = ((x + x1) * cr).sum() / (6 * a)
    cy = ((y + y1) * cr).sum() / (6 * a)
    return float(a), float(cx), float(cy)


def area_geometry(area) -> dict | None:
    """Площадь (UTM 38N), центроид (lat/lon) и кольца для osmium.osm.Area."""
    rings = []  # (sign, latlon array)
    for outer in area.outer_rings():
        o = ring_xy(outer)
        if len(o) >= 3:
            rings.append((1, o))
        for inner in area.inner_rings(outer):
            i = ring_xy(inner)
            if len(i) >= 3:
                rings.append((-1, i))
    if not rings:
        return None
    tot = 0.0
    sx = sy = 0.0
    for sign, ll in rings:
        ex, ny = utm38n_forward(ll[:, 0], ll[:, 1])
        a, cx, cy = poly_area_centroid(np.column_stack([ex, ny]))
        a = sign * abs(a)
        tot += a
        sx += a * cx
        sy += a * cy
    outer_ll = np.vstack([ll for s, ll in rings if s == 1])
    if tot <= 0:
        lat, lon = float(outer_ll[:, 0].mean()), float(outer_ll[:, 1].mean())
    else:
        # центроид обратно в lat/lon: линейная интерполяция по вершинам внешних колец
        ex, ny = utm38n_forward(outer_ll[:, 0], outer_ll[:, 1])
        cx, cy = sx / tot, sy / tot
        A = np.column_stack([ex - ex.mean(), ny - ny.mean(), np.ones(len(ex))])
        k_lat = np.linalg.lstsq(A, outer_ll[:, 0], rcond=None)[0]
        k_lon = np.linalg.lstsq(A, outer_ll[:, 1], rcond=None)[0]
        v = np.array([cx - ex.mean(), cy - ny.mean(), 1.0])
        lat, lon = float(v @ k_lat), float(v @ k_lon)
    return {
        "area_m2": max(tot, 0.0),
        "lat": lat,
        "lon": lon,
        "n_outer": sum(1 for s, _ in rings if s == 1),
        "n_inner": sum(1 for s, _ in rings if s == -1),
        "rings_json": json.dumps([[s, np.round(ll, 6).tolist()] for s, ll in rings]),
    }


def line_length_m(ll: np.ndarray) -> float:
    if len(ll) < 2:
        return 0.0
    ex, ny = utm38n_forward(ll[:, 0], ll[:, 1])
    return float(np.hypot(np.diff(ex), np.diff(ny)).sum())


def main() -> None:
    t0 = time.time()
    streets, trams, poi, buildings, stops, routes, landuse = [], [], [], [], [], [], []
    way_info: dict[int, tuple[float, float, float, str]] = {}  # way_id → (len, mid_lat, mid_lon, kind)
    node_pos: dict[int, tuple[float, float]] = {}               # узлы-остановки (для маршрутов)
    mp_relations = Counter()
    mp_assembled = Counter()   # все собранные площади из relations (вся выборка)
    mp_in_region = Counter()   # из них с центроидом в областях
    n_area = 0

    fp = osmium.FileProcessor(str(EXTRACT)).with_locations().with_areas()
    for obj in fp:
        if obj.is_area():
            n_area += 1
            tags = dict(obj.tags)
            is_b = "building" in tags and tags.get("building") != "no"
            is_l = "landuse" in tags
            is_p = any(k in tags for k in POI_KEYS)
            is_plat = tags.get("public_transport") == "platform"
            if not (is_b or is_l or is_p or is_plat):
                continue
            g = area_geometry(obj)
            otype = "way" if obj.from_way() else "relation"
            oid = int(obj.orig_id())
            if otype == "relation" and (is_b or is_l or is_p):
                kind = "building" if is_b else ("landuse" if is_l else "poi")
                mp_assembled[kind if g is not None else kind + "_empty"] += 1
            if g is None:
                continue
            reg = region_of(g["lat"], g["lon"])
            if reg is None:
                continue
            if otype == "relation" and (is_b or is_l or is_p):
                mp_in_region[kind] += 1
            base = {"osm_type": otype, "osm_id": oid, "region": reg,
                    "in_bbox": in_bbox(g["lat"], g["lon"]), "lat": g["lat"], "lon": g["lon"]}
            if is_b:
                buildings.append({
                    **base,
                    "building": tags.get("building"),
                    "building_levels_raw": tags.get("building:levels"),
                    "levels": parse_levels(tags.get("building:levels")),
                    "building_flats_raw": tags.get("building:flats"),
                    "flats": parse_levels(tags.get("building:flats"), cap=5000),
                    "footprint_m2": g["area_m2"],
                    "n_inner": g["n_inner"],
                    "has_poi_tags": is_p,
                    **{k.replace(":", "_"): tags.get(k) for k in BUILDING_KEEP_TAGS},
                    "rings_json": g["rings_json"],
                })
            if is_l:
                landuse.append({**base, "landuse": tags.get("landuse"), "area_m2": g["area_m2"],
                                "name": tags.get("name"), "rings_json": g["rings_json"]})
            if is_plat and otype == "way":
                stops.append({**base, "osm_type": "way_area", "highway": tags.get("highway"),
                              "public_transport": "platform", "railway": tags.get("railway"),
                              "bus": tags.get("bus"), "trolleybus": tags.get("trolleybus"),
                              "tram": tags.get("tram"), "share_taxi": tags.get("share_taxi"),
                              "name": tags.get("name"), "scheme": "ptv2"})
            if is_p:
                poi.append({**base, "geom_kind": "area", "footprint_m2": g["area_m2"],
                            **{k: tags.get(k) for k in POI_KEYS}, "name": tags.get("name"),
                            "is_building": is_b, "tags_json": json.dumps(tags, ensure_ascii=False)})
            continue

        if obj.is_node():
            loc = obj.location
            if not loc.valid() or not obj.tags:
                continue
            lat, lon = loc.lat, loc.lon
            reg = region_of(lat, lon)
            tags = dict(obj.tags)
            hw, pt, rw = tags.get("highway"), tags.get("public_transport"), tags.get("railway")
            is_stop = hw == "bus_stop" or pt in ("platform", "stop_position") or rw == "tram_stop"
            if is_stop:
                node_pos[int(obj.id)] = (lat, lon)
            if reg is None:
                continue
            base = {"osm_type": "node", "osm_id": int(obj.id), "region": reg, "in_bbox": in_bbox(lat, lon),
                    "lat": lat, "lon": lon}
            if any(k in tags for k in POI_KEYS):
                poi.append({**base, "geom_kind": "node", "footprint_m2": np.nan,
                            **{k: tags.get(k) for k in POI_KEYS}, "name": tags.get("name"),
                            "is_building": False, "tags_json": json.dumps(tags, ensure_ascii=False)})
            if is_stop:
                stops.append({**base, "highway": hw, "public_transport": pt, "railway": rw,
                              "bus": tags.get("bus"), "trolleybus": tags.get("trolleybus"),
                              "tram": tags.get("tram"), "share_taxi": tags.get("share_taxi"),
                              "name": tags.get("name"),
                              "scheme": "ptv2" if pt else "legacy"})
            continue

        if obj.is_way():
            tags = dict(obj.tags)
            hw, rw, pt = tags.get("highway"), tags.get("railway"), tags.get("public_transport")
            if not (hw or rw or pt == "platform"):
                continue
            nodes = [(n.ref, n.lat, n.lon) for n in obj.nodes if n.location.valid()]
            if len(nodes) < 1:
                continue
            ll = np.array([(a, b) for _, a, b in nodes])
            length = line_length_m(ll)
            mid = ll[len(ll) // 2]
            way_info[int(obj.id)] = (length, float(mid[0]), float(mid[1]), hw or rw or "platform")
            reg = region_of(float(ll[:, 0].mean()), float(ll[:, 1].mean()))
            touches_buf = bool(((ll[:, 0] >= SARATOV_BUFFER[0]) & (ll[:, 0] <= SARATOV_BUFFER[2])
                                & (ll[:, 1] >= SARATOV_BUFFER[1]) & (ll[:, 1] <= SARATOV_BUFFER[3])).any())
            touches_bbox = bool(((ll[:, 0] >= SARATOV_BBOX[0]) & (ll[:, 0] <= SARATOV_BBOX[2])
                                 & (ll[:, 1] >= SARATOV_BBOX[1]) & (ll[:, 1] <= SARATOV_BBOX[3])).any())
            if hw in STREET_ROAD_CLASSES and touches_buf and len(nodes) >= 2:
                streets.append({
                    "way_id": int(obj.id), "highway": hw, "oneway": tags.get("oneway"),
                    "junction": tags.get("junction"), "maxspeed": tags.get("maxspeed"),
                    "lanes": tags.get("lanes"), "name": tags.get("name"), "bridge": tags.get("bridge"),
                    "length_m": length, "touches_bbox": touches_bbox,
                    "mid_lat": float(mid[0]), "mid_lon": float(mid[1]),
                    "node_ids": [int(r) for r, _, _ in nodes],
                    "coords_json": json.dumps(np.round(ll, 7).tolist()),
                })
            if rw == "tram" and (touches_buf or reg == "samara"):
                trams.append({"way_id": int(obj.id), "region": reg or "saratov_buffer",
                              "length_m": length, "touches_bbox": touches_bbox,
                              "embedded": tags.get("embedded"), "service": tags.get("service"),
                              "coords_json": json.dumps(np.round(ll, 7).tolist())})
            if pt == "platform" and hw != "bus_stop" and reg is not None and not obj.is_closed():
                # платформы-линии (PTv2); полигональные платформы приходят как area без POI-тегов
                stops.append({"osm_type": "way", "osm_id": int(obj.id), "region": reg,
                              "in_bbox": in_bbox(float(mid[0]), float(mid[1])),
                              "lat": float(ll[:, 0].mean()), "lon": float(ll[:, 1].mean()),
                              "highway": hw, "public_transport": pt, "railway": rw,
                              "bus": tags.get("bus"), "trolleybus": tags.get("trolleybus"),
                              "tram": tags.get("tram"), "share_taxi": tags.get("share_taxi"),
                              "name": tags.get("name"), "scheme": "ptv2"})
            continue

        if obj.is_relation():
            tags = dict(obj.tags)
            if tags.get("type") == "multipolygon":
                if "building" in tags:
                    mp_relations["building"] += 1
                elif "landuse" in tags:
                    mp_relations["landuse"] += 1
                elif any(k in tags for k in POI_KEYS):
                    mp_relations["poi"] += 1
            if tags.get("type") == "route" and tags.get("route") in ROUTE_TYPES:
                ways, stopm, platm, other = [], [], [], 0
                for m in obj.members:
                    role = m.role or ""
                    if m.type == "w" and not role.startswith(("platform", "stop")):
                        ways.append(int(m.ref))
                    elif role.startswith("platform"):
                        platm.append((m.type, int(m.ref)))
                    elif role.startswith("stop") or m.type == "n":
                        stopm.append((m.type, int(m.ref)))
                    else:
                        other += 1
                routes.append({
                    "rel_id": int(obj.id), "route": tags.get("route"), "ref": tags.get("ref"),
                    "name": tags.get("name"), "from": tags.get("from"), "to": tags.get("to"),
                    "network": tags.get("network"), "operator": tags.get("operator"),
                    "ptv": tags.get("public_transport:version"),
                    "n_members": len(obj.members), "n_way_members": len(ways),
                    "n_stop_members": len(stopm), "n_platform_members": len(platm),
                    "way_ids": ways,
                    "stop_members_json": json.dumps(stopm), "platform_members_json": json.dumps(platm),
                })
    t1 = time.time()
    print(f"pass done {t1 - t0:.0f}s; areas seen {n_area:,}", flush=True)

    # маршруты: длина и регион по ways, доступным в выборке
    for r in routes:
        info = [way_info.get(w) for w in r["way_ids"]]
        got = [i for i in info if i is not None]
        r["n_way_with_geom"] = len(got)
        r["length_km"] = sum(i[0] for i in got) / 1000
        r["length_in_saratov_bbox_km"] = sum(i[0] for i in got if in_bbox(i[1], i[2])) / 1000
        r["length_in_samara_bbox_km"] = sum(i[0] for i in got if in_bbox(i[1], i[2], SAMARA_BBOX)) / 1000
        sm = [tuple(x) for x in json.loads(r["stop_members_json"]) + json.loads(r["platform_members_json"])]
        r["n_stop_nodes_with_geom"] = sum(1 for t, i in sm if t == "n" and i in node_pos)
        r["region"] = ("saratov_buffer" if r["length_in_saratov_bbox_km"] > 0 else
                       ("samara" if r["length_in_samara_bbox_km"] > 0 else None))

    out = {
        "osm_v6_streets": pd.DataFrame(streets),
        "osm_v6_tram_tracks": pd.DataFrame(trams),
        "osm_v6_poi": pd.DataFrame(poi),
        "osm_v6_buildings": pd.DataFrame(buildings),
        "osm_v6_pt_stops": pd.DataFrame(stops),
        "osm_v6_pt_routes": pd.DataFrame(routes),
        "osm_v6_landuse": pd.DataFrame(landuse),
    }
    for name, df in out.items():
        p = PARQUET_DIR / f"{name}.parquet"
        df.to_parquet(p, index=False)
        extra = ""
        if "region" in df:
            extra = " | " + ", ".join(f"{k}={v:,}" for k, v in df["region"].value_counts(dropna=False).items())
        print(f"{name}: {len(df):,} rows, {p.stat().st_size / 1e6:.1f} MB{extra}", flush=True)
    print("multipolygon relations seen:", dict(mp_relations))
    print("multipolygon areas assembled (whole extract):", dict(mp_assembled))
    print("multipolygon areas in regions:", dict(mp_in_region))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
