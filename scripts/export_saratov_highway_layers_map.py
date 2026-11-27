"""Export an offline HTML map with toggles for Saratov motor-road classes."""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from pathlib import Path

import osmium
import pandas as pd
from shapely.geometry import LineString, shape
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[1]
PBF = ROOT / "data/raw/osm/volga-fed-district-260603.osm.pbf"
BOUNDARIES = ROOT / "reports/multicity_350k/boundaries.geojson"
ROUTE_COVERAGE = ROOT / "reports/route_proxy_coverage/ways/saratov.parquet"
OUTPUT = Path.home() / "Downloads/Карта_дорог_Саратова_по_классам.html"
PREVIOUS_OUTPUT = Path.home() / "Downloads/Карта_дорог_Саратова_service.html"

ROAD_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "service", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link",
}
SERVICE_TYPES = {"driveway", "parking_aisle", "alley", "drive-through", "emergency_access"}

COLORS = {
    "motorway": "#aa2330",
    "trunk": "#c45b16",
    "primary": "#df8b15",
    "secondary": "#9f9e19",
    "tertiary": "#378e62",
    "unclassified": "#457c91",
    "residential": "#3882c1",
    "living_street": "#78a0c2",
    "motorway_link": "#d06b76",
    "trunk_link": "#d7916d",
    "primary_link": "#e6b66a",
    "secondary_link": "#b5bb73",
    "tertiary_link": "#8abb9d",
    "service:driveway": "#d3543e",
    "service:parking_aisle": "#8e51ad",
    "service:alley": "#277da1",
    "service:drive-through": "#16a085",
    "service:emergency_access": "#c28a1e",
    "service:other": "#6a7787",
    "service:unspecified": "#465569",
}
LABELS = {
    "motorway": "Автомагистрали",
    "trunk": "Дороги trunk",
    "primary": "Основные дороги primary",
    "secondary": "Дороги secondary",
    "tertiary": "Дороги tertiary",
    "unclassified": "Дороги unclassified",
    "residential": "Жилые улицы",
    "living_street": "Жилые зоны",
    "motorway_link": "Съезды motorway",
    "trunk_link": "Съезды trunk",
    "primary_link": "Съезды primary",
    "secondary_link": "Съезды secondary",
    "tertiary_link": "Съезды tertiary",
    "service:driveway": "service: подъезды",
    "service:parking_aisle": "service: парковки",
    "service:alley": "service: переулки",
    "service:drive-through": "service: drive-through",
    "service:emergency_access": "service: экстренный доступ",
    "service:other": "service: прочие",
    "service:unspecified": "service: без подтипа",
}
EXTRA_FIELDS = {
    "combined_speed_length_share": "скорость",
    "combined_surface_coarse_length_share": "покрытие",
    "combined_lanes_length_share": "полосы",
    "combined_width_length_share": "ширина",
    "osm_smoothness_length_share": "smoothness",
}
COMPLETENESS_COLORS = ["#d43f46", "#e47b35", "#cba42a", "#5795c6", "#316fa8", "#26775b"]


def line_parts(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        if geometry.length > 0:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from line_parts(part)


class RoadExtractor(osmium.SimpleHandler):
    def __init__(self, polygon):
        super().__init__()
        self.polygon = polygon
        self.prepared = prep(polygon)
        self.bounds = polygon.bounds
        self.features = []
        self.way_ids = set()
        self.counts = Counter()
        self.scanned = 0

    def way(self, way):
        self.scanned += 1
        highway = way.tags.get("highway")
        if highway not in ROAD_CLASSES:
            return
        try:
            coords = [(n.location.lon, n.location.lat) for n in way.nodes]
        except osmium.InvalidLocationError:
            return
        if len(coords) < 2:
            return
        west, south, east, north = self.bounds
        if (max(x for x, _ in coords) < west or min(x for x, _ in coords) > east
                or max(y for _, y in coords) < south or min(y for _, y in coords) > north):
            return
        line = LineString(coords)
        if not self.prepared.intersects(line):
            return
        subtype = way.tags.get("service", "") or "unspecified"
        if highway == "service":
            category = f"service:{subtype if subtype in SERVICE_TYPES or subtype == 'unspecified' else 'other'}"
        else:
            category = highway
        tags = {
            "id": int(way.id),
            "name": way.tags.get("name", ""),
            "highway": highway,
            "service": subtype if highway == "service" else "",
            "access": way.tags.get("access", ""),
            "surface": way.tags.get("surface", ""),
        }
        kept = False
        for part in line_parts(line.intersection(self.polygon)):
            latlon = [[round(y, 6), round(x, 6)] for x, y in part.coords]
            if len(latlon) < 2:
                continue
            self.features.append([category, latlon, tags])
            kept = True
        if kept:
            self.way_ids.add(int(way.id))
            self.counts[category] += 1


def leaflet_asset(name: str) -> str:
    cache = ROOT / "data/raw/map_assets" / name
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://unpkg.com/leaflet@1.9.4/dist/{name}"
        with urllib.request.urlopen(url, timeout=30) as response:
            cache.write_bytes(response.read())
    return cache.read_text(encoding="utf-8")


def main():
    coverage = pd.read_parquet(ROUTE_COVERAGE, columns=["osm_id", "highway", *EXTRA_FIELDS])
    assert coverage.osm_id.is_unique
    coverage["known_names"] = coverage.apply(
        lambda row: [label for key, label in EXTRA_FIELDS.items() if row[key] >= 1 - 1e-6], axis=1
    )
    known_by_id = dict(zip(coverage.osm_id, coverage.known_names))
    highway_by_id = dict(zip(coverage.osm_id, coverage.highway))
    boundaries = json.loads(BOUNDARIES.read_text())
    feature = next(f for f in boundaries["features"] if f["properties"].get("slug") == "saratov")
    polygon = shape(feature["geometry"])
    extractor = RoadExtractor(polygon)
    extractor.apply_file(str(PBF), locations=True, idx="sparse_mem_array")
    assert extractor.features and extractor.way_ids
    assert extractor.way_ids == set(known_by_id), "OSM map and route-coverage audit must use the same ways"
    completeness_counts = Counter()
    for _, _, tags in extractor.features:
        if tags["highway"] != "service":
            known = known_by_id[tags["id"]]
            tags["known_extra"] = len(known)
            tags["known_names"] = known
    for way_id in extractor.way_ids:
        if highway_by_id[way_id] != "service":
            completeness_counts[len(known_by_id[way_id])] += 1
    assert sum(completeness_counts.values()) == 8783
    data = json.dumps(extractor.features, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    boundary = json.dumps(feature["geometry"], ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    colors = json.dumps(COLORS, ensure_ascii=False)
    labels = json.dumps(LABELS, ensure_ascii=False)
    counts = json.dumps(extractor.counts, ensure_ascii=False)
    completeness_colors = json.dumps(COMPLETENESS_COLORS)
    completeness_counts_json = json.dumps([completeness_counts[i] for i in range(6)])
    css = leaflet_asset("leaflet.css")
    js = leaflet_asset("leaflet.js")
    html = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Саратов — классы дорог и заполненность маршрутных признаков</title>
<style>__LEAFLET_CSS__</style>
<style>
html,body,#map{height:100%;margin:0} #map{background:#f5f6f8}
.panel{position:absolute;z-index:1000;top:14px;left:14px;max-width:310px;background:#fff;
border:1px solid #d7dce3;border-radius:10px;padding:13px 16px;font:14px/1.42 system-ui,sans-serif;
box-shadow:0 3px 14px #25324722;color:#253247}
.panel h1{font-size:18px;line-height:1.2;margin:0 0 6px}.panel p{margin:6px 0}
.panel small{color:#657485}.panel button{margin:7px 6px 0 0;padding:6px 9px;border:1px solid #bcc8d6;
border-radius:6px;background:#f7f9fc;cursor:pointer}.key{display:inline-block;width:12px;height:5px;margin-right:6px;vertical-align:middle}
.legend{margin:7px 0 2px}.legend div{margin:3px 0}.leaflet-control-layers{font:13px system-ui,sans-serif;
max-height:70vh;overflow:auto}.panel details{margin:8px 0}.panel summary{cursor:pointer}
@media(max-width:600px){.panel{max-width:210px;font-size:12px;padding:9px}.panel h1{font-size:15px}}
</style></head><body>
<div id="map"></div><div class="panel"><h1>Дороги Саратова</h1>
<p><b>__WAY_COUNT__</b> автомобильных OSM-ways внутри муниципальной границы.
При открытии включён только <b>highway=service</b>.</p>
<details><summary>Цвета и число дорог по классам</summary><div id="legend" class="legend"></div></details>
<details><summary>Число дополнительных полей</summary><div id="completenessLegend" class="legend"></div></details>
<button onclick="showOnlyService()">Только service</button>
<button onclick="showAll()">Все классы</button>
<button onclick="showCompleteness()">Заполненность 0–5</button>
<button onclick="showCompleteness(0)">Только 0 полей</button>
<button onclick="hideAll()">Скрыть все</button><br>
<button onclick="map.setView([51.5300,46.0347],12)">К центру</button>
<button onclick="map.fitBounds(border.getBounds())">Весь округ</button>
<p><small>OSM: 03.06.2026. Справа можно включать классы и группы по заполненности.
Последние содержат только дороги без service. Считаются скорость, покрытие,
полосы, ширина и smoothness, известные на всей длине OSM-way.
Подложка по умолчанию отключена; её можно включить там же.</small></p>
</div>
<script>__LEAFLET_JS__</script>
<script>
const data=__DATA__, boundary=__BOUNDARY__, colors=__COLORS__, labels=__LABELS__, counts=__COUNTS__;
const completenessColors=__COMPLETENESS_COLORS__, completenessCounts=__COMPLETENESS_COUNTS__;
const canvas=L.canvas({padding:0.5});
const map=L.map('map',{preferCanvas:true,zoomControl:false}).setView([51.5300,46.0347],12);
L.control.zoom({position:'bottomright'}).addTo(map);
const blank=L.layerGroup().addTo(map);
const osm=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19,attribution:'&copy; OpenStreetMap contributors'});
const border=L.geoJSON(boundary,{style:{color:'#172b4d',weight:2,dashArray:'7 5',fill:false,opacity:0.7}}).addTo(map);
const layers={};
const completenessLayers=Array.from({length:6},()=>L.layerGroup());
for(const key of Object.keys(colors)){
  layers[key]=L.layerGroup();
  if(key.startsWith('service:')) layers[key].addTo(map);
}
for(const [category,coords,tags] of data){
  const weight=category.startsWith('service:')?2:(['motorway','trunk','primary'].includes(category)?3.5:2.7);
  const line=L.polyline(coords,{renderer:canvas,color:colors[category],weight,opacity:0.82,
    interactive:true});
  const parts=[`<b>OSM way ${tags.id}</b>`,tags.name?`Название: ${escapeHtml(tags.name)}`:'',
    `highway=${escapeHtml(tags.highway)}`,
    tags.service?`service=${escapeHtml(tags.service)}`:'',tags.access?`access=${escapeHtml(tags.access)}`:'',
    tags.surface?`surface=${escapeHtml(tags.surface)}`:'',
    tags.highway==='service'?'':`Дополнительных полей: ${tags.known_extra} из 5`,
    tags.known_names?.length?`Известны: ${tags.known_names.map(escapeHtml).join(', ')}`:''].filter(Boolean);
  line.bindPopup(parts.join('<br>'));
  layers[category].addLayer(line);
  if(tags.highway!=='service'){
    const completeLine=L.polyline(coords,{renderer:canvas,color:completenessColors[tags.known_extra],
      weight:3.4,opacity:0.94,interactive:true});
    completeLine.bindPopup(parts.join('<br>'));
    completenessLayers[tags.known_extra].addLayer(completeLine);
  }
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',
  '"':'&quot;',"'":'&#39;'}[c]));}
const overlays={'Граница Саратова':border};
for(const key of Object.keys(colors)) if(counts[key]) overlays[`${labels[key]} (${counts[key]})`]=layers[key];
for(let i=0;i<6;i++) overlays[`Заполненность: ${i} из 5 (${completenessCounts[i]})`]=completenessLayers[i];
L.control.layers({'Без подложки':blank,'OpenStreetMap':osm},overlays,{collapsed:false,position:'topright'}).addTo(map);
function hideCompleteness(){for(const layer of completenessLayers) map.removeLayer(layer);}
map.on('overlayadd',event=>{
  if(completenessLayers.includes(event.layer)){
    for(const layer of Object.values(layers)) map.removeLayer(layer);
  }else if(Object.values(layers).includes(event.layer)) hideCompleteness();
});
function showOnlyService(){hideCompleteness();for(const key of Object.keys(layers)){
  if(key.startsWith('service:')) map.addLayer(layers[key]); else map.removeLayer(layers[key]);}}
function showAll(){hideCompleteness();for(const key of Object.keys(layers)) if(counts[key]) map.addLayer(layers[key]);}
function hideAll(){for(const key of Object.keys(layers)) map.removeLayer(layers[key]);hideCompleteness();}
function showCompleteness(only=null){hideAll();for(let i=0;i<6;i++)
  if(only===null||only===i) map.addLayer(completenessLayers[i]);}
document.getElementById('legend').innerHTML=Object.keys(colors).filter(k=>counts[k]).map(k=>
  `<div><span class="key" style="background:${colors[k]}"></span>${labels[k]}: ${counts[k]}</div>`).join('');
document.getElementById('completenessLegend').innerHTML=completenessCounts.map((n,i)=>
  `<div><span class="key" style="background:${completenessColors[i]}"></span>${i} из 5: ${n} дорог</div>`).join('');
if(location.hash==='#all') showAll();
if(location.hash==='#completeness') showCompleteness();
</script></body></html>"""
    html = (html.replace("__LEAFLET_CSS__", css).replace("__LEAFLET_JS__", js)
            .replace("__DATA__", data).replace("__BOUNDARY__", boundary)
            .replace("__COLORS__", colors).replace("__LABELS__", labels)
            .replace("__COUNTS__", counts)
            .replace("__COMPLETENESS_COLORS__", completeness_colors)
            .replace("__COMPLETENESS_COUNTS__", completeness_counts_json)
            .replace("__WAY_COUNT__", f"{len(extractor.way_ids):,}".replace(",", " ")))
    OUTPUT.write_text(html)
    PREVIOUS_OUTPUT.write_text(html)
    print(json.dumps({"output": str(OUTPUT), "ways": len(extractor.way_ids),
                      "line_parts": len(extractor.features), "categories": extractor.counts,
                      "completeness": completeness_counts,
                      "html_mb": round(OUTPUT.stat().st_size / 1e6, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
