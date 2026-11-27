"""Export an offline HTML map of Saratov OSM highway=service roads."""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from pathlib import Path

import osmium
from shapely.geometry import LineString, shape
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[1]
PBF = ROOT / "data/raw/osm/volga-fed-district-260603.osm.pbf"
BOUNDARIES = ROOT / "reports/multicity_350k/boundaries.geojson"
OUTPUT = Path.home() / "Downloads/Карта_дорог_Саратова_service.html"

COLORS = {
    "driveway": "#d3543e",
    "parking_aisle": "#8e51ad",
    "alley": "#277da1",
    "drive-through": "#16a085",
    "emergency_access": "#c28a1e",
    "other": "#6a7787",
    "unspecified": "#465569",
}
LABELS = {
    "driveway": "Подъезды",
    "parking_aisle": "Проезды парковок",
    "alley": "Переулки",
    "drive-through": "Проезды drive-through",
    "emergency_access": "Экстренный доступ",
    "other": "Прочие service",
    "unspecified": "Без подтипа service",
}


def line_parts(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        if geometry.length > 0:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from line_parts(part)


class ServiceExtractor(osmium.SimpleHandler):
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
        if way.tags.get("highway") != "service":
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
        category = subtype if subtype in COLORS else "other"
        tags = {
            "id": int(way.id),
            "name": way.tags.get("name", ""),
            "service": subtype,
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
    boundaries = json.loads(BOUNDARIES.read_text())
    feature = next(f for f in boundaries["features"] if f["properties"].get("slug") == "saratov")
    polygon = shape(feature["geometry"])
    extractor = ServiceExtractor(polygon)
    extractor.apply_file(str(PBF), locations=True, idx="sparse_mem_array")
    assert extractor.features and extractor.way_ids
    data = json.dumps(extractor.features, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    boundary = json.dumps(feature["geometry"], ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    colors = json.dumps(COLORS, ensure_ascii=False)
    labels = json.dumps(LABELS, ensure_ascii=False)
    counts = json.dumps(extractor.counts, ensure_ascii=False)
    css = leaflet_asset("leaflet.css")
    js = leaflet_asset("leaflet.js")
    html = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Саратов — дороги OSM highway=service</title>
<style>__LEAFLET_CSS__</style>
<style>
html,body,#map{height:100%;margin:0} #map{background:#f5f6f8}
.panel{position:absolute;z-index:1000;top:14px;left:14px;max-width:310px;background:#fff;
border:1px solid #d7dce3;border-radius:10px;padding:13px 16px;font:14px/1.42 system-ui,sans-serif;
box-shadow:0 3px 14px #25324722;color:#253247}
.panel h1{font-size:18px;line-height:1.2;margin:0 0 6px}.panel p{margin:6px 0}
.panel small{color:#657485}.panel button{margin:7px 6px 0 0;padding:6px 9px;border:1px solid #bcc8d6;
border-radius:6px;background:#f7f9fc;cursor:pointer}.key{display:inline-block;width:12px;height:5px;margin-right:6px;vertical-align:middle}
.legend{margin:7px 0 2px}.legend div{margin:3px 0}.leaflet-control-layers{font:13px system-ui,sans-serif}
@media(max-width:600px){.panel{max-width:210px;font-size:12px;padding:9px}.panel h1{font-size:15px}}
</style></head><body>
<div id="map"></div><div class="panel"><h1>Дороги Саратова: service</h1>
<p><b>__WAY_COUNT__</b> OSM-ways внутри муниципальной границы. На карте показан только класс <b>highway=service</b>.</p>
<div id="legend" class="legend"></div>
<button onclick="map.setView([51.5300,46.0347],12)">К центру</button>
<button onclick="map.fitBounds(border.getBounds())">Весь округ</button>
<p><small>OSM: 03.06.2026. Подтипы можно включать и выключать справа. Подложка по умолчанию отключена; её можно включить там же.</small></p>
</div>
<script>__LEAFLET_JS__</script>
<script>
const data=__DATA__, boundary=__BOUNDARY__, colors=__COLORS__, labels=__LABELS__, counts=__COUNTS__;
const canvas=L.canvas({padding:0.5});
const map=L.map('map',{preferCanvas:true,zoomControl:false}).setView([51.5300,46.0347],12);
L.control.zoom({position:'bottomright'}).addTo(map);
const blank=L.layerGroup().addTo(map);
const osm=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19,attribution:'&copy; OpenStreetMap contributors'});
const border=L.geoJSON(boundary,{style:{color:'#172b4d',weight:2,dashArray:'7 5',fill:false,opacity:0.7}}).addTo(map);
const layers={};
for(const key of Object.keys(colors)) layers[key]=L.layerGroup().addTo(map);
for(const [category,coords,tags] of data){
  const line=L.polyline(coords,{renderer:canvas,color:colors[category],weight:2,opacity:0.82,
    interactive:true});
  const parts=[`<b>OSM way ${tags.id}</b>`,tags.name?`Название: ${escapeHtml(tags.name)}`:'',
    `service=${escapeHtml(tags.service||'не указан')}`,tags.access?`access=${escapeHtml(tags.access)}`:'',
    tags.surface?`surface=${escapeHtml(tags.surface)}`:''].filter(Boolean);
  line.bindPopup(parts.join('<br>'));
  layers[category].addLayer(line);
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',
  '"':'&quot;',"'":'&#39;'}[c]));}
const overlays={'Граница Саратова':border};
for(const key of Object.keys(colors)) if(counts[key]) overlays[`${labels[key]} (${counts[key]})`]=layers[key];
L.control.layers({'Без подложки':blank,'OpenStreetMap':osm},overlays,{collapsed:false,position:'topright'}).addTo(map);
document.getElementById('legend').innerHTML=Object.keys(colors).filter(k=>counts[k]).map(k=>
  `<div><span class="key" style="background:${colors[k]}"></span>${labels[k]}: ${counts[k]}</div>`).join('');
</script></body></html>"""
    html = (html.replace("__LEAFLET_CSS__", css).replace("__LEAFLET_JS__", js)
            .replace("__DATA__", data).replace("__BOUNDARY__", boundary)
            .replace("__COLORS__", colors).replace("__LABELS__", labels)
            .replace("__COUNTS__", counts).replace("__WAY_COUNT__", f"{len(extractor.way_ids):,}".replace(",", " ")))
    OUTPUT.write_text(html)
    print(json.dumps({"output": str(OUTPUT), "ways": len(extractor.way_ids),
                      "line_parts": len(extractor.features), "categories": extractor.counts,
                      "html_mb": round(OUTPUT.stat().st_size / 1e6, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
