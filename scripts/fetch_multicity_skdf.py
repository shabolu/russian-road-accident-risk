"""Fetch current SKDF normative-condition passports for all existing municipal cohorts."""
from pathlib import Path
import sys,json,time,urllib.request,urllib.parse,concurrent.futures,datetime
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/raw/skdf/route_coverage_20261106';OUT.mkdir(parents=True,exist_ok=True)
WFS='https://xn--d1aluo.xn--p1ai/api-geoserver/skdf_open/wfs';LAYER='skdf_open:lyr_road_conditions_road_normative_condition'
def get(params):
    req=urllib.request.Request(WFS+'?'+urllib.parse.urlencode(params),headers={'User-Agent':'road-accident-risk/research'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)
def fetch(city):
    path=OUT/(city['slug']+'.geojson')
    if path.exists():print('CACHED',city['slug'],flush=True);return
    a,b,c,d=city['extract_bbox'];cql=f"BBOX(geom,{b},{a},{d},{c},'EPSG:4326')";features=[];start=0;matched=None
    while matched is None or start<matched:
        page=get({'service':'WFS','version':'2.0.0','request':'GetFeature','typeNames':LAYER,'outputFormat':'application/json','srsName':'EPSG:4326','CQL_FILTER':cql,'count':2000,'startIndex':start,'sortBy':'id'})
        if matched is None:matched=int(page['numberMatched'])
        f=page['features']
        if not f:raise RuntimeError(f'Empty page {city["slug"]} {start}/{matched}')
        features.extend(f);start+=len(f);print('FETCH',city['slug'],start,'/',matched,flush=True);time.sleep(.3)
    ids=[f['properties']['id'] for f in features];assert len(set(ids))==len(ids)==matched,(city['slug'],len(ids),matched)
    out={'type':'FeatureCollection','features':features,'numberMatched':matched,'provenance':{'endpoint':WFS,'layer':LAYER,'cql':cql,'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'srs':'EPSG:4326'}}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(out,ensure_ascii=False));temp.replace(path)
    print('DONE FETCH',city['slug'],matched,flush=True)
def main():
    cities=json.loads((ROOT/'reports/multicity_350k/cities.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(fetch,cities))
    print('ALL 14 SKDF DOWNLOADS COMPLETE',flush=True)
if __name__=='__main__':main()
