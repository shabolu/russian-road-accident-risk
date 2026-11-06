"""Reproducible OSM extraction for the 14 >350k Volga cities; no outcome-based boundaries."""
from __future__ import annotations
import argparse,json,sys,time,subprocess,concurrent.futures
from pathlib import Path
import numpy as np,pandas as pd,osmium,h3,shapely
from shapely.geometry import shape,mapping
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from road_accident_risk import osm_road_features as o,street_graph_v2 as g
from road_accident_risk.paths import PROJECT_ROOT,OSM_VOLGA_PBF
OUT=PROJECT_ROOT/'reports/multicity_350k'
CITY_IDS={'Казань':367666,'Нижний Новгород':335495,'Уфа':398551,'Самара':287507,'Пермь':1084793,'Саратов':2725551,'Тольятти':1345135,'Ижевск':954515,'Ульяновск':1760251,'Набережные Челны':367728,'Оренбург':1398615,'Чебоксары':1810276,'Пенза':1846781,'Киров':382776}
SLUGS=dict(zip(CITY_IDS,['kazan','nizhny_novgorod','ufa','samara','perm','saratov','tolyatti','izhevsk','ulyanovsk','naberezhnye_chelny','orenburg','cheboksary','penza','kirov']))
def config():
    return json.loads((OUT/'cities.json').read_text())
def boundaries():
    if (OUT/'cities.json').exists():return
    ids=list(CITY_IDS.values());factory=osmium.geom.GeoJSONFactory();geoms={}
    print('Assembling 14 municipal boundaries',flush=True)
    proc=osmium.FileProcessor(str(OSM_VOLGA_PBF)).with_locations('sparse_mem_array').with_areas(osmium.filter.IdFilter(ids)).with_filter(osmium.filter.EntityFilter(osmium.osm.AREA)).with_filter(osmium.filter.IdFilter([2*i+1 for i in ids]))
    for a in proc:
        if not a.from_way() and a.orig_id() in ids:
            geoms[a.orig_id()]=json.loads(factory.create_multipolygon(a));print('Boundary',a.orig_id(),a.tags.get('name'),flush=True)
    assert set(geoms)==set(ids),(set(ids)-set(geoms))
    inventory=pd.read_csv(PROJECT_ROOT/'reports/city_inventory/cities_over_350k.csv',sep=';').set_index('Город')
    cities=[]
    for name,rel in CITY_IDS.items():
        geom=shape(geoms[rel]);assert geom.is_valid and not geom.is_empty
        lon,lat=float(inventory.loc[name,'Долгота центра']),float(inventory.loc[name,'Широта центра'])
        assert geom.covers(shapely.Point(lon,lat)),name
        xmin,ymin,xmax,ymax=geom.bounds;padlat=3000/110540;padlon=3000/(111320*np.cos(np.radians(lat)))
        cities.append({'name':name,'slug':SLUGS[name],'relation':rel,'population':int(inventory.loc[name,'Население']),'lat':lat,'lon':lon,'bbox':[ymin,xmin,ymax,xmax],'extract_bbox':[ymin-padlat,xmin-padlon,ymax+padlat,xmax+padlon],'geometry':geoms[rel]})
    (OUT/'cities.json').write_text(json.dumps(cities,ensure_ascii=False,indent=2))
    (OUT/'boundaries.geojson').write_text(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':{k:v for k,v in c.items() if k!='geometry'},'geometry':c['geometry']} for c in cities]},ensure_ascii=False))
def extract():
    cities=config()
    if all((OUT/'cache'/f"{c['slug']}_roads.parquet").exists() and (OUT/'cache'/f"{c['slug']}_points.json").exists() for c in cities):return
    boxes=np.array([c['extract_bbox'] for c in cities]);rows=[[] for _ in cities];points=[[] for _ in cities]
    print('OSM: motor ways (C++ tag filter, location cache)',flush=True);start=time.time();n=0
    proc=osmium.FileProcessor(str(OSM_VOLGA_PBF)).with_locations('sparse_mem_array').with_filter(osmium.filter.EntityFilter(osmium.osm.WAY)).with_filter(osmium.filter.TagFilter(*[('highway',v) for v in sorted(o.MOTOR_ROAD_CLASSES)]))
    for w in proc:
        n+=1
        if n%100000==0:print('Motor ways scanned',n,'seconds',round(time.time()-start),flush=True)
        cs=np.array([(r.location.lat,r.location.lon) for r in w.nodes])
        if len(cs)<2:continue
        mn,mx=cs.min(axis=0),cs.max(axis=0)
        hits=np.flatnonzero((boxes[:,0]<=mx[0])&(boxes[:,2]>=mn[0])&(boxes[:,1]<=mx[1])&(boxes[:,3]>=mn[1]))
        if not len(hits):continue
        nodes={r.ref:{'lat':float(x[0]),'lon':float(x[1])} for r,x in zip(w.nodes,cs)}
        row=o.row_base_for_way({'id':w.id,'nodes':[r.ref for r in w.nodes],'tags':dict(w.tags)},nodes)
        if row:
            for i in hits:rows[i].append(row)
    for c,rr in zip(cities,rows):pd.DataFrame(rr).to_parquet(OUT/'cache'/f"{c['slug']}_roads.parquet",index=False)
    print('OSM: relevant points (C++ key filter)',flush=True)
    proc=osmium.FileProcessor(str(OSM_VOLGA_PBF)).with_filter(osmium.filter.EntityFilter(osmium.osm.NODE)).with_filter(osmium.filter.KeyFilter('highway','railway','public_transport','traffic_calming'))
    for node in proc:
        a,b=node.location.lat,node.location.lon
        hits=np.flatnonzero((boxes[:,0]<=a)&(boxes[:,2]>=a)&(boxes[:,1]<=b)&(boxes[:,3]>=b))
        if not len(hits):continue
        tags=dict(node.tags)
        if o.is_relevant_point_tags(tags):
            for i in hits:points[i].append({'id':node.id,'lat':a,'lon':b,'tags':tags})
    for c,pp in zip(cities,points):(OUT/'cache'/f"{c['slug']}_points.json").write_text(json.dumps(pp,ensure_ascii=False))
    print('EXTRACT COMPLETE seconds',round(time.time()-start),flush=True)
def build(slug):
    c=next(c for c in config() if c['slug']==slug);dest=OUT/'features'/f'{slug}.parquet'
    if dest.exists():return
    t0=time.time();roads=pd.read_parquet(OUT/'cache'/f'{slug}_roads.parquet');points=json.loads((OUT/'cache'/f'{slug}_points.json').read_text())
    nodes={};ways=[]
    for r in roads.itertuples():
        ids=json.loads(r.nodes_json);cs=json.loads(r.coords_json)
        for i,(a,b) in zip(ids,cs):nodes[i]={'lat':a,'lon':b}
        ways.append({'id':r.osm_id,'nodes':ids,'tags':json.loads(r.tags_json)})
    print('BUILD',slug,'roads',len(roads),flush=True)
    seg=o.build_h3_segments(ways,nodes);f=o.aggregate_h3_road_features(seg,o.build_graph_node_stats(ways,nodes,tagged_nodes=points),o.build_point_feature_stats(points));cells=f.h3
    j=o.aggregate_h3_junction_v2_features(o.build_junction_v2_node_stats(roads,o.traffic_signal_points(points)),cells)
    tags=o.aggregate_h3_tags_v2_features(o.build_h3_tag_v2_segments(seg,roads),cells)
    old=g.data_edge_flags;oldlon=g.UTM38_LON0
    try:
        g.data_edge_flags=lambda lat,lng:old(lat,lng,bbox=c['extract_bbox'],padding_deg=0)
        g.UTM38_LON0=(int((c['lon']+180)//6)+1)*6-183
        n,e=g.build_street_graph_raw(roads);cn,ce=g.consolidate_street_graph(n,e,tol_m=g.CONSOLIDATION_TOL_M,roundabouts=True)
        blocks,_=g.build_street_blocks(cn,ce);raw,_=g.build_street_blocks(n,e)
        cv=g.curve_vertices_for_blocks(raw,n.loc[n.node_type=='junction',['x_utm','y_utm']].to_numpy(),dp_tol_m=g.DP_TOL_M)
        geo=g.aggregate_h3_geometry_v2_features(cells,blocks,cv,g.street_orientation_bins(roads),cn,n,o.build_point_nodes(points))
    finally:g.data_edge_flags=old;g.UTM38_LON0=oldlon
    for part in (j,tags,geo):f=f.drop(columns=[x for x in part if x!='h3' and x in f]).merge(part,on='h3',validate='one_to_one')
    coords=np.array([h3.cell_to_latlng(h) for h in f.h3]);mask=shapely.intersects_xy(shape(c['geometry']),coords[:,1],coords[:,0]);f=f[mask].reset_index(drop=True)
    xy=np.array([h3.cell_to_latlng(h) for h in f.h3]);f['lat']=xy[:,0];f['lon']=xy[:,1]
    f['center_distance_km']=[o.haversine_m(a,b,c['lat'],c['lon'])/1000 for a,b in xy];f['city']=slug
    features=json.loads((PROJECT_ROOT/'reports/transfer_cities/protocol.json').read_text())['features']
    assert f.h3.is_unique and (f.road_length_total_m>0).all() and set(features)<=set(f)
    assert not np.isinf(f[features].to_numpy(dtype=float)).any()
    f.to_parquet(dest,index=False)
    print('DONE',slug,'cells',len(f),'seconds',round(time.time()-t0),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city');ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    for d in ['cache','features','models']:(OUT/d).mkdir(parents=True,exist_ok=True)
    if args.city:build(args.city);return
    boundaries();extract()
    def worker(c):
        logfile=OUT/f"build_{c['slug']}.log"
        with logfile.open('w') as log:
            r=subprocess.run([sys.executable,__file__,'--city',c['slug']],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f"{c['slug']} failed: {logfile}")
        print(logfile.read_text().splitlines()[-1],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(worker,config()))
    print('ALL 14 FEATURE TABLES COMPLETE',flush=True)
if __name__=='__main__':main()
