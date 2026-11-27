"""Length-aware OSM/SKDF attribute coverage, conservative carriageway handling.
This audits attribute availability, not a calibrated speed estimator or a routability validator.
"""
from __future__ import annotations
import sys,json,re,argparse,subprocess,concurrent.futures,time
from pathlib import Path
import numpy as np,pandas as pd,shapely
from shapely.geometry import shape,LineString
from pyproj import Transformer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from road_accident_risk.osm_road_features import MOTOR_ROAD_CLASSES,parse_oneway_v2,MAXSPEED_ZONE_DEFAULTS
from road_accident_risk.skdf_v2 import COATING_CLASS
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reports/multicity_350k';OUT=ROOT/'reports/route_proxy_coverage';RAW=ROOT/'data/raw/skdf/route_coverage_20261106'
SURFACES={'asphalt','concrete','concrete:plates','concrete:lanes','paved','cement','compacted','fine_gravel','paving_stones','metal','bricks','grass','wood','sett','grass_paver','gravel','unpaved','ground','dirt','pebblestone','tartan','cobblestone','clay','earth','stone','rocky','sand','laterite','mud','ice','snow','chipseal','unsealed','rock','woodchips','brick','cobblestone:flattened'}
SMOOTH={'excellent','good','intermediate','bad','very_bad','horrible','very_horrible','impassable'}
COATING_KNOWN=set(COATING_CLASS)-{2043912,2043913} # unconfirmed legend codes excluded
SPEEDS=MAXSPEED_ZONE_DEFAULTS
CRITERIA=['base_class','osm_speed','osm_surface','osm_lanes','osm_width','osm_smoothness','osm_full4','skdf_matched','skdf_unambiguous','combined_speed','combined_surface_coarse','combined_lanes','combined_width','combined_narrow','combined_any_extra','combined_full4_coarse','combined_full4_detailed','potential_full4_coarse','combined_full4_and_smoothness']
def numeric(x,low,high,integer=False,unit=False,speed=False):
    if x is None:return np.nan
    if isinstance(x,bool):return np.nan
    s=str(x).strip().lower().replace(',','.')
    if speed and s in SPEEDS:return SPEEDS[s]
    pattern=r'\d+(?:\.\d+)?(?:\s*m)?' if unit else r'\d+(?:\.\d+)?'
    if speed and s.endswith('mph'):
        try:v=float(s[:-3].strip())*1.609344
        except ValueError:return np.nan
    elif re.fullmatch(pattern,s):v=float(s.removesuffix('m').strip())
    else:return np.nan
    return v if low<=v<=high and (not integer or float(v).is_integer()) else np.nan

def lines(g):
    if g.is_empty:return
    if g.geom_type=='LineString':yield g
    elif hasattr(g,'geoms'):
        for part in g.geoms:yield from lines(part)
def attributes(tags):
    speed=numeric(tags.get('maxspeed'),5,150,speed=True)
    if np.isnan(speed):
        a=numeric(tags.get('maxspeed:forward'),5,150,speed=True);b=numeric(tags.get('maxspeed:backward'),5,150,speed=True)
        if np.isfinite(a) and np.isfinite(b):speed=min(a,b)
    lanes=numeric(tags.get('lanes'),1,20,integer=True)
    if np.isnan(lanes):
        a=numeric(tags.get('lanes:forward'),1,20,integer=True);b=numeric(tags.get('lanes:backward'),1,20,integer=True)
        if np.isfinite(a) and np.isfinite(b) and a+b<=20:lanes=a+b
    return {'speed':speed,'lanes':lanes,'width':numeric(tags.get('width'),1,100,unit=True),'surface':str(tags.get('surface','')).lower() in SURFACES,'smoothness':str(tags.get('smoothness','')).lower() in SMOOTH,'oneway':parse_oneway_v2(tags)}
def analyze(slug):
    dest=OUT/f'{slug}_summary.csv'
    city_outputs = [dest, OUT/f'{slug}_joint_completeness.csv', OUT/f'{slug}_by_class.csv',
                    OUT/f'{slug}_metadata.json', OUT/'ways'/f'{slug}.parquet']
    if all(path.exists() for path in city_outputs):
        return
    c=next(c for c in json.loads((BASE/'cities.json').read_text()) if c['slug']==slug);t0=time.time();zone=int((c['lon']+180)//6)+1;tr=Transformer.from_crs(4326,32600+zone,always_xy=True)
    def project(g):return shapely.transform(g,lambda xy:np.column_stack(tr.transform(xy[:,0],xy[:,1])))
    polygon=project(shape(c['geometry']));roads=pd.read_parquet(BASE/'cache'/f'{slug}_roads.parquet');ways=[];samples=[]
    for r in roads.itertuples():
        coords=np.array(json.loads(r.coords_json));x,y=tr.transform(coords[:,1],coords[:,0]);g=LineString(np.column_stack((x,y))).intersection(polygon)
        if g.is_empty or g.length<=1e-6:continue
        i=len(ways);tags=json.loads(r.tags_json);ways.append({'osm_id':r.osm_id,'highway':r.highway,'name':r.name,'length_m':g.length,**attributes(tags)})
        for line in lines(g):
            xy=np.array(line.coords)[:,:2];d=np.diff(xy,axis=0);length=np.linalg.norm(d,axis=1)
            for a,dd,ll in zip(xy,d,length):
                if ll<=1e-6:continue
                n=max(1,int(np.ceil(ll/20)));f=(np.arange(n)+.5)/n;pp=a+f[:,None]*dd
                samples.append(np.column_stack((np.full(n,i),pp,np.full(n,np.degrees(np.arctan2(dd[0],dd[1]))%180),np.full(n,ll/n))))
    w=pd.DataFrame(ways);p=np.concatenate(samples);wi=p[:,0].astype(int);points=shapely.points(p[:,1:3]);weights=p[:,4]
    assert w.osm_id.is_unique and np.allclose(np.bincount(wi,weights=weights,minlength=len(w)),w.length_m,atol=1e-5)
    raw=json.loads((RAW/f'{slug}.geojson').read_text());passport=[];geoms=[];seg_rec=[];angles=[];clipping=polygon.buffer(50)
    for ft in raw['features']:
        if not ft.get('geometry'):continue
        g=project(shape(ft['geometry'])).intersection(clipping)
        if g.is_empty:continue
        a=ft['properties'];rec=len(passport);speed=numeric(a.get('speed_limit'),5,150);lanes=numeric(a.get('lanes'),1,20,integer=True);width=numeric(a.get('roadway_width'),1,100)
        coating=a.get('pavement_kind_gid');passport.append({'id':a.get('id'),'road_id':str(a.get('road_id') or a.get('id')),'speed':speed,'lanes':lanes,'width':width,'surface_coarse':coating in COATING_KNOWN,'signature':str((speed,lanes,width,coating))})
        for line in lines(g):
            xy=np.array(line.coords)[:,:2]
            for a,b in zip(xy,xy[1:]):
                d=b-a
                if np.linalg.norm(d)<=1e-6:continue
                geoms.append(LineString([a,b]));seg_rec.append(rec);angles.append(np.degrees(np.arctan2(d[0],d[1]))%180)
    n=len(p);best=np.full(n,-1,dtype=int);ambiguous=np.zeros(n,dtype=bool);dist=np.full(n,np.nan)
    sk=pd.DataFrame(passport);sg=np.array(geoms,dtype=object);sr=np.array(seg_rec);angle=np.array(angles)
    if len(sg):
        tree=shapely.STRtree(sg);roadids=sk.road_id.to_numpy();signatures=sk.signature.to_numpy()
        for start in range(0,n,20000):
            end=min(n,start+20000);q,s=tree.query(points[start:end],predicate='dwithin',distance=20)
            if not len(q):continue
            err=abs(angle[s]-p[start+q,3]);err=np.minimum(err,180-err);keep=err<=25;q=q[keep];s=s[keep]
            if not len(q):continue
            distance=shapely.distance(points[start+q],sg[s]);order=np.lexsort((s,distance,q));q=q[order];s=s[order];distance=distance[order]
            _,first=np.unique(q,return_index=True);bq=q[first];br=sr[s[first]];best[start+bq]=br;dist[start+bq]=distance[first]
            close=distance<=dist[start+q]+5;altrec=sr[s];different=(roadids[altrec]!=roadids[best[start+q]])|(signatures[altrec]!=signatures[best[start+q]])
            ambiguous[start+np.unique(q[close&different])]=True
    matched=best>=0;safe=matched&~ambiguous
    if len(sk):
        idx=np.maximum(best,0);sv={k:sk[k].to_numpy()[idx] for k in ['speed','lanes','width','surface_coarse']}
    else:sv={k:np.full(n,np.nan) for k in ['speed','lanes','width']};sv['surface_coarse']=np.zeros(n,bool)
    ow=w.oneway.to_numpy()[wi];osp=np.isfinite(w.speed.to_numpy()[wi]);osl=np.isfinite(w.lanes.to_numpy()[wi]);osw=np.isfinite(w.width.to_numpy()[wi]);oss=w.surface.to_numpy()[wi];osm=w.smoothness.to_numpy()[wi]
    cs=osp|(safe&np.isfinite(sv['speed']));cf=oss|(safe&sv['surface_coarse']);cl=osl|(safe&~ow&np.isfinite(sv['lanes']));cw=osw|(safe&~ow&np.isfinite(sv['width']))
    potential=(osp|(matched&np.isfinite(sv['speed'])))&(oss|(matched&sv['surface_coarse']))&(osl|(matched&np.isfinite(sv['lanes'])))&(osw|(matched&np.isfinite(sv['width'])))
    flags={'base_class':np.ones(n,bool),'osm_speed':osp,'osm_surface':oss,'osm_lanes':osl,'osm_width':osw,'osm_smoothness':osm,'osm_full4':osp&oss&osl&osw,'skdf_matched':matched,'skdf_unambiguous':safe,'combined_speed':cs,'combined_surface_coarse':cf,'combined_lanes':cl,'combined_width':cw,'combined_narrow':cl&cw,'combined_any_extra':cs|cf|cl|cw|osm,'combined_full4_coarse':cs&cf&cl&cw,'combined_full4_detailed':cs&oss&cl&cw,'potential_full4_coarse':potential,'combined_full4_and_smoothness':cs&cf&cl&cw&osm}
    # Joint availability is needed to count missing fields without double counting overlaps.
    known_count=cs.astype(int)+cf.astype(int)+cl.astype(int)+cw.astype(int)
    joint=[{'city':slug,'name':c['name'],'known_features':k,'missing_features':4-k,'known_length_km':float(weights[known_count==k].sum()/1000),'n_samples':int((known_count==k).sum())} for k in range(5)]
    assert np.isclose(sum(r['known_length_km'] for r in joint),weights.sum()/1000)
    pd.DataFrame(joint).to_csv(OUT/f'{slug}_joint_completeness.csv',index=False)
    total=weights.sum();rows=[];classrows=[]
    for name,flag in flags.items():
        known=np.bincount(wi,weights=weights*flag,minlength=len(w));share=known/w.length_m.to_numpy();w[name+'_length_share']=share
        rows.append({'city':slug,'name':c['name'],'criterion':name,'n_osm_ways':len(w),'total_road_km':total/1000,'known_km':known.sum()/1000,'length_share':known.sum()/total,'ways_any':int((share>0).sum()),'ways_ge90pct':int((share>=.9).sum()),'ways_full':int(np.isclose(share,1,atol=1e-9,rtol=0).sum())})
        for cls,group in w.groupby('highway'):
            ids=group.index.to_numpy();L=group.length_m.sum();classrows.append({'city':slug,'highway':cls,'criterion':name,'n_ways':len(group),'total_km':L/1000,'known_km':known[ids].sum()/1000,'length_share':known[ids].sum()/L,'ways_ge90pct':int((share[ids]>=.9).sum())})
    w.to_parquet(OUT/'ways'/f'{slug}.parquet',index=False);pd.DataFrame(classrows).to_csv(OUT/f'{slug}_by_class.csv',index=False)
    meta={'city':slug,'name':c['name'],'fetched_features':len(raw['features']),'features_intersecting_boundary50m':len(sk),'segment_count':len(sg),'n_samples':n,'ambiguous_length_share':weights[ambiguous].sum()/total,'oneway_length_share':weights[ow].sum()/total,'matched_length_share':weights[matched].sum()/total,'skdf_provenance':raw['provenance'],'osm_snapshot':'2026-06-03','elapsed_s':round(time.time()-t0)}
    (OUT/f'{slug}_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));pd.DataFrame(rows).to_csv(dest,index=False);print('MATCH DONE',slug,len(w),'ways',round(total/1000),'km',round(time.time()-t0),'sec',flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city');args=ap.parse_args();(OUT/'ways').mkdir(parents=True,exist_ok=True)
    if args.city:analyze(args.city);return
    cities=json.loads((BASE/'cities.json').read_text())
    def worker(c):
        path=OUT/f"{c['slug']}.log"
        with path.open('w') as log:r=subprocess.run([sys.executable,__file__,'--city',c['slug']],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(str(path))
        print(path.read_text().splitlines()[-1] if path.stat().st_size else 'CACHED '+c['slug'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,cities))
    pd.concat([pd.read_csv(OUT/f"{c['slug']}_summary.csv") for c in cities]).to_csv(OUT/'coverage.csv',index=False)
    pd.concat([pd.read_csv(OUT/f"{c['slug']}_by_class.csv") for c in cities]).to_csv(OUT/'by_class.csv',index=False)
    (OUT/'metadata.json').write_text(json.dumps([json.loads((OUT/f"{c['slug']}_metadata.json").read_text()) for c in cities],ensure_ascii=False,indent=2))
    print('ALL ROUTE ATTRIBUTE COVERAGE COMPLETE',flush=True)
if __name__=='__main__':main()
