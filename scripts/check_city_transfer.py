"""Frozen Saratov-only OSM model, external city evaluation. Outputs isolated in reports/transfer_cities."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import h3, osmium
from road_accident_risk import osm_road_features as o, street_graph_v2 as g
from road_accident_risk import v2_eval_year as vy, catboost_baseline as cb
from road_accident_risk.paths import PROJECT_ROOT, OSM_VOLGA_PBF
OUT = PROJECT_ROOT / 'reports/transfer_cities'
# Fixed geographic windows; whole H3 cells selected by centroid. Not administrative boundaries.
BOXES = {'samara': (53.10,49.95,53.35,50.40), 'ulyanovsk': (54.15,48.15,54.45,48.65)}
def inside(lat,lon,b): return b[0] <= lat <= b[2] and b[1] <= lon <= b[3]
def buffered(b): return (b[0]-.04,b[1]-.06,b[2]+.04,b[3]+.06)
class Extract(osmium.SimpleHandler):
    def __init__(self, names):
        super().__init__(); self.seen_nodes=0; self.seen_ways=0; self.data={n: {'nodes':{},'ways':[],'points':[]} for n in names}
    def node(self,n):
        self.seen_nodes += 1
        if self.seen_nodes % 10000000 == 0: print('OSM nodes scanned',self.seen_nodes,flush=True)
        if not any(inside(n.location.lat,n.location.lon,buffered(BOXES[c])) for c in self.data): return
        tags=dict(n.tags)
        if not o.is_relevant_point_tags(tags): return
        for city,d in self.data.items():
            if inside(n.location.lat,n.location.lon,buffered(BOXES[city])):
                d['points'].append({'id':n.id,'lat':n.location.lat,'lon':n.location.lon,'tags':tags})
    def way(self,w):
        self.seen_ways += 1
        if self.seen_ways % 1000000 == 0: print('OSM ways scanned',self.seen_ways,flush=True)
        if w.tags.get('highway') not in o.MOTOR_ROAD_CLASSES: return
        coords=[(n.location.lat,n.location.lon) for n in w.nodes]
        for city,d in self.data.items():
            if not any(inside(a,b,buffered(BOXES[city])) for a,b in coords): continue
            d['ways'].append({'id':w.id,'tags':dict(w.tags),'nodes':[n.ref for n in w.nodes]})
            for n,(a,b) in zip(w.nodes,coords): d['nodes'][n.ref]={'id':n.ref,'lat':a,'lon':b}
def build(city,d):
    print('BUILD',city,len(d['ways']),flush=True)
    nodes,ways,points=d['nodes'],d['ways'],d['points']
    roads=o.build_raw_roads(ways,nodes); seg=o.build_h3_segments(ways,nodes)
    f=o.aggregate_h3_road_features(seg,o.build_graph_node_stats(ways,nodes,tagged_nodes=points),o.build_point_feature_stats(points))
    cells=f.h3
    j=o.aggregate_h3_junction_v2_features(o.build_junction_v2_node_stats(roads,o.traffic_signal_points(points)),cells)
    t=o.aggregate_h3_tags_v2_features(o.build_h3_tag_v2_segments(seg,roads),cells)
    # Explicitly use local extraction boundary; core has a generous buffer.
    old=g.data_edge_flags
    g.data_edge_flags=lambda lat,lng: old(lat,lng,bbox=buffered(BOXES[city]),padding_deg=0)
    # UTM 39N for target cities; same metric algorithm as Saratov's UTM 38N.
    g.UTM38_LON0=51.0
    n,e=g.build_street_graph_raw(roads)
    cn,ce=g.consolidate_street_graph(n,e,tol_m=g.CONSOLIDATION_TOL_M,roundabouts=True)
    blocks,_=g.build_street_blocks(cn,ce); raw,_=g.build_street_blocks(n,e)
    cv=g.curve_vertices_for_blocks(raw,n.loc[n.node_type=='junction',['x_utm','y_utm']].to_numpy(),dp_tol_m=g.DP_TOL_M)
    geo=g.aggregate_h3_geometry_v2_features(cells,blocks,cv,g.street_orientation_bins(roads),cn,n,o.build_point_nodes(points))
    g.data_edge_flags=old
    for part in (j,t,geo):
        f=f.drop(columns=[c for c in part if c!='h3' and c in f]).merge(part,on='h3',validate='one_to_one')
    f=f[f.h3.map(lambda c: inside(*h3.cell_to_latlng(c),BOXES[city]))].reset_index(drop=True)
    f.to_parquet(OUT/f'{city}_features.parquet',index=False)
    return f

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    missing=[c for c in BOXES if not (OUT/f'{c}_features.parquet').exists()]
    if missing:
        cache=OUT/'extracted_roads.json'
        if cache.exists(): data=json.loads(cache.read_text()); data={k:{**v,'nodes':{int(i):n for i,n in v['nodes'].items()}} for k,v in data.items()}
        else:
            print('Extracting buffered target cities from local Volga PBF',flush=True)
            ex=Extract(missing); ex.apply_file(str(OSM_VOLGA_PBF),locations=True,idx='sparse_mem_array'); data=ex.data
            cache.write_text(json.dumps(data))
        for c in missing: build(c,data[c])
    panel=vy.load_year_panel('road_geo_v2_geometry')
    original=vy.year_variant_features('road_geo_v2_geometry',panel)
    targets={c:pd.read_parquet(OUT/f'{c}_features.parquet') for c in BOXES}
    # Drop only terrain features, unavailable from OSM. Never choose using external outcomes.
    features=[f for f in original if f not in cb.ROAD_GEO_FEATURES]
    for c,f in targets.items():
        assert set(features)<=set(f), (c,set(features)-set(f))
        assert not set(f.h3) & set(panel.h3), 'Target and training cells overlap'
        assert f.h3.is_unique and (f.road_length_total_m>0).all()
    (OUT/'protocol.json').write_text(json.dumps({'train_city':'saratov','train_years':[2015,2022],'test_years':[2023,2025],'bbox':BOXES,'features':features,'excluded':list(set(original)-set(features)),'osm_snapshot':'2026-06-03','seeds':list(vy.SEEDS)},indent=2))
    tr=panel[panel.year<=2022]; caps=cb.fit_p99_clip_caps(tr,features); tr=cb.apply_p99_clipping(tr,caps)
    (OUT/'caps.json').write_text(json.dumps(caps,indent=2))
    targets['saratov']=panel[panel.year==2023].drop_duplicates('h3').copy()
    crashes=pd.read_parquet(PROJECT_ROOT/'parquet_output/crashes_slim.parquet',columns=['datetime','lat','lng'])
    crashes['year']=pd.to_datetime(crashes.datetime).dt.year
    counts={}; audit=[]
    for city,f in targets.items():
        b=BOXES.get(city,o.BBOX); a,bx,c,dx=buffered(b)
        cr=crashes[crashes.year.between(2015,2025)&crashes.lat.between(a,c)&crashes.lng.between(bx,dx)].copy()
        cr['h3']=[h3.latlng_to_cell(a,b,9) for a,b in zip(cr.lat,cr.lng)]
        core=cr[cr.h3.map(lambda x:inside(*h3.cell_to_latlng(x),b))]
        cr=cr[cr.h3.isin(f.h3)]
        cnt=cr.groupby(['h3','year']).size().unstack(fill_value=0).reindex(index=f.h3,columns=range(2015,2026),fill_value=0).fillna(0)
        counts[city]=cnt
        audit.append({'city':city,'n_cells':len(f),'core_test_crashes':int(core.year.between(2023,2025).sum()),'included_test_crashes':int(cnt[[2023,2024,2025]].sum().sum()),**{str(y):int(cnt[y].sum()) for y in range(2015,2026)}})
    pd.DataFrame(audit).to_csv(OUT/'data_audit.csv',index=False)
    rows=[]; predictions=[]
    for seed in vy.SEEDS:
        model=vy.make_model(loss='Tweedie:variance_power=1.3',seed=seed)
        model.fit(tr[features],tr[vy.TARGET_COUNT]); model.save_model(str(OUT/f'model_{seed}.cbm'))
        for city,f in targets.items():
            x=cb.apply_p99_clipping(f,caps); pred=model.predict(x[features],prediction_type='Exponent')
            y=counts[city][[2023,2024,2025]].sum(axis=1).to_numpy()
            rows.append({'city':city,'method':'osm_transfer','seed':seed,**vy.metrics_year(y,3*pred,is_count=True)})
            predictions.append(pd.DataFrame({'city':city,'h3':f.h3.to_numpy(),'seed':seed,'observed':y,'predicted':3*pred}))
            print(city,seed,rows[-1]['coverage_top_10pct'],flush=True)
    for city,f in targets.items():
        y=counts[city][[2023,2024,2025]].sum(axis=1).to_numpy()
        for name,s in [('major_road_length',f.major_road_length_m),('road_length',f.road_length_total_m),('local_history_reference',counts[city][list(range(2015,2023))].mean(axis=1)*3)]:
            rows.append({'city':city,'method':name,'seed':-1,**vy.metrics_year(y,np.asarray(s),is_count=name=='local_history_reference')})
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.concat(predictions).to_parquet(OUT/'predictions.parquet',index=False)
    print(pd.DataFrame(rows).groupby(['city','method'])[['coverage_top_1pct','coverage_top_5pct','coverage_top_10pct','calib_ratio']].mean().to_string(),flush=True)
if __name__=='__main__': main()
