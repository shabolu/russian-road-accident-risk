"""Cell/year counts and coordinate-quality audit, no snapping or deletion of source crashes."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,h3,shapely
from shapely.geometry import shape
from pyproj import Transformer
from scipy.spatial import cKDTree
from road_accident_risk import osm_road_features as o
from road_accident_risk.paths import PROJECT_ROOT
OUT=PROJECT_ROOT/'reports/multicity_350k'
def main():
    cities=json.loads((OUT/'cities.json').read_text());allcounts=[];allcr=[];audit=[];profiles=[]
    crashes=pd.read_parquet(PROJECT_ROOT/'parquet_output/crashes_slim.parquet',columns=['datetime','lat','lng','accident_type','dead_count','injured_count'])
    crashes['year']=pd.to_datetime(crashes.datetime).dt.year;crashes=crashes[crashes.year.between(2015,2025)]
    for c in cities:
        slug=c['slug'];f=pd.read_parquet(OUT/'features'/f'{slug}.parquet');geom=shape(c['geometry']);a,b,d,e=c['bbox']
        z=crashes[crashes.lat.between(a-.01,d+.01)&crashes.lng.between(b-.01,e+.01)].copy();z['h3']=[h3.latlng_to_cell(x,y,9) for x,y in zip(z.lat,z.lng)]
        cent={h:h3.cell_to_latlng(h) for h in z.h3.unique()};inside={h:bool(geom.intersects(shapely.Point(y,x))) for h,(x,y) in cent.items()}
        z=z[z.h3.map(inside)].copy();z['city']=slug;z['in_road_cell']=z.h3.isin(f.h3)
        z['actual_point_in_boundary']=shapely.intersects_xy(geom,z.lng.to_numpy(),z.lat.to_numpy())
        z['center_distance_km']=[o.haversine_m(x,y,c['lat'],c['lon'])/1000 for x,y in zip(z.lat,z.lng)]
        # Sample each road segment at <=20 m. Nearest sample is an upper approximation
        # of point-to-polyline distance, with <=10 m sampling uncertainty.
        roads=pd.read_parquet(OUT/'cache'/f'{slug}_roads.parquet',columns=['coords_json'])
        pts=[]
        for cs in roads.coords_json:
            xy=json.loads(cs)
            for (la,lo),(lb,lp) in zip(xy,xy[1:]):
                k=max(1,math.ceil(o.haversine_m(la,lo,lb,lp)/20))
                v=np.linspace(0,1,k+1);pts.append(np.column_stack((lo+(lp-lo)*v,la+(lb-la)*v)))
        pp=np.concatenate(pts);zone=int((c['lon']+180)//6)+1;proj=Transformer.from_crs('EPSG:4326',f'EPSG:{32600+zone}',always_xy=True)
        rx,ry=proj.transform(pp[:,0],pp[:,1]);tree=cKDTree(np.column_stack((rx,ry)));cx,cy=proj.transform(z.lng.to_numpy(),z.lat.to_numpy());z['road_distance_sample_m']=tree.query(np.column_stack((cx,cy)),workers=2)[0]
        z['same_coordinate_count']=z.groupby(['lat','lng']).datetime.transform('size')
        # Decimal rounding diagnostic: close to a 0.001 degree grid (~60–110 m).
        z['rounded_3dec']=np.isclose(z.lat*1000,np.round(z.lat*1000),atol=1e-5,rtol=0)&np.isclose(z.lng*1000,np.round(z.lng*1000),atol=1e-5,rtol=0)
        for subset,m in [('municipality',np.ones(len(z),bool)),('radius15km',z.center_distance_km<=15)]:
            for year,q in z[m].groupby('year'):
                dist=q.road_distance_sample_m
                audit.append({'city':slug,'subset':subset,'year':int(year),'n_crashes':len(q),'outside_road_cells':int((~q.in_road_cell).sum()),'outside_share':(~q.in_road_cell).mean(),'outside_actual_boundary':int((~q.actual_point_in_boundary).sum()),'distance_p50':dist.median(),'distance_p90':dist.quantile(.9),'distance_p99':dist.quantile(.99),'far_50m_share':(dist>50).mean(),'far_100m_share':(dist>100).mean(),'far_250m_share':(dist>250).mean(),'rounded_3dec_share':q.rounded_3dec.mean(),'unique_coordinates':q[['lat','lng']].drop_duplicates().shape[0]})
        cnt=z[z.in_road_cell].groupby(['h3','year']).size().unstack(fill_value=0).reindex(index=f.h3,columns=range(2015,2026),fill_value=0).fillna(0).astype('int32');cnt.columns=[str(y) for y in cnt.columns];cnt=cnt.reset_index();cnt['city']=slug
        assert int(cnt[[str(y) for y in range(2015,2026)]].to_numpy().sum())==int(z.in_road_cell.sum())
        allcounts.append(cnt);allcr.append(z)
        area=shapely.transform(geom,lambda xy:np.column_stack(proj.transform(xy[:,0],xy[:,1]))).area/1e6
        profiles.append({'city':slug,'name':c['name'],'population':c['population'],'area_km2':area,'n_road_cells':len(f),'road_km':f.road_length_total_m.sum()/1000,'street_km':f.street_length_m.sum()/1000,'road_km_per_km2':f.road_length_total_m.sum()/1000/area,'n_test':int(cnt[['2023','2024','2025']].to_numpy().sum()),'n_early':int(cnt[['2015','2016']].to_numpy().sum())})
        print('AUDIT',slug,len(f),'cells',len(z),'crashes',flush=True)
    count=pd.concat(allcounts,ignore_index=True);assert not count.h3.duplicated().any(),'Overlapping municipal cell cohorts'
    count.to_parquet(OUT/'counts.parquet',index=False);pd.concat(allcr,ignore_index=True).to_parquet(OUT/'crashes.parquet',index=False)
    pd.DataFrame(audit).to_csv(OUT/'coordinate_audit.csv',index=False);pd.DataFrame(profiles).to_csv(OUT/'city_profiles.csv',index=False)
    print('DATA COMPLETE',flush=True)
if __name__=='__main__':main()
