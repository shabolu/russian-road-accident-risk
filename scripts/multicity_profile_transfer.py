"""Post-hoc association of outcome-free city profiles with existing donor-model results.
Closest-three score is mean standalone donor performance, not a fitted group or prediction ensemble.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/analysis/multicity_350k'
city=json.loads((ROOT/'reports/multicity_350k/cities.json').read_text());order=[c['slug'] for c in city]
def main():
    distances={};pairs=[]
    for kind in ['morphology','mapping_quality']:
        x=pd.read_csv(OUT/f'grouping_axes_{kind}.csv',index_col=0).reindex(order);a=np.clip(RobustScaler().fit_transform(x.fillna(x.median())),-5,5)
        d=pd.DataFrame(squareform(pdist(a)),index=order,columns=order);distances[kind]=d
        pairs.extend({'profile':kind,'a':a,'b':b,'distance':d.loc[a,b]} for i,a in enumerate(order) for b in order[i+1:])
    shifts=pd.read_csv(OUT/'pairwise_shift_summary.csv');d=pd.DataFrame(0.,index=order,columns=order)
    for r in shifts.itertuples():d.loc[r.a,r.b]=d.loc[r.b,r.a]=r.mean_ks
    distances['all_features_ks']=d
    m=pd.read_csv(OUT/'metrics_summary.csv');m=m[(m.kind=='local')&(m.window=='all_2015_2022')&(m.subset=='municipality')].pivot(index='source',columns='target',values='coverage_top_10pct').reindex(index=order,columns=order)
    rows=[]
    for kind,d in distances.items():
        for target in order:
            donors=[c for c in order if c!=target];v=m.loc[donors,target];distance=d.loc[donors,target];near=distance.nsmallest(3).index
            rows.append({'profile':kind,'target':target,'similarity_transfer_spearman':spearmanr(-distance,v).statistic,'closest3_mean_coverage':v.loc[near].mean(),'all13_mean_coverage':v.mean(),'delta_closest3_pp':100*(v.loc[near].mean()-v.mean()),'nearest_city':distance.idxmin(),'nearest3':','.join(near)})
    pd.DataFrame(rows).to_csv(OUT/'profile_transfer_association.csv',index=False)
    pd.DataFrame(pairs).sort_values(['profile','distance']).to_csv(OUT/'profile_pair_distances.csv',index=False)
    print(pd.DataFrame(rows).groupby('profile').agg(median_rho=('similarity_transfer_spearman','median'),mean_delta_pp=('delta_closest3_pp','mean')))
if __name__=='__main__':main()
