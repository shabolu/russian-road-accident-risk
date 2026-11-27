"""Equal total city contribution vs cached row-uniform LOCO baseline. No target-city fitting."""
from __future__ import annotations
import argparse,json,sys,subprocess,concurrent.futures,time
from pathlib import Path
import numpy as np,pandas as pd
from multicity_models import load,evaluate,FEATURES,cb,vy,OUT as BASE
OUT=BASE/'equal_city_weights'
SEEDS=[42,7,123];YEARS=list(range(2015,2023))
def task(target):
    dest=OUT/f'{target}_metrics.csv'
    city_outputs = [dest, OUT/f'{target}_caps.json', OUT/f'{target}_weights.csv',
                    OUT/f'{target}_predictions.parquet']
    if all(path.exists() for path in city_outputs):
        return
    _,frames=load();donors={c:f for c,f in frames.items() if c!=target}
    assert not set(frames[target].h3).intersection(pd.concat(list(donors.values())).h3)
    tr=pd.concat([f[FEATURES].assign(city=c,target=f[str(y)].to_numpy()) for c,f in donors.items() for y in YEARS],ignore_index=True)
    counts=tr.groupby('city').size();weight=tr.city.map(len(tr)/(len(counts)*counts)).to_numpy()
    totals=pd.Series(weight).groupby(tr.city).sum();assert np.allclose(totals,totals.iloc[0]) and np.isclose(weight.mean(),1)
    audit=pd.DataFrame({'city':counts.index,'n_train_rows':counts.values,'row_weight':len(tr)/(len(counts)*counts.values),'total_weight':totals.reindex(counts.index).values});audit['target']=target;audit.to_csv(OUT/f'{target}_weights.csv',index=False)
    caps=cb.fit_p99_clip_caps(tr,FEATURES);tr=cb.apply_p99_clipping(tr,caps);f=frames[target];x=cb.apply_p99_clipping(f,caps)[FEATURES]
    (OUT/f'{target}_caps.json').write_text(json.dumps(caps,indent=2))
    rows=[];preds=[];t0=time.time()
    for seed in SEEDS:
        model=vy.make_model(loss='Tweedie:variance_power=1.3',seed=seed,params={'thread_count':3})
        model.fit(tr[FEATURES],tr.target.astype(int),sample_weight=weight)
        pred=model.predict(x,prediction_type='Exponent')
        rows.extend(evaluate(f,pred,{'kind':'loco_equal_city','source':'other_13_cities','target':target,'window':'all_2015_2022','seed':seed}))
        preds.append(pd.DataFrame({'target':target,'seed':seed,'h3':f.h3.to_numpy(),'observed':f[['2023','2024','2025']].sum(axis=1).to_numpy(),'predicted':3*pred,'center_distance_km':f.center_distance_km.to_numpy()}))
        print('FIT equal_city',target,seed,'seconds',round(time.time()-t0),flush=True)
    pd.concat(preds).to_parquet(OUT/f'{target}_predictions.parquet',index=False);pd.DataFrame(rows).to_csv(dest,index=False)
    print('DONE equal_city',target,round(time.time()-t0),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target');args=ap.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    if args.target:task(args.target);return
    cities,_=load();protocol={'objective':'Equal importance of every settlement in evaluation, including lower tail','primary_metric':'macro mean city coverage@10 on completely held-out cities','secondary_metrics':['coverage@1/5','fraction of realized oracle coverage','worst and lower-quartile city coverage','macro abs(log(calibration ratio))'],'comparison':'uniform rows cached baseline vs equal total city training weight','weight':'N_total / (K_train_cities * N_rows_in_city), normalized mean 1','train_years':YEARS,'test_years':[2023,2024,2025],'seeds':SEEDS,'features':FEATURES,'preprocessing':'Unweighted train-only p99 identical to baseline, isolates sample-weight change','scope':'14 municipal cohorts of >350k cities; smaller settlements not validated','selection':'Exploratory fixed weighting formula; no tuning or feature selection; target excluded from training'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    def worker(c):
        path=OUT/f"{c['slug']}.log"
        with path.open('w') as log:r=subprocess.run([sys.executable,__file__,'--target',c['slug']],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(str(path))
        print(path.read_text().splitlines()[-1] if path.stat().st_size else 'CACHED '+c['slug'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,cities))
    pd.concat([pd.read_csv(OUT/f"{c['slug']}_metrics.csv") for c in cities],ignore_index=True).to_csv(OUT/'metrics.csv',index=False)
    pd.concat([pd.read_csv(OUT/f"{c['slug']}_weights.csv") for c in cities],ignore_index=True).to_csv(OUT/'weights.csv',index=False)
    print('EQUAL CITY COMPLETE',flush=True)
if __name__=='__main__':main()
