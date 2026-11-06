"""Fixed-recipe temporal sensitivity and leave-one-city-out (LOCO). Exploratory study."""
from __future__ import annotations
import sys,json,argparse,subprocess,concurrent.futures,time
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from road_accident_risk import v2_eval_year as vy,catboost_baseline as cb
from road_accident_risk.paths import PROJECT_ROOT
OUT=PROJECT_ROOT/'reports/multicity_350k'
FEATURES=json.loads((PROJECT_ROOT/'reports/transfer_cities/protocol.json').read_text())['features']
WINDOWS={'all_2015_2022':list(range(2015,2023)),'clean_2017_2022':list(range(2017,2023)),'control_drop_2017_2018':[2015,2016,2019,2020,2021,2022]}
SEEDS=[42,7,123]
def load():
    cities=json.loads((OUT/'cities.json').read_text());counts=pd.read_parquet(OUT/'counts.parquet')
    frames={c['slug']:pd.read_parquet(OUT/'features'/f"{c['slug']}.parquet").merge(counts[counts.city==c['slug']].drop(columns='city'),on='h3',validate='one_to_one') for c in cities}
    return cities,frames
def fit(train,years,seed):
    # Annual panel preserves the original target and recipe, including target-free train-only clipping.
    tr=pd.concat([f[FEATURES].assign(target=f[str(y)].to_numpy()) for f in train for y in years],ignore_index=True)
    caps=cb.fit_p99_clip_caps(tr,FEATURES);tr=cb.apply_p99_clipping(tr,caps)
    model=vy.make_model(loss='Tweedie:variance_power=1.3',seed=seed,params={'thread_count':3})
    model.fit(tr[FEATURES],tr.target.astype(int));return model,caps

def evaluate(f,pred,meta):
    rows=[]
    for subset,mask in [('municipality',np.ones(len(f),bool)),('radius15km',(f.center_distance_km<=15).to_numpy())]:
        if mask.sum()<10:continue
        z=f.loc[mask];s=pred[mask]
        for period in ['sum','2023','2024','2025']:
            y=z[['2023','2024','2025']].sum(axis=1).to_numpy() if period=='sum' else z[period].to_numpy()
            rows.append({**meta,'subset':subset,'period':period,**vy.metrics_year(y,s*(3 if period=='sum' else 1),is_count=True)})
    return rows

def task(kind,target):
    dest=OUT/f'metrics_{kind}_{target}.csv'
    if dest.exists():print('CACHED',kind,target,flush=True);return
    cities,frames=load();rows=[];preds=[];started=time.time()
    for window,years in WINDOWS.items():
        if kind=='local' and window.startswith('control'):continue
        seeds=SEEDS if kind=='loco' and not window.startswith('control') else ([42,7,123,2024,31337] if kind=='local' and target=='saratov' else [42])
        for seed in seeds:
            train=[f for c,f in frames.items() if c!=target] if kind=='loco' else [frames[target]]
            assert kind!='loco' or not set(frames[target].h3).intersection(set(pd.concat(train).h3))
            m,caps=fit(train,years,seed)
            evalcities=[target] if kind=='loco' else list(frames)
            for city in evalcities:
                f=frames[city];p=m.predict(cb.apply_p99_clipping(f,caps)[FEATURES],prediction_type='Exponent')
                rows.extend(evaluate(f,p,{'kind':kind,'source':'other_13_cities' if kind=='loco' else target,'target':city,'window':window,'seed':seed}))
                if kind=='loco':
                    preds.append(pd.DataFrame({'target':city,'window':window,'seed':seed,'h3':f.h3.to_numpy(),'observed':f[['2023','2024','2025']].sum(axis=1).to_numpy(),'predicted':3*p,'center_distance_km':f.center_distance_km.to_numpy()}))
            print('FIT',kind,target,window,seed,'seconds',round(time.time()-started),flush=True)
    pd.DataFrame(rows).to_csv(dest,index=False)
    if preds:pd.concat(preds).to_parquet(OUT/f'{target}_predictions.parquet',index=False)
    print('DONE',kind,target,round(time.time()-started),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['local','loco']);ap.add_argument('--target');ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    if args.target:task(args.kind,args.target);return
    cities,_=load()
    protocol={'cities':[c['slug'] for c in cities],'boundary':'municipal OSM relations; road-cell centroids inside','secondary_subset':'15km from OSM place node; no refit','features':FEATURES,'target':'annual count; predictions frozen end 2022','test_years':[2023,2024,2025],'windows':WINDOWS,'loco_seeds':SEEDS,'local_seed':42,'saratov_seeds':[42,7,123,2024,31337],'loco_control_seed':42,'recipe':{'iterations':800,'learning_rate':.05,'depth':6,'loss':'Tweedie:variance_power=1.3','thread_count':3},'selection':'Exploratory all-city study; no hyperparameter or cluster selection on these results','pbf_timestamp':'2026-06-03T20:21:00Z'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    jobs=[(kind,c['slug']) for kind in ['loco','local'] for c in cities]
    def worker(job):
        kind,slug=job;path=OUT/f'run_{kind}_{slug}.log'
        with path.open('w') as log:r=subprocess.run([sys.executable,__file__,'--kind',kind,'--target',slug],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(str(path))
        print(path.read_text().splitlines()[-1],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(worker,jobs))
    pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('metrics_*.csv'))],ignore_index=True).to_csv(OUT/'metrics.csv',index=False)
    print('ALL MODELS COMPLETE',flush=True)
if __name__=='__main__':main()
