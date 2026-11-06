"""Post-audit diagnostic for residual 2017/2018 problems; seed42 only, not independent selection."""
import sys,subprocess,concurrent.futures,argparse
from multicity_models import OUT,load,fit,evaluate,FEATURES,cb
import pandas as pd

def task(target):
    path=OUT/f'recent_{target}.csv'
    if path.exists():return
    _,frames=load();train=[f for city,f in frames.items() if city!=target];m,caps=fit(train,list(range(2019,2023)),42);f=frames[target];pred=m.predict(cb.apply_p99_clipping(f,caps)[FEATURES],prediction_type='Exponent')
    pd.DataFrame(evaluate(f,pred,{'kind':'loco','source':'other_13_cities','target':target,'window':'recent_2019_2022','seed':42})).to_csv(path,index=False)
    print('RECENT DONE',target,flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target');args=ap.parse_args()
    if args.target:task(args.target);return
    cities,_=load()
    def worker(c):
        with (OUT/f"recent_{c['slug']}.log").open('w') as log:r=subprocess.run([sys.executable,__file__,'--target',c['slug']],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(c['slug'])
        print('RECENT DONE',c['slug'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,cities))
    pd.concat([pd.read_csv(OUT/f"recent_{c['slug']}.csv") for c in cities],ignore_index=True).to_csv(OUT/'recent_sensitivity.csv',index=False)
if __name__=='__main__':main()
