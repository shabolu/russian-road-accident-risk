"""Spatial block bootstrap of the frozen transfer experiment; no model selection."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import h3,numpy as np,pandas as pd
from road_accident_risk import v2_eval_year as vy
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/transfer_cities'
p=pd.read_parquet(OUT/'predictions.parquet'); metrics=pd.read_csv(OUT/'metrics.csv')
summary=metrics.groupby(['city','method']).mean(numeric_only=True).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
rows=[]
for city,df in p.groupby('city'):
    d=df.groupby('h3').agg(observed=('observed','first'),predicted=('predicted','mean')).reset_index()
    assert df.groupby('h3')['observed'].nunique().max()==1
    if city=='saratov': f=pd.read_parquet(ROOT/'parquet_output/h3_road_features.parquet')
    else: f=pd.read_parquet(OUT/f'{city}_features.parquet')
    d=d.merge(f[['h3','major_road_length_m']],on='h3',validate='one_to_one')
    blocks=d.h3.map(lambda c:h3.cell_to_parent(c,7)); groups=[np.flatnonzero(blocks==b) for b in sorted(blocks.unique())]
    rng=np.random.default_rng(20261106); samples=[]
    for _ in range(500):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]); t=d.iloc[ix]
        a=np.array(vy.coverages(t.observed.to_numpy(),t.predicted.to_numpy())); b=np.array(vy.coverages(t.observed.to_numpy(),t.major_road_length_m.to_numpy()))
        samples.append(np.r_[a,a-b])
    samples=np.array(samples)
    for k,q in enumerate([1,5,10]):
        rows.append({'city':city,'top_pct':q,'blocks':len(groups),'coverage':vy.coverages(d.observed.to_numpy(),d.predicted.to_numpy())[k], 'ci_low':np.quantile(samples[:,k],.025),'ci_high':np.quantile(samples[:,k],.975),'delta_vs_major_low':np.quantile(samples[:,k+3],.025),'delta_vs_major_high':np.quantile(samples[:,k+3],.975)})
pd.DataFrame(rows).to_csv(OUT/'spatial_bootstrap.csv',index=False)
print(summary[['city','method','coverage_top_1pct','coverage_top_5pct','coverage_top_10pct','calib_ratio']].to_string(index=False))
print(pd.DataFrame(rows).to_string(index=False))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
cities=['saratov','samara','ulyanovsk']; x=np.arange(3)
for j,(method,label) in enumerate([('osm_transfer','Модель OSM: обучение на Саратове'),('major_road_length','По длине крупных дорог'),('local_history_reference','По местной истории ДТП (ориентир)')]):
    v=summary[summary.method==method].set_index('city').loc[cities,'coverage_top_10pct']*100
    ax.bar(x+(j-1)*.25,v,.25,label=label)
ax.set_xticks(x,['Саратов','Самара','Ульяновск']); ax.set_ylabel('Доля ДТП за 2023–2025 годы\nв выбранных 10% ячеек, %'); ax.set_ylim(0,100)
ax.legend(loc='upper center',fontsize=8); ax.set_title('Перенос модели из Саратова: 28 признаков OSM')
fig.savefig(OUT/'transfer_top10.png',dpi=180); plt.close(fig)
