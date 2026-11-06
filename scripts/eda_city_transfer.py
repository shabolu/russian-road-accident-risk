"""Comparative exploratory analysis. All post-test experiments are diagnostic, not validation."""
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,h3
from scipy.stats import ks_2samp,spearmanr,wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor,CatBoostClassifier,Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from road_accident_risk import v2_eval_year as vy,catboost_baseline as cb
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'reports/transfer_cities'; OUT=ROOT/'docs/analysis/transfer_eda'; FIG=OUT/'figures'
CITIES=['saratov','samara','ulyanovsk']; LABEL=dict(zip(CITIES,['Саратов','Самара','Ульяновск'])); COLORS=['#2475ad','#e89935','#439555']
features=json.loads((BASE/'protocol.json').read_text())['features']; caps=json.loads((BASE/'caps.json').read_text())
panel=vy.load_year_panel('road_geo_v2_geometry'); frames={'saratov':panel[panel.year==2023].drop_duplicates('h3').copy()}
for c in CITIES[1:]: frames[c]=pd.read_parquet(BASE/f'{c}_features.parquet')
preds=pd.read_parquet(BASE/'predictions.parquet').groupby(['city','h3']).agg(pred=('predicted','mean'),observed=('observed','first')).reset_index()
cr=pd.read_parquet(ROOT/'parquet_output/crashes_slim.parquet'); cr['year']=pd.to_datetime(cr.datetime).dt.year
cr=cr[cr.year.between(2015,2025)&cr.lat.between(51.3,54.6)&cr.lng.between(45.7,50.6)].copy()
cr['h3']=[h3.latlng_to_cell(a,b,9) for a,b in zip(cr.lat,cr.lng)]
counts={}; overview=[]; desc=[]; annual=[]; associations=[]; errors=[]; strata=[]; calibration=[]; sensitivity=[]
model=CatBoostRegressor();model.load_model(str(BASE/'model_42.cbm')); shaprows=[]; imp=[]
for city,f in frames.items():
    f=f.merge(preds[preds.city==city][['h3','pred','observed']],on='h3',validate='one_to_one'); frames[city]=f
    cc=cr[cr.h3.isin(f.h3)]; cnt=cc.groupby(['h3','year']).size().unstack(fill_value=0).reindex(index=f.h3,columns=range(2015,2026),fill_value=0).fillna(0); counts[city]=cnt
    assert np.array_equal(cnt[[2023,2024,2025]].sum(axis=1).to_numpy(),f.observed.to_numpy())
    f['history']=cnt[list(range(2015,2023))].mean(axis=1).to_numpy()*3
    y=f.observed.to_numpy(); s=f.pred.to_numpy(); hist=f.history.to_numpy()
    met=vy.metrics_year(y,s,is_count=True)
    overview.append({'city':city,**met,'zero_share':np.mean(y==0),'history_cov10':vy.coverages(y,hist)[2],'test_vs_history_rate':y.sum()/hist.sum(),'road_km':f.road_length_total_m.sum()/1000,'crashes_per_road_km_year':y.sum()/(3*f.road_length_total_m.sum()/1000),'mean_road_m':f.road_length_total_m.mean(),'history_spearman':spearmanr(y,hist).statistic})
    for yr in range(2015,2026):
        yy=cnt[yr].to_numpy();annual.append({'city':city,'year':yr,'crashes':yy.sum(),'zero_share':np.mean(yy==0),'cov10':vy.coverages(yy,s)[2],'predicted':s.sum()/3,'calib_ratio':s.sum()/3/yy.sum()})
    for feature in features:
        x=f[feature];q=x.quantile([.01,.1,.25,.5,.75,.9,.99])
        desc.append({'city':city,'feature':feature,'missing':x.isna().mean(),'zero':(x==0).mean(),'mean':x.mean(),'std':x.std(),'min':x.min(),'max':x.max(),**{f'p{int(k*100)}':v for k,v in q.items()},'above_saratov_cap':(x>caps.get(feature,np.inf)).mean()})
        valid=x.notna();associations.append({'city':city,'feature':feature,'rho_test':spearmanr(x[valid],y[valid]).statistic if x[valid].nunique()>1 else np.nan,'rho_history':spearmanr(x[valid],hist[valid]).statistic if x[valid].nunique()>1 else np.nan})
        # Bins fixed on Saratov only, missing kept separate.
        edges=np.unique(frames['saratov'][feature].dropna().quantile([.25,.5,.75]).values)
        bins=pd.cut(x,[-np.inf,*edges,np.inf],duplicates='drop').astype(str); bins=bins.where(x.notna(),'missing')
        for b in bins.unique():
            mask=(bins==b).to_numpy();strata.append({'city':city,'feature':feature,'bin':b,'n':mask.sum(),'obs_mean':y[mask].mean(),'pred_mean':s[mask].mean(),'obs_sum':y[mask].sum(),'pred_sum':s[mask].sum(),'history_mean':hist[mask].mean()})
    rank=f.pred.rank(method='first',pct=True); dec=np.minimum(9,(rank*10).astype(int))
    for b in range(10):
        m=dec==b;calibration.append({'city':city,'decile':b+1,'n':m.sum(),'obs':y[m].mean(),'pred':s[m].mean(),'history':hist[m].mean()})
    k=int(len(f)*.1); top=np.argsort(-s)[:k]; true=np.argsort(-y)[:k]
    f['error']=s-y; f['abs_error']=abs(s-y);f['top10_pred']=False;f.loc[f.index[top],'top10_pred']=True
    for kind, ix in [('under',np.argsort(s-y)[:30]),('over',np.argsort(y-s)[:30])]:
        z=f.iloc[ix][['h3','pred','observed','history',*features]].copy();z['city']=city;z['kind']=kind;errors.append(z)
    for name,mask in [('all',np.ones(len(f),bool)),('street_ge200',f.street_length_m>=200),('no_history',hist==0),('has_history',hist>0),('no_service_dominance',f.road_share_service<.5)]:
        m=np.asarray(mask);sensitivity.append({'city':city,'subset':name,'n':m.sum(),'crashes':y[m].sum(),'coverage10':vy.coverages(y[m],s[m])[2],'oracle10':vy.coverages(y[m],y[m])[2],'calib_ratio':s[m].sum()/max(1,y[m].sum())})
    X=cb.apply_p99_clipping(f,caps)[features]
    sv=model.get_feature_importance(Pool(X),type='ShapValues'); assert np.allclose(sv.sum(axis=1),model.predict(X,prediction_type='RawFormulaVal'),atol=1e-6)
    for i,feature in enumerate(features): shaprows.append({'city':city,'feature':feature,'mean_abs_shap':abs(sv[:,i]).mean(),'mean_shap':sv[:,i].mean()})
    # Cell locations for errors and prediction maps.
    latlon=np.array([h3.cell_to_latlng(c) for c in f.h3]);f['lat']=latlon[:,0];f['lon']=latlon[:,1]
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,col,title in zip(axs,['observed','pred','error'],['Факт ДТП, 2023–2025','Прогноз ДТП, 2023–2025','Прогноз − факт']):
        val=f[col]; limit=max(1,float(abs(val).quantile(.98)))
        sc=ax.scatter(f.lon,f.lat,c=val,s=5,cmap='coolwarm' if col=='error' else 'viridis',vmin=-limit if col=='error' else 0,vmax=limit);fig.colorbar(sc,ax=ax,shrink=.7);ax.set_title(title);ax.set_aspect(1/np.cos(np.radians(f.lat.mean())))
    fig.suptitle(LABEL[city]+' — карты ячеек (цвет ограничен 98-м перцентилем)');fig.savefig(FIG/f'map_{city}.png',dpi=160);plt.close(fig)
    print('CITY',city,flush=True)
for name,data in [('overview',overview),('feature_descriptive',desc),('annual',annual),('associations',associations),('feature_bins',strata),('calibration',calibration),('sensitivity',sensitivity),('shap',shaprows)]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
pd.concat(errors).to_csv(OUT/'error_examples.csv',index=False)
shifts=[]
for a,b in itertools.combinations(CITIES,2):
    for feature in features:
        x=frames[a][feature].dropna();y=frames[b][feature].dropna();iqr=x.quantile(.75)-x.quantile(.25);scale=iqr if iqr>0 else x.std()
        shifts.append({'source':a,'target':b,'feature':feature,'ks':ks_2samp(x,y).statistic,'wasserstein_scaled':wasserstein_distance(x,y)/scale if scale>0 else np.nan,'missing_delta':frames[b][feature].isna().mean()-frames[a][feature].isna().mean()})
pd.DataFrame(shifts).to_csv(OUT/'feature_shift.csv',index=False)
# Spatially held-out city classifier: separability of feature distributions, not traffic prediction.
dom=[]
for a,b in itertools.combinations(CITIES,2):
    X=pd.concat([frames[a][features],frames[b][features]],ignore_index=True);yy=np.r_[np.zeros(len(frames[a])),np.ones(len(frames[b]))]
    hh=pd.concat([frames[a].h3,frames[b].h3]);groups=hh.map(lambda c:h3.cell_to_parent(c,7)); pp=np.zeros(len(X))
    for tr,te in GroupKFold(3).split(X,yy,groups):
        m=CatBoostClassifier(iterations=200,depth=4,verbose=False,allow_writing_files=False,random_seed=42);m.fit(X.iloc[tr],yy[tr]);pp[te]=m.predict_proba(X.iloc[te])[:,1]
    dom.append({'a':a,'b':b,'spatial_oof_auc':roc_auc_score(yy,pp)})
pd.DataFrame(dom).to_csv(OUT/'domain_classifier.csv',index=False)
print('DOMAIN',dom,flush=True)
# Diagnostic full source/target matrix, frozen hyperparameters, one seed; no selection.
mat=[]
for source in CITIES:
    f=frames[source]; tr=pd.concat([f[features].assign(target=counts[source][y].to_numpy()) for y in range(2015,2023)],ignore_index=True);cap=cb.fit_p99_clip_caps(tr,features);tr=cb.apply_p99_clipping(tr,cap)
    if source=='saratov':
        tr=panel[panel.year<=2022].copy();cap=cb.fit_p99_clip_caps(tr,features);tr=cb.apply_p99_clipping(tr,cap);tr['target']=tr[vy.TARGET_COUNT]
    m=vy.make_model(loss='Tweedie:variance_power=1.3',seed=42);m.fit(tr[features],tr.target)
    for target in CITIES:
        z=frames[target];ss=3*m.predict(cb.apply_p99_clipping(z,cap)[features],prediction_type='Exponent');mat.append({'source':source,'target':target,**vy.metrics_year(z.observed.to_numpy(),ss,is_count=True)})
pd.DataFrame(mat).to_csv(OUT/'transfer_matrix_diagnostic.csv',index=False)
RU=json.loads((OUT/'feature_labels_ru.json').read_text())
# Plots: all features and joint correlations, concentration, calibration, temporal dynamics.
for start in range(0,len(features),7):
    fig,axs=plt.subplots(4,2,figsize=(12,14),layout='constrained')
    for ax,feature in zip(axs.flat,features[start:start+7]):
        for city,color in zip(CITIES,COLORS):
            x=np.sort(frames[city][feature].dropna());ax.plot(x,np.arange(1,len(x)+1)/len(x),label=LABEL[city],color=color)
        ax.set_title(RU.get(feature,feature),fontsize=9);ax.set_xscale('symlog',linthresh=1);ax.set_ylabel('Накопленная доля');ax.legend(fontsize=7)
    axs.flat[-1].axis('off');fig.suptitle('Распределения признаков: ненулевые и нулевые значения, без NaN');fig.savefig(FIG/f'ecdf_{start//7+1}.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for city,color in zip(CITIES,COLORS):
    f=frames[city];curve=vy.gain_curve(f.observed.to_numpy(),f.pred.to_numpy());axs[0].plot(curve.frac_cells,curve.frac_accidents,label=LABEL[city],color=color)
    curve=vy.gain_curve(f.observed.to_numpy(),f.observed.to_numpy());axs[0].plot(curve.frac_cells,curve.frac_accidents,ls='--',color=color,alpha=.5)
    a=pd.DataFrame(annual);a=a[a.city==city];axs[1].plot(a.year,a.crashes/a[a.year<=2022].crashes.mean(),label=LABEL[city],color=color)
    z=pd.DataFrame(calibration);z=z[z.city==city];axs[2].plot(z.pred,z.obs,'o-',label=LABEL[city],color=color)
axs[0].set(xlabel='Доля выбранных ячеек',ylabel='Доля ДТП',xlim=(0,.3),title='Охват: модель; пунктир — сортировка по факту')
axs[1].set(xlabel='Год',ylabel='К среднему 2015–2022',title='Динамика числа ДТП');axs[1].axvline(2022.5,color='gray',ls=':')
axs[2].plot([0,20],[0,20],'k--');axs[2].set(xlabel='Средний прогноз в дециле',ylabel='Средний факт в дециле',title='Калибровка, ДТП за три года')
for ax in axs:ax.legend(fontsize=8)
fig.savefig(FIG/'performance_context.png',dpi=170);plt.close(fig)
d=pd.DataFrame(desc);fig,axs=plt.subplots(1,2,figsize=(13,10),layout='constrained')
for ax,col,title in zip(axs,['missing','above_saratov_cap'],['Доля пропусков','Доля выше саратовского P99-порога']):
    tab=d.pivot(index='feature',columns='city',values=col).reindex(index=features,columns=CITIES);im=ax.imshow(tab,aspect='auto',vmin=0,vmax=1 if col=='missing' else .06,cmap='YlOrRd');ax.set_yticks(range(len(features)),[RU.get(f,f) for f in features],fontsize=7);ax.set_xticks(range(3),[LABEL[c] for c in CITIES]);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.6)
fig.savefig(FIG/'missing_clipping.png',dpi=170);plt.close(fig)
for city in CITIES:
    corr=frames[city][features].corr(method='spearman');corr.to_csv(OUT/f'correlation_{city}.csv')
sh=pd.DataFrame(shaprows).pivot(index='feature',columns='city',values='mean_abs_shap').reindex(columns=CITIES);sh=sh.loc[sh.mean(axis=1).sort_values().index]
ax=sh.rename(columns=LABEL,index=RU).plot.barh(figsize=(11,10),color=COLORS);ax.set_xlabel('Средний |SHAP| в логарифме ожидаемого числа ДТП');ax.set_title('Какие признаки использует саратовская модель (seed 42)');plt.tight_layout();plt.savefig(FIG/'shap.png',dpi=160);plt.close()
print(pd.DataFrame(overview).to_string(index=False),flush=True);print(pd.DataFrame(mat)[['source','target','coverage_top_10pct','calib_ratio']].to_string(index=False),flush=True)
