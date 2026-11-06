"""Tables and Russian figures for 14 municipal cohorts. Read-only with respect to fitted models."""
from __future__ import annotations
import sys,json,argparse,itertools
from pathlib import Path
import numpy as np,pandas as pd,h3
from scipy.stats import ks_2samp,wasserstein_distance,spearmanr
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from road_accident_risk import v2_eval_year as vy
from road_accident_risk.paths import PROJECT_ROOT
BASE=PROJECT_ROOT/'reports/multicity_350k';OUT=PROJECT_ROOT/'docs/analysis/multicity_350k';FIG=OUT/'figures'
FEATURES=json.loads((PROJECT_ROOT/'reports/transfer_cities/protocol.json').read_text())['features']
RU=json.loads((PROJECT_ROOT/'docs/analysis/transfer_eda/feature_labels_ru.json').read_text())
plt.rcParams.update({'font.size':9})
def save(fig,name):fig.savefig(FIG/f'{name}.png',dpi=160,bbox_inches='tight');plt.close(fig)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--features-only',action='store_true');args=ap.parse_args();FIG.mkdir(parents=True,exist_ok=True)
    cities=json.loads((BASE/'cities.json').read_text());names={c['slug']:c['name'] for c in cities};order=list(names)
    frames={c:pd.read_parquet(BASE/'features'/f'{c}.parquet') for c in order};counts=pd.read_parquet(BASE/'counts.parquet')
    audit=pd.read_csv(BASE/'coordinate_audit.csv');profiles=pd.read_csv(BASE/'city_profiles.csv');desc=[];associations=[];yearprofiles=[];quality=[];strata=[]
    for city,f in frames.items():
        cnt=counts[counts.city==city].set_index('h3').reindex(f.h3);y=cnt[['2023','2024','2025']].sum(axis=1).to_numpy();hist=cnt[[str(a) for a in range(2015,2023)]].mean(axis=1).to_numpy()*3
        for feature in FEATURES:
            x=f[feature];q=x.quantile([.01,.1,.25,.5,.75,.9,.99]);desc.append({'city':city,'feature':feature,'missing_share':x.isna().mean(),'zero_share':(x==0).mean(),'mean':x.mean(),'std':x.std(),'min':x.min(),'max':x.max(),**{f'p{int(k*100)}':v for k,v in q.items()}})
            m=x.notna();rho=spearmanr(x[m],y[m]).statistic if x[m].nunique()>1 else np.nan
            associations.append({'city':city,'feature':feature,'spearman_with_test_count':rho})
        for yr in range(2015,2026):
            yy=cnt[str(yr)].to_numpy();yearprofiles.append({'city':city,'year':yr,'n_road_crashes':yy.sum(),'zero_share':(yy==0).mean(),'realized_cov10':vy.coverages(yy,yy)[2]})
        quality.append({'city':city,'test_count':y.sum(),'zero_test_share':(y==0).mean(),'realized_cov10':vy.coverages(y,y)[2],'history_cov10':vy.coverages(y,hist)[2],'history_calib_ratio':hist.sum()/y.sum(),'test_vs_train_annual_rate':y.sum()/hist.sum(),'surface_tagged_mean':f.surface_tagged_share.mean(),'lanes_tagged_mean':f.lanes_tagged_share.mean(),'lanes_missing':f.lanes_mean_weighted_v2.isna().mean(),'crossing_per_street_km':f.crossing_count.sum()/(f.street_length_m.sum()/1000),'street_junctions_per_street_km':(f.street_junction_3leg_count.sum()+f.street_junction_4leg_count.sum())/(f.street_length_m.sum()/1000),'road_service_length_share':(f.road_share_service*f.road_length_total_m).sum()/f.road_length_total_m.sum(),'street_length_median':f.street_length_m.median(),'block_length_median':f.block_length_median_m.median(),'orientation_median':f.road_orientation_entropy_v2.median()})
    pd.DataFrame(desc).to_csv(OUT/'feature_descriptive.csv',index=False);pd.DataFrame(associations).to_csv(OUT/'feature_associations.csv',index=False);pd.DataFrame(yearprofiles).to_csv(OUT/'year_profiles.csv',index=False)
    prof=profiles.merge(pd.DataFrame(quality),on='city',validate='one_to_one');prof.to_csv(OUT/'city_profiles.csv',index=False)
    shifts=[]
    for a,b in itertools.combinations(order,2):
        for feature in FEATURES:
            x=frames[a][feature].dropna();y=frames[b][feature].dropna();sd=np.sqrt((x.var()+y.var())/2)
            shifts.append({'a':a,'b':b,'feature':feature,'ks':ks_2samp(x,y).statistic,'wasserstein_pooled_sd':wasserstein_distance(x,y)/sd if sd>0 else 0,'missing_delta':frames[b][feature].isna().mean()-frames[a][feature].isna().mean()})
    pd.DataFrame(shifts).to_csv(OUT/'pairwise_feature_shift.csv',index=False)
    baselines=[]
    for city,f in frames.items():
        cnt=counts[counts.city==city].set_index('h3').reindex(f.h3)
        for subset,mask in [('municipality',np.ones(len(f),bool)),('radius15km',(f.center_distance_km<=15).to_numpy())]:
            y=cnt[['2023','2024','2025']].sum(axis=1).to_numpy()[mask]
            for method,s in [('major_road_length',f.major_road_length_m.to_numpy()),('road_length',f.road_length_total_m.to_numpy()),('history_all',cnt[[str(a) for a in range(2015,2023)]].mean(axis=1).to_numpy()*3),('history_clean',cnt[[str(a) for a in range(2017,2023)]].mean(axis=1).to_numpy()*3)]:
                baselines.append({'city':city,'subset':subset,'method':method,**vy.metrics_year(y,s[mask],is_count=method.startswith('history'))})
    pd.DataFrame(baselines).to_csv(OUT/'baselines.csv',index=False)
    comp=[]
    for subset,g in audit.groupby('subset'):
        for city,z in g.groupby('city'):
            early=z[z.year<=2016];late=z[z.year>=2017]
            row={'city':city,'subset':subset,'n_total':z.n_crashes.sum(),'n_early':early.n_crashes.sum(),'early_event_share':early.n_crashes.sum()/z.n_crashes.sum(),'outside_early':early.outside_road_cells.sum(),'outside_late':late.outside_road_cells.sum(),'outside_share_early':early.outside_road_cells.sum()/early.n_crashes.sum(),'outside_share_late':late.outside_road_cells.sum()/late.n_crashes.sum(),'early_share_of_all_outside':early.outside_road_cells.sum()/max(1,z.outside_road_cells.sum())}
            for key in ['far_50m_share','far_100m_share','far_250m_share','rounded_3dec_share']:
                row[f'{key}_early']=np.average(early[key],weights=early.n_crashes);row[f'{key}_late']=np.average(late[key],weights=late.n_crashes)
            comp.append(row)
    comp=pd.DataFrame(comp);comp.to_csv(OUT/'early_late_quality.csv',index=False);audit.to_csv(OUT/'coordinate_audit.csv',index=False)
    # Is the problem caused by service roads? Both all-road and street-only denominators are explicit in profiles.
    for key,label in [('outside_share','ДТП вне дорожных H3-ячеек, %'),('far_100m_share','ДТП далее 100 м от ближайшего образца дороги, %')]:
        fig,axs=plt.subplots(1,2,figsize=(14,8),layout='constrained')
        for ax,subset in zip(axs,['municipality','radius15km']):
            tab=audit[audit.subset==subset].pivot(index='city',columns='year',values=key).reindex(order)*100
            im=ax.imshow(tab,aspect='auto',cmap='YlOrRd',vmin=0,vmax=max(1,np.nanpercentile(tab,98)));ax.set_yticks(range(14),[names[c] for c in order]);ax.set_xticks(range(11),range(2015,2026),rotation=90)
            for i in range(14):
                for j in range(11):ax.text(j,i,f'{tab.iloc[i,j]:.1f}',ha='center',va='center',fontsize=6)
            ax.set_title('Городской округ' if subset=='municipality' else 'Округ ∩ радиус 15 км');fig.colorbar(im,ax=ax,shrink=.6)
        fig.suptitle(label);save(fig,key)
    d=pd.DataFrame(desc)
    for col,title in [('missing_share','Доля пропусков'),('p50','Медианы признаков: масштабирование по городам')]:
        tab=d.pivot(index='feature',columns='city',values=col).reindex(index=FEATURES,columns=order)
        if col=='p50':tab=tab.sub(tab.mean(axis=1),axis=0).div(tab.std(axis=1).replace(0,1),axis=0)
        fig,ax=plt.subplots(figsize=(14,10),layout='constrained');im=ax.imshow(tab,aspect='auto',cmap='YlOrRd' if col=='missing_share' else 'coolwarm',vmin=0 if col=='missing_share' else -2,vmax=1 if col=='missing_share' else 2);ax.set_yticks(range(28),[RU[f] for f in FEATURES],fontsize=7);ax.set_xticks(range(14),[names[c] for c in order],rotation=70,ha='right');ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.6);save(fig,col)
    # Per-feature ECDFs with all city curves, four pages; NaN shown separately in missingness.
    for start in range(0,28,7):
        fig,axs=plt.subplots(4,2,figsize=(13,14),layout='constrained')
        for ax,feature in zip(axs.flat,FEATURES[start:start+7]):
            for city in order:
                x=np.sort(frames[city][feature].dropna());ax.plot(x,np.arange(1,len(x)+1)/len(x),label=names[city],lw=.8)
            ax.set_xscale('symlog',linthresh=1);ax.set_title(RU[feature],fontsize=9);ax.set_ylabel('Накопленная доля')
        axs.flat[-1].axis('off');handles,labels=axs.flat[0].get_legend_handles_labels();axs.flat[-1].legend(handles,labels,ncol=2,fontsize=8,loc='center');save(fig,f'ecdf_{start//7+1}')
    for label,keys in [('morphology',['road_km_per_km2','road_service_length_share','street_length_median','block_length_median','street_junctions_per_street_km','orientation_median']),('mapping_quality',['surface_tagged_mean','lanes_tagged_mean','lanes_missing','crossing_per_street_km'])]:
        X=prof.set_index('city').reindex(order)[keys];X=X.fillna(X.median());scaled=np.clip(RobustScaler().fit_transform(X),-5,5)
        fig,ax=plt.subplots(figsize=(11,5),layout='constrained');dendrogram(linkage(scaled,method='average'),labels=[names[c] for c in order],leaf_rotation=55,ax=ax);ax.set_title('Разведочное сходство: '+('морфология сети' if label=='morphology' else 'полнота разметки и переходы')+'\nГруппы не фиксировались; результаты ДТП не используются');ax.set_ylabel('Расстояние профилей');save(fig,f'profile_similarity_{label}')
        X.to_csv(OUT/f'grouping_axes_{label}.csv')
    print('FEATURE EDA COMPLETE',flush=True)
    if args.features_only:return
    metrics=pd.read_csv(BASE/'metrics.csv')
    if (BASE/'recent_sensitivity.csv').exists():metrics=pd.concat([metrics,pd.read_csv(BASE/'recent_sensitivity.csv')],ignore_index=True)
    metrics.to_csv(OUT/'metrics_all.csv',index=False)
    met=metrics[(metrics.period=='sum')].groupby(['kind','source','target','window','subset']).mean(numeric_only=True).reset_index();met.to_csv(OUT/'metrics_summary.csv',index=False)
    comparison=[];bootstrap=[]
    for city in order:
        data=pd.read_parquet(BASE/f'{city}_predictions.parquet');f=frames[city].set_index('h3')
        for subset in ['municipality','radius15km']:
            z=data if subset=='municipality' else data[data.center_distance_km<=15]
            wide=z.groupby(['h3','window']).predicted.mean().unstack().sort_index();y=z.groupby('h3')['observed'].first().reindex(wide.index).to_numpy();p0=wide['all_2015_2022'].to_numpy();p1=wide['clean_2017_2022'].to_numpy()
            groups=pd.Series(wide.index).map(lambda h:h3.cell_to_parent(h,7));ixs=[np.flatnonzero(groups==g) for g in sorted(groups.unique())];rng=np.random.default_rng(20261106);draws=[]
            for _ in range(500):
                ix=np.concatenate([ixs[i] for i in rng.integers(0,len(ixs),len(ixs))]);draws.append(np.array(vy.coverages(y[ix],p1[ix]))-vy.coverages(y[ix],p0[ix]))
            for i,q in enumerate([1,5,10]):bootstrap.append({'city':city,'subset':subset,'top_pct':q,'delta_ensemble':vy.coverages(y,p1)[i]-vy.coverages(y,p0)[i],'ci_low':np.quantile(np.array(draws)[:,i],.025),'ci_high':np.quantile(np.array(draws)[:,i],.975),'n_blocks':len(ixs)})
            allmet=met[(met.kind=='loco')&(met.target==city)&(met.subset==subset)].set_index('window');a=allmet.loc['all_2015_2022'];b=allmet.loc['clean_2017_2022'];c=allmet.loc['control_drop_2017_2018']
            comparison.append({'city':city,'subset':subset,'all_cov10':a.coverage_top_10pct,'clean_cov10':b.coverage_top_10pct,'delta_cov10':b.coverage_top_10pct-a.coverage_top_10pct,'control_cov10_seed42':c.coverage_top_10pct,'all_calib':a.calib_ratio,'clean_calib':b.calib_ratio,'oracle10':a.coverage_oracle_top_10pct,'all_fraction_realized':a.coverage_top_10pct/a.coverage_oracle_top_10pct,'clean_fraction_realized':b.coverage_top_10pct/b.coverage_oracle_top_10pct})
    compmodel=pd.DataFrame(comparison);compmodel.to_csv(OUT/'window_comparison.csv',index=False);pd.DataFrame(bootstrap).to_csv(OUT/'paired_spatial_bootstrap.csv',index=False)
    # Fixed seed control compared to matching seed, not a three-seed mean.
    controls=metrics[(metrics.kind=='loco')&(metrics.seed==42)&(metrics.period=='sum')].pivot(index=['target','subset'],columns='window',values='coverage_top_10pct').reset_index();controls.to_csv(OUT/'window_control_seed42.csv',index=False)
    # LOCO is the primary transfer test; same-city is a different task.
    local=met[(met.kind=='local')&(met.source==met.target)];local.to_csv(OUT/'local_reference.csv',index=False)
    for subset in ['municipality','radius15km']:
        z=compmodel[compmodel.subset==subset].set_index('city').reindex(order)
        fig,axs=plt.subplots(1,2,figsize=(14,7),layout='constrained');yy=np.arange(14)
        axs[0].barh(yy-.18,z.all_cov10*100,.36,label='Обучение 2015–2022');axs[0].barh(yy+.18,z.clean_cov10*100,.36,label='Обучение 2017–2022');axs[0].set_yticks(yy,[names[c] for c in order]);axs[0].set_xlabel('ДТП в верхних 10% ячеек, %');axs[0].legend(fontsize=8)
        axs[1].barh(yy-.18,z.all_calib,.36,label='2015–2022');axs[1].barh(yy+.18,z.clean_calib,.36,label='2017–2022');axs[1].axvline(1,color='black',ls='--');axs[1].set_yticks(yy,[names[c] for c in order]);axs[1].set_xlabel('Суммарный прогноз / факт');fig.suptitle('Перенос на полностью исключённый город · '+('городские округа' if subset=='municipality' else 'радиус 15 км'));save(fig,f'window_comparison_{subset}')
        matrix=met[(met.kind=='local')&(met.window=='all_2015_2022')&(met.subset==subset)].pivot(index='source',columns='target',values='coverage_top_10pct').reindex(index=order,columns=order)*100
        fig,ax=plt.subplots(figsize=(12,10),layout='constrained');im=ax.imshow(matrix,cmap='YlGnBu',vmin=20,vmax=85)
        for i in range(14):
            for j in range(14):ax.text(j,i,f'{matrix.iloc[i,j]:.0f}',ha='center',va='center',fontsize=6)
        ax.set_yticks(range(14),[names[c] for c in order]);ax.set_xticks(range(14),[names[c] for c in order],rotation=70,ha='right');ax.set_ylabel('Город обучения');ax.set_xlabel('Город оценки');ax.set_title('Матрица переноса: охват ДТП верхними 10% ячеек · '+('городские округа' if subset=='municipality' else 'радиус 15 км'));fig.colorbar(im,ax=ax,shrink=.6);save(fig,f'transfer_matrix_{subset}')
    print('MODEL EDA COMPLETE',flush=True)
if __name__=='__main__':main()
