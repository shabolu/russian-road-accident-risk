"""Post-test sensitivity probes. No claims of independent validation."""
from pathlib import Path
# Reuse data preparation only, without rerunning the main analysis.
p=Path(__file__).with_name('eda_city_transfer.py')
exec(p.read_text().split('counts={}; overview=')[0])
counts={}
for c,f in frames.items():
    f=f.merge(preds[preds.city==c][['h3','pred','observed']],on='h3');frames[c]=f
    cc=cr[cr.h3.isin(f.h3)];counts[c]=cc.groupby(['h3','year']).size().unstack(fill_value=0).reindex(index=f.h3,columns=range(2015,2026),fill_value=0).fillna(0)
tr0=panel[panel.year<=2022]
# Exact original training targets, including original bbox handling.
blocks={'all':features,'no_tag_sensitive':[f for f in features if f not in ['lanes_mean_weighted_v2','bad_surface_share_among_tagged','crossing_count','public_transport_stop_count','oneway_share_v2']], 'no_geometry':[f for f in features if f not in ['min_curve_radius_m','block_length_median_m','road_orientation_entropy_v2','street_dead_end_count','curvature_deg_per_km_v2']], 'no_road_classes':[f for f in features if not f.startswith('road_share_')]}
rows=[]; sh=[]
for variant,fs in blocks.items():
    cp=cb.fit_p99_clip_caps(tr0,fs);tr=cb.apply_p99_clipping(tr0,cp);m=vy.make_model(loss='Tweedie:variance_power=1.3',seed=42);m.fit(tr[fs],tr[vy.TARGET_COUNT])
    for city,f in frames.items():
        s=3*m.predict(cb.apply_p99_clipping(f,cp)[fs],prediction_type='Exponent');rows.append({'variant':variant,'city':city,**vy.metrics_year(f.observed.to_numpy(),s,is_count=True)})
        if variant=='all':
            # Historical calibration uses target history only, for diagnosis. Not zero-history deployment.
            factor=counts[city][list(range(2015,2023))].sum().sum()/8/(s.sum()/3)
            sh.append({'city':city,'historical_scale':factor,'test_calib_before':s.sum()/f.observed.sum(),'test_calib_after':(s*factor).sum()/f.observed.sum()})
pd.DataFrame(rows).to_csv(OUT/'ablation_diagnostic.csv',index=False);pd.DataFrame(sh).to_csv(OUT/'historical_calibration_diagnostic.csv',index=False)
# Source spatial OOF distinguishes known-location performance from unseen-location performance.
f=frames['saratov']; groups=f.h3.map(lambda c:h3.cell_to_parent(c,7)); score=np.zeros(len(f))
for ti,ei in GroupKFold(5).split(f,groups=groups):
    train=tr0[tr0.h3.isin(f.iloc[ti].h3)];cp=cb.fit_p99_clip_caps(train,features);m=vy.make_model(loss='Tweedie:variance_power=1.3',seed=42);m.fit(cb.apply_p99_clipping(train,cp)[features],train[vy.TARGET_COUNT]);score[ei]=3*m.predict(cb.apply_p99_clipping(f.iloc[ei],cp)[features],prediction_type='Exponent')
pd.DataFrame([{'city':'saratov','method':'spatial_oof_5fold',**vy.metrics_year(f.observed.to_numpy(),score,is_count=True)}]).to_csv(OUT/'saratov_spatial_oof.csv',index=False)
# Crashes by type/severity and temporal distributions; registrations, not causal exposure.
types=[]
for c,f in frames.items():
    cc=cr[cr.h3.isin(f.h3)&cr.year.between(2023,2025)]
    for typ,gg in cc.groupby('accident_type'):
        types.append({'city':c,'type':typ,'n':len(gg),'share':len(gg)/len(cc),'fatal_share':gg.has_fatal.mean(),'deaths':gg.dead_count.sum(),'injuries':gg.injured_count.sum()})
pd.DataFrame(types).to_csv(OUT/'accident_types.csv',index=False)
# Feature similarity as profiles, not clusters fitted to only three cities.
d=pd.read_csv(OUT/'feature_descriptive.csv');profile=d.pivot(index='feature',columns='city',values='mean').reindex(columns=CITIES)
scale=d[d.city=='saratov'].set_index('feature')['std'];z=profile.sub(profile.saratov,axis=0).div(scale.replace(0,np.nan),axis=0)
fig,ax=plt.subplots(figsize=(8,10),layout='constrained');im=ax.imshow(z,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1);ax.set_yticks(range(len(z)),z.index,fontsize=8);ax.set_xticks(range(3),[LABEL[c] for c in CITIES]);ax.set_title('Сдвиг средних относительно Саратова\nв стандартных отклонениях Саратова; NaN исключены');fig.colorbar(im,ax=ax);fig.savefig(FIG/'feature_profiles.png',dpi=160);plt.close(fig)
print(pd.DataFrame(rows)[['variant','city','coverage_top_10pct','calib_ratio']].to_string(index=False));print(sh)
