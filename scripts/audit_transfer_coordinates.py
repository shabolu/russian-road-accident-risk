from pathlib import Path
p=Path(__file__).with_name('eda_city_transfer.py');exec(p.read_text().split('counts={}; overview=')[0])
boxes={'saratov':(51.4,45.8,51.7,46.2),'samara':(53.1,49.95,53.35,50.4),'ulyanovsk':(54.15,48.15,54.45,48.65)}
rows=[];bad=[]
for city,f in frames.items():
    a,b,c,d=boxes[city]; q=cr[cr.lat.between(a,c)&cr.lng.between(b,d)].copy();q['included']=q.h3.isin(f.h3)
    for year,z in q.groupby('year'):
        rows.append({'city':city,'year':year,'bbox_crashes':len(z),'included':int(z.included.sum()),'excluded':int((~z.included).sum()),'share_excluded':float((~z.included).mean())})
    z=q[~q.included].groupby(['lat','lng','h3']).agg(n=('year','size'),first_year=('year','min'),last_year=('year','max')).reset_index().nlargest(20,'n');z['city']=city;bad.append(z)
pd.DataFrame(rows).to_csv(OUT/'coordinate_coverage_by_year.csv',index=False);pd.concat(bad).to_csv(OUT/'excluded_coordinate_clusters.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False));print(pd.concat(bad).groupby('city').head(5).to_string(index=False))
