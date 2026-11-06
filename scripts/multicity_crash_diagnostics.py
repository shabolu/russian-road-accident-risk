"""Supplementary crash composition and spatial audit maps, no changes to source data."""
import json,sys
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reports/multicity_350k';OUT=ROOT/'docs/analysis/multicity_350k';FIG=OUT/'figures'
cities=json.loads((BASE/'cities.json').read_text());cr=pd.read_parquet(BASE/'crashes.parquet');rows=[]
for c in cities:
    city=c['slug'];z=cr[cr.city==city]
    for period,mask in [('2015–2016',z.year<=2016),('2017–2022',z.year.between(2017,2022)),('2023–2025',z.year>=2023)]:
        zz=z[mask]
        for typ,t in zz.groupby('accident_type'):
            rows.append({'city':city,'period':period,'type':typ,'n':len(t),'share':len(t)/len(zz),'deaths':int(t.dead_count.sum()),'injuries':int(t.injured_count.sum()),'outside_share':(~t.in_road_cell).mean(),'far100_share':(t.road_distance_sample_m>100).mean()})
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,period,mask in zip(axs,['2015–2016','2017–2025'],[z.year<=2016,z.year>=2017]):
        zz=z[mask];outside=zz[~zz.in_road_cell]
        ax.scatter(zz.lng,zz.lat,s=2,color='#a7b2bf',alpha=.3,label='Все ДТП периода')
        sc=ax.scatter(outside.lng,outside.lat,s=10,color='#c84938',alpha=.8,label='Вне дорожных ячеек')
        xy=np.array([c['lon'],c['lat']]);ax.scatter([xy[0]],[xy[1]],marker='x',s=30,c='black',label='Центральный узел OSM')
        ax.set_title(period+f' · исключено {len(outside)} / {len(zz)}');ax.set_xlabel('Долгота');ax.set_ylabel('Широта');ax.set_aspect(1/np.cos(np.radians(c['lat'])));ax.legend(fontsize=7)
    fig.suptitle(c['name']+' · аудит координат в муниципальной когорте\nСовременная сеть: красная точка не означает доказанную ошибку')
    fig.savefig(FIG/f"coordinate_map_{city}.png",dpi=150);plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'accident_types_by_period.csv',index=False)
print('14 coordinate comparison maps and accident-type table ready')
