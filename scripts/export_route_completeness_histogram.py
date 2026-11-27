"""Export honest whole-way completeness and joint local length distribution to Downloads."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reports/route_proxy_coverage'
KEYS=['combined_speed_length_share','combined_surface_coarse_length_share','combined_lanes_length_share','combined_width_length_share']
COLORS=['#4e9276','#8eb095','#c4bd79','#d89a6a','#bd6e60']
def main():
    dest=Path.home()/'Downloads'/'Полнота_данных_дорог_OSM_СКДФ';dest.mkdir(exist_ok=True)
    cities=json.loads((ROOT/'reports/multicity_350k/cities.json').read_text());count=np.zeros(5,dtype=int);blank=0;zero_partial=0;all_partial=0;bycity=[]
    joint=[]
    for c in cities:
        city=c['slug'];w=pd.read_parquet(BASE/'ways'/f'{city}.parquet');a=w[KEYS].to_numpy()
        full=np.isclose(a,1,atol=1e-9,rtol=0).sum(axis=1);none=(a==0).all(axis=1);partial=((a>0)&~np.isclose(a,1,atol=1e-9,rtol=0)).any(axis=1)
        count+=np.bincount(full,minlength=5);blank+=int(none.sum());zero_partial+=int(((full==0)&~none).sum());all_partial+=int(partial.sum())
        for k in range(5):bycity.append({'Город':c['name'],'Признаков известно на всей длине':k,'Неполных или отсутствующих признаков':4-k,'Число объектов OSM':int((full==k).sum()),'Доля объектов, %':100*(full==k).mean(),'Совсем без четырёх параметров':int(((full==k)&none).sum()),'Есть частичная заполненность хотя бы одного параметра':int(((full==k)&partial).sum())})
        j=pd.read_csv(BASE/f'{city}_joint_completeness.csv');joint.append(j)
    joint=pd.concat(joint,ignore_index=True);length=joint.groupby('known_features').known_length_km.sum().reindex(range(5),fill_value=0);N=int(count.sum());km=length.sum()
    assert N==int(pd.read_csv(BASE/'coverage.csv').query("criterion=='base_class'").n_osm_ways.sum())
    assert count[4]==int(pd.read_csv(BASE/'coverage.csv').query("criterion=='combined_full4_coarse'").ways_full.sum())
    assert np.isclose(length[4],pd.read_csv(BASE/'coverage.csv').query("criterion=='combined_full4_coarse'").known_km.sum())
    assert count[0]==blank+zero_partial
    summary=pd.DataFrame({'Известно признаков':list(range(4,-1,-1)),'Не хватает признаков':list(range(5)),'Число объектов с этим количеством полностью известных параметров':count[::-1],'Доля объектов, %':100*count[::-1]/N,'Длина с этим количеством доступных параметров, км':length.to_numpy()[::-1],'Доля длины, %':100*length.to_numpy()[::-1]/km})
    summary.to_csv(dest/'Сводка_полноты.csv',sep=';',encoding='utf-8-sig',index=False)
    pd.DataFrame(bycity).to_csv(dest/'Полнота_по_городам.csv',sep=';',encoding='utf-8-sig',index=False)
    joint.rename(columns={'city':'Код города','name':'Город','known_features':'Известно признаков','missing_features':'Не хватает признаков','known_length_km':'Длина, км','n_samples':'Число проверенных образцов'}).to_csv(dest/'Полнота_длины_по_городам.csv',sep=';',encoding='utf-8-sig',index=False)
    x=np.arange(5);labels=['Все 4','Не хватает 1','Не хватает 2','Не хватает 3','0 известных*']
    fig,axs=plt.subplots(1,2,figsize=(17,8));fig.subplots_adjust(left=.07,right=.98,top=.72,bottom=.25,wspace=.22)
    vals=[count[::-1],length.to_numpy()[::-1]];denoms=[N,km]
    for ax,v,denom,unit,title in zip(axs,vals,denoms,['объектов OSM','км'],['По числу объектов OSM\nПризнак известен на всей обследованной длине объекта','По длине дорожной сети\nНабор известных признаков определяется локально']):
        bars=ax.bar(x,v,color=COLORS,width=.7)
        for bar,n in zip(bars,v):
            number=f'{int(n):,}'.replace(',',' ') if unit!='км' else f'{n:,.1f}'.replace(',',' ')
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(v)*.025,f'{number}\n{100*n/denom:.1f}%',ha='center',va='bottom',fontsize=11)
        ax.set_xticks(x,labels,fontsize=10);ax.set_ylim(0,max(v)*1.23);ax.set_ylabel(unit,fontsize=12);ax.set_title(title,fontsize=12,pad=15);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p,u=unit:f'{v/1000:g} тыс.' if max(vals[0])>10000 and u!='км' else f'{v:,.0f}'.replace(',',' ')))
    fig.suptitle('Полнота данных OSM + СКДФ для оценки скорости дороги',fontsize=19,y=.96)
    fig.text(.5,.885,f'14 городов · {N:,} объектов OSM · {km:,.1f} км'.replace(',',' '),ha='center',fontsize=13)
    fig.text(.5,.845,'Четыре параметра: ограничение скорости · покрытие/дорожная одежда · полосность · ширина',ha='center',fontsize=12)
    note=(f'* Слева 0 означает: ни один параметр не известен на всей длине объекта.\n'
          f'  Из этой группы {blank:,} объектов совсем без четырёх параметров; {zero_partial:,} имеют частичные данные.\n'
          'Справа 0 означает отсутствие всех четырёх параметров на данном отрезке.\n'
          'OSM-way — текущая единица учёта, до построения маршрутного графа. Класс дороги есть у всех и в четвёрку не входит.\n'
          'Учтены консервативное сопоставление СКДФ и ограничения для раздельных проезжих частей; шаг обследования ≤20 м.').replace(',',' ')
    fig.text(.07,.035,note,fontsize=10,va='bottom',linespacing=1.6)
    fig.savefig(dest/'Гистограмма_полноты_данных.png',dpi=190);fig.savefig(dest/'Гистограмма_полноты_данных.pdf');plt.close(fig)
    # Supplementary per-city breakdown of complete attributes on OSM ways.
    citydf=pd.DataFrame(bycity);tab=citydf.pivot(index='Город',columns='Признаков известно на всей длине',values='Число объектов OSM').reindex([c['name'] for c in cities]);shares=tab.div(tab.sum(axis=1),axis=0)*100
    fig,ax=plt.subplots(figsize=(13,8),layout='constrained');left=np.zeros(14)
    for k,color in zip(range(4,-1,-1),COLORS):ax.barh(np.arange(14),shares[k],left=left,label=f'{k} из 4',color=color);left+=shares[k].to_numpy()
    ax.set_yticks(range(14),shares.index);ax.set_xlim(0,100);ax.set_xlabel('Доля объектов OSM, %');ax.set_title('Число параметров, известных на всей длине объекта\n0 включает частично заполненные объекты; подробности — в сводке');ax.legend(title='Полностью известны',loc='upper left',bbox_to_anchor=(1,1));fig.savefig(dest/'Полнота_по_городам.png',dpi=170);plt.close(fig)
    (dest/'Пояснение.txt').write_text('Исходные источники: OSM 03.06.2026 и СКДФ, выгруженный 06.11.2026.\nЧетыре дополнительных параметра: ограничение скорости, покрытие/крупный тип дорожной одежды, полосность, ширина.\nКласс дороги не включён: он известен у всех отобранных объектов и позволяет назначать базовую скорость.\nЛевая панель: подсчитаны OSM-ways по числу параметров, известных на всей обследованной длине внутри округа. Это не окончательные направленные рёбра маршрутного графа.\nПравая панель: локальная доступность параметров по длине сети. Использует совместное наличие полей, а не сумму их отдельных покрытий.\nУ одного объекта паспорт может меняться по длине. Поэтому 0 полностью известных параметров не всегда означает отсутствие любых данных.\nСовсем без четырёх полей: '+str(blank)+' объектов; с частичными данными, но без полного покрытия хотя бы одного параметра: '+str(zero_partial)+'.\nВсе четыре известны на всей обследованной длине у '+str(count[4])+' объектов.\nГеометрия проверена образцами с шагом не более 20 м. Назначение скоростей и проверка проезжаемости не выполнялись.\nCSV в UTF-8 с BOM, разделитель точка с запятой.\n')
    # Durable numeric audit accompanies the Downloads export.
    cache=ROOT/'docs/analysis/route_proxy/coverage';summary.to_csv(cache/'completeness_histogram.csv',index=False);pd.DataFrame(bycity).to_csv(cache/'completeness_histogram_by_city.csv',index=False)
    print(dest);print(summary.to_string(index=False));print('true_blank',blank,'zero_full_but_partial',zero_partial,'any_partial_objects',all_partial)
if __name__=='__main__':main()
