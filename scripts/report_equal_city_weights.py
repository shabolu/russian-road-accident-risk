"""Equal-city evaluation of training weights, with paired spatial uncertainty."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,h3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from road_accident_risk import v2_eval_year as vy
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reports/multicity_350k';DATA=BASE/'equal_city_weights';OUT=ROOT/'docs/analysis/multicity_350k/equal_city_weights'
def table(t):
    t=t.copy()
    for c in t.select_dtypes('number'):t[c]=t[c].map(lambda v:f'{v:.3f}')
    return '| '+' | '.join(t.columns)+' |\n| '+' | '.join(['---']*len(t.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in t.itertuples(index=False,name=None))
def main():
    OUT.mkdir(exist_ok=True,parents=True);cities=json.loads((BASE/'cities.json').read_text());names={c['slug']:c['name'] for c in cities};order=list(names)
    row=pd.read_csv(BASE/'metrics.csv');row=row[(row.kind=='loco')&(row.window=='all_2015_2022')];eq=pd.read_csv(DATA/'metrics.csv');keys=['target','seed','subset','period'];assert not row[keys].duplicated().any() and not eq[keys].duplicated().any()
    columns=['coverage_top_1pct','coverage_top_5pct','coverage_top_10pct','coverage_oracle_top_10pct','calib_ratio','spearman','poisson_deviance']
    merged=row[keys+columns].merge(eq[keys+columns],on=keys,suffixes=('_row','_equal'),validate='one_to_one');assert len(merged)==14*3*2*4
    merged.to_csv(OUT/'paired_seed_metrics.csv',index=False)
    city=merged.groupby(['target','subset','period']).mean(numeric_only=True).reset_index()
    for pct in [1,5,10]:city[f'delta_cov{pct}_pp']=100*(city[f'coverage_top_{pct}pct_equal']-city[f'coverage_top_{pct}pct_row'])
    city.to_csv(OUT/'city_comparison.csv',index=False)
    macros=[]
    for (subset,period),g in city.groupby(['subset','period']):
        for method in ['row','equal']:
            x=g[f'coverage_top_10pct_{method}'];cal=g[f'calib_ratio_{method}'];oracle=g[f'coverage_oracle_top_10pct_{method}']
            macros.append({'subset':subset,'period':period,'method':method,'n_cities':len(g),'macro_cov1':g[f'coverage_top_1pct_{method}'].mean(),'macro_cov5':g[f'coverage_top_5pct_{method}'].mean(),'macro_cov10':x.mean(),'city_p25_cov10':x.quantile(.25),'worst_city_cov10':x.min(),'worst_city':g.loc[x.idxmin(),'target'],'macro_fraction_realized':(x/oracle).mean(),'macro_abs_log_calib':np.abs(np.log(cal)).mean(),'macro_poisson_deviance':g[f'poisson_deviance_{method}'].mean()})
    macro=pd.DataFrame(macros);macro.to_csv(OUT/'macro_metrics.csv',index=False)
    boot=[]
    for target in order:
        b=pd.read_parquet(BASE/f'{target}_predictions.parquet');b=b[b.window=='all_2015_2022'];e=pd.read_parquet(DATA/f'{target}_predictions.parquet')
        z=b.merge(e,on=['target','seed','h3'],suffixes=('_row','_equal'),validate='one_to_one');assert (z.observed_row==z.observed_equal).all()
        for subset in ['municipality','radius15km']:
            zz=z if subset=='municipality' else z[z.center_distance_km_row<=15]
            g=zz.groupby('h3').agg(y=('observed_row','first'),row=('predicted_row','mean'),equal=('predicted_equal','mean'))
            groups=pd.Series(g.index).map(lambda h:h3.cell_to_parent(h,7));ixs=[np.flatnonzero(groups==v) for v in sorted(groups.unique())];rng=np.random.default_rng(20261106);draws=[]
            y,a,b=g.y.to_numpy(),g.row.to_numpy(),g.equal.to_numpy()
            for _ in range(500):
                ix=np.concatenate([ixs[j] for j in rng.integers(0,len(ixs),len(ixs))]);draws.append(np.array(vy.coverages(y[ix],b[ix]))-vy.coverages(y[ix],a[ix]))
            samples=np.array(draws);delta=np.array(vy.coverages(y,b))-vy.coverages(y,a)
            for k,pct in enumerate([1,5,10]):boot.append({'target':target,'subset':subset,'top_pct':pct,'delta_ensemble_pp':100*delta[k],'ci_low_pp':100*np.quantile(samples[:,k],.025),'ci_high_pp':100*np.quantile(samples[:,k],.975),'n_blocks':len(ixs)})
    pd.DataFrame(boot).to_csv(OUT/'paired_spatial_bootstrap.csv',index=False)
    # Descriptive uncertainty across this finite collection of cities; folds share training data.
    cm=[]
    for subset in ['municipality','radius15km']:
        d=city[(city.subset==subset)&(city.period=='sum')].delta_cov10_pp.to_numpy();rng=np.random.default_rng(20261106);draw=d[rng.integers(0,len(d),(5000,len(d)))].mean(axis=1)
        cm.append({'subset':subset,'mean_delta_pp':d.mean(),'median_delta_pp':np.median(d),'improved':int((d>1e-9).sum()),'worsened':int((d< -1e-9).sum()),'unchanged':int((abs(d)<=1e-9).sum()),'descriptive_city_resampling_low_pp':np.quantile(draw,.025),'descriptive_city_resampling_high_pp':np.quantile(draw,.975)})
    pd.DataFrame(cm).to_csv(OUT/'macro_deltas.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(13,7),layout='constrained')
    for ax,subset in zip(axs,['municipality','radius15km']):
        g=city[(city.period=='sum')&(city.subset==subset)].set_index('target').reindex(order);values=g.delta_cov10_pp
        ax.barh(range(14),values,color=['#308b64' if v>=0 else '#c06048' for v in values]);ax.axvline(0,color='black',lw=1);ax.set_yticks(range(14),[names[c] for c in order]);ax.set_xlabel('Изменение охвата топ-10%, процентные пункты');ax.set_title('Городские округа' if subset=='municipality' else 'Округ ∩ радиус 15 км')
    fig.suptitle('Равный суммарный вес городов против равного веса строк\nКаждый целевой город полностью исключён из обучения');fig.savefig(OUT/'city_deltas.png',dpi=170);plt.close(fig)
    maincity=city[(city.subset=='municipality')&(city.period=='sum')].set_index('target').reindex(order);bt=pd.DataFrame(boot);bt=bt[(bt.subset=='municipality')&(bt.top_pct==10)].set_index('target').reindex(order)
    display=pd.DataFrame({'Город':[names[c] for c in order],'Равные строки, %':100*maincity.coverage_top_10pct_row.to_numpy(),'Равные города, %':100*maincity.coverage_top_10pct_equal.to_numpy(),'Δ, п.п.':maincity.delta_cov10_pp.to_numpy(),'ДИ Δ ансамбля, ниж.':bt.ci_low_pp.to_numpy(),'ДИ Δ ансамбля, верх.':bt.ci_high_pp.to_numpy()})
    macrodisplay=macro[(macro.subset=='municipality')&(macro.period=='sum')][['method','macro_cov10','city_p25_cov10','worst_city_cov10','macro_fraction_realized','macro_abs_log_calib']]
    point=next(c for c in cm if c['subset']=='municipality');delta=point['mean_delta_pp']
    text=f'''# Равный вклад населённых пунктов: проверка обучения с весами городов

2026-11-06. Цель пользователя: одинаково хорошо работать в каждом населённом пункте.
Основная единица оценки — целый исключённый населённый пункт с равным вкладом в среднюю метрику.
Численное равенство coverage между городами не гарантируется и не требуется: концентрация и шум ДТП различаются.
Поэтому показываются также нижняя четверть, худший город и отношение к реализованному ориентиру.

## Протокол

14 современных муниципальных когорт >350 тыс. из основного отчёта; результаты нельзя автоматически
обобщать на малые города, посёлки и сёла. Округ включает сельскую периферию; подвыборка радиусом 15 км
проверяется отдельно. Строго говоря, сейчас уравнивается вклад этих муниципальных когорт.

Для каждого целевого города модель обучается на остальных 13. 28 OSM-признаков, годовые строки
2015–2022, тест 2023–2025, три одинаковых сида 42/7/123. CatBoost Tweedie 1.3, 800/.05/6.
Параметры и формула весов зафиксированы без подбора. Старая модель с равными весами строк взята
из сохранённого основного эксперимента; новые 42 модели обучены с весом строки:

`w_i = N_total / (K_train_cities × N_rows_in_city)`.

Средний вес равен 1, суммарный вес каждого обучающего города одинаков. Проверка сохранена
в `reports/multicity_350k/equal_city_weights/weights.csv` для всех фолдов.
Число ДТП и набор строк не изменяются. Train-only p99-клиппинг остаётся тем же невзвешенным,
чтобы проверить именно изменение sample_weight; далее можно отдельно изучить подготовку признаков.
Информация о ДТП целевого города не участвует в обучении, вычислении весов или порогов.
Равный городской вклад в функцию потерь — проверяемое средство, а не гарантия равного качества.

## Результат

Среднее изменение coverage@10 по 14 городам: **{delta:+.3f} п.п.**.
Улучшение в {point['improved']}, ухудшение в {point['worsened']}, без изменения в {point['unchanged']} городах.
Сравнение остальных показателей и таблица городов ниже важнее одного среднего значения.

{table(macrodisplay)}

Метрики coverage и отношение к реализованному ориентиру в таблице выше даны долями, ошибка
калибровки — `mean_city(abs(log(sum_pred/sum_observed)))`, меньше лучше. `city_p25_cov10` — 25-й
перцентиль качества городов; `worst_city_cov10` — минимальное качество. Это показатели
неоднородности результатов, они не выделяют причинный эффект веса на каждом типе территории.

{table(display)}

![Разность по городам](city_deltas.png)

Основные точки — средние метрик трёх сидов. Интервалы для разности качества **ансамблей прогнозов**:
500 парных пересэмплирований пространственных блоков H3 res 7 внутри каждого города. Поэтому
среднее сидовых Δ и точка ансамбля могут немного различаться. Поправки на множественные сравнения нет.
`macro_deltas.csv` дополнительно содержит описательное пересэмплирование 14 городов;
фолды разделяют обучающие данные, поэтому это не строгий независимый ДИ для всех городов России.

## Интерпретация и ограничения

Предпочтение пользователя относится к цели оценки и сохраняется независимо от того, улучшает ли
конкретная формула весов результат. Нужно сравнивать одновременно macro, нижнюю четверть,
худшие территории, калибровку и нормирование на концентрацию событий.
Сильное улучшение среднего при ухудшении отдельных городов не означает выполнения цели «хорошо везде».
И наоборот, равные веса могут не помочь: разный поток, качество карты, состав ДТП и скрытые признаки
одним изменением весов не устраняются.

Исследование разведочное: эти города уже использованы для анализа и разработки. Любую выбранную
стратегию следует проверять на новых городах; малые населённые пункты нужны отдельно.
Современный OSM позднее теста, история сети не восстановлена; приоритеты и ограничения основного
отчёта сохраняются. Никаких весов по сходству и подбора их параметров в этом эксперименте нет.

## Воспроизведение

```
./venv/bin/python scripts/multicity_equal_city_weights.py
./venv/bin/python scripts/report_equal_city_weights.py
```

Нужны локальные кэши основного опыта. Численные таблицы находятся рядом с отчётом;
протокол, проверки весов и сырые метрики — `reports/multicity_350k/equal_city_weights`.
'''
    (OUT/'README.md').write_text(text);print(macrodisplay.to_string(index=False));print(point);print(OUT/'README.md')
if __name__=='__main__':main()
