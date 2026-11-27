"""Report attribute availability separately from a fully specified speed/routing model."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reports/route_proxy_coverage';OUT=ROOT/'docs/analysis/route_proxy/coverage'
LABELS={'osm_speed':'Скорость-ограничение: OSM','combined_speed':'Скорость-ограничение: OSM + СКДФ','osm_surface':'Покрытие: OSM','combined_surface_coarse':'Покрытие / дорожная одежда: OSM + СКДФ','osm_lanes':'Полосность: OSM','combined_lanes':'Полосность: OSM + СКДФ','osm_width':'Ширина: OSM','combined_width':'Ширина: OSM + СКДФ'}
def table(t):
    t=t.copy()
    for c in t.select_dtypes('number'):
        if c.startswith('Число') or c.startswith('Всего') or c.startswith('Полностью'):t[c]=t[c].map(lambda x:f'{int(x):,}'.replace(',',' '))
        else:t[c]=t[c].map(lambda x:f'{x:.1f}')
    return '| '+' | '.join(t.columns)+' |\n| '+' | '.join(['---']*len(t.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in t.itertuples(index=False,name=None))
def main():
    OUT.mkdir(exist_ok=True,parents=True);data=pd.read_csv(BASE/'coverage.csv');classes=pd.read_csv(BASE/'by_class.csv');meta=json.loads((BASE/'metadata.json').read_text());cities=json.loads((ROOT/'reports/multicity_350k/cities.json').read_text());order=[c['slug'] for c in cities];names={c['slug']:c['name'] for c in cities}
    data.to_csv(OUT/'coverage_long.csv',index=False);classes.to_csv(OUT/'coverage_by_class.csv',index=False)
    full=data[data.criterion=='combined_full4_coarse'].set_index('city').reindex(order);speed=data[data.criterion=='combined_speed'].set_index('city').reindex(order);detail=data[data.criterion=='combined_full4_detailed'].set_index('city').reindex(order);potential=data[data.criterion=='potential_full4_coarse'].set_index('city').reindex(order)
    summary=pd.DataFrame({'Город':[names[c] for c in order],'Всего объектов OSM':full.n_osm_ways.to_numpy(),'Длина сети, км':full.total_road_km.to_numpy(),'Полностью обеспечены, объектов':full.ways_full.to_numpy(),'Число с покрытием ≥90% длины':full.ways_ge90pct.to_numpy(),'Обеспеченная длина, км':full.known_km.to_numpy(),'Обеспеченная длина, %':full.length_share.to_numpy()*100,'Детальное покрытие, % длины':detail.length_share.to_numpy()*100,'Потенциальный набор без фильтров, % длины':potential.length_share.to_numpy()*100,'Известна скорость-ограничение, % длины':speed.length_share.to_numpy()*100})
    summary.to_csv(OUT/'cities_summary_ru.csv',sep=';',encoding='utf-8-sig',index=False)
    attr=data[data.criterion.isin(LABELS)].pivot(index='name',columns='criterion',values='length_share');attr.to_csv(OUT/'attribute_length_shares.csv')
    street=[]
    for city,g in classes[classes.highway!='service'].groupby('city'):
        for criterion,z in g.groupby('criterion'):street.append({'city':city,'name':names[city],'criterion':criterion,'n_ways':int(z.n_ways.sum()),'total_km':z.total_km.sum(),'known_km':z.known_km.sum(),'length_share':z.known_km.sum()/z.total_km.sum(),'ways_ge90pct':int(z.ways_ge90pct.sum())})
    pd.DataFrame(street).to_csv(OUT/'street_without_service.csv',index=False)
    fig,ax=plt.subplots(figsize=(13,7),layout='constrained');y=np.arange(14)
    for j,(criterion,label) in enumerate([('osm_full4','Все четыре поля только OSM'),('combined_full4_coarse','OSM + СКДФ: консервативное сопоставление'),('potential_full4_coarse','Потенциально: до фильтра неоднозначностей')]):
        vals=data[data.criterion==criterion].set_index('city').reindex(order).length_share.to_numpy()*100;ax.barh(y+(j-1)*.23,vals,.23,label=label)
    ax.set_yticks(y,[names[c] for c in order]);ax.set_xlabel('Доля длины автомобильных дорог внутри округа, %');ax.set_title('Данные для расширенной формулы скорости\nОграничение скорости + покрытие/дорожная одежда + полосность + ширина');ax.legend(fontsize=8);fig.savefig(OUT/'full_attribute_coverage.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(16,7),layout='constrained')
    for ax,(os,sk,title) in zip(axs,[('osm_speed','combined_speed','Ограничение скорости'),('osm_surface','combined_surface_coarse','Покрытие / дорожная одежда'),('osm_lanes','combined_lanes','Полосность'),('osm_width','combined_width','Ширина')]):
        for j,(crit,label) in enumerate([(os,'OSM'),(sk,'OSM + СКДФ')]):
            vals=data[data.criterion==crit].set_index('city').reindex(order).length_share.to_numpy()*100;ax.barh(y+(j-.5)*.35,vals,.35,label=label)
        ax.set_yticks(y,[names[c] for c in order]);ax.set_title(title);ax.set_xlabel('% длины');ax.set_xlim(0,100)
    axs[0].legend(fontsize=8);fig.savefig(OUT/'per_attribute_coverage.png',dpi=160);plt.close(fig)
    totalways=int(full.n_osm_ways.sum());totalkm=full.total_road_km.sum();knownkm=full.known_km.sum();fullways=int(full.ways_full.sum());osm=data[data.criterion=='osm_full4'];omshare=osm.known_km.sum()/osm.total_road_km.sum()
    report=f'''# Покрытие атрибутов для формулы скорости: OSM + СКДФ, 14 городов

2026-11-06. Это фактический аудит доступности атрибутов, а не проверка точности расчётной скорости.
Формула скорости пока является конструкцией: численные таблицы классов/покрытий и штрафов не утверждены.

## Главное

- Базовую скорость по классу можно назначить всем {totalways:,} объектам автомобильных дорог
  ({totalkm:,.1f} км внутри округов). Это использование значений по умолчанию, без гарантии точности
  или навигационной доступности. Направления, барьеры, запреты и манёвры этим аудитом не проверены.
- Более требовательный вариант: явно известны ограничение скорости, покрытие/тип дорожной одежды,
  полосность и ширина. После консервативного сопоставления OSM ↔ СКДФ доступны {knownkm:,.1f} км,
  или **{100*knownkm/totalkm:.1f}%** общей длины. У {fullways:,} OSM-ways набор доступен на всей
  обследованной длине внутри округа.
- Только OSM обеспечивал все четыре поля на {100*omshare:.2f}% общей длины.
- Использование точного материала OSM surface строже, чем крупного класса одежды СКДФ:
  эти варианты показаны отдельно. Не декодированные коды материала pavement_type_gid не выдаются за асфальт/грунт.
- Суммы объединённых муниципальных выгрузок описывают объекты внутри своих округов; возможные
  общие линии на административной границе могут считаться в обоих округах. Для сравнения городов
  используются отдельные знаменатели, глобальные суммы не заявляются как уникальная национальная сеть.

## Сколько объектов обеспечено данными

«Дорога» здесь — **OSM-way с ненулевой длиной внутри муниципального полигона**. Это не именованная
улица, не дорога целиком по реестру и не готовое направленное ребро маршрутизатора. Один way может
пересекать несколько паспортных участков с разным заполнением. Длина известной части учитывается
даже тогда, когда весь way не заполнен. Небольшие различия с H3-отчётом объясняются точным обрезанием
геометрии по полигону вместо выбора целых H3-ячеек по центроидам.

{table(summary[['Город','Всего объектов OSM','Длина сети, км','Полностью обеспечены, объектов','Обеспеченная длина, км','Обеспеченная длина, %']])}

Полностью обеспеченный объект — все его образцы имеют четыре атрибута. Дополнительная менее строгая
оценка «≥90% длины way» и детальный вариант покрытия сохранены в `cities_summary_ru.csv`.
Геометрия обследуется образцами не реже 20 м: короткие разрывы между образцами могут быть пропущены,
поэтому «полностью» относится к обследованной длине с этой дискретизацией.

![Полный набор атрибутов](full_attribute_coverage.png)
![Каждый атрибут отдельно](per_attribute_coverage.png)

## Источники и сопоставление

- OSM: локальный PBF 2026-06-03; существующие буферные выгрузки 14 городов; все автомобильные
  классы, включая service. Отдельная таблица улиц без service — `street_without_service.csv`.
- СКДФ: свежая полная выгрузка для bbox каждого буферного округа, слой
  `skdf_open:lyr_road_conditions_road_normative_condition`, WFS
  `https://скдф.рф/api-geoserver/skdf_open/wfs`. Поля `speed_limit`, `lanes`, `roadway_width`,
  `pavement_kind_gid`, дополнительно сохранён `pavement_type_gid`. Старую саратовскую выгрузку не подменяли.
  Счётчики страниц и уникальность паспортных id проверены. Даты/параметры — `reports/route_proxy_coverage/metadata.json`.
- Обрезание дорог по современному муниципальному полигону, проекция местной зоны UTM.
  Участки СКДФ обрезаются полигоном с 50-м буфером для сопоставления у края территории.
- Для каждого образца OSM выбирается ближайший параллельный сегмент СКДФ: расстояние ≤20 м,
  разница направлений по модулю 180° ≤25°. Геометрические расстояния считаются до сегментов точно;
  дискретизация относится к местам проверки вдоль дороги.
- Если в пределах расстояния лучшего кандидата +5 м присутствует другой road_id или конфликтуют
  паспортные атрибуты, сопоставление считается неоднозначным и данные СКДФ для строгой оценки не используются.
  Это эвристика уверенности, не доказательство правильности оставшихся совпадений; названия/топология
  и реальные маршруты не проверены. Самый близкий кандидат при равных расстояниях выбирается детерминированно.
- На односторонние OSM-ways паспортные lanes/roadway_width **не переносятся**, поскольку СКДФ может
  описывать обе проезжие части суммарно. Если нужные поля уже есть в OSM, они используются.
  Это намеренно осторожное правило, которое недооценивает доступные данные для некоторых настоящих
  односторонних дорог. «Потенциальный» вариант показывает наличие до этих фильтров, не готовность к применению.

## Какие значения считаются известными

- Скорость: числовой maxspeed OSM (включая mph и RU:urban/rural/living_street) или speed_limit СКДФ,
  5–150 км/ч. При directional-тегах без общего значения требуются оба направления;
  для оценки заполненности берётся минимум. Полный профиль должен хранить направления раздельно.
- Полосность: целое 1–20. Допускается сумма известных lanes:forward/backward, когда нет общего lanes.
- Ширина: 1–100 м, распознаются обычные числовые значения и единицы m. Диапазоны и неоднозначные
  строки не принимаются. Это широкие фильтры правдоподобия, не сертификат фактической корректности.
- Покрытие: распознанный материал OSM surface или крупная расшифрованная категория дорожной одежды
  СКДФ. Неподтверждённые legacy-коды 2043912/2043913 не используются.
- Smoothness: только распознанные значения OSM, отдельно от типа дорожной одежды. Его наличие
  показывает осведомлённость о состоянии, но impassable означает невозможность проезда.

Назначение измеренной скорости по этим данным не выполняется: maxspeed и паспортная скорость — ограничения,
а действительная скорость зависит также от потока, светофоров и прочего. Отсутствующие параметры в
предложенной формуле могут оставаться нейтральными и давать базовый профиль. Поэтому неполный набор
четырёх атрибутов не означает, что ребро обязательно нужно удалять из графа.

## Проверки и воспроизводимость

В каждой муниципальной выборке проверены уникальность OSM id, баланс длины исходных линий
и суммарных весов образцов, границы долей [0;1]. Данные не переносятся с агрегатов H3 на отдельные дороги.
Подробности по классам — `coverage_by_class.csv`, все критерии — `coverage_long.csv`.
Результаты зависят от геометрической точности реестра и выбранных допусков; ручная проверка соответствий
и исследование других порогов пока не выполнены. Снимки OSM и СКДФ имеют разные даты.

```
./venv/bin/python scripts/fetch_multicity_skdf.py
./venv/bin/python scripts/route_attribute_coverage.py
./venv/bin/python scripts/report_route_attribute_coverage.py
```

Сырые GeoJSON — `data/raw/skdf/route_coverage_20261106`; покрытие каждого way —
`reports/route_proxy_coverage/ways/` (локальные parquet). Итоговые таблицы/метаданные —
`reports/route_proxy_coverage`; данные этого отчёта и рисунки — рядом с README.
'''
    (OUT/'README.md').write_text(report);print(summary.to_string(index=False));print('TOTAL ways',totalways,'full4 ways',fullways,'km',totalkm,'known%',100*knownkm/totalkm)
if __name__=='__main__':main()
