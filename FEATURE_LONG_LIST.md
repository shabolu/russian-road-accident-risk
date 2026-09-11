# Feature Longlist

Цель этого списка - собрать группы внешних признаков, которые можно добавлять постепенно и сравнивать экспериментами.

Основная идея: строить **structural risk model**, то есть модель, которая оценивает риск дорожной среды без опоры на историческую аварийность в конкретной H3-ячейке. Это важно для новых и проектируемых дорог, где история ДТП равна нулю.

Одна строка моделирования:

```text
H3-ячейка x месяц
```

Целевая переменная:

```text
target_had_accident_next_month
```

## 1. Дорожная структура

Основные источники:

- OpenStreetMap;
- Geofabrik Russia extract: https://download.geofabrik.de/russia.html;
- Overpass API для маленьких bbox;
- Overture Maps transportation layer.

Статус: **реализовано** в `h3_road_features.parquet` и `h3_month_road_features.parquet`.

Фактическая реализация использует локальный Geofabrik PBF:

```text
data/raw/osm/volga-fed-district-260603.osm.pbf
```

### 1.1. Длина и количество дорог

| Признак | Смысл | Статус / комментарий |
|---|---|---|
| `road_length_total_m` | Общая длина дорог в H3 | Реализовано; базовая дорожная exposure |
| `road_way_count` | Количество OSM ways в H3 | Реализовано; proxy фрагментации/плотности |
| `road_segment_h3_count` | Количество H3-сегментов дорог | Реализовано; proxy плотности сегментации |
| `major_road_length_m` | Длина major roads | Реализовано |
| `link_road_length_m` | Длина `*_link` дорог | Реализовано |

### 1.2. Абсолютные длины по типам дорог

Эти признаки сохранены в parquet для диагностики, но в основном baseline исключаются из обучения, потому что дублируют связку:

```text
road_length_total_m * road_share_<type>
```

| Признак | Смысл | Статус / комментарий |
|---|---|---|
| `road_length_motorway_m` | Длина motorway | Реализовано, исключается из основного feature set |
| `road_length_motorway_link_m` | Длина motorway_link | Реализовано, исключается из основного feature set |
| `road_length_trunk_m` | Длина trunk | Реализовано, исключается из основного feature set |
| `road_length_trunk_link_m` | Длина trunk_link | Реализовано, исключается из основного feature set |
| `road_length_primary_m` | Длина primary | Реализовано, исключается из основного feature set |
| `road_length_primary_link_m` | Длина primary_link | Реализовано, исключается из основного feature set |
| `road_length_secondary_m` | Длина secondary | Реализовано, исключается из основного feature set |
| `road_length_secondary_link_m` | Длина secondary_link | Реализовано, исключается из основного feature set |
| `road_length_tertiary_m` | Длина tertiary | Реализовано, исключается из основного feature set |
| `road_length_tertiary_link_m` | Длина tertiary_link | Реализовано, исключается из основного feature set |
| `road_length_unclassified_m` | Длина unclassified | Реализовано, исключается из основного feature set |
| `road_length_residential_m` | Длина residential | Реализовано, исключается из основного feature set |
| `road_length_living_street_m` | Длина living_street | Реализовано, исключается из основного feature set |
| `road_length_service_m` | Длина service | Реализовано, исключается из основного feature set |

### 1.3. Доли типов дорог

Основной baseline использует именно `road_length_total_m + road_share_*`, а не абсолютные длины каждого типа.

| Признак | Смысл | Статус / комментарий |
|---|---|---|
| `major_road_share` | Доля крупных дорог | Реализовано |
| `link_road_share` | Доля link-дорог | Реализовано |
| `road_share_motorway` | Доля motorway | Реализовано |
| `road_share_motorway_link` | Доля motorway_link | Реализовано |
| `road_share_trunk` | Доля trunk | Реализовано |
| `road_share_trunk_link` | Доля trunk_link | Реализовано |
| `road_share_primary` | Доля primary | Реализовано |
| `road_share_primary_link` | Доля primary_link | Реализовано |
| `road_share_secondary` | Доля secondary | Реализовано |
| `road_share_secondary_link` | Доля secondary_link | Реализовано |
| `road_share_tertiary` | Доля tertiary | Реализовано |
| `road_share_tertiary_link` | Доля tertiary_link | Реализовано |
| `road_share_unclassified` | Доля unclassified | Реализовано |
| `road_share_residential` | Доля residential | Реализовано |
| `road_share_living_street` | Доля living_street | Реализовано |
| `road_share_service` | Доля service | Реализовано |
| `road_class_entropy` | Разнообразие классов дорог | Не реализовано; кандидат для группы 3 |
| `minor_road_share` | Доля локальных дорог | Не выделено отдельной колонкой; можно получить из `road_share_*` |

### 1.4. Полосность, скорость и атрибуты дороги

| Признак | Смысл | Статус / комментарий |
|---|---|---|
| `lane_meters_estimate` | Длина дорог с учетом числа полос | Реализовано |
| `lanes_mean_weighted` | Среднее число полос, взвешенное длиной | Реализовано |
| `lanes_max` | Максимальное число полос | Реализовано |
| `lanes_tagged_share` | Доля дорог с тегом `lanes` | Реализовано; контроль полноты OSM |
| `maxspeed_mean_weighted` | Среднее ограничение скорости, взвешенное длиной | Реализовано |
| `maxspeed_max` | Максимальное ограничение скорости | Реализовано |
| `maxspeed_tagged_share` | Доля дорог с исходным `maxspeed` | Реализовано |
| `maxspeed_is_imputed_share` | Доля дорог, где `maxspeed` был заполнен эвристикой | Реализовано |
| `oneway_length_m` | Длина односторонних дорог | Реализовано |
| `oneway_share` | Доля односторонних дорог | Реализовано |
| `lit_road_length_m` | Длина дорог с `lit=yes` | Реализовано |
| `lit_share` | Доля освещенных дорог | Реализовано |
| `bridge_length_m` | Длина дорог на мостах | Реализовано |
| `bridge_share` | Доля дорог на мостах | Реализовано |
| `tunnel_length_m` | Длина дорог в тоннелях | Реализовано |
| `tunnel_share` | Доля дорог в тоннелях | Реализовано |
| `asphalt_surface_length_m` | Длина дорог с асфальтовым/похожим покрытием | Реализовано |
| `asphalt_surface_share` | Доля таких дорог | Реализовано |
| `bad_surface_length_m` | Длина дорог с плохим/грунтовым покрытием | Реализовано |
| `bad_surface_share` | Доля плохого/грунтового покрытия | Реализовано |

## 2. Узлы, конфликты и дорожная инфраструктура

Основные источники:

- OpenStreetMap;
- Overpass API;
- муниципальные данные, если найдутся.

Статус: **частично реализовано**. Базовые дорожные точки и графовые перекрестки реализованы; остановки ОТ, POI, парковочные въезды, Ж/Д переезды и раздельные типы переходов еще не реализованы.

### 2.1. Графовые перекрестки

| Признак | Смысл | Статус / комментарий |
|---|---|---|
| `junction_count` | Количество графовых перекрестков | Реализовано |
| `junction_density_per_km` | Перекрестки на км дорог | Реализовано |
| `junction_degree_mean` | Средняя степень узлов | Реализовано |
| `junction_degree_max` | Максимальная степень узла | Реализовано |
| `three_way_junction_count` | Количество 3-way junctions | Реализовано |
| `four_plus_way_junction_count` | Количество junctions со степенью 4+ | Реализовано |
| `major_junction_count` | Узлы с участием major roads | Реализовано |
| `minor_to_major_junction_count` | Minor-to-major junction без близкого светофора | Реализовано; class-rank delta >= 1 и нет traffic signal в радиусе 30 м |
| `minor_to_major_junction_density_per_km` | Плотность таких узлов | Реализовано |
| `legacy_major_minor_junction_count` | Старый вариант major/minor logic | Реализовано для диагностики |

### 2.2. Class-delta conflict признаки

| Признак | Смысл | Статус / комментарий |
|---|---|---|
| `class_delta_junction_count` | Перекрестки дорог разных классов | Реализовано |
| `class_delta_mean` | Средняя разница классов в узле | Реализовано |
| `class_delta_max` | Максимальная разница классов | Реализовано |
| `class_delta_sum` | Сумма class-delta по узлам | Реализовано |
| `class_delta_junction_density_per_km` | Class-delta junctions на км дорог | Реализовано |
| `signalized_class_delta_junction_count` | Class-delta узлы со светофором рядом | Реализовано |
| `unsignalized_class_delta_junction_count` | Class-delta узлы без светофора рядом | Реализовано |
| `unsignalized_high_delta_junction_count` | Несветофорные узлы с высокой разницей классов | Реализовано |
| `unsignalized_class_delta_junction_density_per_km` | Плотность несветофорных class-delta узлов | Реализовано |

### 2.3. Точечная дорожная инфраструктура из OSM

| Признак | Смысл | Статус / комментарий |
|---|---|---|
| `traffic_signal_count` | Светофоры | Реализовано |
| `crossing_count` | Пешеходные переходы | Реализовано |
| `stop_count` | Знаки stop | Реализовано |
| `give_way_count` | Знаки уступи дорогу | Реализовано |
| `mini_roundabout_count` | Mini roundabout | Реализовано |
| `motorway_junction_count` | Motorway junction nodes | Реализовано |
| `traffic_calming_count` | Элементы успокоения движения | Реализовано |
| `crossing_signalized_count` | Регулируемые переходы | Не реализовано |
| `crossing_unsignalized_count` | Нерегулируемые переходы | Не реализовано |
| `bus_stop_count` | Остановки ОТ | Не реализовано |
| `tram_stop_count` | Трамвайные остановки | Не реализовано |
| `public_transport_stop_count` | Все остановки ОТ | Не реализовано |
| `railway_crossing_count` | Ж/Д переезды | Не реализовано |
| `roundabout_count` | Круговые движения | Частично: есть `mini_roundabout_count`, полноценные way/area roundabout еще не выделены |
| `parking_entrance_count` | Въезды/выезды парковок | Не реализовано |
| `driveway_count` | Подъезды/выезды | Не реализовано |
| `speed_bump_count` | Искусственные неровности | Частично входит в `traffic_calming_count`, отдельно не выделено |

## 3. Геометрия и связность дорожной сети

Основные источники:

- OpenStreetMap road geometries;
- Overture transportation;
- проектные линии новых дорог.

Статус: **еще не реализовано**.

Для группы 3 шаблон

```text
count + density + share
```

имеет смысл только для дискретных геометрических объектов и событий: сегментов, тупиков, резких поворотов. Для непрерывных геометрических характеристик дороги лучше использовать `mean / median / max / entropy / distance`, а не натягивать на них `count` и `share`.

### 3.1. Кривизна и поворотность

Это самый перспективный подблок для следующего эксперимента: он должен помогать различать "просто плотный центр" и геометрически сложные участки внутри плотной сети.

| Признак | Смысл | Статус / комментарий |
|---|---|---|
| `road_curvature_mean_weighted` | Средняя кривизна дорог, взвешенная длиной | Кандидат высокого приоритета |
| `road_curvature_max` | Максимальная кривизна | Кандидат высокого приоритета |
| `curvature_deg_per_km` | Сумма поворотов на км дорог | Кандидат высокого приоритета |
| `turn_event_count` | Количество поворотных событий | Имеет смысл, если считаем углы между соседними сегментами |
| `turn_event_density_per_km` | Поворотные события на км дорог | Более интерпретируемо, чем просто count |
| `sharp_turn_count` | Количество резких поворотов | Например, угол >= 45° |
| `sharp_turn_density_per_km` | Резкие повороты на км дорог | Кандидат высокого приоритета |
| `sharp_turn_share` | Доля резких поворотов среди всех поворотов | Осмысленный share внутри родительской группы `turn_event_count` |
| `major_curvature_deg_per_km` | Кривизна на major roads | Имеет смысл как отдельная группа дорог |
| `minor_curvature_deg_per_km` | Кривизна на local/minor roads | Имеет смысл как отдельная группа дорог |
| `link_curvature_deg_per_km` | Кривизна на `*_link` дорогах | Особенно интересно для съездов/примыканий |

### 3.2. Сегментация и фрагментация сети

Здесь count/density/share уже хорошо работают.

| Признак | Смысл | Статус / комментарий |
|---|---|---|
| `road_segment_count` | Число сегментов дорог | Кандидат высокого приоритета |
| `road_segment_density_per_km` | Сегменты на км дорог | Более честная нормировка, чем просто count |
| `road_segment_length_mean` | Средняя длина сегмента | Кандидат высокого приоритета |
| `road_segment_length_median` | Медианная длина сегмента | Устойчивее к выбросам |
| `short_segment_count` | Количество коротких сегментов | Нужен порог, например `< 50 м` |
| `short_segment_density_per_km` | Короткие сегменты на км дорог | Осмысленно |
| `short_segment_share` | Доля коротких сегментов среди всех сегментов | Осмысленный share внутри `road_segment_count` |


### 3.3. Итоговая фильтрация по здравому смыслу

Рекомендуемый MVP группы 3:

| Признак | Почему именно он |
|---|---|
| `curvature_deg_per_km` | Самый понятный общий сигнал геометрической сложности |
| `sharp_turn_density_per_km` | Ловит резкие повороты, которые модель сейчас почти не видит |
| `road_curvature_mean_weighted` | Базовая средняя извилистость |
| `road_curvature_max` | Локальные экстремумы геометрии |
| `road_orientation_entropy` | Различает регулярную сетку и хаотичную геометрию |
| `road_segment_length_median` | Устойчивый proxy плотности/нарезки сети |
| `short_segment_share` | Хорошо ложится в схему `count + share` |
| `dead_end_density_per_km` | Добавляет локальную топологию, почти не дублируя current features |

## 4. Население и застройка

Основные источники:

- WorldPop Russia population raster;
- GHSL population grid;
- Kontur population dataset;
- Overture buildings;
- OSM buildings.

| Признак | Смысл | Комментарий |
|---|---|---|
| `population_count` | Население в H3 | Базовая human exposure |
| `population_density` | Плотность населения | Нормировка на площадь |
| `population_count_250m` | Население в радиусе 250 м | Буфер вокруг ячейки |
| `population_count_500m` | Население в радиусе 500 м | Ближайший контекст |
| `building_count` | Количество зданий | Proxy урбанизации |
| `building_area_total_m2` | Суммарная площадь зданий | Плотность застройки |
| `building_area_share` | Доля застроенной площади | Urban intensity |
| `building_height_mean` | Средняя высота зданий | Если доступно |
| `residential_building_count` | Жилые здания | Жилая активность |
| `commercial_building_count` | Коммерческие здания | Генераторы поездок |
| `industrial_building_count` | Производственные здания | Грузовой/служебный транспорт |
| `built_up_area_share` | Доля застроенной территории | Можно брать из GHSL/OSM/Overture |

## 5. POI и точки притяжения

Основные источники:

- OpenStreetMap POI;
- Overture Places;
- муниципальные справочники.

Для POI лучше считать не только внутри H3, но и в радиусах:

```text
within H3
within 100m
within 250m
within 500m
```

| Признак | Смысл | Комментарий |
|---|---|---|
| `poi_count_total` | Все POI | Общая активность |
| `school_count` | Школы | Дети и пешеходы |
| `kindergarten_count` | Детские сады | Дети и пешеходы |
| `university_count` | Вузы/колледжи | Пешеходный поток |
| `hospital_count` | Больницы | Спецтранспорт и пешеходы |
| `clinic_count` | Поликлиники | Пешеходный поток |
| `retail_count` | Магазины | Точки притяжения |
| `mall_count` | ТЦ | Большой поток |
| `market_count` | Рынки | Пешеходы и парковка |
| `restaurant_cafe_count` | Кафе/рестораны | Активность, в т.ч. вечерняя |
| `bar_count` | Бары | Ночная активность |
| `parking_count` | Парковки | Маневры и въезды/выезды |
| `fuel_station_count` | АЗС | Въезды/выезды |
| `public_transport_poi_count` | Остановки/станции | Пассажирский поток |
| `industrial_poi_count` | Промышленные объекты | Грузовой транспорт |
| `logistics_poi_count` | Склады/логистика | Грузовой транспорт |

## 6. Трафик и экспозиция движения

Основные источники:

- официальные городские/региональные данные по интенсивности, если доступны;
- data.gov.ru и региональные порталы открытых данных;
- коммерческие источники: TomTom, HERE, Яндекс, 2ГИС, если будет доступ;
- proxy из OSM/Overture, если прямых данных нет.

| Признак | Смысл | Комментарий |
|---|---|---|
| `aadt_total` | Среднесуточная интенсивность движения | Лучший exposure-признак, если доступен |
| `aadt_car` | Легковые автомобили | Разделение потока |
| `aadt_truck` | Грузовой поток | Влияет на тяжесть ДТП |
| `truck_share` | Доля грузовиков | Proxy тяжести |
| `peak_hour_flow` | Пиковая интенсивность | Утро/вечер |
| `public_transport_routes_count` | Число маршрутов ОТ | Автобусы/маршрутки |
| `public_transport_frequency_estimate` | Частота ОТ | Если есть расписания |
| `traffic_exposure_proxy` | Расчетный proxy трафика | Если нет реальной интенсивности |

Возможный proxy:

```text
traffic_exposure_proxy =
  road_length_by_class
  + lanes
  + maxspeed
  + proximity_to_major_roads
  + poi density
  + population density
```

## 7. Погода, климат и сезонность

Основные источники:

- ERA5 / Copernicus;
- ERA5-Land hourly;
- локальные метеостанции, если доступны.

Для проектной модели лучше использовать климатические нормы, а не фактическую будущую погоду.

| Признак | Смысл | Комментарий |
|---|---|---|
| `month_num` | Номер месяца | Базовая сезонность |
| `season` | Сезон | Зима/весна/лето/осень |
| `daylight_hours_mean` | Средняя длина светового дня | Освещенность/сезонность |
| `temperature_mean_climatology` | Климатическая средняя температура | Норма для месяца |
| `temperature_min_climatology` | Низкие температуры | Зимний риск |
| `precipitation_sum_climatology` | Осадки | Дождь/снег |
| `snow_depth_mean_climatology` | Снежный покров | Зимний риск |
| `freeze_thaw_days_climatology` | Дни перехода через 0C | Гололед/разрушение покрытия |
| `wet_days_climatology` | Дни с осадками | Мокрое покрытие |

## 8. Освещенность и ночная активность

Основные источники:

- VIIRS nighttime lights;
- NASA Black Marble / VIIRS DNB продукты.

| Признак | Смысл | Комментарий |
|---|---|---|
| `night_light_mean` | Средняя ночная яркость | Proxy урбанизации и активности |
| `night_light_max` | Максимальная ночная яркость | Яркие коммерческие/транспортные зоны |
| `night_light_std` | Неоднородность освещения | Граница активных зон |
| `night_light_change` | Изменение яркости | Развитие территории |
| `dark_road_share_proxy` | Доля темных дорожных участков | Комбинация дорог и ночной яркости |

## 9. Административный и городской контекст

Основные источники:

- geoBoundaries;
- OSM boundaries;
- муниципальные границы;
- landuse из OSM/Overture.

| Признак | Смысл | Комментарий |
|---|---|---|
| `district_id` | Район/муниципалитет | Категориальный контекст |
| `is_urban_core` | Центр города | Центр/периферия |
| `distance_to_city_center_m` | Расстояние до центра | Непрерывный proxy |
| `landuse_residential_share` | Доля жилого landuse | Жилая зона |
| `landuse_commercial_share` | Доля коммерческого landuse | Точки притяжения |
| `landuse_industrial_share` | Доля промзоны | Грузовой/служебный транспорт |
| `landuse_green_share` | Парки/зеленые зоны | Иной паттерн движения |
| `landuse_transport_share` | Транспортные территории | Вокзалы, станции, развязки |

## 10. Признаки, которые лучше исключать из structural-модели

Эти признаки полезны для hotspot-модели, но нежелательны для оценки новых дорог:

| Признак | Почему исключать |
|---|---|
| `accident_count` | Это уже история ДТП в ячейке |
| `had_accident` | То же |
| `accidents_prev_1m` | Исторический риск |
| `accidents_prev_3m` | Исторический риск |
| `accidents_prev_6m` | Исторический риск |
| `accidents_prev_12m` | Исторический риск, текущий baseline почти полностью на нем держится |
| `target_count_next_month` | Target leakage |
| `target_had_accident_next_month` | Target leakage |

## План добавления групп

Предлагаемый порядок экспериментов:

| Шаг | Группа признаков | Цель |
|---:|---|---|
| 1 | Дорожная структура | Проверить, дает ли сама сеть дорог сигнал |
| 2 | Узлы и инфраструктура | Добавить конфликтные точки |
| 3 | Геометрия сети | Проверить сложность конфигурации |
| 4 | Население и застройка | Добавить human exposure |
| 5 | POI | Добавить точки притяжения |
| 6 | Трафик/proxy трафика | Добавить транспортную экспозицию |
| 7 | Погода/климат/сезонность | Добавить временной контекст |
| 8 | Освещенность | Добавить proxy ночной активности |
| 9 | Все structural features | Проверить полную модель без истории ДТП |
| 10 | Structural + historical | Отдельно сравнить с hotspot-моделью |

Метрики для сравнения:

- PR-AUC;
- ROC-AUC;
- Top 1% precision / recall / lift;
- Top 3% precision / recall / lift;
- Top 5% precision / recall / lift;
- Top 10% precision / recall / lift;
- backtest по годам;
- spatial holdout, если будем проверять перенос на новые территории.
