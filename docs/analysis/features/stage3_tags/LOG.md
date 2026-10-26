# Этап 3 (tags): журнал работы

Хронологический журнал этапа 3 из `docs/analysis/features/PLAN.md` — теги OSM
(`surface`, `lanes`, `oneway`): «нет тега» = неизвестно (NaN), а не 0; асфальт → длины
по классам. Стартовый набор — `road_geo_v2_junctions` (32). Время — локальное, 2026-10-16.

## 1. Чтение входных материалов

Прочитаны: `PLAN.md` (правило плана, этап 3, таблица «Порядок и оценка»),
`REVIEW_2026-10-15.md` (A1, A3, A4, B2, B3), EXPERIMENTS.md (этапы 1–2 и прецедент
`lit_road_length_m` от 2026-09-15), `stage2_junctions/README.md` и `LOG.md` (образец),
код `osm_road_features.py` (`row_base_for_way`, `allocate_way_to_h3`,
`aggregate_h3_road_features`, константы), `catboost_baseline.py`, скрипты этапа 2
(`build_junction_v2_features.py`, `run_road_geo_v2_junctions_experiments.py`,
`plot_road_geo_v2_junctions.py`, `validate_junction_v2.py`).

Где лежат теги:
- `parquet_output/osm_roads_raw.parquet` — 30 453 ways; полный словарь тегов в `tags_json`,
  разобранные `surface`, `lanes_num`, `oneway` (bool через `parse_bool_yes`).
- `parquet_output/osm_road_h3_segments.parquet` — 48 247 строк «way × ячейка» с длиной
  куска (`segment_length_m`, сэмплы 20 м, та же функция `allocate_way_to_h3`). Тега
  `junction` здесь нет, поэтому `oneway` v2 берётся из `tags_json` по `osm_id`.

Как старый код трактует отсутствие тега (`row_base_for_way` → `aggregate_h3_road_features`):
- `is_asphalt_surface = surface in ASPHALT_SURFACES` — нет тега → False («не асфальт»);
- `is_bad_surface = surface in BAD_SURFACES` — нет тега → False; `bad_surface_share` делится
  на **всю** длину ячейки;
- `lanes_mean_weighted = weighted_mean(lanes_num, length)` — по размеченным, но 0.0, если
  размеченных нет вовсе (0 ≠ «0 полос»);
- `oneway = parse_bool_yes(tags["oneway"])` — `oneway=-1` → False, `junction=roundabout`
  без `oneway` → False.

Значения тегов по ways: `oneway` — нет 27 983, yes 1 793, no 661, −1 16; `junction` —
roundabout 37 (11 без `oneway`), circular 17 (1 без `oneway`); `surface` — нет 22 653,
asphalt 5 977, unpaved 1 099, ground 441, compacted 93, paved 77, gravel 61, dirt 16,
paving_stones 13, concrete 11, concrete:plates 5, fine_gravel 5, mud 2; `lanes` — нет 28 218.

`major_road_length_m` уже есть в панели (`aggregate_h3_road_features`: длина `is_major`,
`MAJOR_ROAD_CLASSES` = motorway, trunk, primary, secondary и их `_link`; tertiary не входит),
в `DUPLICATE_EXCLUDED_FEATURES` с перехода 3.5 → 4.0. Используется существующее определение.

`fit_p99_clip_caps` / `apply_p99_clipping` с NaN: `DataFrame.quantile` и `.max` пропускают NaN,
`Series.clip` оставляет NaN как есть → правка не нужна (проверено отдельно, см. §4).

## 2. Построение признаков (02:05–02:08)

Код — **новые** функции в `src/road_accident_risk/osm_road_features.py` (старые
`row_base_for_way`, `allocate_way_to_h3`, `aggregate_h3_road_features` не тронуты):
- `parse_oneway_v2(tags)`: `oneway` ∈ {yes, true, 1, −1, reverse} → одностороннее;
  ∈ {no, false, 0} → двустороннее; тега нет → одностороннее только при
  `junction` ∈ {roundabout, circular}, иначе двустороннее (в OSM это дефолт, а не «неизвестно»);
- `road_tags_v2_for_way(tags)`: `surface` нет → `is_asphalt_v2 = is_bad_surface_v2 = NaN`;
  `lanes` нет → NaN. Множества асфальта и плохого покрытия **те же**, что в старом коде:
  `ASPHALT_SURFACES` = {asphalt, paved, concrete, concrete:plates, concrete:lanes},
  `BAD_SURFACES` = {unpaved, gravel, ground, dirt, earth, sand, grass, mud, cobblestone, sett}.
  Прочие значения (compacted 93 ways, paving_stones 13, fine_gravel 5) входят в размеченную
  длину, но ни в асфальт, ни в плохое покрытие (как и раньше);
- `build_h3_tag_v2_segments(h3_segments, roads)`: берёт готовые куски «way × ячейка» из
  `osm_road_h3_segments.parquet` (та же нарезка, что у старых признаков) и теги из `tags_json`;
- `aggregate_h3_tags_v2_features(seg, cells, min_tagged_m=100)` — признаки ячейки;
  порог `TAGS_V2_MIN_TAGGED_M = 100` м — параметр.

Скрипт `scripts/build_tags_v2_features.py` (0.2 с) → `parquet_output/h3_tags_v2_features.parquet`
(4 886 ячеек, как `h3_road_features`). Новый путь `H3_TAGS_V2_FEATURES_PARQUET` в `paths.py`.

Сверка с панелью (те же куски, те же определения): `road_length_total_m`,
`major_road_length_m`, `lanes_tagged_share` — max |Δ| ≤ 1e-12; `lanes_mean_weighted` там,
где v2 не NaN, — 1e-15. То есть `major_road_length_m` в новой таблице — **ровно** тот же
признак, что лежит в панели (удалён как дубль при переходе 3.5 → 4.0); в модель
идёт колонка панели, из новой таблицы подтягиваются только новые имена.

Доли по 4 886 ячейкам: `asphalt_share_among_tagged` и `bad_surface_share_among_tagged` — 46.5 % NaN;
`lanes_mean_weighted_v2` — 73.8 % NaN; `surface_tagged_share` — 41 % нулей; `street_length_m` —
19.6 % нулей (ячейки только с service).

## 3. Код набора признаков (02:08–02:10)

`catboost_baseline.py`: `FEATURE_SET_ROAD_GEO_V2_TAGS`, `V2_TAGS_EXPOSURE_REPLACED`
(`asphalt_surface_length_m` → `major_road_length_m` + `street_length_m`), `V2_TAGS_SEMANTICS_REPLACED`
(`bad_surface_share` → `bad_surface_share_among_tagged`, `lanes_mean_weighted` → `lanes_mean_weighted_v2`,
`oneway_share` → `oneway_share_v2`), `build_road_geo_v2_tags_features(columns, replace_exposure,
replace_semantics, extra)` (замена на месте старого признака), `attach_tags_v2_features(panel)`,
ветки в `build_features` и `train_and_evaluate`.

Ловушка этапа 2 закрыта заранее: `build_road_geo_baseline_features` исключает `TAGS_V2_FEATURES`
(как `JUNCTION_V2_FEATURES`); `major_road_length_m` и так в `DUPLICATE_EXCLUDED_FEATURES`.
Проверка на панели со всеми подтянутыми колонками (1 107 084 × 152):
`road_geo_baseline` 61, `road_geo_v2_prune` 33, `road_geo_v2_junctions` 32 — списки поэлементно
равны спискам на панели без новых колонок. Варианты: B1 — 33 признака, B2 — 32, B — 33.

## 4. p99-клиппинг с NaN (02:10)

На train (`road_length_total_m > 0`) для набора B: `lanes_mean_weighted_v2` — 301 152 NaN до и
после клиппинга, `bad_surface_share_among_tagged` — 185 952 до и после; каппы считаются по
не-NaN значениям (`quantile`/`max` пропускают NaN), `clip` NaN не трогает. Правка функций не нужна.

## 5. Валидация до модели (02:15–02:24)

Скрипт `scripts/validate_tags_v2.py` (~20 с): покрытие тегов, статистики старых/новых признаков,
Spearman, одиночный AUC, «артефакт асфальта», чувствительность порога, ручная проверка; фигуры 1–4.
Две ошибки при первом запуске (передача `cells` в агрегатор при проверке порогов и повторный join
уже подтянутой колонки) исправлены, на результаты не влияли.
CSV: `reports/road_geo_v2_tags_{coverage_by_class,feature_stats,spearman_matrix,spearman_pairs,
asphalt_artifact,threshold_sensitivity}.csv`, `figures/manual_check_cells.csv`.

### 5.1. Покрытие тегов по классам (рис. 1, доля длины в ячейках города)

| класс (с *_link) | длина, км | surface | lanes | oneway (тег явно) | асфальт среди размеч. | плохое среди размеч. |
|---|---|---|---|---|---|---|
| trunk (+motorway) | 130 | 100 % | 99.8 % | 88.9 % | 100 % | 0 % |
| primary | 181 | 100 % | 85.0 % | 65.5 % | 100 % | 0 % |
| secondary | 164 | 99.5 % | 62.1 % | 49.6 % | 100 % | 0 % |
| tertiary | 272 | 84.6 % | 38.1 % | 26.7 % | 99.8 % | 0.2 % |
| unclassified | 525 | 29.1 % | 8.1 % | 4.5 % | 60.1 % | 39.5 % |
| residential | 1 342 | 30.2 % | 7.8 % | 7.3 % | 47.1 % | 50.8 % |
| living_street | 13 | 14.3 % | 1.9 % | 5.0 % | 86.4 % | 13.6 % |
| service | 3 048 | 14.2 % | 0.7 % | 2.1 % | 68.8 % | 28.6 % |
| все классы | 5 676 | 29.9 % | 11.6 % | 10.1 % | 75.8 % | 23.0 % |
| уличная сеть (без service) | 2 628 | 48.1 % | 24.3 % | 19.5 % | 78.2 % | 21.1 % |

Числа ревью **подтверждены** (surface: 100/100/100/85/30/14; lanes: 100/85/63/37/8/0.7);
уточнения — secondary 99.5 % по surface, lanes у tertiary 38 %, unclassified (в ревью не было)
29 % / 8 %. Среди размеченных residential **половина** — грунт: то есть «нет тега» на
residential нельзя трактовать ни как асфальт, ни как не-асфальт.

### 5.2. Старые и новые признаки (4 296 ячеек панели, тест — 154 656 строк, рис. 4)

| признак | нулей | NaN | одиночный AUC | ρ стар.↔нов. | AUC старого на строках, где новый не NaN |
|---|---|---|---|---|---|
| `asphalt_surface_length_m` (стар.) | 50.7 % | — | **0.862** | 0.665 с `asphalt_share_among_tagged` | 0.775 |
| `asphalt_share_among_tagged` | 9.1 % | 45.1 % | 0.589 | | |
| `bad_surface_share` (стар.) | 78.2 % | — | 0.500 | 0.985 | 0.418 (обр. 0.582) |
| `bad_surface_share_among_tagged` | 35.2 % | 45.1 % | 0.413 (обр. **0.587**) | | |
| `lanes_mean_weighted` (стар.) | 70.5 % | — | **0.814** | 1.000 | **0.670** |
| `lanes_mean_weighted_v2` | 0 % | 73.0 % | 0.670 | | |
| `oneway_share` (стар.) | 83.2 % | — | 0.709 | 0.999 | 0.709 |
| `oneway_share_v2` | 83.1 % | — | 0.708 | | |
| `major_road_length_m` | 76.8 % | — | 0.763 | | |
| `street_length_m` | 20.1 % | — | 0.788 | | |
| `surface_tagged_share` (диагн.) | 39.6 % | — | 0.690 | | |
| `road_length_total_m` | 0 % | — | 0.796 | | |

Наблюдения:
- `lanes_mean_weighted`: на ячейках, где полосы размечены, старый и новый признак совпадают
  (ρ 1.000, AUC 0.670 у обоих). Вся разница одиночного AUC 0.814 → 0.670 — это вклад
  «0 = тега нет», то есть **сигнал полноты разметки** (размечены магистрали).
- `bad_surface_share`: старый признак одиночно не различает ничего (AUC 0.500): «грунт» смешан
  с «не размечено». Среди размеченной длины доля плохого покрытия различает места (0.587, знак
  обратный — грунт → меньше ДТП). Переход к NaN **проявил** сигнал.
- `oneway_share_v2` отличается от старого в единичных ячейках (ρ 0.999): `oneway=-1` у 16 ways
  и 12 колец без `oneway`. Правка корректная, но на признак почти не влияет.
- Одиночные AUC: асфальт (0.862) выше и длины сети (0.796), и улиц (0.788), и магистралей (0.763).

### 5.3. «Артефакт карты»: что измеряет `asphalt_surface_length_m` (рис. 2, 3)

`reports/road_geo_v2_tags_asphalt_artifact.csv`:
- Состав асфальта по длине: магистрали (trunk/primary/secondary + links) — **36.9 %**,
  tertiary — 17.6 %, residential — 14.8 %, **service — 23.1 %**. Две пятых признака — размеченные
  дворы и внутриквартальные улицы.
- Spearman по ячейкам: с `major_road_length_m` 0.627, с `street_length_m` 0.513, с
  `road_length_total_m` 0.409, с `surface_tagged_share` **0.704**, с длиной размеченной
  дороги `surface_tagged_length_m` **0.844**. Признак ближе всего к «сколько метров
  размечено», а не к «сколько метров магистралей».
- Частные Spearman: асфальт ~ `surface_tagged_share` | major, street = **0.566**;
  асфальт ~ major | surface_tagged_share, street = 0.417; асфальт ~ street | major, share = 0.358.
  Полнота разметки — самая сильная независимая компонента признака.
- OLS на рангах (R²): major → 0.393; major + street → 0.492; major + surface_tagged_share → 0.604;
  major + street + surface_tagged_share → **0.655**; одна `surface_tagged_length_m` → 0.713.
  Добавление полноты разметки к двум честным длинам поднимает R² с 0.49 до 0.66.
- Одиночные AUC: асфальт 0.862, размеченная длина `surface_tagged_length_m` 0.834, полнота
  `surface_tagged_share` 0.690. Карта полноты разметки **сама по себе** различает ячейки с ДТП:
  картографы размечают центр и магистрали, где и происходят ДТП.
- Остаток ранга асфальта после major + street + surface_tagged_share всё ещё имеет AUC 0.719,
  то есть признак не сводится к трём величинам полностью (линейная модель на рангах грубая; в
  остатке — в том числе общая длина сети и её плотность, т.е. урбанизированность).

Рис. 3 (карта): асфальт (б) и полнота разметки (а) имеют одинаковый рисунок — плотные пятна в
центре, в Ленинском районе и на юго-востоке (Заводской), а не только линии магистралей (в).

### 5.4. Чувствительность к порогу размеченной длины (`threshold_sensitivity.csv`)

| порог | NaN у bad/asphalt share | NaN у lanes_v2 | AUC bad_share (обр.) | ρ bad_share со старым | AUC lanes_v2 |
|---|---|---|---|---|---|
| > 0 м | 39.6 % | 70.5 % | 0.589 | 0.981 | 0.665 |
| 50 м | 42.1 % | 71.5 % | 0.587 | 0.984 | 0.670 |
| **100 м** | 45.1 % | 73.0 % | 0.587 | 0.985 | 0.670 |
| 200 м | 49.9 % | 75.6 % | 0.585 | 0.987 | 0.667 |

Порог почти не влияет ни на ρ, ни на AUC; выбран 100 м (как в плане): отсекает ячейки, где
доля посчитана по одному короткому куску.

### 5.5. Ручная проверка 10 ячеек (`figures/manual_check_cells.csv`)

| случай | ячейка (центр) | что там | старое → новое |
|---|---|---|---|
| асфальт без магистралей | 8910b14984fffff (51.6046, 46.0249) | микрорайон у пр. Героев Отечества: 3.4 км дорог, из них 2.9 км service, **98 %** размечены `asphalt` | асфальт 3 281 м (99-й перцентиль) при major 0 — «образцовый» картограф |
| то же | 8910b14910bffff (51.5993, 45.9633) | ул. Ломоносова, Загороднева (residential/tertiary), 96 % размечено | асфальт 3 137 м, major 0 |
| асфальт 0 при длинной улице | 8910b394d7bffff (51.5723, 45.8522) | Курсантская ул. и безымянные unclassified, 3 км, **ни одного тега** | асфальт 0, bad 0 → NaN, lanes 0 → NaN |
| то же | 8910b386c6bffff (51.5507, 45.9941) | ул. Завокзальный Посёлок, residential, 91 % размечено `unpaved` | bad 0.91 → 1.00 (честно: грунт) |
| bad: мало размеченного | 8910b387583ffff (51.4909, 45.9068) | 3.9 км, размечено 3 % (≈116 м), вся размеченная — грунт | bad 0.03 → **1.00** — ограничение: доля по 116 м из 3.9 км |
| то же | 8910b3b30d3ffff (51.5782, 46.0822) | 3-й пр. Муленкова, 4 % размечено (≈113 м) | bad 0.04 → 1.00 — то же ограничение |
| bad: нет тегов вовсе | 8910b149353ffff (51.5725, 45.9661) | 5.3 км только service, без тегов | bad 0 → NaN, асфальт 0 |
| lanes: магистраль без lanes | 8910b387523ffff (51.4903, 45.9298) | пр. Энтузиастов / Крымская (secondary), `lanes` нет | lanes 0 → NaN (раньше «0 полос» у secondary) |
| oneway=-1 | 8910b3b008bffff (51.6112, 46.1726) | Усть-Курдюмское шоссе (primary): одна проезжая часть `oneway=-1`, вторая `yes` | oneway 0.41 → 1.00 |
| то же | 8910b3b009bffff (51.6120, 46.1677) | то же шоссе | 0.56 → 1.00 |

Все случаи соответствуют задуманной семантике. Найденное ограничение: при длинной сети и
малой абсолютной размеченной длине (100–150 м) доля «среди размеченного» опирается на один
way. Абсолютный порог 100 м это допускает; относительный порог (например, ≥ 10 % длины)
сделал бы признак снова зависимым от полноты разметки — сознательно не вводился.

### 5.6. Правило |ρ| ≥ 0.85 для новых признаков набора B

- `street_length_m` ~ `road_length_total_m`: **ρ 0.582** — не дубликат (54 % длины — service);
  max |ρ| внутри B — 0.802 с `street_junction_3leg_count` (ниже порога). Оставлен.
- `major_road_length_m`: max |ρ| внутри B — 0.671 (`oneway_share_v2`). Оставлен.
- `lanes_mean_weighted_v2` (max 0.400), `oneway_share_v2` (0.671), `bad_surface_share_among_tagged`
  (0.343) — пар ≥ 0.85 внутри B нет (пары со своими старыми версиями 0.985–1.000 не мешают:
  старые в B удалены).
- `asphalt_share_among_tagged` ~ `bad_surface_share_among_tagged`: **ρ −0.977** — одна величина
  (размеченное либо асфальт, либо грунт). В набор берётся только `bad_surface_share_among_tagged`
  (он заменяет старый `bad_surface_share`; по |AUC| они равны 0.587/0.589). Асфальтовая доля
  в B не входит — проверяется отдельным вариантом.

## 6. Абляция A / B1 / B2 / B (02:14–02:27)

`scripts/run_road_geo_v2_tags_experiments.py` (каркас — скрипт этапа 2; рецептура та же;
NaN не клиппируются и не заполняются, в скрипте assert, что число NaN после клиппинга не
изменилось). Сырые прогоны — `reports/road_geo_v2_tags_seed_runs.csv`, разности —
`reports/road_geo_v2_tags_delta.csv` (`scripts/plot_road_geo_v2_tags.py`), ~3 мин на вариант.

```text
PYTHONPATH=src venv/bin/python scripts/run_road_geo_v2_tags_experiments.py \
  --variants road_geo_v2_junctions,tags_exposure_only,tags_semantics_only,road_geo_v2_tags
PYTHONPATH=src venv/bin/python scripts/plot_road_geo_v2_tags.py
```

Воспроизводимость по ходу: A совпал с `reports/road_geo_v2_junctions_seed_runs.csv` на всех 5 сидах
(max |разность| = 0.0; seed 42 — 0.907869 / 0.199600 / 0.3098).

| метрика | A (32) | B1 (33) | Δ B1−A, вердикт | B2 (32) | Δ B2−A, вердикт | B (33) | Δ B−A (2σ) | вердикт 2σ / парный |
|---|---|---|---|---|---|---|---|---|
| ROC-AUC | 0.907944 ±0.000306 | 0.907477 ±0.000208 | −0.000467, на границе (1.41) / парн. шум | 0.907972 | +0.000028, шум | 0.907854 ±0.000143 | −0.000090 (0.000301) | шум / шум |
| PR-AUC | 0.199931 ±0.000593 | 0.199488 | −0.000442, шум | 0.200366 | +0.000435, шум | 0.200480 ±0.000593 | +0.000549 (0.000750) | шум / шум |
| precision@1 % | 0.3132 ±0.0021 | 0.3173 | +0.0041, на границе (1.43) / парн. шум | 0.3135 | +0.0003, шум | 0.3136 ±0.0025 | +0.0004 (0.0029) | шум / шум |
| precision@5 % | 0.2072 | 0.2074 | шум | 0.2072 | шум | 0.2071 | −0.0002 (0.0009) | шум / шум |
| recall@5 % | 0.4438 | 0.4440 | шум | 0.4436 | шум | 0.4434 | −0.0003 (0.0020) | шум / шум |
| recall@10 % | 0.6557 ±0.0015 | 0.6545 | −0.0012, шум | 0.6548 | −0.0008, шум | 0.6531 ±0.0010 | **−0.0026 (0.0016)** | **значимо (1.63)** / шум (парн. 2sd 0.0039) |

- **B1** (только экспозиция): ROC-AUC вниз на границе, precision@1 % вверх на границе; парная
  оценка — шум по всем метрикам.
- **B2** (только NaN-семантика): шум по всем метрикам.
- **B** (итог): шум по всем метрикам, кроме верхнего дециля: precision/recall/lift@10 % −0.4 %
  отн. (recall@10 % 0.6557 → 0.6531), по 2σ «значимо» (1.63), по парной оценке — шум. Та же
  картина B против B2 (1.58). Сдвиг маленький, но направленный (все 5 сидов B ниже среднего A).

Важности (среднее по 5 сидам, рис. 6): асфальт в A — №1 (9.18). В B важность **не перешла
к одному признаку**: `street_length_m` 4.55 (№7), `major_road_length_m` 2.89 (№16); №1 стал
уклон (7.17). Зато `bad_surface_share_among_tagged` поднялся до №3 (6.11; старый
`bad_surface_share` — №13, 3.34); `road_share_service` упал с 5.91 до 4.29. Сумма важности
заменяемых в A — 16.7, новых в B — 18.4.

### 6.1. Почему доля грунта стала №3: пропуск тоже несёт информацию (02:30)

Проверка: у CatBoost NaN по умолчанию — «меньше минимума», то есть дерево может делить
«есть разметка / нет разметки». Одиночный AUC индикатора «значение не NaN» на тесте:
`bad_surface_share_among_tagged` — **0.716** (ρ индикатора с `surface_tagged_share` по ячейкам 0.825),
`lanes_mean_weighted_v2` — **0.772**. AUC `lanes_mean_weighted_v2` с NaN → −1 — 0.807 (старый
признак с нулём — 0.814). Вывод: перевод «нет тега» в NaN **исправляет смысл значения**, но
**не убирает** из модели сигнал полноты разметки — дерево видит маску пропусков. Для
табличного бустинга NaN и «0 вне диапазона» почти эквивалентны. Это проверяется отдельным
диагностическим вариантом с заполнением NaN медианой train (`_fillmedian`): там маска стёрта.

Запущены диагностические варианты (02:28): B + `asphalt_surface_length_m` (есть ли у асфальта
информация сверх длин), B + `surface_tagged_share` (добавляет ли явная полнота разметки),
B + `asphalt_share_among_tagged`.

## 7. Диагностические варианты (02:28–02:44)

```text
PYTHONPATH=src venv/bin/python scripts/run_road_geo_v2_tags_experiments.py \
  --variants tags_plus_asphalt_surface_length_m,tags_plus_surface_tagged_share,tags_plus_asphalt_share_among_tagged
PYTHONPATH=src venv/bin/python scripts/run_road_geo_v2_tags_experiments.py --variants road_geo_v2_tags_fillmedian
```

| вариант | ROC-AUC | PR-AUC | p@1 % | recall@10 % | к B (2σ / парн.) | важность добавленного |
|---|---|---|---|---|---|---|
| B (33) | 0.907854 | 0.200480 | 0.3136 | 0.6531 | — | — |
| E1: B + `asphalt_surface_length_m` (34) | 0.908018 | 0.200226 | 0.3141 | 0.6533 | шум по всем | **8.01, №1**; `bad_surface_share_among_tagged` 6.11 → 2.27 |
| E2: B + `surface_tagged_share` (34) | 0.907682 | 0.200167 | 0.3137 | 0.6525 | шум по всем | 4.75, №6; `bad_surface_share_among_tagged` 6.11 → 4.79 |
| E3: B + `asphalt_share_among_tagged` (34) | 0.908000 | 0.200366 | 0.3154 | 0.6520 | шум по всем | — |
| E4: B, NaN → медиана train (33) | 0.907048 | 0.199914 | 0.3107 | 0.6537 | **ROC-AUC −0.000806 (2σ 0.000274; парн. 0.000390) — значимо**; p@1 на границе; остальное шум | — |

Выводы:
1. **Асфальт не несёт информации сверх B.** Вернув его в B (E1), метрики не меняются (все шум),
   но признак сразу забирает важность №1 (8.0), отбирая её у `bad_surface_share_among_tagged`
   и длин. Важность CatBoost здесь — не мера информативности: модель предпочитает
   «составной» признак, который одним разбиением даёт и длину, и полноту разметки.
2. **Падение recall@10 % у B не из-за асфальта**: E1 (с асфальтом) даёт 0.6533 — столько же,
   сколько B. Все варианты на основе B (E1–E3) — 0.652–0.653, A — 0.6557, B1 и B2 по отдельности —
   0.6545 и 0.6548. Это малый совместный эффект замен (−0.4 % отн.), значимый по 2σ, но не по
   парной оценке; отнесён к шуму с оговоркой.
3. **Явная полнота разметки (E2) ничего не добавляет** к B, но делит важность с
   `bad_surface_share_among_tagged` почти поровну (4.75 и 4.79) — подтверждение, что доля грунта
   в B частично работает как индикатор «размечено/нет» через маску NaN (см. §6.1).
4. **Маска пропусков — реальный сигнал.** Если стереть её (E4: NaN → медиана train только у
   `bad_surface_share_among_tagged` и `lanes_mean_weighted_v2`; первый запуск E4 по ошибке
   заполнял и NaN уклона и угла — перезапущен с исправлением, строки перезаписаны), ROC-AUC
   падает на 0.0008 — значимо и по 2σ, и по парной оценке; верхушка (PR-AUC, @5 %, @10 %) — шум.
   То есть часть сигнала, который раньше нёс асфальт, модель по-прежнему получает из «где
   картографы размечали теги» — теперь через NaN. Это ожидаемо: полнота разметки OSM
   коррелирует с центральностью и активностью района (ρ маски с `surface_tagged_share` 0.825).
   **Решение**: NaN оставлены (семантически корректно, нативно для CatBoost, без потерь); в тексте
   честно указать, что NaN исправляет смысл значения, но не устраняет сигнал полноты разметки.
   Полностью «очищенный» вариант (E4) стоит −0.0009 ROC-AUC к A при той же верхушке.

## 8. Решение по набору (02:47)

Итог — **`road_geo_v2_tags` (B), 33 признака**: 32 из `road_geo_v2_junctions` − 4 заменённых
(`asphalt_surface_length_m`, `bad_surface_share`, `lanes_mean_weighted`, `oneway_share`) + 5 новых
(`major_road_length_m`, `street_length_m`, `bad_surface_share_among_tagged`, `lanes_mean_weighted_v2`,
`oneway_share_v2`). Правило |ρ| ≥ 0.85 внутри B пар не даёт (§5.6). `asphalt_share_among_tagged`
не включён (ρ −0.977 с долей грунта; E3 — шум), `surface_tagged_share` — только диагностика.

Ответ на главный вопрос этапа: важность от асфальта **не ушла** к `major_road_length_m` как к
новому №1 (план ожидал «да»). Она распределилась: `street_length_m` №7, `major_road_length_m` №16,
а часть сигнала полноты разметки перешла в маску NaN доли грунта (№3). Метрики при этом — в
пределах шума (кроме верхнего дециля по 2σ, см. §7 п. 2).

## 9. Проверки воспроизводимости (02:44–02:46)

Штатная команда (каппы и важности — во временную папку scratchpad):
```text
PYTHONPATH=src venv/bin/python scripts/run_catboost_baseline.py --feature-set <набор> \
  --min-road-length-m 0 --clip-p99 --clip-p99-caps-path <scratch>/caps.csv --feature-importance-path <scratch>/imp.csv
```

| набор | признаков | ROC-AUC | PR-AUC | p@1 % | эталон |
|---|---|---|---|---|---|
| `road_geo_baseline` | 61 | 0.908056 | 0.198881 | 0.3072 (475/3 611) | бит в бит; каппы p99 побайтно = `reports/road_geo_baseline_p99_caps.csv` |
| `road_geo_v2_prune` | 33 | 0.908088 | 0.199559 | 0.3118 | = seed 42 в `road_geo_v2_prune_seed_runs.csv` |
| `road_geo_v2_junctions` | 32 | 0.907869 | 0.199600 | 0.3098 | = seed 42 в `road_geo_v2_junctions_seed_runs.csv` |
| `road_geo_v2_tags` | 33 | 0.907708 | 0.199800 | 0.3118 | = seed 42 варианта B в абляции; каппы побайтно = `reports/road_geo_v2_tags_p99_caps.csv` |

Вариант A абляции совпал с `road_geo_v2_junctions_seed_runs.csv` на всех 5 сидах (max |Δ| = 0.0).
Старые parquet не перезаписывались (новый — только `h3_tags_v2_features.parquet`).

## 10. Что не делалось

- viewer, `coursework/`, git — не трогались (по заданию).
- Замена `lanes` на СКДФ (PLAN, этап 3 → этап 5) — не делалась, это этап 5.
- Относительный порог размеченной длины (доля от длины ячейки) — сознательно не вводился (§5.5).
- Субагенты не использовались: валидация — секунды, обучение — последовательно.

## 11. Документы (02:48–02:55)

Написаны `README.md` этапа, запись в EXPERIMENTS.md (сверху), отметка в PLAN.md (этап 3 и строка
таблицы «Порядок и оценка»). Фигуры 1–6 просмотрены глазами; на рис. 4 легенда перекрывала
нижний столбец — перенесена.
