# Этап 7, шаг 1: инфраструктура годовой постановки

ТЗ — [`../../PLAN.md`](../../PLAN.md), этап 7, пункты 7.0, 7.1 (панель) и 7.4 (функции метрик). Журнал —
[LOG.md](LOG.md). Модели в этом шаге не сравнивались, был только дымовой прогон с одним сидом.

Файлы шага:
- модуль `src/road_accident_risk/v2_eval_year.py`;
- скрипты `scripts/build_h3_year_panel.py`, `scripts/run_v2_eval_year.py`, `scripts/check_v2_eval_year.py`,
  `scripts/plot_stage7_infra.py`;
- панель `parquet_output/h3_year_panel.parquet`.

## 7.0. Замороженный набор

Основной набор этапа 7 — `road_geo_v2_skdf`, 32 признака (`vy.FROZEN_FEATURE_SET`). Статичность проверена
на месячной панели `v2_eval.load_panel(attach=[attach_exposure_v6_features])`: у всех семи наборов
v2 `groupby(h3).nunique(dropna=False).max() == 1`. Это `road_geo_baseline` (61 признак), `prune` (33),
`junctions` (32), `tags` (33), `geometry` (30), `skdf` (32) и `exposure_a` (34), всего 83 разных признака.
Значит, признаки ячейки можно брать из одного месяца: строки «ячейка × год» различаются только таргетом и лагами.

**Ловушка.** Билдеры наборов (`build_road_geo_baseline_features` и все наборы на его основе) берут **все
колонки панели, кроме исключённых**. Если отдать им годовую панель, таргет и лаги попадут в признаки.
Поэтому `year_variant_features` передаёт билдерам только колонки признаков ячейки. Проверено: на годовой
панели все семь наборов совпадают с месячными, порядок признаков тот же.

## 7.1. Панель «ячейка × год» (`parquet_output/h3_year_panel.parquet`)

В панели **47 256 строк = 4 296 ячеек × 11 лет (2015–2025)**. Ячейки те же, что в месячной схеме
(`road_length_total_m > 0`). ДТП взяты те же, что в месячной панели: `crashes_slim`, bbox Саратова,
привязка `latlng_to_cell` res 9. Признаки в файл не записаны: `load_year_panel()` подтягивает их по `h3`.

| колонка | смысл |
|---|---|
| `h3`, `year` | ячейка, календарный год |
| `count_year` | число ДТП с пострадавшими за год (у 100 % ДТП выборки есть погибшие или раненые) |
| `had_accident_year` | `count_year ≥ 1` (int8) |
| `count_fatal_year` | число ДТП с погибшими (`has_fatal`) |
| `count_dead_persons_year`, `count_injured_persons_year` | сумма `dead_count`, сумма `injured_count` |
| `severity_weight_year` | сумма `severity_weight` (для этапа 8) |
| `count_type_collision`, `_pedestrian`, `_obstacle`, `_passenger_fall`, `_cyclist`, `_parked_vehicle`, `_rollover`, `_run_off_road`, `_other` | число ДТП по видам `accident_type`; отдельно выделены виды с долей ≥ 1 % в bbox (45.9 / 37.3 / 6.1 / 3.5 / 2.4 / 1.9 / 1.25 / 0.99 %), остальные 8 видов (0.7 %) попали в `other`; нужно для разметки ОДМ (7.7) |
| `accidents_prev_year` | `count_year` за Y−1; NaN в 2015 г. |
| `accidents_prev_3y` | Σ `count_year` за Y−3…Y−1; NaN, если нет трёх полных прошлых лет (2015–2017) |

**Проверки** (`scripts/check_v2_eval_year.py`, все пройдены, 7.5 с):
- сумма ДТП по каждой паре «ячейка × год» равна сумме месячного `accident_count` (всего 15 927 за 2015–2025);
- в bbox 15 994 ДТП, ещё 67 приходятся на ячейки без дорог и, как в месячной схеме, в панель не входят;
- сумма видов равна `count_year`, `fatal ≤ count`, `severity ≥ count`;
- лаги точно равны прошлым годам; если подменить счёты 2023+ на 999, лаги до 2023 г. не меняются;
- каппы p99 не меняются при подмене признаков в годах оценки;
- 204 колонки признаков ячейки совпадают со значениями месячной панели.

**Полнота 2025 г.** В `crashes_slim` 12 месяцев каждого года 2015–2025 и ещё январь 2026 г. (в панель не
входит). В декабре 2025 г. 124 ДТП, обычный уровень для месяца — 74–130. **2025 г. полный, тест 2023–2025
остаётся как в плане.**

| год | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ДТП | 1509 | 1401 | 1311 | 1520 | 1644 | 1595 | 1520 | 1448 | 1388 | 1297 | 1294 |
| ячеек с ≥ 1 ДТП, % | 15.7 | 14.9 | 13.9 | 16.3 | 16.6 | 16.5 | 15.9 | 16.0 | 16.3 | 15.7 | 15.3 |
| дисперсия / среднее | 3.8 | 3.4 | 3.4 | 3.3 | 3.6 | 3.4 | 3.3 | 3.0 | 2.8 | 2.9 | 2.8 |

Ячеек без ДТП за год 83–86 %, максимум в одной ячейке 10–16 ДТП за год. Дисперсия в ≈ 3 раза больше
среднего, т. е. у счёта сверхдисперсия: это отмечено здесь, а описание остаётся за агентом B.
Подробно: `year_panel_by_year.csv`, рис. `fig1_year_panel_counts.png`.

## Сплиты (`vy.YEAR_SPLITS`)

| сплит | обучение (+ каппы p99) | оценка | строк обуч. / оценки | ДТП в оценке | доля ячеек-лет с ≥ 1 ДТП в оценке |
|---|---|---|---|---|---|
| `y2021` | 2015–2020 | 2021 | 25 776 / 4 296 | 1 520 | 15.9 % |
| `y2022` | 2015–2021 | 2022 | 30 072 / 4 296 | 1 448 | 16.0 % |
| `test` | 2015–2022 | 2023, 2024, 2025 | 34 368 / 12 888 | 3 979 (1388 + 1297 + 1294) | 15.8 % |

Группы: `year_folds` = y2021 + y2022, `year_test` = test, `both` = все три. Доля событий в обучении 15.7 %.
Строк в 12 раз меньше, чем в месячной схеме (567 072), а соседние строки больше не дублируют таргет.

На тесте у каждого прогона есть пять значений `eval_period`:
- `2023`, `2024`, `2025` — метрики по каждому году;
- `mean` — среднее этих трёх;
- `sum` — факт за три года вместе; скор равен `3 × прогноз 2023 г.`, т. е. прогнозу, сделанному до 2023 г.

Почему в `sum` не используется сумма погодовых скоров: для скоров с лагами это утечка. Прогноз 2024 г.
по `accidents_prev_year` равен факту 2023 г., а этот факт входит в суммарный таргет. Так первый дымовой
прогон получил Spearman 0.90 вместо 0.62 (найдено и исправлено, см. LOG). Для моделей на статичных
признаках скор одинаков во все годы, поэтому разницы нет.

## API модуля `src/road_accident_risk/v2_eval_year.py` (импорт `from road_accident_risk import v2_eval_year as vy`)

| объект | что делает |
|---|---|
| `build_year_panel()` | строит и пишет панель (`scripts/build_h3_year_panel.py`, 2 с) |
| `load_year_panel(feature_set=None, attach=DEFAULT_ATTACH)` | панель + все 204 признака ячейки (этапы 0–6a, включая `exposure_v6`); 3 с |
| `feature_columns(panel)`, `year_variant_features(name, panel)` | признаки ячейки; признаки варианта — любое имя `v2_eval` (наборы, абляции этапа 5, `ve.register_variant`); суффикс `+lags` добавляет `accidents_prev_year`, `accidents_prev_3y` |
| `YEAR_SPLITS`, `SPLIT_GROUPS`, `split_frames_year`, `split_summary_year` | сплиты |
| `make_model(mode, loss, class_weights, seed, params)` | CatBoost 800 / 0.05 / 6. `binary`: Logloss, `auto_class_weights` = 'Balanced' / None. `count`: 'Poisson' / 'Tweedie:variance_power=p'. `params` переопределяет любые параметры |
| `run_year_variant(panel, name, split, mode, loss, class_weights, seeds, features, params, return_predictions, importance_out)` | обучение и оценка; строка = (split, run_id, seed, eval_period) + метрики; `run_id = "<вариант>\|<модель>"`, модель: `count:Poisson`, `binary:Logloss:Balanced`, `binary:Logloss:None`, … |
| `metrics_year(y, s, severity, is_count)` | метрики 7.4 для одного периода (см. ниже) |
| `coverage_at`, `coverages`, `gain_curve(y, s, n_points=101)` | доля накрытых ДТП в верхних q % ячеек и кривая «доля ячеек → доля ДТП» с честным учётом связей |
| `calibration_table(y, pred, n_bins=10)` | калибровка по децилям: n, Σпрогноз, Σфакт, ratio |
| `bootstrap_coverage(y, s, fracs, B=1000, seed=20261019, s_base=None)` | бутстреп по ячейкам: ДИ доли накрытых ДТП; с `s_base` — парная разность |
| `evaluate_year_predictions(df, score_col, is_count)` | метрики по годам + `mean` + `sum` |
| `period_frame(preds, run_id, split, period)`, `bootstrap_from_predictions(preds, run_id, split, period, base_run_id)` | факт и скор периода из таблицы прогнозов (скор усредняется по сидам), бутстреп по ней |
| `run_baseline(panel, score, split, name, is_count, fillna, return_predictions)` | ориентир без модели: колонка панели или функция `panel -> Series`; seed 0, модель `baseline` |
| `monthly_model_year_scores(name, split, seeds, month_panel, strict=True)` | месячная модель `v2_eval` (Balanced, «ДТП в следующем месяце») → `score_mean`, `score_max` по 12 прогнозам, чьи целевые месяцы лежат в году Y |
| `run_monthly_as_year(year_panel, name, split, seeds, agg, scores=..., return_predictions)` | годовые метрики месячной модели, run_id `<вариант>\|monthly:mean` |
| `compare_year(runs, base, variant, split, period)` | Δ = variant − base, 2σ = 2·√((sd_A² + sd_B²)/n), парная 2·sd разностей, вердикты `v2_eval.verdict`; `higher_is_better`, `better` |
| `compare_folds_year(runs, base, variant)` | среднее Δ по фолдам y2021 и y2022, сидовый 2σ среднего, `delta_y2021`, `delta_y2022`, `same_sign_all_folds` |
| `upsert_year_runs(path, runs)` | дозапись CSV с заменой по (split, run_id, seed, eval_period) |

**Метрики** (`metrics_year`, колонки CSV):
- `coverage_top_{1,5,10}pct` — доля ДТП в верхних 1/5/10 % ячеек;
- `sev_coverage_top_*` — то же по `severity_weight_year`;
- `coverage_oracle_top_*` — потолок: ранжирование по факту;
- `spearman`;
- `roc_auc`, `pr_auc` — для события «≥ 1 ДТП»;
- `poisson_deviance` (среднее), `calib_ratio` (Σпрогноз / Σфакт), `calib_top_decile_ratio` — только при
  `is_count=True`, т. е. для режима `count` или ориентира в единицах ДТП; для бинарных и ранговых скоров NaN;
- `n_cells`, `n_accidents`, `event_rate`, `fit_seconds`.

`k = max(1, int(n·q))`, как в `precision_recall_at_k`. При 4 296 ячейках в верхний 1 % попадают 42 ячейки.
Связи учитываются честно: внутри группы равных скоров кривая линейна, что равно матожиданию при случайном
порядке. Без этого у ориентира `accidents_prev_year`, где у 84 % ячеек 0, результат зависел бы от порядка строк.

**Вердикты.** Используется то же соглашение, что раньше: |Δ|/2σ < 1 — шум, 1–1.5 — на границе, ≥ 1.5 — значимо.
Если 2σ = 0 (один сид или сравнение двух ориентиров), в таблице стоит «нет 2σ». Неопределённость разницы
в этих случаях можно оценить бутстрепом по ячейкам. Для `poisson_deviance` лучше меньшее значение,
для `calib_ratio` — значение ближе к 1. Направление сравнения указано в колонке `better`.

### Раннер `scripts/run_v2_eval_year.py`

```text
PYTHONPATH=src venv/bin/python scripts/run_v2_eval_year.py --variants road_geo_v2_skdf[,…] \
    --mode count|binary [--loss Poisson|'Tweedie:variance_power=1.5'] [--class-weights Balanced|None] \
    --split year_folds|year_test|both [--seeds 42,7,123,2024,31337] [--params '{"depth": 8}'] \
    [--baselines accidents_prev_year,accidents_prev_3y,major_road_length_m] [--monthly mean,max] \
    --tag <tag> [--base '<run_id>' | --base baseline:<колонка>] [--bootstrap] [--no-preds]
```

Раннер пишет в `reports/`:
- `v2_eval_year_<tag>.csv` — сырые прогоны по сидам и периодам;
- `_preds.parquet` — прогнозы `split, run_id, seed, h3, year, pred, count_year, severity_weight_year`;
- `_delta.csv` при `--base` — разности по каждому сплиту и периоду; `eval_period = folds_mean`, если
  прогнаны оба фолда;
- `_bootstrap.csv` при `--bootstrap` — ДИ доли накрытых ДТП на тесте (`sum` и по годам) и парные
  разности к базе.

### Примеры вызова

```python
from road_accident_risk import v2_eval as ve, v2_eval_year as vy
panel = vy.load_year_panel()

# агент A: счётная модель, 5 сидов, оба фолда; Tweedie; бинарная без весов; структура + история
runs = pd.concat([vy.run_year_variant(panel, "road_geo_v2_skdf", s, mode="count", loss="Poisson") for s in ("y2021", "y2022")])
runs_tw = vy.run_year_variant(panel, "road_geo_v2_skdf", "y2022", mode="count", loss="Tweedie:variance_power=1.5")
runs_b = vy.run_year_variant(panel, "road_geo_v2_skdf", "y2022", mode="binary", class_weights=None)
runs_l = vy.run_year_variant(panel, "road_geo_v2_skdf+lags", "y2022", mode="count")
vy.compare_folds_year(all_runs, "road_geo_v2_skdf|count:Poisson", "road_geo_v2_skdf|binary:Logloss:None")
# 7.2: месячная модель как годовой скор (одно обучение на сид, mean и max сразу)
sc = vy.monthly_model_year_scores("road_geo_v2_skdf", "test")
m = pd.concat([vy.run_monthly_as_year(panel, "road_geo_v2_skdf", "test", agg=a, scores=sc) for a in ("mean", "max")])
# прогнозы для ЭБ-ориентира и разметки ОДМ
r, preds = vy.run_year_variant(panel, "road_geo_v2_skdf", "test", mode="count", return_predictions=True)
vy.bootstrap_from_predictions(preds, "road_geo_v2_skdf|count:Poisson", "test", "sum")

# агент B: ориентиры без модели и описательное
b1 = vy.run_baseline(panel, "accidents_prev_3y", "test")
b2 = vy.run_baseline(panel, lambda d: d["accidents_prev_3y"] / 3, "test", name="baseline:prev3y_rate", is_count=True)
b3 = vy.run_baseline(panel, "major_road_length_m", "y2022")
g = vy.gain_curve(y, s)                      # массив для графика
ct = vy.calibration_table(y, pred)
odm = (panel[vy.TYPE_COLS[:-1]].max(axis=1) >= 3) | (panel["count_year"] >= 5)   # очаг ОДМ (без «other»)
```

## Дымовой прогон (seed 42, `--tag stage7_infra_smoke`)

Это не результат, а проверка, что всё работает: сид один, 2σ нет. Указан ряд `sum` теста (2023–2025) и
фолды 2021 / 2022.

| прогон | тест sum: доля ДТП в топ-1 / 5 / 10 % | Spearman | ROC / PR (≥1) | y2021 топ-10 % | y2022 топ-10 % |
|---|---|---|---|---|---|
| `road_geo_v2_skdf\|count:Poisson` | 0.148 / 0.462 / 0.674 | 0.685 | 0.941 / 0.872 | 0.687 | 0.692 |
| `road_geo_v2_skdf\|binary:Logloss:Balanced` | 0.124 / 0.466 / 0.670 | 0.678 | 0.936 / 0.871 | 0.691 | 0.676 |
| `road_geo_v2_skdf\|monthly:mean` (= max) | 0.145 / 0.468 / 0.675 | 0.677 | 0.935 / 0.869 | 0.688 | 0.682 |
| ориентир `accidents_prev_3y` | 0.140 / 0.444 / 0.646 | 0.715 | 0.843 / 0.733 | 0.684 | 0.683 |
| ориентир `accidents_prev_year` | 0.122 / 0.392 / 0.585 | 0.619 | 0.739 / 0.584 | 0.615 | 0.599 |
| ориентир `major_road_length_m` | 0.053 / 0.174 / 0.370 | 0.536 | 0.749 / 0.522 | 0.415 | 0.369 |
| потолок (ранжирование по факту) | 0.171 / 0.521 / 0.746 | — | — | 0.832 | 0.822 |

- **Poisson, тест по годам** 2023 / 2024 / 2025:
  - доля ДТП в топ-10 %: 0.681 / 0.670 / 0.670;
  - Spearman: 0.565 / 0.552 / 0.538;
  - отклонение Пуассона: 0.418 / 0.430 / 0.462;
  - Σпрогноз/Σфакт: 1.08 / 1.15 / 1.15 — модель переоценивает, потому что число ДТП к 2024–2025 гг. упало.
- **Бутстреп** (B = 1000, тест sum), доля ДТП в топ-10 % у Poisson: 0.674 [0.647; 0.698]. Парная разность
  к Poisson у `accidents_prev_3y`: −0.028 [−0.044; −0.015].
- **Месячная модель.** У наборов v2 все признаки статичны, поэтому 12 месячных прогнозов ячейки совпадают
  и `mean = max`. Эффект горизонта в 7.2 сводится к сравнению «модель, обученная на месячном таргете» и
  «модель, обученная на годовом».
- **Время одной подгонки** при load average 4–7: Poisson 4–5 с, бинарная 5–8 с, месячная модель 33–43 с.
  Отсюда оценки на вариант с пятью сидами:
  - `year_folds` ≈ 45–80 с;
  - `year_test` ≈ 25–40 с;
  - `both` ≈ 1.2–2 мин;
  - месячная модель как годовой скор на `both` ≈ 9–10 мин;
  - бутстреп одного прогона на 4 периода ≈ 10 с.
- Рисунок `fig2_gain_curves_smoke.png` (+ CSV) — кривые gain на тесте (sum).

**Месячная схема не изменилась.** `run_v2_eval.py --split test --seeds 42` даёт `road_geo_v2_skdf`
0.908235 / 0.199760 / 0.3157 и `road_geo_baseline` 0.908056 / 0.198881 / 0.3072, обе тройки совпадают с
эталонами (`reports/v2_eval_stage7_infra_month_check.csv`).

## Инструкция для шага 2

**Общее для A и B.**
- Модули `v2_eval_year.py`, `v2_eval.py`, панель `h3_year_panel.parquet` и раннер остаются общими.
  Дополнительные функции размещаются в отдельных скриптах с импортом `vy` и `ve`.
  Варианты регистрируются через `ve.register_variant`.
- Для разделения CSV в тегах отчётов используется префикс агента: `--tag stage7A_…` или
  `stage7B_…`. Прогоны дописываются с заменой по ключу, но параллельная запись в один файл небезопасна.
- Решения отбираются по `year_folds` × 5 сидов по правилу этапа 6: Δ значима и знак в обоих фолдах
  одинаков. `year_test` используется для итоговой оценки.
- Проверка перед запуском — `scripts/check_v2_eval_year.py` (8 с).

**Агент A — модели 7.2, 7.3 (и 7.6, 7.8 позже)**, файлы в `docs/analysis/features/stage7_count12/models/`.
- `run_v2_eval_year.py --mode binary --class-weights Balanced|None`, `--mode count --loss Poisson|'Tweedie:variance_power=1.2'|…`.
- Месячная модель как годовой скор — `--monthly mean` или `monthly_model_year_scores` + `run_monthly_as_year`.
- Структура + история — вариант `road_geo_v2_skdf+lags`.
- Параметр k сверхдисперсии оценивается по столбцам `count` и `pred` в `_preds.parquet`.
- Прогнозы для ЭБ-ориентира лежат в `_preds.parquet`.

**Агент B — описательная часть 7.1, ориентиры 7.5 без моделей, разметка ОДМ 7.7**, файлы в
`docs/analysis/features/stage7_count12/descriptive/`.
- Распределение счёта и сверхдисперсия оцениваются по `h3_year_panel.parquet`. Spearman «год к году» — по
  `pivot(h3, year, count_year)`.
- Ориентиры: `vy.run_baseline(panel, "accidents_prev_year" | "accidents_prev_3y" | "major_road_length_m" |
  "road_length_total_m" | функция, split)` или `--baselines` раннера без `--variants`. Для скоров в
  единицах ДТП (например `prev_3y/3`) передаётся `is_count=True`.
- Очаги ОДМ: `count_type_*` (≥ 3 одного вида; `other` — смесь видов, его не считать) или `count_year ≥ 5`.
  Доля очагов в верхних q % прогноза рассчитывается по `_preds.parquet` агента A. До готовности прогнозов
  ориентиром служит статус очага в прошлом году (сдвиг по `year`).
- Кривые — `vy.gain_curve`; бутстреп — `vy.bootstrap_coverage`.

## Ограничения

- `accidents_prev_3y` определён только с 2018 г. Обучение моделей с `+lags` видит NaN в 2015–2017 гг.
  (CatBoost обрабатывает NaN как отдельное значение).
- Скор месячной модели обучается со `strict=True`: в обучение берутся только метки из годов обучения,
  поэтому для теста строки ограничены `month < 2022-12-01`. Это на один месяц меньше, чем в месячном
  эталоне; `strict=False` воспроизводит границу `v2_eval`.
- В bbox 67 ДТП (0.4 %) приходятся на ячейки без дорог и не учитываются, как и в месячной схеме.
