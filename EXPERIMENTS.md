# Experiment Log

## 2026-07-01 - Road-Only 4.0 Binary 6M Model Family Check

### Что сделано

Проведена финальная проверка постановки абсолютного риска: на том же конфиге, что `CatBoost 4.0: ДТП в следующие 6 месяцев без весов`, обучены альтернативные семейства моделей.

База эксперимента:

```text
target: будет ли ДТП в следующих 6 месяцах
feature set: road_without_type_lengths_no_calendar_slim_dedup
feature count: 60
filter: road_length_total_m > 0
calendar features: исключены
p99 clipping: включен
class weights: нет
train_end: 2023-01-01
```

Дополнительно установлены `xgboost==3.2.0` и `lightgbm==4.6.0`, после чего проверены:

- `CatBoostClassifier` без весов - baseline из предыдущего эксперимента;
- `XGBoost XGBClassifier`;
- `LightGBM LGBMClassifier`;
- `sklearn HistGradientBoostingClassifier`;
- `sklearn RandomForestClassifier`.

Для sklearn-моделей добавлен `SimpleImputer(strategy="median")`, потому что случайный лес хуже переносит пропуски, чем CatBoost. XGBoost и LightGBM использовали нативную обработку пропусков.

Артефакты:

```text
scripts/run_road_only_4_0_binary_6m_model_family_experiments.py
reports/road_only_4_0_binary_6m_model_family_summary.csv
reports/road_only_4_0_binary_6m_model_family_topk.csv
reports/road_only_4_0_binary_6m_model_family_confusion.csv
parquet_output/h3_month_road_only_4_0_binary_next_6m_xgboost_predictions.parquet
parquet_output/h3_month_road_only_4_0_binary_next_6m_lightgbm_predictions.parquet
parquet_output/h3_month_road_only_4_0_binary_next_6m_hist_gradient_boosting_predictions.parquet
parquet_output/h3_month_road_only_4_0_binary_next_6m_random_forest_predictions.parquet
```

В viewer добавлены слои:

- `XGBoost 4.0: ДТП в след. 6м без весов`;
- `LightGBM 4.0: ДТП в след. 6м без весов`;
- `HGB 4.0: ДТП в след. 6м без весов`;
- `RandomForest 4.0: ДТП в след. 6м без весов`.

### Метрики

| Model | ROC-AUC | PR-AUC | Accuracy@0.5 | Precision@0.5 | Recall@0.5 | F1@0.5 | Predicted positive rate@0.5 | Top 1% precision | Top 5% precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CatBoost без весов | 0.925539 | 0.619825 | 0.918724 | 0.646160 | 0.470511 | 0.544521 | 0.075186 | 0.876784 | 0.720487 |
| XGBoost | 0.922847 | 0.613701 | 0.918274 | 0.640553 | 0.475093 | 0.545553 | 0.076583 | 0.878287 | 0.715530 |
| LightGBM | 0.922959 | 0.614722 | 0.918927 | 0.644379 | 0.479383 | 0.549769 | 0.076816 | 0.875282 | 0.718084 |
| HistGradientBoosting | 0.922574 | 0.614190 | 0.919227 | 0.647227 | 0.478583 | 0.550274 | 0.076350 | 0.879790 | 0.715380 |
| RandomForest | 0.908515 | 0.606934 | 0.918424 | 0.639833 | 0.480329 | 0.548725 | 0.077514 | 0.878287 | 0.714028 |

Confusion matrix при threshold `0.5`:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| CatBoost без весов | 115,882 | 3,543 | 7,281 | 6,470 |
| XGBoost | 115,759 | 3,666 | 7,218 | 6,533 |
| LightGBM | 115,787 | 3,638 | 7,159 | 6,592 |
| HistGradientBoosting | 115,838 | 3,587 | 7,170 | 6,581 |
| RandomForest | 115,707 | 3,718 | 7,146 | 6,605 |

### Интерпретация

Смена семейства модели не дала качественного скачка.

`XGBoost`, `LightGBM` и `HistGradientBoosting` почти догнали CatBoost, но не обогнали его по главным ranking-метрикам. У них чуть выше `F1@0.5` и `recall@0.5`, но ниже `ROC-AUC`, `PR-AUC` и `Top 5% precision`.

Лучший из добавленных внешних бустингов - `LightGBM`: `ROC-AUC = 0.922959`, `PR-AUC = 0.614722`, `Top 5% precision = 0.718084`. Это близко к CatBoost, но все еще немного хуже базового CatBoost без весов.

`RandomForest` заметно слабее по `ROC-AUC` (`0.9085` против `0.9255` у CatBoost), но top-k precision остается близким. Это значит, что самые очевидные high-risk ячейки разные модели находят похоже, но общий порядок ранжирования у леса хуже.

Главный вывод: проблема абсолютного риска не в том, что CatBoost "не та модель". При смене модели сигнал остается тем же: дорожная плотность, городская структура и связанные с ними признаки хорошо объясняют, где ДТП вероятнее в абсолютном смысле. Более сложная или другая модель не меняет смысл постановки.

Практически можно оставить CatBoost как основной вариант для абсолютного риска: он дает лучший общий ranking (`ROC-AUC`, `PR-AUC`, `Top 5% precision`) и уже лучше встроен в текущий pipeline/viewer. Эксперимент с model family можно считать закрывающей проверкой перед остановкой направления абсолютного риска.

## 2026-07-01 - CatBoost Road-Only 4.0 Binary 6M Class Weight Experiments

### Что сделано

Проверена гипотеза: слой `binary_next_6m` красит почти весь город в красный, потому что для 6-месячного target положительный класс уже не настолько редкий (`positive rate ~= 10.3%`), а `auto_class_weights="Balanced"` все еще слишком агрессивно разгоняет положительный класс.

Обучены пять вариантов одной и той же модели:

```text
target: будет ли ДТП в следующих 6 месяцах
feature set: road_without_type_lengths_no_calendar_slim_dedup
feature count: 60
filter: road_length_total_m > 0
calendar features: исключены
p99 clipping: включен
train_end: 2023-01-01
```

Варианты весов:

- без весов;
- positive class weight = 2;
- positive class weight = 4;
- positive class weight = 6;
- `auto_class_weights="Balanced"`.

Артефакты:

```text
reports/catboost_road_only_4_0_binary_6m_weight_summary.csv
reports/catboost_road_only_4_0_binary_6m_weight_topk.csv
reports/catboost_road_only_4_0_binary_6m_weight_confusion.csv
reports/catboost_feature_importance_road_only_4_0_binary_6m_*.csv
parquet_output/h3_month_catboost_road_only_4_0_binary_next_6m_weight_*_predictions.parquet
```

Также viewer очищен от старых/диагностических слоев:

- `CatBoost road-only 3.5`;
- старый `CatBoost road-only 4.0`;
- `CatBoost road-only 4.0 >2000м`;
- `CatBoost road-only 4.1 log`;
- `CatBoost 4.0: ДТП в след. месяц`;
- `CatBoost 4.0: кол-во ДТП в след. месяц`;
- старый plain `CatBoost 4.0: ДТП в след. 6м`, заменен на явные weight-варианты.

В viewer добавлены слои:

- `CatBoost 4.0: ДТП в след. 6м без весов`;
- `CatBoost 4.0: ДТП в след. 6м вес 2`;
- `CatBoost 4.0: ДТП в след. 6м вес 4`;
- `CatBoost 4.0: ДТП в след. 6м вес 6`;
- `CatBoost 4.0: ДТП в след. 6м Balanced`.

### Метрики

| Variant | ROC-AUC | PR-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 | Predicted positive rate@0.5 | Score p50 | Score p90 | Score p95 | Score p99 | Top 1% precision | Top 5% precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Без весов | 0.925539 | 0.619825 | 0.646160 | 0.470511 | 0.544521 | 0.075186 | 0.012370 | 0.389421 | 0.624629 | 0.890650 | 0.876784 | 0.720487 |
| Вес 2 | 0.925195 | 0.619662 | 0.561963 | 0.632172 | 0.595003 | 0.116155 | 0.021940 | 0.565151 | 0.767282 | 0.933634 | 0.869271 | 0.723791 |
| Вес 4 | 0.924855 | 0.621230 | 0.475002 | 0.751727 | 0.582153 | 0.163408 | 0.039106 | 0.722774 | 0.866622 | 0.959663 | 0.866266 | 0.725293 |
| Вес 6 | 0.924128 | 0.622503 | 0.422922 | 0.810414 | 0.555797 | 0.197858 | 0.050418 | 0.796142 | 0.902806 | 0.971042 | 0.895567 | 0.725143 |
| Balanced | 0.924312 | 0.622179 | 0.382007 | 0.856883 | 0.528433 | 0.231611 | 0.059493 | 0.845722 | 0.930024 | 0.978536 | 0.884298 | 0.732803 |

Confusion matrix при threshold `0.5`:

| Variant | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Без весов | 115,882 | 3,543 | 7,281 | 6,470 |
| Вес 2 | 112,649 | 6,776 | 5,058 | 8,693 |
| Вес 4 | 108,000 | 11,425 | 3,414 | 10,337 |
| Вес 6 | 104,219 | 15,206 | 2,607 | 11,144 |
| Balanced | 100,363 | 19,062 | 1,968 | 11,783 |

### Интерпретация

Гипотеза подтвердилась: `Balanced` действительно делает 6-месячный binary layer слишком агрессивным.

По мере роста веса положительного класса:

- score distribution сдвигается вверх;
- `predicted_positive_rate@0.5` растет с `7.5%` до `23.2%`;
- recall растет с `0.47` до `0.86`;
- precision падает с `0.65` до `0.38`;
- false positives растут с `3,543` до `19,062`.

То есть красная карта у `Balanced` - не просто проблема палитры. Модель реально ставит высокий score намного большему числу H3-ячеек.

При этом ranking-метрики почти не меняются:

- ROC-AUC держится около `0.924-0.926`;
- PR-AUC держится около `0.620-0.622`;
- Top 5% precision держится около `0.72-0.73`.

Это важный вывод: веса в основном меняют калибровку score и threshold-поведение, но не радикально меняют порядок самых рискованных ячеек.

### Практический вывод

`Balanced` для `binary_next_6m` выглядит чрезмерным: он дает максимальный recall, но слишком сильно красит город и генерирует много false positives.

Если нужна более спокойная карта и меньше ложных high-risk ячеек, лучший кандидат - `weight_2`:

- F1 самый высокий: `0.595`;
- precision заметно выше Balanced: `0.562` против `0.382`;
- recall еще нормальный: `0.632`;
- predicted positive rate близок к реальной частоте positive класса: `11.6%` против `10.3%`.

Если важнее не пропускать потенциально опасные зоны, но хочется быть менее агрессивным, чем Balanced, компромиссный вариант - `weight_4`:

- recall `0.752`;
- precision `0.475`;
- карта должна быть заметно менее красной, чем Balanced, но все еще достаточно чувствительной.

`weight_none` дает самую спокойную карту и высокую precision, но recall слишком низкий: модель пропускает больше половины positive rows. Его полезно держать как нижнюю границу агрессивности, но не как основной слой.

Рабочая рекомендация на текущий момент:

- для просмотра карты риска начать с `weight_2`;
- если визуально слишком спокойно и пропусков многовато - смотреть `weight_4`;
- `Balanced` оставить как верхнюю границу "максимально чувствительного" сценария, но не как основной 6-месячный слой.

## 2026-07-01 - CatBoost Road-Only 4.0 Target Horizon Experiments

### Что сделано

Проверены альтернативные постановки target для основного feature set `CatBoost road-only 4.0`.

База эксперимента:

```text
feature set: road_without_type_lengths_no_calendar_slim_dedup
feature count: 60
filter: road_length_total_m > 0
calendar features: исключены
p99 clipping: включен отдельно на train split для каждого target
train_end: 2023-01-01
```

Проверенные target:

- `binary_next_1m` - будет ли ДТП в следующем месяце; это воспроизведение исходного 4.0 baseline;
- `count_next_1m` - количество ДТП в следующем месяце;
- `count_next_3m` - количество ДТП в следующих 3 месяцах;
- `count_next_6m` - количество ДТП в следующих 6 месяцах;
- `binary_next_3m` - будет ли ДТП в следующих 3 месяцах;
- `binary_next_6m` - будет ли ДТП в следующих 6 месяцах.

Для горизонтов 3 и 6 месяцев последние месяцы по каждому H3 исключаются из оценки, потому что для них нет полного будущего окна. Поэтому `test_rows` меньше, чем у исходного `next_1m`.

Также из viewer удалены диагностические слои ошибок 4.0 (`FP count`, `FN count`, `TP count`, `FP share`, `actual positive rate`, `predicted positive rate`). CSV/parquet артефакты error analysis сохранены, но из интерфейса они больше не торчат.

Артефакты:

```text
reports/catboost_road_only_4_0_target_horizon_summary.csv
reports/catboost_road_only_4_0_target_horizon_topk.csv
reports/catboost_road_only_4_0_target_horizon_confusion.csv
reports/catboost_feature_importance_road_only_4_0_*next*.csv
parquet_output/h3_month_catboost_road_only_4_0_*next*_predictions.parquet
```

### Binary targets

| Target | Test rows | Positive rate | ROC-AUC | PR-AUC | Accuracy@0.5 | Precision@0.5 | Recall@0.5 | F1@0.5 | Top 1% precision | Top 5% precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `binary_next_1m` | 154,656 | 2.335% | 0.907661 | 0.197979 | 0.820110 | 0.100653 | 0.844918 | 0.179878 | 0.303364 | 0.206803 |
| `binary_next_3m` | 146,064 | 6.157% | 0.915870 | 0.441762 | 0.832163 | 0.248052 | 0.849661 | 0.383999 | 0.665753 | 0.491853 |
| `binary_next_6m` | 133,176 | 10.325% | 0.924312 | 0.622179 | 0.842089 | 0.382007 | 0.856883 | 0.528433 | 0.884298 | 0.732803 |

Confusion matrix при threshold `0.5`:

| Target | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| `binary_next_1m` | 123,784 | 27,261 | 560 | 3,051 |
| `binary_next_3m` | 113,908 | 23,163 | 1,352 | 7,641 |
| `binary_next_6m` | 100,363 | 19,062 | 1,968 | 11,783 |

### Count targets

Для count targets CatBoost обучался как регрессор с `Poisson` loss. Для сравнения с бинарными задачами дополнительно считались ROC-AUC/PR-AUC по событию `count > 0`, где predicted count использовался как risk score.

| Target | Test rows | Event rate | Mean count | Total count | ROC-AUC event | PR-AUC event | MAE | RMSE | Poisson deviance | Spearman | Top 1% precision | Top 5% precision | Top 1% count recall | Top 5% count recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `count_next_1m` | 154,656 | 2.335% | 0.025547 | 3,951 | 0.909413 | 0.199752 | 0.045716 | 0.163512 | 0.124485 | 0.214220 | 0.313066 | 0.203958 | 0.150595 | 0.460896 |
| `count_next_3m` | 146,064 | 6.157% | 0.077350 | 11,298 | 0.917292 | 0.440412 | 0.105075 | 0.289655 | 0.234235 | 0.348010 | 0.676027 | 0.483500 | 0.150735 | 0.457869 |
| `count_next_6m` | 133,176 | 10.325% | 0.154397 | 20,562 | 0.925368 | 0.619351 | 0.167051 | 0.422476 | 0.323684 | 0.450269 | 0.899324 | 0.709673 | 0.150861 | 0.458710 |

### Сравнение с исходным 4.0

Контрольный `binary_next_1m` воспроизвел исходный `CatBoost road-only 4.0`:

```text
ROC-AUC: 0.907661
PR-AUC: 0.197979
Precision@0.5: 0.100653
Recall@0.5: 0.844918
Top 1% precision: 0.303364
Top 5% precision: 0.206803
```

То есть новый скрипт действительно сравнивает target-постановки на том же baseline-конфиге.

При переходе от `next_1m` к `next_3m` и `next_6m` positive rate растет:

```text
2.335% -> 6.157% -> 10.325%
```

Из-за этого threshold-метрики становятся намного адекватнее:

```text
Precision@0.5: 0.100653 -> 0.248052 -> 0.382007
F1@0.5:        0.179878 -> 0.383999 -> 0.528433
```

Recall при этом почти не меняется и остается около `0.85`. То есть модель продолжает ловить большую часть positive rows, но при длинном горизонте среди predicted positive становится сильно меньше бессмысленного шума.

PR-AUC растет особенно сильно:

```text
0.197979 -> 0.441762 -> 0.622179
```

Это ожидаемо: положительный класс перестает быть настолько редким. Но это не надо трактовать как чистый прирост качества модели. Задача стала менее шумной: предсказать "будет ли ДТП в течение 6 месяцев" объективно проще, чем "будет ли ДТП ровно в следующем месяце".

Top-k precision тоже становится намного понятнее:

```text
Top 1% precision: 0.303364 -> 0.665753 -> 0.884298
Top 5% precision: 0.206803 -> 0.491853 -> 0.732803
```

Для практического ранжирования это важнее threshold `0.5`: если брать верхние 1-5% самых рискованных H3-ячеек, на горизонте 3/6 месяцев модель уже выделяет зоны, где событие действительно часто наступает.

### Count vs binary

Count-модели не дали принципиально лучшего event-ranking по сравнению с binary-моделями на тех же горизонтах:

- `1m`: count чуть лучше по ROC-AUC/PR-AUC и Top 1%, но почти то же самое;
- `3m`: count и binary практически одинаковы по PR-AUC;
- `6m`: binary чуть лучше по PR-AUC и Top 5%, count чуть лучше по ROC-AUC и Top 1%.

То есть если вопрос звучит как "будет ли ДТП", binary target остается нормальным и даже более прямым вариантом.

Но count target дает другой смысл: он пытается ранжировать не просто вероятность события, а ожидаемую аварийную нагрузку. Это видно по `count_recall`: top 5% ячеек по predicted count забирает примерно `46%` всех будущих ДТП в каждом горизонте:

```text
count_next_1m Top 5% count recall: 46.09%
count_next_3m Top 5% count recall: 45.79%
count_next_6m Top 5% count recall: 45.87%
```

Это полезно, если задача будет формулироваться как "где ожидается больше всего ДТП", а не только "где ДТП случится хотя бы раз".

### Интерпретация

Главный вывод: месячный бинарный target слишком шумный для такой редкой задачи. Он формально работает, но дает много false positives, потому что требует угадать событие в очень узком временном окне.

Горизонт 3/6 месяцев выглядит содержательно лучше для absolute risk:

- target становится менее редким;
- PR-AUC и F1 становятся заметно выше;
- top-k precision становится намного более интерпретируемым;
- сама постановка ближе к практической задаче планирования: не "будет ли авария именно в следующем месяце", а "какие ячейки опасны в ближайший период".

При этом это все еще absolute risk. Модель по-прежнему будет поднимать плотные городские дорожные зоны, потому что там в абсолютном выражении ДТП вероятнее. Но длинный горизонт уменьшает случайный месячный шум и делает результат менее дерганым.

Практический вывод:

- для прогноза "любое ДТП" лучше рассматривать `binary_next_3m` или `binary_next_6m`, а не `binary_next_1m`;
- `binary_next_6m` дает самые сильные и стабильные метрики, но теряет конкретность по времени;
- `binary_next_3m` выглядит как компромисс между устойчивостью и временной конкретностью;
- count target стоит держать как отдельную постановку, если нужно ранжировать места по ожидаемому количеству ДТП, а не только по вероятности хотя бы одного ДТП.

## 2026-07-01 - CatBoost Road-Only 4.0 Error Analysis

### Что сделано

Разобрана ошибка основной модели абсолютного риска `CatBoost road-only 4.0` при threshold `0.5`.

Модель:

```text
baseline: CatBoost road-only 4.0
feature set: road_without_type_lengths_no_calendar_slim_dedup
filter: road_length_total_m > 0
calendar features: исключены
p99 clipping: включен
threshold: 0.5
```

Строки тестовой выборки разделены на четыре группы:

- `TN` - ДТП нет, модель правильно ставит низкий риск;
- `FP` - ДТП нет, но модель ставит высокий риск;
- `FN` - ДТП есть, но модель ставит низкий риск;
- `TP` - ДТП есть, модель правильно ставит высокий риск.

Артефакты:

```text
reports/catboost_road_only_4_0_error_analysis_summary.csv
reports/catboost_road_only_4_0_error_analysis_feature_by_group.csv
reports/catboost_road_only_4_0_error_analysis_feature_contrasts.csv
reports/catboost_road_only_4_0_error_analysis_top_h3.csv
parquet_output/h3_month_catboost_road_only_4_0_error_rows.parquet
parquet_output/h3_catboost_road_only_4_0_error_analysis.parquet
reports/figures/catboost_road_only_4_0_error_analysis/
```

В viewer добавлены слои:

- `Ошибки 4.0: FP count`;
- `Ошибки 4.0: FN count`;
- `Ошибки 4.0: TP count`;
- `Ошибки 4.0: FP share`;
- `Ошибки 4.0: actual positive rate`;
- `Ошибки 4.0: predicted positive rate`.

### Confusion groups

| Group | Rows | Actual positives | Predicted positives | Score mean | Score median | Road length mean, m | Junction count mean | Asphalt length mean, m | Share of test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `TN` | 123,784 | 0 | 0 | 0.0897 | 0.0314 | 1,050.5 | 3.04 | 139.5 | 80.04% |
| `FP` | 27,261 | 0 | 27,261 | 0.7461 | 0.7617 | 2,006.4 | 10.16 | 811.5 | 17.63% |
| `FN` | 560 | 560 | 0 | 0.2738 | 0.2967 | 1,515.4 | 6.16 | 439.0 | 0.36% |
| `TP` | 3,051 | 3,051 | 3,051 | 0.8408 | 0.8772 | 2,309.7 | 13.07 | 1,027.1 | 1.97% |

Итого:

- модель ловит `3,051` из `3,611` positive rows, то есть recall по классу 1 равен примерно `84.5%`;
- модель пропускает `560` positive rows, то есть около `15.5%` будущих ДТП-строк;
- среди всех predicted positive rows только `3,051 / 30,312` являются true positive;
- false positives составляют около `89.9%` всех строк, которым модель поставила высокий риск.

### Что отличает ошибки

`FP` против `TN`: false positives не выглядят случайными пустыми местами. По признакам это гораздо более плотные дорожные/городские ячейки, чем true negatives:

| Feature | FP mean | TN mean | SMD |
|---|---:|---:|---:|
| `class_delta_max` | 4.244 | 1.385 | 1.804 |
| `lane_meters_estimate` | 2,630.8 | 1,117.0 | 1.675 |
| `asphalt_surface_length_m` | 811.5 | 139.5 | 1.674 |
| `lanes_mean_weighted` | 2.082 | 0.345 | 1.612 |
| `unsignalized_high_delta_junction_count` | 7.034 | 1.734 | 1.412 |
| `junction_count` | 10.165 | 3.037 | 1.334 |
| `road_length_total_m` | 2,006.4 | 1,050.5 | 1.236 |

Интерпретация: модель ставит высокий риск не совсем "мимо", а в ячейках, которые по дорожной структуре похожи на рискованные: больше дорог, асфальта, полос, пересечений, крупных перепадов классов дорог. То, что в конкретный месяц там нет ДТП, может быть следствием редкости события, неполноты регистрации ДТП или отсутствия внешних признаков экспозиции: трафика, скорости потока, плотности людей, камер, общественных мест.

`FN` против `TP`: false negatives - это реальные ДТП, которые происходят в менее выраженных "городских" ячейках. Они слабее TP почти по всем признакам дорожной насыщенности:

| Feature | FN mean | TP mean | SMD |
|---|---:|---:|---:|
| `lane_meters_estimate` | 1,787.0 | 3,237.0 | -1.598 |
| `class_delta_max` | 2.421 | 4.755 | -1.522 |
| `legacy_major_minor_junction_count` | 0.389 | 3.777 | -1.479 |
| `major_junction_count` | 0.454 | 4.286 | -1.423 |
| `public_transport_stop_count` | 0.461 | 3.242 | -1.335 |
| `asphalt_surface_length_m` | 439.0 | 1,027.1 | -1.287 |
| `traffic_signal_count` | 0.004 | 0.764 | -1.268 |
| `crossing_count` | 0.209 | 2.937 | -1.187 |

Интерпретация: модель хорошо ловит ДТП в плотной городской дорожной ткани, но хуже ловит ДТП в местах, которые по OSM-структуре выглядят умеренно насыщенными или почти "обычными". Это важный сигнал: часть настоящих аварий может объясняться не только структурой дороги, а факторами, которых у нас пока нет.

`TP` против `FP`: среди high-risk ячеек реальные ДТП отличаются от ложных срабатываний уже не так резко. Самые заметные отличия:

| Feature | TP mean | FP mean | SMD |
|---|---:|---:|---:|
| `lane_meters_estimate` | 3,237.0 | 2,630.8 | 0.630 |
| `traffic_signal_count` | 0.764 | 0.344 | 0.556 |
| `crossing_count` | 2.937 | 1.529 | 0.507 |
| `signalized_class_delta_junction_count` | 0.853 | 0.376 | 0.496 |
| `junction_count` | 13.072 | 10.165 | 0.472 |
| `public_transport_stop_count` | 3.242 | 2.144 | 0.432 |

Интерпретация: TP и FP сильно пересекаются по структуре. Модель умеет отделять "плотно и похоже на риск" от "пусто и безопасно", но ей сложнее отделить "в этой плотной ячейке ДТП реально случится" от "эта ячейка просто похожа на аварийную".

### Карта / H3

По H3-агрегации видны системные зоны:

- есть ячейки с `fp_count = 36`: модель ставила высокий риск каждый тестовый месяц, но целевой ДТП-метки там не было ни разу;
- есть ячейки с `fn_count` до `6`: ДТП несколько раз были, но модель ни разу не поднимала риск выше threshold `0.5`;
- есть ячейки с `tp_count` до `19`: модель стабильно ставит высокий риск, и ДТП там действительно часто происходят.

Это стоит смотреть в viewer слоями `FP count`, `FN count`, `TP count`, потому что таблично видно только общий паттерн, а карта показывает, являются ли ошибки локальными зонами или просто рассыпаны по городу.

### Общая интерпретация

Главный вывод: ошибки модели не выглядят полностью случайными.

False positives в основном похожи на dense urban road cells. Это не "пустые" ячейки, а места, где по дорожной структуре риск действительно кажется высоким. Поэтому низкий precision частично является следствием постановки absolute risk: ДТП - редкое событие, а похожих на опасные дорожных ячеек гораздо больше, чем фактически зафиксированных ДТП в конкретном месяце.

False negatives показывают другой край: реальные ДТП в ячейках, где дорожная структура не настолько явно сигналит опасность. Для таких случаев текущих OSM-признаков может быть недостаточно. Вероятно, нужны признаки экспозиции и контекста: интенсивность движения, население/активность, POI, школы/ТЦ/остановочные узлы, качество покрытия, фактические скорости, ограничения видимости, погодные или сезонные факторы.

Практически это подтверждает предыдущую мысль: absolute risk baseline работает содержательно, но основная его сила - находить плотные городские дорожные зоны. Следующий полезный шаг внутри текущей постановки - детально разобрать FP/FN на карте и понять, какие кластеры ошибок объясняются отсутствующими признаками, а какие являются естественным шумом редкого события.

## 2026-07-01 - CatBoost Road-Only 4.0 with `road_length_total_m > 2000`

### Что сделано

Проверена гипотеза: сделать road-only фильтр жестче и обучать/оценивать модель только на очень насыщенных дорожных ячейках:

```text
road_length_total_m > 2000
```

База эксперимента та же, что у `CatBoost road-only 4.0`:

```text
feature set: road_without_type_lengths_no_calendar_slim_dedup
feature count: 60
p99 clipping: включен
calendar features: исключены
```

Артефакты:

```text
reports/catboost_feature_importance_road_only_4_0_min_2000.csv
reports/catboost_road_only_4_0_min_2000_p99_clip_caps.csv
```

### Размер выборки

Фильтр `>2000м` сильно меняет задачу:

| Filter | Rows | H3 cells | Train rows | Test rows | Test positives | Test positive rate |
|---|---:|---:|---:|---:|---:|---:|
| `road_length_total_m > 0` | 567,072 | 4,296 | 412,416 | 154,656 | 3,611 | 2.335% |
| `road_length_total_m > 2000` | 117,744 | 892 | 85,632 | 32,112 | 2,198 | 6.845% |

То есть модель `>2000м` работает уже не по всем дорожным ячейкам, а по верхнему хвосту дорожной насыщенности. Positive rate становится почти в 3 раза выше, но большая часть road-only ячеек исключается.

### Метрики

Сравнение:

1. основной `4.0`, обученный и оцененный на `>0`;
2. тот же основной `4.0`, но его predictions пересчитаны только на subset `>2000`;
3. новая модель, обученная и оцененная только на `>2000`.

| Metric | 4.0 `>0` full test | 4.0 `>0`, evaluated on `>2000` | 4.0 retrained on `>2000` |
|---|---:|---:|---:|
| Test rows | 154,656 | 32,112 | 32,112 |
| Test positives | 3,611 | 2,198 | 2,198 |
| Positive rate | 2.335% | 6.845% | 6.845% |
| ROC-AUC | 0.907661 | 0.823294 | 0.820683 |
| PR-AUC | 0.197979 | 0.226018 | 0.223919 |
| Accuracy@0.5 | 0.8201 | 0.5551 | 0.7006 |
| Precision@0.5, class 1 | 0.1007 | 0.1260 | 0.1612 |
| Recall@0.5, class 1 | 0.8449 | 0.9268 | 0.8025 |
| F1@0.5, class 1 | 0.1799 | 0.2219 | 0.2685 |
| Top 1% precision | 0.3034 | 0.3333 | 0.3458 |
| Top 3% precision | 0.2432 | 0.3022 | 0.3198 |
| Top 5% precision | 0.2068 | 0.2910 | 0.2903 |
| Top 10% precision | 0.1518 | 0.2551 | 0.2476 |

Confusion matrix для модели, обученной на `>2000`, при threshold `0.5`:

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 20,734 | 9,180 |
| Actual 1 | 434 | 1,764 |

Top-k для модели, обученной на `>2000`:

| Slice | k | Precision | Recall | Lift | TP / positives |
|---|---:|---:|---:|---:|---:|
| Top 1% | 321 | 0.3458 | 0.0505 | 5.05 | 111 / 2,198 |
| Top 3% | 963 | 0.3198 | 0.1401 | 4.67 | 308 / 2,198 |
| Top 5% | 1,605 | 0.2903 | 0.2120 | 4.24 | 466 / 2,198 |
| Top 10% | 3,211 | 0.2476 | 0.3617 | 3.62 | 795 / 2,198 |

### Интерпретация

Фильтр `road_length_total_m > 2000` делает выборку более "городской" и менее дисбалансной: positive rate растет с `2.335%` до `6.845%`.

Из-за этого threshold-метрики выглядят лучше:

- `Precision@0.5`: `0.1007` -> `0.1612`;
- `F1@0.5`: `0.1799` -> `0.2685`;
- false positives становятся менее доминирующими относительно positives.

Но это не означает, что модель стала содержательно сильнее. Часть улучшения возникает потому, что мы заранее оставили только плотные дорожные ячейки, где ДТП в принципе чаще.

Более честное сравнение внутри одного и того же subset `>2000` показывает:

- старая `4.0`, обученная на `>0`, дает ROC-AUC `0.823294`;
- новая модель, обученная только на `>2000`, дает ROC-AUC `0.820683`;
- PR-AUC тоже чуть ниже у retrained `>2000`: `0.223919` против `0.226018`;
- Top 1-3% precision у retrained `>2000` чуть выше;
- Top 5-10% precision чуть ниже или примерно такой же.

Вывод: переобучение только на `>2000м` не дает явного прироста. Оно немного меняет trade-off: становится менее агрессивным на threshold `0.5`, чуть улучшает самый верх рейтинга (`Top 1-3%`), но не улучшает общее ранжирование.

Практический вывод:

- `>2000м` полезен как диагностический срез плотных городских ячеек;
- как основной baseline absolute risk его брать не стоит, потому что он выкидывает большую часть дорожной сети;
- для анализа ошибок и "плотного города" такой фильтр может быть полезен;
- текущий основной baseline `4.0 >0` остается более универсальным.

Топ признаков модели `>2000`:

```text
asphalt_surface_length_m
road_curvature_mean_weighted
asphalt_surface_share
lane_meters_estimate
road_curvature_max
road_share_residential
junction_density_per_km
dead_end_density_per_km
road_orientation_entropy
road_share_service
public_transport_stop_count
class_delta_mean
road_segment_length_median
road_way_count
class_delta_max
road_length_total_m
curvature_deg_per_km
maxspeed_mean_weighted
bad_surface_share
junction_degree_mean
```

## 2026-06-29 - Positive vs Negative Feature Analysis for Road-Only 4.0/4.1

### Что сделано

Проведен отдельный анализ признаков через сравнение строк:

```text
target = 1: в H3-ячейке будет ДТП в следующем месяце
target = 0: в H3-ячейке не будет ДТП в следующем месяце
```

Анализ сделан для двух текущих вариантов absolute-risk модели:

```text
CatBoost road-only 4.0
CatBoost road-only 4.1 log
```

Для каждого признака посчитаны:

- `mean_pos`, `mean_neg`, `mean_ratio_pos_to_neg`;
- `median_pos`, `median_neg`, `median_ratio_pos_to_neg`;
- `p95_pos`, `p95_neg`, `p95_ratio_pos_to_neg`;
- `zero_share_pos`, `zero_share_neg`;
- `KS statistic`;
- `Mann-Whitney p-value`;
- `single-feature ROC-AUC`;
- `standardized mean difference`.

Анализ считается на train-части road-only строк:

```text
month < 2023-01-01
road_length_total_m > 0
```

Данные:

| Показатель | Значение |
|---|---:|
| Train rows | 412,416 |
| Positive rows | 10,661 |
| Positive rate | 2.585% |
| Feature count | 60 |
| Log1p features in 4.1 | 33 |

Созданы notebooks:

```text
notebooks/catboost_road_only_4_0_positive_negative_analysis.ipynb
notebooks/catboost_road_only_4_1_log_positive_negative_analysis.ipynb
```

Воспроизводимый скрипт:

```text
scripts/positive_negative_feature_analysis.py
```

Артефакты:

```text
reports/catboost_road_only_4_0_positive_negative_feature_comparison.csv
reports/catboost_road_only_4_1_log_positive_negative_feature_comparison.csv
reports/catboost_road_only_4_0_positive_negative_top_features.csv
reports/catboost_road_only_4_1_log_positive_negative_top_features.csv
reports/figures/catboost_road_only_4_0_positive_negative/
reports/figures/catboost_road_only_4_1_log_positive_negative/
```

### Основные результаты

Сильнейшие одиночные признаки почти полностью описывают не "опасность сама по себе", а плотность и сложность городской дорожной среды.

Топ признаков по `single-feature ROC-AUC` для `4.0`:

| Feature | ROC-AUC | KS | SMD | mean target=1 | mean target=0 | zero target=1 | zero target=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `asphalt_surface_length_m` | 0.865 | 0.587 | 1.438 | 959.8 | 258.5 | 3.1% | 52.0% |
| `lane_meters_estimate` | 0.854 | 0.565 | 1.532 | 3,005.2 | 1,386.3 | 0.0% | 0.0% |
| `class_delta_max` | 0.834 | 0.573 | 1.453 | 4.45 | 1.89 | 4.2% | 42.4% |
| `unsignalized_high_delta_junction_count` | 0.825 | 0.515 | 1.269 | 8.13 | 2.68 | 5.2% | 44.6% |
| `junction_count` | 0.823 | 0.511 | 1.265 | 11.97 | 4.30 | 3.3% | 34.0% |
| `minor_to_major_junction_count` | 0.822 | 0.508 | 1.252 | 8.69 | 2.96 | 4.4% | 42.5% |
| `road_way_count` | 0.818 | 0.514 | 1.261 | 23.75 | 10.38 | 0.0% | 0.0% |
| `lanes_mean_weighted` | 0.814 | 0.555 | 1.382 | 2.31 | 0.65 | 17.1% | 71.9% |
| `road_length_total_m` | 0.794 | 0.454 | 1.171 | 2,171.1 | 1,220.9 | 0.0% | 0.0% |
| `junction_density_per_km` | 0.783 | 0.455 | 1.120 | 5.23 | 2.58 | 3.3% | 34.0% |

Интерпретация:

- в будущих ДТП-строках намного чаще есть асфальтированные дороги;
- больше суммарная длина дорог и примерная `lane_meters_estimate`;
- больше перекрестков, особенно пересечений разных классов дорог;
- больше признаков городской инфраструктуры: переходы, остановки, размеченные полосы, высокая связность сети;
- много признаков работают как proxy городской плотности.

### Важность CatBoost vs одиночная сила признака

Корреляция между `CatBoost feature importance` и `single-feature_auc_distance` получилась слабой:

| Variant | Pearson | Spearman |
|---|---:|---:|
| 4.0 | 0.057 | 0.081 |
| 4.1 log | 0.026 | 0.069 |

Это важный вывод: `feature importance` нельзя читать как "этот признак сам по себе лучше всего отделяет ДТП от не-ДТП". Часть признаков важна не как одиночный разделитель, а через взаимодействия.

Примеры признаков с высокой importance, но умеренной одиночной разделяющей силой:

| Feature | Importance 4.0 | ROC-AUC | KS | Интерпретация |
|---|---:|---:|---:|---|
| `road_share_residential` | 5.089 | 0.610 | 0.339 | важен как тип городской среды, но не самодостаточный сигнал |
| `road_curvature_mean_weighted` | 4.996 | 0.567 | 0.140 | геометрия работает условно, в связке с плотностью/типами дорог |
| `dead_end_density_per_km` | 3.617 | 0.580 | 0.230 | полезен как локальная топология, но сам по себе слабый |
| `road_share_unclassified` | 3.214 | 0.503 | 0.130 | почти не отделяет классы один, вероятно работает через взаимодействия |
| `road_share_service` | 2.857 | 0.465 | 0.248 | скорее отрицательный/условный сигнал, а не прямой риск |

### Признаки с обратным направлением

Некоторые признаки выше именно у `target=0`, а не у `target=1`.

Главные примеры:

| Feature | ROC-AUC | Oriented AUC | mean target=1 | mean target=0 | Интерпретация |
|---|---:|---:|---:|---:|---|
| `road_segment_length_median` | 0.313 | 0.687 | 24.2 | 43.7 | ДТП-ячейки чаще имеют короткую нарезку дорожной сети |
| `road_segment_length_mean` | 0.359 | 0.641 | 36.3 | 54.3 | длинные сегменты чаще характерны для менее плотных зон |
| `road_share_service` | 0.465 | 0.535 | 0.460 | 0.503 | высокая доля service-дорог сама по себе не повышает риск |

Это логично: короткие сегменты обычно означают более плотную городскую сетку, больше пересечений и больше конфликтных точек.

### Что показал Mann-Whitney

У `57 / 60` признаков `Mann-Whitney p-value < 0.001`.

Но это не значит, что почти все признаки одинаково полезны. При `412,416` train-строках даже слабые отличия становятся статистически значимыми. Поэтому p-value здесь надо читать как фильтр "различие вообще есть", а для практического смысла смотреть на:

- `single-feature ROC-AUC`;
- `KS statistic`;
- `standardized mean difference`;
- связь с CatBoost importance;
- интерпретируемость признака.

### Сравнение 4.0 и 4.1 log

`4.0` и `4.1 log` дают почти одинаковые `single-feature ROC-AUC`, `KS statistic` и `Mann-Whitney`.

Это ожидаемо: `log1p` является монотонным преобразованием. Оно меняет масштаб признака, но почти не меняет порядок объектов по этому признаку. Поэтому одиночная разделяющая способность признаков остается той же.

Что меняется в `4.1`:

- `standardized mean difference` для части признаков меняется из-за смены масштаба;
- CatBoost немного иначе распределяет importance;
- `dead_end_density_per_km` поднялся по importance;
- `road_curvature_mean_weighted` стал выше `road_share_residential`;
- ranking-метрики самой модели при этом не улучшились.

Вывод по `4.1 log` не меняется: это полезный проверенный вариант, но не основной baseline. Он не добавил новой информации, а только изменил шкалу части признаков.

### Общая интерпретация этапа

Анализ `target=1` vs `target=0` подтверждает главный вывод по absolute risk: модель в первую очередь видит не "аномальную опасность", а интенсивность и сложность городской дорожной среды.

Сильные одиночные признаки:

- длина асфальтированных дорог;
- примерная полосо-длина;
- количество дорог;
- количество и плотность перекрестков;
- пересечения дорог разных классов;
- остановки, переходы и другие признаки городской инфраструктуры.

Более тонкие признаки вроде кривизны, тупиков, долей типов дорог и entropy тоже важны, но чаще не как одиночный сигнал. Они помогают CatBoost уточнять риск внутри уже плотной городской среды.

Практический вывод:

- признаки не выглядят случайными: многие реально отличаются между `target=1` и `target=0`;
- при этом основная ось различия снова сводится к urban intensity / road network density;
- для absolute-risk baseline это нормально и ожидаемо;
- если нужна более содержательная задача, следующий шаг все равно должен быть связан с анализом ошибок, преобразованием признаков или постановкой relative/excess risk.

## 2026-06-29 - CatBoost Road-Only 4.1 Log1p Candidates

### Что сделано

Проверена гипотеза, что часть хвостатых числовых признаков лучше подавать в модель через `log1p(x)`.

База эксперимента:

```text
CatBoost road-only 4.0
feature set: road_without_type_lengths_no_calendar_slim_dedup
feature count: 60
p99 clipping: включен
```

Для `4.1` использован тот же набор из 60 признаков, но 33 кандидата были преобразованы через:

```text
log1p(x) = log(1 + x)
```

Кандидаты выбирались по EDA:

- признаки неотрицательные;
- это длины, counts, densities, total/meters-признаки;
- у признака есть правый хвост или много нулей при заметном разбросе ненулевых значений;
- `share`, `entropy`, `mean speed`, `class_delta_mean/max`, `road_curvature_mean/max` не логарифмировались.

Список кандидатов:

```text
road_length_total_m
road_way_count
lit_road_length_m
bridge_length_m
asphalt_surface_length_m
lane_meters_estimate
traffic_signal_count
crossing_count
junction_density_per_km
minor_to_major_junction_density_per_km
junction_count
three_way_junction_count
four_plus_way_junction_count
major_junction_count
minor_to_major_junction_count
legacy_major_minor_junction_count
signalized_class_delta_junction_count
unsignalized_high_delta_junction_count
class_delta_junction_density_per_km
public_transport_stop_count
priority_sign_density_per_km
traffic_calming_density_per_km
railway_crossing_density_per_km
sharp_turn_count
road_segment_count
road_segment_length_mean
road_segment_length_median
dead_end_count
curvature_deg_per_km
turn_event_density_per_km
sharp_turn_density_per_km
road_segment_density_per_km
dead_end_density_per_km
```

Артефакты:

```text
reports/catboost_road_only_4_1_log1p_candidate_analysis.csv
reports/catboost_road_only_4_1_log1p_candidates.csv
reports/catboost_road_only_4_1_log_p99_clip_caps.csv
reports/catboost_feature_importance_road_only_4_1_log.csv
parquet_output/h3_month_catboost_road_only_4_1_log_predictions.parquet
```

### Метрики

Сравнение с `4.0 dedup + p99 clipping`:

| Metric | Road-only 4.0 | Road-only 4.1 log |
|---|---:|---:|
| Feature count | 60 | 60 |
| Log1p transformed features | 0 | 33 |
| ROC-AUC | 0.907661 | 0.907785 |
| PR-AUC | 0.197979 | 0.197511 |
| Accuracy@0.5 | 0.8201 | 0.8213 |
| Precision@0.5, class 1 | 0.1007 | 0.1012 |
| Recall@0.5, class 1 | 0.8449 | 0.8446 |
| F1@0.5, class 1 | 0.1799 | 0.1808 |
| Top 1% precision | 0.3034 | 0.2891 |
| Top 3% precision | 0.2432 | 0.2444 |
| Top 5% precision | 0.2068 | 0.2056 |
| Top 10% precision | 0.1518 | 0.1525 |

Confusion matrix для `4.1 log` при threshold `0.5`:

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 123,963 | 27,082 |
| Actual 1 | 561 | 3,050 |

Top-k для `4.1 log`:

| Slice | k | Precision | Recall | Lift | TP / positives |
|---|---:|---:|---:|---:|---:|
| Top 1% | 1,546 | 0.2891 | 0.1238 | 12.38 | 447 / 3,611 |
| Top 3% | 4,639 | 0.2444 | 0.3140 | 10.47 | 1,134 / 3,611 |
| Top 5% | 7,732 | 0.2056 | 0.4403 | 8.81 | 1,590 / 3,611 |
| Top 10% | 15,465 | 0.1525 | 0.6533 | 6.53 | 2,359 / 3,611 |

### Интерпретация

`log1p` не дал улучшения итоговой модели.

Что изменилось:

- ROC-AUC почти не изменился: `0.907661` -> `0.907785`;
- PR-AUC немного снизился: `0.197979` -> `0.197511`;
- Top 1% precision заметно снизился: `0.3034` -> `0.2891`;
- Top 5% precision чуть снизился: `0.2068` -> `0.2056`;
- threshold-метрики при `0.5` стали чуть аккуратнее по precision/accuracy, но это не главный критерий для ranking-задачи.

Вывод: для CatBoost в этой постановке массовое `log1p`-преобразование хвостатых признаков скорее не нужно. Деревья и так умеют резать исходные числовые признаки по порогам, а логарифмирование частично сгладило верхний хвост, который был полезен для top-1% ranking.

Практический вывод:

- `4.1 log` полезен как проверенный отрицательный эксперимент;
- для основного absolute-risk baseline лучше оставить `4.0 dedup + p99 clipping`;
- если дальше проверять `log1p`, то не массово по 33 признакам, а точечно: например только для самых экстремальных density-признаков вроде `dead_end_density_per_km`, `sharp_turn_density_per_km`, `curvature_deg_per_km`.

Топ признаков `4.1 log`:

```text
asphalt_surface_length_m
road_curvature_mean_weighted
road_share_residential
lane_meters_estimate
dead_end_density_per_km
road_segment_length_median
road_share_service
road_orientation_entropy
road_curvature_max
road_share_unclassified
road_segment_density_per_km
turn_event_density_per_km
road_length_total_m
short_segment_share
class_delta_mean
```

## 2026-06-29 - CatBoost Road-Only 4.0 Dedup + P99 Clipping

### Что сделано

После EDA признаков проведена аккуратная чистка `CatBoost road-only 3.5`:

1. убраны явные дубли признаков;
2. проверен `p99 clipping` для хвостатых числовых признаков;
3. собран новый viewer layer `CatBoost road-only 4.0`.

Новый feature set:

```text
road_without_type_lengths_no_calendar_slim_dedup
```

Из 86 признаков baseline `3.5` осталось 60 признаков. Удалено 26 признаков:

```text
bad_surface_length_m
bridge_share
bus_stop_count
bus_stop_share
class_delta_junction_count
class_delta_sum
crossing_density_per_km
crossing_unsignalized_count
crossing_unsignalized_density_per_km
crossing_unsignalized_share
lanes_max
lit_share
major_road_length_m
maxspeed_is_imputed_share
oneway_length_m
public_transport_stop_density_per_km
railway_crossing_count
road_segment_h3_count
short_segment_count
traffic_calming_count
traffic_signal_density_per_km
tram_stop_count
tram_stop_share
turn_event_count
unsignalized_class_delta_junction_count
unsignalized_class_delta_junction_density_per_km
```

Логика удаления: вырезались признаки, которые по EDA выглядели как точные или почти точные дубли соседних признаков: `count/density`, `length/share`, технические алиасы, а также пары с очень высокой Spearman correlation.

Для clipping:

- p99 caps считались только на train;
- затем те же caps применялись к train/test и к parquet со score для viewer;
- caps сохранены в `reports/catboost_road_only_4_0_p99_clip_caps.csv`;
- для prediction parquet caps сохранены в `reports/catboost_road_only_4_0_prediction_p99_clip_caps.csv`;
- clipping применился к 46 признакам.

Новый prediction layer:

```text
parquet_output/h3_month_catboost_road_only_4_0_predictions.parquet
```

Feature importance:

```text
reports/catboost_feature_importance_road_only_4_0_no_calendar_dedup_clip_p99.csv
reports/catboost_feature_importance_road_only_4_0_dedup_no_clip.csv
```

### Метрики

Сравнение с основным baseline `3.5`:

| Metric | Road-only 3.5 | 4.0 dedup only | 4.0 dedup + p99 clipping |
|---|---:|---:|---:|
| Feature count | 86 | 60 | 60 |
| ROC-AUC | 0.908218 | 0.907839 | 0.907661 |
| PR-AUC | 0.198176 | 0.197867 | 0.197979 |
| Accuracy@0.5 | 0.820 | 0.8197 | 0.8201 |
| Precision@0.5, class 1 | 0.1005 | 0.1006 | 0.1007 |
| Recall@0.5, class 1 | 0.8438 | 0.8463 | 0.8449 |
| F1@0.5, class 1 | 0.180 | 0.1798 | 0.1799 |
| Top 1% precision | 0.3066 | 0.2937 | 0.3034 |
| Top 3% precision | - | 0.2457 | 0.2432 |
| Top 5% precision | 0.2059 | 0.2050 | 0.2068 |
| Top 10% precision | - | 0.1521 | 0.1518 |

Confusion matrix для `4.0 dedup + p99 clipping` при threshold `0.5`:

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 123,784 | 27,261 |
| Actual 1 | 560 | 3,051 |

Top-k для `4.0 dedup + p99 clipping`:

| Slice | k | Precision | Recall | Lift | TP / positives |
|---|---:|---:|---:|---:|---:|
| Top 1% | 1,546 | 0.3034 | 0.1299 | 12.99 | 469 / 3,611 |
| Top 3% | 4,639 | 0.2432 | 0.3124 | 10.41 | 1,128 / 3,611 |
| Top 5% | 7,732 | 0.2068 | 0.4428 | 8.86 | 1,599 / 3,611 |
| Top 10% | 15,465 | 0.1518 | 0.6502 | 6.50 | 2,348 / 3,611 |

### Интерпретация

Удаление явных дублей почти не ухудшило модель: feature count снизился с 86 до 60, а ROC-AUC/PR-AUC остались практически на том же уровне. Это хороший результат: часть признаков действительно была избыточной.

`p99 clipping` не дал сильного прироста, но немного стабилизировал ranking относительно `dedup only`:

- `PR-AUC`: `0.197867` -> `0.197979`;
- `Top 1% precision`: `0.2937` -> `0.3034`;
- `Top 5% precision`: `0.2050` -> `0.2068`.

По сравнению с `3.5` итоговая `4.0` почти не меняет общее качество:

- ROC-AUC чуть ниже: `0.908218` -> `0.907661`;
- PR-AUC почти тот же: `0.198176` -> `0.197979`;
- Top 5% precision чуть выше: `0.2059` -> `0.2068`;
- Top 1% precision чуть ниже: `0.3066` -> `0.3034`.

Вывод: `road-only 4.0` лучше как более чистая и интерпретируемая версия baseline: признаков меньше, явных дублей меньше, хвосты числовых признаков ограничены, а качество практически не просело.

Топ признаков `4.0 dedup + p99 clipping`:

```text
asphalt_surface_length_m
road_share_residential
road_curvature_mean_weighted
lane_meters_estimate
road_segment_length_median
dead_end_density_per_km
road_orientation_entropy
road_length_total_m
road_share_unclassified
short_segment_share
turn_event_density_per_km
road_share_service
curvature_deg_per_km
road_curvature_max
road_segment_density_per_km
```

## 2026-06-29 - CatBoost Road-Only 3.5 Class Imbalance Experiments

### Что сделано

Проведен отдельный диагностический этап по дисбалансу классов для финального baseline абсолютного риска `CatBoost road-only 3.5`.

Цель этапа:

```text
понять, являются ли низкий precision и большое число false positives следствием auto_class_weights="Balanced",
или это в большей степени свойство самой постановки absolute risk
```

Для этого создан notebook:

```text
notebooks/catboost_road_only_3_5_imbalance_experiments.ipynb
```

В notebook проверены стратегии:

```text
auto_class_weights="Balanced"
без весов
scale_pos_weight = 1, 3, 5, 10, 20, 40
undersampling negatives: 1:5, 1:10, 1:20
Platt calibration
isotonic calibration
threshold sweep от 0.05 до 0.90
```

Результаты сохранены:

```text
reports/catboost_road_only_3_5_imbalance_summary.csv
reports/catboost_road_only_3_5_imbalance_thresholds.csv
reports/catboost_road_only_3_5_imbalance_calibration.csv
```

Все эксперименты используют тот же feature set, что и финальная модель абсолютного риска:

```text
road_without_type_lengths_no_calendar_slim
```

Данные:

| Показатель | Значение |
|---|---:|
| Train rows | 412,416 |
| Test rows | 154,656 |
| Test positive rate | 2.3349% |
| Feature count | 86 |
| Target | `target_had_accident_next_month` |

### Сводные результаты

Ключевые варианты:

| Experiment | ROC-AUC | PR-AUC | Brier | Precision@0.5 | Recall@0.5 | FP@0.5 | FN@0.5 | Top 1% precision | Top 5% precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `balanced` | 0.908218 | 0.198176 | 0.122445 | 0.1005 | 0.8438 | 27,265 | 564 | 0.3066 | 0.2059 |
| `scale_pos_weight_20` | 0.908703 | 0.198478 | 0.085589 | 0.1246 | 0.7724 | 19,603 | 822 | 0.3092 | 0.2059 |
| `scale_pos_weight_10` | 0.909060 | 0.197053 | 0.055074 | 0.1594 | 0.6181 | 11,772 | 1,379 | 0.3118 | 0.2047 |
| `scale_pos_weight_5` | 0.909415 | 0.196762 | 0.035211 | 0.2143 | 0.4101 | 5,431 | 2,130 | 0.3118 | 0.2056 |
| `no_weights` | 0.909704 | 0.196999 | 0.020410 | 0.3194 | 0.0064 | 49 | 3,588 | 0.3124 | 0.2059 |
| `undersample_1_to_20` | 0.909539 | 0.195790 | 0.022530 | 0.3151 | 0.1194 | 937 | 3,180 | 0.3072 | 0.2064 |
| `balanced_platt` | 0.906261 | 0.197655 | 0.020405 | 0.0000 | 0.0000 | 0 | 3,611 | 0.2956 | 0.2067 |
| `balanced_isotonic` | 0.906132 | 0.188615 | 0.020320 | 0.0000 | 0.0000 | 0 | 3,611 | 0.2962 | 0.2054 |

### Что происходит при threshold tuning

Порог `0.5` оказался не универсальным и не естественным для этой задачи.

Для `balanced` при росте threshold:

| Threshold | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|
| 0.50 | 0.1005 | 0.8438 | 27,265 | 564 |
| 0.70 | 0.1349 | 0.7341 | 17,005 | 960 |
| 0.80 | 0.1630 | 0.5982 | 11,088 | 1,451 |
| 0.90 | 0.2431 | 0.3295 | 3,706 | 2,421 |

То есть threshold можно использовать как ручку управления trade-off:

```text
ниже threshold -> больше пойманных ДТП, но больше false positives
выше threshold -> меньше false positives, но больше пропущенных ДТП
```

Однако изменение threshold не меняет порядок ячеек по risk score. Оно только выбирает, где провести бинарную границу.

### Calibration

Калибровка резко улучшила Brier score:

| Experiment | Brier |
|---|---:|
| `balanced` | 0.122445 |
| `balanced_platt` | 0.020405 |
| `balanced_isotonic` | 0.020320 |
| `no_weights` | 0.020410 |

Вывод: score модели с `auto_class_weights="Balanced"` нельзя интерпретировать как честную вероятность ДТП. Это скорее ranking score.

После Platt/isotonic calibration score становится ближе к вероятности, но threshold `0.5` становится бесполезным: реальные вероятности ДТП низкие, поэтому calibrated-модель почти никому не ставит score выше `0.5`.

Для calibrated score надо использовать низкие пороги вроде `0.05`, `0.10`, `0.15`, либо top-k/top-percentile.

Проверка threshold для calibrated-моделей:

| Experiment | Threshold | Precision | Recall | FP | FN | Комментарий |
|---|---:|---:|---:|---:|---:|---|
| `balanced_platt` | 0.05 | 0.1204 | 0.7851 | 20,709 | 776 | высокий охват ДТП, но очень много ложных срабатываний |
| `balanced_platt` | 0.10 | 0.1490 | 0.6583 | 13,571 | 1,234 | более умеренный вариант, но false positives все еще много |
| `balanced_platt` | 0.15 | 0.1917 | 0.4893 | 7,449 | 1,844 | лучший F1 для Platt calibration |
| `balanced_platt` | 0.20 | 0.2514 | 0.2983 | 3,207 | 2,534 | меньше шума, но пропускается большая часть ДТП |
| `balanced_isotonic` | 0.10 | 0.1541 | 0.6361 | 12,607 | 1,314 | близко к Platt calibration |
| `balanced_isotonic` | 0.15 | 0.2089 | 0.4290 | 5,867 | 2,062 | лучший F1 для isotonic calibration |
| `balanced_isotonic` | 0.20 | 0.2495 | 0.3060 | 3,323 | 2,506 | более строгий, но с низким recall |

По F1 calibrated-модели не дают принципиального выигрыша:

| Experiment | Best F1 threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `balanced` | 0.90 | 0.2431 | 0.3295 | 0.2798 |
| `balanced_platt` | 0.15 | 0.1917 | 0.4893 | 0.2755 |
| `balanced_isotonic` | 0.15 | 0.2089 | 0.4290 | 0.2809 |
| `scale_pos_weight_10` | 0.65 | 0.2112 | 0.4317 | 0.2837 |

Интерпретация: калибровка полезна, если нужен score, похожий на вероятность ДТП в ячейке в следующем месяце. Но она почти не улучшает разделение классов и не решает проблему большого числа false positives. Иными словами, calibration меняет смысл шкалы score, но не создает новую информацию для модели.

### Интерпретация

Эксперименты с дисбалансом показали две разные вещи.

Во-первых, class weights действительно сильно влияют на бинарную классификацию при фиксированном пороге.

Например:

- `balanced` дает высокий recall `0.8438`, но очень много false positives: `27,265`;
- `scale_pos_weight_20` снижает false positives до `19,603`, сохраняя recall `0.7724`;
- `scale_pos_weight_10` снижает false positives до `11,772`, но recall падает до `0.6181`;
- `no_weights` почти не ловит positive class при threshold `0.5`: recall `0.0064`.

То есть текущий режим `auto_class_weights="Balanced"` действительно делает модель агрессивной: она лучше ловит ДТП, но сильно шумит.

Во-вторых, top-k ranking почти не меняется:

| Experiment | Top 1% precision | Top 5% precision |
|---|---:|---:|
| `balanced` | 0.3066 | 0.2059 |
| `scale_pos_weight_20` | 0.3092 | 0.2059 |
| `scale_pos_weight_10` | 0.3118 | 0.2047 |
| `scale_pos_weight_5` | 0.3118 | 0.2056 |
| `no_weights` | 0.3124 | 0.2059 |

Это главный результат этапа.

Разные стратегии работы с дисбалансом сильно меняют:

- precision/recall при фиксированном threshold;
- количество false positives;
- количество false negatives;
- масштаб score.

Но они почти не меняют:

- качество верхнего risk ranking;
- концентрацию ДТП в top 1%;
- концентрацию ДТП в top 5%.

### Вывод

Проблема не сводится к неудачному threshold `0.5` или конкретно к `auto_class_weights="Balanced"`.

`Balanced` действительно завышает агрессивность бинарных предсказаний, но модель устойчиво ранжирует примерно одни и те же дорожные зоны как наиболее рискованные. Это значит, что содержательное ограничение остается тем же: в постановке absolute risk модель в первую очередь выделяет зоны с высокой городской и дорожной активностью.

Практический вывод:

- для ranking-задачи `CatBoost road-only 3.5` остается устойчивым baseline;
- `score >= 0.5` не стоит использовать как основной критерий "опасно / не опасно";
- если нужен более умеренный binary classifier, лучше смотреть на `scale_pos_weight_10` или `scale_pos_weight_20` и подбирать threshold;
- если score нужно трактовать как вероятность, нужна calibration;
- эксперименты с дисбалансом подтверждают, что следующий содержательный шаг не в threshold tuning, а в анализе признаков, ошибок и, позже, в переходе к relative/excess risk.

## 2026-06-10 - CatBoost Road-Only 3.5 Slim Absolute-Risk Baseline

### Что сделано

После просмотра хвоста importance у `road-only 3.0` из основного absolute-risk baseline удалены признаки, которые в текущих данных либо:

- имеют нулевой или почти нулевой вклад;
- являются константами или почти константами для региона;
- выглядят как слабые дубли более сильных соседних признаков.

В код добавлен отдельный воспроизводимый feature set:

```text
road_without_type_lengths_no_calendar_slim
```

Из обучения вырезаны:

```text
road_share_motorway
road_share_motorway_link
mini_roundabout_count
mini_roundabout_share
mini_roundabout_density_per_km
roundabout_total_count
roundabout_count
roundabout_density_per_km
roundabout_total_density_per_km
full_roundabout_share
priority_sign_count
give_way_share
motorway_junction_count
stop_count
stop_density_per_km
stop_share
give_way_count
give_way_density_per_km
tunnel_length_m
tunnel_share
link_road_length_m
link_road_share
road_share_primary_link
road_share_secondary_link
road_share_tertiary_link
road_share_trunk_link
crossing_signalized_count
crossing_signalized_share
crossing_signalized_density_per_km
```

Дополнительно собран отдельный viewer layer:

```text
parquet_output/h3_month_catboost_road_only_3_5_predictions.parquet
```

Feature importance:

```text
reports/catboost_feature_importance_road_only_3_5_no_calendar.csv
```

### Данные и признаки

| Показатель | Значение |
|---|---:|
| Rows после фильтра `road_length_total_m > 0` | 567,072 |
| H3 в road-only 3.5 viewer layer | 4,296 |
| Columns в parquet | 134 |
| Feature count | 86 |
| Было в road-only 3.0 | 115 |
| Удалено из baseline | 29 |
| Train rows | 412,416 |
| Test rows | 154,656 |
| Train positive rate | 2.5850% |
| Test positive rate | 2.3349% |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.908218 |
| PR-AUC | 0.198176 |
| Accuracy | 0.8201 |
| Precision, класс `1` | 0.1005 |
| Recall, класс `1` | 0.8438 |
| F1-score, класс `1` | 0.1796 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 123,780 | 27,265 |
| Actual 1 | 564 | 3,047 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 1,546 | 0.3066 | 0.1313 | 13.13 | 474 / 3,611 |
| Top 3% | 4,639 | 0.2460 | 0.3160 | 10.53 | 1,141 / 3,611 |
| Top 5% | 7,732 | 0.2059 | 0.4409 | 8.82 | 1,592 / 3,611 |
| Top 10% | 15,465 | 0.1528 | 0.6544 | 6.54 | 2,363 / 3,611 |

### Сравнение с road-only 3.0

| Метрика | Road-only 3.0 | Road-only 3.5 | Изменение |
|---|---:|---:|---:|
| Feature count | 115 | 86 | -29 |
| ROC-AUC | 0.908145 | 0.908218 | +0.000073 |
| PR-AUC | 0.198121 | 0.198176 | +0.000055 |
| Accuracy | 0.8197 | 0.8201 | +0.0004 |
| Precision, класс `1` | 0.1005 | 0.1005 | +0.0000 |
| Recall, класс `1` | 0.8455 | 0.8438 | -0.0017 |
| F1-score, класс `1` | 0.1796 | 0.1796 | +0.0000 |
| Top 1% precision | 0.3060 | 0.3066 | +0.0006 |
| Top 1% recall | 0.1310 | 0.1313 | +0.0003 |
| Top 3% precision | 0.2470 | 0.2460 | -0.0010 |
| Top 3% recall | 0.3174 | 0.3160 | -0.0014 |
| Top 5% precision | 0.2067 | 0.2059 | -0.0008 |
| Top 5% recall | 0.4425 | 0.4409 | -0.0016 |
| Top 10% precision | 0.1529 | 0.1528 | -0.0001 |
| Top 10% recall | 0.6549 | 0.6544 | -0.0005 |

### На что модель теперь опирается

Top-20 признаков по встроенной важности CatBoost:

| Признак | Importance |
|---|---:|
| `asphalt_surface_length_m` | 7.044883 |
| `road_share_residential` | 4.555123 |
| `dead_end_density_per_km` | 4.115389 |
| `road_curvature_mean_weighted` | 3.787710 |
| `road_segment_length_median` | 3.671419 |
| `lane_meters_estimate` | 3.619972 |
| `road_share_unclassified` | 3.416933 |
| `road_orientation_entropy` | 3.246253 |
| `road_share_service` | 2.911895 |
| `road_segment_density_per_km` | 2.848555 |
| `road_curvature_max` | 2.696781 |
| `turn_event_density_per_km` | 2.472281 |
| `road_length_total_m` | 2.327385 |
| `class_delta_max` | 2.250703 |
| `road_segment_length_mean` | 2.115298 |
| `maxspeed_mean_weighted` | 2.113359 |
| `sharp_turn_share` | 1.974212 |
| `short_segment_share` | 1.886005 |
| `junction_density_per_km` | 1.767289 |
| `class_delta_junction_density_per_km` | 1.671659 |

### Интерпретация

Это хорошая чистка.

Удаление `29` слабых признаков не ухудшило модель. Наоборот:

- `ROC-AUC` и `PR-AUC` чуть выросли;
- `Top 1%` тоже слегка улучшился;
- остальной `top-k` практически не изменился.

То есть длинный хвост редких и почти пустых признаков в основном не добавлял полезного сигнала, а скорее раздувал baseline.

Практически это означает:

- `road-only 3.5` лучше подходит как основной **absolute-risk baseline**;
- модель стала компактнее и проще для сопровождения;
- importance теперь чище отражает реальное ядро сигнала.

После pruning основной сигнал окончательно сосредоточился в четырех блоках:

- geometry / network structure;
- типы дорог;
- capacity / lane proxy;
- покрытие и качество дорожной среды.

Особенно важно, что geometry-признаки не исчезли после чистки, а остались в топе:

- `dead_end_density_per_km`
- `road_curvature_mean_weighted`
- `road_segment_length_median`
- `road_orientation_entropy`
- `road_curvature_max`

Это хороший признак: модель опирается не на длинный мусорный хвост, а на структурно понятные дорожные сигналы.

Практический вывод:

- для абсолютного риска `road-only 3.5` сейчас выглядит как лучший и самый аккуратный CatBoost baseline;
- вырезанные признаки стоит оставить в parquet и viewer для анализа, но не обязательно держать в основном baseline;
- следующий шаг уже не в том, чтобы еще сильнее чистить хвост, а в том, чтобы отделять absolute risk от excess risk отдельной постановкой задачи.

### Статус

`CatBoost road-only 3.5` принят как основной baseline для задачи прогнозирования **абсолютного риска**.

В viewer добавлен отдельный слой:

```text
CatBoost road-only 3.5
```

## 2026-06-09 - CatBoost Road-Only 3.0 With Group 3 Geometry Features

### Что сделано

В road-only 2.5 модель добавлены признаки из группы 3, которые должны лучше различать "просто плотный центр" и реальную геометрическую сложность дорожной сети:

```text
curvature_deg_per_km
sharp_turn_density_per_km
road_curvature_mean_weighted
road_curvature_max
road_orientation_entropy
road_segment_length_median
short_segment_share
dead_end_density_per_km
```

Технически при reconcile были также посчитаны промежуточные и вспомогательные geometry-агрегаты, которые остались в parquet и тоже попали в feature set:

```text
turn_event_count
sharp_turn_count
road_segment_count
road_segment_length_mean
short_segment_count
dead_end_count
turn_event_density_per_km
sharp_turn_share
road_segment_density_per_km
```

Обновлены:

```text
parquet_output/h3_road_features.parquet
parquet_output/h3_month_road_features.parquet
```

Собран отдельный viewer layer:

```text
parquet_output/h3_month_catboost_road_only_3_0_predictions.parquet
```

Feature importance:

```text
reports/catboost_feature_importance_road_only_3_0_no_calendar.csv
```

### Данные и признаки

| Показатель | Значение |
|---|---:|
| Rows после фильтра `road_length_total_m > 0` | 567,072 |
| H3 в road-only 3.0 viewer layer | 4,296 |
| Columns в parquet | 134 |
| Feature count | 115 |
| Train rows | 412,416 |
| Test rows | 154,656 |
| Train positive rate | 2.5850% |
| Test positive rate | 2.3349% |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.908145 |
| PR-AUC | 0.198121 |
| Accuracy | 0.8197 |
| Precision, класс `1` | 0.1005 |
| Recall, класс `1` | 0.8455 |
| F1-score, класс `1` | 0.1796 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 123,714 | 27,331 |
| Actual 1 | 558 | 3,053 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 1,546 | 0.3060 | 0.1310 | 13.10 | 473 / 3,611 |
| Top 3% | 4,639 | 0.2470 | 0.3174 | 10.58 | 1,146 / 3,611 |
| Top 5% | 7,732 | 0.2067 | 0.4425 | 8.85 | 1,598 / 3,611 |
| Top 10% | 15,465 | 0.1529 | 0.6549 | 6.55 | 2,365 / 3,611 |

### Сравнение с road-only 2.5

| Метрика | Road-only 2.5 | Road-only 3.0 | Изменение |
|---|---:|---:|---:|
| ROC-AUC | 0.907646 | 0.908145 | +0.000499 |
| PR-AUC | 0.196893 | 0.198121 | +0.001228 |
| Accuracy | 0.8235 | 0.8197 | -0.0038 |
| Precision, класс `1` | 0.1023 | 0.1005 | -0.0018 |
| Recall, класс `1` | 0.8433 | 0.8455 | +0.0022 |
| F1-score, класс `1` | 0.1824 | 0.1796 | -0.0028 |
| Top 1% precision | 0.3079 | 0.3060 | -0.0019 |
| Top 1% recall | 0.1318 | 0.1310 | -0.0008 |
| Top 3% precision | 0.2483 | 0.2470 | -0.0013 |
| Top 3% recall | 0.3190 | 0.3174 | -0.0016 |
| Top 5% precision | 0.2068 | 0.2067 | -0.0001 |
| Top 5% recall | 0.4428 | 0.4425 | -0.0003 |
| Top 10% precision | 0.1519 | 0.1529 | +0.0010 |
| Top 10% recall | 0.6505 | 0.6549 | +0.0044 |

### На что модель опирается

Top-20 признаков по встроенной важности CatBoost:

| Признак | Importance |
|---|---:|
| `asphalt_surface_length_m` | 7.572617 |
| `road_share_residential` | 5.347439 |
| `road_curvature_max` | 4.349820 |
| `dead_end_density_per_km` | 4.219866 |
| `road_curvature_mean_weighted` | 4.059841 |
| `road_segment_length_median` | 3.637872 |
| `lane_meters_estimate` | 3.195405 |
| `road_orientation_entropy` | 3.105398 |
| `road_share_unclassified` | 2.823850 |
| `curvature_deg_per_km` | 2.451299 |
| `short_segment_share` | 2.389202 |
| `road_segment_density_per_km` | 2.299487 |
| `turn_event_density_per_km` | 2.275866 |
| `road_segment_length_mean` | 2.152094 |
| `road_share_service` | 2.007390 |
| `road_length_total_m` | 2.002895 |
| `class_delta_max` | 1.846633 |
| `sharp_turn_share` | 1.827162 |
| `sharp_turn_density_per_km` | 1.819544 |
| `junction_density_per_km` | 1.793786 |

Целевые новые geometry-признаки по importance:

| Признак | Importance |
|---|---:|
| `road_curvature_max` | 4.349820 |
| `dead_end_density_per_km` | 4.219866 |
| `road_curvature_mean_weighted` | 4.059841 |
| `road_segment_length_median` | 3.637872 |
| `road_orientation_entropy` | 3.105398 |
| `curvature_deg_per_km` | 2.451299 |
| `short_segment_share` | 2.389202 |
| `sharp_turn_density_per_km` | 1.819544 |

### Интерпретация

Это первый road-only эксперимент, где признаки из группы 3 дали не только локальный косметический эффект, а заметный вклад в само ранжирование.

Главный сигнал:

- `PR-AUC` вырос с `0.196893` до `0.198121`;
- `ROC-AUC` тоже вырос;
- `Top 10%` стал лучше по precision и recall;
- `Top 1%` почти не изменился и даже чуть просел относительно `2.5`.

То есть geometry-признаки не сделали shortlist "самых красных" ячеек радикально лучше, но улучшили более широкий порядок внутри road-only выборки. Это хороший знак: модель стала чуть лучше различать дорожные ячейки по структуре, а не просто усилила уже существующий top-tail.

Самое важное здесь не только прирост метрик, а то, **какие именно признаки поднялись в importance**. Сразу несколько geometry-признаков вошли в верхнюю часть списка:

- `road_curvature_max`
- `dead_end_density_per_km`
- `road_curvature_mean_weighted`
- `road_segment_length_median`
- `road_orientation_entropy`
- `curvature_deg_per_km`

Это означает, что гипотеза была верной: модель действительно начала опираться не только на "сколько дорог и какого типа", но и на форму сети:

- насколько дорога ломаная или плавно изогнутая;
- насколько сеть регулярная или хаотичная по направлениям;
- насколько коротко нарезаны сегменты;
- насколько много тупиков и локальных окончаний.

При этом "весь центр красный" одним этим шагом не исчезнет. Причина простая: базовые сильные признаки вроде `asphalt_surface_length_m`, `road_share_residential`, `lane_meters_estimate`, `road_share_unclassified` и `road_length_total_m` никуда не делись. Они все еще несут большой proxy-сигнал городской плотности и насыщенности сети.

Практический вывод:

- `road-only 3.0` сейчас выглядит как **лучший road-only baseline по общему качеству ранжирования**;
- если целевая метрика - `PR-AUC` и более устойчивый ranking по всей дорожной выборке, `3.0` лучше `2.5`;
- если нужна максимально агрессивная точность именно в `top 1%`, `2.5` остается почти на том же уровне и даже немного сильнее;
- geometry-блок стоит сохранять и развивать дальше, потому что он уже дает не декоративный, а реальный структурный сигнал.

В viewer добавлен отдельный слой:

```text
CatBoost road-only 3.0
```

## 2026-06-09 - CatBoost Road-Only 2.5 With Consistent Group 2 Density/Share Features

### Что сделано

В road-only 2.0 модель добавлены дополнительные group-2 признаки там, где для них есть осмысленный шаблон `count + density + share`:

```text
traffic_signal_density_per_km
priority_sign_count
priority_sign_density_per_km
stop_density_per_km
give_way_density_per_km
stop_share
give_way_share
traffic_calming_density_per_km
mini_roundabout_density_per_km
roundabout_total_count
roundabout_total_density_per_km
mini_roundabout_share
full_roundabout_share
motorway_junction_density_per_km
crossing_signalized_density_per_km
```

Обновлены:

```text
parquet_output/h3_road_features.parquet
parquet_output/h3_month_road_features.parquet
```

Собран отдельный viewer layer:

```text
parquet_output/h3_month_catboost_road_only_2_5_predictions.parquet
```

Feature importance:

```text
reports/catboost_feature_importance_road_only_2_5_no_calendar.csv
```

### Данные и признаки

| Показатель | Значение |
|---|---:|
| Rows после фильтра `road_length_total_m > 0` | 567,072 |
| H3 в road-only 2.5 viewer layer | 4,296 |
| Columns в parquet | 117 |
| Feature count | 98 |
| Train rows | 412,416 |
| Test rows | 154,656 |
| Train positive rate | 2.5850% |
| Test positive rate | 2.3349% |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.907646 |
| PR-AUC | 0.196893 |
| Accuracy | 0.8235 |
| Precision, класс `1` | 0.1023 |
| Recall, класс `1` | 0.8433 |
| F1-score, класс `1` | 0.1824 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 124,318 | 26,727 |
| Actual 1 | 566 | 3,045 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 1,546 | 0.3079 | 0.1318 | 13.19 | 476 / 3,611 |
| Top 3% | 4,639 | 0.2483 | 0.3190 | 10.64 | 1,152 / 3,611 |
| Top 5% | 7,732 | 0.2068 | 0.4428 | 8.86 | 1,599 / 3,611 |
| Top 10% | 15,465 | 0.1519 | 0.6505 | 6.51 | 2,349 / 3,611 |

### Сравнение с road-only 2.0

| Метрика | Road-only 2.0 | Road-only 2.5 | Изменение |
|---|---:|---:|---:|
| ROC-AUC | 0.907886 | 0.907646 | -0.000240 |
| PR-AUC | 0.197587 | 0.196893 | -0.000694 |
| Accuracy | 0.8207 | 0.8235 | +0.0028 |
| Precision, класс `1` | 0.1012 | 0.1023 | +0.0011 |
| Recall, класс `1` | 0.8471 | 0.8433 | -0.0038 |
| F1-score, класс `1` | 0.1807 | 0.1824 | +0.0017 |
| Top 1% precision | 0.2956 | 0.3079 | +0.0123 |
| Top 1% recall | 0.1266 | 0.1318 | +0.0052 |
| Top 3% precision | 0.2477 | 0.2483 | +0.0006 |
| Top 3% recall | 0.3182 | 0.3190 | +0.0008 |
| Top 5% precision | 0.2063 | 0.2068 | +0.0005 |
| Top 5% recall | 0.4417 | 0.4428 | +0.0011 |
| Top 10% precision | 0.1523 | 0.1519 | -0.0004 |
| Top 10% recall | 0.6522 | 0.6505 | -0.0017 |

### На что модель опирается

Top-20 признаков по встроенной важности CatBoost:

| Признак | Importance |
|---|---:|
| `asphalt_surface_length_m` | 10.583793 |
| `lane_meters_estimate` | 6.623033 |
| `road_share_residential` | 6.219757 |
| `road_length_total_m` | 5.278291 |
| `maxspeed_mean_weighted` | 3.918426 |
| `junction_density_per_km` | 3.527215 |
| `road_share_unclassified` | 3.502778 |
| `asphalt_surface_share` | 3.177235 |
| `road_share_service` | 3.007574 |
| `road_way_count` | 2.799385 |
| `class_delta_max` | 2.597789 |
| `bad_surface_length_m` | 2.385776 |
| `road_segment_h3_count` | 2.341141 |
| `class_delta_mean` | 2.320951 |
| `lanes_mean_weighted` | 2.289622 |
| `class_delta_junction_density_per_km` | 2.207877 |
| `road_share_tertiary` | 2.145607 |
| `lanes_tagged_share` | 1.898843 |
| `major_road_length_m` | 1.867045 |
| `bad_surface_share` | 1.733281 |

Новые 2.5 признаки по importance:

| Признак | Importance |
|---|---:|
| `traffic_calming_density_per_km` | 0.459572 |
| `traffic_signal_density_per_km` | 0.183869 |
| `crossing_signalized_density_per_km` | 0.079117 |
| `give_way_density_per_km` | 0.037238 |
| `motorway_junction_density_per_km` | 0.031404 |
| `priority_sign_density_per_km` | 0.029856 |
| `crossing_signalized_share` | 0.024301 |

### Интерпретация

Road-only 2.5 не улучшил модель по общему качеству ранжирования: `PR-AUC` немного снизился относительно `2.0`, `ROC-AUC` тоже слегка ниже. Значит добавленные density/share признаки не дали чистого общего выигрыша.

Но при этом верхний хвост стал лучше: `Top 1% precision` вырос с `0.2956` до `0.3079`, `Top 1% recall` - с `0.1266` до `0.1318`. Для сценария, где нас интересует именно самый узкий список "самых подозрительных" дорожных ячеек, это полезный сдвиг.

Новые признаки оказывают умеренный, а не доминирующий вклад. Лучше всех из этого пакета зашли:

- `traffic_calming_density_per_km`
- `traffic_signal_density_per_km`
- `crossing_signalized_density_per_km`
- `give_way_density_per_km`
- `motorway_junction_density_per_km`

Это похоже на более тонкий слой локальной организации движения: регулирование, calming, signalized crossing и priority logic. Но эти признаки все еще намного слабее основных групп - surface, lanes, road type mix, road amount и junction graph.

Практический вывод:

- набор `2.5` стоит сохранить в parquet и viewer;
- если целевая метрика - `PR-AUC` на всем road-only тесте, то `2.0` пока лучше;
- если важнее именно `top 1%` shortlist, `2.5` выглядит интереснее;
- для заметного снижения "весь центр красный" вероятнее всего уже нужен следующий блок из группы 3: геометрия и связность сети.

В viewer добавлен отдельный слой:

```text
CatBoost road-only 2.5
```

## 2026-06-09 - CatBoost Road-Only 2.0 With Added Group 2 Infrastructure

### Что сделано

В road-only no-calendar модель добавлены новые признаки из группы 2:

```text
crossing_signalized_count
crossing_unsignalized_count
crossing_density_per_km
crossing_signalized_share
crossing_unsignalized_share
crossing_unsignalized_density_per_km
public_transport_stop_count
public_transport_stop_density_per_km
bus_stop_count
tram_stop_count
bus_stop_share
tram_stop_share
railway_crossing_count
railway_crossing_density_per_km
roundabout_count
roundabout_density_per_km
```

Источник - расширенный перескан `osm_road_point_nodes.parquet` из локального PBF и reconcile поверх существующих road parquet. Полный road graph и road lengths не пересобирались; обновлялись только group-2 point/infrastructure признаки.

Дополнительно собран отдельный viewer layer:

```text
parquet_output/h3_month_catboost_road_only_2_0_predictions.parquet
```

Feature importance:

```text
reports/catboost_feature_importance_road_only_2_0_no_calendar.csv
```

### Данные и признаки

| Показатель | Значение |
|---|---:|
| Rows после фильтра `road_length_total_m > 0` | 567,072 |
| H3 в road-only 2.0 viewer layer | 4,296 |
| Columns в parquet | 102 |
| Feature count | 83 |
| Train rows | 412,416 |
| Test rows | 154,656 |
| Train positive rate | 2.5850% |
| Test positive rate | 2.3349% |

Новые point/infrastructure totals после reconcile:

| Признак | Сумма по H3 |
|---|---:|
| `crossing_signalized_count` | 442 |
| `crossing_unsignalized_count` | 1,232 |
| `public_transport_stop_count` | 2,463 |
| `bus_stop_count` | 2,087 |
| `tram_stop_count` | 229 |
| `railway_crossing_count` | 437 |
| `roundabout_count` | 45 |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.907886 |
| PR-AUC | 0.197587 |
| Accuracy | 0.8207 |
| Precision, класс `1` | 0.1012 |
| Recall, класс `1` | 0.8471 |
| F1-score, класс `1` | 0.1807 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 123,864 | 27,181 |
| Actual 1 | 552 | 3,059 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 1,546 | 0.2956 | 0.1266 | 12.66 | 457 / 3,611 |
| Top 3% | 4,639 | 0.2477 | 0.3182 | 10.61 | 1,149 / 3,611 |
| Top 5% | 7,732 | 0.2063 | 0.4417 | 8.84 | 1,595 / 3,611 |
| Top 10% | 15,465 | 0.1523 | 0.6522 | 6.52 | 2,355 / 3,611 |

### Сравнение с прошлым road-only

Сравнение с экспериментом `CatBoost Road-Only Without Calendar Features`.

| Метрика | Road-only | Road-only 2.0 | Изменение |
|---|---:|---:|---:|
| ROC-AUC | 0.907575 | 0.907886 | +0.000311 |
| PR-AUC | 0.196892 | 0.197587 | +0.000695 |
| Accuracy | 0.8229 | 0.8207 | -0.0022 |
| Precision, класс `1` | 0.1020 | 0.1012 | -0.0008 |
| Recall, класс `1` | 0.8438 | 0.8471 | +0.0033 |
| F1-score, класс `1` | 0.1820 | 0.1807 | -0.0013 |
| Top 1% precision | 0.3001 | 0.2956 | -0.0045 |
| Top 1% recall | 0.1285 | 0.1266 | -0.0019 |
| Top 3% precision | 0.2479 | 0.2477 | -0.0002 |
| Top 3% recall | 0.3185 | 0.3182 | -0.0003 |
| Top 5% precision | 0.2065 | 0.2063 | -0.0002 |
| Top 5% recall | 0.4423 | 0.4417 | -0.0006 |
| Top 10% precision | 0.1512 | 0.1523 | +0.0011 |
| Top 10% recall | 0.6475 | 0.6522 | +0.0047 |

### На что модель опирается

Top-20 признаков по встроенной важности CatBoost:

| Признак | Importance |
|---|---:|
| `asphalt_surface_length_m` | 10.621772 |
| `lane_meters_estimate` | 6.493301 |
| `road_share_residential` | 6.281752 |
| `road_share_unclassified` | 5.139819 |
| `road_length_total_m` | 4.622093 |
| `junction_density_per_km` | 3.707032 |
| `road_share_service` | 3.676258 |
| `maxspeed_mean_weighted` | 3.354100 |
| `road_way_count` | 2.981514 |
| `class_delta_mean` | 2.958713 |
| `asphalt_surface_share` | 2.557931 |
| `class_delta_max` | 2.319591 |
| `bad_surface_length_m` | 2.281542 |
| `bad_surface_share` | 2.205793 |
| `junction_degree_mean` | 2.131923 |
| `lanes_tagged_share` | 1.995900 |
| `class_delta_sum` | 1.922367 |
| `road_share_tertiary` | 1.840504 |
| `lanes_mean_weighted` | 1.824917 |
| `major_road_length_m` | 1.824790 |

Новые признаки из группы 2 по importance:

| Признак | Importance |
|---|---:|
| `public_transport_stop_density_per_km` | 1.013052 |
| `railway_crossing_density_per_km` | 0.966888 |
| `public_transport_stop_count` | 0.881647 |
| `crossing_unsignalized_density_per_km` | 0.663072 |
| `bus_stop_count` | 0.532250 |
| `bus_stop_share` | 0.483744 |
| `crossing_density_per_km` | 0.446192 |

Группировка признаков по смыслу:

| Группа признаков | Importance | Доля |
|---|---:|---:|
| Junction / graph structure | 24.699 | 24.7% |
| Road type shares | 18.491 | 18.5% |
| Surface quality / coverage | 17.667 | 17.7% |
| Lanes / capacity | 11.553 | 11.6% |
| Road amount / density | 11.447 | 11.4% |
| Point infrastructure | 6.459 | 6.5% |
| Speed | 5.852 | 5.9% |
| Other road attributes | 3.832 | 3.8% |

### Интерпретация

Добавление новых group-2 point/infrastructure признаков дало небольшой, но реальный прирост по общему ранжированию: PR-AUC вырос с `0.196892` до `0.197587`, ROC-AUC - с `0.907575` до `0.907886`. Это не перелом качества, а аккуратный инкремент.

При этом top-1/3/5 почти не изменились и местами слегка просели. Значит новые признаки пока не помогают сильнее вытащить самый верхний хвост риска. Их вклад скорее в более тонкую перестановку внутри дорожных ячеек и в чуть лучшее общее разделение positives/negatives.

Из новых признаков лучше всего зашли:

- `public_transport_stop_density_per_km`
- `railway_crossing_density_per_km`
- `public_transport_stop_count`
- `crossing_unsignalized_density_per_km`
- `bus_stop_count`

Это выглядит логично: остановки ОТ и переезды добавляют сигнал человеческой и транспортной активности, а нерегулируемые переходы дают дополнительный конфликтный сигнал поверх уже имеющейся дорожной структуры.

Но по масштабу этот новый блок все еще не доминирует. Главные группы остаются теми же: junction graph, road type shares, surface quality, lanes/capacity и road amount. То есть новые признаки не перевернули модель, а аккуратно расширили ее представление о конфликтных точках.

Практический вывод:

- group 2.0 признаки полезны и стоит оставить;
- ждать от них резкого "обеления центра" не стоит;
- чтобы реально разбить весь красный центр на более осмысленные паттерны риска, вероятно, уже нужен следующий блок - геометрия и связность сети из группы 3.

В viewer добавлен отдельный слой:

```text
CatBoost road-only 2.0
```

## 2026-06-09 - CatBoost Road-Only Without Calendar Features

### Что сделано

Проведен CatBoost-прогон только на H3-month строках, где в H3 есть дороги:

```text
road_length_total_m > 0
```

Запуск:

```text
scripts/run_catboost_baseline.py \
  --feature-set road_without_type_lengths_no_calendar \
  --min-road-length-m 0
```

Для viewer собран отдельный parquet:

```text
parquet_output/h3_month_catboost_road_only_predictions.parquet
```

Feature importance сохранен:

```text
reports/catboost_feature_importance_road_only_no_calendar.csv
```

### Данные и признаки

Таблица:

```text
parquet_output/h3_month_road_features.parquet
```

Использован тот же no-calendar feature set, что в прошлом эксперименте:

```text
road / junction / speed features
road_length_total_m
road_share_*
```

Исключены:

```text
year
month_num
road_length_<highway>_m
h3
month
target_had_accident_next_month
```

| Показатель | Значение |
|---|---:|
| Rows после фильтра | 567,072 |
| H3 в road-only viewer layer | 4,296 |
| Feature count | 67 |
| Target | `target_had_accident_next_month` |
| Train period | 2015-01 - 2022-12 |
| Test period | 2023-01 - 2025-12 |
| Train rows | 412,416 |
| Test rows | 154,656 |
| Train positive rate | 2.5850% |
| Test positive rate | 2.3349% |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.907575 |
| PR-AUC | 0.196892 |
| Accuracy | 0.8229 |
| Precision, класс `1` | 0.1020 |
| Recall, класс `1` | 0.8438 |
| F1-score, класс `1` | 0.1820 |
| Support, класс `0` | 151,045 |
| Support, класс `1` | 3,611 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 124,212 | 26,833 |
| Actual 1 | 564 | 3,047 |

Top-k метрики считаются внутри road-only test set:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 1,546 | 0.3001 | 0.1285 | 12.85 | 464 / 3,611 |
| Top 3% | 4,639 | 0.2479 | 0.3185 | 10.62 | 1,150 / 3,611 |
| Top 5% | 7,732 | 0.2065 | 0.4423 | 8.85 | 1,597 / 3,611 |
| Top 10% | 15,465 | 0.1512 | 0.6475 | 6.47 | 2,338 / 3,611 |

### Сравнение с прошлым CatBoost no-calendar

Предыдущий no-calendar эксперимент обучался и оценивался на всех H3, включая ячейки без дорог. Текущий road-only эксперимент обучается и оценивается только на дорожных H3.

| Метрика | CatBoost no calendar, all H3 | CatBoost road-only | Изменение |
|---|---:|---:|---:|
| Test rows | 301,932 | 154,656 | -147,276 |
| Test positive rate | 1.1966% | 2.3349% | +1.1383 п.п. |
| ROC-AUC | 0.952030 | 0.907575 | -0.044455 |
| PR-AUC | 0.195761 | 0.196892 | +0.001131 |
| Accuracy | 0.8742 | 0.8229 | -0.0513 |
| Precision, класс `1` | 0.0806 | 0.1020 | +0.0214 |
| Recall, класс `1` | 0.9142 | 0.8438 | -0.0704 |
| F1-score, класс `1` | 0.1482 | 0.1820 | +0.0338 |
| Top 1% precision | 0.2719 | 0.3001 | +0.0282 |
| Top 1% recall | 0.2272 | 0.1285 | -0.0987 |
| Top 1% lift | 22.73 | 12.85 | -9.88 |
| Top 3% precision | 0.1942 | 0.2479 | +0.0537 |
| Top 3% recall | 0.4869 | 0.3185 | -0.1684 |
| Top 5% precision | 0.1530 | 0.2065 | +0.0535 |
| Top 5% recall | 0.6391 | 0.4423 | -0.1968 |
| Top 5% lift | 12.78 | 8.85 | -3.93 |
| Top 10% precision | 0.1011 | 0.1512 | +0.0501 |
| Top 10% recall | 0.8453 | 0.6475 | -0.1978 |

### На что модель опирается

Top-20 признаков по встроенной важности CatBoost:

| Признак | Importance |
|---|---:|
| `asphalt_surface_length_m` | 10.118432 |
| `road_share_residential` | 7.054723 |
| `lane_meters_estimate` | 6.650928 |
| `road_share_unclassified` | 4.920717 |
| `road_length_total_m` | 4.779366 |
| `junction_density_per_km` | 4.496844 |
| `asphalt_surface_share` | 4.004012 |
| `maxspeed_mean_weighted` | 3.988701 |
| `road_share_service` | 3.208791 |
| `road_way_count` | 2.786031 |
| `class_delta_max` | 2.689784 |
| `road_segment_h3_count` | 2.461212 |
| `class_delta_mean` | 2.307932 |
| `bad_surface_length_m` | 2.264525 |
| `road_share_tertiary` | 2.237910 |
| `lanes_mean_weighted` | 2.219995 |
| `lanes_tagged_share` | 1.879241 |
| `bad_surface_share` | 1.799062 |
| `junction_degree_mean` | 1.792238 |
| `class_delta_junction_density_per_km` | 1.777130 |

Группировка признаков по смыслу:

| Группа признаков | Importance | Доля |
|---|---:|---:|
| Junction / graph structure | 25.798 | 25.8% |
| Road type shares | 19.606 | 19.6% |
| Surface quality / coverage | 18.186 | 18.2% |
| Lanes / capacity | 11.967 | 12.0% |
| Road amount / density | 11.560 | 11.6% |
| Speed | 6.530 | 6.5% |
| Other road attributes | 3.907 | 3.9% |
| Point infrastructure | 2.447 | 2.4% |

### Интерпретация

Road-only эксперимент делает задачу заметно честнее: модель больше не может получать большой выигрыш просто от различения "дорожная городская ячейка" против "пустая ячейка без дорог". Поэтому ROC-AUC ожидаемо падает с `0.952030` до `0.907575`: исчезает легкая часть задачи.

При этом PR-AUC почти не падает и даже чуть растет: `0.195761` -> `0.196892`. Это важный сигнал. Внутри только дорожных ячеек модель все еще умеет отделять будущие ДТП от нулей, хотя задача стала сложнее.

Top-k precision внутри road-only выборки стал выше: Top 1% precision вырос с `0.2719` до `0.3001`, Top 5% precision - с `0.1530` до `0.2065`. Но эти числа считаются на другом denominator: Top 1% теперь это `1,546` строк, а не `3,019`; Top 5% - `7,732`, а не `15,096`. Поэтому recall ниже: мы смотрим меньше абсолютных ячеек.

Практический смысл:

- all-H3 CatBoost no-calendar отвечает на вопрос: "где в городе вообще вероятнее ДТП";
- road-only CatBoost отвечает на более узкий вопрос: "какие дорожные H3 рискованнее среди H3, где есть дороги".

Для инженерной интерпретации road-only слой полезнее, потому что он меньше сводится к факту наличия городской дорожной сети. При этом feature importance почти не поменялся по группам: junction graph, road type shares, surface, lanes/capacity и road amount остаются главными блоками. Это значит, что даже после удаления пустых H3 модель продолжает использовать структуру дорожной сети, а не только бинарный признак "есть дорога".

В viewer добавлен отдельный слой:

```text
CatBoost road-only
```

Этот слой показывает только H3, где `road_length_total_m > 0`.

## 2026-06-09 - CatBoost Without Calendar Features

### Что сделано

Проведен CatBoost-прогон без календарных признаков:

```text
year
month_num
```

Запуск:

```text
scripts/run_catboost_baseline.py --feature-set road_without_type_lengths_no_calendar
```

Для viewer дополнительно собран parquet с помесячными score/rank:

```text
parquet_output/h3_month_catboost_no_calendar_predictions.parquet
```

Feature importance сохранен:

```text
reports/catboost_feature_importance_without_type_lengths_no_calendar.csv
```

### Данные и признаки

Таблица:

```text
parquet_output/h3_month_road_features.parquet
```

Использован тот же набор road/junction/speed признаков, что в CatBoost baseline, но исключены `year` и `month_num`.

| Показатель | Значение |
|---|---:|
| Rows | 1,107,084 |
| Columns в parquet | 86 |
| Feature count | 67 |
| Target | `target_had_accident_next_month` |
| Train period | 2015-01 - 2022-12 |
| Test period | 2023-01 - 2025-12 |
| Train rows | 805,152 |
| Test rows | 301,932 |
| Train positive rate | 1.3319% |
| Test positive rate | 1.1966% |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.952030 |
| PR-AUC | 0.195761 |
| Accuracy | 0.8742 |
| Precision, класс `1` | 0.0806 |
| Recall, класс `1` | 0.9142 |
| F1-score, класс `1` | 0.1482 |
| Support, класс `0` | 298,319 |
| Support, класс `1` | 3,613 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 260,654 | 37,665 |
| Actual 1 | 310 | 3,303 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 3,019 | 0.2719 | 0.2272 | 22.73 | 821 / 3,613 |
| Top 3% | 9,057 | 0.1942 | 0.4869 | 16.23 | 1,759 / 3,613 |
| Top 5% | 15,096 | 0.1530 | 0.6391 | 12.78 | 2,309 / 3,613 |
| Top 10% | 30,193 | 0.1011 | 0.8453 | 8.45 | 3,054 / 3,613 |

### Сравнение с CatBoost baseline

Baseline содержит `year` и `month_num`; текущий эксперимент исключает оба календарных признака.

| Метрика | CatBoost baseline | CatBoost no calendar | Изменение |
|---|---:|---:|---:|
| ROC-AUC | 0.952714 | 0.952030 | -0.000684 |
| PR-AUC | 0.193741 | 0.195761 | +0.002020 |
| Accuracy | 0.8868 | 0.8742 | -0.0126 |
| Precision, класс `1` | 0.0872 | 0.0806 | -0.0066 |
| Recall, класс `1` | 0.8937 | 0.9142 | +0.0205 |
| F1-score, класс `1` | 0.1589 | 0.1482 | -0.0107 |
| Top 1% precision | 0.2719 | 0.2719 | +0.0000 |
| Top 1% recall | 0.2272 | 0.2272 | +0.0000 |
| Top 1% lift | 22.73 | 22.73 | +0.00 |
| Top 3% precision | 0.1938 | 0.1942 | +0.0004 |
| Top 3% recall | 0.4857 | 0.4869 | +0.0012 |
| Top 5% precision | 0.1533 | 0.1530 | -0.0003 |
| Top 5% recall | 0.6405 | 0.6391 | -0.0014 |
| Top 5% lift | 12.81 | 12.78 | -0.03 |
| Top 10% precision | 0.1016 | 0.1011 | -0.0005 |
| Top 10% recall | 0.8489 | 0.8453 | -0.0036 |

### На что модель опирается

Top-20 признаков по встроенной важности CatBoost:

| Признак | Importance |
|---|---:|
| `asphalt_surface_length_m` | 9.089848 |
| `road_share_residential` | 7.824023 |
| `lane_meters_estimate` | 7.051420 |
| `road_length_total_m` | 4.809633 |
| `junction_density_per_km` | 4.203420 |
| `road_share_unclassified` | 4.101311 |
| `asphalt_surface_share` | 4.092155 |
| `road_share_service` | 3.232170 |
| `maxspeed_mean_weighted` | 2.833115 |
| `road_way_count` | 2.776708 |
| `road_segment_h3_count` | 2.712748 |
| `bad_surface_length_m` | 2.693374 |
| `bad_surface_share` | 2.498803 |
| `road_share_tertiary` | 2.452101 |
| `class_delta_mean` | 2.322211 |
| `lanes_mean_weighted` | 2.157615 |
| `class_delta_max` | 2.157351 |
| `junction_degree_mean` | 2.137256 |
| `maxspeed_is_imputed_share` | 1.989274 |
| `class_delta_sum` | 1.879884 |

Группировка признаков по смыслу:

| Группа признаков | Importance | Доля |
|---|---:|---:|
| Junction / graph structure | 25.488 | 25.5% |
| Road type shares | 19.256 | 19.3% |
| Surface quality / coverage | 18.374 | 18.4% |
| Lanes / capacity | 11.952 | 12.0% |
| Road amount / density | 11.731 | 11.7% |
| Speed | 6.366 | 6.4% |
| Other road attributes | 4.129 | 4.1% |
| Point infrastructure | 2.704 | 2.7% |

### Интерпретация

Удаление `year` и `month_num` почти не ухудшило ранжирование top-k. Top 1% остался тем же, Top 3% даже чуть вырос, Top 5-10% просели минимально. PR-AUC вырос с `0.193741` до `0.195761`, а ROC-AUC снизился совсем немного.

Это хороший сигнал: CatBoost не держался критически на календарных признаках. Основной ранжирующий сигнал сохраняется в дорожной структуре: покрытие/asphalt, состав типов дорог, lane meters, общая длина дорог, плотность перекрестков, speed и class-delta признаки.

По группам видно, что самый сильный блок - это не один отдельный тег OSM, а структура дорожного графа: плотность перекрестков, class-delta признаки, minor-to-major junction density, degree перекрестков и несветофорные пересечения дорог разного класса. Это хорошо совпадает с исходной гипотезой: риск должен расти не просто от наличия дороги, а от сложности дорожной конфигурации внутри H3.

Второй сильный блок - доли типов дорог (`road_share_residential`, `road_share_unclassified`, `road_share_service`, `road_share_tertiary`). Модель различает жилую уличную сетку, служебные/дворовые дороги, неклассифицированные городские улицы и более связные tertiary-улицы. Это важнее, чем абсолютная длина каждого типа дороги, поэтому схема `road_length_total_m + road_share_*` выглядит оправданной.

Третий сильный блок - surface признаки (`asphalt_surface_length_m`, `asphalt_surface_share`, `bad_surface_*`). Их нужно трактовать осторожно: они могут отражать не только реальное качество покрытия, но и полноту OSM-разметки. Более подробно размеченные дороги часто находятся в городской активной зоне, поэтому часть сигнала может быть proxy для плотности/важности места, а не физического состояния дороги.

Capacity-признаки (`lane_meters_estimate`, `lanes_mean_weighted`, `lanes_tagged_share`, `lanes_max`) тоже заметны. Это похоже на сигнал масштаба дороги: больше полос и больше lane-meters обычно соответствуют более загруженным и крупным участкам.

Threshold-метрики при пороге `0.5` стали менее аккуратными по precision/accuracy, зато recall класса `1` вырос. Для нашей задачи это вторично: мы в первую очередь используем модель как ранжировщик риска по top-k, а не как бинарный классификатор с фиксированным порогом.

Практический вывод: вариант без календаря выглядит более честным кандидатом на основной baseline, потому что он почти не теряет top-k качество и меньше рискует кодировать временные особенности датасета вместо дорожного риска.

## 2026-06-05 - CatBoost Baseline On Road Structure Features

### Что сделано

Добавлен и запущен CatBoost baseline:

```text
scripts/run_catboost_baseline.py
```

Модель:

```text
CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="PRAUC",
    auto_class_weights="Balanced",
    random_seed=42,
)
```

Важно: `eval_set` в запуске использовался только для мониторинга качества во время обучения, `use_best_model=False`, поэтому модель не выбирала лучшую итерацию по тесту. Для сравнения с логистической регрессией используем внешние sklearn-метрики, напечатанные после обучения.

### Данные и признаки

Таблица:

```text
parquet_output/h3_month_road_features.parquet
```

Использован тот же feature set, что в актуальном логрег-эксперименте:

```text
road_length_total_m
road_share_*
остальные road / junction / speed / calendar features
```

Исключены:

```text
h3
month
target_had_accident_next_month
road_length_<highway>_m
```

| Показатель | Значение |
|---|---:|
| Rows | 1,107,084 |
| Columns в parquet | 86 |
| Feature count | 69 |
| Target | `target_had_accident_next_month` |
| Train period | 2015-01 - 2022-12 |
| Test period | 2023-01 - 2025-12 |
| Train rows | 805,152 |
| Test rows | 301,932 |
| Train positive rate | 1.3319% |
| Test positive rate | 1.1966% |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.952714 |
| PR-AUC | 0.193741 |
| Accuracy | 0.8868 |
| Precision, класс `1` | 0.0872 |
| Recall, класс `1` | 0.8937 |
| F1-score, класс `1` | 0.1589 |
| Support, класс `0` | 298,319 |
| Support, класс `1` | 3,613 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 264,516 | 33,803 |
| Actual 1 | 384 | 3,229 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 3,019 | 0.2719 | 0.2272 | 22.73 | 821 / 3,613 |
| Top 3% | 9,057 | 0.1938 | 0.4857 | 16.19 | 1,755 / 3,613 |
| Top 5% | 15,096 | 0.1533 | 0.6405 | 12.81 | 2,314 / 3,613 |
| Top 10% | 30,193 | 0.1016 | 0.8489 | 8.49 | 3,067 / 3,613 |

### Сравнение с логистической регрессией

Сравнение с экспериментом `Road Structure Logistic Regression Without Per-Type Road Lengths`.

| Метрика | Logistic Regression | CatBoost | Изменение |
|---|---:|---:|---:|
| ROC-AUC | 0.948101 | 0.952714 | +0.004613 |
| PR-AUC | 0.165792 | 0.193741 | +0.027949 |
| Accuracy | 0.8738 | 0.8868 | +0.0130 |
| Precision, класс `1` | 0.0793 | 0.0872 | +0.0079 |
| Recall, класс `1` | 0.8990 | 0.8937 | -0.0053 |
| F1-score, класс `1` | 0.1457 | 0.1589 | +0.0132 |
| Top 1% precision | 0.2262 | 0.2719 | +0.0457 |
| Top 1% recall | 0.1890 | 0.2272 | +0.0382 |
| Top 1% lift | 18.91 | 22.73 | +3.82 |
| Top 3% precision | 0.1788 | 0.1938 | +0.0150 |
| Top 3% recall | 0.4481 | 0.4857 | +0.0376 |
| Top 5% precision | 0.1456 | 0.1533 | +0.0077 |
| Top 5% recall | 0.6084 | 0.6405 | +0.0321 |
| Top 5% lift | 12.17 | 12.81 | +0.64 |
| Top 10% precision | 0.0987 | 0.1016 | +0.0029 |
| Top 10% recall | 0.8245 | 0.8489 | +0.0244 |

### На что модель опирается

Feature importance сохранен:

```text
reports/catboost_feature_importance_without_type_lengths.csv
```

Top-20 признаков по встроенной важности CatBoost:

| Признак | Importance |
|---|---:|
| `year` | 10.830584 |
| `month_num` | 7.607264 |
| `asphalt_surface_length_m` | 7.066964 |
| `road_share_residential` | 6.313651 |
| `lane_meters_estimate` | 5.493864 |
| `road_length_total_m` | 4.432938 |
| `asphalt_surface_share` | 4.274054 |
| `road_share_unclassified` | 3.597239 |
| `road_share_tertiary` | 3.375276 |
| `junction_density_per_km` | 3.308660 |
| `maxspeed_mean_weighted` | 3.028835 |
| `road_share_service` | 2.466923 |
| `road_segment_h3_count` | 2.380671 |
| `bad_surface_length_m` | 2.276151 |
| `class_delta_mean` | 1.857546 |
| `bad_surface_share` | 1.830343 |
| `class_delta_max` | 1.798765 |
| `lanes_tagged_share` | 1.661979 |
| `major_road_length_m` | 1.497707 |
| `class_delta_sum` | 1.426817 |

### Интерпретация

CatBoost дал заметный прирост именно там, где нам важнее всего: в ранжировании самых рискованных ячеек. Top 1% precision вырос с 22.62% до 27.19%, то есть среди 3,019 самых рискованных h3-month строк модель нашла 821 будущую ДТП-строку вместо 683 у логистической регрессии. Top 5% recall вырос с 60.84% до 64.05%, то есть при просмотре верхних 5% риска CatBoost покрывает больше реальных ДТП.

Threshold-метрики тоже стали лучше по precision/F1, но recall класса `1` на пороге `0.5` слегка ниже. Для нашей задачи это не выглядит проблемой, потому что практический сценарий больше похож на сортировку ячеек по риску и работу с top-k, а не на жесткое бинарное решение по порогу `0.5`.

Главный вывод: CatBoost выглядит сильнее логистической регрессии как ранжирующая модель риска. При этом важность `year` и `month_num` очень высокая, поэтому следующим шагом стоит проверить устойчивость на временных срезах и отдельно прогнать вариант без календарных признаков или с более аккуратной валидацией по годам.

## 2026-06-05 - Road Structure Logistic Regression Without Per-Type Road Lengths

### Что сделано

Обучена логистическая регрессия на текущей таблице:

```text
parquet_output/h3_month_road_features.parquet
```

Из признаков исключены per-type абсолютные длины дорог:

```text
road_length_living_street_m
road_length_motorway_m
road_length_motorway_link_m
road_length_primary_m
road_length_primary_link_m
road_length_residential_m
road_length_secondary_m
road_length_secondary_link_m
road_length_service_m
road_length_tertiary_m
road_length_tertiary_link_m
road_length_trunk_m
road_length_trunk_link_m
road_length_unclassified_m
```

Оставлены:

```text
road_length_total_m
road_share_*
остальные road / junction / speed / calendar features
```

Идея: не давать логистической регрессии одновременно `road_length_total_m`, `road_share_*` и `road_length_<type>_m`, потому что per-type length почти равен `road_length_total_m * road_share_<type>` и создает сильную корреляцию признаков.

### Данные

| Показатель | Значение |
|---|---:|
| Rows | 1,107,084 |
| Columns в parquet | 86 |
| Feature count после исключения | 69 |
| Target | `target_had_accident_next_month` |
| Train period | 2015-01 - 2022-12 |
| Test period | 2023-01 - 2025-12 |
| Train rows | 805,152 |
| Test rows | 301,932 |
| Train positive rate | 1.3319% |
| Test positive rate | 1.1966% |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.948101 |
| PR-AUC | 0.165792 |
| Accuracy | 0.8738 |
| Precision, класс `1` | 0.0793 |
| Recall, класс `1` | 0.8990 |
| F1-score, класс `1` | 0.1457 |
| Support, класс `0` | 298,319 |
| Support, класс `1` | 3,613 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 260,592 | 37,727 |
| Actual 1 | 365 | 3,248 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 3,019 | 0.2262 | 0.1890 | 18.91 | 683 / 3,613 |
| Top 3% | 9,057 | 0.1788 | 0.4481 | 14.94 | 1,619 / 3,613 |
| Top 5% | 15,096 | 0.1456 | 0.6084 | 12.17 | 2,198 / 3,613 |
| Top 10% | 30,193 | 0.0987 | 0.8245 | 8.25 | 2,979 / 3,613 |

### Сравнение с прошлым записанным road-экспериментом

Важно: это сравнение не является чистой ablation только per-type lengths. После прошлого записанного эксперимента в таблицу были добавлены/изменены признаки: `road_share_*`, `class_delta_*`, новая логика `minor_to_major`, маппинг `RU:*` maxspeed и `maxspeed_is_imputed_share`.

| Метрика | Road structure previous | Current without per-type lengths | Изменение |
|---|---:|---:|---:|
| ROC-AUC | 0.947687 | 0.948101 | +0.000414 |
| PR-AUC | 0.160188 | 0.165792 | +0.005604 |
| Precision, класс `1` | 0.0785 | 0.0793 | +0.0008 |
| Recall, класс `1` | 0.8934 | 0.8990 | +0.0056 |
| F1-score, класс `1` | 0.1443 | 0.1457 | +0.0014 |
| Top 1% precision | 0.2242 | 0.2262 | +0.0020 |
| Top 1% recall | 0.1874 | 0.1890 | +0.0016 |
| Top 1% lift | 18.74 | 18.91 | +0.17 |
| Top 5% precision | 0.1453 | 0.1456 | +0.0003 |
| Top 5% recall | 0.6073 | 0.6084 | +0.0011 |
| Top 5% lift | 12.15 | 12.17 | +0.02 |
| Top 10% precision | 0.0985 | 0.0987 | +0.0002 |
| Top 10% recall | 0.8229 | 0.8245 | +0.0016 |
| Top 10% lift | 8.23 | 8.25 | +0.02 |

### Контроль: текущая таблица со всеми признаками

Для проверки эффекта удаления per-type `road_length_*_m` дополнительно обучена модель на текущей таблице со всеми 83 признаками.

| Метрика | Current all features | Current without per-type lengths | Разница without - all |
|---|---:|---:|---:|
| ROC-AUC | 0.948634 | 0.948101 | -0.000533 |
| PR-AUC | 0.165626 | 0.165792 | +0.000166 |
| Precision, класс `1` | 0.0792 | 0.0793 | +0.0001 |
| Recall, класс `1` | 0.8995 | 0.8990 | -0.0005 |
| Top 1% precision | 0.2256 | 0.2262 | +0.0006 |
| Top 3% precision | 0.1835 | 0.1788 | -0.0047 |
| Top 5% precision | 0.1464 | 0.1456 | -0.0008 |
| Top 10% precision | 0.0984 | 0.0987 | +0.0003 |

### Интерпретация

Метрики без per-type lengths не просели относительно прошлого записанного road baseline и немного выросли по PR-AUC и top-k. Это говорит, что замена абсолютных длин типов на `road_length_total_m + road_share_*` как минимум не ломает качество.

Контрольный прогон на текущей таблице со всеми признаками показывает, что per-type lengths дают очень небольшой и неоднозначный эффект: ROC-AUC чуть выше со всеми признаками, PR-AUC и Top 1% precision чуть выше без них, Top 3-5% precision чуть выше со всеми. Разницы маленькие, поэтому для логистической регрессии более чистая схема `total length + shares` выглядит предпочтительнее: она проще интерпретируется и меньше дублирует один и тот же сигнал.

Практический вывод: для следующего основного эксперимента стоит использовать `road_length_total_m + road_share_*`, а per-type `road_length_*_m` держать в parquet для диагностики, но не включать в базовый feature set.

## 2026-06-04 - Road Structure Features From Local Geofabrik PBF

### Что сделано

Пользователь добавил локальный Geofabrik PBF:

```text
volga-fed-district-260603.osm.pbf
```

Файл перенесен в raw data:

```text
data/raw/osm/volga-fed-district-260603.osm.pbf
```

Добавлен и запущен скрипт:

```text
scripts/build_osm_road_features.py build-pbf
```

Скрипт читает PBF, фильтрует bbox Саратова, строит дорожные признаки по H3 и объединяет их с target из `h3_month_panel.parquet`.

### Настройки

```text
H3_RESOLUTION = 9
BBOX = (51.40, 45.80, 51.70, 46.20)
SAMPLE_STEP_M = 20.0
```

### Что сохранено

| Файл | Строки | Смысл |
|---|---:|---|
| `parquet_output/osm_roads_raw.parquet` | 30,453 | OSM road ways после bbox-фильтра |
| `parquet_output/osm_road_h3_segments.parquet` | 48,247 | Куски дорог, распределенные по H3 |
| `parquet_output/h3_road_features.parquet` | 4,886 | Статические road features по H3 |
| `parquet_output/h3_month_road_features.parquet` | 1,107,084 | `H3 x month` + road features + target |

### PBF processing stats

| Показатель | Значение |
|---|---:|
| PBF size | 729.13 MB |
| Ways processed, pass 1 | 11,135,748 |
| Motor-road ways collected before bbox filter | 1,144,884 |
| Required node ids | 7,087,424 |
| Nodes processed, pass 2 | 110,900,367 |
| Filtered road nodes | 148,010 |
| Filtered road ways | 30,453 |
| Filtered tagged road nodes | 2,335 |
| Dropped ways outside bbox | 1,114,431 |
| Total elapsed | 1,418.6s |

### Использованные группы признаков

В этот parquet вошли только structural road features и календарные признаки, без исторических лагов ДТП:

| Группа | Примеры |
|---|---|
| Road length | `road_length_total_m`, `road_length_primary_m`, `road_length_residential_m`, `road_length_service_m` |
| Road class shares | `major_road_share`, `link_road_share` |
| Capacity/speed | `lane_meters_estimate`, `lanes_mean_weighted`, `maxspeed_mean_weighted`, `maxspeed_max` |
| Road attributes | `oneway_share`, `lit_share`, `bridge_share`, `tunnel_share`, `bad_surface_share`, `asphalt_surface_share` |
| Junction graph | `junction_count`, `three_way_junction_count`, `four_plus_way_junction_count`, `major_junction_count`, `minor_to_major_junction_count` |
| Road point infrastructure | `traffic_signal_count`, `crossing_count`, `stop_count`, `give_way_count`, `traffic_calming_count` |
| Calendar | `year`, `month_num` |

### Sanity checks

| Показатель | Значение |
|---|---:|
| H3 с road features | 4,886 |
| H3 в merged ML table | 8,387 |
| H3 с `road_length_total_m > 0` | 4,296 |
| Rows с `road_length_total_m > 0` | 567,072 |
| Positive rate для H3 с дорогами | 2.5168% |
| Positive rate для H3 без дорог | 0.0120% |

### Логистическая регрессия на road features

Модель:

```text
LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
```

Train/test split:

```text
train: 2015-01 - 2022-12
test:  2023-01 - 2025-12
```

Размер:

| Часть | Строки | Positive rate |
|---|---:|---:|
| Train | 805,152 | 1.3319% |
| Test | 301,932 | 1.1966% |

Метрики:

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.947687 |
| PR-AUC | 0.160188 |
| Accuracy | 0.8732 |
| Precision, класс `1` | 0.0785 |
| Recall, класс `1` | 0.8934 |
| F1-score, класс `1` | 0.1443 |
| Support, класс `0` | 298,319 |
| Support, класс `1` | 3,613 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 260,405 | 37,914 |
| Actual 1 | 385 | 3,228 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 3,019 | 0.2242 | 0.1874 | 18.74 | 677 / 3,613 |
| Top 3% | 9,057 | 0.1756 | 0.4401 | 14.67 | 1,590 / 3,613 |
| Top 5% | 15,096 | 0.1453 | 0.6073 | 12.15 | 2,194 / 3,613 |
| Top 10% | 30,193 | 0.0985 | 0.8229 | 8.23 | 2,973 / 3,613 |

### Сравнение

| Метрика | Calendar only | Historical baseline | Road structure |
|---|---:|---:|---:|
| ROC-AUC | 0.524515 | 0.829779 | 0.947687 |
| PR-AUC | 0.012750 | 0.149182 | 0.160188 |
| Top 1% precision | 0.0149 | 0.2411 | 0.2242 |
| Top 1% lift | 1.25 | 20.15 | 18.74 |
| Top 5% precision | 0.0132 | 0.1328 | 0.1453 |
| Top 5% recall | 0.0554 | 0.5549 | 0.6073 |
| Top 5% lift | 1.11 | 11.10 | 12.15 |
| Top 10% precision | 0.0136 | 0.0852 | 0.0985 |
| Top 10% lift | 1.13 | 7.12 | 8.23 |

### Интерпретация

Road structure features без исторических лагов уже дают сильный сигнал:

- PR-AUC выше historical baseline: `0.160188` против `0.149182`.
- Top 5% lift выше historical baseline: `12.15` против `11.10`.
- Top 5% recall выше historical baseline: `60.73%` против `55.49%`.
- Top 1% precision чуть ниже historical baseline: `22.42%` против `24.11%`.

Вывод: дорожная структура уже полезнее календаря и сопоставима или лучше исторических лагов для top-k risk ranking. Это важный шаг к structural risk model, применимой к новым дорогам.

Ограничение: PBF parsing пока читает весь `volga-fed-district` extract и занял около 24 минут. Следующая инженерная оптимизация - сделать локальный bbox extract или трехпроходный bbox-first parser, чтобы не держать миллионы road nodes всего федерального округа.

## 2026-06-04 - Calendar-Only Logistic Regression Baseline

### Что сделано

Добавлен скрипт:

```text
scripts/make_calendar_baseline.py
```

Он создает новый parquet-файл:

```text
parquet_output/calendar_baseline.parquet
```

Источник данных:

```text
parquet_output/h3_month_panel.parquet
```

После этого обучена логистическая регрессия только на календарных признаках, без исторических лагов ДТП и без дорожных/внешних признаков.

Команда создания датасета:

```bash
python scripts/make_calendar_baseline.py
```

Команда обучения:

```bash
python scripts/run_logreg_baseline.py \
  --panel parquet_output/calendar_baseline.parquet \
  --features year,month_num,month_sin,month_cos,is_winter,is_spring,is_summer,is_autumn
```

### Что прогнозировалось

Одна строка датасета - это:

```text
H3-ячейка x месяц
```

Целевая переменная:

```text
target_had_accident_next_month
```

Модель прогнозировала, будет ли хотя бы одно ДТП в данной H3-ячейке в следующем месяце.

### Использованные признаки

| Признак | Смысл |
|---|---|
| `year` | Год наблюдения |
| `month_num` | Номер месяца |
| `month_sin` | Циклическое представление месяца |
| `month_cos` | Циклическое представление месяца |
| `is_winter` | Зимний месяц |
| `is_spring` | Весенний месяц |
| `is_summer` | Летний месяц |
| `is_autumn` | Осенний месяц |

Важно: в этом эксперименте нет признаков, которые различают H3-ячейки внутри одного месяца. Поэтому модель может учить только общую сезонность/тренд, но не может строить реальную spatial risk map.

### Размер датасета

| Часть | Строки | Период | Positive rate |
|---|---:|---|---:|
| Panel | 1,107,084 | 2015-01 - 2026-01 | 1.2950% |
| Train | 805,152 | 2015-01 - 2022-12 | 1.3319% |
| Test | 301,932 | 2023-01 - 2025-12 | 1.1966% |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.524515 |
| PR-AUC | 0.012750 |
| Accuracy | 0.3384 |
| Precision, класс `1` | 0.0128 |
| Recall, класс `1` | 0.7127 |
| F1-score, класс `1` | 0.0251 |
| Support, класс `0` | 298,319 |
| Support, класс `1` | 3,613 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 99,606 | 198,713 |
| Actual 1 | 1,038 | 2,575 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 3,019 | 0.0149 | 0.0125 | 1.25 | 45 / 3,613 |
| Top 3% | 9,057 | 0.0136 | 0.0340 | 1.13 | 123 / 3,613 |
| Top 5% | 15,096 | 0.0132 | 0.0554 | 1.11 | 200 / 3,613 |
| Top 10% | 30,193 | 0.0136 | 0.1135 | 1.13 | 410 / 3,613 |

### Сравнение с historical baseline

Historical baseline - это предыдущий эксперимент, где использовались календарные признаки и лаги ДТП:

```text
year
month_num
accidents_prev_1m
accidents_prev_3m
accidents_prev_6m
accidents_prev_12m
```

| Метрика | Calendar only | Historical baseline | Изменение |
|---|---:|---:|---:|
| ROC-AUC | 0.524515 | 0.829779 | -0.305264 |
| PR-AUC | 0.012750 | 0.149182 | -0.136432 |
| Precision, класс `1` | 0.0128 | 0.1049 | -0.0921 |
| Recall, класс `1` | 0.7127 | 0.7088 | +0.0039 |
| F1-score, класс `1` | 0.0251 | 0.1828 | -0.1577 |
| Top 1% precision | 0.0149 | 0.2411 | -0.2262 |
| Top 1% lift | 1.25 | 20.15 | -18.90 |
| Top 5% precision | 0.0132 | 0.1328 | -0.1196 |
| Top 5% recall | 0.0554 | 0.5549 | -0.4995 |
| Top 5% lift | 1.11 | 11.10 | -9.99 |
| Top 10% precision | 0.0136 | 0.0852 | -0.0716 |
| Top 10% lift | 1.13 | 7.12 | -5.99 |

### Интерпретация

Calendar-only baseline почти не дает полезного сигнала:

- ROC-AUC близок к `0.5`, то есть почти случайное ранжирование.
- PR-AUC `0.012750` почти равен test positive rate `0.011966`.
- Top-k lift находится около `1.1`, то есть top зоны почти не лучше случайного выбора.
- Высокий recall при threshold `0.5` не является хорошим результатом: модель просто помечает слишком много строк как positive, из-за чего precision падает почти до базовой частоты ДТП.

Вывод: календарные признаки сами по себе не решают задачу. Они могут быть полезны как слабый временной контекст, но spatial risk map без дорожных, инфраструктурных, population/POI и traffic features построить нельзя.

## 2026-06-04 - Logistic Regression Baseline With All H3 Cells In Saratov Bbox

### Что сделано

Пересобрана H3 x month panel для bbox Саратова с включением всех H3-ячеек внутри bbox:

```text
INCLUDE_ALL_H3_CELLS_IN_BBOX = True
H3_RESOLUTION = 9
BBOX = (51.40, 45.80, 51.70, 46.20)
```

После этого обучена baseline-модель логистической регрессии:

```text
model = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
```

Train/test split сделан по времени:

```text
train: 2015-01 - 2022-12
test:  2023-01 - 2025-12
```

### Что прогнозировалось

Одна строка датасета - это одна пара:

```text
H3-ячейка x месяц
```

Целевая переменная:

```text
target_had_accident_next_month
```

То есть модель прогнозировала, будет ли хотя бы одно ДТП в данной H3-ячейке в следующем месяце.

### Использованные признаки

В этом baseline использовались только календарные признаки и исторические лаги ДТП:

| Признак | Смысл |
|---|---|
| `year` | Год наблюдения |
| `month_num` | Номер месяца |
| `accidents_prev_1m` | Количество ДТП в этой H3-ячейке за предыдущий месяц |
| `accidents_prev_3m` | Сумма ДТП за предыдущие 3 месяца |
| `accidents_prev_6m` | Сумма ДТП за предыдущие 6 месяцев |
| `accidents_prev_12m` | Сумма ДТП за предыдущие 12 месяцев |

Важно: модель в этом виде в основном оценивает повторяемость ДТП в местах, где уже была аварийность. Для задачи оценки новых дорог такие признаки методологически слабые, потому что у новой дороги исторические лаги будут нулевыми.

### Размер датасета

| Часть | Строки | Период | Positive rate |
|---|---:|---|---:|
| Panel | 1,107,084 | 2015-01 - 2026-01 | - |
| Train | 805,152 | 2015-01 - 2022-12 | 1.3319% |
| Test | 301,932 | 2023-01 - 2025-12 | 1.1966% |

Дополнительно:

| Показатель | Значение |
|---|---:|
| H3-ячеек с ДТП | 1,795 |
| H3-ячеек внутри bbox | 8,383 |
| Всего H3-ячеек в panel | 8,387 |
| ДТП после bbox-фильтра | 16,058 |

### Метрики

| Метрика | Значение |
|---|---:|
| ROC-AUC | 0.829779 |
| PR-AUC | 0.149182 |
| Accuracy | 0.9242 |
| Precision, класс `1` | 0.1049 |
| Recall, класс `1` | 0.7088 |
| F1-score, класс `1` | 0.1828 |
| Support, класс `0` | 298,319 |
| Support, класс `1` | 3,613 |

Confusion matrix при threshold `0.5`:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 276,470 | 21,849 |
| Actual 1 | 1,052 | 2,561 |

Top-k метрики:

| Срез | k | Precision | Recall | Lift | TP |
|---|---:|---:|---:|---:|---:|
| Top 1% | 3,019 | 0.2411 | 0.2015 | 20.15 | 728 / 3,613 |
| Top 3% | 9,057 | 0.1695 | 0.4249 | 14.16 | 1,535 / 3,613 |
| Top 5% | 15,096 | 0.1328 | 0.5549 | 11.10 | 2,005 / 3,613 |
| Top 10% | 30,193 | 0.0852 | 0.7122 | 7.12 | 2,573 / 3,613 |

### На что опирается логистическая регрессия

Коэффициенты ниже посчитаны после `StandardScaler`, поэтому их можно сравнивать между собой. Чем больше модуль коэффициента, тем сильнее признак влияет на score модели.

| Признак | Стандартизированный коэффициент | Модуль коэффициента | Множитель odds на +1 std | Среднее в train | Std в train |
|---|---:|---:|---:|---:|---:|
| `accidents_prev_12m` | 1.119982 | 1.119982 | 3.064798 | 0.166551 | 0.768639 |
| `year` | -0.116480 | 0.116480 | 0.890048 | 2018.500000 | 2.291288 |
| `accidents_prev_6m` | 0.029664 | 0.029664 | 1.030108 | 0.086105 | 0.445533 |
| `accidents_prev_1m` | 0.024845 | 0.024845 | 1.025156 | 0.014781 | 0.134840 |
| `month_num` | 0.024624 | 0.024624 | 1.024930 | 6.500000 | 3.452053 |
| `accidents_prev_3m` | 0.017034 | 0.017034 | 1.017179 | 0.043819 | 0.270628 |

Интерпретация:

- `accidents_prev_12m` доминирует: увеличение этого признака на 1 стандартное отклонение умножает odds прогноза ДТП примерно на `3.06`.
- Остальные признаки влияют слабо: их множители odds близки к `1.0`.
- Отрицательный коэффициент у `year` означает, что при прочих равных более поздние годы получают чуть меньший score. Это может отражать тренд в данных, неполноту последних периодов или изменение регистрации ДТП; без отдельной проверки причин интерпретировать это как реальное снижение риска нельзя.
- Вывод: текущая логистическая регрессия почти полностью является моделью исторической аварийности за последние 12 месяцев, а не моделью структурного риска дороги.

### Интерпретация

Модель стала более честной для карты риска по всему bbox, потому что теперь negative examples включают все H3-ячейки bbox, а не только ячейки с историей ДТП.

При этом текущий baseline остается неудовлетворительным для задачи оценки новых или проектируемых дорог: модель почти полностью опирается на историю ДТП. Следующий содержательный шаг - собрать structural risk features без исторических лагов:

- длина дорог в H3;
- классы дорог;
- количество перекрестков;
- speed limit;
- число полос;
- пешеходные переходы;
- остановки общественного транспорта;
- школы, детсады, ТЦ и другие POI;
- плотность населения;
- расчетная или наблюдаемая интенсивность движения.
