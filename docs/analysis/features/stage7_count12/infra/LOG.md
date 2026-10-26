# Этап 7, шаг 1 — инфраструктура годовой постановки: журнал

ТЗ: `docs/analysis/features/PLAN.md`, этап 7 (7.0, 7.1 — панель, 7.4 — функции метрик).
Окружение: `venv/bin/python`, `PYTHONPATH=src`. Записи в хронологическом порядке; по ним можно продолжить с места.

## 2026-10-19

- Прочитаны: PLAN.md (этап 6 закрыт, этап 7.0–7.9), README 6.0 (`stage6_exposure/validation`),
  `v2_eval.py`, `run_v2_eval.py`, `catboost_baseline.py` (attach, p99), `h3_month_panel.py`.
- Данные: `crashes_slim.parquet` — 1 601 656 ДТП (вся РФ), 2015-01-01 … 2026-01-31; 12 месяцев в каждом
  году 2015–2025 + январь 2026. Месячная панель `h3_month_panel.parquet`: месяцы 2015-01…2025-12
  (последний месяц 2026-01 отброшен как «без таргета», но он входит в таргет строки 2025-12).
- `h3_month_road_features.parquet`: 1 107 084 строки = 8 387 ячеек bbox × 132 мес.; после фильтра
  `road_length_total_m > 0` — 4 296 ячеек.
- ДТП в bbox Саратова: 16 058 (с январём 2026); 2015–2025 — 15 994, по годам 1532 / 1422 / 1325 / 1523 /
  1645 / 1595 / 1522 / 1449 / 1388 / 1298 / 1295 (совпадает с суммой месячного `accident_count` по всем
  ячейкам bbox). **2025 год полный**: 12 месяцев, декабрь 2025 — 124 ДТП (обычный уровень, 74–130 в мес.).
  Все ДТП в выборке с пострадавшими (dead+injured > 0 у 100 %).
- Виды ДТП в bbox: столкновение 45.9 %, наезд на пешехода 37.3 %, на препятствие 6.1 %, падение пассажира
  3.5 %, на велосипедиста 2.4 %, на стоящее ТС 1.9 %, опрокидывание 1.25 %, съезд с дороги 0.99 %, прочие
  0.7 % (8 редких видов) → 8 колонок `count_type_*` + `count_type_other`.
- **7.0 статичность** (`v2_eval.load_panel(attach=[attach_exposure_v6_features])`, 567 072 × 209, 7.5 с):
  для 7 наборов (`road_geo_baseline` 61, `prune` 33, `junctions` 32, `tags` 33, `geometry` 30, `skdf` 32,
  `exposure_a` 34; объединение 83 признака) `groupby(h3).nunique(dropna=False).max() == 1` — **все
  признаки статичны по ячейке**. Значит, признаки ячейки можно брать из одного месяца панели.
- Найдено: билдеры наборов (`build_road_geo_baseline_features` и производные) берут **все колонки панели
  минус исключения** → в годовой панели нельзя передавать им колонки таргета/лагов (они попали бы в
  признаки). Решение: `year_variant_features` передаёт в `v2_eval.variant_features` только колонки
  признаков ячейки (в порядке месячной панели); лаги добавляются явно суффиксом `+lags`.
- Написан модуль `src/road_accident_risk/v2_eval_year.py` (панель, сплиты, обучение, метрики 7.4, ориентиры,
  месячная → годовая, сравнения). Решения:
  - доля накрытых ДТП считается **с честным учётом связей** (внутри группы равных скоров — линейная
    интерполяция кривой gain = матожидание при случайном порядке): для ориентиров `accidents_prev_year`
    (у ~84 % ячеек 0) сортировка без этого давала бы произвольный результат; k = max(1, int(n·q)) как в
    `precision_recall_at_k`;
  - `count_fatal_year` — число ДТП с погибшими (has_fatal); дополнительно `count_dead_persons_year`;
  - `accidents_prev_3y` — NaN, если нет полных трёх прошлых лет (2015–2017), `accidents_prev_year` — NaN в 2015;
  - месячная модель для годового скора обучается строго (`strict=True`): только строки, чей таргет лежит в
    годах обучения (для теста month < 2022-12-01), иначе метка декабря = январь года оценки;
  - признаки ячейки — из одного месяца `v2_eval.load_panel(attach=[attach_exposure_v6_features])` (кэш).
- `scripts/build_h3_year_panel.py` → `parquet_output/h3_year_panel.parquet` (47 256 × 19, 2 с).
  ДТП в ячейках с дорогами 15 927 из 15 994 (67 ДТП — в ячейках bbox без дорог, вне панели, как и в
  месячной схеме).
- `scripts/check_v2_eval_year.py` — 12 проверок, все OK (7.5 с): синтетика, суммы = месячные по каждой
  ячейке-году, лаги, каппы, наборы признаков = месячные.
- `scripts/run_v2_eval_year.py` написан. Первый дымовой прогон (Poisson, seed 42, both + ориентиры) показал
  **утечку в периоде 'sum'**: скор считался как сумма погодовых скоров, а у `accidents_prev_year` скор 2024 г.
  = факт 2023 г., входящий в суммарный таргет (Spearman 0.896 против 0.57 по годам). Исправлено: скор 'sum' =
  n_лет × скор первого года оценки (прогноз, сделанный до 2023 г.); для моделей на статичных признаках это то же
  самое. Добавлена проверка в `check_v2_eval_year.py`; файлы первого прогона удалены, прогон повторён.
- Прогоны с одним сидом давали вердикт «значимо» при 2σ = 0 (ve.verdict: ratio = inf). Введён `_verdict`:
  при 2σ = 0 или NaN — «нет 2σ» (1 сид, ориентир против ориентира — для них бутстреп по ячейкам).
- Добавлены `coverages` (несколько долей за одну сортировку), `period_frame`, `bootstrap_from_predictions`,
  флаг `--bootstrap` раннера; прогнозы месячной модели тоже пишутся в `_preds.parquet`.
- **Дымовой прогон** (load average 4–7, параллельно другие процессы), seed 42, `--split both`:
  ```
  PYTHONPATH=src venv/bin/python scripts/run_v2_eval_year.py --variants road_geo_v2_skdf --mode count --loss Poisson \
      --split both --seeds 42 --tag stage7_infra_smoke --baselines accidents_prev_year,accidents_prev_3y,major_road_length_m
  PYTHONPATH=src venv/bin/python scripts/run_v2_eval_year.py --variants road_geo_v2_skdf --mode binary --class-weights Balanced \
      --split both --seeds 42 --tag stage7_infra_smoke --monthly mean,max --base 'road_geo_v2_skdf|count:Poisson' --bootstrap
  ```
  время одной подгонки: Poisson 4–5 с, бинарная Balanced 5–8 с, месячная модель (для 7.2) 33–43 с; первый
  запуск 16 с, второй (с месячной моделью и бутстрепом) 173 с. Файлы: `reports/v2_eval_year_stage7_infra_smoke.csv`,
  `_delta.csv`, `_preds.parquet`, `_bootstrap.csv`.
- Месячная схема не сломана: `run_v2_eval.py --variants road_geo_v2_skdf,road_geo_baseline --split test --seeds 42
  --tag stage7_infra_month_check` → 0.908235 / 0.199760 / 0.3157 и 0.908056 / 0.198881 / 0.3072 (= эталоны), 76 с.
- `scripts/plot_stage7_infra.py` → `split_summary.csv`, `year_panel_by_year.csv`, `fig1_year_panel_counts.png`,
  `fig2_gain_curves_smoke.png` (+ CSV). Проверены глазами; на fig1 подписи столбцов перекрывались линией — поднят
  предел оси.
- PLAN.md (7.0, 7.1, 7.4, таблица) и EXPERIMENTS.md обновлены. Шаг 1 завершён.
