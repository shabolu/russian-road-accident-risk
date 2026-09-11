# Russian Road Accident Risk

Исследовательский пайплайн для подготовки данных ДТП и baseline-модели риска по H3-ячейкам и месяцам.

## Что лежит в проекте

- `src/road_accident_risk/` - основной Python-пакет с воспроизводимой логикой.
- `main.py`, `to_parq.py`, `slim_parq.py`, `to_h3_month.py` - старые совместимые entrypoint-скрипты. Они сохранены и теперь делегируют в пакет.
- `logreg.ipynb`, `temp.ipynb` - исследовательские ноутбуки.
- `dataset.json` - исходный большой JSON, локальный артефакт.
- `parquet_output/` - локальные parquet-артефакты пайплайна.

## Пайплайн

1. Построить схему исходного JSON:

```bash
python main.py
```

или:

```bash
python -m road_accident_risk.schema_inference
```

2. Разложить `dataset.json` в нормализованные parquet-таблицы:

```bash
python to_parq.py
```

Результаты:

- `parquet_output/crashes.parquet`
- `parquet_output/vehicles.parquet`
- `parquet_output/persons.parquet`

3. Собрать компактную таблицу ДТП:

```bash
python slim_parq.py
```

Результат:

- `parquet_output/crashes_slim.parquet`

4. Собрать H3 x month panel:

```bash
python to_h3_month.py
```

По умолчанию используется bbox Саратова и H3 resolution 9. Для более честных negative examples можно включить все H3-ячейки bbox:

```bash
python to_h3_month.py --include-all-h3-cells-in-bbox
```

5. Запустить baseline логистической регрессии:

```bash
python scripts/run_logreg_baseline.py
```

или после установки пакета:

```bash
road-accidents-logreg
```

## Установка

Для текущей локальной работы достаточно существующего `venv`. Для новой среды:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Для установки пакета в editable-режиме:

```bash
python -m pip install -e ".[notebook]"
```

## Важные детали данных

В исходнике scalar-поля `LAT` и `LNG` перепутаны: `LAT` содержит longitude, `LNG` содержит latitude. В `slim`-этапе это учтено так:

- `lat = POINT_LAT`, fallback на `LNG`
- `lng = POINT_LNG`, fallback на `LAT`

Поэтому downstream-код должен использовать `crashes_slim.parquet`, а не напрямую `LAT/LNG` из `crashes.parquet`.
