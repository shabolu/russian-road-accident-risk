"""Выгрузка интенсивности движения из открытого WFS ФГИС СКДФ.

Слой skdf_open:lyr_road_conditions_traffic содержит участки дорог с геометрией
и полем traffic — среднесуточной интенсивностью в авт/сут.

Авторизация не нужна. Из-за рубежа работает.

Использование:
    python scripts/fetch_skdf_traffic.py --out parquet_output/skdf_traffic_saratov.geojson

Грабли, на которые я уже наступил:

1. KVP-параметр `bbox=` слой игнорирует (отдаёт единицы объектов). Фильтровать надо
   через CQL_FILTER.
2. Геометрия хранится в EPSG:3857, поэтому в `BBOX(...)` обязательно указывать CRS
   пятым аргументом: BBOX(geom, minLon, minLat, maxLon, maxLat, 'EPSG:4326').
   Без него фильтр молча вернёт 0.
3. `startIndex` без `sortBy` даёт HTTP 400 — "Cannot do natural order without a
   primary key". У слоя нет PK, порядок задаём вручную (sortBy=id).
4. `region_gid` — НЕ код субъекта РФ. Значение 22960 даёт 51 тыс. участков,
   размазанных по всей стране от Калининграда до Владивостока. Не использовать
   как географический фильтр.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request

WFS = "https://xn--d1aluo.xn--p1ai/api-geoserver/skdf_open/wfs"
LAYER = "skdf_open:lyr_road_conditions_traffic"
PAGE = 2000          # GeoServer отдаёт стабильно до ~2-5 тыс. за раз
PAUSE_S = 0.4        # вежливая пауза между страницами


def _get(params: dict, timeout: int = 120) -> bytes:
    url = f"{WFS}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "road-accident-risk/research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def count(cql: str) -> int:
    """resultType=hits возвращает только счётчик, без данных — дёшево."""
    xml = _get({
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": LAYER, "resultType": "hits", "CQL_FILTER": cql,
    }).decode("utf-8", "replace")
    for key in ('numberMatched="', 'numberOfFeatures="'):
        if key in xml:
            return int(xml.split(key, 1)[1].split('"', 1)[0])
    raise RuntimeError(f"не нашёл счётчик в ответе: {xml[:300]}")


def fetch_page(cql: str, start: int, n: int) -> list[dict]:
    raw = _get({
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeNames": LAYER, "outputFormat": "application/json",
        "srsName": "EPSG:4326",          # иначе отдаст в проекции сервера
        "CQL_FILTER": cql,
        "count": n, "startIndex": start,
        # без sortBy сервер отвечает 400: "Cannot do natural order without a
        # primary key" — у слоя нет PK, поэтому порядок задаём вручную
        "sortBy": "id",
    })
    return json.loads(raw)["features"]


def fetch_all(cql: str) -> list[dict]:
    total = count(cql)
    print(f"всего объектов по фильтру: {total:,}", file=sys.stderr)
    out: list[dict] = []
    start = 0
    while start < total:
        for attempt in range(4):        # сервер изредка отдаёт 404/500 под нагрузкой
            try:
                page = fetch_page(cql, start, PAGE)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  retry {attempt + 1} на startIndex={start}: {e}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        if not page:
            break
        out.extend(page)
        start += len(page)
        print(f"  {len(out):,} / {total:,}", file=sys.stderr)
        time.sleep(PAUSE_S)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    # bbox проекта: тот же, что в h3_month_panel.py — (lat_min, lng_min, lat_max, lng_max)
    ap.add_argument("--bbox", default="51.40,45.80,51.70,46.20",
                    help="lat_min,lng_min,lat_max,lng_max")
    ap.add_argument("--out", default="parquet_output/skdf_traffic.geojson")
    ap.add_argument("--only-with-traffic", action="store_true",
                    help="брать только участки с traffic > 0")
    args = ap.parse_args()

    lat1, lon1, lat2, lon2 = (float(x) for x in args.bbox.split(","))
    # CQL BBOX ждёт minLon, minLat, maxLon, maxLat + CRS
    cql = f"BBOX(geom,{lon1},{lat1},{lon2},{lat2},'EPSG:4326')"
    if args.only_with_traffic:
        cql += " AND traffic > 0"

    feats = fetch_all(cql)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)

    vals = [x["properties"].get("traffic") for x in feats]
    nonzero = [v for v in vals if v]
    print(
        f"\nсохранено: {args.out}\n"
        f"  объектов: {len(feats):,}\n"
        f"  с traffic > 0: {len(nonzero):,} ({len(nonzero) / max(len(feats), 1):.1%})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
