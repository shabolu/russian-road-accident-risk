from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h3
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PROJECT_ROOT / "parquet_output" / "h3_month_panel.parquet"
DEFAULT_ROAD_FEATURES = PROJECT_ROOT / "parquet_output" / "h3_road_features.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "h3_hex_map.html"


MONTH_COLUMNS = [
    "h3",
    "month",
    "accident_count",
    "target_count_next_month",
    "target_had_accident_next_month",
]

ROAD_COLUMNS = [
    "h3",
    "road_length_total_m",
    "major_road_length_m",
    "major_road_share",
    "maxspeed_mean_weighted",
    "lanes_mean_weighted",
    "lane_meters_estimate",
    "junction_count",
    "minor_to_major_junction_count",
    "traffic_signal_count",
    "crossing_count",
    "stop_count",
    "give_way_count",
    "traffic_calming_count",
]


def cell_boundary_lng_lat(cell: str) -> list[list[float]]:
    boundary = h3.cell_to_boundary(cell)
    coords = [[lng, lat] for lat, lng in boundary]
    coords.append(coords[0])
    return coords


def cell_boundary_lat_lng(cell: str) -> list[list[float]]:
    return [[round(float(lat), 6), round(float(lng), 6)] for lat, lng in h3.cell_to_boundary(cell)]


def cell_center(cell: str) -> tuple[float, float]:
    if hasattr(h3, "cell_to_latlng"):
        lat, lng = h3.cell_to_latlng(cell)
        return float(lat), float(lng)
    lat, lng = h3.h3_to_geo(cell)
    return float(lat), float(lng)


def safe_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def color_for_total(total: int, max_total: int) -> str:
    if total <= 0 or max_total <= 0:
        return "#eef2f4"

    value = math.log1p(total) / math.log1p(max_total)
    stops = [
        (0.00, (238, 242, 244)),
        (0.18, (186, 216, 220)),
        (0.38, (112, 175, 178)),
        (0.60, (238, 177, 82)),
        (0.82, (220, 106, 68)),
        (1.00, (154, 54, 57)),
    ]
    for idx in range(len(stops) - 1):
        left, left_rgb = stops[idx]
        right, right_rgb = stops[idx + 1]
        if left <= value <= right:
            t = (value - left) / max(right - left, 1e-9)
            rgb = [
                round(left_rgb[i] + (right_rgb[i] - left_rgb[i]) * t)
                for i in range(3)
            ]
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return "#9a3639"


def build_map_payload(panel_path: Path, road_features_path: Path) -> dict[str, Any]:
    panel = pd.read_parquet(panel_path, columns=MONTH_COLUMNS)
    panel["month"] = pd.to_datetime(panel["month"]).dt.strftime("%Y-%m")

    road_features = (
        pd.read_parquet(road_features_path, columns=ROAD_COLUMNS)
        if road_features_path.exists()
        else pd.DataFrame(columns=ROAD_COLUMNS)
    )
    road_value_columns = [column for column in ROAD_COLUMNS if column != "h3"]
    road_by_h3 = {
        row["h3"]: [round(safe_float(row.get(column)), 4) for column in road_value_columns]
        for row in road_features.to_dict(orient="records")
    }

    summary = (
        panel.groupby("h3", as_index=False)
        .agg(
            total_accidents=("accident_count", "sum"),
            active_months=("accident_count", lambda s: int((s > 0).sum())),
            target_positive_months=("target_had_accident_next_month", "sum"),
            max_month_accidents=("accident_count", "max"),
        )
    )

    monthly_by_h3: dict[str, list[list[Any]]] = {}
    for cell, group in panel.groupby("h3", sort=False):
        monthly_by_h3[cell] = [
            [
                row["month"],
                int(row["accident_count"]),
                int(row["target_count_next_month"]),
                int(bool(row["target_had_accident_next_month"])),
            ]
            for row in group.to_dict(orient="records")
        ]

    max_total = int(summary["total_accidents"].max()) if len(summary) else 0
    cells = []
    center_lats = []
    center_lngs = []

    for row in summary.to_dict(orient="records"):
        cell = row["h3"]
        lat, lng = cell_center(cell)
        center_lats.append(lat)
        center_lngs.append(lng)
        total = int(row["total_accidents"])
        cells.append([
            cell,
            cell_boundary_lat_lng(cell),
            [
                total,
                int(row["active_months"]),
                int(row["target_positive_months"]),
                int(row["max_month_accidents"]),
            ],
            road_by_h3.get(cell, []),
            monthly_by_h3[cell],
        ])

    return {
        "generated_from": {
            "panel": str(panel_path),
            "road_features": str(road_features_path),
        },
        "center": [
            sum(center_lats) / len(center_lats),
            sum(center_lngs) / len(center_lngs),
        ],
        "max_total_accidents": max_total,
        "cell_count": len(cells),
        "road_columns": road_value_columns,
        "cells": cells,
    }


def render_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>H3 карта ДТП</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1d2428;
      background: #f6f7f8;
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 420px;
      height: 100%;
      width: 100%;
    }}
    #map {{
      height: 100%;
      min-width: 0;
    }}
    .side {{
      border-left: 1px solid #d8dee2;
      background: #ffffff;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}
    .header {{
      padding: 16px 18px 12px;
      border-bottom: 1px solid #e1e6e9;
    }}
    .title {{
      font-size: 16px;
      font-weight: 700;
      margin: 0 0 6px;
    }}
    .sub {{
      font-size: 12px;
      color: #65727a;
      margin: 0;
      line-height: 1.4;
    }}
    .content {{
      padding: 14px 18px;
      overflow: auto;
    }}
    .h3id {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      word-break: break-all;
      color: #44515a;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 14px 0;
    }}
    .metric {{
      border: 1px solid #e1e6e9;
      border-radius: 6px;
      padding: 9px 10px;
      background: #fbfcfc;
    }}
    .metric .label {{
      font-size: 11px;
      color: #65727a;
      margin-bottom: 4px;
    }}
    .metric .value {{
      font-size: 15px;
      font-weight: 700;
    }}
    .section-title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
      color: #66737b;
      margin: 16px 0 8px;
      font-weight: 700;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #e5eaed;
      padding: 6px 5px;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #ffffff;
      color: #596870;
      z-index: 1;
    }}
    .months {{
      max-height: 420px;
      overflow: auto;
      border: 1px solid #e1e6e9;
      border-radius: 6px;
    }}
    .empty {{
      color: #65727a;
      font-size: 13px;
      line-height: 1.45;
    }}
    .legend {{
      position: absolute;
      left: 14px;
      bottom: 18px;
      z-index: 600;
      background: rgba(255,255,255,0.94);
      border: 1px solid #d8dee2;
      border-radius: 6px;
      padding: 9px 10px;
      font-size: 12px;
      box-shadow: 0 4px 14px rgba(30, 42, 50, 0.12);
    }}
    .swatches {{
      display: flex;
      gap: 3px;
      margin-top: 6px;
    }}
    .swatch {{
      width: 28px;
      height: 10px;
      border-radius: 2px;
    }}
    @media (max-width: 900px) {{
      .app {{
        grid-template-columns: 1fr;
        grid-template-rows: 58vh 42vh;
      }}
      .side {{
        border-left: 0;
        border-top: 1px solid #d8dee2;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <div id="map"></div>
    <aside class="side">
      <div class="header">
        <p class="title">H3 карта ДТП</p>
        <p class="sub">Гексагоны окрашены по сумме ДТП за весь период. Ячеек: {payload["cell_count"]:,}. Max ДТП в ячейке: {payload["max_total_accidents"]}.</p>
      </div>
      <div class="content" id="details">
        <p class="empty">Выберите H3-гексагон на карте.</p>
      </div>
    </aside>
  </div>
  <div class="legend">
    <div>Сумма ДТП</div>
    <div class="swatches">
      <div class="swatch" style="background:#eef2f4"></div>
      <div class="swatch" style="background:#bad8dc"></div>
      <div class="swatch" style="background:#70afb2"></div>
      <div class="swatch" style="background:#eeb152"></div>
      <div class="swatch" style="background:#dc6a44"></div>
      <div class="swatch" style="background:#9a3639"></div>
    </div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const DATA = {data_json};
    const ROAD_INDEX = Object.fromEntries(DATA.road_columns.map((name, index) => [name, index]));
    const map = L.map('map', {{ zoomControl: true }}).setView(DATA.center, 11);
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    let selectedLayer = null;

    function fmtNumber(value, digits = 0) {{
      const number = Number(value || 0);
      return number.toLocaleString('ru-RU', {{ maximumFractionDigits: digits }});
    }}

    function metric(label, value) {{
      return `<div class="metric"><div class="label">${{label}}</div><div class="value">${{value}}</div></div>`;
    }}

    function roadValue(road, key) {{
      const index = ROAD_INDEX[key];
      return index === undefined ? 0 : Number(road[index] || 0);
    }}

    function colorForTotal(total) {{
      const maxTotal = DATA.max_total_accidents || 0;
      if (total <= 0 || maxTotal <= 0) return '#eef2f4';
      const value = Math.log1p(total) / Math.log1p(maxTotal);
      const stops = [
        [0.00, [238, 242, 244]],
        [0.18, [186, 216, 220]],
        [0.38, [112, 175, 178]],
        [0.60, [238, 177, 82]],
        [0.82, [220, 106, 68]],
        [1.00, [154, 54, 57]]
      ];
      for (let i = 0; i < stops.length - 1; i++) {{
        const [left, leftRgb] = stops[i];
        const [right, rightRgb] = stops[i + 1];
        if (value >= left && value <= right) {{
          const t = (value - left) / Math.max(right - left, 1e-9);
          const rgb = leftRgb.map((v, idx) => Math.round(v + (rightRgb[idx] - v) * t));
          return '#' + rgb.map(v => v.toString(16).padStart(2, '0')).join('');
        }}
      }}
      return '#9a3639';
    }}

    function roadRows(road) {{
      const rows = [
        ['Длина дорог, м', fmtNumber(roadValue(road, 'road_length_total_m'), 1)],
        ['Крупные дороги, м', fmtNumber(roadValue(road, 'major_road_length_m'), 1)],
        ['Доля крупных дорог', `${{fmtNumber(roadValue(road, 'major_road_share') * 100, 1)}}%`],
        ['Средняя скорость', fmtNumber(roadValue(road, 'maxspeed_mean_weighted'), 1)],
        ['Среднее число полос', fmtNumber(roadValue(road, 'lanes_mean_weighted'), 2)],
        ['Lane-meters', fmtNumber(roadValue(road, 'lane_meters_estimate'), 1)],
        ['Перекрестки', fmtNumber(roadValue(road, 'junction_count'))],
        ['Примыкания minor→major', fmtNumber(roadValue(road, 'minor_to_major_junction_count'))],
        ['Светофоры', fmtNumber(roadValue(road, 'traffic_signal_count'))],
        ['Переходы', fmtNumber(roadValue(road, 'crossing_count'))]
      ];
      return rows.map(([k, v]) => `<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join('');
    }}

    function monthRows(months) {{
      return months.map(row => `
        <tr>
          <td>${{row[0]}}</td>
          <td>${{fmtNumber(row[1])}}</td>
          <td>${{fmtNumber(row[2])}}</td>
          <td>${{row[3] ? 'да' : 'нет'}}</td>
        </tr>
      `).join('');
    }}

    function renderDetails(props) {{
      const road = props.road || [];
      document.getElementById('details').innerHTML = `
        <div class="h3id">${{props.h3}}</div>
        <div class="metrics">
          ${{metric('ДТП всего', fmtNumber(props.total_accidents))}}
          ${{metric('Месяцев с ДТП', fmtNumber(props.active_months))}}
          ${{metric('Max за месяц', fmtNumber(props.max_month_accidents))}}
          ${{metric('Target+ месяцев', fmtNumber(props.target_positive_months))}}
        </div>

        <div class="section-title">Дорожные признаки</div>
        <table>
          <tbody>${{roadRows(road)}}</tbody>
        </table>

        <div class="section-title">По месяцам</div>
        <div class="months">
          <table>
            <thead>
              <tr>
                <th>Месяц</th>
                <th>ДТП</th>
                <th>ДТП след. мес.</th>
                <th>Target</th>
              </tr>
            </thead>
            <tbody>${{monthRows(props.months || [])}}</tbody>
          </table>
        </div>
      `;
    }}

    function propsFromCell(cell) {{
      const [h3, boundary, summary, road, months] = cell;
      return {{
        h3,
        boundary,
        total_accidents: summary[0],
        active_months: summary[1],
        target_positive_months: summary[2],
        max_month_accidents: summary[3],
        road,
        months
      }};
    }}

    function baseStyle(props) {{
      return {{
        color: '#47565e',
        weight: 0.55,
        opacity: 0.55,
        fillColor: colorForTotal(props.total_accidents),
        fillOpacity: props.total_accidents > 0 ? 0.72 : 0.30
      }};
    }}

    const group = L.featureGroup();
    for (const cell of DATA.cells) {{
      const props = propsFromCell(cell);
      const polygon = L.polygon(props.boundary, baseStyle(props));
      polygon.__props = props;
      polygon.on('mouseover', () => {{
        if (polygon !== selectedLayer) polygon.setStyle({{ weight: 1.4, opacity: 0.9 }});
      }});
      polygon.on('mouseout', () => {{
        if (polygon !== selectedLayer) polygon.setStyle(baseStyle(props));
      }});
      polygon.on('click', () => {{
        if (selectedLayer) selectedLayer.setStyle(baseStyle(selectedLayer.__props));
        selectedLayer = polygon;
        polygon.setStyle({{ weight: 2.2, color: '#111820', opacity: 1, fillOpacity: 0.9 }});
        renderDetails(props);
      }});
      polygon.addTo(group);
    }}
    group.addTo(map);
    map.fitBounds(group.getBounds(), {{ padding: [16, 16] }});
  </script>
</body>
</html>
"""


def build_html(panel_path: Path, road_features_path: Path, output_path: Path) -> Path:
    payload = build_map_payload(panel_path, road_features_path)
    html = render_html(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved map to: {output_path}")
    print(f"Cells: {payload['cell_count']:,}")
    print(f"HTML size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a temporary HTML H3 accident map.")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--road-features", type=Path, default=DEFAULT_ROAD_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_html(args.panel, args.road_features, args.output)


if __name__ == "__main__":
    main()
