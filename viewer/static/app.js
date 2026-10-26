const map = L.map("map", { zoomControl: true }).setView([51.53, 46.0], 11);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let group = L.featureGroup().addTo(map);
let intersectionGroup = L.featureGroup().addTo(map);
let roadGroup = L.featureGroup().addTo(map);
let selectedLayer = null;
let currentCells = [];
let currentMetric = document.getElementById("metric").value;
let currentMaxMetric = 0;
let focused = false;
let savedView = null;

const metricNames = {
  total_accidents: "ДТП всего",
  road_length_total_m: "Длина дорог",
  junction_count: "Перекрестки",
  crossing_count: "Переходы",
  catboost_road_only_4_0_count_next_3m_score_latest: "CatBoost 4.0: кол-во ДТП в след. 3м",
  catboost_road_only_4_0_count_next_6m_score_latest: "CatBoost 4.0: кол-во ДТП в след. 6м",
  catboost_road_only_4_0_binary_next_3m_score_latest: "CatBoost 4.0: ДТП в след. 3м",
  catboost_road_only_4_0_binary_next_6m_weight_none_score_latest: "CatBoost 4.0: ДТП в след. 6м без весов",
  road_only_4_0_binary_next_6m_xgboost_score_latest: "XGBoost 4.0: ДТП в след. 6м без весов",
  road_only_4_0_binary_next_6m_lightgbm_score_latest: "LightGBM 4.0: ДТП в след. 6м без весов",
  road_only_4_0_binary_next_6m_hist_gradient_boosting_score_latest: "HGB 4.0: ДТП в след. 6м без весов",
  road_only_4_0_binary_next_6m_random_forest_score_latest: "RandomForest 4.0: ДТП в след. 6м без весов",
  catboost_road_only_4_0_binary_next_6m_weight_2_score_latest: "CatBoost 4.0: ДТП в след. 6м вес 2",
  catboost_road_only_4_0_binary_next_6m_weight_4_score_latest: "CatBoost 4.0: ДТП в след. 6м вес 4",
  catboost_road_only_4_0_binary_next_6m_weight_6_score_latest: "CatBoost 4.0: ДТП в след. 6м вес 6",
  catboost_road_only_4_0_binary_next_6m_weight_balanced_score_latest: "CatBoost 4.0: ДТП в след. 6м Balanced",
  catboost_road_only_4_0_grade2_score_latest: "CatBoost 4.0 + уклон (2 признака)",
  catboost_road_only_4_0_grade2_delta_score_latest: "Δ logit: уклон − база 4.0",
};

const ROAD_ONLY_3_5_FEATURE_ORDER = [
  "road_length_total_m",
  "road_way_count",
  "road_segment_h3_count",
  "major_road_length_m",
  "oneway_length_m",
  "lit_road_length_m",
  "bridge_length_m",
  "bad_surface_length_m",
  "asphalt_surface_length_m",
  "lanes_mean_weighted",
  "lanes_max",
  "lane_meters_estimate",
  "lanes_tagged_share",
  "major_road_share",
  "oneway_share",
  "lit_share",
  "bridge_share",
  "bad_surface_share",
  "asphalt_surface_share",
  "traffic_signal_count",
  "crossing_count",
  "traffic_calming_count",
  "junction_density_per_km",
  "minor_to_major_junction_density_per_km",
  "junction_count",
  "junction_degree_mean",
  "junction_degree_max",
  "three_way_junction_count",
  "four_plus_way_junction_count",
  "major_junction_count",
  "minor_to_major_junction_count",
  "legacy_major_minor_junction_count",
  "class_delta_junction_count",
  "class_delta_mean",
  "class_delta_max",
  "class_delta_sum",
  "signalized_class_delta_junction_count",
  "unsignalized_class_delta_junction_count",
  "unsignalized_high_delta_junction_count",
  "class_delta_junction_density_per_km",
  "unsignalized_class_delta_junction_density_per_km",
  "road_share_living_street",
  "road_share_primary",
  "road_share_residential",
  "road_share_secondary",
  "road_share_service",
  "road_share_tertiary",
  "road_share_trunk",
  "road_share_unclassified",
  "maxspeed_mean_weighted",
  "maxspeed_max",
  "maxspeed_tagged_share",
  "maxspeed_is_imputed_share",
  "crossing_unsignalized_count",
  "public_transport_stop_count",
  "bus_stop_count",
  "tram_stop_count",
  "railway_crossing_count",
  "traffic_signal_density_per_km",
  "crossing_density_per_km",
  "crossing_unsignalized_density_per_km",
  "priority_sign_density_per_km",
  "traffic_calming_density_per_km",
  "public_transport_stop_density_per_km",
  "railway_crossing_density_per_km",
  "motorway_junction_density_per_km",
  "crossing_unsignalized_share",
  "bus_stop_share",
  "tram_stop_share",
  "road_curvature_mean_weighted",
  "road_curvature_max",
  "turn_event_count",
  "sharp_turn_count",
  "road_segment_count",
  "road_segment_length_mean",
  "road_segment_length_median",
  "short_segment_count",
  "road_orientation_entropy",
  "dead_end_count",
  "curvature_deg_per_km",
  "turn_event_density_per_km",
  "sharp_turn_density_per_km",
  "sharp_turn_share",
  "road_segment_density_per_km",
  "short_segment_share",
  "dead_end_density_per_km",
];

const ROAD_ONLY_4_0_EXCLUDED_FEATURES = new Set([
  "road_segment_h3_count",
  "maxspeed_is_imputed_share",
  "unsignalized_class_delta_junction_count",
  "unsignalized_class_delta_junction_density_per_km",
  "traffic_calming_count",
  "bridge_share",
  "tram_stop_share",
  "railway_crossing_count",
  "traffic_signal_density_per_km",
  "crossing_density_per_km",
  "crossing_unsignalized_count",
  "crossing_unsignalized_density_per_km",
  "crossing_unsignalized_share",
  "lit_share",
  "public_transport_stop_density_per_km",
  "bus_stop_count",
  "bus_stop_share",
  "tram_stop_count",
  "oneway_length_m",
  "bad_surface_length_m",
  "major_road_length_m",
  "lanes_max",
  "class_delta_sum",
  "class_delta_junction_count",
  "turn_event_count",
  "short_segment_count",
]);

const ROAD_ONLY_4_0_FEATURE_ORDER = ROAD_ONLY_3_5_FEATURE_ORDER
  .filter((feature) => !ROAD_ONLY_4_0_EXCLUDED_FEATURES.has(feature));

const modelConfigs = {
  catboost_road_only_4_0_count_next_3m_score_latest: {
    label: "4.0 кол-во ДТП в след. 3м",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_count_next_3m_score",
    rankColumn: "catboost_road_only_4_0_count_next_3m_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_count_next_3m_score_latest",
    scoreFormat: "count",
  },
  catboost_road_only_4_0_count_next_6m_score_latest: {
    label: "4.0 кол-во ДТП в след. 6м",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_count_next_6m_score",
    rankColumn: "catboost_road_only_4_0_count_next_6m_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_count_next_6m_score_latest",
    scoreFormat: "count",
  },
  catboost_road_only_4_0_binary_next_3m_score_latest: {
    label: "4.0 ДТП в след. 3м",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_binary_next_3m_score",
    rankColumn: "catboost_road_only_4_0_binary_next_3m_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_binary_next_3m_score_latest",
  },
  catboost_road_only_4_0_binary_next_6m_weight_none_score_latest: {
    label: "4.0 ДТП 6м без весов",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_none_score",
    rankColumn: "catboost_road_only_4_0_binary_next_6m_weight_none_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_none_score_latest",
  },
  road_only_4_0_binary_next_6m_xgboost_score_latest: {
    label: "4.0 ДТП 6м XGBoost",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "road_only_4_0_binary_next_6m_xgboost_score",
    rankColumn: "road_only_4_0_binary_next_6m_xgboost_rank_in_month",
    latestScoreColumn: "road_only_4_0_binary_next_6m_xgboost_score_latest",
  },
  road_only_4_0_binary_next_6m_lightgbm_score_latest: {
    label: "4.0 ДТП 6м LightGBM",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "road_only_4_0_binary_next_6m_lightgbm_score",
    rankColumn: "road_only_4_0_binary_next_6m_lightgbm_rank_in_month",
    latestScoreColumn: "road_only_4_0_binary_next_6m_lightgbm_score_latest",
  },
  road_only_4_0_binary_next_6m_hist_gradient_boosting_score_latest: {
    label: "4.0 ДТП 6м HGB",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "road_only_4_0_binary_next_6m_hist_gradient_boosting_score",
    rankColumn: "road_only_4_0_binary_next_6m_hist_gradient_boosting_rank_in_month",
    latestScoreColumn: "road_only_4_0_binary_next_6m_hist_gradient_boosting_score_latest",
  },
  road_only_4_0_binary_next_6m_random_forest_score_latest: {
    label: "4.0 ДТП 6м RandomForest",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "road_only_4_0_binary_next_6m_random_forest_score",
    rankColumn: "road_only_4_0_binary_next_6m_random_forest_rank_in_month",
    latestScoreColumn: "road_only_4_0_binary_next_6m_random_forest_score_latest",
  },
  catboost_road_only_4_0_binary_next_6m_weight_2_score_latest: {
    label: "4.0 ДТП 6м вес 2",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_2_score",
    rankColumn: "catboost_road_only_4_0_binary_next_6m_weight_2_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_2_score_latest",
  },
  catboost_road_only_4_0_binary_next_6m_weight_4_score_latest: {
    label: "4.0 ДТП 6м вес 4",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_4_score",
    rankColumn: "catboost_road_only_4_0_binary_next_6m_weight_4_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_4_score_latest",
  },
  catboost_road_only_4_0_binary_next_6m_weight_6_score_latest: {
    label: "4.0 ДТП 6м вес 6",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_6_score",
    rankColumn: "catboost_road_only_4_0_binary_next_6m_weight_6_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_6_score_latest",
  },
  catboost_road_only_4_0_binary_next_6m_weight_balanced_score_latest: {
    label: "4.0 ДТП 6м Balanced",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_balanced_score",
    rankColumn: "catboost_road_only_4_0_binary_next_6m_weight_balanced_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_binary_next_6m_weight_balanced_score_latest",
  },
  catboost_road_only_4_0_grade2_score_latest: {
    label: "4.0 + уклон (2)",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_grade2_score",
    rankColumn: "catboost_road_only_4_0_grade2_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_grade2_score_latest",
  },
  catboost_road_only_4_0_grade2_delta_score_latest: {
    label: "Δ logit уклон − база",
    featureOrder: ROAD_ONLY_4_0_FEATURE_ORDER,
    scoreColumn: "catboost_road_only_4_0_grade2_delta_score",
    rankColumn: "catboost_road_only_4_0_grade2_delta_rank_in_month",
    latestScoreColumn: "catboost_road_only_4_0_grade2_delta_score_latest",
  },
};

function currentModelConfig(metric = currentMetric) {
  return modelConfigs[metric] || null;
}

function fmtNumber(value, digits = 0) {
  const number = Number(value || 0);
  return number.toLocaleString("ru-RU", { maximumFractionDigits: digits });
}

function fmtPercent(value, digits = 1) {
  return `${fmtNumber(Number(value || 0) * 100, digits)}%`;
}

function formatModelScore(config, value) {
  if (config.scoreFormat === "count") return fmtNumber(value, 3);
  return fmtPercent(value, 1);
}

function colorForValue(value, maxValue) {
  if (!value || !maxValue) return "#eef2f4";
  const ratio = Math.log1p(value) / Math.log1p(maxValue);
  const stops = [
    [0.0, [238, 242, 244]],
    [0.18, [186, 216, 220]],
    [0.38, [112, 175, 178]],
    [0.6, [238, 177, 82]],
    [0.82, [220, 106, 68]],
    [1.0, [154, 54, 57]],
  ];

  for (let i = 0; i < stops.length - 1; i += 1) {
    const [left, leftRgb] = stops[i];
    const [right, rightRgb] = stops[i + 1];
    if (ratio >= left && ratio <= right) {
      const t = (ratio - left) / Math.max(right - left, 1e-9);
      const rgb = leftRgb.map((v, idx) => Math.round(v + (rightRgb[idx] - v) * t));
      return `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
    }
  }
  return "#9a3639";
}

function metricCard(label, value) {
  return `<div class="metric-card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

function currentMetricValueDisplay(summary, cell, metric = currentMetric) {
  const config = modelConfigs[metric];
  if (config) {
    return {
      label: config.label,
      value: formatModelScore(config, summary[config.latestScoreColumn]),
    };
  }

  const label = metricNames[metric] || metric;
  const value = cell.metric_value;
  if (metric.includes("rate") || metric.includes("share")) {
    return { label, value: fmtPercent(value, 1) };
  }
  return { label, value: fmtNumber(value, 1) };
}

function formatFeatureValue(key, value) {
  if (key.includes("share")) return fmtNumber(value, 4);
  if (key.includes("entropy")) return fmtNumber(value, 3);
  if (key.includes("density") || key.includes("mean") || key.includes("curvature")) return fmtNumber(value, 2);
  if (
    key.includes("length") ||
    key.endsWith("_m") ||
    key === "lane_meters_estimate" ||
    key === "maxspeed_mean_weighted" ||
    key === "maxspeed_max"
  ) {
    return fmtNumber(value, 1);
  }
  return fmtNumber(value);
}

function roadRows(road, metric = currentMetric) {
  const config = modelConfigs[metric];
  const featureOrder = config ? config.featureOrder : Object.keys(road);
  return featureOrder
    .filter((key) => Object.prototype.hasOwnProperty.call(road, key))
    .map((key) => `<tr><td>${key}</td><td>${formatFeatureValue(key, road[key])}</td></tr>`)
    .join("");
}

function monthRows(months, config) {
  return months.map((row) => `
    <tr>
      <td>${row.month}</td>
      <td>${fmtNumber(row.accident_count)}</td>
      ${config ? `<td>${formatModelScore(config, row[config.scoreColumn])}</td>` : ""}
      ${config ? `<td>${fmtNumber(row[config.rankColumn])}</td>` : ""}
      <td>${fmtNumber(row.accidents_prev_12m)}</td>
      <td>${fmtNumber(row.target_count_next_month)}</td>
      <td>${row.target_had_accident_next_month ? "да" : "нет"}</td>
    </tr>
  `).join("");
}

function setMapLocked(locked) {
  const actions = [
    map.dragging,
    map.touchZoom,
    map.doubleClickZoom,
    map.scrollWheelZoom,
    map.boxZoom,
    map.keyboard,
  ];
  for (const action of actions) {
    if (!action) continue;
    if (locked) action.disable();
    else action.enable();
  }
  document.body.classList.toggle("focus-mode", locked);
}

function neutralizeCells(selected) {
  for (const layer of group.getLayers()) {
    if (layer === selected) {
      layer.setStyle({
        color: "#111820",
        weight: 2.2,
        opacity: 1,
        fillColor: "#f8fafb",
        fillOpacity: 0.06,
      });
    } else {
      layer.setStyle({
        color: "#b8c2c7",
        weight: 0.35,
        opacity: 0.25,
        fillColor: "#eef2f4",
        fillOpacity: 0.08,
      });
    }
  }
}

function restoreCells() {
  for (const layer of group.getLayers()) {
    layer.setStyle(styleForCell(layer.__cell, currentMaxMetric));
  }
  if (selectedLayer) {
    selectedLayer.setStyle({ weight: 2.2, color: "#111820", opacity: 1, fillOpacity: 0.9 });
  }
}

function intersectionColor(item) {
  if (item.is_minor_to_major_junction) return "#d1493f";
  if (item.is_major_junction) return "#8f45a3";
  if (item.is_class_change) return "#da8a24";
  return "#176f7d";
}

const roadTypeColors = {
  motorway: "#bf1d2d",
  motorway_link: "#df5a45",
  trunk: "#d75f00",
  trunk_link: "#ef8f35",
  primary: "#f2b134",
  primary_link: "#f6c15b",
  secondary: "#2b8a6e",
  secondary_link: "#56aa85",
  tertiary: "#2374ab",
  tertiary_link: "#5a98c8",
  unclassified: "#6f7d86",
  residential: "#8a6fbd",
  living_street: "#b2769a",
  service: "#8c99a3",
};

function roadColor(highway) {
  return roadTypeColors[highway] || "#303840";
}

function roadDisplayName(road) {
  if (road.name) return road.name;
  if (road.ref) return road.ref;
  return `osm ${road.osm_id}`;
}

function intersectionRows(items) {
  if (!items.length) {
    return `<tr><td colspan="6">В этом H3 перекрестки не найдены.</td></tr>`;
  }
  return items.map((item) => `
    <tr>
      <td>${item.node_id}</td>
      <td>${item.junction_way_count}</td>
      <td>${fmtNumber(item.class_rank_delta)}</td>
      <td>${item.has_traffic_signal_nearby ? "да" : "нет"}</td>
      <td>${item.connected_highway_classes || ""}</td>
      <td>${item.is_minor_to_major_junction ? "да" : "нет"}</td>
    </tr>
  `).join("");
}

function roadFocusRows(items) {
  if (!items.length) {
    return `<tr><td colspan="4">В этом H3 дороги не найдены.</td></tr>`;
  }
  return items.map((item) => `
    <tr>
      <td><span class="road-chip" style="background:${roadColor(item.highway)}"></span>${item.highway}</td>
      <td>${roadDisplayName(item)}</td>
      <td>${fmtNumber(item.length_m, 1)}</td>
      <td>${item.surface || ""}</td>
    </tr>
  `).join("");
}

async function enterIntersectionFocus(cell, detail) {
  if (!selectedLayer || focused) return;
  const response = await fetch(`/api/cell/${cell.h3}/intersections`);
  if (!response.ok) return;
  const payload = await response.json();
  const intersections = payload.intersections || [];

  focused = true;
  savedView = {
    center: map.getCenter(),
    zoom: map.getZoom(),
  };
  document.getElementById("cancel-focus").hidden = false;
  document.getElementById("metric").disabled = true;
  document.getElementById("reload").disabled = true;
  document.getElementById("status").textContent = `перекрестков: ${intersections.length}`;
  setMapLocked(true);
  neutralizeCells(selectedLayer);
  intersectionGroup.clearLayers();
  roadGroup.clearLayers();

  map.fitBounds(selectedLayer.getBounds(), { padding: [80, 80], animate: true });

  for (const item of intersections) {
    const marker = L.circleMarker([item.lat, item.lng], {
      radius: item.is_minor_to_major_junction ? 6 : 4.5,
      color: "#ffffff",
      weight: 1.4,
      fillColor: intersectionColor(item),
      fillOpacity: 0.95,
    });
    marker.bindTooltip(
      `node ${item.node_id}<br>${item.connected_highway_classes || ""}<br>class_delta: ${fmtNumber(item.class_rank_delta)}<br>signal nearby: ${item.has_traffic_signal_nearby ? "да" : "нет"}<br>minor_to_major: ${item.is_minor_to_major_junction ? "да" : "нет"}`,
      { sticky: true }
    );
    marker.addTo(intersectionGroup);
  }

  const oldDetails = document.getElementById("details").innerHTML;
  const listHtml = `
    <div class="section-title">Перекрестки в H3</div>
    <p class="focus-note">Карта заблокирована. Нажмите “Отмена”, чтобы вернуться.</p>
    <div class="intersection-list">
      <table>
        <thead>
          <tr>
            <th>node</th>
            <th>ways</th>
            <th>delta</th>
            <th>signal</th>
            <th>классы</th>
            <th>minor→major</th>
          </tr>
        </thead>
        <tbody>${intersectionRows(intersections)}</tbody>
      </table>
    </div>
  `;
  document.getElementById("details").innerHTML = oldDetails + listHtml;
}

async function enterRoadFocus(cell) {
  if (!selectedLayer || focused) return;
  const response = await fetch(`/api/cell/${cell.h3}/roads`);
  if (!response.ok) return;
  const payload = await response.json();
  const roads = payload.roads || [];

  focused = true;
  savedView = {
    center: map.getCenter(),
    zoom: map.getZoom(),
  };
  document.getElementById("cancel-focus").hidden = false;
  document.getElementById("metric").disabled = true;
  document.getElementById("reload").disabled = true;
  document.getElementById("status").textContent =
    `дорог: ${roads.length}, ${fmtNumber(payload.total_length_m, 1)} м`;
  setMapLocked(true);
  neutralizeCells(selectedLayer);
  intersectionGroup.clearLayers();
  roadGroup.clearLayers();

  map.fitBounds(selectedLayer.getBounds(), { padding: [80, 80], animate: true });

  for (const road of roads) {
    for (const part of road.parts || []) {
      const line = L.polyline(part.coords, {
        color: roadColor(road.highway),
        weight: road.is_major ? 5.5 : 4,
        opacity: 0.95,
        lineCap: "round",
        lineJoin: "round",
        dashArray: road.is_link ? "7 5" : null,
      });
      line.bindTooltip(
        `${road.highway}<br>${roadDisplayName(road)}<br>в H3: ${fmtNumber(road.length_m, 1)} м<br>кусок: ${fmtNumber(part.length_m, 1)} м`,
        { sticky: true }
      );
      line.addTo(roadGroup);
    }
  }

  const oldDetails = document.getElementById("details").innerHTML;
  const listHtml = `
    <div class="section-title">Дороги в H3</div>
    <p class="focus-note">Цвет линии соответствует OSM highway. Наведите на линию, чтобы увидеть длину.</p>
    <div class="road-legend">
      ${Object.entries(roadTypeColors).map(([type, color]) => `
        <span><i style="background:${color}"></i>${type}</span>
      `).join("")}
    </div>
    <div class="road-list">
      <table>
        <thead>
          <tr>
            <th>тип</th>
            <th>имя/ref</th>
            <th>м в H3</th>
            <th>surface</th>
          </tr>
        </thead>
        <tbody>${roadFocusRows(roads)}</tbody>
      </table>
    </div>
  `;
  document.getElementById("details").innerHTML = oldDetails + listHtml;
}

function exitIntersectionFocus() {
  focused = false;
  intersectionGroup.clearLayers();
  roadGroup.clearLayers();
  document.getElementById("cancel-focus").hidden = true;
  document.getElementById("metric").disabled = false;
  document.getElementById("reload").disabled = false;
  setMapLocked(false);
  restoreCells();
  document.getElementById("status").textContent =
    `${fmtNumber(currentCells.length)} H3, max=${fmtNumber(currentMaxMetric, 1)}`;
  if (savedView) {
    map.setView(savedView.center, savedView.zoom, { animate: true });
  }
}

async function showCell(cell, metricForDetail = currentMetric) {
  const details = document.getElementById("details");
  details.innerHTML = `<p class="empty">Загрузка ${cell.h3}...</p>`;
  const response = await fetch(`/api/cell/${cell.h3}`);
  if (!response.ok) {
    details.innerHTML = `<p class="empty">Не удалось загрузить ячейку.</p>`;
    return;
  }
  const data = await response.json();
  const summary = cell.summary;
  const config = currentModelConfig(metricForDetail);
  const selectedMetric = currentMetricValueDisplay(summary, cell, metricForDetail);
  const modelHeader = config ? `<th>${config.label}</th><th>Rank</th>` : "";
  details.innerHTML = `
    <div class="h3id">${data.h3}</div>
    <div class="metrics">
      ${metricCard("ДТП всего", fmtNumber(summary.total_accidents))}
      ${metricCard(selectedMetric.label, selectedMetric.value)}
      ${metricCard("Месяцев с ДТП", fmtNumber(summary.active_months))}
      ${metricCard("Target+ месяцев", fmtNumber(summary.target_positive_months))}
    </div>
    <div class="action-row">
      <button id="show-intersections">Показать перекрестки</button>
      <button id="show-roads">Показать дороги</button>
    </div>
    <div class="section-title">Дорожные признаки</div>
    <table><tbody>${roadRows(data.road_features, metricForDetail)}</tbody></table>
    <div class="section-title">По месяцам</div>
    <div class="months">
      <table>
        <thead>
          <tr>
            <th>Месяц</th>
            <th>ДТП</th>
            ${modelHeader}
            <th>ДТП за 12м</th>
            <th>ДТП след. мес.</th>
            <th>Target</th>
          </tr>
        </thead>
        <tbody>${monthRows(data.months, config)}</tbody>
      </table>
    </div>
  `;
  document.getElementById("show-intersections").addEventListener("click", () => {
    enterIntersectionFocus(cell, data);
  });
  document.getElementById("show-roads").addEventListener("click", () => {
    enterRoadFocus(cell);
  });
}

function styleForCell(cell, maxMetric) {
  return {
    color: "#47565e",
    weight: 0.55,
    opacity: 0.55,
    fillColor: colorForValue(cell.metric_value, maxMetric),
    fillOpacity: cell.metric_value > 0 ? 0.72 : 0.28,
  };
}

function renderCells(payload) {
  exitIntersectionFocus();
  group.clearLayers();
  selectedLayer = null;
  currentCells = payload.cells;
  currentMetric = payload.metric;
  currentMaxMetric = payload.max_metric;
  document.getElementById("legend-title").textContent = metricNames[payload.metric] || payload.metric;
  document.getElementById("details").innerHTML = `<p class="empty">Выберите гексагон на карте.</p>`;

  for (const cell of payload.cells) {
    const polygon = L.polygon(cell.boundary, styleForCell(cell, payload.max_metric));
    polygon.__cell = cell;
    polygon.on("mouseover", () => {
      if (polygon !== selectedLayer) polygon.setStyle({ weight: 1.4, opacity: 0.9 });
    });
    polygon.on("mouseout", () => {
      if (polygon !== selectedLayer) polygon.setStyle(styleForCell(cell, payload.max_metric));
    });
    polygon.on("click", () => {
      if (focused) return;
      if (selectedLayer) {
        selectedLayer.setStyle(styleForCell(selectedLayer.__cell, payload.max_metric));
      }
      selectedLayer = polygon;
      polygon.setStyle({ weight: 2.2, color: "#111820", opacity: 1, fillOpacity: 0.9 });
      showCell(cell, payload.metric);
    });
    polygon.addTo(group);
  }

  if (group.getLayers().length) {
    map.fitBounds(group.getBounds(), { padding: [16, 16] });
  }
}

async function loadMeta() {
  const response = await fetch("/api/meta");
  const meta = await response.json();
  document.getElementById("meta").textContent =
    `${fmtNumber(meta.h3_count)} H3, ${fmtNumber(meta.rows)} строк, период ${meta.min_month} - ${meta.max_month}`;
}

async function loadCells() {
  const metric = document.getElementById("metric").value;
  document.getElementById("status").textContent = "загрузка...";
  const response = await fetch(`/api/cells?metric=${encodeURIComponent(metric)}`);
  const payload = await response.json();
  renderCells(payload);
  document.getElementById("status").textContent =
    `${fmtNumber(payload.cells.length)} H3, max=${fmtNumber(payload.max_metric, 1)}`;
}

document.getElementById("reload").addEventListener("click", loadCells);
document.getElementById("metric").addEventListener("change", loadCells);
document.getElementById("cancel-focus").addEventListener("click", exitIntersectionFocus);

document.getElementById("metric").value = currentMetric;
loadMeta();
loadCells();
