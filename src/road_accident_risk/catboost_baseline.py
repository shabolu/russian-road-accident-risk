from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score

from road_accident_risk.logreg_baseline import TARGET, TRAIN_END, precision_recall_at_k
from road_accident_risk.exposure_v6 import EXPOSURE_V6_FEATURES
from road_accident_risk.osm_road_features import JUNCTION_V2_FEATURES, MOTOR_ROAD_CLASSES, TAGS_V2_FEATURES
from road_accident_risk.paths import (
    H3_EXPOSURE_V6_FEATURES_PARQUET,
    H3_GEOMETRY_V2_FEATURES_PARQUET,
    H3_JUNCTION_V2_FEATURES_PARQUET,
    H3_MONTH_ROAD_FEATURES_PARQUET,
    H3_ROAD_GRADE_FEATURES_PARQUET,
    H3_SKDF_V2_FEATURES_PARQUET,
    H3_TAGS_V2_FEATURES_PARQUET,
    PROJECT_ROOT,
)
from road_accident_risk.skdf_v2 import SKDF_V2_ALL_FEATURES, SKDF_V2_FEATURES
from road_accident_risk.street_graph_v2 import GEOMETRY_V2_FEATURES


FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS = "road_without_type_lengths"
FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR = "road_without_type_lengths_no_calendar"
FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM = "road_without_type_lengths_no_calendar_slim"
FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP = (
    "road_without_type_lengths_no_calendar_slim_dedup"
)
# Бейзлайн с 2026-09-16: road-only 4.0 (59 признаков) + два признака продольного уклона
# из Copernicus DEM (см. EXPERIMENTS.md, «Road+Geo Baseline»). Признаки уклона живут в
# отдельном parquet (H3_ROAD_GRADE_FEATURES_PARQUET) и подтягиваются по h3.
FEATURE_SET_ROAD_GEO_BASELINE = "road_geo_baseline"
ROAD_GEO_FEATURES = ["road_grade_abs_mean_weighted", "grade_change_density_per_km"]

FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP_LOG1P = (
    "road_without_type_lengths_no_calendar_slim_dedup_log1p"
)

# Road+Geo v2, этап 1 (2026-10-15): чистка бейзлайна 61 -> 33 без пересчёта признаков.
# Основание: docs/analysis/features/REVIEW_2026-10-15.md, PLAN.md (этап 1),
# фактические Spearman-пары: reports/road_geo_baseline_spearman_pairs.csv,
# итог: docs/analysis/features/stage1_prune/README.md. Бейзлайн road_geo_baseline не меняется.
FEATURE_SET_ROAD_GEO_V2_PRUNE = "road_geo_v2_prune"

# Группа -> {признак: причина удаления}. Группы использованы в абляции по шагам.
V2_PRUNE_EXCLUDED_GROUPS: dict[str, dict[str, str]] = {
    "osm_tag_completeness": {
        "lanes_tagged_share": "полнота тега lanes (прилежание картографа); rho 0.956 с lanes_mean_weighted",
        "maxspeed_tagged_share": "полнота тега maxspeed (прилежание картографа); 78 % нулей",
    },
    "length": {
        "road_way_count": "дубликат road_length_total_m, rho 0.933 (остаток - нарезка ways)",
        "lane_meters_estimate": "дубликат road_length_total_m, rho 0.966 (lanes=1 при пропуске)",
    },
    "speed": {
        "maxspeed_max": "4 уникальных значения, импутация по классу дороги",
        "maxspeed_mean_weighted": "перекодировка долей классов, rho -0.978 с road_share_service",
    },
    "junctions": {
        "legacy_major_minor_junction_count": "дубликат major_junction_count, rho 0.980",
        "class_delta_junction_density_per_km": "дубликат minor_to_major_junction_density_per_km, rho 0.997",
        "minor_to_major_junction_density_per_km": "density-форма; rho 0.941 с minor_to_major_junction_count",
        "junction_density_per_km": "density-форма junction_count, rho 0.930",
        "junction_degree_max": "дубликат junction_degree_mean, rho 0.943",
        "class_delta_mean": "дубликат class_delta_max, rho 0.934",
        "signalized_class_delta_junction_count": "95 % нулей; rho 0.922 с traffic_signal_count",
        # отклонение от PLAN.md: в плане оставался, но по факту дубликат
        "minor_to_major_junction_count": "дубликат unsignalized_high_delta_junction_count, rho 0.978 "
                                          "(и junction_count, rho 0.908)",
    },
    "curvature": {
        "turn_event_density_per_km": "плотность вершин OSM; rho 0.931 с road_segment_density_per_km",
        "sharp_turn_count": "rho 0.896 с sharp_turn_share, 0.929 с sharp_turn_density_per_km",
        "sharp_turn_density_per_km": "дубликат sharp_turn_share, rho 0.968",
    },
    "segmentation": {
        "road_segment_count": "число вершин; rho 0.906 с длиной, 0.926 с road_way_count",
        "road_segment_length_mean": "дубликат road_segment_length_median, rho 0.891",
        "road_segment_density_per_km": "плотность оцифровки; rho 0.892 с short_segment_share",
    },
    "dead_ends": {
        # отклонение от PLAN.md: план оставлял density; оставлен count (см. README этапа)
        "dead_end_density_per_km": "density-форма dead_end_count (rho 0.846); делитель с service, max 631/км",
    },
    "surface": {
        "asphalt_surface_share": "пара length/share с asphalt_surface_length_m, rho 0.918",
    },
    "dead_weight": {
        "motorway_junction_density_per_km": "99.4 % нулей, важность 0.05",
        "road_share_living_street": "99.3 % нулей, важность 0.06",
        "priority_sign_density_per_km": "98.6 % нулей, важность 0.04",
        "traffic_calming_density_per_km": "98.3 % нулей, важность 0.19",
        "bridge_length_m": "98.1 % нулей, важность 0.14",
    },
    "composition": {
        "major_road_share": "линейная комбинация road_share_{trunk,primary,secondary} (+ исключённые links)",
    },
}
V2_PRUNE_EXCLUDED_FEATURES = {
    feature for group in V2_PRUNE_EXCLUDED_GROUPS.values() for feature in group
}

# Road+Geo v2, этап 2 (2026-10-16): перекрёстки с честной топологией.
# Новые признаки ячейки лежат в H3_JUNCTION_V2_FEATURES_PARQUET (scripts/build_junction_v2_features.py),
# подтягиваются по h3 (attach_junction_v2_features). Итог: docs/analysis/features/stage2_junctions/README.md.
# road_geo_baseline и road_geo_v2_prune не меняются.
FEATURE_SET_ROAD_GEO_V2_JUNCTIONS = "road_geo_v2_junctions"

# Старые признаки перекрёстков, оставшиеся в road_geo_v2_prune, -> причина замены.
V2_JUNCTIONS_REPLACED_FEATURES: dict[str, str] = {
    "junction_count": "66 % узлов с service, 828 узлов степени 2; -> street_junction_*_count + driveway_count",
    "junction_degree_mean": "степень = число ways в узле, а не подходов",
    "three_way_junction_count": "12 751 Т-образных учтены как степень 2; -> street_junction_3leg_count",
    "four_plus_way_junction_count": "1 204 крестообразных учтены как степень 2; -> street_junction_4leg_count",
    "major_junction_count": "любой узел с major-классом, включая въезды service; -> major_x_street_unsignalized_count",
    "class_delta_max": "ранги с service (delta 6 = primary x service); -> street_class_delta_max",
    "unsignalized_high_delta_junction_count": "rho 0.953 с driveway_count; -> driveway_count",
    "traffic_signal_count": "счёт узлов-светофоров зависит от стиля разметки; -> signalized_junction_count",
}
# Правило «одна величина — один признак» (|rho| >= 0.85) внутри новых признаков.
V2_JUNCTIONS_DEDUP_EXCLUDED: dict[str, str] = {
    "street_junction_4leg_unsignalized_count": "rho 0.949 с street_junction_4leg_count (светофор у 6 % "
                                               "крестов); одиночный AUC ниже (0.629 против 0.696)",
}
V2_JUNCTIONS_NEW_FEATURES = [f for f in JUNCTION_V2_FEATURES if f not in V2_JUNCTIONS_DEDUP_EXCLUDED]

# Road+Geo v2, этап 3 (2026-10-16): теги OSM — «нет тега» = неизвестно (NaN), а не 0.
# Новые признаки ячейки лежат в H3_TAGS_V2_FEATURES_PARQUET (scripts/build_tags_v2_features.py),
# подтягиваются по h3 (attach_tags_v2_features). Итог: docs/analysis/features/stage3_tags/README.md.
# road_geo_baseline, road_geo_v2_prune и road_geo_v2_junctions не меняются.
FEATURE_SET_ROAD_GEO_V2_TAGS = "road_geo_v2_tags"

# B1: экспозиция вместо асфальта. Старый признак -> признаки-замены.
V2_TAGS_EXPOSURE_REPLACED: dict[str, list[str]] = {
    "asphalt_surface_length_m": ["major_road_length_m", "street_length_m"],
}
V2_TAGS_EXPOSURE_REASON = (
    "нет тега surface = «не асфальт»; surface размечен у 100 % длины trunk/primary/secondary и 14–30 % "
    "residential/service -> длина магистралей + полнота разметки (см. stage3_tags/README.md)"
)
# B2: семантика пропусков. Старый признак -> v2-версия (NaN при отсутствии тега / < 100 м размеченной длины).
V2_TAGS_SEMANTICS_REPLACED: dict[str, str] = {
    "bad_surface_share": "bad_surface_share_among_tagged",
    "lanes_mean_weighted": "lanes_mean_weighted_v2",
    "oneway_share": "oneway_share_v2",
}

# Road+Geo v2, этап 4 (2026-10-16): геометрия улиц без service — граф «перекрёсток -> перекрёсток».
# Новые признаки ячейки лежат в H3_GEOMETRY_V2_FEATURES_PARQUET (scripts/build_geometry_v2_features.py),
# подтягиваются по h3 (attach_geometry_v2_features). Итог: docs/analysis/features/stage4_geometry/README.md.
# road_geo_baseline, road_geo_v2_prune, road_geo_v2_junctions и road_geo_v2_tags не меняются.
FEATURE_SET_ROAD_GEO_V2_GEOMETRY = "road_geo_v2_geometry"

# Старый признак -> признаки-замены (на месте старого; пустой список — удалить без замены).
# B1: сегментация (шаг вершин OSM) -> длина квартала.
V2_GEOMETRY_BLOCKS_REPLACED: dict[str, list[str]] = {
    "road_segment_length_median": ["block_length_median_m"],
    "short_segment_share": [],
}
# B2: углы в вершинах OSM (все классы, вкл. service и углы на перекрёстках) -> кривизна и радиус после DP.
V2_GEOMETRY_CURVATURE_REPLACED: dict[str, list[str]] = {
    "road_curvature_mean_weighted": [],
    "road_curvature_max": ["min_curve_radius_m"],
    "curvature_deg_per_km": ["curvature_deg_per_km_v2"],
    "sharp_turn_share": [],
}
# B3: тупики, ориентация, переезды.
V2_GEOMETRY_OTHER_REPLACED: dict[str, list[str]] = {
    "dead_end_count": ["street_dead_end_count"],
    "road_orientation_entropy": ["road_orientation_entropy_v2"],
    "railway_crossing_density_per_km": ["level_crossing_count"],
}
V2_GEOMETRY_REPLACED_REASON: dict[str, str] = {
    "road_segment_length_median": "медианный шаг вершин OSM (плотность оцифровки), 58 % сегментов — service",
    "short_segment_share": "доля шагов вершин < 50 м; порог выше медианы шага",
    "road_curvature_mean_weighted": "угол на вершину зависит от частоты вершин; service и углы на перекрёстках",
    "road_curvature_max": "p50 = 49°: прямой угол парковочного проезда",
    "curvature_deg_per_km": "числитель и знаменатель с service, углы на перекрёстках",
    "sharp_turn_share": "порог 45° произволен, ловит прямые углы парковок",
    "dead_end_count": "85 % — концы service-проездов",
    "road_orientation_entropy": "с service (парковочные ряды под любыми углами)",
    "railway_crossing_density_per_km": "смешаны автопереезды и пешеходные переходы через пути; делитель с service",
}


# Road+Geo v2, этап 5 (2026-10-16): паспортные атрибуты ФГИС СКДФ.
# Признаки ячейки лежат в H3_SKDF_V2_FEATURES_PARQUET (scripts/build_skdf_v2_features.py),
# подтягиваются по h3 (attach_skdf_v2_features). Итог: docs/analysis/features/stage5_skdf/README.md.
# road_geo_baseline и все v2-наборы этапов 1–4 не меняются.
FEATURE_SET_ROAD_GEO_V2_SKDF = "road_geo_v2_skdf"
# Блоки абляции 5.9 (после дедупликации по Spearman, README этапа 5).
V2_SKDF_WIDTH = ["skdf_roadway_width_mean_w", "skdf_width_per_lane_mean_w"]            # B1
V2_SKDF_COATING_REPLACED = {"bad_surface_share_among_tagged": "skdf_low_coating_share"}  # B2
V2_SKDF_HIERARCHY = ["skdf_cat_sp_rank_mean_w"]                                         # B3
V2_SKDF_LANES_REPLACED = {"lanes_mean_weighted_v2": "lanes_skdf_or_osm"}                # B4a
V2_SKDF_LANES_BOTH = ["skdf_lanes_mean_w"]                                              # B4b
V2_SKDF_SPEED = ["skdf_speed_mean_w"]                                                   # B5
V2_SKDF_AXLE = ["skdf_axle_load_mean_w"]                                                # B5 (нагрузка)
V2_SKDF_TRAFFIC = ["skdf_real_traffic_max"]                                             # B6
# Итоговая комбинация блоков (аргументы build_road_geo_v2_skdf_features), решение — stage5_skdf/README.md:
# только B1 (ширина): комбинации с иерархией/скоростью/нагрузкой/traffic прироста сверх B1 не дают,
# а «ширина + иерархия + скорость + нагрузка» роняет precision@1 % сверх 2σ.
ROAD_GEO_V2_SKDF_FINAL: dict = dict(width=True)
# Все блоки, прошедшие правило 5.9 поодиночке (вариант skdf_C_all_passed, 36 признаков).
ROAD_GEO_V2_SKDF_ALL_PASSED: dict = dict(width=True, hierarchy=True, speed=True, axle=True, traffic=True)

# Road+Geo v2, этап 6.2–6.5 (запуск 6a, 2026-10-19): прокси экспозиции из OSM — положение, POI, застройка, ОТ.
# Признаки ячейки лежат в H3_EXPOSURE_V6_FEATURES_PARQUET (scripts/build_exposure_v6_features.py),
# подтягиваются по h3 (attach_exposure_v6_features). Отбор — по двум валидационным фолдам 2021 + 2022
# (src/road_accident_risk/v2_eval.py), итог: docs/analysis/features/stage6_exposure/blocks_a/README.md.
# Промежуточный набор этапа 6 (до центральности 6.6 и соседей 6.7): A = road_geo_v2_skdf + блоки,
# прошедшие правило и жадную сборку. Прошёл только E3 (застройка); E1 (положение), E2 (POI) — не прошли
# ограничитель recall@10 %, E4 (ОТ) — без значимого эффекта; маршруты ОТ ухудшают PR-AUC.
FEATURE_SET_ROAD_GEO_V2_EXPOSURE_A = "road_geo_v2_exposure_a"
V2_EXPOSURE_A_FEATURES = ["building_footprint_m2", "small_building_share"]  # E3


SLIM_EXCLUDED_FEATURES = {
    "road_share_motorway",
    "road_share_motorway_link",
    "mini_roundabout_count",
    "mini_roundabout_share",
    "mini_roundabout_density_per_km",
    "roundabout_total_count",
    "roundabout_count",
    "roundabout_density_per_km",
    "roundabout_total_density_per_km",
    "full_roundabout_share",
    "priority_sign_count",
    "give_way_share",
    "motorway_junction_count",
    "stop_count",
    "stop_density_per_km",
    "stop_share",
    "give_way_count",
    "give_way_density_per_km",
    "tunnel_length_m",
    "tunnel_share",
    "link_road_length_m",
    "link_road_share",
    "road_share_primary_link",
    "road_share_secondary_link",
    "road_share_tertiary_link",
    "road_share_trunk_link",
    "crossing_signalized_count",
    "crossing_signalized_share",
    "crossing_signalized_density_per_km",
}

DUPLICATE_EXCLUDED_FEATURES = {
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
}

# Признаки с некорректной семантикой, а не дубликаты.
# parse_bool_yes трактует ОТСУТСТВУЮЩИЙ тег lit как False, но в OSM отсутствие
# lit означает "неизвестно", а не "не освещено" (в отличие от oneway/bridge/tunnel,
# где дефолт действительно отрицательный). Тег проставлен лишь у 5.1% длины сети
# Саратова, поэтому lit_road_length_m измеряет полноту разметки, а не освещённость.
# Удаление признака не ухудшило метрики и подняло PR-AUC на +0.0011 (2σ=0.0008).
INVALID_SEMANTICS_EXCLUDED_FEATURES = {
    "lit_road_length_m",
}

ROAD_ONLY_4_1_LOG1P_FEATURES = {
    "asphalt_surface_length_m",
    "bridge_length_m",
    "class_delta_junction_density_per_km",
    "crossing_count",
    "curvature_deg_per_km",
    "dead_end_count",
    "dead_end_density_per_km",
    "four_plus_way_junction_count",
    "junction_count",
    "junction_density_per_km",
    "lane_meters_estimate",
    "legacy_major_minor_junction_count",
    "lit_road_length_m",
    "major_junction_count",
    "minor_to_major_junction_count",
    "minor_to_major_junction_density_per_km",
    "priority_sign_density_per_km",
    "public_transport_stop_count",
    "railway_crossing_density_per_km",
    "road_length_total_m",
    "road_segment_count",
    "road_segment_density_per_km",
    "road_segment_length_mean",
    "road_segment_length_median",
    "road_way_count",
    "sharp_turn_count",
    "sharp_turn_density_per_km",
    "signalized_class_delta_junction_count",
    "three_way_junction_count",
    "traffic_calming_density_per_km",
    "traffic_signal_count",
    "turn_event_density_per_km",
    "unsignalized_high_delta_junction_count",
}


def build_road_features_without_type_lengths(columns: list[str]) -> list[str]:
    per_type_road_lengths = {f"road_length_{highway}_m" for highway in MOTOR_ROAD_CLASSES}
    excluded = {"h3", "month", TARGET} | per_type_road_lengths
    return [column for column in columns if column not in excluded]


def build_road_features_without_type_lengths_no_calendar(columns: list[str]) -> list[str]:
    excluded = {"year", "month_num"}
    return [
        feature
        for feature in build_road_features_without_type_lengths(columns)
        if feature not in excluded
    ]


def build_road_features_without_type_lengths_no_calendar_slim(columns: list[str]) -> list[str]:
    return [
        feature
        for feature in build_road_features_without_type_lengths_no_calendar(columns)
        if feature not in SLIM_EXCLUDED_FEATURES
    ]


def build_road_features_without_type_lengths_no_calendar_slim_dedup(columns: list[str]) -> list[str]:
    return [
        feature
        for feature in build_road_features_without_type_lengths_no_calendar_slim(columns)
        if feature not in DUPLICATE_EXCLUDED_FEATURES
        and feature not in INVALID_SEMANTICS_EXCLUDED_FEATURES
    ]


def build_road_geo_baseline_features(columns: list[str]) -> list[str]:
    base = [
        f for f in build_road_features_without_type_lengths_no_calendar_slim_dedup(columns)
        if f not in ROAD_GEO_FEATURES  # базовый билдер берёт все колонки панели — не дублируем
        and f not in JUNCTION_V2_FEATURES  # признаки этапа 2 в бейзлайн не входят, даже если подтянуты
        and f not in TAGS_V2_FEATURES  # признаки этапа 3 — тоже
        and f not in GEOMETRY_V2_FEATURES  # признаки этапа 4 — тоже
        and f not in SKDF_V2_ALL_FEATURES  # признаки этапа 5 (СКДФ) — тоже
        and f not in EXPOSURE_V6_FEATURES  # признаки этапа 6a (экспозиция) — тоже
    ]
    missing = [f for f in ROAD_GEO_FEATURES if f not in columns]
    if missing:
        raise ValueError(
            f"road_geo_baseline: нет колонок {missing}; вызовите attach_road_grade_features(panel) "
            f"или соберите их scripts/build_road_grade_features.py"
        )
    return [*base, *ROAD_GEO_FEATURES]


def build_road_geo_v2_prune_features(
    columns: list[str], keep_excluded: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    """road_geo_baseline минус V2_PRUNE_EXCLUDED_FEATURES (порядок признаков сохраняется).

    keep_excluded - признаки из списка исключений, которые нужно вернуть (абляция по группам).
    """
    return [
        f for f in build_road_geo_baseline_features(columns)
        if f not in V2_PRUNE_EXCLUDED_FEATURES or f in keep_excluded
    ]


def build_road_geo_v2_junctions_features(
    columns: list[str],
    keep_replaced: set[str] | frozenset[str] = frozenset(),
    new_features: list[str] | None = None,
) -> list[str]:
    """road_geo_v2_prune минус старые признаки перекрёстков плюс новые (этап 2).

    keep_replaced - старые признаки, которые нужно оставить (вариант C и поиск недостающего);
    new_features - список новых признаков (по умолчанию V2_JUNCTIONS_NEW_FEATURES).
    """
    new = list(V2_JUNCTIONS_NEW_FEATURES if new_features is None else new_features)
    missing = [f for f in new if f not in columns]
    if missing:
        raise ValueError(
            f"road_geo_v2_junctions: нет колонок {missing}; вызовите attach_junction_v2_features(panel) "
            f"или соберите их scripts/build_junction_v2_features.py"
        )
    base = [
        f for f in build_road_geo_v2_prune_features(columns)
        if f not in V2_JUNCTIONS_REPLACED_FEATURES or f in keep_replaced
    ]
    return [*base, *[f for f in new if f not in base]]


def build_road_geo_v2_tags_features(
    columns: list[str],
    replace_exposure: bool = True,
    replace_semantics: bool = True,
    extra: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """road_geo_v2_junctions с заменами этапа 3 (на месте старых признаков, порядок сохраняется).

    replace_exposure  - B1: asphalt_surface_length_m -> major_road_length_m + street_length_m;
    replace_semantics - B2: bad_surface_share, lanes_mean_weighted, oneway_share -> v2 с NaN;
    оба True - итоговый набор road_geo_v2_tags (B); extra - признаки, добавляемые в конец (абляция).
    """
    need = []
    if replace_exposure:
        need += [f for new in V2_TAGS_EXPOSURE_REPLACED.values() for f in new]
    if replace_semantics:
        need += list(V2_TAGS_SEMANTICS_REPLACED.values())
    missing = [f for f in [*need, *extra] if f not in columns]
    if missing:
        raise ValueError(
            f"road_geo_v2_tags: нет колонок {missing}; вызовите attach_tags_v2_features(panel) "
            f"или соберите их scripts/build_tags_v2_features.py"
        )
    out: list[str] = []
    for f in build_road_geo_v2_junctions_features(columns):
        if replace_exposure and f in V2_TAGS_EXPOSURE_REPLACED:
            out += V2_TAGS_EXPOSURE_REPLACED[f]
        elif replace_semantics and f in V2_TAGS_SEMANTICS_REPLACED:
            out.append(V2_TAGS_SEMANTICS_REPLACED[f])
        else:
            out.append(f)
    out += [f for f in extra if f not in out]
    if len(out) != len(set(out)):
        raise ValueError(f"road_geo_v2_tags: повтор признаков {out}")
    return out


def build_road_geo_v2_geometry_features(
    columns: list[str],
    replace_blocks: bool = True,
    replace_curvature: bool = True,
    replace_other: bool = True,
    extra: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """road_geo_v2_tags с заменами этапа 4 (на месте старых признаков, порядок сохраняется).

    replace_blocks    - B1: road_segment_length_median, short_segment_share -> block_length_median_m;
    replace_curvature - B2: 4 старых признака кривизны -> curvature_deg_per_km_v2, min_curve_radius_m;
    replace_other     - B3: dead_end_count, road_orientation_entropy, railway_crossing_density_per_km ->
                        street_dead_end_count, road_orientation_entropy_v2, level_crossing_count;
    все True - итоговый набор road_geo_v2_geometry (B); extra - признаки, добавляемые в конец (абляция).
    """
    groups = [(replace_blocks, V2_GEOMETRY_BLOCKS_REPLACED), (replace_curvature, V2_GEOMETRY_CURVATURE_REPLACED),
              (replace_other, V2_GEOMETRY_OTHER_REPLACED)]
    mapping = {old: new for on, g in groups if on for old, new in g.items()}
    need = [f for new in mapping.values() for f in new]
    missing = [f for f in [*need, *extra] if f not in columns]
    if missing:
        raise ValueError(
            f"road_geo_v2_geometry: нет колонок {missing}; вызовите attach_geometry_v2_features(panel) "
            f"или соберите их scripts/build_geometry_v2_features.py"
        )
    out: list[str] = []
    for f in build_road_geo_v2_tags_features(columns):
        out += mapping.get(f, [f])
    out += [f for f in extra if f not in out]
    if len(out) != len(set(out)):
        raise ValueError(f"road_geo_v2_geometry: повтор признаков {out}")
    return out


def build_road_geo_v2_skdf_features(
    columns: list[str],
    width: bool = False,
    coating: str | None = None,
    hierarchy: bool = False,
    lanes: str | None = None,
    speed: bool = False,
    axle: bool = False,
    traffic: bool = False,
    extra: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """road_geo_v2_geometry (30) + блоки этапа 5. Замены — на месте старого признака, добавления — в конец.

    width     - B1: + ширина проезжей части и ширина на полосу;
    coating   - B2: "replace" — bad_surface_share_among_tagged -> skdf_low_coating_share; "both" — добавить;
    hierarchy - B3: + ранг категории улицы по СП;
    lanes     - B4: "replace" — lanes_mean_weighted_v2 -> lanes_skdf_or_osm; "both" — + skdf_lanes_mean_w;
    speed/axle - B5: + паспортная скорость / осевая нагрузка;  traffic - B6: + skdf_real_traffic_max;
    extra     - признаки, добавляемые в конец (абляция).
    Итоговый набор road_geo_v2_skdf — по решению абляции (ROAD_GEO_V2_SKDF_FINAL).
    """
    mapping: dict[str, str] = {}
    if coating == "replace":
        mapping.update(V2_SKDF_COATING_REPLACED)
    if lanes == "replace":
        mapping.update(V2_SKDF_LANES_REPLACED)
    add: list[str] = []
    if width:
        add += V2_SKDF_WIDTH
    if coating == "both":
        add += list(V2_SKDF_COATING_REPLACED.values())
    if hierarchy:
        add += V2_SKDF_HIERARCHY
    if lanes == "both":
        add += V2_SKDF_LANES_BOTH
    if speed:
        add += V2_SKDF_SPEED
    if axle:
        add += V2_SKDF_AXLE
    if traffic:
        add += V2_SKDF_TRAFFIC
    need = [*mapping.values(), *add, *extra]
    missing = [f for f in need if f not in columns]
    if missing:
        raise ValueError(
            f"road_geo_v2_skdf: нет колонок {missing}; вызовите attach_skdf_v2_features(panel) "
            f"или соберите их scripts/build_skdf_v2_features.py"
        )
    out = [mapping.get(f, f) for f in build_road_geo_v2_geometry_features(columns)]
    out += [f for f in [*add, *extra] if f not in out]
    if len(out) != len(set(out)):
        raise ValueError(f"road_geo_v2_skdf: повтор признаков {out}")
    return out


def attach_exposure_v6_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Подтягивает признаки этапа 6a (положение, POI, застройка, ОТ) по h3, если их ещё нет в панели.

    NaN сохраняются: small_building_share и levels_tagged_share не определены в ячейке без зданий.
    """
    if all(f in panel.columns for f in EXPOSURE_V6_FEATURES):
        return panel
    if not H3_EXPOSURE_V6_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{H3_EXPOSURE_V6_FEATURES_PARQUET} не найден — запустите scripts/build_exposure_v6_features.py"
        )
    ex = pd.read_parquet(H3_EXPOSURE_V6_FEATURES_PARQUET, columns=["h3", *EXPOSURE_V6_FEATURES])
    return panel.merge(ex, on="h3", how="left")


def build_road_geo_v2_exposure_a_features(columns: list[str]) -> list[str]:
    """road_geo_v2_skdf (32) + признаки этапа 6a, прошедшие отбор (V2_EXPOSURE_A_FEATURES), в конец."""
    missing = [f for f in V2_EXPOSURE_A_FEATURES if f not in columns]
    if missing:
        raise ValueError(
            f"road_geo_v2_exposure_a: нет колонок {missing}; вызовите attach_exposure_v6_features(panel) "
            f"или соберите их scripts/build_exposure_v6_features.py"
        )
    base = build_road_geo_v2_skdf_features(columns, **ROAD_GEO_V2_SKDF_FINAL)
    return [*base, *[f for f in V2_EXPOSURE_A_FEATURES if f not in base]]


def attach_skdf_v2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Подтягивает признаки СКДФ этапа 5 по h3 и строит производные (lanes_skdf_or_osm, skdf_has_passport).

    NaN сохраняются: ячейка без паспортизованной дороги (37 % ячеек панели) — «нет на балансе».
    lanes_skdf_or_osm требует lanes_mean_weighted_v2 (attach_tags_v2_features).
    """
    if all(f in panel.columns for f in SKDF_V2_ALL_FEATURES):
        return panel
    if not H3_SKDF_V2_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{H3_SKDF_V2_FEATURES_PARQUET} не найден — запустите scripts/build_skdf_v2_features.py"
        )
    sk = pd.read_parquet(H3_SKDF_V2_FEATURES_PARQUET, columns=["h3", *SKDF_V2_FEATURES])
    out = panel.merge(sk, on="h3", how="left")
    out["skdf_has_passport"] = out["skdf_len_m"].notna().astype(float)
    if "lanes_mean_weighted_v2" in out.columns:
        out["lanes_skdf_or_osm"] = out["skdf_lanes_mean_w"].fillna(out["lanes_mean_weighted_v2"])
    return out

def attach_geometry_v2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Подтягивает признаки геометрии этапа 4 по h3, если их ещё нет в панели. NaN сохраняются
    (нет кварталов / уличной длины < 200 м в ячейке), CatBoost обрабатывает их сам."""
    if all(f in panel.columns for f in GEOMETRY_V2_FEATURES):
        return panel
    if not H3_GEOMETRY_V2_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{H3_GEOMETRY_V2_FEATURES_PARQUET} не найден — запустите scripts/build_geometry_v2_features.py"
        )
    geom = pd.read_parquet(H3_GEOMETRY_V2_FEATURES_PARQUET, columns=["h3", *GEOMETRY_V2_FEATURES])
    return panel.merge(geom, on="h3", how="left")


def attach_tags_v2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Подтягивает признаки тегов этапа 3 по h3, если их ещё нет в панели.

    major_road_length_m и lanes_tagged_share в панели уже есть (то же определение, сверено
    в scripts/build_tags_v2_features.py), поэтому подтягиваются только TAGS_V2_FEATURES.
    NaN сохраняются: это «тег неизвестен», CatBoost обрабатывает их сам.
    """
    if all(f in panel.columns for f in TAGS_V2_FEATURES):
        return panel
    if not H3_TAGS_V2_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{H3_TAGS_V2_FEATURES_PARQUET} не найден — запустите scripts/build_tags_v2_features.py"
        )
    tags = pd.read_parquet(H3_TAGS_V2_FEATURES_PARQUET, columns=["h3", *TAGS_V2_FEATURES])
    return panel.merge(tags, on="h3", how="left")


def attach_junction_v2_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Подтягивает признаки перекрёстков этапа 2 по h3, если их ещё нет в панели."""
    if all(f in panel.columns for f in JUNCTION_V2_FEATURES):
        return panel
    if not H3_JUNCTION_V2_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{H3_JUNCTION_V2_FEATURES_PARQUET} не найден — запустите scripts/build_junction_v2_features.py"
        )
    junc = pd.read_parquet(H3_JUNCTION_V2_FEATURES_PARQUET, columns=["h3", *JUNCTION_V2_FEATURES])
    return panel.merge(junc, on="h3", how="left")


def attach_road_grade_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Подтягивает признаки уклона по h3, если их ещё нет в панели."""
    if all(f in panel.columns for f in ROAD_GEO_FEATURES):
        return panel
    if not H3_ROAD_GRADE_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{H3_ROAD_GRADE_FEATURES_PARQUET} не найден — запустите scripts/build_road_grade_features.py"
        )
    grade = pd.read_parquet(H3_ROAD_GRADE_FEATURES_PARQUET, columns=["h3", *ROAD_GEO_FEATURES])
    return panel.merge(grade, on="h3", how="left")


def build_road_features_without_type_lengths_no_calendar_slim_dedup_log1p(columns: list[str]) -> list[str]:
    return build_road_features_without_type_lengths_no_calendar_slim_dedup(columns)


def build_features(panel: pd.DataFrame, feature_set: str, features_arg: str | None = None) -> list[str]:
    if features_arg:
        return [feature.strip() for feature in features_arg.split(",") if feature.strip()]

    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS:
        return build_road_features_without_type_lengths(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR:
        return build_road_features_without_type_lengths_no_calendar(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM:
        return build_road_features_without_type_lengths_no_calendar_slim(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP:
        return build_road_features_without_type_lengths_no_calendar_slim_dedup(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP_LOG1P:
        return build_road_features_without_type_lengths_no_calendar_slim_dedup_log1p(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_GEO_BASELINE:
        return build_road_geo_baseline_features(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_GEO_V2_PRUNE:
        return build_road_geo_v2_prune_features(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_GEO_V2_JUNCTIONS:
        return build_road_geo_v2_junctions_features(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_GEO_V2_TAGS:
        return build_road_geo_v2_tags_features(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_GEO_V2_GEOMETRY:
        return build_road_geo_v2_geometry_features(list(panel.columns))
    if feature_set == FEATURE_SET_ROAD_GEO_V2_SKDF:
        return build_road_geo_v2_skdf_features(list(panel.columns), **ROAD_GEO_V2_SKDF_FINAL)
    if feature_set == FEATURE_SET_ROAD_GEO_V2_EXPOSURE_A:
        return build_road_geo_v2_exposure_a_features(list(panel.columns))

    raise ValueError(f"Unknown feature set: {feature_set}")


def default_log1p_features(feature_set: str, features: list[str]) -> list[str]:
    if feature_set == FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS_NO_CALENDAR_SLIM_DEDUP_LOG1P:
        return [feature for feature in features if feature in ROAD_ONLY_4_1_LOG1P_FEATURES]
    return []


def fit_p99_clip_caps(train: pd.DataFrame, features: list[str]) -> dict[str, float]:
    quantiles = train[features].quantile(0.99, numeric_only=True)
    max_values = train[features].max(numeric_only=True)
    caps: dict[str, float] = {}
    for feature, cap in quantiles.items():
        if not np.isfinite(cap) or cap <= 0:
            continue
        max_value = max_values.get(feature)
        if pd.isna(max_value) or max_value <= cap:
            continue
        caps[feature] = float(cap)
    return caps


def apply_p99_clipping(df: pd.DataFrame, caps: dict[str, float]) -> pd.DataFrame:
    if not caps:
        return df
    result = df.copy()
    for feature, cap in caps.items():
        if feature in result.columns:
            result[feature] = result[feature].clip(upper=cap)
    return result


def apply_log1p_transform(df: pd.DataFrame, log1p_features: list[str]) -> pd.DataFrame:
    if not log1p_features:
        return df
    result = df.copy()
    for feature in log1p_features:
        if feature not in result.columns:
            continue
        min_value = result[feature].min()
        if pd.notna(min_value) and min_value < 0:
            raise ValueError(f"Cannot apply log1p to negative feature: {feature}")
        result[feature] = np.log1p(result[feature])
    return result


def train_and_evaluate(
    panel_path: Path = H3_MONTH_ROAD_FEATURES_PARQUET,
    train_end: str = TRAIN_END,
    feature_set: str = FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS,
    features_arg: str | None = None,
    min_road_length_m: float | None = None,
    clip_p99: bool = False,
    clip_p99_caps_path: Path | None = None,
    log1p_features_path: Path | None = None,
    iterations: int = 800,
    learning_rate: float = 0.05,
    depth: int = 6,
    random_seed: int = 42,
    feature_importance_path: Path = PROJECT_ROOT / "reports" / "catboost_feature_importance_without_type_lengths.csv",
) -> None:
    panel = pd.read_parquet(panel_path)
    ex6 = (FEATURE_SET_ROAD_GEO_V2_EXPOSURE_A,)
    if feature_set in (FEATURE_SET_ROAD_GEO_BASELINE, FEATURE_SET_ROAD_GEO_V2_PRUNE, FEATURE_SET_ROAD_GEO_V2_JUNCTIONS,
                       FEATURE_SET_ROAD_GEO_V2_TAGS, FEATURE_SET_ROAD_GEO_V2_GEOMETRY, FEATURE_SET_ROAD_GEO_V2_SKDF, *ex6):
        panel = attach_road_grade_features(panel)
    if feature_set in (FEATURE_SET_ROAD_GEO_V2_JUNCTIONS, FEATURE_SET_ROAD_GEO_V2_TAGS, FEATURE_SET_ROAD_GEO_V2_GEOMETRY,
                       FEATURE_SET_ROAD_GEO_V2_SKDF, *ex6):
        panel = attach_junction_v2_features(panel)
    if feature_set in (FEATURE_SET_ROAD_GEO_V2_TAGS, FEATURE_SET_ROAD_GEO_V2_GEOMETRY, FEATURE_SET_ROAD_GEO_V2_SKDF, *ex6):
        panel = attach_tags_v2_features(panel)
    if feature_set in (FEATURE_SET_ROAD_GEO_V2_GEOMETRY, FEATURE_SET_ROAD_GEO_V2_SKDF, *ex6):
        panel = attach_geometry_v2_features(panel)
    if feature_set in (FEATURE_SET_ROAD_GEO_V2_SKDF, *ex6):
        panel = attach_skdf_v2_features(panel)
    if feature_set in ex6:
        panel = attach_exposure_v6_features(panel)
    if min_road_length_m is not None:
        if "road_length_total_m" not in panel.columns:
            raise ValueError("road_length_total_m is required when --min-road-length-m is used")
        panel = panel[panel["road_length_total_m"] > min_road_length_m].copy()

    features = build_features(panel, feature_set=feature_set, features_arg=features_arg)

    missing = [feature for feature in features if feature not in panel.columns]
    if missing:
        raise ValueError(f"Missing feature columns in panel: {missing}")

    df = panel[["h3", "month", *features, TARGET]].copy()
    df["month"] = pd.to_datetime(df["month"])
    df[TARGET] = df[TARGET].astype(int)

    train = df[df["month"] < train_end].copy()
    test = df[df["month"] >= train_end].copy()
    clip_caps: dict[str, float] = {}
    log1p_features = default_log1p_features(feature_set, features)
    if clip_p99:
        clip_caps = fit_p99_clip_caps(train, features)
        train = apply_p99_clipping(train, clip_caps)
        test = apply_p99_clipping(test, clip_caps)
        if clip_p99_caps_path is not None:
            clip_p99_caps_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"feature": feature, "p99_cap": cap} for feature, cap in clip_caps.items()]
            ).to_csv(clip_p99_caps_path, index=False)
    if log1p_features:
        train = apply_log1p_transform(train, log1p_features)
        test = apply_log1p_transform(test, log1p_features)
        if log1p_features_path is not None:
            log1p_features_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"feature": log1p_features}).to_csv(log1p_features_path, index=False)

    X_train = train[features]
    y_train = train[TARGET]
    X_test = test[features]
    y_test = test[TARGET]

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=random_seed,
        allow_writing_files=False,
        verbose=100,
    )
    train_pool = Pool(X_train, y_train, feature_names=features)
    test_pool = Pool(X_test, y_test, feature_names=features)
    model.fit(train_pool, eval_set=test_pool, use_best_model=False)

    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= 0.5).astype(int)

    print(f"Panel: {panel_path}")
    print(f"Panel shape: {panel.shape}")
    print(f"Feature set: {feature_set}")
    if min_road_length_m is not None:
        print(f"Row filter: road_length_total_m > {min_road_length_m}")
    if clip_p99:
        print(f"P99 clipping: enabled | clipped_features={len(clip_caps)}")
        if clip_p99_caps_path is not None:
            print(f"P99 clipping caps: {clip_p99_caps_path}")
    if log1p_features:
        print(f"Log1p transform: enabled | features={len(log1p_features)}")
        if log1p_features_path is not None:
            print(f"Log1p features: {log1p_features_path}")
    print(f"Feature count: {len(features)}")
    print(f"Features: {', '.join(features)}")
    print(f"Train: {train.shape} | {train['month'].min()} to {train['month'].max()} | rate={y_train.mean():.6f}")
    print(f"Test:  {test.shape} | {test['month'].min()} to {test['month'].max()} | rate={y_test.mean():.6f}")
    print(f"CatBoost: iterations={iterations}, learning_rate={learning_rate}, depth={depth}")
    print(f"ROC-AUC: {roc_auc_score(y_test, test_scores):.6f}")
    print(f"PR-AUC:  {average_precision_score(y_test, test_scores):.6f}")
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_pred))
    print()
    print(classification_report(y_test, test_pred, digits=4))
    print("Top-k metrics:")

    for k_frac in [0.01, 0.03, 0.05, 0.10]:
        metrics = precision_recall_at_k(y_test.to_numpy(), test_scores, k_frac=k_frac)
        print(
            f"Top {k_frac:.0%}: "
            f"k={metrics['k']:,}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"lift={metrics['lift']:.2f}, "
            f"tp={metrics['positives_in_top_k']:,}/{metrics['total_positives']:,}"
        )

    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.get_feature_importance(train_pool, type="FeatureImportance"),
        }
    ).sort_values("importance", ascending=False)
    feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(feature_importance_path, index=False)

    print()
    print(f"Feature importance: {feature_importance_path}")
    print("Top feature importances:")
    for row in feature_importance.head(20).itertuples(index=False):
        print(f"{row.feature}: {row.importance:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a CatBoost baseline.")
    parser.add_argument("--panel", type=Path, default=H3_MONTH_ROAD_FEATURES_PARQUET)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--feature-set", default=FEATURE_SET_ROAD_WITHOUT_TYPE_LENGTHS)
    parser.add_argument("--features", default=None, help="Comma-separated feature columns to use.")
    parser.add_argument("--min-road-length-m", type=float, default=None)
    parser.add_argument("--clip-p99", action="store_true")
    parser.add_argument("--clip-p99-caps-path", type=Path, default=None)
    parser.add_argument("--log1p-features-path", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--feature-importance-path",
        type=Path,
        default=PROJECT_ROOT / "reports" / "catboost_feature_importance_without_type_lengths.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_and_evaluate(
        panel_path=args.panel,
        train_end=args.train_end,
        feature_set=args.feature_set,
        features_arg=args.features,
        min_road_length_m=args.min_road_length_m,
        clip_p99=args.clip_p99,
        clip_p99_caps_path=args.clip_p99_caps_path,
        log1p_features_path=args.log1p_features_path,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        random_seed=args.random_seed,
        feature_importance_path=args.feature_importance_path,
    )


if __name__ == "__main__":
    main()
