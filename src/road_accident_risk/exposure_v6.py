"""Road+Geo v2, этап 6.2–6.5 (запуск 6a): прокси экспозиции и активности из снимка OSM.

Признаки ячейки H3 res 9 лежат в H3_EXPOSURE_V6_FEATURES_PARQUET (scripts/build_exposure_v6_features.py),
подтягиваются по h3 (catboost_baseline.attach_exposure_v6_features). Итог и определения:
docs/analysis/features/stage6_exposure/blocks_a/README.md.

Все расстояния — от центроида ячейки (h3.cell_to_latlng), в UTM 38N (cKDTree), никогда от точек ДТП.
"""
from __future__ import annotations

# --- константы правил (обоснование — README) ------------------------------------------------------
# E1: точки центров для способа (а) — узлы OSM (как в EDA 6.1, LOG §3).
SARATOV_CENTER = (51.53338, 46.03414)   # node 3620668550 «Театральная площадь»
ENGELS_CENTER = (51.50138, 46.12331)    # node 101480194 place=city «Энгельс»
# E1, способ (б): пики сглаженной плотности POI-аттракторов.
POI_PEAK_GRID_M = 100.0        # шаг сетки гистограммы
POI_PEAK_SIGMA_M = 1000.0      # σ гауссова ядра
POI_PEAK_WINDOW_M = 3000.0     # окно поиска локального максимума
POI_PEAK_MIN_REL = 0.20        # пик — не ниже 20 % от главного
MAJOR_CLASSES = ("trunk", "primary")
LINE_SAMPLE_M = 10.0           # шаг сэмплов по линиям для расстояния до ребра

# E2: POI-аттракторы — правило PLAN 6.3 / EDA 6.1 (без парковок, уличной мебели, leisure, nightclub).
ATTRACTOR_AMENITY = frozenset({"restaurant", "cafe", "fast_food", "pharmacy", "bank", "hospital", "clinic",
                               "marketplace", "bar", "pub"})
SHOP_EXCLUDED = frozenset({"vacant", "no"})
SCHOOL_AMENITY = frozenset({"school", "kindergarten"})
HEALTH_AMENITY = frozenset({"hospital", "clinic", "doctors", "dentist", "pharmacy"})
POI_BUFFERS_M = (300, 500)

# E3: порог «мелкого» здания — по распределению площади основания (бимодальное, см. README).
SMALL_BUILDING_M2 = 200.0

# E4: дедуп остановок, маршруты.
STOP_DEDUP_M = 30.0
ROUTE_TYPES = ("bus", "trolleybus")
TRAM_SERVICE_EXCLUDED = frozenset({"yard", "siding", "spur"})

# --- признаки ------------------------------------------------------------------------------------
E1_CENTER_CANDIDATES = ["dist_to_center_two_m", "dist_to_center_poi_m"]
E1_OTHER = ["dist_to_nearest_major_m", "dist_to_nearest_signal_m"]
E2_MODES = ("cell", "r300", "r500")
E2_BASE = ["poi_attractor_count", "school_kindergarten_count", "healthcare_count"]


def e2_names(mode: str) -> list[str]:
    """Имена счётов POI для способа подсчёта: cell — в ячейке, r300/r500 — в круге от центроида."""
    return list(E2_BASE) if mode == "cell" else [f"{f}_{mode}" for f in E2_BASE]


E3_FEATURES = ["building_footprint_m2", "small_building_share", "attractor_building_footprint_m2"]
E4_FEATURES = ["pt_stop_count_v6", "dist_to_nearest_stop_m", "tram_track_length_m", "pt_route_count"]
DIAGNOSTIC_FEATURES = ["levels_tagged_share", "building_count", "pt_stop_raw_count"]

EXPOSURE_V6_FEATURES = [
    *E1_CENTER_CANDIDATES, *E1_OTHER,
    *[f for m in E2_MODES for f in e2_names(m)],
    *E3_FEATURES, *E4_FEATURES, *DIAGNOSTIC_FEATURES,
]
