"""Этап 3 плана docs/analysis/features/PLAN.md: теги OSM — «нет тега» = неизвестно (NaN), а не 0.

Строит из osm_road_h3_segments.parquet (куски way × ячейка) и tags_json из osm_roads_raw.parquet:
    parquet_output/h3_tags_v2_features.parquet — признаки ячейки (TAGS_V2_FEATURES + сверочные колонки)
Старые parquet (h3_road_features, панель, сегменты) не трогаются.

Запуск: PYTHONPATH=src venv/bin/python scripts/build_tags_v2_features.py [--min-tagged-m 100]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.osm_road_features import (  # noqa: E402
    TAGS_V2_FEATURES, TAGS_V2_MIN_TAGGED_M, aggregate_h3_tags_v2_features, build_h3_tag_v2_segments,
)
from road_accident_risk.paths import (  # noqa: E402
    H3_ROAD_FEATURES_PARQUET, H3_TAGS_V2_FEATURES_PARQUET, OSM_ROAD_H3_SEGMENTS_PARQUET, OSM_ROADS_RAW_PARQUET,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tagged-m", type=float, default=TAGS_V2_MIN_TAGGED_M)
    args = ap.parse_args()
    t0 = time.time()
    seg = pd.read_parquet(OSM_ROAD_H3_SEGMENTS_PARQUET)
    roads = pd.read_parquet(OSM_ROADS_RAW_PARQUET, columns=["osm_id", "tags_json"])
    old = pd.read_parquet(H3_ROAD_FEATURES_PARQUET)
    st = build_h3_tag_v2_segments(seg, roads)
    feats = aggregate_h3_tags_v2_features(st, cells=old["h3"], min_tagged_m=args.min_tagged_m)

    # сверка с существующими признаками панели (те же куски, то же определение)
    chk = old.set_index("h3").join(feats.set_index("h3"), rsuffix="_new")
    for a in ["road_length_total_m", "major_road_length_m", "lanes_tagged_share"]:
        diff = (chk[a] - chk[f"{a}_new"]).abs().max()
        print(f"сверка {a}: max |старое − новое| = {diff:.3e}")
        if diff > 1e-6:
            raise SystemExit(f"расхождение в {a}")
    lm = chk["lanes_mean_weighted"].where(chk["lanes_mean_weighted_v2"].notna())
    print(f"сверка lanes_mean_weighted там, где v2 не NaN: max |Δ| = {(lm - chk['lanes_mean_weighted_v2']).abs().max():.3e}")

    feats.to_parquet(H3_TAGS_V2_FEATURES_PARQUET, index=False)
    print(f"кусков way × ячейка: {len(st):,}; ячеек: {len(feats):,}; порог размеченной длины {args.min_tagged_m:.0f} м")
    print(f"-> {H3_TAGS_V2_FEATURES_PARQUET}")
    d = feats[TAGS_V2_FEATURES + ["major_road_length_m"]]
    print(pd.DataFrame({"nan_share": d.isna().mean(), "zero_share": (d == 0).mean(),
                        "median": d.median(), "max": d.max()}).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"{time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
