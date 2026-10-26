"""Этап 5 плана docs/analysis/features/PLAN.md: паспортные атрибуты ФГИС СКДФ в ячейках H3 res 9.

Строит из parquet_output/skdf_traffic_saratov.geojson (5 594 участка):
    parquet_output/skdf_v2_segments.parquet     — участки с расшифрованными gid (без геометрии)
    parquet_output/skdf_v2_h3_pieces.parquet    — куски участок × ячейка (сэмплы ~20 м, длина)
    parquet_output/h3_skdf_v2_features.parquet  — признаки ячейки (SKDF_V2_FEATURES)
Старый h3_skdf_features.parquet (EDA 2026-09-16) не трогается; сверка с ним — в выводе
(тот же способ агрегации должен дать те же длину, ширину, полосность и скорость).

Запуск: PYTHONPATH=src venv/bin/python scripts/build_skdf_v2_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.paths import (  # noqa: E402
    H3_SKDF_V2_FEATURES_PARQUET, H3_TAGS_V2_FEATURES_PARQUET, PARQUET_DIR, SKDF_TRAFFIC_GEOJSON,
    SKDF_V2_H3_PIECES_PARQUET, SKDF_V2_SEGMENTS_PARQUET,
)
from road_accident_risk.skdf_v2 import (  # noqa: E402
    SKDF_V2_FEATURES, aggregate_h3_skdf_v2_features, allocate_skdf_to_h3, load_skdf_segments,
)


def main() -> None:
    t0 = time.time()
    seg = load_skdf_segments(SKDF_TRAFFIC_GEOJSON)
    print(f"участков: {len(seg):,}; уникальных id: {seg.skdf_id.nunique():,}")
    pieces = allocate_skdf_to_h3(seg)
    print(f"кусков участок × ячейка: {len(pieces):,}; ячеек: {pieces.h3.nunique():,}; "
          f"длина {pieces.piece_length_m.sum() / 1000:.1f} км ({time.time() - t0:.1f} с)")
    street = pd.read_parquet(H3_TAGS_V2_FEATURES_PARQUET, columns=["h3", "street_length_m"]).set_index("h3")["street_length_m"]
    feats = aggregate_h3_skdf_v2_features(seg, pieces, street)

    old = pd.read_parquet(PARQUET_DIR / "h3_skdf_features.parquet").set_index("h3")
    new = feats.set_index("h3")
    print(f"сверка со старым h3_skdf_features: ячеек старых {len(old):,}, новых {len(new):,}, "
          f"общих {old.index.isin(new.index).sum():,}")
    for a, b in [("skdf_len_m", "skdf_len_m"), ("skdf_width_mean_w", "skdf_roadway_width_mean_w"),
                 ("skdf_lanes_mean_w", "skdf_lanes_mean_w"), ("skdf_speed_mean_w", "skdf_speed_mean_w"),
                 ("skdf_placeholder_share", "skdf_placeholder_share")]:
        x = old[a]; y = new[b].reindex(old.index)
        both = x.notna() & y.notna()
        print(f"  {a:>24} ~ {b:<28} общих {int(both.sum()):5d}; max |Δ| = {(x - y)[both].abs().max():.4g}; "
              f"доля |Δ| > 1e-6: {((x - y)[both].abs() > 1e-6).mean():.3%}; NaN старое/новое {int(x.isna().sum())}/{int(y.isna().sum())}")

    seg.drop(columns=["coords"]).to_parquet(SKDF_V2_SEGMENTS_PARQUET, index=False)
    pieces.to_parquet(SKDF_V2_H3_PIECES_PARQUET, index=False)
    feats.to_parquet(H3_SKDF_V2_FEATURES_PARQUET, index=False)
    d = feats[SKDF_V2_FEATURES]
    print(pd.DataFrame({"nan_share": d.isna().mean(), "zero_share": (d == 0).mean(), "median": d.median(),
                        "p99": d.quantile(0.99), "max": d.max()}).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"ячеек: {len(feats):,} -> {H3_SKDF_V2_FEATURES_PARQUET}; {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
