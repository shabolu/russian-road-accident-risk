"""Этап 7.1: панель «ячейка × год» -> parquet_output/h3_year_panel.parquet (см. src/road_accident_risk/v2_eval_year.py).

Запуск:
    PYTHONPATH=src venv/bin/python scripts/build_h3_year_panel.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from road_accident_risk.v2_eval_year import YEAR_PANEL_PARQUET, build_year_panel  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    p = build_year_panel(write=True)
    print(f"-> {YEAR_PANEL_PARQUET} [{time.time() - t0:.0f} с]")
    print(p.groupby("year")[["count_year", "had_accident_year"]].agg(["sum", "mean"]).to_string())
