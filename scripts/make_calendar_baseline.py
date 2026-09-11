from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from road_accident_risk.paths import CALENDAR_BASELINE_PARQUET, H3_MONTH_PANEL_PARQUET


TARGET = "target_had_accident_next_month"
CALENDAR_COLUMNS = [
    "h3",
    "month",
    "year",
    "month_num",
    "month_sin",
    "month_cos",
    "is_winter",
    "is_spring",
    "is_summer",
    "is_autumn",
    TARGET,
]


def make_calendar_baseline(
    input_path: Path = H3_MONTH_PANEL_PARQUET,
    output_path: Path = CALENDAR_BASELINE_PARQUET,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input panel not found: {input_path}")

    panel = pd.read_parquet(input_path)
    required = {"h3", "month", "year", "month_num", TARGET}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Missing required columns in input panel: {missing}")

    df = panel[["h3", "month", "year", "month_num", TARGET]].copy()
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["year"].astype("int16")
    df["month_num"] = df["month_num"].astype("int8")

    angle = 2 * math.pi * (df["month_num"].astype(float) - 1) / 12
    df["month_sin"] = angle.map(math.sin)
    df["month_cos"] = angle.map(math.cos)

    df["is_winter"] = df["month_num"].isin([12, 1, 2])
    df["is_spring"] = df["month_num"].isin([3, 4, 5])
    df["is_summer"] = df["month_num"].isin([6, 7, 8])
    df["is_autumn"] = df["month_num"].isin([9, 10, 11])
    df[TARGET] = df[TARGET].astype(bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = df[CALENDAR_COLUMNS]
    df.to_parquet(output_path, index=False)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"Positive rate: {df[TARGET].mean():.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a calendar-only baseline parquet from H3 panel.")
    parser.add_argument("--input", type=Path, default=H3_MONTH_PANEL_PARQUET)
    parser.add_argument("--output", type=Path, default=CALENDAR_BASELINE_PARQUET)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    make_calendar_baseline(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
