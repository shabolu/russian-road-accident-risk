from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import h3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from road_accident_risk.paths import CRASHES_SLIM_PARQUET, H3_MONTH_PANEL_PARQUET
from road_accident_risk.utils import format_size


H3_RESOLUTION = 9
START_DATE = None
END_DATE = None
BBOX = (51.40, 45.80, 51.70, 46.20)
INCLUDE_ALL_H3_CELLS_IN_BBOX = True
READ_BATCH_SIZE = 250_000
CELL_CHUNK_SIZE = 50_000
DROP_LAST_MONTH_WITHOUT_TARGET = True
COMPRESSION = "snappy"
PROGRESS_EVERY_SECONDS = 3


def latlng_to_h3_cell(lat: float, lng: float, resolution: int) -> str:
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)

    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lng, resolution)

    raise RuntimeError("Unsupported h3 package version")


def h3_cells_from_bbox(
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    resolution: int,
) -> set[str]:
    outer_latlng = [
        (min_lat, min_lng),
        (min_lat, max_lng),
        (max_lat, max_lng),
        (max_lat, min_lng),
        (min_lat, min_lng),
    ]

    if hasattr(h3, "LatLngPoly") and hasattr(h3, "polygon_to_cells"):
        poly = h3.LatLngPoly(outer_latlng)
        return set(h3.polygon_to_cells(poly, resolution))

    if hasattr(h3, "polyfill"):
        geojson_polygon = {
            "type": "Polygon",
            "coordinates": [[
                [min_lng, min_lat],
                [max_lng, min_lat],
                [max_lng, max_lat],
                [min_lng, max_lat],
                [min_lng, min_lat],
            ]],
        }
        return set(
            h3.polyfill(
                geojson_polygon,
                resolution,
                geo_json_conformant=True,
            )
        )

    raise RuntimeError("Unsupported h3 package version for bbox polyfill")


def validate_bbox(bbox: tuple[float, float, float, float] | None) -> None:
    if bbox is None:
        return

    min_lat, min_lng, max_lat, max_lng = bbox

    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("BBOX latitudes must be between -90 and 90")

    if not (-180 <= min_lng <= 180 and -180 <= max_lng <= 180):
        raise ValueError("BBOX longitudes must be between -180 and 180")

    if min_lat >= max_lat:
        raise ValueError("BBOX min_lat must be < max_lat")

    if min_lng >= max_lng:
        raise ValueError("BBOX min_lng must be < max_lng")


def parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None

    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be MIN_LAT,MIN_LNG,MAX_LAT,MAX_LNG")

    bbox = (parts[0], parts[1], parts[2], parts[3])
    validate_bbox(bbox)
    return bbox


def apply_bbox_filter(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None,
) -> pd.DataFrame:
    if bbox is None:
        return df

    min_lat, min_lng, max_lat, max_lng = bbox

    return df[
        df["lat"].between(min_lat, max_lat)
        & df["lng"].between(min_lng, max_lng)
    ].copy()


def make_month_range(min_month: pd.Timestamp, max_month: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(
        min_month.to_period("M").to_timestamp(),
        max_month.to_period("M").to_timestamp(),
        freq="MS",
    )


def chunked(items: list[str], chunk_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def aggregate_crashes_to_h3_month(
    input_path: Path = CRASHES_SLIM_PARQUET,
    h3_resolution: int = H3_RESOLUTION,
    bbox: tuple[float, float, float, float] | None = BBOX,
    start_date: str | None = START_DATE,
    end_date: str | None = END_DATE,
    read_batch_size: int = READ_BATCH_SIZE,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    validate_bbox(bbox)

    pf = pq.ParquetFile(input_path)

    required_columns = ["datetime", "lat", "lng"]
    existing_columns = set(pf.schema_arrow.names)
    missing = [c for c in required_columns if c not in existing_columns]
    if missing:
        raise ValueError(f"Missing columns in input parquet: {missing}")

    input_size = input_path.stat().st_size
    total_rows = pf.metadata.num_rows

    print("=" * 100)
    print("STEP 1: Aggregate crashes to H3 x month")
    print(f"Input: {input_path}")
    print(f"Input size: {format_size(input_size)}")
    print(f"Rows: {total_rows:,}")
    print(f"H3 resolution: {h3_resolution}")
    print(f"BBOX: {bbox}")
    print(f"START_DATE: {start_date}")
    print(f"END_DATE: {end_date}")

    agg_parts: list[pd.DataFrame] = []

    rows_read = 0
    rows_after_filters = 0
    started_at = time.time()
    last_progress_at = started_at

    columns_to_read = ["datetime", "lat", "lng"]

    for batch in pf.iter_batches(batch_size=read_batch_size, columns=columns_to_read):
        rows_read += batch.num_rows

        df = batch.to_pandas()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
        df = df.dropna(subset=["datetime", "lat", "lng"])

        if start_date is not None:
            df = df[df["datetime"] >= pd.Timestamp(start_date)]

        if end_date is not None:
            df = df[df["datetime"] <= pd.Timestamp(end_date)]

        df = apply_bbox_filter(df, bbox)

        if len(df) > 0:
            rows_after_filters += len(df)
            df["h3"] = [
                latlng_to_h3_cell(lat, lng, h3_resolution)
                for lat, lng in zip(df["lat"], df["lng"])
            ]
            df["month"] = df["datetime"].dt.to_period("M").dt.to_timestamp()

            part = (
                df.groupby(["h3", "month"], as_index=False)
                  .size()
                  .rename(columns={"size": "accident_count"})
            )
            agg_parts.append(part)

        now = time.time()
        if now - last_progress_at >= PROGRESS_EVERY_SECONDS or rows_read == total_rows:
            elapsed = max(now - started_at, 1e-9)
            speed = rows_read / elapsed
            pct = rows_read / total_rows * 100 if total_rows else 100
            eta = (total_rows - rows_read) / speed if speed > 0 else None
            eta_text = f"{eta / 60:.1f} min" if eta is not None else "unknown"

            print(
                f"[read] rows={rows_read:,}/{total_rows:,} ({pct:.2f}%) | "
                f"kept={rows_after_filters:,} | "
                f"speed={speed:,.0f} rows/s | eta={eta_text}",
                flush=True,
            )
            last_progress_at = now

    if not agg_parts:
        raise RuntimeError("No rows after filters. Check bbox/date filters.")

    positive = pd.concat(agg_parts, ignore_index=True)
    positive = (
        positive.groupby(["h3", "month"], as_index=False)["accident_count"]
        .sum()
    )
    positive["accident_count"] = positive["accident_count"].astype("int32")

    min_month = positive["month"].min()
    max_month = positive["month"].max()

    print()
    print("Positive H3-month rows:", f"{len(positive):,}")
    print("Unique H3 cells with accidents:", f"{positive['h3'].nunique():,}")
    print("Month range:", min_month.date(), "to", max_month.date())
    print("Total accidents after filters:", f"{positive['accident_count'].sum():,}")

    return positive, min_month, max_month


def get_panel_cells(
    positive: pd.DataFrame,
    bbox: tuple[float, float, float, float] | None = BBOX,
    h3_resolution: int = H3_RESOLUTION,
    include_all_h3_cells_in_bbox: bool = INCLUDE_ALL_H3_CELLS_IN_BBOX,
) -> list[str]:
    observed_cells = set(positive["h3"].unique())

    if bbox is not None and include_all_h3_cells_in_bbox:
        min_lat, min_lng, max_lat, max_lng = bbox

        print()
        print("Generating all H3 cells inside bbox...")
        bbox_cells = h3_cells_from_bbox(
            min_lat=min_lat,
            min_lng=min_lng,
            max_lat=max_lat,
            max_lng=max_lng,
            resolution=h3_resolution,
        )

        cells = sorted(bbox_cells | observed_cells)

        print("BBox H3 cells:", f"{len(bbox_cells):,}")
        print("Observed H3 cells:", f"{len(observed_cells):,}")
        print("Total panel H3 cells:", f"{len(cells):,}")

        return cells

    cells = sorted(observed_cells)
    print()
    print("Using only observed H3 cells.")
    print("Total panel H3 cells:", f"{len(cells):,}")

    return cells


def add_lags_and_targets(
    panel: pd.DataFrame,
    drop_last_month_without_target: bool = DROP_LAST_MONTH_WITHOUT_TARGET,
) -> pd.DataFrame:
    panel = panel.sort_values(["h3", "month"]).reset_index(drop=True)
    g = panel.groupby("h3", sort=False)["accident_count"]

    panel["had_accident"] = panel["accident_count"] > 0
    panel["accidents_prev_1m"] = g.shift(1).fillna(0).astype("int32")

    for window in [3, 6, 12]:
        panel[f"accidents_prev_{window}m"] = (
            panel.groupby("h3", sort=False)["accident_count"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
            .fillna(0)
            .astype("int32")
        )

    panel["target_count_next_month"] = g.shift(-1)

    if drop_last_month_without_target:
        panel = panel[panel["target_count_next_month"].notna()].copy()

    panel["target_count_next_month"] = (
        panel["target_count_next_month"]
        .fillna(0)
        .astype("int32")
    )
    panel["target_had_accident_next_month"] = panel["target_count_next_month"] > 0

    panel["year"] = panel["month"].dt.year.astype("int16")
    panel["month_num"] = panel["month"].dt.month.astype("int8")

    return panel[[
        "h3",
        "month",
        "year",
        "month_num",
        "accident_count",
        "had_accident",
        "accidents_prev_1m",
        "accidents_prev_3m",
        "accidents_prev_6m",
        "accidents_prev_12m",
        "target_count_next_month",
        "target_had_accident_next_month",
    ]]


def build_panel_and_write(
    positive: pd.DataFrame,
    min_month: pd.Timestamp,
    max_month: pd.Timestamp,
    output_path: Path = H3_MONTH_PANEL_PARQUET,
    bbox: tuple[float, float, float, float] | None = BBOX,
    h3_resolution: int = H3_RESOLUTION,
    include_all_h3_cells_in_bbox: bool = INCLUDE_ALL_H3_CELLS_IN_BBOX,
    cell_chunk_size: int = CELL_CHUNK_SIZE,
    drop_last_month_without_target: bool = DROP_LAST_MONTH_WITHOUT_TARGET,
    compression: str = COMPRESSION,
) -> None:
    if output_path.exists():
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cells = get_panel_cells(
        positive=positive,
        bbox=bbox,
        h3_resolution=h3_resolution,
        include_all_h3_cells_in_bbox=include_all_h3_cells_in_bbox,
    )
    months = make_month_range(min_month, max_month)

    estimated_rows = len(cells) * len(months)
    if drop_last_month_without_target:
        estimated_rows -= len(cells)

    print()
    print("=" * 100)
    print("STEP 2: Build full H3 x month panel")
    print(f"Months: {len(months):,}")
    print(f"Cells: {len(cells):,}")
    print(f"Estimated output rows: {estimated_rows:,}")
    print(f"Output: {output_path}")
    print(f"Cell chunk size: {cell_chunk_size:,}")

    positive = positive.copy()
    positive["month"] = pd.to_datetime(positive["month"])

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    started_at = time.time()
    last_progress_at = started_at
    cell_chunks = list(chunked(cells, cell_chunk_size))

    try:
        for chunk_idx, cell_chunk in enumerate(cell_chunks, start=1):
            base = pd.MultiIndex.from_product(
                [cell_chunk, months],
                names=["h3", "month"],
            ).to_frame(index=False)

            part_positive = positive[positive["h3"].isin(cell_chunk)]
            panel = base.merge(part_positive, on=["h3", "month"], how="left")
            panel["accident_count"] = panel["accident_count"].fillna(0).astype("int32")
            panel = add_lags_and_targets(
                panel,
                drop_last_month_without_target=drop_last_month_without_target,
            )

            table = pa.Table.from_pandas(panel, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    where=output_path,
                    schema=table.schema.remove_metadata(),
                    compression=compression,
                    version="2.6",
                    use_compliant_nested_type=True,
                    store_schema=False,
                )

            table = table.cast(writer.schema)
            writer.write_table(table)
            rows_written += panel.shape[0]

            now = time.time()
            if now - last_progress_at >= PROGRESS_EVERY_SECONDS or chunk_idx == len(cell_chunks):
                elapsed = max(now - started_at, 1e-9)
                speed = rows_written / elapsed
                pct = chunk_idx / len(cell_chunks) * 100
                eta = (estimated_rows - rows_written) / speed if speed > 0 else None
                eta_text = f"{eta / 60:.1f} min" if eta is not None else "unknown"

                print(
                    f"[write] chunks={chunk_idx:,}/{len(cell_chunks):,} ({pct:.2f}%) | "
                    f"rows_written={rows_written:,}/{estimated_rows:,} | "
                    f"speed={speed:,.0f} rows/s | eta={eta_text}",
                    flush=True,
                )
                last_progress_at = now

    finally:
        if writer is not None:
            writer.close()

    print()
    print("Done.")
    print(f"Rows written: {rows_written:,}")
    print(f"Saved to: {output_path}")


def build_h3_month_panel(
    input_path: Path = CRASHES_SLIM_PARQUET,
    output_path: Path = H3_MONTH_PANEL_PARQUET,
    h3_resolution: int = H3_RESOLUTION,
    bbox: tuple[float, float, float, float] | None = BBOX,
    include_all_h3_cells_in_bbox: bool = INCLUDE_ALL_H3_CELLS_IN_BBOX,
    start_date: str | None = START_DATE,
    end_date: str | None = END_DATE,
    read_batch_size: int = READ_BATCH_SIZE,
    cell_chunk_size: int = CELL_CHUNK_SIZE,
    drop_last_month_without_target: bool = DROP_LAST_MONTH_WITHOUT_TARGET,
    compression: str = COMPRESSION,
) -> None:
    positive, min_month, max_month = aggregate_crashes_to_h3_month(
        input_path=input_path,
        h3_resolution=h3_resolution,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        read_batch_size=read_batch_size,
    )
    build_panel_and_write(
        positive=positive,
        min_month=min_month,
        max_month=max_month,
        output_path=output_path,
        bbox=bbox,
        h3_resolution=h3_resolution,
        include_all_h3_cells_in_bbox=include_all_h3_cells_in_bbox,
        cell_chunk_size=cell_chunk_size,
        drop_last_month_without_target=drop_last_month_without_target,
        compression=compression,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an H3 x month modeling panel.")
    parser.add_argument("--input", type=Path, default=CRASHES_SLIM_PARQUET)
    parser.add_argument("--output", type=Path, default=H3_MONTH_PANEL_PARQUET)
    parser.add_argument("--h3-resolution", type=int, default=H3_RESOLUTION)
    parser.add_argument("--bbox", type=parse_bbox, default=BBOX)
    parser.add_argument(
        "--include-all-h3-cells-in-bbox",
        action=argparse.BooleanOptionalAction,
        default=INCLUDE_ALL_H3_CELLS_IN_BBOX,
    )
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--read-batch-size", type=int, default=READ_BATCH_SIZE)
    parser.add_argument("--cell-chunk-size", type=int, default=CELL_CHUNK_SIZE)
    parser.add_argument("--keep-last-month-without-target", action="store_true")
    parser.add_argument("--compression", default=COMPRESSION)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_h3_month_panel(
        input_path=args.input,
        output_path=args.output,
        h3_resolution=args.h3_resolution,
        bbox=args.bbox,
        include_all_h3_cells_in_bbox=args.include_all_h3_cells_in_bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        read_batch_size=args.read_batch_size,
        cell_chunk_size=args.cell_chunk_size,
        drop_last_month_without_target=not args.keep_last_month_without_target,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
