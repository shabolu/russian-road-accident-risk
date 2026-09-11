from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from road_accident_risk.paths import CRASHES_PARQUET, CRASHES_SLIM_PARQUET
from road_accident_risk.utils import format_size


BATCH_SIZE = 100_000
COMPRESSION = "snappy"
PROGRESS_EVERY_SECONDS = 3


SLIM_SCHEMA = pa.schema([
    ("datetime", pa.timestamp("us")),
    ("lat", pa.float64()),
    ("lng", pa.float64()),
    ("light_type", pa.string()),
    ("traffic_area_state", pa.string()),
    ("weather_json", pa.string()),
    ("accident_type", pa.string()),
    ("dead_count", pa.int16()),
    ("injured_count", pa.int16()),
    ("has_fatal", pa.bool_()),
    ("has_injured", pa.bool_()),
    ("severity_weight", pa.int16()),
])


def progress_line(
    rows_read: int,
    rows_written: int,
    total_rows: int,
    started_at: float,
) -> str:
    elapsed = max(time.time() - started_at, 1e-9)
    speed = rows_read / elapsed
    pct = rows_read / total_rows * 100 if total_rows else 100
    eta = (total_rows - rows_read) / speed if speed > 0 else None
    eta_text = f"{eta / 60:.1f} min" if eta is not None else "unknown"

    return (
        "[progress] "
        f"read={rows_read:,}/{total_rows:,} ({pct:.2f}%) | "
        f"written={rows_written:,} | "
        f"speed={speed:,.0f} rows/s | "
        f"eta={eta_text}"
    )


def ensure_columns_exist(schema: pa.Schema, required_columns: list[str]) -> None:
    existing = set(schema.names)
    missing = [col for col in required_columns if col not in existing]

    if missing:
        raise ValueError(
            "Missing required columns in input parquet: "
            + ", ".join(missing)
        )


def coalesce(left: pa.Array, right: pa.Array) -> pa.Array:
    return pc.if_else(pc.is_null(left), right, left)


def to_float64(array: pa.Array) -> pa.Array:
    return pc.cast(array, pa.float64(), safe=False)


def to_int16_fill_zero(array: pa.Array) -> pa.Array:
    filled = pc.fill_null(array, 0)
    return pc.cast(filled, pa.int16(), safe=False)


def parse_datetime(array: pa.Array) -> pa.Array:
    return pc.strptime(
        array,
        format="%Y-%m-%dT%H:%M:%S",
        unit="us",
        error_is_null=True,
    )


def make_slim_table(batch: pa.RecordBatch) -> pa.Table:
    table = pa.Table.from_batches([batch])

    date_time = table["DATE_TIME"]
    point_lat = to_float64(table["POINT_LAT"])
    point_lng = to_float64(table["POINT_LNG"])

    # In the source LAT/LNG scalar fields are swapped:
    # LAT = longitude, LNG = latitude. GeoJSON POINT_* is preferred.
    fallback_lat = to_float64(table["LNG"])
    fallback_lng = to_float64(table["LAT"])

    lat = coalesce(point_lat, fallback_lat)
    lng = coalesce(point_lng, fallback_lng)

    dead_count = to_int16_fill_zero(table["LOST_AMOUNT"])
    injured_count = to_int16_fill_zero(table["SUFFER_AMOUNT"])

    has_fatal = pc.greater(dead_count, pa.scalar(0, pa.int16()))
    has_injured = pc.greater(injured_count, pa.scalar(0, pa.int16()))

    severity_weight = pc.add(
        pc.add(pa.scalar(1, pa.int16()), injured_count),
        pc.multiply(dead_count, pa.scalar(5, pa.int16())),
    )
    severity_weight = pc.cast(severity_weight, pa.int16(), safe=False)

    slim = pa.Table.from_arrays(
        [
            parse_datetime(date_time),
            lat,
            lng,
            table["LIGHT_TYPE"],
            table["TRAFFIC_AREA_STATE"],
            table["CLOUDS_JSON"],
            table["EM_TYPE"],
            dead_count,
            injured_count,
            has_fatal,
            has_injured,
            severity_weight,
        ],
        schema=SLIM_SCHEMA,
    )

    mask = pc.and_(
        pc.and_(
            pc.and_(
                pc.invert(pc.is_null(slim["datetime"])),
                pc.invert(pc.is_null(slim["lat"])),
            ),
            pc.invert(pc.is_null(slim["lng"])),
        ),
        pc.and_(
            pc.and_(
                pc.greater_equal(slim["lat"], pa.scalar(41.0)),
                pc.less_equal(slim["lat"], pa.scalar(82.0)),
            ),
            pc.and_(
                pc.greater_equal(slim["lng"], pa.scalar(19.0)),
                pc.less_equal(slim["lng"], pa.scalar(180.0)),
            ),
        ),
    )

    return slim.filter(mask).cast(SLIM_SCHEMA)


def make_crashes_slim(
    input_path: Path = CRASHES_PARQUET,
    output_path: Path = CRASHES_SLIM_PARQUET,
    batch_size: int = BATCH_SIZE,
    compression: str = COMPRESSION,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    required_columns = [
        "DATE_TIME",
        "LAT",
        "LNG",
        "POINT_LAT",
        "POINT_LNG",
        "LIGHT_TYPE",
        "TRAFFIC_AREA_STATE",
        "CLOUDS_JSON",
        "LOST_AMOUNT",
        "SUFFER_AMOUNT",
        "EM_TYPE",
    ]

    parquet_file = pq.ParquetFile(input_path)
    ensure_columns_exist(parquet_file.schema_arrow, required_columns)

    total_rows = parquet_file.metadata.num_rows
    input_size = input_path.stat().st_size

    print(f"Input: {input_path}")
    print(f"Input size: {format_size(input_size)}")
    print(f"Rows: {total_rows:,}")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size:,}")
    print("Coordinate rule: lat = POINT_LAT fallback LNG; lng = POINT_LNG fallback LAT")
    print("Starting slim conversion...")

    writer: pq.ParquetWriter | None = None
    rows_read = 0
    rows_written = 0
    started_at = time.time()
    last_progress_at = started_at

    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=required_columns,
        ):
            rows_read += batch.num_rows
            slim_table = make_slim_table(batch)

            if writer is None:
                writer = pq.ParquetWriter(
                    where=output_path,
                    schema=SLIM_SCHEMA,
                    compression=compression,
                    version="2.6",
                    use_compliant_nested_type=True,
                    store_schema=False,
                )

            if slim_table.num_rows > 0:
                writer.write_table(slim_table)
                rows_written += slim_table.num_rows

            now = time.time()
            if now - last_progress_at >= PROGRESS_EVERY_SECONDS or rows_read == total_rows:
                print(
                    progress_line(
                        rows_read=rows_read,
                        rows_written=rows_written,
                        total_rows=total_rows,
                        started_at=started_at,
                    ),
                    flush=True,
                )
                last_progress_at = now

    finally:
        if writer is not None:
            writer.close()

    print("\nDone.")
    print(f"Rows read: {rows_read:,}")
    print(f"Rows written: {rows_written:,}")
    print(f"Dropped rows: {rows_read - rows_written:,}")
    print(f"Saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a compact crash parquet table.")
    parser.add_argument("--input", type=Path, default=CRASHES_PARQUET)
    parser.add_argument("--output", type=Path, default=CRASHES_SLIM_PARQUET)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--compression", default=COMPRESSION)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    make_crashes_slim(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
