from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

from road_accident_risk.paths import DATASET_JSON, PARQUET_DIR
from road_accident_risk.utils import format_size, json_dumps, to_float, to_int, to_str


ITEMS_PATH = "item"
CHUNK_SIZE = 50_000
PROGRESS_EVERY_SECONDS = 5
COMPRESSION = "zstd"


CRASH_SCHEMA = pa.schema([
    ("crash_uid", pa.int64()),
    ("ADDITIONAL_ATTR", pa.int64()),
    ("CHILDREN_ATTR", pa.int64()),
    ("DATE_TIME", pa.string()),
    ("EMTP_NUMBER", pa.int64()),
    ("EM_NUMBER", pa.int64()),
    ("EM_TYPE", pa.string()),
    ("HOUSE", pa.string()),
    ("LAT", pa.float64()),
    ("LNG", pa.float64()),
    ("LIGHT_TYPE", pa.string()),
    ("LOST_AMOUNT", pa.int64()),
    ("MT_RATE", pa.string()),
    ("NP", pa.string()),
    ("PERS_AMOUNT", pa.int64()),
    ("PLACE", pa.string()),
    ("ROAD_KM_M", pa.float64()),
    ("ROADS", pa.string()),
    ("ROAD_SIGN", pa.string()),
    ("ROAD_TYPE", pa.string()),
    ("SCHEMA_ID", pa.int64()),
    ("STREET", pa.string()),
    ("STREET_SIGN", pa.string()),
    ("SUFFER_AMOUNT", pa.int64()),
    ("TRAFFIC_AREA_STATE", pa.string()),
    ("VEH_AMOUNT", pa.int64()),
    ("POINT_TYPE", pa.string()),
    ("POINT_LNG", pa.float64()),
    ("POINT_LAT", pa.float64()),
    ("CLOUDS_JSON", pa.string()),
    ("DEFECTS_JSON", pa.string()),
    ("MOTION_INFLUENCES_JSON", pa.string()),
    ("RD_CONSTR_HERES_JSON", pa.string()),
    ("RD_CONSTR_THERES_JSON", pa.string()),
    ("REGIONS_JSON", pa.string()),
])

VEHICLE_SCHEMA = pa.schema([
    ("crash_uid", pa.int64()),
    ("vehicle_index", pa.int64()),
    ("N", pa.int64()),
    ("COLOR", pa.string()),
    ("ESCAPE", pa.string()),
    ("MARK", pa.string()),
    ("MODEL", pa.string()),
    ("OKFS", pa.string()),
    ("OKOPF", pa.string()),
    ("PROD_TYPE", pa.string()),
    ("RUDDER_TYPE", pa.string()),
    ("VEHICLE_YEAR", pa.int64()),
    ("DAMAGE_DISPOSITION_JSON", pa.string()),
    ("TECH_FAILURE_TYPE_JSON", pa.string()),
])

PERSON_SCHEMA = pa.schema([
    ("crash_uid", pa.int64()),
    ("person_scope", pa.string()),
    ("vehicle_index", pa.int64()),
    ("person_index", pa.int64()),
    ("N", pa.int64()),
    ("PART_TYPE", pa.string()),
    ("SEX", pa.string()),
    ("ESCAPE", pa.string()),
    ("HV_TYPE", pa.string()),
    ("CHILD_SAFETY_TYPE", pa.string()),
    ("DRIVER_SERVICE_LENGTH", pa.int64()),
    ("INJURED_CARD", pa.string()),
    ("MED_RESULT_PERMILLE", pa.int64()),
    ("SAFETY_BELT", pa.string()),
    ("ATTENDANT_PDD_DERANGEMENT_JSON", pa.string()),
    ("MAIN_PDD_DERANGEMENT_JSON", pa.string()),
])


def get_point_lng_lat(point: Any) -> tuple[str | None, float | None, float | None]:
    if not isinstance(point, dict):
        return None, None, None

    point_type = to_str(point.get("type"))
    coords = point.get("coordinates")

    if not isinstance(coords, list) or len(coords) < 2:
        return point_type, None, None

    return point_type, to_float(coords[0]), to_float(coords[1])


def make_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    if not rows:
        return pa.Table.from_pylist([], schema=schema)

    field_names = set(schema.names)
    normalized_rows = [
        {field: row.get(field) for field in field_names}
        for row in rows
    ]

    return pa.Table.from_pylist(normalized_rows, schema=schema)


def print_progress(
    processed_items: int,
    written_crashes: int,
    written_vehicles: int,
    written_persons: int,
    current_pos: int,
    file_size: int,
    started_at: float,
) -> None:
    elapsed = max(time.time() - started_at, 1e-9)
    percent = current_pos / file_size * 100 if file_size else 0
    speed_items = processed_items / elapsed
    speed_bytes = current_pos / elapsed

    eta = None
    if speed_bytes > 0 and file_size > 0:
        eta = (file_size - current_pos) / speed_bytes

    eta_text = f"{eta / 60:.1f} min" if eta is not None else "unknown"

    print(
        "[progress] "
        f"items={processed_items:,} | "
        f"crashes={written_crashes:,} | "
        f"vehicles={written_vehicles:,} | "
        f"persons={written_persons:,} | "
        f"file={format_size(current_pos)}/{format_size(file_size)} "
        f"({percent:.2f}%) | "
        f"speed={speed_items:,.0f} items/s, {format_size(int(speed_bytes))}/s | "
        f"eta={eta_text}",
        flush=True,
    )


def build_crash_row(item: dict[str, Any], crash_uid: int) -> dict[str, Any]:
    point_type, point_lng, point_lat = get_point_lng_lat(item.get("POINT"))

    return {
        "crash_uid": crash_uid,
        "ADDITIONAL_ATTR": to_int(item.get("ADDITIONAL_ATTR")),
        "CHILDREN_ATTR": to_int(item.get("CHILDREN_ATTR")),
        "DATE_TIME": to_str(item.get("DATE_TIME")),
        "EMTP_NUMBER": to_int(item.get("EMTP_NUMBER")),
        "EM_NUMBER": to_int(item.get("EM_NUMBER")),
        "EM_TYPE": to_str(item.get("EM_TYPE")),
        "HOUSE": to_str(item.get("HOUSE")),
        "LAT": to_float(item.get("LAT")),
        "LNG": to_float(item.get("LNG")),
        "LIGHT_TYPE": to_str(item.get("LIGHT_TYPE")),
        "LOST_AMOUNT": to_int(item.get("LOST_AMOUNT")),
        "MT_RATE": to_str(item.get("MT_RATE")),
        "NP": to_str(item.get("NP")),
        "PERS_AMOUNT": to_int(item.get("PERS_AMOUNT")),
        "PLACE": to_str(item.get("PLACE")),
        "ROAD_KM_M": to_float(item.get("ROAD_KM_M")),
        "ROADS": to_str(item.get("ROADS")),
        "ROAD_SIGN": to_str(item.get("ROAD_SIGN")),
        "ROAD_TYPE": to_str(item.get("ROAD_TYPE")),
        "SCHEMA_ID": to_int(item.get("SCHEMA_ID")),
        "STREET": to_str(item.get("STREET")),
        "STREET_SIGN": to_str(item.get("STREET_SIGN")),
        "SUFFER_AMOUNT": to_int(item.get("SUFFER_AMOUNT")),
        "TRAFFIC_AREA_STATE": to_str(item.get("TRAFFIC_AREA_STATE")),
        "VEH_AMOUNT": to_int(item.get("VEH_AMOUNT")),
        "POINT_TYPE": point_type,
        "POINT_LNG": point_lng,
        "POINT_LAT": point_lat,
        "CLOUDS_JSON": json_dumps(item.get("CLOUDS")),
        "DEFECTS_JSON": json_dumps(item.get("DEFECTS")),
        "MOTION_INFLUENCES_JSON": json_dumps(item.get("MOTION_INFLUENCES")),
        "RD_CONSTR_HERES_JSON": json_dumps(item.get("RD_CONSTR_HERES")),
        "RD_CONSTR_THERES_JSON": json_dumps(item.get("RD_CONSTR_THERES")),
        "REGIONS_JSON": json_dumps(item.get("REGIONS")),
    }


def build_vehicle_row(vehicle: dict[str, Any], crash_uid: int, vehicle_index: int) -> dict[str, Any]:
    return {
        "crash_uid": crash_uid,
        "vehicle_index": vehicle_index,
        "N": to_int(vehicle.get("N")),
        "COLOR": to_str(vehicle.get("COLOR")),
        "ESCAPE": to_str(vehicle.get("ESCAPE")),
        "MARK": to_str(vehicle.get("MARK")),
        "MODEL": to_str(vehicle.get("MODEL")),
        "OKFS": to_str(vehicle.get("OKFS")),
        "OKOPF": to_str(vehicle.get("OKOPF")),
        "PROD_TYPE": to_str(vehicle.get("PROD_TYPE")),
        "RUDDER_TYPE": to_str(vehicle.get("RUDDER_TYPE")),
        "VEHICLE_YEAR": to_int(vehicle.get("VEHICLE_YEAR")),
        "DAMAGE_DISPOSITION_JSON": json_dumps(vehicle.get("DAMAGE_DISPOSITION")),
        "TECH_FAILURE_TYPE_JSON": json_dumps(vehicle.get("TECH_FAILURE_TYPE")),
    }


def build_person_row(
    person: dict[str, Any],
    crash_uid: int,
    person_scope: str,
    person_index: int,
    vehicle_index: int | None = None,
) -> dict[str, Any]:
    return {
        "crash_uid": crash_uid,
        "person_scope": person_scope,
        "vehicle_index": vehicle_index,
        "person_index": person_index,
        "N": to_int(person.get("N")),
        "PART_TYPE": to_str(person.get("PART_TYPE")),
        "SEX": to_str(person.get("SEX")),
        "ESCAPE": to_str(person.get("ESCAPE")),
        "HV_TYPE": to_str(person.get("HV_TYPE")),
        "CHILD_SAFETY_TYPE": to_str(person.get("CHILD_SAFETY_TYPE")),
        "DRIVER_SERVICE_LENGTH": to_int(person.get("DRIVER_SERVICE_LENGTH")),
        "INJURED_CARD": to_str(person.get("INJURED_CARD")),
        "MED_RESULT_PERMILLE": to_int(person.get("MED_RESULT_PERMILLE")),
        "SAFETY_BELT": to_str(person.get("SAFETY_BELT")),
        "ATTENDANT_PDD_DERANGEMENT_JSON": json_dumps(person.get("ATTENDANT_PDD_DERANGEMENT")),
        "MAIN_PDD_DERANGEMENT_JSON": json_dumps(person.get("MAIN_PDD_DERANGEMENT")),
    }


def split_item(
    item: dict[str, Any],
    crash_uid: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    crash_row = build_crash_row(item, crash_uid)
    vehicle_rows: list[dict[str, Any]] = []
    person_rows: list[dict[str, Any]] = []

    vehicles = item.get("VEHICLES")
    if isinstance(vehicles, list):
        for vehicle_index, vehicle in enumerate(vehicles, start=1):
            if not isinstance(vehicle, dict):
                continue

            vehicle_rows.append(build_vehicle_row(vehicle, crash_uid, vehicle_index))

            vehicle_persons = vehicle.get("PERSONS")
            if isinstance(vehicle_persons, list):
                for person_index, person in enumerate(vehicle_persons, start=1):
                    if isinstance(person, dict):
                        person_rows.append(
                            build_person_row(
                                person=person,
                                crash_uid=crash_uid,
                                person_scope="vehicle",
                                person_index=person_index,
                                vehicle_index=vehicle_index,
                            )
                        )

    top_persons = item.get("PERSONS")
    if isinstance(top_persons, list):
        for person_index, person in enumerate(top_persons, start=1):
            if isinstance(person, dict):
                person_rows.append(
                    build_person_row(
                        person=person,
                        crash_uid=crash_uid,
                        person_scope="top_level",
                        person_index=person_index,
                        vehicle_index=None,
                    )
                )

    return crash_row, vehicle_rows, person_rows


class ParquetSink:
    def __init__(self, path: Path, schema: pa.Schema, compression: str = COMPRESSION) -> None:
        self.path = path
        self.schema = schema
        self.compression = compression
        self.writer: pq.ParquetWriter | None = None
        self.rows_written = 0

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        table = make_table(rows, self.schema)

        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                where=self.path,
                schema=self.schema,
                compression=self.compression,
            )

        self.writer.write_table(table)
        self.rows_written += len(rows)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def convert_json_to_parquet(
    input_path: Path = DATASET_JSON,
    output_dir: Path = PARQUET_DIR,
    items_path: str = ITEMS_PATH,
    chunk_size: int = CHUNK_SIZE,
    compression: str = COMPRESSION,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    file_size = input_path.stat().st_size

    crashes_sink = ParquetSink(output_dir / "crashes.parquet", CRASH_SCHEMA, compression)
    vehicles_sink = ParquetSink(output_dir / "vehicles.parquet", VEHICLE_SCHEMA, compression)
    persons_sink = ParquetSink(output_dir / "persons.parquet", PERSON_SCHEMA, compression)

    crashes_buffer: list[dict[str, Any]] = []
    vehicles_buffer: list[dict[str, Any]] = []
    persons_buffer: list[dict[str, Any]] = []

    processed_items = 0
    crash_uid = 0
    started_at = time.time()
    last_progress_at = started_at
    last_file_pos = 0

    print(f"Input file: {input_path}")
    print(f"File size: {format_size(file_size)}")
    print(f"Output dir: {output_dir}")
    print(f"Items path: {items_path}")
    print(f"Chunk size: {chunk_size:,}")
    print("Starting conversion...")

    try:
        with input_path.open("rb") as f:
            for item in ijson.items(f, items_path):
                if not isinstance(item, dict):
                    continue

                crash_uid += 1
                crash_row, vehicle_rows, person_rows = split_item(item, crash_uid)

                crashes_buffer.append(crash_row)
                vehicles_buffer.extend(vehicle_rows)
                persons_buffer.extend(person_rows)

                processed_items += 1

                if len(crashes_buffer) >= chunk_size:
                    crashes_sink.write_rows(crashes_buffer)
                    vehicles_sink.write_rows(vehicles_buffer)
                    persons_sink.write_rows(persons_buffer)

                    crashes_buffer.clear()
                    vehicles_buffer.clear()
                    persons_buffer.clear()

                now = time.time()
                if now - last_progress_at >= PROGRESS_EVERY_SECONDS:
                    last_file_pos = f.tell()
                    print_progress(
                        processed_items=processed_items,
                        written_crashes=crashes_sink.rows_written + len(crashes_buffer),
                        written_vehicles=vehicles_sink.rows_written + len(vehicles_buffer),
                        written_persons=persons_sink.rows_written + len(persons_buffer),
                        current_pos=last_file_pos,
                        file_size=file_size,
                        started_at=started_at,
                    )
                    last_progress_at = now

            crashes_sink.write_rows(crashes_buffer)
            vehicles_sink.write_rows(vehicles_buffer)
            persons_sink.write_rows(persons_buffer)

            crashes_buffer.clear()
            vehicles_buffer.clear()
            persons_buffer.clear()

            last_file_pos = f.tell()
            print_progress(
                processed_items=processed_items,
                written_crashes=crashes_sink.rows_written,
                written_vehicles=vehicles_sink.rows_written,
                written_persons=persons_sink.rows_written,
                current_pos=last_file_pos,
                file_size=file_size,
                started_at=started_at,
            )

    finally:
        crashes_sink.close()
        vehicles_sink.close()
        persons_sink.close()

    print("\nDone.")
    print(f"crashes:  {crashes_sink.rows_written:,} rows -> {output_dir / 'crashes.parquet'}")
    print(f"vehicles: {vehicles_sink.rows_written:,} rows -> {output_dir / 'vehicles.parquet'}")
    print(f"persons:  {persons_sink.rows_written:,} rows -> {output_dir / 'persons.parquet'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert dataset.json into normalized parquet tables.")
    parser.add_argument("--input", type=Path, default=DATASET_JSON)
    parser.add_argument("--output-dir", type=Path, default=PARQUET_DIR)
    parser.add_argument("--items-path", default=ITEMS_PATH)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--compression", default=COMPRESSION)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    convert_json_to_parquet(
        input_path=args.input,
        output_dir=args.output_dir,
        items_path=args.items_path,
        chunk_size=args.chunk_size,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
