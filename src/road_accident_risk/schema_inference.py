from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import ijson

from road_accident_risk.paths import DATASET_JSON
from road_accident_risk.utils import format_size


ITEMS_PATH = "item"
PROGRESS_EVERY_ITEMS = 10_000
PROGRESS_EVERY_SECONDS = 5
MAX_EXAMPLES = 5
MAX_STRING_EXAMPLE_LEN = 200


def get_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def make_node() -> dict[str, Any]:
    return {
        "types": Counter(),
        "count": 0,
        "null_count": 0,
        "examples": [],
        "properties": {},
        "items": None,
    }


def add_example(node: dict[str, Any], value: Any) -> None:
    if isinstance(value, str):
        example = value[:MAX_STRING_EXAMPLE_LEN]
    elif value is None or isinstance(value, (bool, int, float)):
        example = value
    else:
        return

    if example not in node["examples"] and len(node["examples"]) < MAX_EXAMPLES:
        node["examples"].append(example)


def update_node(node: dict[str, Any], value: Any) -> None:
    value_type = get_type_name(value)

    node["types"][value_type] += 1
    node["count"] += 1

    if value is None:
        node["null_count"] += 1
        add_example(node, value)
        return

    if isinstance(value, dict):
        for key, child_value in value.items():
            if key not in node["properties"]:
                node["properties"][key] = make_node()
            update_node(node["properties"][key], child_value)

    elif isinstance(value, list):
        if node["items"] is None:
            node["items"] = make_node()

        for item in value:
            update_node(node["items"], item)

    else:
        add_example(node, value)


def finalize_node(node: dict[str, Any], parent_count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "types": dict(node["types"]),
        "observed_count": node["count"],
        "null_count": node["null_count"],
    }

    if parent_count is not None:
        result["missing_count"] = max(parent_count - node["count"], 0)
        result["presence_ratio"] = round(node["count"] / parent_count, 6) if parent_count else None

    if node["examples"]:
        result["examples"] = node["examples"]

    if node["properties"]:
        result["properties"] = {
            key: finalize_node(child, parent_count=node["count"])
            for key, child in sorted(node["properties"].items())
        }

    if node["items"] is not None:
        result["items"] = finalize_node(node["items"])

    return result


def print_progress(
    processed_items: int,
    file_obj,
    file_size: int,
    started_at: float,
) -> None:
    elapsed = max(time.time() - started_at, 1e-9)
    current_pos = file_obj.tell()
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
        f"file={format_size(current_pos)}/{format_size(file_size)} "
        f"({percent:.2f}%) | "
        f"speed={speed_items:,.0f} items/s, {format_size(int(speed_bytes))}/s | "
        f"eta={eta_text}",
        flush=True,
    )


def infer_schema_streaming(
    path: Path,
    items_path: str = ITEMS_PATH,
    max_items: int | None = None,
) -> dict[str, Any]:
    root = make_node()

    file_size = path.stat().st_size
    started_at = time.time()
    last_progress_at = started_at
    processed_items = 0
    last_file_pos = 0

    print(f"Input file: {path}")
    print(f"File size: {format_size(file_size)}")
    print(f"Items path: {items_path}")
    print("Starting streaming schema inference...")

    with path.open("rb") as f:
        for item in ijson.items(f, items_path):
            update_node(root, item)
            processed_items += 1

            now = time.time()
            last_file_pos = f.tell()

            should_print_by_items = processed_items % PROGRESS_EVERY_ITEMS == 0
            should_print_by_time = now - last_progress_at >= PROGRESS_EVERY_SECONDS

            if should_print_by_items or should_print_by_time:
                print_progress(processed_items, f, file_size, started_at)
                last_progress_at = now

            if max_items is not None and processed_items >= max_items:
                print(f"Reached max_items={max_items:,}. Stopping.")
                last_file_pos = f.tell()
                break

        print_progress(processed_items, f, file_size, started_at)

    schema = finalize_node(root)
    schema["records_total_streamed"] = processed_items
    schema["items_path"] = items_path
    schema["source_file_size_bytes"] = file_size
    schema["last_file_position_bytes"] = last_file_pos

    return schema


def write_schema(
    input_path: Path = DATASET_JSON,
    output_path: Path | None = None,
    items_path: str = ITEMS_PATH,
    max_items: int | None = None,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_schema.json")

    schema = {
        "source_file": str(input_path),
        "schema": infer_schema_streaming(
            path=input_path,
            items_path=items_path,
            max_items=max_items,
        ),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"Saved schema to: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer a streaming schema for dataset.json.")
    parser.add_argument("--input", type=Path, default=DATASET_JSON)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--items-path", default=ITEMS_PATH)
    parser.add_argument("--max-items", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    write_schema(
        input_path=args.input,
        output_path=args.output,
        items_path=args.items_path,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()
