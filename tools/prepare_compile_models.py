#!/usr/bin/env python3
"""Create AI Hub-compatible graph directories from exported XR-1 graphs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import onnx


ALLOWED_SUFFIXES = {".onnx", ".data", ".encodings", ".bin"}


def external_locations(model: onnx.ModelProto) -> list[str]:
    locations = []
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        values = {item.key: item.value for item in initializer.external_data}
        location = values.get("location")
        if not location:
            raise ValueError(f"external initializer has no location: {initializer.name}")
        locations.append(location)
    return locations


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def copy_external_data_to_single_file(
    model: onnx.ModelProto,
    source_dir: Path,
    target_dir: Path,
) -> None:
    segments: list[tuple[str, int, int]] = []
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        values = {item.key: item.value for item in initializer.external_data}
        location = values.get("location")
        if not location:
            raise ValueError(f"external initializer has no location: {initializer.name}")
        source = source_dir / location
        if not source.is_file():
            raise FileNotFoundError(source)
        source_size = source.stat().st_size
        source_offset = int(values.get("offset", "0"))
        source_length = int(values.get("length", str(source_size - source_offset)))
        if source_offset < 0 or source_length < 0 or source_offset + source_length > source_size:
            raise ValueError(
                f"invalid external range for {initializer.name}: "
                f"offset={source_offset} length={source_length} file_size={source_size}"
            )
        segment = (location, source_offset, source_length)
        if segment not in segments:
            segments.append(segment)

    offsets: dict[tuple[str, int, int], tuple[int, int]] = {}
    with (target_dir / "model.data").open("wb") as output:
        offset = 0
        for location, source_offset, source_length in segments:
            source = source_dir / location
            with source.open("rb") as input_file:
                input_file.seek(source_offset)
                remaining = source_length
                while remaining:
                    chunk = input_file.read(min(16 * 1024 * 1024, remaining))
                    if not chunk:
                        raise OSError(f"unexpected EOF while reading {source}")
                    output.write(chunk)
                    remaining -= len(chunk)
            offsets[(location, source_offset, source_length)] = (offset, source_length)
            offset += source_length

    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        values = {item.key: item.value for item in initializer.external_data}
        location = values["location"]
        source = source_dir / location
        source_size = source.stat().st_size
        source_offset = int(values.get("offset", "0"))
        source_length = int(values.get("length", str(source_size - source_offset)))
        offset, length = offsets[(location, source_offset, source_length)]
        del initializer.external_data[:]
        for key, value in (
            ("location", "model.data"),
            ("offset", str(offset)),
            ("length", str(length)),
        ):
            item = initializer.external_data.add()
            item.key = key
            item.value = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.export_report.read_text(encoding="utf-8"))
    source_root = args.export_report.parent
    args.output_root.mkdir(parents=True, exist_ok=True)

    for graph in report["graphs"]:
        name = str(graph["name"])
        source_dir = source_root / name
        target_dir = args.output_root / name
        if target_dir.exists() and any(target_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty directory: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)

        source_model_path = source_dir / "model.onnx"
        target_model_path = target_dir / "model.onnx"
        model = onnx.load(str(source_model_path), load_external_data=False)
        locations = external_locations(model)
        if not locations:
            copy_file(source_model_path, target_model_path)
        else:
            copy_external_data_to_single_file(model, source_dir, target_dir)
            onnx.save_model(model, str(target_model_path), save_as_external_data=False)

        onnx.checker.check_model(str(target_model_path), full_check=False)
        unexpected = [path.name for path in target_dir.iterdir() if path.suffix not in ALLOWED_SUFFIXES]
        if unexpected:
            raise RuntimeError(f"AI Hub-incompatible files in {target_dir}: {unexpected}")
        print(f"{name}: {len(locations)} external tensors", flush=True)

    args.output_root.joinpath("export_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output_root / 'export_report.json'}")


if __name__ == "__main__":
    main()
