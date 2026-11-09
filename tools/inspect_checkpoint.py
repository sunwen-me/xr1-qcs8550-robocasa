#!/usr/bin/env python3
"""Inspect an XR-1 checkpoint without materializing its tensors."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr1_port.contract import XR1Contract


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def _dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else None


def inspect(path: Path, sample_limit: int) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="meta",
        mmap=True,
        weights_only=False,
    )
    report: dict[str, Any] = {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "top_level_type": type(checkpoint).__name__,
        "top_level_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else None,
    }
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("module", "state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                report["state_key"] = key
                break
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state is not a mapping: {type(state).__name__}")

    prefixes = collections.Counter()
    dtypes = collections.Counter()
    samples: list[dict[str, Any]] = []
    xr1_like = 0
    for index, (name, value) in enumerate(state.items()):
        name = str(name)
        prefixes[name.split(".", 1)[0]] += 1
        dtypes[_dtype(value)] += 1
        if index < sample_limit:
            samples.append({"name": name, "shape": _shape(value), "dtype": _dtype(value)})
        if "vlm" in name or "dit" in name:
            xr1_like += 1

    report.update(
        {
            "tensor_count": len(state),
            "prefix_counts": dict(prefixes),
            "dtype_counts": dict(dtypes),
            "xr1_named_tensor_count": xr1_like,
            "samples": samples,
            "expected_contract": XR1Contract().to_dict(),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="path to the trained checkpoint")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect(Path(args.checkpoint), args.sample_limit)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
