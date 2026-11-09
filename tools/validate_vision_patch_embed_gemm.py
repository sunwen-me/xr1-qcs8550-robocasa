#!/usr/bin/env python3
"""Compare the original and Gemm-patched Vision patch-stem outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    difference = actual - expected
    return {
        "shape": list(actual.shape),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(expected), 1e-12)),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(np.max(np.abs(difference))),
        "finite": bool(np.isfinite(actual).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.inputs, allow_pickle=False) as archive:
        pixel_key = "vision__pixel_values" if "vision__pixel_values" in archive else "pixel_values"
        pixel_values = np.asarray(archive[pixel_key], dtype=np.float32)
    feed = {"pixel_values": pixel_values}

    original = ort.InferenceSession(str(args.original), providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(str(args.candidate), providers=["CPUExecutionProvider"])
    original_names = [output.name for output in original.get_outputs()]
    candidate_names = [output.name for output in candidate.get_outputs()]
    if original_names != candidate_names:
        raise RuntimeError(f"output contract changed: {original_names} != {candidate_names}")
    expected = original.run(original_names, feed)
    actual = candidate.run(candidate_names, feed)
    metrics = {
        name: compare(candidate_value, original_value)
        for name, candidate_value, original_value in zip(candidate_names, actual, expected)
    }
    report = {
        "original": str(args.original),
        "candidate": str(args.candidate),
        "inputs": str(args.inputs),
        "input_shape": list(pixel_values.shape),
        "outputs": metrics,
        "pass": all(
            bool(row["finite"])
            and float(row["relative_l2"]) <= 1e-5
            and float(row["max_abs"]) <= 1e-4
            for row in metrics.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit("Vision Gemm candidate failed output equivalence")


if __name__ == "__main__":
    main()
