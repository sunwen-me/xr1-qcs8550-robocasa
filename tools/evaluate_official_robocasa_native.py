#!/usr/bin/env python3
"""Compare official RoboCasa native QNN outputs with fixed BF16 references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    active = np.broadcast_to(np.asarray(mask, dtype=bool), pred.shape)
    p = pred[active]
    r = ref[active]
    delta = p - r
    denom = float(np.linalg.norm(r))
    cosine = float(np.dot(p, r) / (np.linalg.norm(p) * denom + 1e-12))
    return {
        "finite": bool(np.isfinite(pred).all() and np.isfinite(ref).all()),
        "active_value_count": int(active.sum()),
        "relative_l2_percent": float(np.linalg.norm(delta) / (denom + 1e-12) * 100.0),
        "cosine_similarity": cosine,
        "active_mae": float(np.mean(np.abs(delta))),
        "active_max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "pred_active_l2": float(np.linalg.norm(p)),
        "reference_active_l2": denom,
        "pred_to_reference_norm_ratio": float(np.linalg.norm(p) / (denom + 1e-12)),
        "full60_mae": float(np.mean(np.abs(pred - ref))),
        "full60_max_abs_error": float(np.max(np.abs(pred - ref))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.bundle / "validation_cases.json").read_text(encoding="utf-8"))
    rows = []
    for case in manifest["cases"]:
        name = case["sample"]
        pred_path = args.output / name / "raw_action.npy"
        request_path = args.bundle / case["request"]
        reference_path = args.bundle / case["reference"]
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        pred = np.load(pred_path, allow_pickle=False)
        with np.load(reference_path, allow_pickle=False) as ref_archive:
            ref = np.asarray(ref_archive["normalized_action"], dtype=np.float32)
        with np.load(request_path, allow_pickle=False) as request:
            mask = np.asarray(request["dit__action_mask"], dtype=np.float32)
        if pred.shape != (1, 10, 60) or ref.shape != pred.shape or mask.shape != pred.shape:
            raise ValueError(f"{name}: shape mismatch pred={pred.shape} ref={ref.shape} mask={mask.shape}")
        row = {
            "sample": name,
            "task": case["task"],
            "finite": bool(np.isfinite(pred).all()),
            "metrics": metrics(pred, ref, mask > 0),
        }
        stats_path = args.bundle / "stats/action_stats_10.npz"
        with np.load(stats_path, allow_pickle=False) as stats:
            decoded_pred = pred * np.asarray(stats["std"], dtype=np.float32) + np.asarray(stats["mean"], dtype=np.float32)
        with np.load(reference_path, allow_pickle=False) as ref_archive:
            decoded_ref = np.asarray(ref_archive["decoded_action"], dtype=np.float32)
        row["decoded_metrics"] = metrics(decoded_pred, decoded_ref, mask > 0)
        row["near_zero_collapse"] = bool(
            row["metrics"]["reference_active_l2"] > 1e-6
            and row["metrics"]["pred_to_reference_norm_ratio"] < 0.5
        )
        rows.append(row)

    tasks = {}
    for task in sorted({row["task"] for row in rows}):
        selected = [row for row in rows if row["task"] == task]
        keys = (
            "relative_l2_percent",
            "cosine_similarity",
            "active_mae",
            "active_max_abs_error",
            "pred_to_reference_norm_ratio",
        )
        tasks[task] = {
            "sample_count": len(selected),
            **{
                key: {
                    "mean": float(np.mean([row["metrics"][key] for row in selected])),
                    "max": float(np.max([row["metrics"][key] for row in selected])),
                }
                for key in keys
            },
        }
    report = {
        "schema_version": 1,
        "checkpoint_sha256": json.loads((args.bundle / "board_manifest.json").read_text())["provenance"]["checkpoint_sha256"],
        "sample_count": len(rows),
        "all_finite": all(row["finite"] for row in rows),
        "near_zero_collapse_count": sum(row["near_zero_collapse"] for row in rows),
        "cross_sample_active_variation": {},
        "overall": {
            key: {
                "mean": float(np.mean([row["metrics"][key] for row in rows])),
                "max": float(np.max([row["metrics"][key] for row in rows])),
            }
            for key in (
                "relative_l2_percent",
                "cosine_similarity",
                "active_mae",
                "active_max_abs_error",
                "pred_to_reference_norm_ratio",
            )
        },
        "by_task": tasks,
        "samples": rows,
    }
    prediction_stack = []
    reference_stack = []
    mask_stack = []
    decoded_prediction_stack = []
    decoded_reference_stack = []
    for case in manifest["cases"]:
        name = case["sample"]
        prediction_stack.append(np.load(args.output / name / "raw_action.npy", allow_pickle=False))
        with np.load(args.bundle / case["reference"], allow_pickle=False) as archive:
            reference_stack.append(np.asarray(archive["normalized_action"], dtype=np.float32))
            decoded_reference_stack.append(np.asarray(archive["decoded_action"], dtype=np.float32))
        with np.load(args.bundle / case["request"], allow_pickle=False) as archive:
            mask_stack.append(np.asarray(archive["dit__action_mask"], dtype=bool))
        with np.load(args.bundle / "stats/action_stats_10.npz", allow_pickle=False) as stats:
            decoded_prediction_stack.append(
                prediction_stack[-1] * np.asarray(stats["std"], dtype=np.float32)
                + np.asarray(stats["mean"], dtype=np.float32)
            )
    pred_stack = np.concatenate(prediction_stack, axis=0)
    ref_stack = np.concatenate(reference_stack, axis=0)
    masks = np.concatenate(mask_stack, axis=0)
    pred_decoded_stack = np.concatenate(decoded_prediction_stack, axis=0)
    ref_decoded_stack = np.concatenate(decoded_reference_stack, axis=0)
    active = np.broadcast_to(masks, pred_stack.shape)
    report["cross_sample_active_variation"] = {
        "qnn_mean_per_time_dimension_std": float(np.mean(np.std(pred_stack[active].reshape(len(rows), -1), axis=0))),
        "bf16_mean_per_time_dimension_std": float(np.mean(np.std(ref_stack[active].reshape(len(rows), -1), axis=0))),
        "qnn_active_std": float(np.std(pred_stack[active])),
        "bf16_active_std": float(np.std(ref_stack[active])),
        "active_dimensions": np.where(np.any(masks, axis=(0, 1)))[0].tolist(),
        "active_dimension_count": int(np.any(masks, axis=(0, 1)).sum()),
    }
    report["overall"]["decoded_active_relative_l2_percent"] = {
        "mean": float(np.mean([row["decoded_metrics"]["relative_l2_percent"] for row in rows])),
        "max": float(np.max([row["decoded_metrics"]["relative_l2_percent"] for row in rows])),
    }
    report["overall"]["decoded_active_mae"] = {
        "mean": float(np.mean([row["decoded_metrics"]["active_mae"] for row in rows])),
        "max": float(np.max([row["decoded_metrics"]["active_mae"] for row in rows])),
    }
    report["overall"]["decoded_active_max_abs_error"] = {
        "mean": float(np.mean([row["decoded_metrics"]["active_max_abs_error"] for row in rows])),
        "max": float(np.max([row["decoded_metrics"]["active_max_abs_error"] for row in rows])),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "comparison_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
