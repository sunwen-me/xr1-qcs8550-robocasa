#!/usr/bin/env python3
"""Capture official HF BF16 action references for fixed RoboCasa requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--gpu-memory", default="10GiB")
    parser.add_argument("--cpu-memory", default="24GiB")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    manifest_path = args.requests / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output / "manifest.json"
    existing: dict[str, Any] = {}
    if output_manifest.exists():
        old = json.loads(output_manifest.read_text(encoding="utf-8"))
        if old.get("checkpoint") != str(args.checkpoint) or old.get("processor") != str(args.processor):
            raise SystemExit("existing BF16 reference manifest belongs to different checkpoint/processor")
        existing = {row["sample"]: row for row in old.get("samples", [])}

    print(f"loading official BF16 checkpoint from {args.checkpoint}", flush=True)
    start = time.time()
    model = AutoModel.from_pretrained(
        str(args.checkpoint),
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.processor), trust_remote_code=True, use_fast=False)
    print(
        f"loaded in {time.time()-start:.1f}s; device_map={getattr(model, 'hf_device_map', {})}; "
        f"gpu_allocated={torch.cuda.memory_allocated()/1024**3:.2f}GiB",
        flush=True,
    )

    output_rows = list(existing.values())
    for index, row in enumerate(manifest["samples"]):
        sample = str(row["sample"])
        request_path = Path(row["request"])
        if existing.get(sample, {}).get("request_sha256") == row["request_sha256"]:
            prior = existing[sample]
            prior_path = Path(prior["path"])
            if prior_path.exists() and sha256(prior_path) == prior.get("sha256"):
                print(f"skip existing {sample}", flush=True)
                continue
        with np.load(request_path, allow_pickle=False) as archive:
            request = {name: archive[name].copy() for name in archive.files}
        inputs = {
            name: torch.from_numpy(request[name])
            for name in ("input_ids", "attention_mask", "image_grid_thw", "pixel_values", "state", "action_mask")
        }
        for name, value in inputs.items():
            if value.is_floating_point():
                inputs[name] = value.to(device="cuda", dtype=torch.bfloat16)
            else:
                inputs[name] = value.to("cuda")
        seed = int(request["seed"].reshape(-1)[0])
        inputs["seed"] = seed
        tic = time.time()
        with torch.inference_mode():
            output = model(**inputs).actions.detach().float().cpu()
        normalized = output.numpy().astype(np.float32, copy=False)
        if normalized.shape != (1, 10, 60) or not np.isfinite(normalized).all():
            raise ValueError(f"{sample}: invalid BF16 action shape/values: {normalized.shape}")
        decoded = processor.decode_action(output, robot_type=str(manifest["robot_type"])).float().numpy()
        if decoded.shape != (1, 10, 60) or not np.isfinite(decoded).all():
            raise ValueError(f"{sample}: invalid decoded BF16 action: {decoded.shape}")
        reference_path = args.output / f"{index:03d}_{sample}_bf16.npz"
        np.savez(reference_path, normalized_action=normalized, decoded_action=decoded)
        record = {
            "sample": sample,
            "task": row["task"],
            "seed": int(row["seed"]),
            "request": str(request_path),
            "request_sha256": row["request_sha256"],
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "inference_seconds": time.time() - tic,
            "normalized_action_shape": list(normalized.shape),
            "decoded_action_shape": list(decoded.shape),
            "finite": True,
            "normalized_range": [float(normalized.min()), float(normalized.max())],
            "decoded_range": [float(decoded.min()), float(decoded.max())],
        }
        existing[sample] = record
        print(
            f"[{index+1}/{len(manifest['samples'])}] {sample}: "
            f"{record['inference_seconds']:.2f}s decoded={record['decoded_range']}",
            flush=True,
        )

    ordered = [existing[row["sample"]] for row in manifest["samples"]]
    result = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest_tree(args.checkpoint),
        "processor": str(args.processor),
        "processor_sha256": digest_tree(args.processor),
        "reference_runtime": "official AutoModel BF16 forward; matching official processor inputs and action decoding",
        "device_map": getattr(model, "hf_device_map", {}),
        "torch_version": torch.__version__,
        "samples": ordered,
    }
    output_manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"captured {len(ordered)} BF16 references in {args.output}", flush=True)


if __name__ == "__main__":
    main()
