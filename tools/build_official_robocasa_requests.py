#!/usr/bin/env python3
"""Build official-checkpoint processor requests and fixed QCS8550 bundles.

This path intentionally mirrors the official RoboCasa evaluator: two base
views followed by the wrist view, 0.95 center-cropped 256px images, raw padded
proprioception, and action statistics from the checkpoint processor. It does
not apply the fold-training q01/q99 normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from xr1_qwen_geometry import (
    ACTION_DIM,
    HIDDEN_SIZE,
    SEQUENCE_LENGTH,
    XR1Geometry,
)


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


def numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    return np.ascontiguousarray(value)


def summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()) if value.size else None,
        "max": float(value.max()) if value.size else None,
        "finite": bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True,
    }


def messages(images: list[Image.Image], instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "The following observations are captured from multiple views.\n# Base View\n"},
                {"type": "image", "image": images[0]},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": "\n# Left-Wrist View\n"},
                {"type": "image", "image": images[2]},
                {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


def build_internal(
    request: dict[str, np.ndarray], geometry: XR1Geometry, pad_id: int, action_length: int
) -> dict[str, np.ndarray]:
    raw_ids = request["input_ids"].astype(np.int32, copy=False)
    raw_attn = request["attention_mask"].astype(np.int32, copy=False)
    grid = request["image_grid_thw"].astype(np.int32, copy=False)
    if raw_ids.ndim != 2 or raw_ids.shape[0] != 1 or raw_ids.shape[1] > SEQUENCE_LENGTH:
        raise ValueError(f"unsupported prompt token shape {raw_ids.shape}")
    if raw_attn.shape != raw_ids.shape or grid.shape != (3, 3):
        raise ValueError(f"bad processor geometry: attention={raw_attn.shape}, grid={grid.shape}")
    pixels = request["pixel_values"].astype(np.float32, copy=False)
    if pixels.shape != (768, 1536) or not np.isfinite(pixels).all():
        raise ValueError(f"bad official pixel tensor: {pixels.shape}")

    raw_positions = geometry.position_ids(raw_ids, grid, raw_attn)
    input_ids = np.full((1, SEQUENCE_LENGTH), pad_id, dtype=np.int32)
    attention = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int32)
    positions = np.ones((3, 1, SEQUENCE_LENGTH), dtype=np.int32)
    length = raw_ids.shape[1]
    input_ids[:, :length] = raw_ids
    attention[:, :length] = raw_attn
    positions[:, :, :length] = raw_positions
    visual_mask = (input_ids == geometry.image_token_id).astype(np.float32)[..., None]

    source_mask = request["action_mask"].astype(np.float32, copy=False)
    if source_mask.ndim != 3 or source_mask.shape[0] != 1 or source_mask.shape[2] != ACTION_DIM:
        raise ValueError(f"unexpected official action mask shape: {source_mask.shape}")
    if source_mask.shape[1] != action_length:
        raise ValueError(f"graph action horizon {action_length} differs from processor horizon {source_mask.shape[1]}")
    action_mask = source_mask

    state = request["state"].astype(np.float32, copy=False)
    if state.shape == (1, ACTION_DIM):
        state = state[:, None, :]
    if state.shape != (1, 1, ACTION_DIM) or not np.isfinite(state).all():
        raise ValueError(f"bad raw official state: {state.shape}")
    # The official inference server casts floating processor tensors to BF16
    # before the model call. Preserve that input rounding in the FP32 QNN bundle.
    state = torch.from_numpy(state).to(torch.bfloat16).float().numpy()

    # The HF inference server seeds torch immediately before randn_like on its
    # BF16 CUDA action mask. Use CUDA here to reproduce its initial noise; CPU
    # torch uses a different random stream.
    seed = int(request.get("seed", np.asarray(42)).reshape(-1)[0])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the official BF16 action-noise stream")
    torch.manual_seed(seed)
    bf16_mask = torch.from_numpy(action_mask).to(device="cuda", dtype=torch.bfloat16)
    noisy_action = torch.randn_like(bf16_mask).cpu().float().numpy()
    query_length = action_length + 2
    position_ids = torch.arange(query_length, dtype=torch.long).view(1, 1, -1)
    position_ids = position_ids.repeat(3, 1, 1) + int(raw_positions.max()) + 1
    dummy_action = torch.zeros((1, action_length, ACTION_DIM), dtype=torch.float32)
    with torch.no_grad():
        cos, sin = geometry.rotary(dummy_action, position_ids)
    cos, sin = cos.numpy().astype(np.float32), sin.numpy().astype(np.float32)
    cache_mask = np.where(attention[:, None, None, :] > 0, 0.0, -10000.0)
    cache_mask = np.broadcast_to(cache_mask, (1, 1, query_length, SEQUENCE_LENGTH))
    causal = np.tril(np.ones((query_length, query_length), dtype=bool))
    query_mask = np.where(causal, 0.0, -10000.0).astype(np.float32)[None, None]
    dit_attention = np.concatenate([cache_mask, query_mask], axis=-1).astype(np.float32, copy=False)
    visual = np.zeros((1, SEQUENCE_LENGTH, HIDDEN_SIZE), dtype=np.float32)

    return {
        "vision__pixel_values": pixels,
        "qwen__input_ids": input_ids,
        "qwen__attention_mask": attention,
        "qwen__position_ids": positions,
        "qwen__visual_pos_mask": visual_mask,
        "qwen__vision_embeds_padded": visual.copy(),
        "qwen__deepstack_visual_embeds_0_padded": visual.copy(),
        "qwen__deepstack_visual_embeds_1_padded": visual.copy(),
        "qwen__deepstack_visual_embeds_2_padded": visual.copy(),
        "dit__noisy_action": noisy_action,
        "dit__action_mask": action_mask,
        "dit__state": state,
        "dit__cos": cos,
        "dit__sin": sin,
        "dit__attention_mask": dit_attention,
        "official_seed": np.asarray([seed], dtype=np.int64),
        "official_raw_token_count": np.asarray([length], dtype=np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--robot-type", default="robocasa_mg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--action-length", type=int, default=10)
    args = parser.parse_args()
    if args.robot_type != "robocasa_mg":
        raise ValueError("this graph builder requires robot_type=robocasa_mg")

    frame_manifest = json.loads((args.frames / "manifest.json").read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(str(args.processor), trust_remote_code=True, use_fast=False)
    robot_types = processor.list_robot_types()
    if robot_types != [args.robot_type]:
        raise ValueError(f"processor robot types mismatch: {robot_types}")
    stats = processor.action_config[args.robot_type]
    mean, std = numpy(stats["mean"]), numpy(stats["std"])
    mask = numpy(processor.get_action_mask(args.robot_type))
    if mean.shape != std.shape or mean.ndim != 3 or mean.shape[2] != ACTION_DIM:
        raise ValueError(f"unexpected official action stats: {mean.shape}/{std.shape}")
    if mean.shape[1] != args.action_length or mask.shape[1] != args.action_length:
        raise ValueError(f"requested graph horizon {args.action_length} differs from official processor stats/mask: {mean.shape}/{mask.shape}")
    processor_sha = digest_tree(args.processor)
    geometry = XR1Geometry(processor_config=args.processor)
    pad_id = int(processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)

    args.output.mkdir(parents=True, exist_ok=True)
    request_dir = args.output / "processor_requests"
    bundle_dir = args.output / "internal_bundles"
    request_dir.mkdir(exist_ok=True)
    bundle_dir.mkdir(exist_ok=True)
    records = []
    for index, row in enumerate(frame_manifest["samples"]):
        sample = str(row["sample"])
        source = args.frames / sample
        images = [Image.open(source / f"camera_{i}.png").convert("RGB") for i in range(3)]
        if any(image.size != (args.image_size, args.image_size) for image in images):
            raise ValueError(f"{sample}: frame size differs from {args.image_size} square")
        raw_state = np.load(source / "state.npy", allow_pickle=False).astype(np.float32).reshape(-1)
        state = np.zeros(ACTION_DIM, dtype=np.float32)
        state[: raw_state.size] = raw_state
        processed = processor.apply_chat_template(
            messages(images, str(row["instruction"])),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            state=state.reshape(1, 1, ACTION_DIM),
            robot_type=args.robot_type,
        )
        request = {name: numpy(processed[name]) for name in (
            "input_ids", "attention_mask", "image_grid_thw", "pixel_values", "state", "action_mask"
        )}
        request["seed"] = np.asarray([args.seed], dtype=np.int64)
        request_path = request_dir / f"{index:03d}_{sample}.npz"
        np.savez(request_path, **request)
        bundle = build_internal(request, geometry, pad_id, args.action_length)
        bundle_path = bundle_dir / f"{index:03d}_{sample}.npz"
        np.savez(bundle_path, **bundle)
        records.append({
            "sample": sample,
            "task": row["task"],
            "seed": int(row["seed"]),
            "instruction": row["instruction"],
            "request": str(request_path),
            "request_sha256": sha256(request_path),
            "bundle": str(bundle_path),
            "bundle_sha256": sha256(bundle_path),
            "processor_inputs": {name: summary(value) for name, value in request.items()},
            "internal_inputs": {name: summary(value) for name, value in bundle.items()},
            "active_action_dimensions_per_step": int(np.count_nonzero(mask[0, 0])),
        })
        print(f"built {sample}: token_count={request['input_ids'].shape[-1]} pixels={request['pixel_values'].shape}", flush=True)

    np.savez(args.output / f"official_action_stats_{args.action_length}.npz", mean=mean, std=std, action_mask=mask)
    manifest = {
        "schema_version": 1,
        "source": "official XR-1 RoboCasa evaluator input contract",
        "frames": str(args.frames),
        "processor": str(args.processor),
        "processor_sha256": processor_sha,
        "robot_type": args.robot_type,
        "state_normalization": "none; preserve raw processor BF16 state values in FP32 container",
        "action_statistics_source": "official checkpoint preprocessor_config.json",
        "initial_noise_source": "torch.randn_like(BF16 action_mask on CUDA; manual_seed(seed))",
        "source_action_horizon": int(mean.shape[1]),
        "graph_action_horizon": args.action_length,
        "action_statistics_extended_after_time_constant_check": False,
        "pixel_value_shape": [768, 1536],
        "sequence_length": SEQUENCE_LENGTH,
        "samples": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    bundle_manifest = {
        "schema_version": 1,
        "source_manifest": str(args.output / "manifest.json"),
        "robot_type": args.robot_type,
        "action_length": args.action_length,
        "samples": [
            {
                "sample": row["sample"],
                "task": row["task"],
                "instruction": row["instruction"],
                "path": row["bundle"],
                "sha256": row["bundle_sha256"],
            }
            for row in records
        ],
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"built {len(records)} official processor requests and internal bundles in {args.output}")


if __name__ == "__main__":
    main()
