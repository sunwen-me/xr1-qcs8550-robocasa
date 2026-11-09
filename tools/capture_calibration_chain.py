#!/usr/bin/env python3
"""Capture real-sample XR-1 Vision, Qwen, DiT, and action references."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xr1_port.contract import XR1Contract
from xr1_port.graph_models import XR1DiTStep, XR1QwenPart, XR1VisionEncoder
from xr1_port.model_loader import load_model




def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def tensor(value: np.ndarray, *, long: bool = False) -> torch.Tensor:
    result = torch.from_numpy(np.ascontiguousarray(value))
    return result.long() if long else result.float()


def to_numpy(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(value.detach().float().cpu().numpy())


def capture_sample(
    model: torch.nn.Module,
    vision: XR1VisionEncoder,
    dit: XR1DiTStep,
    contract: XR1Contract,
    bundle_path: Path,
    output_path: Path,
) -> dict[str, object]:
    with np.load(bundle_path, allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    pixel_values = tensor(bundle["vision__pixel_values"])
    input_ids = tensor(bundle["qwen__input_ids"], long=True)
    attention_mask = tensor(bundle["qwen__attention_mask"])
    position_ids = tensor(bundle["qwen__position_ids"], long=True)
    visual_pos_mask = tensor(bundle["qwen__visual_pos_mask"])
    mask = visual_pos_mask.bool().squeeze(-1)
    padded = []
    for name in (
        "vision_embeds_padded",
        "deepstack_visual_embeds_0_padded",
        "deepstack_visual_embeds_1_padded",
        "deepstack_visual_embeds_2_padded",
    ):
        value = bundle.get(f"qwen__{name}")
        if value is None:
            value = np.zeros(
                (1, input_ids.shape[1], contract.vision_output_size),
                dtype=np.float32,
            )
        padded.append(tensor(value))

    outputs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        vision_outputs = vision(pixel_values)
        for name, value in zip(
            (
                "image_features",
                "deepstack_visual_embeds_0",
                "deepstack_visual_embeds_1",
                "deepstack_visual_embeds_2",
            ),
            vision_outputs,
        ):
            outputs[f"vision__{name}"] = to_numpy(value)
        for target, source in zip(padded, vision_outputs):
            target[mask] = source

        hidden: torch.Tensor | None = None
        all_k: list[torch.Tensor] = []
        all_v: list[torch.Tensor] = []
        for start in range(contract.text_layers):
            name = f"qwen_{start:02d}_{start + 1:02d}"
            part = XR1QwenPart(
                model.vlm.model,
                start,
                start + 1,
                first=start == 0,
                final=start + 1 == contract.text_layers,
            ).float().eval()
            if start == 0:
                hidden, key, value = part(
                    input_ids,
                    attention_mask,
                    position_ids,
                    padded[0],
                    visual_pos_mask,
                    padded[1],
                    padded[2],
                    padded[3],
                )
            else:
                assert hidden is not None
                hidden, key, value = part(
                    hidden,
                    attention_mask,
                    position_ids,
                    padded[0],
                    visual_pos_mask,
                    padded[1],
                    padded[2],
                    padded[3],
                )
            outputs[f"{name}__hidden_states"] = to_numpy(hidden)
            outputs[f"{name}__k_cache"] = to_numpy(key)
            outputs[f"{name}__v_cache"] = to_numpy(value)
            all_k.append(key)
            all_v.append(value)
            print(f"{bundle_path.stem}: {name}", flush=True)

        k_cache = torch.cat(all_k, dim=0)
        v_cache = torch.cat(all_v, dim=0)
        outputs["k_cache"] = to_numpy(k_cache)
        outputs["v_cache"] = to_numpy(v_cache)

        action = tensor(bundle["dit__noisy_action"])
        action_mask = tensor(bundle["dit__action_mask"])
        state = tensor(bundle["dit__state"])
        cos = tensor(bundle["dit__cos"])
        sin = tensor(bundle["dit__sin"])
        dit_attention = tensor(bundle["dit__attention_mask"])
        for step in range(5):
            timestep = torch.full((1, 1, 1), step / 5.0, dtype=torch.float32)
            velocity = dit(
                action,
                timestep,
                action_mask,
                state,
                cos,
                sin,
                dit_attention,
                k_cache,
                v_cache,
            )
            outputs[f"velocity_step_{step}"] = to_numpy(velocity)
            action = action + velocity / 5.0
        outputs["raw_action"] = to_numpy(action)

    finite = bool(all(np.isfinite(value).all() for value in outputs.values()))
    if not finite:
        raise ValueError(f"non-finite chain reference for {bundle_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **outputs)
    return {
        "bundle": str(bundle_path),
        "bundle_sha256": sha256(bundle_path),
        "path": str(output_path),
        "sha256": sha256(output_path),
        "bytes": output_path.stat().st_size,
        "finite": finite,
        "arrays": len(outputs),
        "action": {
            "minimum": float(outputs["raw_action"].min()),
            "maximum": float(outputs["raw_action"].max()),
            "l2": float(np.linalg.norm(outputs["raw_action"].astype(np.float64))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", action="append", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--allow-inference-missing", action="store_true")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    bundle_manifest = json.loads(
        (args.bundles / "manifest.json").read_text(encoding="utf-8")
    )
    selected = set(args.sample_index or range(len(bundle_manifest["samples"])))
    model = load_model(
        args.checkpoint,
        assign=True,
        export_compatible=True,
        allow_inference_missing=args.allow_inference_missing,
    ).float().eval()
    contract = XR1Contract(image_height=args.image_height, image_width=args.image_width)
    vision = XR1VisionEncoder(
        model.vlm.model.visual,
        [[1, contract.grid_h, contract.grid_w]] * contract.camera_count,
    ).float().eval()
    dit = XR1DiTStep(model, flat_norm=True).float().eval()
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, row in enumerate(bundle_manifest["samples"]):
        if index not in selected:
            continue
        bundle_path = Path(str(row.get("path", row.get("bundle"))))
        if not bundle_path.is_file():
            raise FileNotFoundError(f"missing internal bundle for {row['sample']}: {bundle_path}")
        with np.load(bundle_path, allow_pickle=False) as archive:
            action_length = int(archive["dit__noisy_action"].shape[1])
        if action_length <= 0:
            raise ValueError(f"invalid action horizon in bundle {bundle_path}: {action_length}")
        contract = XR1Contract(
            image_height=args.image_height,
            image_width=args.image_width,
            action_length=action_length,
        )
        output_path = args.output / f"{index:02d}_{row['sample']}.npz"
        metadata_path = output_path.with_suffix(".json")
        if output_path.exists() and metadata_path.exists() and not args.overwrite:
            record = json.loads(metadata_path.read_text(encoding="utf-8"))
            if sha256(output_path) != record["sha256"]:
                raise ValueError(f"existing reference hash mismatch: {output_path}")
            print(f"reuse {output_path}", flush=True)
        else:
            record = capture_sample(
                model, vision, dit, contract, bundle_path, output_path
            )
            record.update(
                {
                    "sample": row["sample"],
                    "task": row.get("task", ""),
                    "instruction": row.get("instruction", ""),
                }
            )
            save_json(metadata_path, record)
            print(f"saved {output_path}", flush=True)
        records.append(record)
        gc.collect()

    save_json(
        args.output / "manifest.json",
        {
            "schema_version": 1,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256_path(Path(args.checkpoint)),
            "calculation": "FP32 B06 Vision + piecewise Qwen + five-step DiT",
            "action_length": contract.action_length,
            "samples": records,
        },
    )
    print(f"captured {len(records)} real chain references")


if __name__ == "__main__":
    main()
