#!/usr/bin/env python3
"""Export the fixed-shape XR-1 vision, Qwen, and DiT graphs."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
XR1_SOURCE = Path(os.environ.get("XR1_UPSTREAM_SOURCE", "")).expanduser()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if XR1_SOURCE.is_dir() and str(XR1_SOURCE) not in sys.path:
    sys.path.insert(0, str(XR1_SOURCE))

from xr1_port.contract import XR1Contract
from xr1_port.graph_models import XR1DiTStep, XR1QwenPart, XR1VisionEncoder
from xr1_port.model_loader import load_model


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _export(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    output_dir: Path,
) -> tuple[list[list[int]], list[list[int]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"
    with torch.no_grad():
        outputs = module(*args)
        output_values = outputs if isinstance(outputs, tuple) else (outputs,)
        output_shapes = [list(value.shape) for value in output_values]
        torch.onnx.export(
            module,
            args,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            dynamo=False,
            external_data=True,
            optimize=False,
            verbose=False,
        )
    input_shapes = [list(value.shape) for value in args]
    np.savez(
        output_dir / "reference.npz",
        **{
            f"input__{name}": _to_numpy(value)
            for name, value in zip(input_names, args)
        },
        **{
            f"output__{name}": _to_numpy(value)
            for name, value in zip(output_names, output_values)
        },
    )
    return input_shapes, output_shapes


def _messages(images: list[Image.Image], instruction: str) -> list[dict[str, object]]:
    content: list[dict[str, object]] = [
        {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
        {"type": "image", "image": images[0]},
        {"type": "text", "text": "\n# Left-Wrist View\n"},
        {"type": "image", "image": images[1]},
        {"type": "text", "text": "\n# Right-Wrist View\n"},
        {"type": "image", "image": images[2]},
        {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
    ]
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
    ]


def _build_inputs(
    model: torch.nn.Module,
    processor: AutoProcessor,
    contract: XR1Contract,
    sequence_length: int,
    instruction: str,
    robot_type: str,
) -> dict[str, torch.Tensor]:
    images = [Image.new("RGB", (contract.image_width, contract.image_height), color=(0, 0, 0)) for _ in range(3)]
    processor_kwargs: dict[str, object] = {}
    if hasattr(processor, "get_action_mask"):
        # The released XR-1 MiBotProcessor requires these fields even though
        # graph export only consumes its token and image tensors.
        processor_kwargs.update(
            {
                "state": torch.zeros((1, 1, contract.state_dim), dtype=torch.float32),
                "robot_type": robot_type,
            }
        )
    raw = processor.apply_chat_template(
        [_messages(images, instruction)],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        images_kwargs={"do_resize": False},
        **processor_kwargs,
    )
    raw_length = int(raw["input_ids"].shape[1])
    if raw_length > sequence_length:
        raise ValueError(f"prompt has {raw_length} tokens, larger than fixed sequence {sequence_length}")
    input_ids = F.pad(raw["input_ids"], (0, sequence_length - raw_length), value=int(processor.tokenizer.pad_token_id))
    attention_mask = F.pad(raw["attention_mask"], (0, sequence_length - raw_length), value=0)
    position_ids = model.vlm.model.get_rope_index(
        input_ids[:, :raw_length],
        image_grid_thw=raw["image_grid_thw"],
        attention_mask=raw["attention_mask"],
    )[0]
    position_ids = F.pad(position_ids, (0, sequence_length - raw_length), value=1)

    image_token_id = int(model.vlm.model.config.image_token_id)
    visual_mask = input_ids.eq(image_token_id).unsqueeze(-1).to(torch.float32)
    with torch.no_grad():
        vision = copy.deepcopy(XR1VisionEncoder(
            model.vlm.model.visual,
            [[1, contract.grid_h, contract.grid_w]] * contract.camera_count,
        )).float().eval()
        image_features, deep0, deep1, deep2 = vision(raw["pixel_values"].float())
    padded = torch.zeros((1, sequence_length, contract.text_hidden_size), dtype=torch.float32)
    deepstack = [torch.zeros_like(padded) for _ in range(3)]
    raw_mask = input_ids[:, :raw_length].eq(image_token_id)
    padded[:, :raw_length][raw_mask] = image_features.float()
    for target, source in zip(deepstack, (deep0, deep1, deep2)):
        target[:, :raw_length][raw_mask] = source.float()
    return {
        "input_ids": input_ids.to(torch.int32),
        "attention_mask": attention_mask.to(torch.int32),
        "position_ids": position_ids.to(torch.int32),
        "vision_embeds_padded": padded,
        "visual_pos_mask": visual_mask,
        "deepstack_visual_embeds_0_padded": deepstack[0],
        "deepstack_visual_embeds_1_padded": deepstack[1],
        "deepstack_visual_embeds_2_padded": deepstack[2],
        "pixel_values": raw["pixel_values"].float(),
        "image_grid_thw": raw["image_grid_thw"].to(torch.int32),
        "raw_length": torch.tensor(raw_length, dtype=torch.int32),
    }


def _export_vision(
    model: torch.nn.Module,
    contract: XR1Contract,
    output_root: Path,
    stable_gelu_threshold: float | None = None,
    stable_gelu_series_terms: int = 5,
    stable_gelu_block_count: int | None = None,
    stable_gelu_exp_floor: float = -16.0,
) -> dict[str, object]:
    wrapper = copy.deepcopy(XR1VisionEncoder(
        model.vlm.model.visual,
        [[1, contract.grid_h, contract.grid_w]] * contract.camera_count,
        stable_gelu_threshold=stable_gelu_threshold,
        stable_gelu_series_terms=stable_gelu_series_terms,
        stable_gelu_block_count=stable_gelu_block_count,
        stable_gelu_exp_floor=stable_gelu_exp_floor,
    )).float().eval()
    pixel_values = torch.zeros(contract.pixel_value_shape, dtype=torch.float32)
    input_shapes, output_shapes = _export(
        wrapper,
        (pixel_values,),
        ["pixel_values"],
        ["image_features", "deepstack_visual_embeds_0", "deepstack_visual_embeds_1", "deepstack_visual_embeds_2"],
        output_root / "vision",
    )
    return {
        "name": "vision",
        "inputs": {"pixel_values": input_shapes[0]},
        "outputs": dict(
            zip(
                [
                    "image_features",
                    "deepstack_visual_embeds_0",
                    "deepstack_visual_embeds_1",
                    "deepstack_visual_embeds_2",
                ],
                output_shapes,
            )
        ),
    }


def _export_qwen(
    model: torch.nn.Module,
    contract: XR1Contract,
    qwen_inputs: dict[str, torch.Tensor],
    output_root: Path,
    layers_per_part: int,
    selected_parts: set[str] | None = None,
    down_proj_as_conv: bool = False,
) -> list[dict[str, object]]:
    reports = []
    total_layers = contract.text_layers
    for start in range(0, total_layers, layers_per_part):
        end = min(total_layers, start + layers_per_part)
        name = f"qwen_{start:02d}_{end:02d}"
        if selected_parts is not None and name not in selected_parts:
            continue
        first = start == 0
        final = end == total_layers
        wrapper = copy.deepcopy(XR1QwenPart(
            model.vlm.model,
            start,
            end,
            first=first,
            final=final,
            down_proj_as_conv=down_proj_as_conv,
        )).float().eval()
        if first:
            names = [
                "input_ids",
                "attention_mask",
                "position_ids",
                "vision_embeds_padded",
                "visual_pos_mask",
                "deepstack_visual_embeds_0_padded",
                "deepstack_visual_embeds_1_padded",
                "deepstack_visual_embeds_2_padded",
            ]
            args = tuple(qwen_inputs[name] for name in names)
        else:
            names = ["hidden_states", "attention_mask", "position_ids"]
            sequence_length = int(qwen_inputs["attention_mask"].shape[1])
            args = (
                torch.zeros((1, sequence_length, contract.text_hidden_size)),
                qwen_inputs["attention_mask"],
                qwen_inputs["position_ids"],
            )
            if start < 3:
                # Layers 1 and 2 still consume their corresponding DeepStack
                # features even though they are no longer the first graph.
                names.extend(
                    [
                        "vision_embeds_padded",
                        "visual_pos_mask",
                        "deepstack_visual_embeds_0_padded",
                        "deepstack_visual_embeds_1_padded",
                        "deepstack_visual_embeds_2_padded",
                    ]
                )
                args += (
                    qwen_inputs["vision_embeds_padded"],
                    qwen_inputs["visual_pos_mask"],
                    qwen_inputs["deepstack_visual_embeds_0_padded"],
                    qwen_inputs["deepstack_visual_embeds_1_padded"],
                    qwen_inputs["deepstack_visual_embeds_2_padded"],
                )
        input_shapes, output_shapes = _export(
            wrapper,
            args,
            names,
            ["hidden_states", "k_cache", "v_cache"],
            output_root / name,
        )
        if first:
            first_input_count = 6 + int(end > 1) + int(end > 2)
            report_names = names[:first_input_count]
            report_shapes = input_shapes[:first_input_count]
        elif start < 3:
            deepstack_index = 5 + start
            report_names = ["hidden_states", "attention_mask", "position_ids", names[deepstack_index]]
            report_shapes = [input_shapes[index] for index in (0, 1, 2, deepstack_index)]
        else:
            report_names = names[:3]
            report_shapes = input_shapes[:3]
        reports.append(
            {
                "name": name,
                "layer_start": start,
                "layer_end": end,
                "inputs": dict(zip(report_names, report_shapes)),
                "outputs": dict(zip(["hidden_states", "k_cache", "v_cache"], output_shapes)),
            }
        )
        del wrapper
        gc.collect()
    return reports


def _dit_inputs(model: torch.nn.Module, contract: XR1Contract, sequence_length: int) -> tuple[torch.Tensor, ...]:
    query_length = contract.dit_query_length
    state = torch.zeros((1, 1, contract.state_dim), dtype=torch.float32)
    noisy_action = torch.zeros((1, contract.action_length, contract.action_dim), dtype=torch.float32)
    action_mask = torch.ones_like(noisy_action)
    timestep = torch.zeros((1, 1, 1), dtype=torch.float32)
    position_ids = torch.arange(query_length, dtype=torch.long).view(1, 1, -1).repeat(3, 1, 1)
    position_ids = position_ids + 1
    position_ids[:, :, -contract.action_length :] += 10
    cos, sin = model.rotary_emb(noisy_action, position_ids)
    attention_mask = torch.zeros((1, 1, query_length, sequence_length + query_length), dtype=torch.float32)
    attention_mask[:, :, :, :sequence_length] = 0.0
    causal = torch.tril(torch.ones((query_length, query_length), dtype=torch.bool))
    attention_mask[:, :, :, sequence_length:] = torch.where(
        causal.view(1, 1, query_length, query_length),
        torch.zeros((), dtype=torch.float32),
        torch.full((), -10000.0, dtype=torch.float32),
    )
    k_cache = torch.zeros(
        (contract.dit_layers, 1, contract.text_kv_heads, sequence_length, contract.text_head_dim),
        dtype=torch.float32,
    )
    return noisy_action, timestep, action_mask, state, cos.float(), sin.float(), attention_mask, k_cache, k_cache.clone()


def _export_dit(model: torch.nn.Module, contract: XR1Contract, output_root: Path, sequence_length: int) -> dict[str, object]:
    wrapper = copy.deepcopy(XR1DiTStep(model, flat_norm=True)).float().eval()
    args = _dit_inputs(model, contract, sequence_length)
    names = [
        "noisy_action",
        "timestep",
        "action_mask",
        "state",
        "cos",
        "sin",
        "attention_mask",
        "k_cache",
        "v_cache",
    ]
    input_shapes, output_shapes = _export(wrapper, args, names, ["velocity"], output_root / "dit_step")
    return {"name": "dit_step", "inputs": dict(zip(names, input_shapes)), "outputs": {"velocity": output_shapes[0]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "graphs")
    parser.add_argument("--sequence-length", type=int, default=288)
    parser.add_argument(
        "--action-length",
        type=int,
        default=30,
        help="DiT action horizon; the official RoboCasa checkpoint uses 10.",
    )
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--layers-per-part", type=int, default=1)
    parser.add_argument(
        "--graphs",
        action="append",
        choices=("vision", "qwen", "dit"),
        help="export only the selected graph families; repeat for multiple families",
    )
    parser.add_argument(
        "--qwen-part",
        action="append",
        dest="qwen_parts",
        help="when exporting Qwen, select exact partitions such as qwen_01_02; repeat to select more",
    )
    parser.add_argument(
        "--qwen-down-proj-conv",
        action="store_true",
        help="export Qwen down_proj as an equivalent 1x1 Conv2d; Qwen uses the native Linear by default",
    )
    parser.add_argument("--instruction", default="put the object in the bin")
    parser.add_argument(
        "--robot-type",
        choices=("robocasa_mg", "so101_dual"),
        default="robocasa_mg",
        help="robot type passed to processors that build robot-specific action masks",
    )
    parser.add_argument("--allow-inference-missing", action="store_true")
    parser.add_argument(
        "--stable-vision-gelu-threshold",
        type=float,
        help="replace Vision tanh-GELU below this negative logistic threshold",
    )
    parser.add_argument(
        "--stable-vision-gelu-series-terms",
        type=int,
        default=5,
        help="number of alternating logistic-series terms in the stable GELU tail",
    )
    parser.add_argument(
        "--stable-vision-gelu-block-count",
        type=int,
        help="apply stable GELU only to this many leading Vision blocks",
    )
    parser.add_argument(
        "--stable-vision-gelu-exp-floor",
        type=float,
        default=-16.0,
        help="lower bound for the stable Vision GELU Exp input",
    )
    args = parser.parse_args()
    if args.layers_per_part <= 0 or 36 % args.layers_per_part != 0:
        raise ValueError("--layers-per-part must be a positive divisor of 36")

    if args.image_height % 16 or args.image_width % 16:
        raise ValueError("--image-height and --image-width must be divisible by the 16-pixel patch size")
    contract = XR1Contract(
        image_height=args.image_height,
        image_width=args.image_width,
        action_length=args.action_length,
    )
    model = load_model(
        args.checkpoint,
        assign=True,
        export_compatible=True,
        allow_inference_missing=args.allow_inference_missing,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.processor,
        trust_remote_code=True,
        use_fast=False,
    )
    qwen_inputs = _build_inputs(
        model,
        processor,
        contract,
        args.sequence_length,
        args.instruction,
        args.robot_type,
    )
    selected_graphs = set(args.graphs or ("vision", "qwen", "dit"))

    reports: dict[str, object] = {
        "contract": contract.to_dict(),
        "sequence_length": args.sequence_length,
        "layers_per_part": args.layers_per_part,
        "qwen_down_proj_conv": args.qwen_down_proj_conv,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": _sha256_path(Path(args.checkpoint)),
        "processor": args.processor,
        "processor_sha256": _sha256_path(Path(args.processor)),
        "robot_type": args.robot_type,
        "instruction": args.instruction,
        "graphs": [],
    }
    if args.stable_vision_gelu_threshold is not None:
        reports["vision_gelu"] = {
            "threshold": args.stable_vision_gelu_threshold,
            "series_terms": args.stable_vision_gelu_series_terms,
            "block_count": args.stable_vision_gelu_block_count,
            "exp_floor": args.stable_vision_gelu_exp_floor,
        }
    if "vision" in selected_graphs:
        reports["graphs"].append(
            _export_vision(
                model,
                contract,
                args.output_root,
                stable_gelu_threshold=args.stable_vision_gelu_threshold,
                stable_gelu_series_terms=args.stable_vision_gelu_series_terms,
                stable_gelu_block_count=args.stable_vision_gelu_block_count,
                stable_gelu_exp_floor=args.stable_vision_gelu_exp_floor,
            )
        )
    if "qwen" in selected_graphs:
        reports["graphs"].extend(
            _export_qwen(
                model,
                contract,
                qwen_inputs,
                args.output_root,
                args.layers_per_part,
                set(args.qwen_parts) if args.qwen_parts else None,
                args.qwen_down_proj_conv,
            )
        )
    if "dit" in selected_graphs:
        reports["graphs"].append(_export_dit(model, contract, args.output_root, args.sequence_length))
    _save_json(args.output_root / "export_report.json", reports)
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
