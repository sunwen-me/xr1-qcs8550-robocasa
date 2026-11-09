"""Load the released XR-1 training checkpoint in the upstream code path."""

from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

XR1_SOURCE = Path(os.environ.get("XR1_UPSTREAM_SOURCE", "")).expanduser()
if XR1_SOURCE.is_dir() and str(XR1_SOURCE) not in sys.path:
    sys.path.insert(0, str(XR1_SOURCE))


def checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("module", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint state must be a mapping, got {type(checkpoint).__name__}")

    state = dict(checkpoint)
    if state and all(str(key).startswith("model.") for key in state):
        state = {str(key)[len("model.") :]: value for key, value in state.items()}
    return state


def load_checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir() or checkpoint_path.suffix == ".safetensors":
        return _load_safetensors_state(checkpoint_path)
    raw = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    return checkpoint_state(raw)


def _safetensor_shards(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
        return [path / name for name in names]
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found in {path}")
    return shards


def _load_safetensors_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("safetensors is required to load this checkpoint") from error

    state: dict[str, torch.Tensor] = {}
    for shard in _safetensor_shards(path):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in state:
                    raise ValueError(f"duplicate safetensors key: {name}")
                state[name] = handle.get_tensor(name)
    return checkpoint_state(state)


INFERENCE_ONLY_MISSING_KEYS = frozenset(
    {
        "action_projector_choice.0.layers.0.weight",
        "action_projector_choice.0.layers.2.weight",
        "action_projector_choice.0.layers.4.weight",
        "action_projector_choice.0.layers.6.weight",
        "action_projector_choice.1.layers.0.weight",
        "score_projector_choice.0.layers.0.weight",
        "score_projector_choice.0.layers.2.weight",
        "score_projector_choice.0.layers.4.weight",
        "score_projector_choice.0.layers.6.weight",
        "score_projector_choice.1.layers.0.weight",
        "state_projector_choice.layers.0.weight",
        "state_projector_choice.layers.2.weight",
        "vlm.lm_head.weight",
        "vlm.model.action_embed.weight",
        "vlm.model.score_embed.weight",
    }
)


def build_model(*, meta: bool = False):
    if not XR1_SOURCE.is_dir():
        raise RuntimeError("Set XR1_UPSTREAM_SOURCE to the matching Xiaomi Robotics-1 xr1 source directory")
    from mibot.models.VLA.XR1 import xr1

    if meta:
        previous = torch.distributions.Distribution._validate_args
        torch.distributions.Distribution.set_default_validate_args(False)
        try:
            with torch.device("meta"):
                return xr1(ffn_gradient_checkpointing=False, async_train=False)
        finally:
            torch.distributions.Distribution.set_default_validate_args(previous)
    return xr1(ffn_gradient_checkpointing=False, async_train=False).eval()


class ExportRMSNorm(nn.Module):
    """Portable RMSNorm equivalent to the upstream Liger implementation."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.weight = source.weight
        self.variance_epsilon = float(getattr(source, "variance_epsilon", 1e-6))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states * self.weight


def export_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch replacement for the upstream Liger RoPE kernel."""

    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    query_first, query_second = query.chunk(2, dim=-1)
    key_first, key_second = key.chunk(2, dim=-1)
    rotated_query = torch.cat((-query_second, query_first), dim=-1)
    rotated_key = torch.cat((-key_second, key_first), dim=-1)
    return query * cos + rotated_query * sin, key * cos + rotated_key * sin


def replace_liger_rmsnorm(module: nn.Module) -> int:
    """Replace Liger RMSNorm kernels for CPU/ONNX export and return the count."""

    replaced = 0
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "LigerRMSNorm":
            setattr(module, name, ExportRMSNorm(child))
            replaced += 1
        else:
            replaced += replace_liger_rmsnorm(child)
    return replaced


def use_eager_text_attention(model: nn.Module) -> None:
    """Select the portable eager attention path for reference/export runs."""

    from mibot.models.VLM import qwen3vl

    qwen3vl.apply_rotary_pos_emb = export_rotary_pos_emb

    for config in (
        getattr(getattr(model, "vlm", None), "config", None),
        getattr(getattr(getattr(model, "vlm", None), "model", None), "config", None),
        getattr(getattr(getattr(getattr(model, "vlm", None), "model", None), "language_model", None), "config", None),
    ):
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

    embedder = getattr(model, "t_embedder", None)
    if embedder is not None:
        def portable_timestep_forward(self: nn.Module, timestep: torch.Tensor) -> torch.Tensor:
            half = self.frequency_embedding_size // 2
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, dtype=torch.float32, device=timestep.device)
                / half
            )
            args = timestep[:, None].float() * frequencies[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            embedding = embedding.to(dtype=self.mlp[0].weight.dtype)
            return self.mlp(embedding)[:, None]

        embedder.forward = types.MethodType(portable_timestep_forward, embedder)


def materialize_meta_rope_buffers(model: nn.Module) -> None:
    """Recreate non-persistent RoPE buffers omitted from checkpoints."""

    for module in model.modules():
        buffer = getattr(module, "inv_freq", None)
        if buffer is None or not buffer.is_meta:
            continue
        class_name = module.__class__.__name__
        if class_name == "Qwen3VLVisionRotaryEmbedding":
            dimension = int(buffer.numel() * 2)
            values = torch.arange(0, dimension, 2, dtype=torch.float32)
            inv_freq = 1.0 / (10000.0 ** (values / dimension))
            module.inv_freq = inv_freq.to(dtype=buffer.dtype)
        elif class_name == "Qwen3VLTextRotaryEmbedding":
            inv_freq, attention_scaling = module.rope_init_fn(
                module.config,
                device=torch.device("cpu"),
            )
            module.inv_freq = inv_freq.to(dtype=buffer.dtype)
            module.attention_scaling = attention_scaling
        else:
            raise RuntimeError(f"cannot materialize unknown meta RoPE buffer: {module}")
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = module.inv_freq


def load_model(
    path: str | Path,
    *,
    assign: bool = True,
    export_compatible: bool = False,
    allow_inference_missing: bool = False,
):
    model = build_model(meta=assign)
    state = load_checkpoint_state(path)
    incompatible = model.load_state_dict(state, strict=False, assign=assign)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = INFERENCE_ONLY_MISSING_KEYS if allow_inference_missing else frozenset()
    unexpected_missing = missing - allowed_missing
    if unexpected or unexpected_missing:
        raise RuntimeError(
            "XR-1 checkpoint load failed: "
            f"missing={sorted(unexpected_missing)}, unexpected={sorted(unexpected)}"
        )
    materialize_meta_rope_buffers(model)
    if export_compatible:
        replace_liger_rmsnorm(model)
        use_eager_text_attention(model)
    return model
