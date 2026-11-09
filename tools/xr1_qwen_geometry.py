#!/usr/bin/env python3
"""Qwen3-VL rotary geometry used by the XR-1 RoboCasa request builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

SEQUENCE_LENGTH = 288
PIXEL_VALUE_SHAPE = (768, 1536)
ACTION_LENGTH = 10
ACTION_DIM = 60
DIT_QUERY_LENGTH = ACTION_LENGTH + 2
HIDDEN_SIZE = 2560
IMAGE_TOKEN_ID = 151655


def tensor_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    result = np.asarray(value)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return np.ascontiguousarray(result)


class XR1Geometry:
    """Reproduce the Qwen3-VL rotary positions from processor metadata."""

    def __init__(self, processor_config: Path | None = None) -> None:
        self.vlm = None
        if processor_config is not None:
            # Processor metadata is sufficient to reproduce M-RoPE without loading weights.
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                str(processor_config), local_files_only=True, trust_remote_code=True
            )
            vlm_config = getattr(config, "vlm_config", config)
            self.image_token_id = int(getattr(vlm_config, "image_token_id", IMAGE_TOKEN_ID))
            self.vision_start_token_id = int(getattr(vlm_config, "vision_start_token_id", 151652))
            self.spatial_merge_size = int(getattr(vlm_config.vision_config, "spatial_merge_size", 2))
            text_config = vlm_config.text_config
            self.rotary = Qwen3VLTextRotaryEmbedding(text_config)
            self.pad_token_id = int(
                getattr(text_config, "pad_token_id", None)
                or getattr(text_config, "eos_token_id", 151645)
            )
            return
        # The meta model supplies the exact upstream Qwen3-VL RoPE indexing
        # implementation without loading the multi-gigabyte checkpoint.
        from xr1_port.model_loader import build_model

        meta = build_model(meta=True)
        self.vlm = meta.vlm.model
        self.rotary = Qwen3VLTextRotaryEmbedding(meta.vlm.config.text_config)
        self.image_token_id = int(getattr(self.vlm.config, "image_token_id", IMAGE_TOKEN_ID))
        text_config = meta.vlm.config.text_config
        self.pad_token_id = int(
            getattr(text_config, "pad_token_id", None)
            or getattr(text_config, "eos_token_id", 151645)
        )

    def _position_ids_without_model(
        self,
        input_ids: np.ndarray,
        image_grid_thw: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        ids = input_ids[0][attention_mask[0] == 1].tolist()
        starts = [index for index, token in enumerate(ids) if token == self.vision_start_token_id]
        image_positions = [index + 1 for index in starts if index + 1 < len(ids)]
        if len(image_positions) != len(image_grid_thw):
            raise ValueError(
                f"processor token layout has {len(image_positions)} images, "
                f"but image_grid_thw has {len(image_grid_thw)} entries"
            )
        pieces: list[torch.Tensor] = []
        start = 0
        for image_index, image_token_position in enumerate(image_positions):
            t, h, w = (int(value) for value in image_grid_thw[image_index])
            grid_h = h // self.spatial_merge_size
            grid_w = w // self.spatial_merge_size
            text_len = image_token_position - start
            offset = pieces[-1].max() + 1 if pieces else 0
            if text_len:
                pieces.append(torch.arange(text_len).view(1, -1).expand(3, -1) + offset)
            t_index = torch.arange(t).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            h_index = torch.arange(grid_h).view(1, -1, 1).expand(t, -1, grid_w).flatten()
            w_index = torch.arange(grid_w).view(1, 1, -1).expand(t, grid_h, -1).flatten()
            pieces.append(torch.stack((t_index, h_index, w_index)) + text_len + offset)
            start = image_token_position + t * grid_h * grid_w
        if start < len(ids):
            offset = pieces[-1].max() + 1 if pieces else 0
            pieces.append(torch.arange(len(ids) - start).view(1, -1).expand(3, -1) + offset)
        positions = torch.cat(pieces, dim=1).numpy().astype(np.int32, copy=False)
        result = np.ones((3, 1, input_ids.shape[1]), dtype=np.int32)
        result[:, 0, attention_mask[0] == 1] = positions
        return result

    def position_ids(
        self,
        input_ids: np.ndarray,
        image_grid_thw: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        if self.vlm is None:
            return self._position_ids_without_model(input_ids, image_grid_thw, attention_mask)
        with torch.no_grad():
            positions = self.vlm.get_rope_index(
                torch.from_numpy(input_ids.astype(np.int64, copy=False)),
                image_grid_thw=torch.from_numpy(image_grid_thw.astype(np.int64, copy=False)),
                attention_mask=torch.from_numpy(attention_mask.astype(np.int64, copy=False)),
            )[0]
        return positions.cpu().numpy().astype(np.int32, copy=False)

    def dit_rope(self, maximum_text_position: int, prefix_length: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if prefix_length < 0 or prefix_length > ACTION_LENGTH:
            raise ValueError(f"prefix_length must be in [0, {ACTION_LENGTH}], got {prefix_length}")
        position_ids = torch.arange(DIT_QUERY_LENGTH, dtype=torch.long).view(1, 1, -1)
        position_ids = position_ids.repeat(3, 1, 1) + int(maximum_text_position) + 1
        suffix_length = ACTION_LENGTH - prefix_length
        if suffix_length:
            position_ids[:, :, -suffix_length:] += 10
        dummy_action = torch.zeros((1, ACTION_LENGTH, ACTION_DIM), dtype=torch.float32)
        with torch.no_grad():
            cos, sin = self.rotary(dummy_action, position_ids)
        return cos.cpu().numpy().astype(np.float32), sin.cpu().numpy().astype(np.float32)
