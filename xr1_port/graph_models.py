"""ONNX-friendly XR-1 graph wrappers.

The wrappers keep preprocessing, random noise, Euler integration, and robot
decoding outside the compiled graphs. Each graph has static sequence shapes so
that AI Hub can produce QNN context binaries for the QCS8550.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value[..., : value.shape[-1] // 2], value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


def _repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return value
    batch, heads, sequence, head_dim = value.shape
    value = value[:, :, None, :, :].expand(batch, heads, repeats, sequence, head_dim)
    return value.reshape(batch, heads * repeats, sequence, head_dim)


def _static_causal_mask(attention_mask: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    sequence = hidden_states.shape[1]
    indices = torch.arange(sequence, device=hidden_states.device)
    causal = indices.view(1, -1) <= indices.view(-1, 1)
    valid = attention_mask.to(torch.bool).view(attention_mask.shape[0], 1, 1, sequence)
    allowed = causal.view(1, 1, sequence, sequence) & valid
    minimum = torch.finfo(hidden_states.dtype).min
    return torch.where(
        allowed,
        torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype),
        torch.full((), minimum, device=hidden_states.device, dtype=hidden_states.dtype),
    )


class LinearAsConv1x1(nn.Conv2d):
    """Exact 1x1 Conv2d representation of a Linear layer for HTP export."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__(
            linear.in_features,
            linear.out_features,
            kernel_size=1,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
        )
        self.weight.data.copy_(linear.weight.data[:, :, None, None])
        if linear.bias is not None and self.bias is not None:
            self.bias.data.copy_(linear.bias.data)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError(f"expected [batch, sequence, channels], got {value.shape}")
        value = super().forward(value.permute(0, 2, 1).unsqueeze(-1))
        return value.squeeze(-1).permute(0, 2, 1)


class StableTanhGELU(nn.Module):
    """Algebraically preserve tanh-GELU while bounding HTP's Exp input."""

    def __init__(
        self,
        threshold: float = -3.0,
        series_terms: int = 5,
        exp_floor: float = -16.0,
    ) -> None:
        super().__init__()
        if threshold >= 0.0:
            raise ValueError("stable GELU threshold must be negative")
        if series_terms < 1:
            raise ValueError("stable GELU series must have at least one term")
        if exp_floor >= threshold:
            raise ValueError("stable GELU Exp floor must be below the tail threshold")
        self.threshold = float(threshold)
        self.series_terms = int(series_terms)
        self.exp_floor = float(exp_floor)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        cubic = value * value * value
        tanh_argument = math.sqrt(2.0 / math.pi) * (value + 0.044715 * cubic)
        logit = 2.0 * tanh_argument
        # Where evaluates both branches in the exported graph. Keep Exp's
        # negative input in the range that QNN HTP handles reliably even when
        # the selected branch is the regular sigmoid path.
        tail_logit = torch.clamp(logit, min=self.exp_floor, max=self.threshold)
        u = torch.exp(tail_logit)
        tail_polynomial: torch.Tensor | float = 1.0
        for _ in range(self.series_terms - 1):
            tail_polynomial = 1.0 - u * tail_polynomial
        tail = u * tail_polynomial
        probability = torch.where(logit < self.threshold, tail, torch.sigmoid(logit))
        return value * probability


class XR1VisionEncoder(nn.Module):
    """Encode three fixed 224x224 camera images into Qwen visual tokens."""

    def __init__(
        self,
        visual: nn.Module,
        grid_thw: Iterable[Iterable[int]],
        *,
        stable_gelu_threshold: float | None = None,
        stable_gelu_series_terms: int = 5,
        stable_gelu_block_count: int | None = None,
        stable_gelu_exp_floor: float = -16.0,
    ) -> None:
        super().__init__()
        self.visual = visual
        if stable_gelu_threshold is not None:
            block_count = (
                len(self.visual.blocks)
                if stable_gelu_block_count is None
                else stable_gelu_block_count
            )
            if not 0 <= block_count <= len(self.visual.blocks):
                raise ValueError(
                    f"stable GELU block count must be in [0, {len(self.visual.blocks)}]"
                )
            for block in self.visual.blocks[:block_count]:
                block.mlp.act_fn = StableTanhGELU(
                    stable_gelu_threshold,
                    stable_gelu_series_terms,
                    stable_gelu_exp_floor,
                )
        self.register_buffer("grid_thw", torch.tensor(list(grid_thw), dtype=torch.long), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features, deepstack = self.visual(pixel_values, grid_thw=self.grid_thw)
        return image_features, deepstack[0], deepstack[1], deepstack[2]


class XR1QwenPart(nn.Module):
    """Run a static prefill chunk and return its hidden state and per-layer KV."""

    def __init__(
        self,
        vlm_model: nn.Module,
        start: int,
        end: int,
        *,
        first: bool = False,
        final: bool = False,
        down_proj_as_conv: bool = False,
    ) -> None:
        super().__init__()
        language_model = vlm_model.language_model
        self.layers = nn.ModuleList(language_model.layers[start:end])
        self.rotary_emb = language_model.rotary_emb
        self.embed_tokens = language_model.embed_tokens if first else None
        self.final_norm = language_model.norm if final else None
        self.start = int(start)
        self.end = int(end)
        self.first = bool(first)
        self.down_proj_as_conv = bool(down_proj_as_conv)
        self.down_projs = nn.ModuleList(
            LinearAsConv1x1(layer.mlp.down_proj)
            for layer in self.layers
        ) if self.down_proj_as_conv else None

    def _attention(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = _apply_rotary(query, key, position_embeddings)

        repeated_key = _repeat_kv(key, attention.num_key_value_groups)
        repeated_value = _repeat_kv(value, attention.num_key_value_groups)
        logits = torch.matmul(query, repeated_key.transpose(2, 3)) * attention.scaling
        logits = logits + causal_mask
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, repeated_value)
        output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return attention.o_proj(output), key, value

    def _layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask: torch.Tensor,
        down_proj: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attention_output, key, value = self._attention(
            layer.self_attn,
            normalized,
            position_embeddings,
            causal_mask,
        )
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        mlp = layer.mlp
        gate = mlp.gate_proj(hidden_states)
        hidden_states = down_proj(F.silu(gate) * mlp.up_proj(hidden_states))
        return residual + hidden_states, key, value

    def forward(
        self,
        hidden_states_or_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        vision_embeds_padded: torch.Tensor | None = None,
        visual_pos_mask: torch.Tensor | None = None,
        deepstack_visual_embeds_0_padded: torch.Tensor | None = None,
        deepstack_visual_embeds_1_padded: torch.Tensor | None = None,
        deepstack_visual_embeds_2_padded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.first:
            if self.embed_tokens is None or vision_embeds_padded is None or visual_pos_mask is None:
                raise ValueError("first XR-1 Qwen part requires embedding and vision inputs")
            input_ids = hidden_states_or_input_ids.to(torch.long)
            hidden_states = self.embed_tokens(input_ids)
            mask = visual_pos_mask.to(hidden_states.device, hidden_states.dtype)
            hidden_states = hidden_states * (1.0 - mask) + vision_embeds_padded.to(
                hidden_states.device, hidden_states.dtype
            ) * mask
            deepstack = (
                deepstack_visual_embeds_0_padded,
                deepstack_visual_embeds_1_padded,
                deepstack_visual_embeds_2_padded,
            )
        else:
            hidden_states = hidden_states_or_input_ids
            deepstack = (
                deepstack_visual_embeds_0_padded,
                deepstack_visual_embeds_1_padded,
                deepstack_visual_embeds_2_padded,
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = _static_causal_mask(attention_mask, hidden_states)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for offset, layer in enumerate(self.layers):
            layer_index = self.start + offset
            down_proj = (
                self.down_projs[offset]
                if self.down_projs is not None
                else layer.mlp.down_proj
            )
            hidden_states, key, value = self._layer(
                layer,
                hidden_states,
                position_embeddings,
                causal_mask,
                down_proj,
            )
            keys.append(key)
            values.append(value)
            # Qwen injects DeepStack after layers 0, 1, and 2.  Layers 1
            # and 2 may live in a non-first graph partition, so `first`
            # only controls token embedding and must not gate this path.
            if layer_index < 3:
                visual = deepstack[layer_index]
                if visual is None:
                    raise ValueError("missing XR-1 DeepStack input")
                hidden_states = hidden_states + visual.to(hidden_states.device, hidden_states.dtype)

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        return hidden_states, torch.stack(keys, dim=0), torch.stack(values, dim=0)


class XR1DiTStep(nn.Module):
    """One XR-1 DiT velocity step with a packed 36-layer Qwen KV cache."""

    def __init__(self, action_model: nn.Module, *, flat_norm: bool = True) -> None:
        super().__init__()
        self.state_projector = action_model.state_projector
        self.action_projector = action_model.action_projector
        self.action_output_layer = action_model.action_output_layer
        self.t_embedder = action_model.t_embedder
        self.t_projector = action_model.t_projector
        self.layers = action_model.dit.layers
        self.sink = action_model.sink
        self.flat_norm = bool(flat_norm)

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        embedder = self.t_embedder
        dimension = int(embedder.frequency_embedding_size)
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / half
        )
        args = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embedding = embedding.to(embedder.mlp[0].weight.dtype)
        return embedder.mlp(embedding)[:, None]

    @staticmethod
    def _norm_flat(norm: nn.Module, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        return norm(value.reshape(-1, shape[-1])).reshape(shape)

    def _attention(
        self,
        attention: nn.Module,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_length, _ = hidden_states.shape
        qkv = attention.qkv_proj(hidden_states).view(
            batch, query_length, 3, attention.num_heads, attention.head_dim
        )
        query, key, value = qkv.unbind(2)
        if self.flat_norm:
            query = self._norm_flat(attention.q_norm, query)
            key = self._norm_flat(attention.k_norm, key)
        else:
            query = attention.q_norm(query)
            key = attention.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = _apply_rotary(query, key, (cos, sin))
        key_cache = _repeat_kv(key_cache, attention.kv_group)
        value_cache = _repeat_kv(value_cache, attention.kv_group)
        key = torch.cat([key_cache, key], dim=-2)
        value = torch.cat([value_cache, value], dim=-2)
        logits = torch.matmul(query, key.transpose(2, 3)) * (attention.head_dim**-0.5)
        probabilities = F.softmax(logits + attention_mask, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).contiguous().reshape(batch, query_length, -1)
        return attention.o_proj(output)

    def _layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            layer.adaln_table[None] + timestep
        ).chunk(6, dim=1)
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = hidden_states * (1.0 + scale_attn) + shift_attn
        hidden_states = residual + gate_attn * self._attention(
            layer.attn,
            hidden_states,
            cos,
            sin,
            attention_mask,
            key_cache,
            value_cache,
        )
        residual = hidden_states
        hidden_states = layer.post_layernorm(hidden_states)
        hidden_states = hidden_states * (1.0 + scale_mlp) + shift_mlp
        return residual + gate_mlp * layer.mlp(hidden_states)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        action_mask: torch.Tensor,
        state: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        time = self._time_embedding(timestep[:, 0, 0] * 1000.0)
        time = self.t_projector(time).view(time.shape[0], 6, -1)
        action_hidden = self.action_projector(noisy_action * action_mask)
        state_hidden = self.state_projector(state)
        sink = self.sink.weight[None].repeat(state_hidden.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_hidden, action_hidden], dim=1).contiguous()
        for layer_index, layer in enumerate(self.layers):
            hidden_states = self._layer(
                layer,
                hidden_states,
                time,
                cos,
                sin,
                attention_mask,
                k_cache[layer_index],
                v_cache[layer_index],
            )
        return self.action_output_layer(hidden_states[:, -action_hidden.shape[1] :])


class XR1DiTFlow5(nn.Module):
    """Five fixed Euler/flow-matching steps fused into one graph.

    ``update_mask`` keeps the committed asynchronous prefix unchanged while
    moving the action recursion into the exported graph.
    """

    def __init__(self, action_model: nn.Module, *, flat_norm: bool = True, steps: int = 5) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.step = XR1DiTStep(action_model, flat_norm=flat_norm)
        self.steps = int(steps)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        action_mask: torch.Tensor,
        update_mask: torch.Tensor,
        state: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        action = noisy_action
        for step in range(self.steps):
            step_timestep = timestep * 0.0 + float(step) / float(self.steps)
            velocity = self.step(
                action,
                step_timestep,
                action_mask,
                state,
                cos,
                sin,
                attention_mask,
                k_cache,
                v_cache,
            )
            action = action + velocity / float(self.steps) * update_mask
        return action
