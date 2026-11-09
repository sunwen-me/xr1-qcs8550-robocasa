"""Static tensor contract used by the XR-1 export pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class XR1Contract:
    camera_count: int = 3
    # This is the RoboCasa production contract. The retired 224x224 export
    # has 588 patches and must only be selected explicitly for old artifacts.
    image_height: int = 256
    image_width: int = 256
    temporal_patch_size: int = 2
    patch_size: int = 16
    spatial_merge_size: int = 2
    vision_hidden_size: int = 1024
    vision_output_size: int = 2560
    text_hidden_size: int = 2560
    text_layers: int = 36
    text_heads: int = 32
    text_kv_heads: int = 8
    text_head_dim: int = 128
    dit_layers: int = 36
    dit_hidden_size: int = 1024
    state_dim: int = 60
    action_length: int = 30
    action_dim: int = 60
    euler_steps: int = 5
    qairt_api_version: str = "2.50"
    qairt_version: str = "2.50.0.260828221209"

    @property
    def grid_t(self) -> int:
        return 1

    @property
    def grid_h(self) -> int:
        return self.image_height // self.patch_size

    @property
    def grid_w(self) -> int:
        return self.image_width // self.patch_size

    @property
    def vision_patch_count(self) -> int:
        return self.camera_count * self.grid_t * self.grid_h * self.grid_w

    @property
    def vision_token_count(self) -> int:
        return self.vision_patch_count // (self.spatial_merge_size**2)

    @property
    def pixel_value_shape(self) -> tuple[int, int]:
        return (
            self.vision_patch_count,
            self.temporal_patch_size * self.patch_size * self.patch_size * 3,
        )

    @property
    def dit_query_length(self) -> int:
        return 1 + 1 + self.action_length

    @property
    def dit_kv_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.dit_layers,
            1,
            self.text_kv_heads,
            0,
            self.text_head_dim,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "grid_thw": [self.grid_t, self.grid_h, self.grid_w],
                "pixel_value_shape": list(self.pixel_value_shape),
                "vision_patch_count": self.vision_patch_count,
                "vision_token_count": self.vision_token_count,
                "dit_query_length": self.dit_query_length,
            }
        )
        return data
