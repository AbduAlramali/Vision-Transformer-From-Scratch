from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ViTConfig:
    # Input and patch embedding
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Transformer architecture
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0

    # Architectural choices
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6
    pool: Literal["cls", "mean"] = "cls"

    # Task
    num_classes: int = 10

    @staticmethod
    def _to_pair(
        value: int | tuple[int, int],
    ) -> tuple[int, int]:
        if isinstance(value, int):
            return (value, value)
        if len(value) != 2:
            raise ValueError("Expected an integer or a pair of integers.")
        return tuple(value)

    @property
    def image_hw(self) -> tuple[int, int]:
        return self._to_pair(self.image_size)

    @property
    def patch_hw(self) -> tuple[int, int]:
        return self._to_pair(self.patch_size)

    @property
    def grid_size(self) -> tuple[int, int]:
        image_h, image_w = self.image_hw
        patch_h, patch_w = self.patch_hw
        return (image_h // patch_h, image_w // patch_w)

    @property
    def num_patches(self) -> int:
        grid_h, grid_w = self.grid_size
        return grid_h * grid_w

    @property
    def seq_length(self) -> int:
        # One extra token for the [CLS] representation.
        return self.num_patches + 1 if self.pool == "cls" else self.num_patches

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def mlp_dim(self) -> int:
        return int(self.embed_dim * self.mlp_ratio)
