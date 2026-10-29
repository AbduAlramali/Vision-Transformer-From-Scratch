import torch
import torch.nn as nn

from configs.vit import ViTConfig


class PatchEmbedding(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        P = config.patch_size
        I = config.image_size
        self.image_size = I
        self.patch_size = P
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim

        self.num_patches = (I // P) ** 2

        patch_dim = config.in_channels * P * P

        self.projection = nn.Linear(
            patch_dim,
            config.embed_dim,
        )

    def _extract_patches(self, x: torch.Tensor):
        """
        Convert an image into a sequence of image patches.

        Input:
            x: (B, C, H, W)

        Output:
            patches: (B, N, C, P, P)
        """
        B, C, H, W = x.shape
        P = self.patch_size

        patches = x.unfold(2, P, P).unfold(3, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(
            B,
            self.num_patches,
            C,
            P,
            P,
        )
        return patches

    def _flatten_patches(self, patches):
        """
        Flatten each patch into a vector.

        Input:
            (B, N, C, P, P)

        Output:
            (B, N, patch_dim)
        """
        B, N, C, P, _ = patches.shape

        return patches.reshape(
            B,
            N,
            C * P * P,
        )

    def forward(self, x):
        patches = self._extract_patches(x)

        patches = self._flatten_patches(patches)

        embeddings = self.projection(patches)

        return embeddings


class PositionalEmbedding(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        num_tokens = config.num_patches + (config.pool == "cls")

        # one learnable vector for each position
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_tokens, config.embed_dim)
        )

    def forward(self, x):
        return x + self.position_embeddings
