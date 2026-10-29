import torch
import torch.nn as nn

from configs.vit import ViTConfig

from .embedding import PatchEmbedding, PositionalEmbedding
from .encoder import Encoder


class ViT(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config

        self.encoders = nn.ModuleList([Encoder(config) for _ in range(config.depth)])
        self.patcher = PatchEmbedding(config)
        self.positional_encoding = PositionalEmbedding(config)

        if config.pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        else:
            self.register_parameter("cls_token", None)

        self.norm = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
        )
        self.classification_head = nn.Linear(
            config.embed_dim,
            config.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, N, embed_dim)
        x = self.patcher(x)
        if self.config.pool == "cls":
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)

        x = self.positional_encoding(x)

        for encoder in self.encoders:
            x = encoder(x)

        if self.config.pool == "cls":
            x = x[:, 0]  # (B, D)
        else:
            x = x.mean(dim=1)  # (B, D)

        x = self.norm(x)
        logits = self.classification_head(x)

        return logits
