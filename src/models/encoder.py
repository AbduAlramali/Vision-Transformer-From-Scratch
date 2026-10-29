import torch
import torch.nn as nn

from configs.vit import ViTConfig

from .attention import MultiheadAttention


class Encoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.multihead = MultiheadAttention(config)
        self.layernorm_1 = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
        )
        self.layernorm_2 = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
        )

        self.feednn = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layernorm_1(x)
        out_attn = x + self.multihead(x_norm)

        out_attn_norm = self.layernorm_2(out_attn)
        out_feednn = out_attn + self.feednn(out_attn_norm)
        return out_feednn
