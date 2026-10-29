import math

import torch
import torch.nn as nn

from configs.vit import ViTConfig


class AttentionHead(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()

        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim
        self.W_q = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.W_k = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.W_v = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)

    def forward(self, x: torch.Tensor):
        q = self.W_q(x)
        v = self.W_v(x)
        k = self.W_k(x)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        att = torch.softmax(att, dim=-1)

        out = att @ v

        return out


class MultiheadAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()

        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim

        self.heads = nn.ModuleList(
            [AttentionHead(config) for _ in range(self.num_heads)]
        )
        self.W_o = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x) for head in self.heads]

        concatenated = torch.cat(head_outputs, dim=-1)

        out = self.W_o(concatenated)

        return out
