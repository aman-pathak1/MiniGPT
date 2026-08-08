import torch
import torch.nn as nn

from src.config import GPTConfig


class DecoderBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(config.embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.embedding_dim)
        )

        self.norm2 = nn.LayerNorm(config.embedding_dim)

    def forward(self, x):

        attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            need_weights=False
        )

        x = self.norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(x + feed_forward_output)

        return x