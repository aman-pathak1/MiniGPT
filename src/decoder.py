import torch
import torch.nn as nn

from src.config import GPTConfig


class DecoderBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(
            config.embedding_dim,
            eps=config.layer_norm_eps
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                config.embedding_dim,
                config.ffn_dim,
                bias=config.use_bias
            ),
            nn.GELU(),
            nn.Linear(
                config.ffn_dim,
                config.embedding_dim,
                bias=config.use_bias
            )
        )

        self.norm2 = nn.LayerNorm(
            config.embedding_dim,
            eps=config.layer_norm_eps
        )

    def forward(self, x):

        seq_len = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            need_weights=False
        )

        x = self.norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(x + feed_forward_output)

        return x