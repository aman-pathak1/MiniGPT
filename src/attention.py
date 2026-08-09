import torch
import torch.nn as nn
import math

from src.config import GPTConfig


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.embedding_dim % config.num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        # Query, Key and Value projection
        self.qkv = nn.Linear(
            config.embedding_dim,
            3 * config.embedding_dim,
            bias=config.use_bias
        )

        # Final output projection
        self.out_proj = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.use_bias
        )

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        mask = torch.triu(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool
            ),
            diagonal=1
        )

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False
        )

    def forward(self, x):

        # x:
        # [batch_size, sequence_length, embedding_dim]

        batch_size, seq_length, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)

        # Split into Query, Key and Value
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape:
        # [B, T, C]
        # →
        # [B, num_heads, T, head_dim]

        q = q.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Attention scores
        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Prevent looking at future tokens
        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:seq_length, :seq_length],
            float("-inf")
        )

        # Convert scores to probabilities
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.dropout(attention_weights)

        # Weighted sum of Values
        attention_output = attention_weights @ v

        # [B, heads, T, head_dim]
        # →
        # [B, T, embedding_dim]

        attention_output = attention_output.transpose(1, 2).contiguous()

        attention_output = attention_output.view(
            batch_size,
            seq_length,
            self.embedding_dim
        )

        # Final projection
        attention_output = self.out_proj(attention_output)

        return attention_output