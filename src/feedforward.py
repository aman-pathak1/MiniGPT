import torch.nn as nn

from src.config import GPTConfig


class FeedForward(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.network = nn.Sequential(

            # Expand hidden dimension
            nn.Linear(
                config.embedding_dim,
                config.ffn_dim,
                bias=config.use_bias
            ),

            # Non-linear activation
            nn.GELU(),

            # Project back to embedding dimension
            nn.Linear(
                config.ffn_dim,
                config.embedding_dim,
                bias=config.use_bias
            )
        )

    def forward(self, x):

        return self.network(x)