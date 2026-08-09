import torch
import torch.nn as nn

from src.config import GPTConfig


class Embeddings(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Token Embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim
        )

        # Learned Positional Embedding
        self.position_embedding = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.embedding_dim
        )

    def forward(self, input_ids):

        # input_ids shape:
        # [batch_size, sequence_length]

        seq_length = input_ids.size(1)

        # Create position IDs:
        # [0, 1, 2, ..., seq_length - 1]
        positions = torch.arange(
            seq_length,
            device=input_ids.device
        ).unsqueeze(0)

        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)

        # Positional embeddings
        position_embeddings = self.position_embedding(positions)

        # Add token + positional embeddings
        embeddings = token_embeddings + position_embeddings

        return embeddings