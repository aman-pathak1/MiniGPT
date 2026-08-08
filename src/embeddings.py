import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_position_embeddings):
        super().__init__()

        # Token Embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Positional Embedding (GPT Style)
        self.position_embedding = nn.Embedding(
            max_position_embeddings,
            embedding_dim
        )

    def forward(self, input_ids):

        seq_length = input_ids.size(1)

        positions = torch.arange(
            seq_length,
            device=input_ids.device
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(positions)

        embeddings = token_embeddings + position_embeddings

        return embeddings