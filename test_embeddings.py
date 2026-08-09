import torch

from src.config import GPTConfig
from src.tokenizer import GPTTokenizer
from src.embeddings import Embeddings


config = GPTConfig()

tokenizer = GPTTokenizer()

config.vocab_size = tokenizer.vocab_size


embedding = Embeddings(config)


text = "Hello GPT!"


token_ids = tokenizer.encode(text)


token_tensor = torch.tensor(
    token_ids,
    dtype=torch.long
).unsqueeze(0)


output = embedding(token_tensor)


print("=" * 50)
print("MiniGPT Embeddings Test")
print("=" * 50)


print("\nInput Text:")
print(text)


print("\nToken IDs:")
print(token_ids)


print("\nInput Shape:")
print(token_tensor.shape)


print("\nEmbedding Shape:")
print(output.shape)


# ============================================================
# Shape Test
# ============================================================

expected_shape = (
    1,
    len(token_ids),
    config.embedding_dim
)

assert output.shape == expected_shape, (
    f"Expected {expected_shape}, "
    f"got {output.shape}"
)


# ============================================================
# Embedding Dimension Test
# ============================================================

assert output.size(-1) == config.embedding_dim, (
    "Embedding dimension is incorrect"
)


# ============================================================
# NaN / Inf Test
# ============================================================

assert torch.isfinite(output).all(), (
    "Embedding contains NaN or Inf"
)


print("\nExpected Shape:")
print(expected_shape)


print("\nActual Shape:")
print(output.shape)


print()
print("=" * 50)
print("Embeddings Test Passed Successfully!")
print("=" * 50)