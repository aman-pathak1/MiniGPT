import torch

from src.config import GPTConfig
from src.decoder import DecoderBlock


config = GPTConfig()


decoder = DecoderBlock(config)


dummy_input = torch.randn(
    2,
    config.context_length,
    config.embedding_dim
)


output = decoder(dummy_input)


print("=" * 50)
print("MiniGPT Decoder Test")
print("=" * 50)


print("Input Shape :", dummy_input.shape)
print("Output Shape:", output.shape)


# ============================================================
# Shape Test
# ============================================================

assert output.shape == dummy_input.shape, (
    "Decoder output shape must match input shape"
)


# ============================================================
# Batch Size Test
# ============================================================

assert output.size(0) == 2, (
    "Batch size changed"
)


# ============================================================
# Context Length Test
# ============================================================

assert output.size(1) == config.context_length, (
    "Context length changed"
)


# ============================================================
# Embedding Dimension Test
# ============================================================

assert output.size(2) == config.embedding_dim, (
    "Embedding dimension changed"
)


# ============================================================
# NaN / Inf Test
# ============================================================

assert torch.isfinite(output).all(), (
    "Decoder output contains NaN or Inf"
)


print()
print("Expected Shape :", dummy_input.shape)
print("Actual Shape   :", output.shape)


print()
print("=" * 50)
print("Decoder Test Passed Successfully!")
print("=" * 50)