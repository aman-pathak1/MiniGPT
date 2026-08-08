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

print("Input Shape :", dummy_input.shape)
print("Output Shape:", output.shape)