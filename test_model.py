import torch

from src.config import GPTConfig
from src.tokenizer import GPTTokenizer
from src.model import GPTModel

config = GPTConfig()

tokenizer = GPTTokenizer()

config.vocab_size = tokenizer.vocab_size

model = GPTModel(config)

text = "Hello GPT!"

token_ids = tokenizer.encode(text)

input_ids = torch.tensor([token_ids])

logits = model(input_ids)

print("Input Shape :", input_ids.shape)
print("Output Shape:", logits.shape)