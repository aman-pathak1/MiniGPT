import torch

from src.config import GPTConfig
from src.tokenizer import GPTTokenizer
from src.embeddings import TokenEmbedding

config = GPTConfig()

tokenizer = GPTTokenizer()

config.vocab_size = tokenizer.vocab_size

embedding = TokenEmbedding(
    vocab_size=config.vocab_size,
    embedding_dim=config.embedding_dim
)

text = "Hello GPT!"

token_ids = tokenizer.encode(text)

token_tensor = torch.tensor(token_ids)

output = embedding(token_tensor)

print("Input Text:")
print(text)

print("\nToken IDs:")
print(token_ids)

print("\nEmbedding Shape:")
print(output.shape)

print("\nEmbedding:")
print(output)