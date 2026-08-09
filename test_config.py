from src.config import GPTConfig


config = GPTConfig()


assert config.vocab_size == 50257
assert config.context_length == 512
assert config.embedding_dim == 384
assert config.num_heads == 6
assert config.num_layers == 6
assert config.ffn_dim == 1536

assert config.embedding_dim % config.num_heads == 0

assert config.batch_size > 0
assert config.learning_rate > 0
assert config.epochs > 0

print("=" * 50)
print("MiniGPT Configuration Test")
print("=" * 50)

print(f"Vocabulary Size : {config.vocab_size}")
print(f"Context Length  : {config.context_length}")
print(f"Embedding Dim   : {config.embedding_dim}")
print(f"Attention Heads : {config.num_heads}")
print(f"Decoder Layers  : {config.num_layers}")
print(f"FFN Dimension   : {config.ffn_dim}")
print(f"Batch Size      : {config.batch_size}")
print(f"Learning Rate   : {config.learning_rate}")
print(f"Epochs          : {config.epochs}")
print(f"Device          : {config.device}")

print("=" * 50)
print("Configuration Test Passed!")
print("=" * 50)