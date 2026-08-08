from src.config import GPTConfig

config = GPTConfig()

print("=" * 50)
print("MiniGPT Configuration")
print("=" * 50)

print(f"Vocabulary Size   : {config.vocab_size}")
print(f"Context Length    : {config.context_length}")
print(f"Embedding Dim     : {config.embedding_dim}")
print(f"Attention Heads   : {config.num_heads}")
print(f"Decoder Layers    : {config.num_layers}")
print(f"FFN Dimension     : {config.ffn_dim}")

print("-" * 50)

print(f"Batch Size        : {config.batch_size}")
print(f"Learning Rate     : {config.learning_rate}")
print(f"Epochs            : {config.epochs}")

print("-" * 50)

print(f"Device            : {config.device}")

print("=" * 50)
print("Configuration Loaded Successfully!")
print("=" * 50)