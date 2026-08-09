from src.config import GPTConfig
from src.tokenizer import GPTTokenizer
from src.model import GPTModel


config = GPTConfig()

tokenizer = GPTTokenizer()

# Keep vocabulary consistent with tokenizer
config.vocab_size = tokenizer.vocab_size


model = GPTModel(config)


print("=" * 60)
print("MiniGPT Model Summary")
print("=" * 60)

print(model)


# ============================================================
# Parameter Count
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


non_trainable_params = (
    total_params - trainable_params
)


print()
print("=" * 60)

print(
    f"Total Parameters     : {total_params:,}"
)

print(
    f"Trainable Parameters : {trainable_params:,}"
)

print(
    f"Non-Trainable        : {non_trainable_params:,}"
)

print("=" * 60)


# ============================================================
# Basic Architecture Checks
# ============================================================

assert total_params > 0, (
    "Model has no parameters"
)

assert trainable_params > 0, (
    "Model has no trainable parameters"
)

assert config.embedding_dim % config.num_heads == 0, (
    "Embedding dimension must be divisible by number of heads"
)


print()
print("Architecture checks passed!")
print("Model Summary Test Passed Successfully!")