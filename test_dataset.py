import torch

from src.config import GPTConfig
from src.dataset import GPTDataset


config = GPTConfig()


dataset = GPTDataset(
    file_path="data/tiny_shakespeare.txt",
    config=config
)


print("=" * 50)
print("MiniGPT Dataset Test")
print("=" * 50)


print(f"Dataset Length : {len(dataset):,}")


x, y = dataset[0]


print(f"Input Shape    : {x.shape}")
print(f"Target Shape   : {y.shape}")

print(f"Input Dtype    : {x.dtype}")
print(f"Target Dtype   : {y.dtype}")


# ============================================================
# Shape Tests
# ============================================================

assert x.shape == (
    config.context_length,
), "Input shape is incorrect"


assert y.shape == (
    config.context_length,
), "Target shape is incorrect"


# ============================================================
# Data Type Tests
# ============================================================

assert x.dtype == torch.long, (
    "Input IDs must use torch.long"
)

assert y.dtype == torch.long, (
    "Target IDs must use torch.long"
)


# ============================================================
# Next Token Prediction Test
# ============================================================

assert torch.equal(
    x[1:],
    y[:-1]
), "Target is not shifted by one token"


# ============================================================
# Dataset Length Test
# ============================================================

assert len(dataset) == (
    len(dataset.tokens) - config.context_length
), "Dataset length is incorrect"


print()
print("First Input Tokens:")
print(x[:20])


print()
print("First Target Tokens:")
print(y[:20])


print()
print("=" * 50)
print("Dataset Test Passed Successfully!")
print("=" * 50)