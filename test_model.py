import torch

from src.config import GPTConfig
from src.tokenizer import GPTTokenizer
from src.model import GPTModel


config = GPTConfig()

tokenizer = GPTTokenizer()

# Keep vocabulary consistent with tokenizer
config.vocab_size = tokenizer.vocab_size


model = GPTModel(config)

model.eval()


text = "Hello GPT!"

token_ids = tokenizer.encode(text)


input_ids = torch.tensor(
    [token_ids],
    dtype=torch.long
)


with torch.no_grad():

    logits = model(input_ids)


print("=" * 50)
print("MiniGPT Model Test")
print("=" * 50)


print("\nInput Text:")
print(text)


print("\nToken IDs:")
print(token_ids)


print("\nInput Shape:")
print(input_ids.shape)


print("\nOutput Shape:")
print(logits.shape)


# ============================================================
# Shape Tests
# ============================================================

expected_input_shape = (
    1,
    len(token_ids)
)

expected_output_shape = (
    1,
    len(token_ids),
    config.vocab_size
)


assert input_ids.shape == expected_input_shape, (
    f"Expected input shape {expected_input_shape}, "
    f"got {input_ids.shape}"
)


assert logits.shape == expected_output_shape, (
    f"Expected output shape {expected_output_shape}, "
    f"got {logits.shape}"
)


# ============================================================
# Vocabulary Dimension Test
# ============================================================

assert logits.size(-1) == config.vocab_size, (
    "LM Head output does not match vocabulary size"
)


# ============================================================
# NaN / Inf Test
# ============================================================

assert torch.isfinite(logits).all(), (
    "Model output contains NaN or Inf"
)


print("\nExpected Input Shape:")
print(expected_input_shape)


print("\nExpected Output Shape:")
print(expected_output_shape)


print()
print("=" * 50)
print("Model Test Passed Successfully!")
print("=" * 50)