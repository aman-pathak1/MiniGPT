from src.config import GPTConfig
from src.tokenizer import GPTTokenizer
from src.model import GPTModel

config = GPTConfig()

tokenizer = GPTTokenizer()

config.vocab_size = tokenizer.vocab_size

model = GPTModel(config)

print(model)
total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"\nTotal Parameters : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")