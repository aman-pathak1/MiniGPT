from src.config import GPTConfig
from src.dataset import GPTDataset

config = GPTConfig()

dataset = GPTDataset(
    file_path="data/tiny_shakespeare.txt",
    config=config
)

print()

print("Dataset Length :", len(dataset))

print()

x, y = dataset[0]

print("Input Shape :", x.shape)
print("Target Shape:", y.shape)

print()

print("Input IDs:")
print(x)

print()

print("Target IDs:")
print(y)