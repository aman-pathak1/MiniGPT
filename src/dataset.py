import torch
from torch.utils.data import Dataset

from src.config import GPTConfig
from src.tokenizer import GPTTokenizer


class GPTDataset(Dataset):

    def __init__(self, file_path: str, config: GPTConfig):

        self.config = config

        self.tokenizer = GPTTokenizer()

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        print(f"Dataset Size : {len(text):,} characters")

        self.tokens = self.tokenizer.encode(text)

        print(f"Total Tokens : {len(self.tokens):,}")

    def __len__(self):

        return len(self.tokens) - self.config.context_length

    def __getitem__(self, index):

        input_ids = self.tokens[
            index : index + self.config.context_length
        ]

        target_ids = self.tokens[
            index + 1 : index + self.config.context_length + 1
        ]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )