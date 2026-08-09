import torch
from dataclasses import dataclass


@dataclass
class GPTConfig:

    # ==========================
    # Model Architecture
    # ==========================

    vocab_size: int = 50257

    context_length: int = 512
    max_position_embeddings: int = 512

    embedding_dim: int = 384

    num_heads: int = 6
    num_layers: int = 6

    ffn_dim: int = 1536

    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    use_bias: bool = True
    tie_word_embeddings: bool = True

    activation: str = "gelu"

    # ==========================
    # Training
    # ==========================

    batch_size: int = 8
    epochs: int = 10

    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    gradient_clip: float = 1.0

    warmup_steps: int = 1000

    # ==========================
    # Generation
    # ==========================

    temperature: float = 0.8

    top_k: int = 50
    top_p: float = 0.95

    max_new_tokens: int = 150

    # ==========================
    # Checkpoints
    # ==========================

    save_every: int = 1000
    eval_every: int = 500

    checkpoint_dir: str = "checkpoints"

    # ==========================
    # Misc
    # ==========================

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    seed: int = 42