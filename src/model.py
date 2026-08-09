import torch
import torch.nn as nn

from src.config import GPTConfig
from src.embeddings import Embeddings
from src.decoder import DecoderBlock


class GPTModel(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # Token + Positional Embeddings
        self.embeddings = Embeddings(config)

        # Transformer Decoder Blocks
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        # Final Layer Normalization
        self.final_layer_norm = nn.LayerNorm(
            config.embedding_dim,
            eps=config.layer_norm_eps
        )

        # Language Modeling Head
        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Weight Tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.token_embedding.weight

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

        elif isinstance(module, nn.LayerNorm):

            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids):

        # Token + positional embeddings
        x = self.embeddings(input_ids)

        # Pass through all decoder blocks
        for decoder in self.decoder_blocks:
            x = decoder(x)

        # Final LayerNorm
        x = self.final_layer_norm(x)

        # Convert hidden states to vocabulary logits
        logits = self.lm_head(x)

        return logits