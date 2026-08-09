import torch

from src.config import GPTConfig
from src.tokenizer import GPTTokenizer


@torch.no_grad()
def generate(
    model,
    prompt: str,
    config: GPTConfig
):
    model.eval()

    tokenizer = GPTTokenizer()

    # Convert prompt to token IDs
    input_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=config.device
    ).unsqueeze(0)

    for _ in range(config.max_new_tokens):

        # Keep only the latest context
        input_ids = input_ids[:, -config.context_length:]

        # Model prediction
        logits = model(input_ids)

        # Get logits of the last token
        next_token_logits = logits[:, -1, :]

        # Temperature
        next_token_logits = (
            next_token_logits / config.temperature
        )

        # Top-K filtering
        if config.top_k is not None:

            top_k = min(
                config.top_k,
                next_token_logits.size(-1)
            )

            top_k_values, _ = torch.topk(
                next_token_logits,
                top_k
            )

            threshold = top_k_values[:, -1].unsqueeze(-1)

            next_token_logits = torch.where(
                next_token_logits < threshold,
                torch.full_like(
                    next_token_logits,
                    float("-inf")
                ),
                next_token_logits
            )

        # Top-P filtering
        if config.top_p is not None:

            sorted_logits, sorted_indices = torch.sort(
                next_token_logits,
                descending=True
            )

            sorted_probabilities = torch.softmax(
                sorted_logits,
                dim=-1
            )

            cumulative_probabilities = torch.cumsum(
                sorted_probabilities,
                dim=-1
            )

            sorted_indices_to_remove = (
                cumulative_probabilities > config.top_p
            )

            # Keep at least one token
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = torch.zeros_like(
                next_token_logits,
                dtype=torch.bool
            )

            indices_to_remove.scatter_(
                1,
                sorted_indices,
                sorted_indices_to_remove
            )

            next_token_logits = next_token_logits.masked_fill(
                indices_to_remove,
                float("-inf")
            )

        # Convert logits into probabilities
        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        # Sample next token
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        # Add new token to sequence
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

    # Convert token IDs back to text
    generated_text = tokenizer.decode(
        input_ids[0].tolist()
    )

    return generated_text