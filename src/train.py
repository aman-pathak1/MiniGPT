import os
import math
import time
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config import GPTConfig
from src.dataset import GPTDataset
from src.tokenizer import GPTTokenizer
from src.model import GPTModel


def set_seed(seed: int):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, device, loss_fn, use_amp):

    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for input_ids, target_ids in data_loader:

            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    logits = model(input_ids)

                    loss = loss_fn(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1)
                    )

            else:

                logits = model(input_ids)

                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1)
                )

            total_loss += loss.item()
            total_batches += 1

    model.train()

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches


def save_checkpoint(
    model,
    optimizer,
    epoch,
    step,
    train_loss,
    val_loss,
    config
):

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"checkpoint_epoch_{epoch}_step_{step}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
        },
        checkpoint_path
    )

    print(f"\nCheckpoint saved: {checkpoint_path}")


def main():

    # ==========================
    # Configuration
    # ==========================

    config = GPTConfig()

    set_seed(config.seed)

    device = torch.device(config.device)

    print("=" * 60)
    print("MiniGPT Training")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":

        print(f"GPU: {torch.cuda.get_device_name(0)}")

        print(
            f"GPU Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # ==========================
    # Tokenizer
    # ==========================

    tokenizer = GPTTokenizer()

    config.vocab_size = tokenizer.vocab_size

    print(f"Vocabulary Size: {config.vocab_size}")

    # ==========================
    # Dataset
    # ==========================

    dataset = GPTDataset(
        file_path="data/tiny_shakespeare.txt",
        config=config
    )

    print(f"Total Dataset Samples: {len(dataset):,}")

    # ==========================
    # Train / Validation Split
    # ==========================

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    print(f"Training Samples:   {len(train_dataset):,}")
    print(f"Validation Samples: {len(val_dataset):,}")

    # ==========================
    # DataLoader
    # ==========================

    num_workers = 0

    if device.type == "cuda":
        num_workers = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda"
    )

    print(f"Training Batches:   {len(train_loader):,}")
    print(f"Validation Batches: {len(val_loader):,}")

    # ==========================
    # Model
    # ==========================

    model = GPTModel(config)

    model = model.to(device)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nTotal Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # ==========================
    # Loss
    # ==========================

    loss_fn = nn.CrossEntropyLoss()

    # ==========================
    # Optimizer
    # ==========================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(
            config.adam_beta1,
            config.adam_beta2
        ),
        eps=config.adam_eps,
        weight_decay=config.weight_decay
    )

    # ==========================
    # Mixed Precision
    # ==========================

    use_amp = device.type == "cuda"

    if use_amp:

        scaler = torch.amp.GradScaler("cuda")

        print("Mixed Precision: Enabled")

    else:

        scaler = None

        print("Mixed Precision: Disabled")

    # ==========================
    # Training
    # ==========================

    global_step = 0

    best_val_loss = float("inf")

    model.train()

    for epoch in range(1, config.epochs + 1):

        epoch_start = time.time()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.epochs}"
        )

        for input_ids, target_ids in progress_bar:

            input_ids = input_ids.to(
                device,
                non_blocking=True
            )

            target_ids = target_ids.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            # ==========================
            # Forward Pass
            # ==========================

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    logits = model(input_ids)

                    loss = loss_fn(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1)
                    )

                # ==========================
                # Backward Pass
                # ==========================

                scaler.scale(loss).backward()

                # Unscale before gradient clipping

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip
                )

                scaler.step(optimizer)

                scaler.update()

            else:

                logits = model(input_ids)

                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1)
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip
                )

                optimizer.step()

            # ==========================
            # Logging
            # ==========================

            loss_value = loss.item()

            running_loss += loss_value

            global_step += 1

            average_loss = running_loss / (
                global_step % len(train_loader)
                if global_step % len(train_loader) != 0
                else len(train_loader)
            )

            progress_bar.set_postfix(
                loss=f"{loss_value:.4f}"
            )

        # ==========================
        # Epoch Statistics
        # ==========================

        train_loss = running_loss / len(train_loader)

        val_loss = evaluate(
            model,
            val_loader,
            device,
            loss_fn,
            use_amp
        )

        epoch_time = time.time() - epoch_start

        train_perplexity = math.exp(
            min(train_loss, 20)
        )

        val_perplexity = math.exp(
            min(val_loss, 20)
        )

        print("\n" + "=" * 60)

        print(f"Epoch: {epoch}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        print(f"Train PPL:  {train_perplexity:.2f}")
        print(f"Val PPL:    {val_perplexity:.2f}")

        print(f"Time:       {epoch_time:.2f}s")

        print("=" * 60)

        # ==========================
        # Best Model
        # ==========================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            os.makedirs(
                config.checkpoint_dir,
                exist_ok=True
            )

            best_model_path = os.path.join(
                config.checkpoint_dir,
                "best_model.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": config,
                },
                best_model_path
            )

            print(
                f"Best model saved: {best_model_path}"
            )

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()