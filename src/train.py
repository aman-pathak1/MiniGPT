import os
import math
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import GPTConfig
from src.dataset import GPTDataset
from src.tokenizer import GPTTokenizer
from src.model import GPTModel


def set_seed(seed: int):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model,
    data_loader,
    device,
    loss_fn,
    use_amp
):

    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for input_ids, target_ids in data_loader:

            input_ids = input_ids.to(
                device,
                non_blocking=True
            )

            target_ids = target_ids.to(
                device,
                non_blocking=True
            )

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
    scheduler,
    epoch,
    step,
    train_loss,
    val_loss,
    config,
    filename
):

    os.makedirs(
        config.checkpoint_dir,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        filename
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": config,
        },
        checkpoint_path
    )

    print(
        f"\nCheckpoint saved: {checkpoint_path}"
    )


def create_scheduler(
    optimizer,
    warmup_steps,
    total_steps
):

    def lr_lambda(current_step):

        # Warmup
        if current_step < warmup_steps:

            return float(current_step + 1) / max(
                1,
                warmup_steps
            )

        # Cosine decay
        progress = (
            current_step - warmup_steps
        ) / max(
            1,
            total_steps - warmup_steps
        )

        progress = min(
            max(progress, 0.0),
            1.0
        )

        return 0.5 * (
            1.0 + math.cos(
                math.pi * progress
            )
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )


def main():

    # ============================================================
    # Configuration
    # ============================================================

    config = GPTConfig()

    set_seed(config.seed)

    device = torch.device(
        config.device
    )

    print("=" * 60)
    print("MiniGPT Training")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # ============================================================
    # Tokenizer
    # ============================================================

    tokenizer = GPTTokenizer()

    config.vocab_size = tokenizer.vocab_size

    print(
        f"Vocabulary Size: {config.vocab_size}"
    )

    # ============================================================
    # Dataset
    # ============================================================

    dataset = GPTDataset(
        file_path="data/tiny_shakespeare.txt",
        config=config
    )

    print(
        f"Total Dataset Samples: {len(dataset):,}"
    )

    # ============================================================
    # Train / Validation Split
    # ============================================================

    total_samples = len(dataset)

    split_index = int(
        0.9 * total_samples
    )

    # Keep a context-length gap between
    # training and validation windows.
    train_end = max(
        0,
        split_index - config.context_length
    )

    train_indices = range(
        0,
        train_end
    )

    val_indices = range(
        split_index,
        total_samples
    )

    train_dataset = Subset(
        dataset,
        train_indices
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    print(
        f"Training Samples:   {len(train_dataset):,}"
    )

    print(
        f"Validation Samples: {len(val_dataset):,}"
    )

    # ============================================================
    # DataLoader
    # ============================================================

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

    print(
        f"Training Batches:   {len(train_loader):,}"
    )

    print(
        f"Validation Batches: {len(val_loader):,}"
    )

    # ============================================================
    # Model
    # ============================================================

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

    print(
        f"\nTotal Parameters:     {total_params:,}"
    )

    print(
        f"Trainable Parameters: {trainable_params:,}"
    )

    # ============================================================
    # Loss
    # ============================================================

    loss_fn = nn.CrossEntropyLoss()

    # ============================================================
    # Optimizer
    # ============================================================

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

    # ============================================================
    # Learning Rate Scheduler
    # ============================================================

    total_steps = (
        config.epochs *
        len(train_loader)
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps
    )

    print(
        f"Total Training Steps: {total_steps:,}"
    )

    print(
        f"Warmup Steps:         {config.warmup_steps:,}"
    )

    # ============================================================
    # Mixed Precision
    # ============================================================

    use_amp = device.type == "cuda"

    if use_amp:

        scaler = torch.amp.GradScaler(
            "cuda"
        )

        print(
            "Mixed Precision: Enabled"
        )

    else:

        scaler = None

        print(
            "Mixed Precision: Disabled"
        )

    # ============================================================
    # Training State
    # ============================================================

    global_step = 0

    best_val_loss = float("inf")

    running_loss = 0.0

    model.train()

    # ============================================================
    # Training Loop
    # ============================================================

    for epoch in range(
        1,
        config.epochs + 1
    ):

        epoch_start = time.time()

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

            optimizer.zero_grad(
                set_to_none=True
            )

            # ====================================================
            # Forward Pass
            # ====================================================

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    logits = model(
                        input_ids
                    )

                    loss = loss_fn(
                        logits.reshape(
                            -1,
                            logits.size(-1)
                        ),
                        target_ids.reshape(-1)
                    )

                # =================================================
                # Backward Pass
                # =================================================

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                logits = model(
                    input_ids
                )

                loss = loss_fn(
                    logits.reshape(
                        -1,
                        logits.size(-1)
                    ),
                    target_ids.reshape(-1)
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip
                )

                optimizer.step()

            # ====================================================
            # Learning Rate Update
            # ====================================================

            scheduler.step()

            # ====================================================
            # Logging
            # ====================================================

            loss_value = loss.item()

            running_loss += loss_value

            global_step += 1

            current_lr = optimizer.param_groups[0]["lr"]

            progress_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                lr=f"{current_lr:.2e}"
            )

            # ====================================================
            # Evaluation Every N Steps
            # ====================================================

            if (
                global_step % config.eval_every == 0
            ):

                val_loss = evaluate(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    loss_fn=loss_fn,
                    use_amp=use_amp
                )

                val_perplexity = math.exp(
                    min(val_loss, 20)
                )

                print(
                    f"\nStep {global_step:,}"
                )

                print(
                    f"Validation Loss: {val_loss:.4f}"
                )

                print(
                    f"Validation PPL:  {val_perplexity:.2f}"
                )

                # ================================================
                # Save Best Model
                # ================================================

                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=global_step,
                        train_loss=loss_value,
                        val_loss=val_loss,
                        config=config,
                        filename="best_model.pt"
                    )

            # ====================================================
            # Save Checkpoint Every N Steps
            # ====================================================

            if (
                global_step % config.save_every == 0
            ):

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    train_loss=loss_value,
                    val_loss=best_val_loss,
                    config=config,
                    filename=f"checkpoint_step_{global_step}.pt"
                )

        # ========================================================
        # Epoch Statistics
        # ========================================================

        train_loss = (
            running_loss /
            len(train_loader)
        )

        running_loss = 0.0

        val_loss = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            use_amp=use_amp
        )

        epoch_time = (
            time.time() -
            epoch_start
        )

        train_perplexity = math.exp(
            min(train_loss, 20)
        )

        val_perplexity = math.exp(
            min(val_loss, 20)
        )

        print(
            "\n" + "=" * 60
        )

        print(
            f"Epoch: {epoch}/{config.epochs}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f}"
        )

        print(
            f"Train PPL:  {train_perplexity:.2f}"
        )

        print(
            f"Val PPL:    {val_perplexity:.2f}"
        )

        print(
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}"
        )

        print(
            f"Time:       {epoch_time:.2f}s"
        )

        print(
            "=" * 60
        )

        # ========================================================
        # Epoch-End Best Model Check
        # ========================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                train_loss=train_loss,
                val_loss=val_loss,
                config=config,
                filename="best_model.pt"
            )

    print(
        "\nTraining Complete!"
    )


if __name__ == "__main__":
    main()