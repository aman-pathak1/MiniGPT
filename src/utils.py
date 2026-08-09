import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    """
    Return the device specified by the configuration.
    """

    return torch.device(config.device)


def count_parameters(model):
    """
    Return total and trainable parameter counts.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def get_learning_rate(optimizer):
    """
    Return the current learning rate.
    """

    return optimizer.param_groups[0]["lr"]


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
    """
    Save model, optimizer and scheduler states.
    """

    os.makedirs(
        config.checkpoint_dir,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        filename
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(
        f"Checkpoint saved: {checkpoint_path}"
    )

    return checkpoint_path


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu"
):
    """
    Load a saved checkpoint.

    Returns:
        checkpoint,
        epoch,
        step,
        train_loss,
        val_loss
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    epoch = checkpoint.get(
        "epoch",
        0
    )

    step = checkpoint.get(
        "step",
        0
    )

    train_loss = checkpoint.get(
        "train_loss",
        None
    )

    val_loss = checkpoint.get(
        "val_loss",
        None
    )

    return (
        checkpoint,
        epoch,
        step,
        train_loss,
        val_loss
    )


def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.
    """

    return np.exp(
        min(loss, 20)
    )


def get_model_device(model):
    """
    Return the device on which the model is located.
    """

    return next(
        model.parameters()
    ).device