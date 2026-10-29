import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.data import DataConfig
from configs.training import TrainingConfig
from configs.vit import ViTConfig
from src.data import Food101Dataset, get_transforms
from src.models import ViT


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    data_config: DataConfig,
    training_config: TrainingConfig,
    vit_config: ViTConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create Food-101 training and validation dataloaders."""

    train_dataset = Food101Dataset(
        root=data_config.root,
        transform=get_transforms(data_config.image_size, split="train"),
        split="train",
        val_split=data_config.val_split,
        seed=training_config.seed,
    )
    val_dataset = Food101Dataset(
        root=data_config.root,
        transform=get_transforms(data_config.image_size, split="val"),
        split="val",
        val_split=data_config.val_split,
        seed=training_config.seed,
    )

    pin_memory = data_config.pin_memory and device.type == "cuda"
    generator = torch.Generator().manual_seed(training_config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return the mean loss."""

    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the model and return the mean loss, accuracy, and macro F1."""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_classes = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            num_classes = logits.shape[1]
            total_correct += (predictions == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_predictions.append(predictions)
            all_labels.append(labels)

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    f1_scores = []
    for class_index in range(num_classes):
        true_positive = ((predictions == class_index) & (labels == class_index)).sum()
        false_positive = ((predictions == class_index) & (labels != class_index)).sum()
        false_negative = ((predictions != class_index) & (labels == class_index)).sum()

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(torch.nan_to_num(f1))

    macro_f1 = torch.stack(f1_scores).mean().item()

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        macro_f1,
    )


def main() -> None:
    data_config = DataConfig()
    training_config = TrainingConfig()
    vit_config = ViTConfig()

    set_seed(training_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(
        data_config,
        training_config,
        vit_config,
        device,
    )

    model = ViT(vit_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(training_config.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_loss, val_accuracy, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{training_config.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "vit_config": vars(vit_config),
                    "training_config": vars(training_config),
                    "data_config": vars(data_config),
                },
                checkpoint_dir / "best_vit.pt",
            )
            print(f"Saved new best model (val_acc={best_accuracy:.4f})")


if __name__ == "__main__":
    main()
