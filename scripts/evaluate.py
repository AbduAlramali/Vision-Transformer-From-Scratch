from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.data import DataConfig
from configs.training import TrainingConfig
from configs.vit import ViTConfig
from src.data import Food101Dataset, get_transforms
from src.models import ViT


def load_checkpoint(path: Path) -> dict:
    """Load a training checkpoint onto the CPU."""
    return torch.load(path, map_location="cpu", weights_only=True)


def macro_f1(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute macro-averaged F1 over all classes."""
    f1_scores = []

    for class_index in range(num_classes):
        true_positive = ((predictions == class_index) & (labels == class_index)).sum()
        false_positive = ((predictions == class_index) & (labels != class_index)).sum()
        false_negative = ((predictions != class_index) & (labels == class_index)).sum()

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(torch.nan_to_num(f1))

    return torch.stack(f1_scores).mean().item()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run the model over the dataloader and compute test metrics."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0
    num_classes = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            num_classes = logits.shape[1]
            predictions = logits.argmax(dim=1)
            top5_predictions = logits.topk(min(5, num_classes), dim=1).indices

            top1_correct += (predictions == labels).sum().item()
            top5_correct += (
                (top5_predictions == labels.unsqueeze(1)).any(dim=1).sum().item()
            )
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_predictions.append(predictions)
            all_labels.append(labels)

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)

    return {
        "loss": total_loss / total_samples,
        "accuracy": top1_correct / total_samples,
        "top5_accuracy": top5_correct / total_samples,
        "macro_f1": macro_f1(predictions, labels, num_classes),
    }


def main() -> None:
    checkpoint_path = Path("checkpoints/best_vit.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path)

    vit_config = ViTConfig(**checkpoint["vit_config"])
    data_config = DataConfig(**checkpoint["data_config"])
    training_config = TrainingConfig(**checkpoint["training_config"])

    model = ViT(vit_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_dataset = Food101Dataset(
        root=data_config.root,
        transform=get_transforms(data_config.image_size, split="test"),
        split="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory and device.type == "cuda",
    )

    metrics = evaluate_model(
        model,
        test_loader,
        nn.CrossEntropyLoss(),
        device,
    )

    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test top-5 accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
