from torchvision import transforms as T


def get_transforms(image_size: int, split: str) -> T.Compose:
    """Build the image preprocessing pipeline for a dataset split."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")

    operations = [
        T.Resize((image_size, image_size)),
    ]

    if split == "train":
        operations.append(T.RandomHorizontalFlip(p=0.5))

    operations.extend(
        [
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return T.Compose(operations)
