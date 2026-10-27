from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class Food101Dataset(Dataset):
    """Food-101 dataset with an optional deterministic validation split."""

    def __init__(self, root, transform, split="train", val_split=0.1, seed=42):
        self.root = Path(root).expanduser()
        self.images_dir = self.root / "images"
        self.meta_dir = self.root / "meta"
        self.transform = transform

        with (self.meta_dir / "classes.txt").open() as file:
            self.classes = [line.strip() for line in file if line.strip()]

        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        if split == "test":
            self.samples = self._load_samples("test")
        else:
            samples = self._load_samples("train")

            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(samples), generator=generator).tolist()

            val_size = round(len(samples) * val_split)

            if split == "val":
                selected_indices = indices[:val_size]
            else:
                selected_indices = indices[val_size:]

            self.samples = [samples[i] for i in selected_indices]

    def _load_samples(self, split):
        samples = []

        with (self.meta_dir / f"{split}.txt").open() as file:
            for line in file:
                relative_path = line.strip()

                if not relative_path:
                    continue

                class_name = relative_path.split("/")[0]
                image_path = self.images_dir / f"{relative_path}.jpg"
                label = self.class_to_idx[class_name]

                samples.append((image_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        image = self.transform(image)
        return image, label
