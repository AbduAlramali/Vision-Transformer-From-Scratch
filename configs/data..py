from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DataConfig:
    root: str = "~/datasets/food101/food-101"
    image_size: int = 224
    num_workers: int = 2
    pin_memory: bool = True
    val_split: float = 0.1
