from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    seed: int = 42
