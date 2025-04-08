from dataclasses import dataclass
from typing import Literal
from typing_extensions import TypedDict

@dataclass
class TrainingConfig(TypedDict):
    lr: float
    optmizer: Literal['adam', 'adamw', 'sgd']

@dataclass
class ModelConfig(TypedDict):
    model_name: str
    num_classes: int

