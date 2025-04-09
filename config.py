from dataclasses import dataclass
from typing import Literal
from typing_extensions import TypedDict

@dataclass
class TrainingConfig(TypedDict):
    optmizer: Literal['adam', 'adamw', 'sgd']
    distributed_strategy: Literal["single", "FSDP", "DDP"]

@dataclass
class ModelConfig(TypedDict):
    model_name: str
    num_classes: int