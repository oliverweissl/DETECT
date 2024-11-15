from dataclasses import dataclass, field
from typing import Callable, Type

from .learner import Learner


@dataclass
class ExperimentConfig:
    """A simple Dataclass to store experiment configs."""

    samples_per_class: int  # How candidates should be searched for per class
    generations: int  # How many generations we search for candidates.
    mix_dim_range: tuple[int, int]  # The range of mixing dimensions used.
    predictor: str  # The predictor network.
    generator: str  # The generator network.
    genome_size: int = field(init=False)  # The size of the genome.
    learner: Type[Learner]  # The learner type.
    metric: Type[Callable]

    def __post_init__(self) -> None:
        # Calculate genome size from range of mixing.
        self.genome_size = self.mix_dim_range[1] - self.mix_dim_range[0]
