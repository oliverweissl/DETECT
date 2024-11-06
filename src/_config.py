from dataclasses import dataclass, field
from typing import Type
from .learner import Learner


@dataclass
class Config:
    """A simple Dataclass to store experiment configs."""
    samples_per_class: int  # How candidates should be searched for per class
    generations: int  # How many generations we search for candidates.
    population_size: int  # The size of candidate populations.
    continuous_search: bool  # Whether the search algorithm outputs continuous or discrete weights.
    mix_dim_range: tuple[int, int]  # The range of mixing dimensions used.
    predictor: str  # The predictor network.
    generator: str  # The generator network.
    genome_size: int = field(init=False)  # The size of the genome.
    learner: Type[Learner]  # The learner type.

    def __post_init__(self) -> None:
        self.genome_size = self.mix_dim_range[1]-self.mix_dim_range[0]  # Calculate genome size from range of mixing.

