from dataclasses import dataclass, field
from typing import Type
from .learner import Learner


@dataclass
class Config:
    """A simple Dataclass to store experiment configs."""
    generations: int
    population_size: int
    continuous_search: bool
    mix_dim_range: tuple[int, int]
    predictor: str
    generator: str
    genome_size: int = field(init=False)
    learner: Type[Learner]

    def __post_init__(self) -> None:
        self.genome_size = self.mix_dim_range[1]-self.mix_dim_range[0]

