from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union
from numpy.typing import NDArray


@dataclass
class LearnerCandidate:
    """A candidate solution found by the learner."""

    solution: Optional[NDArray]
    fitness: Union[tuple[float, ...], float, list[float]]

    # Additional data that might be used.
    data: Optional[Any] = field(default=None)
    parents: Optional[list[LearnerCandidate]] = field(default=None)

    def __post_init__(self) -> None:
        """Post init processing of data."""
        if isinstance(self.fitness, float):
            self.fitness: tuple[float, ...] = (self.fitness,)
        elif isinstance(self.fitness, list):
            self.fitness: tuple[float, ...] = tuple(self.fitness)
