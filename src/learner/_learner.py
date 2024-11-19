from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Type, Union

import numpy as np
from numpy.typing import NDArray

from .auxiliary_components import LearnerCandidate


class Learner(ABC):
    """An abstract learner class."""

    # Standard elements.
    _best_candidates: list[LearnerCandidate]
    _x_current: NDArray
    _fitness: tuple[NDArray, ...]

    _learner_type: Type
    _num_objectives: int
    _bounds: tuple[int, int]

    @abstractmethod
    def new_population(self) -> None:
        """
        Generate a new population.
        """
        ...

    @abstractmethod
    def get_x_current(self) -> tuple[Union[NDArray, None], NDArray]:
        """
        Return the current population in specific format.

        :return: The population as array of smx indices and smx weights.
        """
        ...

    def assign_fitness(self, fitness: Iterable[NDArray], data: Optional[Iterable]) -> None:
        """
        Assign fitness to the current population and extract the best individual if it beats champion.

        :param fitness: The fitness to assign.
        :param data: Additional data for the candidates.
        """
        # Format fitness into tuple if it is list or singular item.
        fitness = tuple(fitness)

        assert (
            len(fitness) == self._num_objectives
        ), f"Error: The number of fitness values is off. {len(fitness)} found, {self._num_objectives} needed."

        self._fitness = fitness
        data = data or [None] * len(self._fitness)
        self._get_best_candidates(data=data)

    def _get_best_candidates(self, data: list[Any]) -> None:
        """
        Find and assign the best candidate in the current population.

        :param data: Additional data for the candidates.
        """
        # Get the new candidates with the best fitness.
        new_candidates: set[np.int_] = set([np.argmin(metric) for metric in self._fitness])

        # We check against all current best candidates.
        # And compare them to the new candidates individually.
        for i, current_best in enumerate(self._best_candidates[:]):
            for new_index in new_candidates:
                # If any of the metrics for a current candidate is bigger than the new candidate we discard it.
                new_f = tuple([metric[new_index] for metric in self._fitness])
                if any((c > m for c, m in zip(current_best.fitness, new_f))):
                    self._best_candidates.pop(i)
                    c = LearnerCandidate(
                        self._x_current[new_index],
                        new_f,
                        data[new_index],
                    )
                    self._best_candidates.append(c)

    def reset(self) -> None:
        """Reset the learner to default."""
        self._best_candidates = [LearnerCandidate(None, np.inf)]
        self._x_current = np.random.uniform(
            low=self._bounds[0], high=self._bounds[1], size=self._x_current.shape
        )

    @property
    def best_candidates(self) -> list[LearnerCandidate]:
        """
        Get the best candidates so far (if more than one it is a pareto frontier).

        :return: The candidate.
        """
        return self._best_candidates

    @property
    def learner_type(self) -> Type:
        """
        Get the type of learner.

        :returns: The type.
        """
        return self._learner_type

    def _normalize_to_bounds(self, element: NDArray) -> NDArray:
        """
        Normalize array to range [0,1].

        :param element: The array to be normalized.
        :returns: The normalized array.
        """
        return (element - self._bounds[0]) / (self._bounds[1] - self._bounds[0])
