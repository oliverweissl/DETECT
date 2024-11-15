from abc import ABC, abstractmethod
from typing import Type, Union

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

    def assign_fitness(self, fitness: tuple[NDArray]) -> None:
        """
        Assign fitness to the current population and extract the best individual if it beats champion.

        :param fitness: The fitness to assign.
        """
        assert (
            len(fitness) == self._num_objectives
        ), f"Error: The number of fitness values is off. {len(fitness)} found, {self._num_objectives} needed."
        self._fitness = fitness
        self._get_best_candidates()

    def _get_best_candidates(self) -> None:
        """Find and assign the best candidate in the current population."""
        cand_dict = {}
        for i, metric in enumerate(self._fitness):
            c = self._x_current[np.argmin(metric)]
            if not cand_dict.get(c):
                cand_dict[c] = {i: np.min(metric)}
            else:
                cand_dict[c] |= {i: np.min(metric)}

        # We check against all current best candidates.
        # And compare them to the new candidates individually.
        for i, current_best in enumerate(self._best_candidates[:]):
            for candidate, metric_dict in cand_dict.items():
                # If any of the metrics for a current candidate is bigger than the new candidate we discard it.
                if any((c > m for c, m in zip(current_best.fitness, metric_dict.values()))):
                    self._best_candidates.pop(i)
                    self._best_candidates.append(candidate)

    def reset(self) -> None:
        """Reset the learner to default."""
        self._best_candidates = [LearnerCandidate(None, np.inf)]
        self._x_current = np.random.rand(*self._x_current.shape)

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
