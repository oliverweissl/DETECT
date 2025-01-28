from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Type, Union

import numpy as np
from numpy.typing import NDArray

from .auxiliary_components import OptimizerCandidate


class Learner(ABC):
    """An abstract learner class."""

    # Standard elements.
    _best_candidates: list[OptimizerCandidate]
    _previous_best: list[OptimizerCandidate]
    _x_current: NDArray
    _fitness: tuple[NDArray, ...]

    _learner_type: Type
    _num_objectives: int
    _bounds: tuple[int, int]

    _n_var: int

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

    def assign_fitness(self, fitness: Iterable[NDArray], *data: Optional[Iterable]) -> None:
        """
        Assign fitness to the current population and extract the best individual using pareto frontier.

        :param fitness: The fitness to assign.
        :param data: Additional data for the candidates.
        """
        # Format fitness into tuple if it is list or singular item.
        fitness = tuple(fitness)

        assert (
            len(fitness) == self._num_objectives
        ), f"Error: {len(fitness)} Fitness values found, {self._num_objectives} needed."

        self._fitness = fitness

        """Since we can have an arbitrary amount of metrics we extract best candidates using a pareto frontier."""
        new_metrics = np.asarray(
            fitness
        ).T  # Metrics are rows, instances are columns -> we transpose.
        old_metrics = np.asarray([cand.fitness for cand in self._best_candidates])
        metrics = np.vstack((new_metrics, old_metrics))

        new_data: list[Any] = [None] * new_metrics.shape[0] if data is None else list(zip(*data))
        data = new_data + [cand.data for cand in self._best_candidates]

        solutions = np.vstack(
            (self._x_current, np.array([cand.solution for cand in self._best_candidates]))
        )

        sorted_indices = metrics.sum(1).argsort()
        for i in range(metrics.shape[0]):
            n = sorted_indices.shape[0]
            on_pareto = np.ones(n, dtype=bool)
            if i >= n:
                break
            on_pareto[i + 1 : n] = (
                metrics[sorted_indices][i + 1 :] <= metrics[sorted_indices][i]
            ).all(axis=1) & (metrics[sorted_indices][i + 1 :] < metrics[sorted_indices][i]).any(
                axis=1
            )
            sorted_indices = sorted_indices[on_pareto[:n]]

        candidates = []
        for index in sorted_indices:
            candidates.append(OptimizerCandidate(solution=solutions[index], fitness=metrics[index], data=data[index]))
        self._previous_best = self._best_candidates
        self._best_candidates = candidates

    def reset(self) -> None:
        """Reset the learner to default."""
        self._x_current = np.random.uniform(
            low=self._bounds[0], high=self._bounds[1], size=self._x_current.shape
        )
        self._best_candidates = [
            OptimizerCandidate(self._x_current[0], [np.inf] * self._num_objectives)
        ]
        self._previous_best = self._best_candidates.copy()

    @property
    def best_candidates(self) -> list[OptimizerCandidate]:
        """
        Get the best candidates so far (if more than one it is a pareto frontier).

        :return: The candidate.
        """
        return self._best_candidates

    @property
    def previous_best(self) -> list[OptimizerCandidate]:
        """
        Get the previously best candidates.

        :return: The candidate.
        """
        return self._previous_best

    @property
    def learner_type(self) -> Type:
        """
        Get the type of learner.

        :returns: The type.
        """
        return self._learner_type

    @property
    def n_var(self) -> int:
        """
        Get size of genome for optimizer.

        :returns: The size of the genome.
        """
        return self._n_var

    def _normalize_to_bounds(self, element: NDArray) -> NDArray:
        """
        Normalize array to range [0,1].

        :param element: The array to be normalized.
        :returns: The normalized array.
        """
        return (element - self._bounds[0]) / (self._bounds[1] - self._bounds[0])
