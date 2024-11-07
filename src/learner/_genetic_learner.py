from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

from ._learner import Learner


class GeneticLearner(Learner):
    """A learner based on genetic algorithm."""

    _mutation_rate: float
    _population_size: int
    _x_previous: NDArray

    _vec_mutate: Callable[[NDArray, NDArray, float], NDArray]

    @staticmethod
    def _mutate(x: int, p: float, mr: float) -> int:
        return 1 - x if p < mr else x

    def __init__(self, x0: NDArray, population_size: int, mutation_rate: float = 0.05) -> None:
        """
        Initialize the genetic learner.

        :param x0: The initial population.
        :param population_size: The population size.
        :param mutation_rate: The mutation rate.
        """
        self._x_current = x0
        self._population_size = population_size
        self._mutation_rate = mutation_rate

        self._vec_mutate = np.vectorize(self._mutate)
        self._best_candidate = (None, np.inf)

    def new_population(self) -> None:
        """Generate a new population based on fitness of old population."""
        parents = self._tournament_selection()

        # Generate parent pairs. If the number of parents is not divisible by two, discard random elements.
        all_indices = np.random.choice(
            np.arange(self._population_size), self._population_size, replace=False
        )
        parent_indices = np.stack(
            np.split(all_indices[: (self._population_size // 2) * 2], 2), axis=-1
        )

        # Generate new individuals from parent pairs.
        new_individuals = np.array(
            [self._crossover(parents[a], parents[b]) for a, b in parent_indices]
        )

        # Mutate the new individuals to get the children
        mutation_probabilities = np.random.rand(parent_indices.shape[0])
        mutation_array = np.tile(mutation_probabilities[:, None], (1, new_individuals.shape[1]))
        children = self._vec_mutate(new_individuals, mutation_array, self._mutation_rate)

        self._x_current = np.concatenate((parents, children), axis=0)

    def _tournament_selection(self, k: int = 2) -> NDArray:
        """
        Generate a new subset of individuals based on tournament.

        :param k: The number of contestants.
        :returns: The new population.
        """
        p, f = self._x_current.copy(), self._fitness.copy()
        available_indices = np.arange(len(p))
        winners = []
        for _ in range(self._population_size):
            individuals = np.random.choice(available_indices, size=k, replace=False)
            winner_index = np.argmin(f[individuals])
            winners.append(p[individuals[winner_index]])

            available_indices = np.delete(
                available_indices, np.where(available_indices == individuals[winner_index])
            )
        return np.array(winners)

    @staticmethod
    def _crossover(xa: NDArray, xb: NDArray) -> NDArray:
        """
        One point crossover.

        :param xa: The first individual.
        :param xb: The second individual.
        :returns: The new genome.
        """
        assert (l := len(xa)) == len(xb)
        return np.concatenate((xa[:l], xb[l:]))

    def get_x_current(self) -> tuple[Union[NDArray, None], NDArray]:
        """
        Return the current population in specific format.

        :return: The population as array of smx indices and smx weights.
        """
        smx_cond = np.zeros_like(
            self._x_current
        )  # TODO: for now only one element can be used to mix styles -> should be n elements.
        smx_weights = self._x_current
        return smx_cond, smx_weights
