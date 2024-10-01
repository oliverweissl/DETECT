from typing import Callable
from numpy.typing import NDArray
import numpy as np
from ._learner import Learner

class GeneticLearner(Learner):
    """A learner based on genetic algorithm."""

    _mutation_rate: float
    _population_size: int
    _x_previous: NDArray

    _vec_mutate: Callable[[NDArray, NDArray, float], NDArray]

    @staticmethod
    def _mutate(x: int, p: float, mr: float) -> int:
        return 1-x if p < mr else x


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

    def new_population(self, fitnesses: NDArray) -> None:
        """
        Generate a new population based on fitness of old population.

        :param fitnesses: The fitness of old population.
        """
        parents = self._tournament_selection(self._x_current, fitnesses, self._population_size)
        indices = np.vstack(np.split(np.random.choice(self._population_size, replace=False), 2))
        probabilities = np.random.rand(len(indices))
        new_individuals = np.array([self._crossover(parents[a] ,parents[b]) for a,b in indices])
        children = self._vec_mutate(new_individuals, probabilities, self._mutation_rate)

        self._x_current = parents + children

    @staticmethod
    def _tournament_selection(population: NDArray, fitnesses: NDArray, to_keep: int, k: int = 2) -> NDArray:
        """
        Generate a new subset of individuals based on tournament.


        :param population: The population
        :param fitnesses: The fitness of individuals.
        :param to_keep: The number of individuals to keep.
        :param k: The number of contestants.
        :returns: The new population.
        """
        all_indices = np.arange(len(population))
        winners = []
        for _ in range(to_keep):
            individuals = np.random.choice(all_indices, size=k, replace=False)
            winner = np.argmax(fitnesses[individuals])
            winners.append(population[winner])

            all_indices = np.delete(all_indices, winner)
            population = np.delete(population, winner)
            fitnesses = np.delete(fitnesses, winner)
        return np.array(winners)

    @staticmethod
    def _crossover(xa:NDArray, xb:NDArray) -> NDArray:
        """
        One point crossover.

        :param xa: The first individual.
        :param xb: The second individual.
        :returns: The new genome.
        """
        assert (l:=len(xa)) == len(xb)
        return np.concatenate(xa[:l], xb[l:])
