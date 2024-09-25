import numpy as np
from numpy.typing import NDArray


class RevDELearner:
    """
    A Reverse Differential Evolutionary Learner.
    For theory see: https://arxiv.org/pdf/2002.02869.
    """
    # General states
    _generation: int
    _best_candidate: tuple[NDArray, float]
    _mean_first: float

    # Parameters
    _population_size: int
    _bounds: tuple[int, int]
    F: float
    CR: float

    # Population stuff
    _x_current: NDArray
    _x_previous: NDArray
    _fitness_current: NDArray

    def __init__(
            self,
            x0: NDArray,
            population_size: int,
            bounds:tuple[int, int] = (0,1),
            f: float = 0.9,
            cr: float = 0.5,
    ) -> None:
        """
        Initialize the Reverse Differential Evolutionary Learner.

        :param x0: The initial genetic material.
        :param population_size: The size of the population.
        :param bounds: The bounds of the population.
        :param f: The scaling factor.
        :param cr: The crossover rate.
        """
        self._generation = 0
        self._best_fitness = np.inf
        self._population_size = population_size

        self._bounds = bounds  # The bounds of genome values.
        self.F = f  # The scaling factor.
        self.CR = cr # The crossover rate.

        self._x_current = x0
        self._x_previous = np.empty(shape=x0.shape, dtype=x0.dtype)
        self._fitness = np.empty(shape=x0.shape, dtype=float)

    def new_population(self, fitnesses: NDArray) -> NDArray:
        self._mean_first = np.mean(fitnesses) if self._generation == 0 else self._mean_first

        x, f = self._select(self._x_current, fitnesses)
        x_cand, f_min = x[np.argmin(f)], np.min(f)
        self._best_candidate = (x_cand, f_min) if f_min < self._best_candidate[1] else self._best_candidate

        self._x_current = self._recombination(x)
        self._x_previous = x
        self._fitness = f
        self._generation += 1

    @property
    def best_candidate(self) -> tuple[NDArray, float]:
        """
        Get the best candidate in the RevDE.

        :return: The candidate.
        """
        return self._best_candidate

    @property
    def x_current(self) -> NDArray:
        """
        Get the current population of genomes.

        :return: The population.
        """
        return self._x_current


    def _recombination(self, x: NDArray) -> NDArray:
        """
        Recombination method of genetic material.

        This approach discretizes the values since our genome is in N.

        :param x: The population to do recombination with.
        :return: The recombined population.
        """
        indices_1 = np.arange(x.shape[0])
        x_1 = x[indices_1]
        indices_2 = np.random.permutation(x.shape[0])
        x_2 = x_1[indices_2]
        indices_3 = np.random.permutation(x.shape[0])
        x_3 = x_2[indices_3]

        y_1 = np.clip(x_1 + self.F * (x_2 - x_3), *self._bounds)
        y_2 = np.clip(x_2 + self.F * (x_3 - y_1), *self._bounds)
        y_3 = np.clip(x_3 + self.F * (y_1 - y_2), *self._bounds)

        # uniform crossover
        if self.CR < 1.:
            p_1 = np.random.binomial(1, self.CR, y_1.shape)
            p_2 = np.random.binomial(1, self.CR, y_2.shape)
            p_3 = np.random.binomial(1, self.CR, y_3.shape)
            y_1 = np.round(p_1 * y_1 + (1. - p_1) * x_1, decimals=0)
            y_2 = np.round(p_2 * y_2 + (1. - p_2) * x_2, decimals=0)
            y_3 = np.round(p_3 * y_3 + (1. - p_3) * x_3, decimals=0)

        x[[indices_1, indices_2, indices_3]] = y_1, y_2, y_3
        return x

    def _select(self, x: NDArray, f:NDArray) -> tuple[NDArray, NDArray]:
        """
        Select a subset of a population x by their fitness values.

        :param x: The population of candidates.
        :param f: The fitness values.
        :return: The selected population with its fitness.
        """
        indices = np.argsort(f)
        x_new, f_new = x[indices], f[indices]
        return x_new[:self._population_size], f_new[:self._population_size]
