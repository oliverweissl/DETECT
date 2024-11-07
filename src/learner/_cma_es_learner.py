from datetime import datetime
from typing import Union

import cma
import numpy as np
from numpy.typing import NDArray

from src.learner import Learner


class CMAESLearner(Learner):
    """A CMA-ES based learner."""

    _optimizer: cma.CMAEvolutionStrategy
    _x_current_continuous: NDArray[np.float64]

    def __init__(self, x0: NDArray, std: float = 0.5) -> None:
        """
        Initialize the CMA-ES Learner.

        :param x0: Initial population.
        :param std: Standard deviation of the populations dsitribution.
        """
        cma_seed = (
            int(datetime.now().timestamp() * 1e6) % 2**32
        )  # Cma seed must be smaller than 2**32.
        options = cma.CMAOptions()
        options.set("bounds", [0, 1])
        options.set("seed", cma_seed)

        initial_mean = np.ones_like(x0) / 2

        self._x_current = x0
        self._x_current_continuous = initial_mean
        self._optimizer = cma.CMAEvolutionStrategy(initial_mean, std, options)

    def new_population(self) -> None:
        """Generate a new population based on fitness of old population."""
        self._optimizer.tell(self._x_current_continuous, self._fitness)
        new_candidates = self._optimizer.ask()

        self._x_current_continuous = new_candidates
        self._x_current = np.round(new_candidates, 0)

    def get_x_current(self) -> tuple[Union[NDArray, None], NDArray]:
        """
        Return the current population in specific format.

        :return: The population as array of smx indices and smx weights.
        """
        raise NotImplemented
