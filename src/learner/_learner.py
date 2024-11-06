from abc import abstractmethod, ABC
from numpy.typing import NDArray
from typing import Union
import numpy as np
import logging


class Learner(ABC):
    """An abstract learner class."""
    # Standard elements.
    _best_candidate: tuple[Union[NDArray, None], float]
    _x_current: NDArray


    @abstractmethod
    def new_population(self, fitnesses: NDArray) -> None:
        """
        Generate a new population based on fitnesses of current population.

        :param fitnesses: The evaluated fitnesses.
        """
        ...

    @abstractmethod
    def get_x_current(self) -> tuple[NDArray, NDArray]:
        """
        Return the current population in specific format.

        :return: The population as array of smx indices and smx weights.
        """
        ...

    @property
    def best_candidate(self) -> tuple[NDArray, float]:
        """
        Get the best candidate so far.

        :return: The candidate.
        """
        return self._best_candidate

    def reset(self) -> None:
        """Reset the learner to default."""
        self._best_candidate = (None, np.inf)
        self._x_current = np.random.rand(*self._x_current.shape)
