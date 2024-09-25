from abc import abstractmethod, ABC
from numpy.typing import NDArray


class Learner(ABC):
    """An abstract learner class."""
    # Standard elements.
    _best_candidate: tuple[NDArray, float]
    _x_current: NDArray

    @abstractmethod
    def new_population(self, fitnesses: NDArray) -> None:
        """
        Generate a new population based on fitnesses of current population.

        :param fitnesses: The evaluated fitnesses.
        """
        ...

    @property
    def best_candidate(self) -> tuple[NDArray, float]:
        """
        Get the best candidate so far.

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

