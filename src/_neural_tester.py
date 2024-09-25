from torch import nn
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from objective_functions import get_accuracy, get_penalized_distance
from learner import Learner
from typing import Callable, Any
from numpy.typing import NDArray

class NeuralTester:
    """A tester class for neural style mixing."""

    _predictor: nn.Module
    _generator: nn.Module
    _learner: Learner

    _predictor_evaluation_function: Callable[[Tensor, Tensor], float]
    _learner_evaluation_function: Callable[[NDArray, NDArray, int, int], float]

    _num_generations: int

    def __init__(
            self,
            *,
            predictor: nn.Module,
            generator: nn.Module,
            learner: Learner,
            predictor_evaluation_function: Callable[[Tensor, Tensor], float],
            learner_evaluation_function: Callable[[NDArray, NDArray, int, int], float],
            generations: int,
            ):
        """
        Initialize the Neural Tester.

        :param predictor: The predictor network to test boundaries for.
        :param generator: Thy style mixer, that generates new inputs.
        :param learner: The learner to find boundary candidates.
        :param predictor_evaluation_function: The evaluation function  for the predictor network.
        :param learner_evaluation_function: The evaluation function for the learner.
        :param generations: The number of generations for the Learner.
        """
        self._predictor = predictor
        self._generator = generator
        self._learner = learner

        self._predictor_evaluation_function = predictor_evaluation_function
        self._learner_evaluation_function = learner_evaluation_function

        self._generations = generations

        self._predictor.eval()

    def test(self, input: Dataset):
        """
        Testing the predictor for its desicion boundary using a set of Inputs.

        :param input: The dataset to test.
        """
        for X, y in input:
            y_hat = self._predictor(X)
            first, second, *_ = torch.argsort(y_hat)

            if torch.argmax(y) != first:  # We are only interested in checking the boundary if the prediction matches the label
                continue

            seed, seed_p = self._get_seeds(y, second)
            pop_size = len(self._learner.x_current)
            for _ in range(self._generations):
                predictions = self._predictor()


                fitnesses = self._predictor_evaluation_function(np.full(pop_size, fill_value=X), )

                self._learner.new_population()


    def _get_seeds(self,y, y_p) -> tuple[Any, Any]:
        pass

    def _mix_seeds(self, seed, seed_p, genomes:NDArray) -> Tensor:
        """
        Mix seeds to get new input images based on genomes.

        :param seed:
        :param seed_p:
        :param genomes: The genomes for mixing.
        :return: The new images.
        """
        seed[]






