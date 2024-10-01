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

    def test(self, input_dataset: Dataset):
        """
        Testing the predictor for its decision boundary using a set of Inputs.

        :param input_dataset: The dataset to test.
        """
        for X, y in input_dataset:
            y_hat = self._predictor(X)
            first, second, *_ = torch.argsort(y_hat)

            if torch.argmax(y) != first:  # We are only interested in checking the boundary if the prediction matches the label
                continue

            seed, seed_p = self._get_seed(y), self._get_seed(second)
            for _ in range(self._generations):
                mixed_images = self._mix_seeds(seed, seed_p, self._learner.x_current)
                predictions = self._predictor(mixed_images)

                fitnesses = np.array([self._learner_evaluation_function(X, Xp, y, yp) for Xp, yp in zip(mixed_images, predictions)])
                self._learner.new_population(fitnesses)


    def _get_seed(self, y: NDArray) -> Any:
        """
        Get a seed from a specific label.

        :param y: The class lable.
        :returns: The seed element.
        """
        pass

    def _mix_seeds(self, seed, seed_p, genomes:NDArray) -> Tensor:
        """
        Mix seeds to get new input images based on genomes.

        :param seed: The initial seed.
        :param seed_p: The second seed.
        :param genomes: The genomes for mixing.
        :return: The generated images.
        """
        pass

    # TODO -> Class as vector
    # All classes concatenate
    # seeds are a list ?? tf is this
    # w0 seeds are matrices -> num(seeds) x label vec


    # All zs -> latent noise x number of seeds

    # -> then map zs to classes using mapping network
    # for final W`s -> wetake all weights of seeds and apply their weights --> is this new dimension to our genome??





