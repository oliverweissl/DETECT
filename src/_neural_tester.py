from torch import nn
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime
from objective_functions import get_accuracy, get_penalized_distance
from models import StyleMixer, CandidateList, MixCandidate
from learner import Learner
from typing import Callable, Any
from numpy.typing import NDArray

class NeuralTester:
    """A tester class for neural style mixing."""

    _predictor: nn.Module
    _generator: nn.Module
    _mixer: StyleMixer
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
        self._mixer = StyleMixer(generator, torch.device("cuda"), (1, 15))

        self._predictor_evaluation_function = predictor_evaluation_function
        self._learner_evaluation_function = learner_evaluation_function

        self._generations = generations

        self._predictor.eval()

    def test(self, input_dataset: Dataset):
        """
        Testing the predictor for its decision boundary using a set of Inputs.

        :param input_dataset: The dataset to test.
        """
        for X, y in input_dataset:  # X: img, y: 1xC
            y_hat = self._predictor(X)
            first, second, *_ = torch.argsort(y_hat)

            if torch.argmax(y) != first:  # We are only interested in checking the boundary if the prediction matches the label
                continue

            """The candidate usage could be better!!"""
            c1, c2 = MixCandidate(label=first, is_w0=True), MixCandidate(label=second)
            candidates = CandidateList(c1, c2)

            for _ in range(self._generations):
                smx_indices_arr, smx_weights_arr = self._learner.get_x_current

                images = []
                for smx_indices, smx_weights in zip(smx_indices_arr, smx_weights_arr):
                    mixed_image = self._mixer.mix(
                        candidates=candidates,
                        smx_indices=smx_indices,
                        smx_weights=smx_weights,
                        random_seed=self._get_random_seed(),
                    )
                    images.append(mixed_image)

                predictions = self._predictor(images)
                fitness = np.array([self._learner_evaluation_function(X, Xp, y, yp) for Xp, yp in zip(images, predictions)])
                self._learner.new_population(fitness)

    @staticmethod
    def _get_random_seed() -> int:
        now = datetime.now()
        rand = np.random.randint(int(round(now.timestamp())))
        return rand
