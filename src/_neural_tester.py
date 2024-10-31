import torch
import numpy as np
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from datetime import datetime
from typing import Callable

from .style_mixing import StyleMixer, CandidateList, MixCandidate
from .learner import Learner

class NeuralTester:
    """A tester class for neural style mixing."""
    """Used Components."""
    _predictor: nn.Module
    _generator: nn.Module
    _mixer: StyleMixer
    _learner: Learner

    """Evaluation functions."""
    _objective_function: Callable[[Tensor, Tensor, int, int], float]

    """Additional Parameters."""
    _num_generations: int
    _num_w0: int
    _num_ws: int

    def __init__(
            self,
            *,
            predictor: nn.Module,
            generator: nn.Module,
            learner: Learner,
            objective_function: Callable[[Tensor, Tensor, int, int], float],
            num_generations: int,
            mix_dim_range: tuple[int, int],
            num_w0: int = 1,
            num_ws: int = 1,
            ):
        """
        Initialize the Neural Tester.

        :param predictor: The predictor network to test boundaries for.
        :param generator: Thy style mixer, that generates new inputs.
        :param learner: The learner to find boundary candidates.
        :param objective_function: The evaluation function for the learner.
        :param num_generations: The number of generations for the Learner.
        :param mix_dim_range: The range of layers available for style mixing (default 1-15).
        :param num_w0: The number of w0 seeds to be generated.
        :param num_ws: The number of w seeds to be generated.
        """
        self._predictor = predictor
        self._generator = generator
        self._learner = learner
        self._mixer = StyleMixer(generator, torch.device("cuda"), mix_dim_range)

        self._objective_function = objective_function

        self._num_generations = num_generations
        self._num_w0 = num_w0
        self._num_ws = num_ws

        self._predictor.eval()

    def test(self, input_dataset: Dataset):
        """
        Testing the predictor for its decision boundary using a set of (test!) Inputs.

        :param input_dataset: The dataset to test.
        """
        for X, y in input_dataset:  # X: img, y: 1xC
            y_hat = self._predictor(X)
            first, second, *_ = torch.argsort(y_hat)

            if torch.argmax(y) != first:  # We are only interested in checking the boundary if the prediction matches the label
                continue

            """
            We generate w0 and w candidates for seed generation.
            
            Not that these do not have to share a label, but for this implementation we do not control the labels seperately.
            """
            w0c = [MixCandidate(label=first, is_w0=True) for _ in range(self._num_w0)]
            wsc = [MixCandidate(label=second) for _ in range(self._num_ws)]
            candidates = CandidateList(*w0c, *wsc)

            # Now we run a search-based optimization strategy to find a good boundary candidate.
            for _ in range(self._generations):
                smx_cond_arr, smx_weights_arr = self._learner.get_x_current  # Get the initial population of style mixing conditions and weights
                assert 0 <= max(smx_cond_arr) < len(wsc), f"Error: StyleMixing Conditions reference indices of {max(smx_cond_arr)}, but we only have {len(wsc)} elements."

                images = []
                # TODO: investigate if this mixing can be parallelized
                for smx_cond, smx_weights in zip(smx_cond_arr, smx_weights_arr):
                    mixed_image = self._mixer.mix(
                        candidates=candidates,
                        smx_cond=smx_cond,
                        smx_weights=smx_weights,
                        random_seed=self._get_time_seed(),
                    )
                    images.append(mixed_image)

                predictions = self._predictor(images)
                fitness = np.array([self._objective_function(X, Xp, y, yp) for Xp, yp in zip(images, predictions)])
                self._learner.new_population(fitness)  # Generate a new population based on previous performance

    @staticmethod
    def _get_time_seed() -> int:
        """A simple function ot make a seed from the current timestamp."""
        now = datetime.now()
        return int(round(now.timestamp()))
