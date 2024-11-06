import torch
import logging
import numpy as np
from torch import nn
from torch import Tensor
from datetime import datetime
from typing import Callable
from itertools import product

from .style_mixing import StyleMixer, CandidateList, MixCandidate
from .learner import Learner

class NeuralTester:
    """A tester class for neural style mixing."""
    """Used Components."""
    _predictor: nn.Module
    _generator: nn.Module
    _mixer: StyleMixer
    _learner: Learner
    _device: torch.device

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
            device: torch.device,
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
        :param device: The device to use.
        :param num_w0: The number of w0 seeds to be generated.
        :param num_ws: The number of w seeds to be generated.
        """
        self._device = device
        self._predictor = predictor.to(device)
        self._generator = generator.to(device)
        self._learner = learner
        self._mixer = StyleMixer(generator, device, mix_dim_range)

        self._objective_function = objective_function

        self._num_generations = num_generations
        self._num_w0 = num_w0
        self._num_ws = num_ws

        self._predictor.eval()

    def test(self,*, samples_per_class: int, num_classes:int):
        """
        Testing the predictor for its decision boundary using a set of (test!) Inputs.

        :param samples_per_class: The number of samples per class to test for boundaries.
        :param num_classes: The number of classes.
        """

        logging.info(f"Start testing. Number of classes: {num_classes}, iterations per class: {samples_per_class}, total iterations: {num_classes*samples_per_class}\n")
        for class_idx, sample_id in product(range(num_classes), range(samples_per_class)):
            first = None
            while_counter =  0  # For logging purposes to see how many samples we need to find valid seed.
            w0_tensors: list[Tensor] = []
            logging.info(f"Generate seed(s) for class: {class_idx}.")
            while len(w0_tensors) < self._num_w0:  # Generate base seeds.
                while_counter += 1
                X, w = self._mixer.generate_w0_X(self._get_time_seed(), class_idx)

                X_rgb = self._assure_rgb(X)
                y_hat = self._predictor(X_rgb)
                first, second, *_ = torch.argsort(y_hat, descending=True)[0]
                if first.item() == class_idx:  # We are only interested in checking the boundary if the prediction matches the label
                    w0_tensors.append(w)
            logging.info(f"\tFound {self._num_w0} valid seed(s) after: {while_counter} iterations.")

            """
            We generate w0 and w candidates for seed generation.
            
            Not that these do not have to share a label, but for this implementation we do not control the labels seperately.
            """
            w0c = [MixCandidate(label=first.item(), is_w0=True, w_tensor=tensor) for tensor in w0_tensors]
            wsc = [MixCandidate(label=second.item()) for _ in range(self._num_ws)]
            candidates = CandidateList(*w0c, *wsc)

            # Now we run a search-based optimization strategy to find a good boundary candidate.
            logging.info(f"Running Search-Algorithm for {self._num_generations} generations.")
            for _ in range(self._num_generations):
                smx_cond_arr, smx_weights_arr = self._learner.get_x_current()  # Get the initial population of style mixing conditions and weights
                assert 0 <= smx_cond_arr.max() < len(wsc), f"Error: StyleMixing Conditions reference indices of {smx_cond_arr.max()}, but we only have {len(wsc)} elements."

                images = []
                for smx_cond, smx_weights in zip(smx_cond_arr, smx_weights_arr):
                    mixed_image = self._mixer.mix(
                        candidates=candidates,
                        smx_cond=smx_cond,
                        smx_weights=smx_weights,
                        random_seed=self._get_time_seed(),
                    )
                    images.append(mixed_image)
                images = [self._assure_rgb(img) for img in images]  # Convert images to RGB if they are grayscale
                images_tensor = torch.stack(images)  # Make list of tensors into batch

                """We predict the label from the mixed images."""
                predictions: Tensor = self._predictor(images_tensor)
                predictions_labels = torch.argsort(predictions, dim=1, descending=True)[:,0].flatten()

                fitness = np.array([self._objective_function(X.squeeze(0), Xp, class_idx, yp) for Xp, yp in zip(images, predictions_labels)])
                self._learner.new_population(fitness)  # Generate a new population based on previous performance
            logging.info(f"\tBest candidate has a fitness of: {self._learner.best_candidate[1]}")
            self._learner.reset()  # Reset the learner for new candidate.
            logging.info("\tReset learner!")

    @staticmethod
    def _get_time_seed() -> int:
        """A simple function ot make a seed from the current timestamp."""
        now = datetime.now()
        return int(round(now.timestamp()))

    @staticmethod
    def _assure_rgb(image: Tensor) -> Tensor:
        """For now we only convert Grayscale to RGB."""
        if len(image.shape) == 4 and image.shape[1] == 1:
            return image.repeat(1,3,1,1)
        elif len(image.shape) == 3 and image.shape[0] == 1:
            return image.repeat(3,1,1)
        elif len(image.shape) == 2:
            return torch.unsqueeze(image, 0).repeat(3,1,1)



