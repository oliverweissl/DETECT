import torch
import logging
import wandb
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
        exp_start = datetime.now()
        for class_idx, sample_id in product(range(num_classes), range(samples_per_class)):
            self._init_wandb(exp_start, class_idx)  # Initialize Wandb run for logging

            w0_tensors: list[Tensor] = []
            logging.info(f"Generate seed(s) for class: {class_idx}.")
            while_counter = 0  # For logging purposes to see how many samples we need to find valid seed.
            while len(w0_tensors) < self._num_w0:  # Generate base seeds.
                while_counter += 1
                img, w = self._mixer.generate_X_w0(self._get_time_seed(), class_idx)  # We generate w0 vector and the corresponding image X.
                img_rgb = self._assure_rgb(img)  # We transform the image to RGB if it is in Grayscale.

                y_hat = self._predictor(img_rgb)
                """
                Now we select primary and secondary predictions for further style mixing.
                
                Note this can be extended to any n predictions, but for this approach we limited it to 2.
                """
                first, second, *_ = torch.argsort(y_hat, descending=True)[0]
                if first.item() == class_idx:  # We are only interested in checking the boundary if the prediction matches the label
                    w0_tensors.append(w)

            # Logging Operations
            logging.info(f"\tFound {self._num_w0} valid seed(s) after: {while_counter} iterations.")
            wandb.log({"base_image": wandb.Image(img_rgb, caption="Base Image")})
            wandb.summary["w0_trials"] = while_counter

            """
            We generate w0 and w candidates for seed generation.
            
            Not that the w0s and ws' do not have to share a label, but for this implementation we do not control the labels seperately.
            """
            w0c = [MixCandidate(label=first.item(), is_w0=True, w_tensor=tensor) for tensor in w0_tensors]  # To save compute we parse tha cached tensors of w0 vectors as we generated them already for getting the initial prediction.
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

                """We predict the label from the mixed images."""
                predictions: Tensor = self._predictor(torch.stack(images))
                predictions_labels = torch.argsort(predictions, dim=1, descending=True)[:,0].flatten()  # We extract the predicted labels

                fitness = np.array([self._objective_function(img_rgb.squeeze(0), Xp, class_idx, yp) for Xp, yp in zip(images, predictions_labels)])

                # Logging Operations
                wandb.log(
                    {
                        "min_fitness":  fitness.min(),
                        "max_fitness": fitness.max(),
                        "mean_fitness": fitness.mean(),
                        "std_fitness": fitness.std(),
                    }
                )
                if fitness.min() < self._learner.best_candidate[1]:
                    best_candidate_image = images[np.argmin(fitness)]

                self._learner.new_population(fitness)  # Generate a new population based on previous performance

            logging.info(f"\tBest candidate has a fitness of: {self._learner.best_candidate[1]}")
            wandb.summary["best_fitness"] = self._learner.best_candidate[1]
            wandb.summary["best_config"] = self._learner.best_candidate[0]
            wandb.log({"candidate_image": wandb.Image(best_candidate_image, caption="Best candidate image")})
            self._learner.reset()  # Reset the learner for new candidate.
            logging.info("\tReset learner!")

    def _init_wandb(self, exp_start: datetime, class_idx: int) -> None:
        """Initialize Wandb Run for logging"""
        wandb.init(
            project="NeuralStyleSearch",
            config={
                "num_gen": self._num_generations,
                "num_w0s": self._num_w0,
                "num_wns": self._num_ws,
                "mix_dims": self._mixer._mix_dims,
                "pop_size": self._learner._x_current.shape[0],
                "experiment_start": exp_start,
                "label": class_idx,
            }
        )

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
