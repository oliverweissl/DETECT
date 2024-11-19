from __future__ import annotations

import logging
from datetime import datetime
from itertools import product

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

import wandb

from ._experiment_config import ExperimentConfig
from .criteria import DefaultArguments
from .learner import Learner
from .style_mixing import CandidateList, MixCandidate, StyleMixer


class NeuralTester:
    """A tester class for DNN using latent space manipulation in generative models."""

    """Used Components."""
    _predictor: nn.Module
    _generator: nn.Module
    _mixer: StyleMixer
    _learner: Learner
    _device: torch.device
    _softmax: nn.Module  # Allows to apply softmax to output of predictor

    """Additional Parameters."""
    _config: ExperimentConfig
    _num_w0: int
    _num_ws: int

    """Temporary Variables."""
    _img_rgb: Tensor

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        learner: Learner,
        device: torch.device,
        num_w0: int = 1,
        num_ws: int = 1,
    ):
        """
        Initialize the Neural Tester.

        :param config: The experiment configuration.
        :param learner: The learner to find boundary candidates.
        :param device: The device to use.
        :param num_w0: The number of w0 seeds to be generated.
        :param num_ws: The number of w seeds to be generated.
        """
        self._device = device
        self._predictor = config.predictor.to(device)
        self._generator = config.generator.to(device)
        self._learner = learner
        self._mixer = StyleMixer(self._generator, device, config.mix_dim_range)

        self._num_w0 = num_w0
        self._num_ws = num_ws

        self._config = config
        self._predictor.eval()
        self._softmax = torch.nn.Softmax(dim=1)

    def test(self):
        """Testing the predictor for its decision boundary using a set of (test!) Inputs."""
        spc, nc = self._config.samples_per_class, self._config.classes

        logging.info(
            f"Start testing. Number of classes: {nc}, iterations per class: {spc}, total iterations: {nc*spc}\n"
        )
        exp_start = datetime.now()
        for class_idx, sample_id in product(range(nc), range(spc)):
            self._init_wandb(exp_start, class_idx)  # Initialize Wandb run for logging

            w0_tensors: list[Tensor] = []
            logging.info(f"Generate seed(s) for class: {class_idx}.")
            # For logging purposes to see how many samples we need to find valid seed.
            while_counter = 0
            while len(w0_tensors) < self._num_w0:  # Generate base seeds.
                while_counter += 1
                # We generate w0 vector and the corresponding image X.
                img, w = self._mixer.generate_X_w0(self._get_time_seed(), class_idx)
                # We transform the image to RGB if it is in Grayscale.
                self._img_rgb = self._assure_rgb(img)

                y_hat = self._predictor(self._img_rgb)
                """
                Now we select primary and secondary predictions for further style mixing.
                
                Note this can be extended to any n predictions, but for this approach we limited it to 2.
                """
                first, second, *_ = torch.argsort(y_hat, descending=True)[0]
                # We are only interested in checking the boundary if the prediction matches the label
                if first.item() == class_idx:
                    w0_tensors.append(w)

            # Logging Operations
            logging.info(f"\tFound {self._num_w0} valid seed(s) after: {while_counter} iterations.")
            wandb.log({"base_image": wandb.Image(self._img_rgb, caption="Base Image")})
            wandb.summary["w0_trials"] = while_counter

            """
            We generate w0 and w candidates for seed generation.
            
            Not that the w0s and ws' do not have to share a label, but for this implementation we do not control the labels seperately.
            """
            # To save compute we parse tha cached tensors of w0 vectors as we generated them already for getting the initial prediction.
            w0c = [
                MixCandidate(label=first.item(), is_w0=True, w_tensor=tensor)
                for tensor in w0_tensors
            ]
            wsc = [MixCandidate(label=second.item()) for _ in range(self._num_ws)]
            candidates = CandidateList(*w0c, *wsc)

            # Now we run a search-based optimization strategy to find a good boundary candidate.
            logging.info(f"Running Search-Algorithm for {self._config.generations} generations.")

            # We define the inner loop with its parameters.
            for _ in range(self._config.generations):
                images, fitness = self._inner_loop(
                    candidates,
                    class_idx,
                    second,
                )
                # Assign fitness to current population and additional data (in our case images).
                self._learner.assign_fitness(fitness, data=images)
                # Generate a new population based on previous performance.
                self._learner.new_population()
            else:
                images, fitness = self._inner_loop(candidates, class_idx, second)
                self._learner.assign_fitness(fitness, data=images)

            logging.info(
                f"\tBest candidate(s) have a fitness of: {', '.join([str(c.fitness) for c in self._learner.best_candidates])}"
            )
            wandb.summary["boundary_to"] = second.item()

            wandb.log(
                {
                    "best_candidates": wandb.Table(
                        columns=[f"Obj{i}" for i in range(len(self._config.metrics))]
                        + [f"Genome_{i}" for i in range(*self._config.mix_dim_range)]
                        + ["Image"],
                        data=[
                            [
                                *c.fitness,
                                *c.solution,
                                wandb.Image(c.data),
                            ]
                            for c in self._learner.best_candidates
                        ],
                    ),
                }
            )
            self._learner.reset()  # Reset the learner for new candidate.
            logging.info("\tReset learner!")

    def _inner_loop(
        self,
        candidates: CandidateList,
        y: int,
        y2: int,
    ) -> tuple[list[Tensor], tuple[NDArray, ...]]:
        """
        The inner loop for the learner.

        :param candidates: The mixing candidates to be used.
        :param y: The base class label.
        :param y2: The second most likely label.
        :returns: The images generated, and the corresponding fitness.
        """
        # Get the initial population of style mixing conditions and weights
        smx_cond_arr, smx_weights_arr = self._learner.get_x_current()
        assert (
            0 <= smx_cond_arr.max() < self._num_ws
        ), f"Error: StyleMixing Conditions reference indices of {smx_cond_arr.max()}, but we only have {self._num_ws} elements."

        images = []
        for smx_cond, smx_weights in zip(smx_cond_arr, smx_weights_arr):
            mixed_image = self._mixer.mix(
                candidates=candidates,
                smx_cond=smx_cond,
                smx_weights=smx_weights,
                random_seed=self._get_time_seed(),
            )
            images.append(mixed_image)
        # Convert images to RGB if they are grayscale
        images = [self._assure_rgb(img) for img in images]

        """We predict the label from the mixed images."""
        predictions: Tensor = self._predictor(torch.stack(images))
        predictions_softmax = self._softmax(predictions)
        fitness = tuple(
            [
                np.array(
                    [
                        criterion.evaluate(
                            default_args=DefaultArguments(
                                i1=self._img_rgb.squeeze(0),
                                i2=Xp,
                                y1=y,
                                y2=y2,
                                yp=yp,
                            )
                        )
                        for Xp, yp in zip(images, predictions_softmax)
                    ]
                )
                for criterion in self._config.metrics
            ]
        )

        # Logging Operations
        results = {}
        # Log statistics for each objective function seperatly.
        for i, obj in enumerate(fitness):
            results |= {
                f"min_obj{i}": obj.min(),
                f"max_obj{i}": obj.max(),
                f"mean_obj{i}": obj.mean(),
                f"std_obj{i}": obj.std(),
            }
        wandb.log(results)

        return images, fitness

    def _init_wandb(self, exp_start: datetime, class_idx: int) -> None:
        """
        Initialize Wandb Run for logging

        :param exp_start: The start of the experiment (for grouping purposes).
        :param class_idx: The class index to search boundary candidates for.
        """
        wandb.init(
            project="NeuralStyleSearch",
            config={
                "num_gen": self._config.generations,
                "num_w0s": self._num_w0,
                "num_wns": self._num_ws,
                "mix_dims": self._mixer._mix_dims,
                "pop_size": self._learner._x_current.shape[0],
                "experiment_start": exp_start,
                "label": class_idx,
                "learner_type": self._learner.learner_type,
            },
        )

    @staticmethod
    def _get_time_seed() -> int:
        """
        A simple function ot make a seed from the current timestamp.

        :returns: A seed based on the timestamp.
        """
        now = datetime.now()
        return int(round(now.timestamp()))

    @staticmethod
    def _assure_rgb(image: Tensor) -> Tensor:
        """
        For now we only convert Grayscale to RGB.

        :param image: The image to be converted.
        :returns: The converted image (3 x H x W).
        """
        if len(image.shape) == 4 and image.shape[1] == 1:
            return image.repeat(1, 3, 1, 1)
        elif len(image.shape) == 3 and image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        elif len(image.shape) == 2:
            return torch.unsqueeze(image, 0).repeat(3, 1, 1)
