from __future__ import annotations

import logging
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
import torch
import wandb
from numpy.typing import NDArray
from torch import Tensor, nn
from wandb import UsageError

from ._experiment_config import ExperimentConfig
from .criteria import CriteriaArguments, Criterion
from .manipulator import CandidateList, Manipulator, MixCandidate
from .optimizer import Learner
from .persistence import DefaultDF


class NeuralTester:
    """A tester class for DNN using latent space manipulation in generative models."""

    """Used Components."""
    _sut: nn.Module
    _manipulator: Manipulator
    _optimizer: Learner
    _objectives: list[Criterion]

    _softmax: nn.Module

    """Additional Parameters."""
    _config: ExperimentConfig
    _num_w0: int
    _num_ws: int

    """Temporary Variables."""
    _img_rgb: Tensor

    def __init__(
        self,
        *,
        sut: nn.Module,
        manipulator: Manipulator,
        optimizer: Learner,
        objectives: list[Criterion],
        config: ExperimentConfig,
        frontier_pairs: bool,
        num_w0: int = 1,
        num_ws: int = 1,
        silent_wandb: bool = False,
    ):
        """
        Initialize the Neural Tester.

        :param sut: The system under test.
        :param manipulator: The manipulator object.
        :param optimizer: The optimizer object.
        :param objectives: The objectives list.
        :param config: The experiment config.
        :param frontier_pairs: Whether the frontier pairs should be searched for.
        :param num_w0: The number of primary seeds.
        :param num_ws: The number of target seeds.
        :param silent_wandb: Whether to silence wandb.
        """

        self._sut = sut
        self._manipulator = manipulator
        self._optimizer = optimizer
        self._objectives = objectives

        self._num_w0 = num_w0
        self._num_ws = num_ws

        self._config = config
        self._softmax = torch.nn.Softmax(dim=1)  # TODO: should be refractored probably

        self._df = DefaultDF(pairs=frontier_pairs)
        self._silent = silent_wandb

    def test(self, validity_domain: bool = False) -> None:
        """
        Testing the predictor for its decision boundary using a set of (test!) Inputs.

        :param validity_domain: Whether the validity domain should be tested for.
        """
        spc, nc = self._config.samples_per_class, self._config.classes

        logging.info(
            f"Start testing. Number of classes: {nc}, iterations per class: {spc}, total iterations: {nc*spc}\n"
        )
        exp_start = datetime.now()
        for class_idx, sample_id in product(range(nc), range(spc)):
            self._init_wandb(exp_start, class_idx, self._silent)  # Initialize Wandb run for logging

            w0_tensors, w0_images, w0_ys, w0_trials = self._generate_seeds(self._num_w0, class_idx)

            """
            Now we select primary and secondary predictions for further style mixing.
            Note this can be extended to any n predictions, but for this approach we limited it to 2.
            Additionally this can be generalized to N w0 vectors, but now we only consider one.
            """
            _, second, *_ = torch.argsort(w0_ys[0], descending=True)[0]
            self._img_rgb = w0_images[0]

            wn_tensors, wn_images, wn_ys, wn_trials = (
                self._generate_noise(self._num_ws)
                if validity_domain
                else self._generate_seeds(self._num_ws, second)
            )

            self._maybe_log({"base_image": wandb.Image(self._img_rgb, caption="Base Image")})
            self._maybe_summary("w0_trials", wn_trials)
            self._maybe_summary("wn_trials", wn_trials)

            """
            Note that the w0s and ws' do not have to share a label, but for this implementation we do not control the labels separately.
            """
            # To save compute we parse tha cached tensors of w vectors as we generated them already for getting the initial prediction.
            w0c = [
                MixCandidate(label=class_idx, is_w0=True, w_tensor=tensor) for tensor in w0_tensors
            ]
            wsc = [MixCandidate(label=second.item(), w_tensor=tensor) for tensor in wn_tensors]
            candidates = CandidateList(*w0c, *wsc)

            # Now we run a search-based optimization strategy to find a good boundary candidate.
            logging.info(f"Running Search-Algorithm for {self._config.generations} generations.")
            for _ in range(self._config.generations):
                # We define the inner loop with its parameters.
                images, fitness, preds = self._inner_loop(candidates, class_idx, second)
                # Assign fitness to current population and additional data (in our case images).
                self._optimizer.assign_fitness(fitness, images, preds.tolist())
                self._optimizer.new_population()
            # Evaluate the last generation.
            images, fitness, preds = self._inner_loop(candidates, class_idx, second)
            self._optimizer.assign_fitness(fitness, images, preds.tolist())

            logging.info(
                f"\tBest candidate(s) have a fitness of: {', '.join([str(c.fitness) for c in self._optimizer.best_candidates])}"
            )
            self._maybe_summary("expected_boundary", second.item())
            wnb_results = {
                "best_candidates": wandb.Table(
                    columns=[metric.name for metric in self._objectives]
                    + [f"Genome_{i}" for i in range(self._optimizer.n_var)]
                    + ["Image"]
                    + [f"Conf_{i}" for i in range(self._config.classes)],
                    data=[
                        [
                            *c.fitness,
                            *c.solution,
                            wandb.Image(c.data[0]),
                            *c.data[1],
                        ]
                        for c in self._optimizer.best_candidates
                    ],
                ),
            }
            self._maybe_log(wnb_results)

            Xp, yp = self._optimizer.best_candidates[0].data
            results = [
                self._img_rgb.tolist(),
                w0_ys[0].tolist(),
                Xp.tolist(),
                yp,
                datetime.now() - exp_start,
            ]
            self._df.append_row(results)
            self._optimizer.reset()  # Reset the learner for new candidate.
            logging.info("\tReset learner!")

        if self._config.save_to is not None:
            self._df.to_csv(f"{self._config.save_to}.csv", index=False)

    def _inner_loop(
        self,
        candidates: CandidateList,
        c1: int,
        c2: int,
    ) -> tuple[list[Tensor], tuple[NDArray, ...], Tensor]:
        """
        The inner loop for the learner.

        :param candidates: The mixing candidates to be used.
        :param c1: The base class label.
        :param c2: The second most likely label.
        :returns: The images generated, and the corresponding fitness and the softmax predictions.
        """
        # Get the initial population of style mixing conditions and weights
        sm_cond_arr, sm_weights_arr = self._optimizer.get_x_current()
        assert (
            0 <= sm_cond_arr.max() < self._num_ws
        ), f"Error: StyleMixing Conditions reference indices of {sm_cond_arr.max()}, but we only have {self._num_ws} elements."

        images = []
        for sm_cond, sm_weights in zip(sm_cond_arr, sm_weights_arr):
            mixed_image = self._manipulator.manipulate(
                candidates=candidates,
                cond=sm_cond,
                weights=sm_weights,
                random_seed=self._get_time_seed(),
            )
            images.append(mixed_image)
        # Convert images to RGB if they are grayscale
        images = [self._assure_rgb(img) for img in images]

        """We predict the label from the mixed images."""
        predictions: Tensor = self._sut(torch.stack(images))
        predictions_softmax = self._softmax(predictions)
        fitness = tuple(
            [
                np.array(
                    [
                        criterion.evaluate(
                            default_args=CriteriaArguments(
                                i1=self._img_rgb,
                                i2=Xp,
                                c1=c1,
                                c2=c2,
                                yp=yp,
                            )
                        )
                        for Xp, yp in zip(images, predictions_softmax)
                    ]
                )
                for criterion in self._objectives
            ]
        )

        # Logging Operations
        results = {}
        # Log statistics for each objective function seperatly.
        for metric, obj in zip(self._objectives, fitness):
            results |= {
                f"min_{metric.name}": obj.min(),
                f"max_{metric.name}": obj.max(),
                f"mean_{metric.name}": obj.mean(),
                f"std_{metric.name}": obj.std(),
            }
        self._maybe_log(results)

        return images, fitness, predictions_softmax

    @staticmethod
    def _maybe_log(results: dict) -> None:
        """
        Logs to Wandb if initialized.

        :param results: The results to log.
        """
        try:
            wandb.log(results)
        except wandb.errors.Error as e:
            logging.error(e)
            pass

    @staticmethod
    def _maybe_summary(field: str, summary: Any) -> None:
        """
        Add elements to wandb Summary if initialized.

        :param field: The field to add an element to.
        :param summary: The element to add.
        """
        try:
            wandb.summary[field] = summary
        except wandb.errors.Error as e:
            logging.error(e)
            pass

    def _generate_seeds(
        self, amount: int, cls: int
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], int]:
        """
        Generate seeds for a specific class.

        :param amount: The amount of seeds to be generated.
        :param cls: The class to be generated.
        :returns: The w vectors generated, the corresponding images, confidence values and the amount of trials needed.
        """
        ws: list[Tensor] = []
        imgs: list[Tensor] = []
        y_hats: list[Tensor] = []

        logging.info(f"Generate seed(s) for class: {cls}.")
        # For logging purposes to see how many samples we need to find valid seed.
        trials = 0
        while len(ws) < amount:
            trials += 1
            # We generate w latent vector.
            w = self._manipulator.get_w(self._get_time_seed(), cls)
            # We generate and transform the image to RGB if it is in Grayscale.
            img = self._manipulator.get_image(w)
            img = self._assure_rgb(img)
            y_hat = self._sut(img.unsqueeze(0))

            # We are only interested in candidate if the prediction matches the label
            if y_hat.argmax() == cls:
                ws.append(w)
                imgs.append(img)
                y_hats.append(y_hat)
        logging.info(f"\tFound {amount} valid seed(s) after: {trials} iterations.")
        return ws, imgs, y_hats, trials

    def _generate_noise(self, amount: int) -> tuple[list[Tensor], list[Tensor], list[Tensor], int]:
        """
        Generate noise.

        :param amount: The amount of seeds to be generated.
        :returns: The w vectors generated, the corresponding images, confidence values and the amount of trials needed.
        """
        logging.info("Generate noise seeds.")
        # For logging purposes to see how many samples we need to find valid seed.
        w: Tensor = self._manipulator.get_w(self._get_time_seed(), 0)
        ws = [torch.randn(w.size(), device=w.device) for _ in range(amount)]
        imgs = [self._assure_rgb(self._manipulator.get_image(w)) for w in ws]
        y_hats = [self._sut(img.unsqueeze(0)) for img in imgs]

        logging.info(f"\tFound {amount} valid seed(s).")
        return ws, imgs, y_hats, 0

    def _init_wandb(self, exp_start: datetime, class_idx: int, silent: bool) -> None:
        """
        Initialize Wandb Run for logging

        :param exp_start: The start of the experiment (for grouping purposes).
        :param class_idx: The class index to search boundary candidates for.
        :param silent: Whether wandb should be silenced.
        """
        try:
            wandb.init(
                project="NeuralStyleSearch",
                config={
                    "num_gen": self._config.generations,
                    "num_w0s": self._num_w0,
                    "num_wns": self._num_ws,
                    "pop_size": self._optimizer._x_current.shape[0],
                    "experiment_start": exp_start,
                    "label": class_idx,
                    "learner_type": self._optimizer.learner_type,
                },
                settings=wandb.Settings(silent=silent),
            )
        except UsageError as e:
            logging.error(f"Raised error {e}, \n continuing...")
            pass

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
        Assure that image is or can be converted to RGB.

        :param image: The image to be converted.
        :returns: The converted image (3 x H x W).
        :raises ValueError: If the image shape is not recognized.
        """
        # We check if the input has a channel dimension.
        channel = None if len(image.shape) == 2 else len(image.shape) - 3
        # If we don`t have channels we add a dimension.
        image = image.unsqueeze(0) if channel is None else image

        rep_mask = [1] * len(image.shape)  # A repetition mask for channel extrusions
        if image.shape[channel] == 1:
            # If we only have one channel we repeat it 3 times to make it rgb.
            rep_mask[channel] = 3
            return image.repeat(*rep_mask)
        elif image.shape[channel] == 3:
            return image
        else:
            raise ValueError(f"Unknown image shape. {image.shape}")
