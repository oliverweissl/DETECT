import os
from abc import ABC, abstractmethod
import torch
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

from src.backpropagation import (backpropagation_gradients_s_space, smoothgrad_s_space)
from src.occlusion import occlusions_s_space


class BaseManipulatorSSpace(ABC):
    """
    Abstract base class for style space manipulators across different classification tasks.

    This class provides common functionality for manipulating generator style spaces
    to achieve desired changes in classifier predictions.
    """

    def __init__(self, generator, classifier, segmenter, preprocess_fn, save_dir, confidence_drop_threshold, device=None):
        """
        Initialize the base manipulator.

        Args:
            generator: Pre-trained generator for image synthesis.
            classifier: Pre-trained classifier for evaluating confidence.
            segmenter: Pre-trained segmentation model for image segmentation.
            preprocess_fn: Preprocessing function for classifier inputs.
            save_dir: Directory to save results.
            device: Torch device ('cuda' or 'cpu').
        """
        self.generator = generator
        self.classifier = classifier
        self.preprocess_fn = preprocess_fn
        self.segmenter = segmenter
        self.save_dir = save_dir
        self.confidence_drop_threshold = confidence_drop_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    @abstractmethod
    def _get_target_value(self, prediction):
        """
        Extract target value from classifier prediction based on task type.

        Args:
            prediction: Classifier output tensor

        Returns:
            Target prediction value for manipulation
        """
        pass

    @abstractmethod
    def _evaluate_oracle(self, original_pred, perturbed_pred, oracle):
        """
        Evaluate if the perturbation achieved the desired outcome.

        Args:
            original_pred: Original prediction value
            perturbed_pred: Perturbed prediction value
            oracle: Success criteria ("confidence_drop", "misclassification", etc.)

        Returns:
            bool: Whether perturbation was successful
        """
        pass

    @abstractmethod
    def _get_confidence_drop(self, initial_confidence, adjusted_confidence):

        """
        Calculate the confidence drop between the original and perturbed predictions.

        Args:
            initial_confidence (float): Original prediction confidence.
            adjusted_confidence (float): Perturbed prediction confidence.

        Returns:
            float: Confidence drop between the original and perturbed predictions.
        """
        pass

    def plot_comparison(self, original_img, img_perturbed, prediction_delta, save_path,
                        predicate_target, predicate_perturbed_target):
        """
        Plot comparison between original and perturbed images.
        Common implementation across all manipulator types.
        """
        diff_img = np.abs(img_perturbed - original_img)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title(f"Original: {predicate_target:.2f}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(img_perturbed)
        plt.title(f"Perturbed: {predicate_perturbed_target:.2f}")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(diff_img)
        plt.title(f"Decline: {prediction_delta:.2f}")
        plt.axis("off")

        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def save_image_np(img_np, img_dir):
        """Save numpy image array as PNG file."""
        Image.fromarray((img_np * 255).astype(np.uint8)).save(img_dir)

    def compute_gradients(self, config, w_latents):
        """
        Compute gradients using different methods.
        Common implementation with task-specific target handling.
        """
        if config == "gradient":
            return backpropagation_gradients_s_space(
                synthesis_net=self.generator.synthesis,
                classifier=self.classifier,
                preprocess=self.preprocess_fn,
                w_latents=w_latents,
                target_class=self._get_target_class(),
            )
        elif config == "occlusion":
            print("Computing occlusion gradients... this may take a while")
            return occlusions_s_space(
                synthesis_net=self.generator.synthesis,
                classifier=self.classifier,
                preprocess=self.preprocess_fn,
                w_latents=w_latents,
                target_class=self._get_target_class(),
                epsilon=0.1,
                device=self.device
            )
        elif config == "smoothgrad":
            return smoothgrad_s_space(
                synthesis_net=self.generator.synthesis,
                classifier=self.classifier,
                preprocess=self.preprocess_fn,
                w_latents=w_latents,
                target_class=self._get_target_class(),
                n_samples=10,
                noise_scale=0.2,
                device=self.device
            )

    @abstractmethod
    def _get_target_class(self):
        """Get the target class index for gradient computation."""
        pass

    @abstractmethod
    def handle_one_seed(self, torch_seed, **kwargs):
        """
        Main method to handle seed processing.
        Must be implemented by each specific manipulator.
        """
        pass
