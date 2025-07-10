import os
import torch
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from utils import rank_gradient_info, perturbate_s_latents, convert_to_serializable
from src_.backpropagation import (generate_image_with_s_latents,
                                  backpropagation_gradients_s_space, smoothgrad_s_space)
from src_.occlusion import occlusions_s_space


class ManipulatorSSpace:
    """
    A class to automatically select and manipulate the channels in style space of the generator to achieve confidence
    reduction of the target class and generate perturbed images for analysis.
    """
    def __init__(self, generator, classifier, segmenter, preprocess_fn, target_logit, save_dir, device=None):
        """
        Initialize the ConfidenceAdjuster.

        Args:
            generator: Pre-trained generator for image synthesis.
            classifier: Pre-trained classifier for evaluating confidence.
            segmenter: Pre-trained segmentation model for image segmentation.
            preprocess_fn: Preprocessing function for classifier inputs.
            target_logit: Target logit index.
            save_dir: Directory to save results.
            device: Torch device ('cuda' or 'cpu').
        """
        self.generator = generator
        self.classifier = classifier
        self.preprocess_fn = preprocess_fn
        self.target_logit = target_logit

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.segmenter = segmenter
        self.save_dir = save_dir

    def compare_perturbed(self, s_gradients, layer_name, rank_data, prediction_target, extent_factor=40, top_n=0):
        """
        Perturb one image in s-space with the given layer and top-n important channels, generate the perturbed image,
        and calculate the confidence drop.

        Args:
            s_gradients: Gradients in s-space.
            layer_name: Name of the layer to perturb.
            rank_data: Ranked gradient data for a specific layer.
            prediction_target: Target prediction value for comparison.
            extent_factor: Scale of perturbation to apply.
            top_n: Index of the channel to perturb based on importance ranking.

        Returns:
            tuple: (confidence_drop, img_perturbed, adjusted_confidence)
                - confidence_drop: The reduction in confidence due to the perturbation.
                - img_perturbed: The perturbed image.
                - adjusted_confidence: The new confidence after perturbation.
        """
        location = rank_data['ranked_indices'][top_n]
        gradient_single = rank_data['gradients'][top_n]

        # Apply perturbation
        extent = np.sign(prediction_target) * (-extent_factor) * np.sign(gradient_single)
        gradient_perturbed = perturbate_s_latents(s_gradients, layer_name, location, extent)

        # Generate the perturbed image
        img_perturbed_tensor = generate_image_with_s_latents(
            synthesis_net=self.generator.synthesis,
            s_latents=gradient_perturbed
        )
        img_perturbed = img_perturbed_tensor.cpu().detach().numpy()[0].transpose(1, 2, 0)

        # Get the classifier confidence for the perturbed image
        prediction_perturbed = self.classifier(self.preprocess_fn(img_perturbed_tensor.squeeze(0)).unsqueeze(0)).squeeze(0)
        adjusted_confidence = prediction_perturbed[self.target_logit].cpu().detach().numpy()

        confidence_drop = np.sign(prediction_target) * (prediction_target - adjusted_confidence)
        return confidence_drop, img_perturbed, adjusted_confidence

    def bisection_factor_adjustment(self, current_factor, initial_confidence, current_confidence,tolerance, s_gradients, layer_name,
                                    rank_data, prediction_target, top_n, max_iterations=10):
        """
        Dynamically adjust the extent_factor using the bisection method to achieve misclassification
        with the adjusted confidence closest to 0.

        Args:
            current_factor (float): Initial upper bound for factor adjustment.
            initial_confidence (float): Confidence value of the original (unperturbed) image.
            current_confidence (float): Current confidence after perturbation.
            tolerance (float): Precision tolerance for bisection convergence.
            s_gradients: Gradients in s-space.
            layer_name: Name of the layer to perturb.
            rank_data: Ranked gradient data for a specific layer.
            prediction_target: Target prediction value for comparison.
            top_n: Index of the channel to perturb based on importance ranking.

        Returns:
            tuple: (best_factor, best_adjusted_confidence)
                - best_factor: The optimal extent_factor for misclassification close to 0 confidence.
                - best_adjusted_confidence: The adjusted confidence for the optimal factor.
        """
        low = 1e-5  # Smallest extent_factor
        high = current_factor  # Upper bound
        mid = (low + high) / 2.0  # Initial midpoint

        best_factor = current_factor
        best_adjusted_confidence = current_confidence # Start with infinity for comparison
        best_img_perturbed = None
        best_confidence_drop = None

        ierations = 0
        # Iteratively perform bisection search
        while high - low > tolerance and ierations < max_iterations:
            confidence_drop, img_perturbed, adjusted_confidence = self.compare_perturbed(
                s_gradients, layer_name, rank_data, prediction_target, extent_factor=mid, top_n=top_n
            )
            print(f"low: {low:.2f}, high: {high:.2f}, mid: {mid:.2f}, adjusted_confidence: {adjusted_confidence:.2f}, initial_confidence: {initial_confidence:.2f}, best_adjusted_confidence: {best_adjusted_confidence:.2f}")

            if  adjusted_confidence * initial_confidence < 0:
                high = mid
                if abs(adjusted_confidence) < abs(best_adjusted_confidence):
                    best_factor = mid
                    best_adjusted_confidence = adjusted_confidence
                    best_img_perturbed = img_perturbed
                    best_confidence_drop = confidence_drop
            else:
                low = mid  # Narrow search to larger factors

            mid = (low + high) / 2.0  # Update midpoint for next iteration
            ierations += 1
        return best_factor, best_adjusted_confidence, best_img_perturbed, best_confidence_drop

    def plot_comparison(self, original_img, img_perturbed, prediction_delta, save_path,
                        predicate_target, predicate_perturbed_target):
        """
        Plot the original, perturbed, and difference images, and save the results to a file.

        Args:
            original_img: Original image as a NumPy array.
            img_perturbed: Perturbed image as a NumPy array.
            prediction_delta: Difference in prediction confidence after perturbation.
            save_path: File path to save the plot.
        """
        diff_img = np.abs(img_perturbed - original_img)
        # TODO: The segmentation algorithm to find the affected region does not work well
        # region_means = segmenter.compute_region_means(diff_img, mask)
        # affect_region = segmenter.part_labels[max(region_means, key=region_means.get)]
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
        # plt.show()
        plt.savefig(save_path)
        #plt.close()

    @staticmethod
    def save_image_np(img_np, img_dir):
        #assert max(img_np) <= 1 and min(img_np) >= 0, "image not in range [0, 1]"
        Image.fromarray((img_np * 255).astype(np.uint8)).save(img_dir)

    def handle_one_seed(self,
                        torch_seed,
                        top_channels=1, # not used with adaptive strategy
                        default_extent_factor=10,
                        tolerance_of_extent_bisection=1,
                        confidence_drop_threshold=0.3,
                        oracle="confidence_drop", # "misclassification"
                        specified_layer=None,
                        skip_rgb_layer=True,
                        truncation_psi = 0.7,
                        config = "gradient"):
        """
        Handles the generation, perturbation, and analysis of synthetic image seeds by manipulating latent space attributes
        in a generator and evaluating the effects on a classifier prediction.

        The method takes a random seed to generate an image using a pre-defined generator model. It computes perturbations
        in the latent space to measure the impact on classifier prediction confidence. The function iterates over layers
        and channels based on computed gradients and ranks their significance to identify critical factors affecting
        classifier confidence. Perturbations are applied using various configurations, adjusted dynamically through
        bisection technique if necessary, and the effects are visualized and saved for analysis.

        Attributes
        ----------
        torch_seed : int
            Random seed for generating synthetic data.
        top_channels : int, optional
            Number of top-ranked gradient channels considered for perturbation (default is 1).
        default_extent_factor : int, optional
            Initial perturbation extent factor (default is 10).
        tolerance_of_extent_bisection : int, optional
            Tolerance for bisection adjustment of extent factor (default is 1).
        confidence_drop_threshold : float, optional
            Threshold of confidence drop below which changes are ignored (default is 0.3).
        oracle : str, optional
            Defines target mode: "confidence_drop" or "misclassification" (default is "confidence_drop").
        specified_layer : str or None, optional
            Specifies which layer to perturb, if any (default is None).
        skip_rgb_layer : bool, optional
            Whether to skip layers containing "rgb" in their name (default is True).
        truncation_psi : float, optional
            Controls truncation of latent vectors (default is 0.7).
        config : str, optional
            Method for generating perturbations: "gradient", "occlusion", or "smoothgrad" (default is "gradient").

        Parameters
        ----------
        torch_seed : int
            Random seed used for reproducible synthetic data generation.
        top_channels : int, optional
            Limits the number of gradient-ranked channels to analyze (default is 1).
        default_extent_factor : int, optional
            Initial magnitude used for latent space perturbations (default is 10).
        tolerance_of_extent_bisection : int, optional
            Precision tolerance for adjustment of perturbation magnitudes via bisection (default is 1).
        confidence_drop_threshold : float, optional
            Minimum proportion of confidence drop to qualify a perturbation as significant (default is 0.3).
        oracle : str, optional
            Performance target mode: validation via confidence drop or success of misclassification
            (default is "confidence_drop").
        specified_layer : str or None, optional
            Restricts the perturbation analysis to only a specific generator layer (default is None).
        skip_rgb_layer : bool, optional
            Avoids analyzing generator layers responsible for RGB translation when True (default is True).
        truncation_psi : float, optional
            Adjusts latent vector optimization for synthetic generation (default is 0.7).
        config : str, optional
            Perturbation analysis configuration, options are "gradient", "occlusion", or "smoothgrad" (default is "gradient").

        Raises
        ------
        FileNotFoundError
            Raised when the output directory for saving results cannot be created.
        ValueError
            Raised when unsupported `config` options are provided.
        KeyError
            Raised when the generator's latent space objects are missing or misspecified.

        Returns
        -------
        None
            No explicit return value; results are saved as image and JSON artifacts in the specified directory.
        """

        assert oracle in ["confidence_drop", "misclassification"], "oracle must be either 'confidence_drop' or 'misclassification'"
        assert config in ["gradient", "occlusion", "smoothgrad"], "config must be either 'gradient', 'occlusion', or 'smoothgrad'"

        torch.manual_seed(torch_seed)
        # generate one random seed from z latent space
        z = torch.randn([1, self.generator.z_dim], device=self.device)

        if self.generator.c_dim != 0:
            label = torch.zeros([1, self.generator.c_dim], device=self.device)
            target_class = 207  #  SELECT CLASS, not for facial task!
            label[:, target_class] = 1
        else:
            label = None

        img_tensor = self.generator(z, label,
                               truncation_psi=truncation_psi, noise_mode='const')# ensure deterministic and reproducible
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2  # normalize to [0, 1] ss

        img = img_tensor.cpu().detach().numpy()[0]
        img = img.transpose(1, 2, 0)

        # get prediction of the original image
        prediction = self.classifier(self.preprocess_fn(img_tensor.squeeze(0)).unsqueeze(0)).squeeze(0)
        prediction_target = prediction[self.target_logit].cpu().detach().numpy()

        # get mask
        if self.segmenter is not None:
            mask = self.segmenter.predict(img)
        else:
            mask = np.zeros_like(img) # dummy mask

        # calculate w vector
        w = self.generator.mapping(z, c=label)
        if truncation_psi != 1:
            w = self.generator.mapping.w_avg + (w - self.generator.mapping.w_avg) * truncation_psi
        if config == "gradient":
            s_gradients, classifier_output, img_tensor = backpropagation_gradients_s_space(
                synthesis_net = self.generator.synthesis,
                classifier = self.classifier,
                preprocess= self.preprocess_fn,
                w_latents= w,
                target_class= self.target_logit, # glasses
            )
        elif config == "occlusion":
            print("computing occlusion gradients... this may take a while")
            s_gradients, classifier_output, img_tensor = occlusions_s_space(
                synthesis_net = self.generator.synthesis,
                classifier = self.classifier,
                preprocess= self.preprocess_fn,
                w_latents= w,
                target_class=self.target_logit,
                epsilon=0.1, #1e-6,
                device = self.device
            )
        elif config == "smoothgrad":
            s_gradients, classifier_output, img_tensor = smoothgrad_s_space(
                synthesis_net = self.generator.synthesis,
                classifier = self.classifier,
                preprocess= self.preprocess_fn,
                w_latents= w,
                target_class=self.target_logit,
                n_samples=10,
                noise_scale=0.2,
                device=self.device
            )
        # rank gradient to get the most important channel of each layer
        top_channels = self.get_adaptive_top_channels("_")
        ranked_gradient_info = rank_gradient_info(s_gradients, top=top_channels)

        if specified_layer is None:
            for i, (layer_name, rank_data) in enumerate(ranked_gradient_info.items()):
                if "rgb" in layer_name and skip_rgb_layer: # skip rgb layer
                    continue
                # get adaptive top channels
                adaptive_top_channels = self.get_adaptive_top_channels(layer_name) # top_channels
                for top_n in range(adaptive_top_channels):
                    confidence_drop, img_perturbed, prediction_perturbed_target = self.compare_perturbed(
                        s_gradients, layer_name, rank_data, prediction_target, extent_factor=default_extent_factor, top_n=top_n)
                    save_flag = False
                    # RQ1 we don't need threshold
                    if oracle == "confidence_drop" and confidence_drop > confidence_drop_threshold * np.abs(prediction_target):
                        save_flag = True
                        print(f"Layer: {layer_name}, Ranking: {top_n} confidence_drop {confidence_drop:.2f} , "
                              f"confidence {prediction_target:.2f} {prediction_perturbed_target:.2f}")
                        if self.segmenter is not None:
                            mask_perturbed = self.segmenter.predict(img_perturbed)
                            seg_result = self.segmenter.detect_changes(img, img_perturbed, mask, mask_perturbed)
                        else:
                            seg_result = None

                    elif oracle == "misclassification" and prediction_target * prediction_perturbed_target < 0:
                        (best_factor, best_adjusted_confidence, best_img_perturbed,
                         best_confidence_drop) = self.bisection_factor_adjustment(default_extent_factor,
                                                                                  prediction_target,
                                                                                  prediction_perturbed_target,
                                                                                  tolerance=tolerance_of_extent_bisection,
                                                                                  s_gradients=s_gradients,
                                                                                  layer_name=layer_name,
                                                                                  rank_data=rank_data,
                                                                                  prediction_target=prediction_target,
                                                                                  top_n=top_n
                                                                                  )
                        if best_img_perturbed is not None:
                            default_extent_factor = best_factor
                            prediction_perturbed_target = best_adjusted_confidence
                            img_perturbed = best_img_perturbed
                            confidence_drop = best_confidence_drop
                        print(f"misclassification!! adjusted factor {default_extent_factor}")
                        print(f"Layer: {layer_name}, Ranking: {top_n} confidence_drop {confidence_drop:.2f} , "
                              f"confidence {prediction_target:.2f} {prediction_perturbed_target:.2f}")
                        # hill climbing to find the best factor
                        # print("Using constrained optimization to ensure prediction flip")
                        (best_factor, best_adjusted_confidence, best_img_perturbed,
                         best_confidence_drop) = self.constrained_hill_climbing_factor_adjustment(
                            default_extent_factor, prediction_target, prediction_perturbed_target,
                            s_gradients, layer_name, rank_data, prediction_target, top_n)

                        # check if prediction flip is guaranteed
                        if (best_img_perturbed is not None and
                            best_adjusted_confidence * prediction_target < 0):  #
                            default_extent_factor = best_factor
                            prediction_perturbed_target = best_adjusted_confidence
                            img_perturbed = best_img_perturbed
                            confidence_drop = best_confidence_drop
                            print(f"✓ Prediction flip guaranteed! factor={default_extent_factor:.3f}")

                            save_flag = True
                            if self.segmenter is not None:
                                mask_perturbed = self.segmenter.predict(img_perturbed)
                                seg_result = self.segmenter.detect_changes(img, img_perturbed, mask, mask_perturbed)
                            else:
                                seg_result = None
                        else:
                            print("✗ Failed to maintain prediction flip, using fallback")
                    if save_flag:
                        seed_dir = os.path.join(self.save_dir, f"{torch_seed}")
                        if not os.path.exists(seed_dir):
                            os.makedirs(seed_dir, exist_ok=True)
                            self.save_image_np(img, os.path.join(seed_dir, "1_original.png"))

                        file_name = f"{layer_name.replace('.', '_')}_{rank_data['ranked_indices'][top_n]}"

                        img_perturbed_path = os.path.join(seed_dir, file_name + ".png")
                        self.save_image_np(img_perturbed, img_perturbed_path)

                        json_path = os.path.join(seed_dir, file_name + ".json")

                        info = {
                            'seed': torch_seed,
                            "oracle": oracle,
                            "confidence_drop": confidence_drop,
                            "config": config,
                            "prediction_target": prediction_target,
                            "prediction_perturbed_target": prediction_perturbed_target,
                            "extent_factor": default_extent_factor,
                            "top_n": top_n,
                            "layer_name": layer_name,
                            "channel_id": rank_data['ranked_indices'][top_n],
                            "gradient": rank_data['gradients'][top_n],
                            "img_path": img_perturbed_path,
                            "significant_changes": seg_result[:5] if seg_result is not None else None
                        }

                        with open(json_path, 'w') as f:
                            json.dump(convert_to_serializable(info), f, indent=4)
                        self.plot_comparison(img, img_perturbed,
                                             confidence_drop,
                                             img_perturbed_path.replace(".png", "_comparison.png"),
                                             prediction_target, prediction_perturbed_target)
        else:
            # The case when we only want to perturb a specific layer
            print("Not complete yet!!! ask XC to finish this")
            pass

    def constrained_hill_climbing_factor_adjustment(self, current_factor, initial_confidence, current_confidence,
                                                    s_gradients, layer_name, rank_data, prediction_target, top_n,
                                                    step_size=0.1, max_iterations=20, patience=3):
        """
        Use constrained hill climbing to adjust extent_factor ensuring prediction flip is maintained.

        Args:
            current_factor (float): Initial factor value
            initial_confidence (float): Original confidence before perturbation
            current_confidence (float): Current confidence after perturbation
            s_gradients: Gradients in s-space
            layer_name: Name of the layer to perturb
            rank_data: Ranked gradient data
            prediction_target: Target prediction value
            top_n: Channel index
            step_size (float): Initial step size for hill climbing
            max_iterations (int): Maximum number of iterations
            patience (int): Number of iterations to wait before reducing step size

        Returns:
            tuple: (best_factor, best_adjusted_confidence, best_img_perturbed, best_confidence_drop)
        """

        def is_misclassified(confidence, initial_conf):
            """Check if prediction is flipped (signs are different)"""
            return confidence * initial_conf < 0

        def evaluate_factor(factor):
            """Evaluate a factor and return results only if misclassification occurs"""
            confidence_drop, img_perturbed, adjusted_confidence = self.compare_perturbed(
                s_gradients, layer_name, rank_data, prediction_target,
                extent_factor=factor, top_n=top_n
            )

            if is_misclassified(adjusted_confidence, initial_confidence):
                return True, confidence_drop, img_perturbed, adjusted_confidence
            else:
                return False, None, None, None

        # Verify initial condition
        if not is_misclassified(current_confidence, initial_confidence):
            print("Warning: Initial condition does not satisfy misclassification!")
            return current_factor, current_confidence, None, None

        best_factor = current_factor
        best_adjusted_confidence = current_confidence
        best_img_perturbed = None
        best_confidence_drop = None

        factor = current_factor
        no_improvement_count = 0

        for iteration in range(max_iterations):
            # Try both directions with constraint checking
            candidates = [
                max(1e-5, factor - step_size),  # Decrease factor
                factor + step_size  # Increase factor
            ]

            improved = False
            for candidate_factor in candidates:
                is_valid, confidence_drop, img_perturbed, adjusted_confidence = evaluate_factor(candidate_factor)

                if is_valid and abs(adjusted_confidence) < abs(best_adjusted_confidence):
                    best_factor = candidate_factor
                    best_adjusted_confidence = adjusted_confidence
                    best_img_perturbed = img_perturbed
                    best_confidence_drop = confidence_drop
                    factor = candidate_factor
                    improved = True
                    no_improvement_count = 0
                    print(f"Constrained hill climbing iteration {iteration}: factor={factor:.3f}, "
                          f"confidence={adjusted_confidence:.3f} (flip maintained)")
                    break

            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    step_size *= 0.5  # Reduce step size
                    no_improvement_count = 0
                    print(f"Reducing step size to {step_size:.3f}")

                    if step_size < 1e-3:  # Stop if step size becomes too small
                        break

        # Final verification
        if not is_misclassified(best_adjusted_confidence, initial_confidence):
            print("Error: Final result does not maintain prediction flip!")
            return None, None, None, None

        return best_factor, best_adjusted_confidence, best_img_perturbed, best_confidence_drop

    @staticmethod
    def get_adaptive_top_channels(layer_name):
        """
        Get adaptive top_channels based on layer characteristics.

        Args:
            layer_name (str): Name of the layer

        Returns:
            int: Number of top channels to use for this layer
        """
        # define different layer strategies
        layer_strategies = {
            'early_layers': {
                'patterns': ['b4.', 'b8.', 'b16.'],
                'top_channels': 15
            },

            'middle_layers': {
                'patterns': ['b32.', 'b64.', 'b128.', 'b256.'],
                'top_channels': 15
            },

            'late_layers': {
                'patterns': ['b512.', 'b1024.'],
                'top_channels': 5
            }
        }

        # check if the layer name contains any of the specified patterns
        for strategy_name, strategy in layer_strategies.items():
            for pattern in strategy['patterns']:
                if pattern in layer_name:
                    # print(f"Layer {layer_name} matched {strategy_name} (pattern: {pattern}): using {strategy['top_channels']} top channels")
                    return strategy['top_channels']

        # if no match found, use the default, return max channels
        if layer_name != "_":
            # print anomaly layer when layer name is not "_" the placeholder
            print(f"Layer {layer_name} using max top channels")
        return max(strategy['top_channels'] for strategy in layer_strategies.values())

