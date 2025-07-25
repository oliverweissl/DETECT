import os
import numpy as np
import torch
import json
from src.utils import rank_gradient_info, convert_to_serializable, perturbate_s_latents
from src.s_manipulator import BaseManipulatorSSpace
from src.backpropagation import generate_image_with_s_latents

class BinaryManipulatorSSpace(BaseManipulatorSSpace):
    """Manipulator for binary classification tasks."""

    def __init__(self, generator, classifier, segmenter, preprocess_fn, target_logit, save_dir, confidence_drop_threshold = 0.4,device=None):
        super().__init__(generator, classifier, segmenter, preprocess_fn, save_dir, confidence_drop_threshold,device)
        self.target_logit = target_logit

    def _get_target_value(self, prediction):
        """
        Extract the target prediction value from the classifier's output.
        """
        return prediction[self.target_logit].cpu().detach().numpy()

    def _get_target_class(self):
        return self.target_logit

    def _evaluate_oracle(self, original_pred, perturbed_pred, oracle):
        if oracle == "confidence_drop":
            confidence_drop = self._get_confidence_drop(original_pred, perturbed_pred)
            return confidence_drop > self.confidence_drop_threshold * np.abs(original_pred)
        elif oracle == "misclassification":
            return original_pred * perturbed_pred < 0
        return False

    def _get_confidence_drop(self, initial_confidence, adjusted_confidence):
        return np.sign(initial_confidence) * (initial_confidence - adjusted_confidence)

    def compare_perturbed(self, s_gradients, layer_name, rank_data, prediction_target, extent_factor=40, top_n=0):
        """
        Perturb one image in s-space and calculate the effect.
        Common implementation across all manipulator types.
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

        # Get the classifier prediction for the perturbed image
        prediction_perturbed = self.classifier(
            self.preprocess_fn(img_perturbed_tensor.squeeze(0)).unsqueeze(0)).squeeze(0)
        adjusted_confidence = self._get_target_value(prediction_perturbed)

        #confidence_drop = np.sign(prediction_target) * (prediction_target - adjusted_confidence)
        confidence_drop = self._get_confidence_drop(prediction_target, adjusted_confidence)
        return confidence_drop, img_perturbed, adjusted_confidence


    def bisection_factor_adjustment(self, current_factor, initial_confidence, current_confidence,
                                    tolerance, s_gradients, layer_name, rank_data, prediction_target,
                                    top_n, max_iterations=10):
        """
        Dynamically adjust the extent_factor using the bisection method to achieve misclassification
        with the adjusted confidence closest to 0.
        """
        low = 1e-5  # Smallest extent_factor
        high = current_factor  # Upper bound
        mid = (low + high) / 2.0  # Initial midpoint

        best_factor = current_factor
        best_adjusted_confidence = current_confidence
        best_img_perturbed = None
        best_confidence_drop = None

        iterations = 0
        # Iteratively perform bisection search
        while high - low > tolerance and iterations < max_iterations:
            confidence_drop, img_perturbed, adjusted_confidence = self.compare_perturbed(
                s_gradients, layer_name, rank_data, prediction_target, extent_factor=mid, top_n=top_n
            )
            print(f"low: {low:.2f}, high: {high:.2f}, mid: {mid:.2f}, adjusted_confidence: {adjusted_confidence:.2f}, "
                  f"initial_confidence: {initial_confidence:.2f}, best_adjusted_confidence: {best_adjusted_confidence:.2f}")

            if adjusted_confidence * initial_confidence < 0:
                high = mid
                if abs(adjusted_confidence) < abs(best_adjusted_confidence):
                    best_factor = mid
                    best_adjusted_confidence = adjusted_confidence
                    best_img_perturbed = img_perturbed
                    best_confidence_drop = confidence_drop
            else:
                low = mid  # Narrow search to larger factors

            mid = (low + high) / 2.0  # Update midpoint for next iteration
            iterations += 1

        return best_factor, best_adjusted_confidence, best_img_perturbed, best_confidence_drop

    def constrained_hill_climbing_factor_adjustment(self, current_factor, initial_confidence, current_confidence,
                                                    s_gradients, layer_name, rank_data, prediction_target, top_n,
                                                    step_size=0.1, max_iterations=20, patience=3):
        """
        Use constrained hill climbing to adjust extent_factor ensuring prediction flip is maintained.
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

    def handle_one_seed(self,
                        torch_seed,
                        default_extent_factor=10,
                        tolerance_of_extent_bisection=1,
                        oracle="confidence_drop",  # "misclassification"
                        specified_layer=None,
                        skip_rgb_layer=True,
                        truncation_psi=0.7,
                        config="gradient"):
        """
        Handle the generation, perturbation, and analysis of synthetic image seeds for binary classification.

        This method generates a synthetic image from a random seed, computes perturbations in the style space
        to measure the impact on binary classifier prediction confidence, and saves results for analysis.
        """
        assert oracle in ["confidence_drop",
                          "misclassification"], "oracle must be either 'confidence_drop' or 'misclassification'"
        assert config in ["gradient", "occlusion",
                          "smoothgrad"], "config must be either 'gradient', 'occlusion', or 'smoothgrad'"

        torch.manual_seed(torch_seed)
        # Generate one random seed from z latent space
        z = torch.randn([1, self.generator.z_dim], device=self.device)

        label_sg = None

        img_tensor = self.generator(z, label_sg,
                                    truncation_psi=truncation_psi,
                                    noise_mode='const')  # ensure deterministic and reproducible
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2  # normalize to [0, 1]

        img = img_tensor.cpu().detach().numpy()[0]
        img = img.transpose(1, 2, 0)

        # Get prediction of the original image
        prediction = self.classifier(self.preprocess_fn(img_tensor.squeeze(0)).unsqueeze(0)).squeeze(0)
        prediction_target = self._get_target_value(prediction)

        # Get mask
        if self.segmenter is not None:
            mask = self.segmenter.predict(img)
        else:
            mask = np.zeros_like(img)  # dummy mask

        # Calculate w vector
        w = self.generator.mapping(z, c=label_sg)
        if truncation_psi != 1:
            w = self.generator.mapping.w_avg + (w - self.generator.mapping.w_avg) * truncation_psi

        # Compute gradients using specified method
        s_gradients, classifier_output, img_tensor  = self.compute_gradients(config, w)

        # Rank gradient to get the most important channel of each layer
        ranked_gradient_info = rank_gradient_info(s_gradients, top="adaptive", layer_name = "all")

        info = {
            'seed': torch_seed,
            "oracle": oracle,
            "config": config,
            "prediction_target": prediction_target,
        }

        if specified_layer is None:
            for i, (layer_name, rank_data) in enumerate(ranked_gradient_info.items()):
                if "rgb" in layer_name and skip_rgb_layer:  # skip rgb layer
                    continue

                # Get adaptive top channels
                # adaptive_top_channels = self.get_adaptive_top_channels(layer_name)

                for top_n in range(len(rank_data['ranked_indices'])):
                    confidence_drop, img_perturbed, prediction_perturbed_target = self.compare_perturbed(
                        s_gradients, layer_name, rank_data, prediction_target,
                        extent_factor=default_extent_factor, top_n=top_n)

                    save_flag = False

                    if oracle == "confidence_drop" and self._evaluate_oracle(
                            prediction_target, prediction_perturbed_target, oracle):
                        save_flag = True
                        print(f"Layer: {layer_name}, Ranking: {top_n} confidence_drop {confidence_drop:.2f}, "
                              f"confidence {prediction_target:.2f} {prediction_perturbed_target:.2f}")

                        if self.segmenter is not None:
                            mask_perturbed = self.segmenter.predict(img_perturbed)
                            seg_result = self.segmenter.detect_changes(img, img_perturbed, mask, mask_perturbed)
                        else:
                            seg_result = None

                    elif oracle == "misclassification" and self._evaluate_oracle(
                            prediction_target, prediction_perturbed_target, oracle):
                        (best_factor, best_adjusted_confidence, best_img_perturbed,
                         best_confidence_drop) = self.constrained_hill_climbing_factor_adjustment(
                            current_factor=default_extent_factor,
                            initial_confidence=prediction_target,
                            current_confidence=prediction_perturbed_target,
                            s_gradients=s_gradients,
                            layer_name=layer_name,
                            rank_data=rank_data,
                            prediction_target=prediction_target,
                            top_n=top_n,
                            step_size=5,
                        )

                        # Check if prediction flip is guaranteed
                        if (best_img_perturbed is not None and
                                best_adjusted_confidence * prediction_target < 0):
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

                        info.update({
                            "prediction_perturbed_target": prediction_perturbed_target,
                            "confidence_drop": confidence_drop,
                            "extent_factor": default_extent_factor,
                            "top_n": top_n,
                            "layer_name": layer_name,
                            "channel_id": rank_data['ranked_indices'][top_n],
                            "gradient": rank_data['gradients'][top_n],
                            "img_path": img_perturbed_path,
                            "significant_changes": seg_result[:5] if seg_result is not None else None
                        })

                        with open(json_path, 'w') as f:
                            json.dump(convert_to_serializable(info), f, indent=4)

                        self.plot_comparison(img, img_perturbed,
                                             confidence_drop,
                                             img_perturbed_path.replace(".png", "_comparison.png"),
                                             prediction_target, prediction_perturbed_target)
        else:
            # The case when we only want to perturb a specific layer (layer_name, channel_id)
            print("Specific layer perturbation not implemented yet! Please implement this feature.")


