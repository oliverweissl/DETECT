import numpy as np
import torch
import os
import json
from src.utils import perturbate_s_latents, rank_gradient_info, convert_to_serializable

from src.s_manipulator import BaseManipulatorSSpace
from src.backpropagation import generate_image_with_s_latents

class MulticlassManipulatorSSpace(BaseManipulatorSSpace):
    """Manipulator for multiclass classification tasks."""

    def __init__(self, generator, classifier, segmenter, preprocess_fn,  save_dir,
                 class_dict, confidence_drop_threshold=0.3, device=None):
        super().__init__(generator, classifier, segmenter, preprocess_fn, save_dir, confidence_drop_threshold, device)
        self.target_logit = None
        self.class_dict = class_dict

    def _get_target_value(self, prediction):
        """
        Extract the target prediction value from the classifier's output.
        For multiclass, this is the confidence of the target class.
        """
        if isinstance(prediction, np.ndarray):
            predicted_class = np.argmax(prediction)
            highest_confidence = prediction[predicted_class]
            if self.target_logit is None:
                self._set_target_class(predicted_class)
            return prediction[self.target_logit], predicted_class, highest_confidence
        elif isinstance(prediction, torch.Tensor):
            print("_get_target_value",prediction.shape)
            result = self._get_target_value(prediction.squeeze(0).cpu().detach().numpy())
            return result
        else:
            raise ValueError(f"Unsupported prediction type: {type(prediction)}")

    def _set_target_class(self, target_class):
        self.target_logit = target_class

    def _reset_target_class(self):
        self.target_logit = None

    def _get_target_class(self):
        """Return the target class for gradient computation."""
        return self.target_logit

    def _get_confidence_drop(self, initial_confidence, adjusted_confidence):
        """Calculate the confidence drop between original and perturbed predictions."""
        initial_target = self._get_target_value(initial_confidence)[0]  if isinstance(initial_confidence, np.ndarray) else initial_confidence
        adjusted_target = self._get_target_value(adjusted_confidence)[0] if isinstance(adjusted_confidence, np.ndarray) else adjusted_confidence

        return initial_target - adjusted_target

    def _evaluate_oracle(self, original_pred, perturbed_pred, oracle):
        """
        Evaluate if the perturbation achieved the desired outcome based on the specified oracle.

        Args:
            original_pred: Original prediction value
            perturbed_pred: [Perturbed prediction target value, predicted class,]
            oracle: Success criteria ("confidence_drop" or "misclassification")
            predicted_class: Predicted class
        Returns:
            bool: Whether the perturbation was successful according to the oracle
        """
        perturbed_target = perturbed_pred[0]
        perturbed_class = perturbed_pred[1]
        if oracle == "confidence_drop":
            confidence_drop = self._get_confidence_drop(original_pred, perturbed_target)
            return confidence_drop > self.confidence_drop_threshold * np.abs(original_pred)
        elif oracle == "misclassification":
            # For multiclass, check if predicted class changed
            #perturbed_pred = perturbed_pred if isinstance(perturbed_pred, np.ndarray) else perturbed_pred.cpu().detach().numpy()
            #perturbed_class = np.argmax(perturbed_pred) if isinstance(perturbed_pred, np.ndarray) else perturbed_pred
            return self.target_logit != perturbed_class
        return False

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

            # For multiclass, we're checking if the predicted class has changed
            original_class = np.argmax(self.classifier(self.preprocess_fn(
                generate_image_with_s_latents(self.generator.synthesis, s_gradients).squeeze(0)
            ).unsqueeze(0)).squeeze(0).cpu().detach().numpy())

            perturbed_class = np.argmax(self.classifier(self.preprocess_fn(
                generate_image_with_s_latents(self.generator.synthesis,
                                              perturbate_s_latents(s_gradients, layer_name,
                                                                   rank_data['ranked_indices'][top_n],
                                                                   mid * -np.sign(rank_data['gradients'][top_n]))
                                              ).squeeze(0)
            ).unsqueeze(0)).squeeze(0).cpu().detach().numpy())

            class_changed = original_class != perturbed_class

            if class_changed:
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

    def constrained_hill_climbing_factor_adjustment(self, current_factor,
                                                    s_gradients, layer_name, rank_data, prediction_target, top_n,
                                                    original_predicted_class, step_size=0.2, max_iterations=20,
                                                    patience=3, verbose=False):
        """
        Use constrained hill climbing to adjust extent_factor ensuring prediction flip is maintained.

        Args:
            current_factor (float): Initial factor value
            #current_confidence (float): Current confidence after perturbation
            s_gradients: Gradients in s-space
            layer_name: Name of the layer to perturb
            rank_data: Ranked gradient data
            prediction_target: Target prediction value
            top_n: Channel index
            original_predicted_class: Original predicted class before perturbation
            step_size (float): Initial step size for hill climbing
            max_iterations (int): Maximum number of iterations
            patience (int): Number of iterations to wait before reducing step size

        Returns:
            tuple: (best_factor, best_adjusted_confidence, best_img_perturbed, best_confidence_drop)
        """

        def is_misclassified(predicted_class, original_class):
            """Check if prediction class has changed from original"""
            return predicted_class != original_class
        def evaluate_factor(factor):
            """Evaluate a factor and return results only if misclassification occurs"""
            # ([confidence_drop, prediction_perturbed_target], img_perturbed,
            #                      [predicted_class, predicted_top_confidence])
            confidence_info, img_perturbed, prediction_info = self.compare_perturbed(
                s_gradients, layer_name, rank_data, prediction_target,
                extent_factor=factor, top_n=top_n
            )

            confidence_drop, adjusted_confidence = confidence_info
            predicted_class, top_confidence = prediction_info

            if is_misclassified(predicted_class, original_predicted_class):
                decision_margin = abs(top_confidence - adjusted_confidence)
                return True, confidence_drop, img_perturbed, decision_margin, predicted_class, confidence_info, prediction_info
            else:
                return False, None, None, float('inf'), None, None, None

        # Get initial predicted class for verification
        confidence_info, _, prediction_info = self.compare_perturbed(
            s_gradients, layer_name, rank_data, prediction_target,
            extent_factor=current_factor, top_n=top_n
        )
        initial_predicted_class = prediction_info[0]  # predicted_class is at index 0

        # Verify initial condition
        if not is_misclassified(initial_predicted_class, original_predicted_class):
            print(f"Warning: Initial condition does not satisfy misclassification! {initial_predicted_class} == {original_predicted_class}")
            return current_factor, None,  confidence_info, prediction_info

        best_factor = current_factor
        best_decision_margin = float('inf')
        best_img_perturbed = None
        best_confidence_info = confidence_info
        best_prediction_info = prediction_info

        _, _, _, initial_margin, _, _, _ = evaluate_factor(current_factor)
        best_decision_margin = initial_margin

        factor = current_factor
        no_improvement_count = 0

        #print(f"Initial decision margin: {initial_margin:.3f}")

        for iteration in range(max_iterations):
            # Try both directions with constraint checking
            candidates = [
                max(1e-5, factor - step_size),  # Decrease factor
                factor + step_size  # Increase factor
            ]

            improved = False
            for candidate_factor in candidates:
                is_valid, confidence_drop, img_perturbed, decision_margin, predicted_class, conf_info, pred_info = evaluate_factor(
                    candidate_factor)

                # to find better decision margin
                if is_valid and decision_margin < best_decision_margin:
                    best_factor = candidate_factor
                    best_decision_margin = decision_margin
                    best_img_perturbed = img_perturbed
                    best_confidence_info = conf_info
                    best_prediction_info = pred_info
                    factor = candidate_factor
                    improved = True
                    no_improvement_count = 0

                    # print debugging info
                    if verbose:
                        _, adjusted_conf = conf_info
                        _, top_conf= pred_info
                        print(f"Hill climbing iteration {iteration}: factor={factor:.3f}, "
                              f"decision_margin={decision_margin:.3f}, "
                              f"top_confidence={top_conf:.3f}, adjusted_confidence={adjusted_conf:.3f}, "
                              f"predicted_class={predicted_class} (misclassification maintained)")
                    break

            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    step_size *= 0.5  # Reduce step size
                    no_improvement_count = 0
                    # print(f"Reducing step size to {step_size:.3f}")

                    if step_size < 1e-3:  # Stop if step size becomes too small
                        break

        # Final verification
        confidence_info, final_img_perturbed, prediction_info = self.compare_perturbed(
            s_gradients, layer_name, rank_data, prediction_target,
            extent_factor=best_factor, top_n=top_n
        )
        final_predicted_class = prediction_info[0]

        if not is_misclassified(final_predicted_class, original_predicted_class):
            print("Error: Final result does not maintain prediction flip!")
            return current_factor, None, confidence_info, prediction_info

        print(f"Final result: factor={best_factor:.3f}, decision_margin={best_decision_margin:.3f}")
        return best_factor, best_img_perturbed, best_confidence_info, best_prediction_info

    def compare_perturbed(self, s_gradients, layer_name, rank_data, prediction_target, extent_factor, top_n=0):
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
        extent =  (-extent_factor) *  np.sign(gradient_single)# np.sign(prediction_target) *is always decreasing
        gradient_perturbed = perturbate_s_latents(s_gradients, layer_name, location, extent)

        # Generate the perturbed image
        img_perturbed_tensor = generate_image_with_s_latents(
            synthesis_net=self.generator.synthesis,
            s_latents=gradient_perturbed
        )
        img_perturbed = img_perturbed_tensor.cpu().detach().numpy()[0].transpose(1, 2, 0)

        # Get the classifier confidence for the perturbed image

        prediction_perturbed = self.classifier(self.preprocess_fn(img_perturbed_tensor.squeeze(0)).unsqueeze(0)).squeeze(0).cpu().detach().numpy()
        adjusted_confidence,predicted_class, predicted_top_confidence = self._get_target_value(prediction_perturbed)

        confidence_drop = self._get_confidence_drop(prediction_target, prediction_perturbed)
        confidence_info = [confidence_drop, adjusted_confidence]
        prediction_info = [predicted_class, predicted_top_confidence]

        return confidence_info, img_perturbed,  prediction_info
        #return prediction_perturbed, img_perturbed

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
        Handles the generation, perturbation, and analysis of synthetic image seeds for multiclass classification.

        Args:
            torch_seed: Random seed for generating synthetic data
            default_extent_factor: Initial perturbation extent factor
            tolerance_of_extent_bisection: Tolerance for bisection adjustment
            oracle: Success criteria ("confidence_drop" or "misclassification")
            specified_layer: Optional layer to target specifically
            skip_rgb_layer: Whether to skip RGB layers
            truncation_psi: Controls variation of the generated images (lower = less variation)
            config: Method for computing gradients ("gradient", "smoothgrad", or "occlusion")

        Returns:
            Dictionary with results of the manipulation process
        """
        assert oracle in ["confidence_drop", "misclassification"], \
            "oracle must be either 'confidence_drop' or 'misclassification'"
        assert config in ["gradient", "smoothgrad", "occlusion"], \
            "config must be either 'gradient', 'smoothgrad', or 'occlusion'"

        # Set random seed for reproducibility
        torch.manual_seed(torch_seed)
        # Generate random latent vector
        z = torch.randn([1, self.generator.z_dim], device=self.device)

        # Generate image
        img_tensor = self.generator(z, None, truncation_psi=truncation_psi, noise_mode='const')
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]

        img = img_tensor.cpu().detach().numpy()[0].transpose(1, 2, 0)

        # Get prediction for the original image
        prediction = self.classifier(self.preprocess_fn(img_tensor.squeeze(0)).unsqueeze(0)).squeeze(0)
        original_prediction = prediction.cpu().detach().numpy()
        # Get target prediction value and set target class
        original_prediction_target, original_class, _ = self._get_target_value(original_prediction)

        # Get segmentation mask if available
        #if self.segmenter is not None:
        #    mask = self.segmenter.predict(img)
        #else:
        #    mask = np.zeros_like(img)

        # Calculate w vector for StyleGAN
        w = self.generator.mapping(z, c=None)
        if truncation_psi != 1:
            w = self.generator.mapping.w_avg + (w - self.generator.mapping.w_avg) * truncation_psi

        # Compute gradients using specified method
        s_gradients, classifier_output, img_tensor = self.compute_gradients(config, w)

        # Rank gradients
        ranked_gradient_info = rank_gradient_info(s_gradients, top="adaptive", layer_name = "all")

        info = {
            'seed': torch_seed,
            "oracle": oracle,
            "config": config,
            "prediction_target": original_prediction_target
        }

        # Process each layer
        if specified_layer is None:
            for layer_name, rank_data in ranked_gradient_info.items():
                if "rgb" in layer_name and skip_rgb_layer:
                    continue

                # Try each top channel
                for top_n in range(len(rank_data['ranked_indices'])):
                    [confidence_drop,adjusted_confidence], img_perturbed,  [predicted_class,
                          predicted_top_confidence]= self.compare_perturbed(
                        s_gradients, layer_name, rank_data, original_prediction_target,
                        extent_factor=default_extent_factor, top_n=top_n
                    )

                    save_flag = False
                    if oracle == "confidence_drop" and self._evaluate_oracle(
                            original_prediction_target, [adjusted_confidence, predicted_class], oracle):
                        save_flag = True

                        print(f"Layer: {layer_name}, Ranking: {top_n} confidence_drop {confidence_drop:.2f} , "
                              f"confidence {original_prediction_target:.2f} {adjusted_confidence:.2f}")
                        #if self.segmenter is not None:
                        #    mask_perturbed = self.segmenter.predict(img_perturbed)
                        #    seg_result = self.segmenter.detect_changes(img, img_perturbed, mask, mask_perturbed)
                        #else:
                        #    # mask_perturbed = np.zeros_like(img_perturbed)
                        #    seg_result = None
                    elif oracle == "misclassification" and self._evaluate_oracle(
                            original_prediction_target, [adjusted_confidence, predicted_class], oracle):
                        # misclassification
                        print("perturbed_top_class: ", predicted_class)
                        (best_factor, img_perturbed, [confidence_drop, adjusted_confidence],
                         [predicted_class,
                          predicted_top_confidence]) = self.constrained_hill_climbing_factor_adjustment(
                            current_factor=default_extent_factor,
                            s_gradients=s_gradients,
                            layer_name=layer_name,
                            rank_data=rank_data,
                            prediction_target=original_prediction_target,
                            original_predicted_class=self.target_logit,
                            top_n=top_n,
                            step_size=5,
                            max_iterations=20,
                            patience=3
                        )

                        # check if prediction flip is guaranteed
                        if (img_perturbed is not None and
                                predicted_class != self.target_logit):  #
                            save_flag = True
                            default_extent_factor = best_factor

                            print(f"✓ Prediction flip guaranteed! factor={default_extent_factor:.3f}")

                            #if self.segmenter is not None:
                            #    mask_perturbed = self.segmenter.predict(img_perturbed)
                            #    seg_result = self.segmenter.detect_changes(img, img_perturbed, mask, mask_perturbed)
                            #else:
                            #    seg_result = None
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
                            "confidence_drop": confidence_drop,
                            "original_class": f"{self.target_logit} : {self.class_dict[int(self.target_logit)]}",
                            "perturbed_class": f"{predicted_class} : {self.class_dict[int(predicted_class)]}",
                            "perturbed_confidence": adjusted_confidence,
                            "perturbed_top_confidence": predicted_top_confidence,
                            "extent_factor": default_extent_factor,
                            "top_n": top_n,
                            "layer_name": layer_name,
                            "channel_id": rank_data['ranked_indices'][top_n],
                            "gradient": rank_data['gradients'][top_n],
                            "img_path": img_perturbed_path,
                            # "significant_changes": seg_result[:8] if seg_result is not None else None
                        })

                        with open(json_path, 'w') as f:
                            json.dump(convert_to_serializable(info), f, indent=4)
                        self.plot_comparison(img, img_perturbed,
                                             confidence_drop,
                                             img_perturbed_path.replace(".png", "_comparison.png"),
                                             original_prediction_target, adjusted_confidence)
        else:
            # The case when we only want to perturb a specific layer
            print("Specific layer perturbation not implemented yet! Please implement this feature.")
            pass
        self._reset_target_class()
