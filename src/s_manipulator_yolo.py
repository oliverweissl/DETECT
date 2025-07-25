import numpy as np
import torch
import os
import json
from src.utils import perturbate_s_latents, rank_gradient_info, convert_to_serializable, predict_yolo

from src.s_manipulator import BaseManipulatorSSpace
from src.backpropagation import generate_image_with_s_latents
from src.backpropagation import generate_image_with_s_latents, backpropagation_gradients_s_space_yolo, smoothgrad_s_space_yolo

class ObjectDetectionManipulatorSSpace(BaseManipulatorSSpace):
    """Manipulator for multiclass classification tasks."""

    def __init__(self, generator, classifier, segmenter,  save_dir,
                 class_dict, confidence_drop_threshold=0.3, device=None, preprocess_fn=None):
        super().__init__(generator, classifier, segmenter, preprocess_fn, save_dir, confidence_drop_threshold, device)
        self.target_logit = None
        self.class_dict = class_dict

    def _get_target_value(self, prediction):
        """
        Extract the target prediction value from the classifier's output.
        For multiclass, this is the confidence of the target class.
        TODO
        """
        if isinstance(prediction, np.ndarray):
            predicted_class = np.argmax(prediction)
            highest_confidence = prediction[predicted_class]

            return prediction[self.target_logit], predicted_class, highest_confidence
        elif isinstance(prediction, torch.Tensor):
            #print("_get_target_value",prediction.shape)
            result = self._get_target_value(prediction.squeeze(0).cpu().detach().numpy())
            return result
        else:
            raise ValueError(f"Unsupported prediction type: {type(prediction)}")

    def _set_target_class(self, target_logit, target_idx):
        self.target_logit = target_logit
        self.target_idx = target_idx

    def _get_target_class(self):
        """Return the target class for gradient computation."""
        return self.target_idx, self.target_logit

    def _get_confidence_drop(self, initial_confidence, adjusted_confidence):
        """Calculate the confidence drop between original and perturbed predictions."""
        # TODO
        initial_target = self._get_target_value(initial_confidence)[0]  if isinstance(initial_confidence, np.ndarray) else initial_confidence
        adjusted_target = self._get_target_value(adjusted_confidence)[0] if isinstance(adjusted_confidence, np.ndarray) else adjusted_confidence

        return initial_target - adjusted_target

    def _evaluate_oracle(self, original_pred, perturbed_pred, oracle):
        """
        Evaluate if the perturbation achieved the desired outcome based on the specified oracle.

        Args:
            original_pred: Original prediction value
            perturbed_pred: [confidence_drop, predicted class,]
            oracle: Success criteria ("confidence_drop" or "misclassification")
            predicted_class: Predicted class
        Returns:
            bool: Whether the perturbation was successful according to the oracle
        """
        confidence_drop = perturbed_pred[0]
        perturbed_class = perturbed_pred[1]
        if oracle == "confidence_drop":
            return confidence_drop > self.confidence_drop_threshold * np.abs(original_pred)
        elif oracle == "misclassification":
            return self.target_logit != perturbed_class
        return False


    def constrained_hill_climbing_factor_adjustment(self, current_factor, current_confidence,
                                                    s_gradients, layer_name, rank_data, prediction_target, top_n,
                                                    original_predicted_class, step_size=0.2, max_iterations=20,
                                                    patience=3):
        """
        Use constrained hill climbing to adjust extent_factor ensuring prediction flip is maintained.

        Args:
            current_factor (float): Initial factor value
            current_confidence (float): Current confidence after perturbation
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

        # [confidence_drop, target_confidence], img_perturbed,  [predicted_idx, predicted_class,top_confidence,boxes_xywh]
        def evaluate_factor(factor):
            """Evaluate a factor and return results only if misclassification occurs"""
            # YOLO version returns: [confidence_drop, target_confidence], img_perturbed, [predicted_idx, predicted_class, top_confidence, boxes_xywh]
            confidence_info, img_perturbed, prediction_info = self.compare_perturbed(
                s_gradients, layer_name, rank_data, prediction_target,
                extent_factor=factor, top_n=top_n
            )

            confidence_drop, adjusted_confidence = confidence_info
            _, predicted_class, top_confidence, _ = prediction_info

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
        initial_predicted_class = prediction_info[1]  # predicted_class is at index 1

        # Verify initial condition
        if not is_misclassified(initial_predicted_class, original_predicted_class):
            print("Warning: Initial condition does not satisfy misclassification!")
            return current_factor, None, confidence_info, prediction_info

        best_factor = current_factor
        best_decision_margin = float('inf')
        best_img_perturbed = None
        best_confidence_info = confidence_info
        best_prediction_info = prediction_info

        _, _, _, initial_margin, _, _, _ = evaluate_factor(current_factor)
        best_decision_margin = initial_margin

        factor = current_factor
        no_improvement_count = 0

        # print(f"Initial decision margin: {initial_margin:.3f}")

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
                    _, adjusted_conf = conf_info
                    _, _, top_conf, _ = pred_info
                    # print(f"Hill climbing iteration {iteration}: factor={factor:.3f}, "
                    #      f"decision_margin={decision_margin:.3f}, "
                    #      f"top_confidence={top_conf:.3f}, adjusted_confidence={adjusted_conf:.3f}, "
                    #      f"predicted_class={predicted_class} (misclassification maintained)")
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
        final_predicted_class = prediction_info[1]

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
        extent = (-extent_factor) * np.sign(gradient_single)# np.sign(prediction_target) * in yolo is always decreasing
        gradient_perturbed = perturbate_s_latents(s_gradients, layer_name, location, extent)

        # Generate the perturbed image
        img_perturbed_tensor = generate_image_with_s_latents(
            synthesis_net=self.generator.synthesis,
            s_latents=gradient_perturbed
        )
        img_perturbed = img_perturbed_tensor.cpu().detach().numpy()[0].transpose(1, 2, 0)

        # Get the classifier confidence for the perturbed image
        predicted_idx, predicted_class, boxes_xywh, top_confidence, target_confidence, _ = predict_yolo(self.classifier, img_perturbed_tensor, self.device, target_class=self.target_logit)
        confidence_drop =  (prediction_target - target_confidence) # np.sign(prediction_target) *
        confidence_info = [confidence_drop, target_confidence]
        prediction_info = [predicted_idx, predicted_class, top_confidence, boxes_xywh]
        return confidence_info, img_perturbed,  prediction_info

        #return prediction_perturbed, img_perturbed
    def compute_gradients(self, config, w_latents):
        """
        Compute gradients using different methods.
        Common implementation with task-specific target handling.
        """
        if config == "gradient":
            return backpropagation_gradients_s_space_yolo(
                synthesis_net=self.generator.synthesis,
                classifier=self.classifier,
                w_latents=w_latents,
                target_idx=self.target_idx,
                target_class=self.target_logit
             )
        elif config == "smoothgrad":
            return smoothgrad_s_space_yolo(
                synthesis_net=self.generator.synthesis,
                classifier=self.classifier,
                w_latents=w_latents,
                target_idx=self.target_idx,
                target_class=self.target_logit,
                n_samples=10,
                noise_scale=0.2,
                device=self.device
             )

    def handle_one_seed(self,
                        torch_seed,
                        default_extent_factor=10,
                        oracle="confidence_drop",  # "misclassification"
                        specified_layer=None,
                        skip_rgb_layer=True,#
                        truncation_psi=0.7,
                        config="gradient"):
        """
        handle one seed

        Args:
            torch_seed: seed to generate
        """

        assert oracle in ["confidence_drop",
                          "misclassification"], "oracle must be either 'confidence_drop' or 'misclassification'"
        assert config in ["gradient", "smoothgrad"], "config must be either 'gradient' or 'smoothgrad'"

        torch.manual_seed(torch_seed)
        # generate one random seed from z latent space
        z = torch.randn([1, self.generator.z_dim], device=self.device)

        img_tensor = self.generator(z, None,
                                    truncation_psi=truncation_psi,
                                    noise_mode='const')  # ensure deterministic and reproducible
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2  # normalize to [0, 1] ss

        img = img_tensor.cpu().detach().numpy()[0].transpose(1, 2, 0)

        # get prediction of the original image
        # default target is car 2
        original_top_idx, original_top_class, original_boxes_xywh, original_top_confidence, _, original_top_confidence_norm = predict_yolo(
            self.classifier, img_tensor,
            self.device, 2)
        if original_top_class != 2:  # or original_top_confidence_norm<0.72:
            # the predicted class is not car
            print(f"the original image is not a car or the confidence is too low, skip it")
            return

        self._set_target_class(target_logit=original_top_class, target_idx=original_top_idx)

        # get mask
        #if self.segmenter is not None:
        #    mask = self.segmenter.predict(img)
        #else:
        #    mask = np.zeros_like(img)  # dummy mask

        # calculate w vector
        w = self.generator.mapping(z, c=None)
        if truncation_psi != 1:
            w = self.generator.mapping.w_avg + (w - self.generator.mapping.w_avg) * truncation_psi

        # Compute gradients using specified method
        s_gradients, classifier_output, img_tensor = self.compute_gradients(config, w)

        # rank gradient to get the most important channel of each layer
        ranked_gradient_info = rank_gradient_info(s_gradients,  top="adaptive", layer_name = "all")

        if specified_layer is None:
            for i, (layer_name, rank_data) in enumerate(ranked_gradient_info.items()):
                if "rgb" in layer_name and skip_rgb_layer:  # skip rgb layer
                    continue

                for top_n in range(len(rank_data['ranked_indices'])):
                    [confidence_drop, perturbed_confidence], img_perturbed, [perturbed_idx, perturbed_top_class,
                                                                             perturbed_top_confidence,
                                                                             perturbed_boxes_xywh] = self.compare_perturbed(
                        s_gradients, layer_name, rank_data,
                        original_top_confidence, extent_factor=default_extent_factor, top_n=top_n)
                    save_flag = False
                    if oracle == "confidence_drop" and confidence_drop > self._evaluate_oracle(
                            original_top_confidence, [confidence_drop, perturbed_top_class], oracle):
                        save_flag = True

                        print(f"Layer: {layer_name}, Ranking: {top_n} confidence_drop {confidence_drop:.2f} , "
                              f"confidence {original_top_confidence:.2f} {perturbed_confidence:.2f}")
                        #if self.segmenter is not None:
                        #    mask_perturbed = self.segmenter.predict(img_perturbed)
                        #    seg_result = self.segmenter.detect_changes(img, img_perturbed, mask, mask_perturbed)
                        #else:
                        #    # mask_perturbed = np.zeros_like(img_perturbed)
                        #    seg_result = None

                    elif oracle == "misclassification" and perturbed_top_class != original_top_class:
                        print("perturbed_top_class: ", perturbed_top_class, self.class_dict[perturbed_top_class])
                        (best_factor, img_perturbed, [confidence_drop, perturbed_confidence],
                         [predicted_idx, perturbed_top_class, perturbed_top_confidence,
                          perturbed_boxes_xywh]) = self.constrained_hill_climbing_factor_adjustment(
                            current_factor=default_extent_factor,
                            current_confidence=perturbed_top_confidence,
                            s_gradients=s_gradients,
                            layer_name=layer_name,
                            rank_data=rank_data,
                            prediction_target=original_top_confidence,
                            original_predicted_class=original_top_class,
                            top_n=top_n,
                            step_size=5,
                            max_iterations=20,
                            patience=3
                        )
                        # check if prediction flip is guaranteed
                        if (img_perturbed is not None and
                                perturbed_top_class != original_top_class):  #
                            save_flag = True
                            default_extent_factor = best_factor

                            print(f"✓ Prediction flip guaranteed! factor={default_extent_factor:.3f}")
                            """if self.segmenter is not None:
                                mask_perturbed = self.segmenter.predict(img_perturbed)
                                seg_result = self.segmenter.detect_changes(img, img_perturbed, mask, mask_perturbed)
                            else:
                                seg_result = None"""
                    if save_flag:
                        # file_name = f"seed_{torch_seed}_{layer_name.replace('.', '_')}_{rank_data['ranked_indices'][top_n]}"
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
                            "config": config,
                            "confidence_drop": confidence_drop,
                            "original_idx": self.target_idx,
                            "original_class": f"{self.target_logit} : {self.class_dict[int(self.target_logit)]}",
                            "original_confidence": original_top_confidence,
                            "original_boxes_xywh": original_boxes_xywh,
                            "perturbed_idx": perturbed_idx,
                            "perturbed_class": f"{perturbed_top_class} : {self.class_dict[int(perturbed_top_class)]}",
                            "perturbed_confidence": perturbed_confidence,
                            "perturbed_top_confidence": perturbed_top_confidence,
                            "perturbed_boxes_xywh": perturbed_boxes_xywh,
                            "extent_factor": default_extent_factor,
                            "top_n": top_n,
                            "layer_name": layer_name,
                            "channel_id": rank_data['ranked_indices'][top_n],
                            "gradient": rank_data['gradients'][top_n],
                            "img_path": img_perturbed_path,
                            # "significant_changes": seg_result[:8] if seg_result is not None else None
                        }

                        with open(json_path, 'w') as f:
                            json.dump(convert_to_serializable(info), f, indent=4)
                        self.plot_comparison(img, img_perturbed,
                                             confidence_drop,
                                             img_perturbed_path.replace(".png", "_comparison.png"),
                                             original_top_confidence, perturbed_confidence)
        else:
            # The case when we only want to perturb a specific layer
            print("Specific layer perturbation not implemented yet! Please implement this feature.")
            pass
