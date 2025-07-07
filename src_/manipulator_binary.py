import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from utils import rank_gradient_info, perturbate_s_latents, convert_to_serializable
from src_.backpropagation import generate_image_with_s_latents, backpropagation_gradients_s_space
from src_.occlusion import occlusions_s_space


class ManipulatorSSpace:
    """
    A class to automatically select and manipulate the channels in style space of the generator to achieve confidence
    reduction of the target class and generate perturbed images for analysis.
    """
    def __init__(self, generator, classifier, segmenter, preprocess_fn, target_logit, save_dir,device=None):
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
        extent = np.sign(prediction_target) * (-extent_factor) * gradient_single
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

    def handle_one_seed(self,
                        torch_seed,
                        top_channels=1,
                        default_extent_factor=40,
                        tolerance_of_extent_bisection=1,
                        confidence_drop_threshold=0.3,
                        criterion='entropy',
                        specified_layer=None,
                        skip_rgb_layer=True,
                        truncation_psi = 0.7):
        """
        handle one seed

        Args:
            torch_seed: seed to generate
        """
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
                               truncation_psi=1, noise_mode='const')# ensure deterministic and reproducible
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
            mask = np.zeros_like(img)

        # calculate w vector
        w = self.generator.mapping(z, c=label)

        s_gradients, classifier_output, img_tensor = backpropagation_gradients_s_space(
            synthesis_net = self.generator.synthesis,
            classifier = self.classifier,
            preprocess= self.preprocess_fn,
            w_latents= w,
            target_class= self.target_logit, # glasses
        )
        s_relevance, classifier_output, img = occlusions_s_space(
            synthesis_net = self.generator.synthesis,
            classifier = self.classifier,
            preprocess= self.preprocess_fn,
            w_latents= w,
            target_class=self.target_logit,
            epsilon=1e-6, gamma=0.25
        )

        # rank gradient to get the most important channel of each layer
        ranked_gradient_info = rank_gradient_info(s_gradients, top=top_channels)

        if specified_layer is None:
            for i, (layer_name, rank_data) in enumerate(ranked_gradient_info.items()):
                # layer_name, rank_data = list(ranked_gradient_info.items())[0]
                if "rgb" in layer_name and skip_rgb_layer: # skip rgb layer
                    continue
                for top_n in range(top_channels):
                    confidence_drop, img_perturbed, prediction_perturbed_target = self.compare_perturbed(
                        s_gradients, layer_name, rank_data, prediction_target, extent_factor=default_extent_factor, top_n=top_n)

                    if confidence_drop < confidence_drop_threshold* np.abs(prediction_target):
                        continue
                    print(f"Layer: {layer_name}, Ranking: {top_n} confidence_drop {confidence_drop:.2f} , "
                          f"confidence {prediction_target:.2f} {prediction_perturbed_target:.2f}")

                    if prediction_target * prediction_perturbed_target<0:

                        (best_factor, best_adjusted_confidence, best_img_perturbed, best_confidence_drop) = self.bisection_factor_adjustment(default_extent_factor,
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
                    file_name = f"seed_{torch_seed}_{layer_name.replace('.', '_')}_{rank_data['ranked_indices'][top_n]}"

                    os.makedirs(self.save_dir, exist_ok=True)
                    img_perturbed_path = os.path.join(self.save_dir, file_name + ".png")
                    json_path = os.path.join(self.save_dir, file_name + ".json")

                    if self.segmenter is not None:
                        mask_perturbed = self.segmenter.predict(img_perturbed)
                        result = self.segmenter.detect_changes(img, img_perturbed, mask, mask_perturbed)
                    else:
                        #mask_perturbed = np.zeros_like(img_perturbed)
                        result = None


                    info = {
                        'seed': torch_seed,
                        "confidence_drop": confidence_drop,
                        "prediction_target": prediction_target,
                        "prediction_perturbed_target": prediction_perturbed_target,
                        "extent_factor": default_extent_factor,
                        "top_n": top_n,
                        "layer_name": layer_name,
                        "channel_id": rank_data['ranked_indices'][top_n],
                        "gradient": rank_data['gradients'][top_n],
                        "img_path": img_perturbed_path,
                        "significant_changes": result[:8] if result is not None else None
                    }

                    with open(json_path, 'w') as f:
                        json.dump(convert_to_serializable(info), f, indent=4)
                    self.plot_comparison(img, img_perturbed,
                                           confidence_drop,
                                           img_perturbed_path,
                                         prediction_target, prediction_perturbed_target)
        else:
            # The case when we only want to perturb a specific layer

            pass