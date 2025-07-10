def hill_climbing_factor_adjustment(self, current_factor, initial_confidence, current_confidence,
                                    s_gradients, layer_name, rank_data, prediction_target, top_n,
                                    step_size=0.1, max_iterations=20, patience=3):
    """
    Use hill climbing to adjust the extent_factor for misclassification with confidence closest to 0.

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
    best_factor = current_factor
    best_adjusted_confidence = current_confidence
    best_img_perturbed = None
    best_confidence_drop = None

    factor = current_factor
    no_improvement_count = 0

    for iteration in range(max_iterations):
        # Try both directions
        candidates = [
            max(1e-5, factor - step_size),  # Decrease factor
            factor + step_size  # Increase factor
        ]

        improved = False
        for candidate_factor in candidates:
            confidence_drop, img_perturbed, adjusted_confidence = self.compare_perturbed(
                s_gradients, layer_name, rank_data, prediction_target,
                extent_factor=candidate_factor, top_n=top_n
            )

            # Check if we achieved misclassification and if it's better (closer to 0)
            if adjusted_confidence * initial_confidence < 0:
                if abs(adjusted_confidence) < abs(best_adjusted_confidence):
                    best_factor = candidate_factor
                    best_adjusted_confidence = adjusted_confidence
                    best_img_perturbed = img_perturbed
                    best_confidence_drop = confidence_drop
                    factor = candidate_factor
                    improved = True
                    no_improvement_count = 0
                    print(f"Hill climbing iteration {iteration}: factor={factor:.3f}, "
                          f"confidence={adjusted_confidence:.3f}")
                    break

        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                step_size *= 0.5  # Reduce step size
                no_improvement_count = 0
                print(f"Reducing step size to {step_size:.3f}")

                if step_size < 1e-3:  # Stop if step size becomes too small
                    break

    return best_factor, best_adjusted_confidence, best_img_perturbed, best_confidence_drop