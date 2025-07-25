
import torch


def occlusions_s_space(synthesis_net: torch.nn.Module,
                       classifier: torch.nn.Module,
                       preprocess: torch.nn.Module,
                       w_latents: torch.Tensor,
                       target_class=None,
                       epsilon=1e-6,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Compute Layer-wise Relevance Propagation (LRP) relevance scores for StyleGAN S-space.

    Parameters:
      synthesis_net: The StyleGAN synthesis network.
      classifier: The classifier network.
      preprocess: Preprocessing transform for classifier input.
      w_latents: The input tensor requiring explanation; shape: (N, ...).
      target_class: int or None. If the model outputs a vector, target_class selects the index to explain.
      epsilon: Small constant for numerical stability in LRP-ε rule.
      gamma: Parameter for LRP-γ rule (higher values emphasize positive contributions).
      device: The device to run the computation on.

    Returns:
      s_relevance: A dictionary with relevance scores for each S-space layer.
      classifier_output: Output logits from the classifier.
      synthesized_image: The generated image corresponding to the W-space.
    """
    synthesis_net = synthesis_net.to(device).eval()
    classifier = classifier.to(device).eval()
    w_latents = w_latents.to(device).detach()

    # Dictionary to store S-space activations and relevance scores
    s_activations = {}
    s_relevance = {}

    # Hook function to capture S-space activations
    def capture_s_activations(module, input, output):
        layer_name = module.name_module if hasattr(module, "name_module") else str(module)
        s_activations[layer_name] = output.detach().clone()

    # Register hooks for all affine layers in the synthesis network
    hooks = _register_affine_hooks(synthesis_net, capture_s_activations)

    # Forward pass to generate synthetic image and capture S-space activations
    with torch.no_grad():
        img = synthesis_net(w_latents)
        img = (img.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]

        # Classifier preprocessing & prediction
        img_clf = preprocess(img.squeeze(0)).unsqueeze(0).to(device)
        classifier_output = classifier(img_clf)

        # Determine target output
        if target_class is not None:
            target_output = classifier_output[:, target_class]
        else:
            target_output = classifier_output[:, classifier_output.argmax(dim=1)]

    # Remove hooks after forward pass
    _remove_hooks(hooks)

    # Compute relevance for each S-space layer using perturbation analysis
    for layer_name, s_activation in s_activations.items():
        layer_relevance = compute_layer_relevance_perturbation(
            synthesis_net, w_latents, layer_name, classifier, preprocess,
            target_class, target_output, epsilon, device
        )

        s_relevance[layer_name] = {
            'values': s_activation,
            'grad': layer_relevance
        }

    return s_relevance, classifier_output, img


def _register_affine_hooks(synthesis_net, hook_fn):
    """Register hooks for all affine layers and return hook handles."""
    hooks = []
    for name, module in synthesis_net.named_modules():
        if "affine" in name:
            module.name_module = name
            hooks.append(module.register_forward_hook(hook_fn))
    return hooks


def _remove_hooks(hooks):
    """Remove all hooks in the list."""
    for hook in hooks:
        hook.remove()


def compute_layer_relevance_perturbation(synthesis_net, w_latents, target_layer_name,
                                         classifier, preprocess, target_class,
                                         original_output, epsilon, device):
    """
    Compute layer relevance using small perturbations for each channel.
    """
    # First, get the original layer activation to know the shape
    original_activation = None

    def capture_activation_hook(module, input, output):
        nonlocal original_activation
        if hasattr(module, 'name_module') and module.name_module == target_layer_name:
            original_activation = output.detach().clone()
        return output

    # Register hook to capture original activation
    hooks = _register_affine_hooks(synthesis_net, capture_activation_hook)

    # Forward pass to get original activation
    with torch.no_grad():
        synthesis_net(w_latents)

    # Remove hooks
    _remove_hooks(hooks)

    if original_activation is None:
        return 0.0

    # Get the shape of the activation
    activation_shape = original_activation.shape

    # For each channel, compute relevance
    channel_relevances = []

    # Assuming the activation has shape [batch, channels, ...] or [batch, features]
    if len(activation_shape) == 2:  # [batch, features]
        num_channels = activation_shape[1]
        for channel_idx in range(num_channels):
            relevance = compute_channel_relevance(
                synthesis_net, w_latents, target_layer_name, channel_idx,
                classifier, preprocess, target_class, original_output, epsilon, device
            )
            channel_relevances.append(relevance)
    else:  # Multi-dimensional activation
        # Flatten all dimensions except batch to treat as channels
        total_elements = 1
        for dim in activation_shape[1:]:
            total_elements *= dim

        for channel_idx in range(total_elements):
            relevance = compute_channel_relevance(
                synthesis_net, w_latents, target_layer_name, channel_idx,
                classifier, preprocess, target_class, original_output, epsilon, device,
                activation_shape
            )
            channel_relevances.append(relevance)

    return torch.tensor(channel_relevances, device=device)


def compute_channel_relevance(synthesis_net, w_latents, target_layer_name, channel_idx,
                              classifier, preprocess, target_class, original_output,
                              epsilon, device, activation_shape=None):
    """
    Compute relevance for a specific channel by adding epsilon to that channel.
    """

    def perturb_channel_hook(module, input, output):
        if hasattr(module, 'name_module') and module.name_module == target_layer_name:
            # Clone the output to avoid modifying the original
            perturbed_output = output.clone()

            if activation_shape is None or len(output.shape) == 2:
                # Simple case: [batch, features]
                perturbed_output[:, channel_idx] += epsilon
            else:
                # Multi-dimensional case: convert flat index to multi-dimensional indices
                flat_output = perturbed_output.view(output.shape[0], -1)
                flat_output[:, channel_idx] += epsilon
                perturbed_output = flat_output.view(output.shape)

            return perturbed_output
        return output

    # Register perturbation hook
    hooks = _register_affine_hooks(synthesis_net, perturb_channel_hook)

    # Forward pass with perturbation
    with torch.no_grad():
        img_perturbed = synthesis_net(w_latents)
        img_perturbed = (img_perturbed.clamp(-1, 1) + 1) / 2

        img_clf_perturbed = preprocess(img_perturbed.squeeze(0)).unsqueeze(0).to(device)
        classifier_output_perturbed = classifier(img_clf_perturbed)

        if target_class is not None:
            target_output_perturbed = classifier_output_perturbed[:, target_class]
        else:
            target_output_perturbed = classifier_output_perturbed[:, classifier_output_perturbed.argmax(dim=1)]

    # Remove hooks
    _remove_hooks(hooks)

    # Compute relevance as sensitivity to perturbation
    output_change = (target_output_perturbed - original_output)
    relevance = output_change# / epsilon

    return relevance.item()


if __name__ == "__main__":
    import os
    import time
    start_time = time.time()

    os.chdir("..")
    from configs import gan_facial_ckpt_path, sut_facial_path, preprocess_celeb_classifier
    from utils import load_generator, load_facial_classifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan = load_generator(gan_facial_ckpt_path, device)
    classifier = load_facial_classifier(sut_facial_path, device)
    torch.manual_seed(0)
    z = torch.randn([1, gan.z_dim], device=device)
    target_class = 15
    truncation_psi = 0.7
    w = gan.mapping(z, None)
    w = gan.mapping.w_avg + (w - gan.mapping.w_avg) * truncation_psi

    img_tensor = gan.synthesis(w, noise_mode='const')
    prediction = classifier(preprocess_celeb_classifier(img_tensor))[0, target_class]
    s_relevance, classifier_output, img = occlusions_s_space(
        gan.synthesis, classifier, preprocess_celeb_classifier, w,
        target_class=15, epsilon=1e-6
    )
    print(s_relevance)
    print("Execution time: {:.2f} seconds".format(time.time() - start_time))
    # 485.38 seconds

    from backpropagation import backpropagation_gradients_s_space
    start_time = time.time()
    s_gradients, classifier_output, img = backpropagation_gradients_s_space(
        gan.synthesis, classifier, preprocess_celeb_classifier, w,
        target_class=15
    )
    print(s_gradients)
    print("Execution time: {:.2f} seconds".format(time.time() - start_time))
    # Execution time: 0.88 seconds