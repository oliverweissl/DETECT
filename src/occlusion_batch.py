import torch
import torch.nn.functional as F
import gc


def compute_layer_relevance_vectorized_perturbation(synthesis_net, w_latents, target_layer_name,
                                                    classifier, preprocess, target_class,
                                                    original_output, epsilon, device, chunk_size=32):
    """
    Memory-efficient vectorized version - process perturbations in chunks to avoid OOM.
    """
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # First, get the original layer activation to know the shape
    original_activation = None

    def capture_activation_hook(module, input, output):
        nonlocal original_activation
        if hasattr(module, 'name_module') and module.name_module == target_layer_name:
            original_activation = output.detach().clone()
        return output

    # Register hook to capture original activation
    hooks = []
    for name, module in synthesis_net.named_modules():
        if "affine" in name:
            module.name_module = name
            hooks.append(module.register_forward_hook(capture_activation_hook))

    # Forward pass to get original activation
    with torch.no_grad():
        synthesis_net(w_latents)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    if original_activation is None:
        return torch.tensor(0.0, device=device)

    # Get the shape of the activation
    activation_shape = original_activation.shape
    batch_size = activation_shape[0]

    # Flatten all dimensions except batch to treat as channels
    if len(activation_shape) == 2:  # [batch, features]
        num_channels = activation_shape[1]
        flat_shape = activation_shape
    else:  # Multi-dimensional activation
        num_channels = 1
        for dim in activation_shape[1:]:
            num_channels *= dim
        flat_shape = (batch_size, num_channels)

    # Determine optimal chunk size based on available memory
    if torch.cuda.is_available():
        # Get available GPU memory
        free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        # Estimate memory per sample (conservative estimate)
        estimated_memory_per_sample = w_latents.numel() * w_latents.element_size() * 50  # Factor for intermediate activations
        max_chunk_size = max(1, int(free_memory * 0.5 / estimated_memory_per_sample))  # Use 50% of available memory
        chunk_size = min(chunk_size, max_chunk_size, num_channels)
    else:
        chunk_size = min(chunk_size, num_channels)

    print(f"Processing {num_channels} channels in chunks of {chunk_size}")

    # Process in chunks to avoid memory issues
    all_relevances = []

    for start_idx in range(0, num_channels, chunk_size):
        end_idx = min(start_idx + chunk_size, num_channels)
        current_chunk_size = end_idx - start_idx

        # Create identity matrix for current chunk
        identity_chunk = torch.eye(current_chunk_size, device=device) * epsilon

        # Expand w_latents to current chunk size
        w_shape = w_latents.shape
        if len(w_shape) == 4:  # [batch, seq, layers, features]
            w_batch = w_latents.unsqueeze(0).expand(current_chunk_size, -1, -1, -1, -1).contiguous()
            w_batch = w_batch.view(current_chunk_size * batch_size, w_shape[1], w_shape[2], w_shape[3])
        elif len(w_shape) == 3:  # [batch, seq, features]
            w_batch = w_latents.unsqueeze(0).expand(current_chunk_size, -1, -1, -1).contiguous()
            w_batch = w_batch.view(current_chunk_size * batch_size, w_shape[1], w_shape[2])
        else:  # [batch, features]
            w_batch = w_latents.unsqueeze(0).expand(current_chunk_size, -1, -1).contiguous()
            w_batch = w_batch.view(current_chunk_size * batch_size, w_shape[1])

        # Vectorized perturbation hook for current chunk
        def vectorized_perturb_hook(module, input, output):
            if hasattr(module, 'name_module') and module.name_module == target_layer_name:
                output_shape = output.shape
                if len(activation_shape) == 2:
                    output_reshaped = output.view(current_chunk_size, batch_size, -1)
                    # Only perturb the relevant channels for this chunk
                    output_perturbed = output_reshaped.clone()
                    for i in range(current_chunk_size):
                        channel_idx = start_idx + i
                        if channel_idx < output_reshaped.shape[2]:
                            output_perturbed[i, :, channel_idx] += epsilon
                    return output_perturbed.view(current_chunk_size * batch_size, -1)
                else:
                    output_reshaped = output.view(current_chunk_size, batch_size, *activation_shape[1:])
                    output_perturbed = output_reshaped.clone()
                    # Create perturbation mask for current chunk
                    flat_output = output_reshaped.view(current_chunk_size, batch_size, -1)
                    for i in range(current_chunk_size):
                        channel_idx = start_idx + i
                        if channel_idx < flat_output.shape[2]:
                            flat_output[i, :, channel_idx] += epsilon
                    return flat_output.view(current_chunk_size * batch_size, *activation_shape[1:])
            return output

        # Register hook for current chunk
        hooks = []
        for name, module in synthesis_net.named_modules():
            if "affine" in name:
                module.name_module = name
                hooks.append(module.register_forward_hook(vectorized_perturb_hook))

        # Process current chunk
        with torch.no_grad():
            try:
                # Forward pass with batch of perturbations
                img_batch = synthesis_net(w_batch)
                img_batch = (img_batch.clamp(-1, 1) + 1) / 2

                # Reshape img_batch back to [current_chunk_size, batch_size, ...]
                img_batch = img_batch.view(current_chunk_size, batch_size, *img_batch.shape[1:])

                # Process each perturbation
                classifier_outputs = []
                for i in range(current_chunk_size):
                    img_i = img_batch[i]  # [batch_size, ...]

                    # Process each item in the batch
                    batch_outputs = []
                    for j in range(batch_size):
                        img_clf = preprocess(img_i[j]).unsqueeze(0).to(device)
                        classifier_output = classifier(img_clf)
                        batch_outputs.append(classifier_output)

                    # Stack batch outputs
                    batch_outputs = torch.cat(batch_outputs, dim=0)  # [batch_size, num_classes]
                    classifier_outputs.append(batch_outputs)

                classifier_outputs = torch.stack(classifier_outputs,
                                                 dim=0)  # [current_chunk_size, batch_size, num_classes]

                if target_class is not None:
                    target_outputs = classifier_outputs[:, :, target_class]  # [current_chunk_size, batch_size]
                else:
                    target_outputs = classifier_outputs.gather(2,
                                                               classifier_outputs.argmax(dim=2, keepdim=True)).squeeze(
                        2)

                # Average over batch dimension if needed
                if batch_size > 1:
                    target_outputs = target_outputs.mean(dim=1)  # [current_chunk_size]
                else:
                    target_outputs = target_outputs.squeeze(1)  # [current_chunk_size]

                # Compute relevance for current chunk
                output_changes = torch.abs(target_outputs - original_output.item())
                all_relevances.append(output_changes)

            except torch.cuda.OutOfMemoryError:
                print(f"OOM error with chunk size {chunk_size}, switching to CPU processing")
                # Fallback to CPU processing
                w_batch = w_batch.cpu()
                synthesis_net = synthesis_net.cpu()
                classifier = classifier.cpu()

                # Process on CPU
                img_batch = synthesis_net(w_batch)
                img_batch = (img_batch.clamp(-1, 1) + 1) / 2
                img_batch = img_batch.view(current_chunk_size, batch_size, *img_batch.shape[1:])

                classifier_outputs = []
                for i in range(current_chunk_size):
                    img_i = img_batch[i]
                    batch_outputs = []
                    for j in range(batch_size):
                        img_clf = preprocess(img_i[j]).unsqueeze(0)
                        classifier_output = classifier(img_clf)
                        batch_outputs.append(classifier_output)
                    batch_outputs = torch.cat(batch_outputs, dim=0)
                    classifier_outputs.append(batch_outputs)

                classifier_outputs = torch.stack(classifier_outputs, dim=0)

                if target_class is not None:
                    target_outputs = classifier_outputs[:, :, target_class]
                else:
                    target_outputs = classifier_outputs.gather(2,
                                                               classifier_outputs.argmax(dim=2, keepdim=True)).squeeze(
                        2)

                if batch_size > 1:
                    target_outputs = target_outputs.mean(dim=1)
                else:
                    target_outputs = target_outputs.squeeze(1)

                output_changes = target_outputs - original_output.item()
                all_relevances.append(output_changes.to(device))

                # Move local_models back to GPU
                synthesis_net = synthesis_net.to(device)
                classifier = classifier.to(device)

        # Remove hooks after processing chunk
        for hook in hooks:
            hook.remove()

        # Clear cache after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

    # Concatenate all relevances
    relevances = torch.cat(all_relevances, dim=0)

    return relevances


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
    hooks = []
    for name, module in synthesis_net.named_modules():
        if "affine" in name:
            module.name_module = name
            hooks.append(module.register_forward_hook(capture_s_activations))

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
    for hook in hooks:
        hook.remove()

    # Initialize relevance propagation from classifier output
    current_relevance = target_output.clone()

    # Compute relevance for each S-space layer using batch perturbation analysis
    for layer_name, s_activation in s_activations.items():
        layer_relevance = compute_layer_relevance_vectorized_perturbation(
            synthesis_net, w_latents, layer_name, classifier, preprocess,
            target_class, target_output, epsilon, device
        )
        s_relevance[layer_name] = {
            'values': s_activation,
            'relevance': layer_relevance
        }

    return s_relevance, classifier_output, img


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
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    # Execution time: 1487.92 seconds