import torch

def backpropagation_gradients_w_space(synthesis_net: torch.nn.Module,
                                      classifier: torch.nn.Module,
                                      preprocess: torch.nn.Module,
                                      w_latents: torch.Tensor,
                                      target_class=None,
                                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Compute gradients of the classifier's output with respect to the input_tensor via backpropagation.

    Parameters:
      synthesis_net: The StyleGAN synthesis network.
      classifier: The classifier network.
      preprocess: Preprocessing transform for classifier input.
      w_latents: The input tensor requiring explanation; shape: (N, ...).
      target_class: int or None. If the model outputs a vector, target_class selects the index to explain.
                    If None, the output is assumed to be a scalar.
      device: The device to run the computation on.

    Returns:
      gradients: A tensor of the same shape as input_tensor containing the computed gradients.
      classifier_output: Output logits from classifier.
      synthesized_image: The generated image.
    """
    # Move the input tensor to the specified device and ensure it requires gradient computation
    w_latents = w_latents.to(device)
    w_latents = w_latents.detach().clone().requires_grad_(True)

    # Forward pass: compute model output
    # ==== synthesis image ====
    img = synthesis_net(w_latents)  # input per-layer w，shape: (1, 3, H, W)
    img = (img.clamp(-1, 1) + 1) / 2  # necessary otherwise predicted as wrong class

    # ==== classifier output ====
    if device.type == 'cuda':
        img_clf = preprocess(img.squeeze(0)).unsqueeze(0).cuda()
    else:
        img_clf = preprocess(img.squeeze(0)).unsqueeze(0)

    # get logits from classifier, resnet18 do not include a softmax layer in their architecture. So the output of the model is already the logits.
    classifier_output = classifier(img_clf)
    print(f"output range {classifier_output.min()}, {classifier_output.max()}")  # to check if it has softmax
    predict_class = classifier_output.argmax(dim=1, keepdim=True)

    # Select the output corresponding to target_class if specified, else use the output directly
    if target_class is not None:
        # Assuming model output shape is (batch_size, num_classes)
        target = classifier_output[:, target_class]
    else:
        target = classifier_output[:, predict_class]

    # Clear any existing gradients
    synthesis_net.zero_grad()
    classifier.zero_grad()

    # Backward pass: compute gradients
    # Create a tensor of ones with the same shape as output for gradient computation
    grad_outputs = torch.ones_like(
        target)  # effectively setting the initial gradient of the computational graph when backpropagation begins.
    target.backward(grad_outputs)  # backward pass

    # The gradients of input_tensor with respect to the model output are now stored in input_tensor.grad
    gradients = w_latents.grad.clone()
    return gradients, classifier_output, img


def get_s_vectors(synthesis_net, w_latents, device):
    """
    From a PyTorch StyleGAN Synthesis Network, extract style vectors (S-space)
    from layers with '.affine' in their names.

    Args:
        synthesis_net (nn.Module): The StyleGAN Synthesis network.
        w_latents (torch.Tensor): Latent codes in the W space, shape [N, num_ws, w_dim].
        device (torch.device): The computation device.

    Returns:
        dict: A dictionary with keys as layer names ending in '.affine' and
              values as the corresponding S vectors (style vectors).
    """
    synthesis_net = synthesis_net.to(device)
    w_latents = w_latents.to(device)

    # Dictionary to capture S vectors based on layer names
    s_vectors = {}

    # Define Hook function
    def hook_fn(module, input, output):
        layer_name = module.name_module if hasattr(module, "name_module") else str(module)
        s_vectors[layer_name] = output.detach().cpu()

    # Register hooks for all layers containing 'affine'
    hooks = []
    for name, module in synthesis_net.named_modules():
        if "affine" in name:  # Capture affine layers
            module.name_module = name  # Save layer name for identification
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        synthesis_net(w_latents)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return s_vectors


def backpropagation_gradients_s_space(synthesis_net: torch.nn.Module,
                                      classifier: torch.nn.Module,
                                      preprocess: torch.nn.Module,
                                      w_latents: torch.Tensor,
                                      target_class=None,
                                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Compute gradients of the classifier's output with respect to the input_tensor via backpropagation.

    Parameters:
      synthesis_net: The StyleGAN synthesis network.
      classifier: The classifier network.
      preprocess: Preprocessing transform for classifier input.
      w_latents: The input tensor requiring explanation; shape: (N, ...).
      target_class: int or None. If the model outputs a vector, target_class selects the index to explain.
                    If None, the output is assumed to be a scalar.
      device: The device to run the computation on.

    Returns:
      s_gradients: A dictionary with values and gradients for each S-space layer.
      classifier_output: Output logits from the classifier.
      synthesized_image: The generated image corresponding to the W-space.
    """
    synthesis_net = synthesis_net.to(device).eval()
    classifier = classifier.to(device).eval()
    w_latents = w_latents.to(device).detach()

    # Dictionary to store gradients
    s_gradients = {}

    # Define Hook function to capture S vectors and retain gradient computation
    def hook_fn(module, input, output):
        output.requires_grad_(True)  # Ensure S vectors have gradients
        layer_name = module.name_module if hasattr(module, "name_module") else str(module)
        s_gradients[layer_name] = {'values': output, 'grad': None}

    # Register hooks for all affine layers in the synthesis network
    hooks = []
    for name, module in synthesis_net.named_modules():
        if "affine" in name:  # Assuming S-space corresponds to affine layers
            module.name_module = name
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass to generate synthetic image
    # with torch.no_grad():
    img = synthesis_net(w_latents)  # Generate image from W space
    img = (img.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]

    # Classifier preprocessing & prediction
    img_clf = preprocess(img.squeeze(0)).unsqueeze(0).to(device)
    classifier_output = classifier(img_clf)

    # Specify the target class for backpropagation
    if target_class is not None:
        target_output = classifier_output[:, target_class]
    else:
        target_output = classifier_output[:, classifier_output.argmax(dim=1)]

    # Backpropagation to compute gradients
    grad_outputs = torch.ones_like(target_output)
    target_output.backward(grad_outputs)

    # Extract gradients for each S layer (updated logic)
    for layer_name, data in s_gradients.items():
        if data['values'].grad is None:
            raise RuntimeError(
                f"Gradient for layer '{layer_name}' is None. "
                f"Ensure that 'data['values']' has 'requires_grad=True' and "
                f"that the computation graph is intact."
            )

        # Safeguard: Ensure gradients exist before cloning
        s_gradients[layer_name]['grad'] = data['values'].grad.clone()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Return gradients, classifier output, and synthesized image
    return s_gradients, classifier_output, img


def smoothgrad_s_space(synthesis_net: torch.nn.Module,
                       classifier: torch.nn.Module,
                       preprocess: torch.nn.Module,
                       w_latents: torch.Tensor,
                       target_class=None,
                       n_samples=10,
                       noise_scale=0.15,
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Compute SmoothGrad for S-space representations.

    Parameters:
      synthesis_net: The StyleGAN synthesis network.
      classifier: The classifier network.
      preprocess: Preprocessing transform for classifier input.
      w_latents: The input tensor requiring explanation; shape: (N, ...).
      target_class: int or None. If the model outputs a vector, target_class selects the index to explain.
      n_samples: Number of noisy samples to generate.
      noise_scale: Standard deviation of the noise to add.
      device: The device to run the computation on.

    Returns:
      smoothed_s_gradients: A dictionary with smoothed gradients for each S-space layer.
      classifier_output: Output logits from classifier (from clean input).
      synthesized_image: The generated image (from clean input).
    """
    # Get the clean prediction first
    smoothed_s_gradients, classifier_output, synthesized_image = backpropagation_gradients_s_space(
        synthesis_net, classifier, preprocess, w_latents, target_class, device)

    # Initialize gradients accumulator dictionary
    #smoothed_s_gradients = {}

    # Generate n_samples noisy versions and compute gradients
    for i in range(n_samples-1):
        # Add Gaussian noise to the input
        noise = torch.randn_like(w_latents) * noise_scale
        noisy_w_latents = w_latents + noise

        # Compute gradients for the noisy input
        s_gradients, _, _ = backpropagation_gradients_s_space(
            synthesis_net, classifier, preprocess, noisy_w_latents, target_class, device)

        # Accumulate gradients
        for layer_name, data in s_gradients.items():
            smoothed_s_gradients[layer_name]['grad'] += data['grad']

    # Average the gradients
    for layer_name in smoothed_s_gradients.keys():
        smoothed_s_gradients[layer_name]['grad'] /= n_samples

    return smoothed_s_gradients, classifier_output, synthesized_image

def backpropagation_gradients_s_space_yolo(synthesis_net: torch.nn.Module,
                                      classifier: torch.nn.Module,
                                      #preprocess: torch.nn.Module,
                                      w_latents: torch.Tensor,
                                      target_idx: int,
                                      target_class: int,
                                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Compute gradients of the classifier's output with respect to the input_tensor via backpropagation.

    Parameters:
      synthesis_net: The StyleGAN synthesis network.
      classifier: The classifier network.
      #preprocess: Preprocessing transform for classifier input.
      w_latents: The input tensor requiring explanation; shape: (N, ...).
      target_idx: int
      target_class: int
      device: The device to run the computation on.

    Returns:
      s_gradients: A dictionary with values and gradients for each S-space layer.
      classifier_output: Output logits from the classifier.
      synthesized_image: The generated image corresponding to the W-space.
    """
    synthesis_net = synthesis_net.to(device).eval()
    classifier = classifier.to(device).eval()
    w_latents = w_latents.to(device).detach()

    # Dictionary to store gradients
    s_gradients = {}

    # Define Hook function to capture S vectors and retain gradient computation
    def hook_fn(module, input, output):
        output.requires_grad_(True)  # Ensure S vectors have gradients
        layer_name = module.name_module if hasattr(module, "name_module") else str(module)
        s_gradients[layer_name] = {'values': output, 'grad': None}

    # Register hooks for all affine layers in the synthesis network
    hooks = []
    for name, module in synthesis_net.named_modules():
        if "affine" in name:  # Assuming S-space corresponds to affine layers
            module.name_module = name
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass to generate synthetic image
    # with torch.no_grad():
    img = synthesis_net(w_latents)  # Generate image from W space
    img = (img.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]

    # Classifier preprocessing & prediction
    #img_clf = preprocess(img.squeeze(0)).unsqueeze(0).to(device)
    img_clf = img.squeeze(0).unsqueeze(0).to(device)
    classifier_output = classifier(img_clf)
    classifier_output = classifier_output[0].squeeze(0).T
    # Specify the target class for backpropagation
    target_output = classifier_output[target_idx][4+target_class]

    # Backpropagation to compute gradients
    grad_outputs = torch.ones_like(target_output)
    target_output.backward(grad_outputs)

    # Extract gradients for each S layer (updated logic)
    for layer_name, data in s_gradients.items():
        if data['values'].grad is None:
            raise RuntimeError(
                f"Gradient for layer '{layer_name}' is None. "
                f"Ensure that 'data['values']' has 'requires_grad=True' and "
                f"that the computation graph is intact."
            )

        # Safeguard: Ensure gradients exist before cloning
        s_gradients[layer_name]['grad'] = data['values'].grad.clone()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Return gradients, classifier output, and synthesized image
    return s_gradients, classifier_output, img


def generate_image_with_s_latents(synthesis_net: torch.nn.Module, s_latents: dict, device=torch.device('cuda')):
    """
    Generate an image using pre-defined S-space latents.

    Args:
        synthesis_net (torch.nn.Module): The StyleGAN synthesis network.
        s_latents (dict): Dictionary containing S-space latents for each layer.
                          Format: {'layer_name': torch.Tensor, ...}, where the tensors are the affine outputs.
        device (torch.device): Device for computation.

    Returns:
        torch.Tensor: The generated image.
    """
    synthesis_net = synthesis_net.to(device).eval()

    # Define a hook function to overwrite S-space activations
    def overwrite_s_latents(module, input, output):
        layer_name = module.name_module if hasattr(module, "name_module") else str(module)
        if layer_name in s_latents.keys():
            return s_latents[layer_name]['values']
        return output

    # Register hooks on affine layers of the synthesis network
    hooks = []
    for name, module in synthesis_net.named_modules():
        if "affine" in name:  # Assuming S-space corresponds to affine layers
            module.name_module = name
            hooks.append(module.register_forward_hook(overwrite_s_latents))

    # Generate an image by running the synthesis network
    with torch.no_grad():
        # Use a dummy W latent as input (we bypass it using the hooks)
        num_layers = synthesis_net.num_ws
        dummy_w = torch.zeros(1, num_layers, 512).to(device)  # Shape depends on the StyleGAN config
        img = synthesis_net(dummy_w)

    # Normalize the image to [0, 1]
    img = (img.clamp(-1, 1) + 1) / 2

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return img

def integrated_gradients(synthesis_net: torch.nn.Module,
                         classifier: torch.nn.Module,
                         preprocess: torch.nn.Module,
                         input_tensor: torch.Tensor,
                         baseline=None,
                         target_class=None,
                         steps=50,
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Compute the Integrated Gradients for a given input tensor.

    Parameters:
      synthesis_net: The StyleGAN synthesis network.
      classifier: The classifier network.
      preprocess: Preprocessing transform for classifier input.
      input_tensor: The input tensor for the synthesis network (e.g., latent vector), requiring explanation.
      baseline: The baseline tensor (must have the same shape as input_tensor). If None, a tensor of zeros is used.
      target_class: int or None. If the model outputs a vector, target_class selects the output index to explain.
      steps: The number of interpolation steps between baseline and input_tensor.
      device: Device to use for computation.

    Returns:
      integrated_grad: A tensor of the same shape as input_tensor representing the integrated gradients.
    """
    # Move the input_tensor to the specified device
    input_tensor = input_tensor.to(device)

    # If baseline is not provided, use a tensor of zeros as the baseline
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)
    else:
        baseline = baseline.to(device)

    # Calculate the difference between the input and the baseline
    diff = input_tensor - baseline

    # Construct the interpolation path between baseline and input_tensor
    scaled_inputs = [baseline + (float(i) / steps) * diff for i in range(steps + 1)]
    # Concatenate the interpolated tensors along a new batch dimension, shape: (steps+1, *input_tensor.shape)
    scaled_inputs = torch.cat([x.unsqueeze(0) for x in scaled_inputs], dim=0)

    # Enable gradient computation for the interpolated inputs
    scaled_inputs.requires_grad = True

    scaled_inputs = scaled_inputs.squeeze()
    print("--------------------------------------------------------")
    print(scaled_inputs.shape)
    # Forward pass through synthesis network and classifier for all scaled inputs
    synthesized_images = synthesis_net(scaled_inputs)  # Generate images
    synthesized_images = (synthesized_images.clamp(-1, 1) + 1) / 2  # Normalize image output

    if device.type == 'cuda':
        processed_images = preprocess(synthesized_images).cuda()
    else:
        processed_images = preprocess(synthesized_images)
    outputs = classifier(processed_images)

    if target_class is not None:
        # If model outputs a vector (e.g., (steps+1, num_classes)), select the target_class for each sample
        outputs = outputs[:, target_class]
    else:
        # If the model output is a scalar, squeeze the output to get (steps+1,)
        outputs = outputs.squeeze()

    # Backward pass: compute gradients of the outputs with respect to the interpolated inputs
    grads = torch.autograd.grad(outputs, scaled_inputs,
                                grad_outputs=torch.ones_like(outputs),
                                create_graph=False, retain_graph=False)[0]

    # Approximate the integral using the trapezoidal rule:
    # Calculate the average gradients along the interpolation path;
    # note that the first and last gradients each have half weight.
    grads = grads.detach()
    avg_grads = (grads[0] + grads[-1]) / 2.0
    if steps > 1:
        avg_grads += grads[1:-1].sum(dim=0)
        avg_grads /= steps

    # Calculate the integrated gradients:
    # Integrated gradients = (input_tensor - baseline) * average gradient
    integrated_grad = diff * avg_grads
    return integrated_grad


def aggregate_gradients(gradients,
                        option: str = 'mean'):
    assert gradients.ndim == 3
    assert gradients.size(2) == 512
    if option == 'l2':
        aggregated_gradients = torch.norm(gradients, dim=2)
    elif option == 'l1':
        aggregated_gradients = torch.abs(gradients).mean(dim=2)
    elif option == 'mean':
        aggregated_gradients = torch.mean(gradients, dim=2)
    elif option == 'max':
        aggregated_gradients = torch.max(gradients, dim=2)[0]
    elif option == 'sum':
        aggregated_gradients = torch.sum(gradients, dim=2)
    return aggregated_gradients.squeeze().cpu().numpy()
