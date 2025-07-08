import torch
import os
import json
from torch import nn
from collections import OrderedDict
import numpy as np


def tensor_to_ndarray(img_tensor):
    if img_tensor.min()<-0.1:
        img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2  # normalize to [0, 1] ss

    img = img_tensor.cpu().detach().numpy()[0]
    img = img.transpose(1, 2, 0)
    return img

# Add a helper function to handle JSON serialization of numpy types
def convert_to_serializable(obj):
    """
    Recursively convert numpy types in a dictionary to Python native types.
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.float32, np.float64)):  # Handle numpy float types
        return float(obj)  # Convert numpy float to native Python float
    elif isinstance(obj, (np.int32, np.int64)):  # Handle numpy integer types
        return int(obj)  # Convert numpy int to native Python int
    return obj  # Return the object unchanged if it doesn't match any special types

def remove_prefix_from_state_dict(state_dict, prefix='module.'):
    """
    Remove a prefix (e.g., 'module.') from all keys in a PyTorch state_dict.

    Args:
        state_dict (dict): The original state_dict from a saved model.
        prefix (str): The prefix string to remove from the beginning of keys.

    Returns:
        OrderedDict: A new state_dict with the prefix removed from keys.
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
    """code to remove prefix """
    """from utils import remove_prefix_from_state_dict
    input_path = sut_facial_path    # 
    output_path = sut_facial_path.replace('resnet_celeb_40','resnet_celeb_40_single')    
    
    # load parallel model
    state_dict = torch.load(input_path, map_location='cpu')
    
    # remove 'module.' prefix
    clean_state_dict = remove_prefix_from_state_dict(state_dict)
    
    # save new file
    torch.save(clean_state_dict, output_path)"""

def load_generator(gan_ckpt_path, device):
    from src.manipulator._style_gan_manipulator.legacy import load_network_pkl
    with open(gan_ckpt_path, 'rb') as f:
        return load_network_pkl(f)['G_ema'].to(device)

def load_facial_classifier(ckpt_path, device):
    from local_models.classifiers.celeb_resnet_model import AttributeClassifier

    model = AttributeClassifier()
    model = model.to(device)
    # model = torch.nn.DataParallel(model)  # covert to DataParallel
    model.load_state_dict(torch.load(ckpt_path, weights_only=False))
    model.eval()
    return model

def load_facial_large_classifier(ckpt_path, device):
    from local_models.classifiers.celeb_swag_model import SWAGCelebAClassifier

    model = SWAGCelebAClassifier()
    model = model.to(device)
    # model = torch.nn.DataParallel(model)  # covert to DataParallel
    model.load_state_dict(torch.load(ckpt_path, weights_only=False))
    model.eval()
    return model

def load_animal_classifier(device):
    import torchvision.models as models
    return models.resnet18(pretrained=True).eval().to(device)

def load_rexnet_dog_classifier(ckpt_path, device):
    import timm
    model_name = 'rexnet_150'
    num_classes = 78
    model = timm.create_model(model_name=model_name, num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model

def load_data(data_path, target_class_id, sample_id):
    """
    Load all gradient values across the population and normalize them.

    Args:
        data_path (str): Path to the data directory.
        target_class_id (int): List of target class IDs (e.g., [205, 206...]).
        sample_id (int): Number of samples per class.
    Returns:
        tuple: A tuple containing (gradient dict, metadata dict) for the requested sample. Returns (None, None) if files not found.
    """

    # Construct the path to the gradient file
    gradient_path = os.path.join(data_path, str(target_class_id), str(sample_id), f'gradient_{sample_id}.pt')
    
    # Load the .pt file
    if not os.path.isfile(gradient_path):
        print(f"Warning: {gradient_path} not found, skipping.")
        gradient = None
    else:    
        gradient = torch.load(gradient_path, weights_only=True)

    # load predictions
    metadata_path = os.path.join(data_path, str(target_class_id), str(sample_id), f'metadata_{sample_id}.json')
    if not os.path.isfile(metadata_path):
        print(f"Warning: {metadata_path} not found, skipping.")
        metadata = None
    else:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    return gradient, metadata


def rank_gradient_info(s_latents, top = 5, layer_name=None):
    """
    Find and rank the indices of the gradients based on their absolute values.

    Args:
        s_latents (dict): Dictionary containing S-space latents for each layer.
                          Format: {'layer_name': {'grad': torch.Tensor}, ...},
                          where the tensors are the affine outputs with gradient values.
        top (int, optional): Number of top gradients to return for each layer.
                                If None, return all gradients.
        layer_name (str or list, optional): Name of the layer(s) to rank gradients.
                                            If None or "all", includes all layers.
                                            If a string, uses that single layer.
                                            If a list of strings, uses the specified layers.

    Returns:
        dict or list: Dictionary for all layers or list for a single layer:
                      - 'ranked_indices': Ranked indices of gradients based on their absolute values.
                      - 'gradient_raw': Corresponding raw gradient values.
    """
    def process_gradients(layer_data, top=None):
        if 'grad' not in layer_data:
            raise ValueError(f"'grad' key missing in layer data.")
        grad = layer_data['grad']
        grad_abs = torch.abs(grad)  # Absolute values for ranking
        ranked_indices = torch.argsort(grad_abs, descending=True).squeeze().tolist()
        gradient_raw = grad.view(-1)[ranked_indices].tolist()

        # Truncate to the top gradients if specified
        if top is not None:
            ranked_indices = ranked_indices[:top]
            gradient_raw = gradient_raw[:top]

        return ranked_indices, gradient_raw

    if layer_name is None or layer_name == 'all':
        # Handle the case for all layers
        gradient_info_by_layer = {}
        for layer_name, layer_data in s_latents.items():
            ranked_indices, gradient_raw = process_gradients(layer_data, top)
            gradient_info_by_layer[layer_name] = {
                'ranked_indices': ranked_indices,
                'gradients': gradient_raw
            }
        return gradient_info_by_layer

    elif isinstance(layer_name, str):
        # Handle ranking for a single layer
        if layer_name not in s_latents:
            raise ValueError(f"Layer name '{layer_name}' not found in s_latents.")
        ranked_indices, gradient_raw = process_gradients(s_latents[layer_name], top)
        return {'ranked_indices': ranked_indices, 'gradients': gradient_raw}

    elif isinstance(layer_name, list):
        # Handle ranking for specified layers
        gradient_info_by_layer = {}
        for layer in layer_name:
            if layer not in s_latents:
                raise ValueError(f"Layer name '{layer}' not found in s_latents.")
            ranked_indices, gradient_raw = process_gradients(s_latents[layer], top)
            gradient_info_by_layer[layer] = {
                'ranked_indices': ranked_indices,
                'gradients': gradient_raw
            }
        return gradient_info_by_layer
    else:
        raise TypeError("layer_name must be None, 'all', a string, or a list of strings.")


def perturbate_s_latents(s_latent, layer_name, index, extent=0.1):
    """
    Perturbate the S-space latents based on the given layer and index.

    Args:
        s_latent (dict): Dictionary containing S-space latents for each layer.
                          Format: {'layer_name': {'values': torch.Tensor}, ...},
                          where the tensors are the affine outputs with gradient values.
        layer_name (str): Name of the layer to perturbate.
        index (int): Index of the gradient to perturbate.
    Returns:
        dict: Dictionary containing perturbed S-space latents for each layer.
    """
    from copy import deepcopy
    s_latent_perturbed = deepcopy(s_latent)

    # Ensure the layer exists in s_latent
    if layer_name not in s_latent:
        raise ValueError(f"Layer '{layer_name}' not found in s_latent.")

    # Ensure the layer contains values
    if 'values' not in s_latent[layer_name]:
        raise ValueError(f"'values' key missing in layer '{layer_name}'.")

    # Ensure the specified index is valid
    if index >= s_latent[layer_name]['values'].numel():
        raise IndexError(f"Index {index} is out of range for the layer '{layer_name}'.")

    with torch.no_grad():
        tensor = s_latent_perturbed[layer_name]['values']
        flat_tensor = tensor.view(-1)  # Flatten the tensor
        flat_tensor[index] = (flat_tensor[index] + extent).detach()  # Modify the specified index
        s_latent_perturbed[layer_name]['values'] = flat_tensor.view(tensor.size())  # Reshape back

    return s_latent_perturbed


def predict_yolo(model, img_tensor, device, target_class):
    model.to(device).eval()

    prediction = model(img_tensor)[0].squeeze(0).T
    class_probs = prediction[:, 4:]                   # shape: [5376, 79]
    confidences = class_probs.max(1).values  # shape: [5376]

    top_idx = int(confidences.argmax())
    top_class = int(class_probs[top_idx].argmax())
    boxes_xywh = prediction[top_idx, :4].cpu().detach().numpy()

    top_confidence = prediction[top_idx, 4+top_class].cpu().detach().numpy()
    target_confidence = prediction[:, 4+target_class].max().cpu().detach().numpy()
    return top_idx, top_class, boxes_xywh, top_confidence, target_confidence

if __name__ == "__main__":
    print("--------------------------------------------------------")
    os.chdir('..')
    sut_facial_lagre_path = os.path.join('local_models/classifiers','checkpoints','swag_celeb_40_parallel.pth')
    input_path = sut_facial_lagre_path    #
    output_path = sut_facial_lagre_path.replace('_parallel','_single')

    # load parallel model
    state_dict = torch.load(input_path, map_location='cpu')

    # remove 'module.' prefix
    clean_state_dict = remove_prefix_from_state_dict(state_dict)

    # save new file
    torch.save(clean_state_dict, output_path)
    print(f"model saved in {output_path}")