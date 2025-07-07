""" Reproduce Spectral Relevance Analysis (SpRAy) based on the description in paper
 *Unmasking Clever Hans Predictors and Assessing What Machines Really Learn*
#"""

import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
import pandas as pd

from configs import sut_facial_path, gan_facial_ckpt_path
from src_.utils import load_facial_classifier, load_generator

from local_models.classifiers.celeb_resnet_model import CelebADataset
from local_models.classifiers.celeb_configs import dataset_path

# Step 1: Generate Relevance Maps using LRP
def generate_relevance_maps(model, img_tensor, target_logit, composite_config):
    """
    Generate relevance map for input image tensor using LRP.

    Parameters:
        model: Trained neural network model.
        img_tensor: input in tensor form
        composite_config: Configuration for the LRP composite.

    Returns:
        relevance map as numpy arrays.
    """

    # Configure the Canonizers and Composite for LRP
    canonizers = [SequentialMergeBatchNorm()]  # Merge BatchNorm with linear layers
    composite = EpsilonGammaBox(
        low=composite_config.get('low', -3.0),
        high=composite_config.get('high', 3.0),
        canonizers=canonizers
    )
    with Gradient(model=model, composite=composite) as attributor:

        target_tensor = torch.zeros(1, 40, device=img_tensor.device)  # Adjust target size
        target_tensor[0, target_logit] = 1  # Example: set logit target for class 15

        _, relevance = attributor(img_tensor, target_tensor)
        relevance_np = relevance.cpu().detach().numpy()[0]
    return relevance_np


# Step 2: Preprocess Relevance Maps
def preprocess_relevance_maps(relevance_maps, target_size=(32, 32)):
    """
    Downsize and normalize relevance maps to a uniform shape and size.

    Parameters:
        relevance_maps: List of raw relevance maps (as numpy arrays).
        target_size: Desired output size for relevance maps.

    Returns:
        Processed relevance maps as a 2D array where each map is flattened.
    """
    from skimage.transform import resize
    processed_maps = []
    for relevance_map in relevance_maps:
        # Reshape relevance map from (C, H, W) to (H, W, C)
        relevance_map = np.transpose(relevance_map, (1, 2, 0))
        # Normalize relevance map to [0, 1]
        normalized_map = (
                                 relevance_map - relevance_map.min()) / (
                                     relevance_map.max() - relevance_map.min() + 1e-8)
        # Resize relevance map to the target size
        resized_map = resize(normalized_map, target_size, anti_aliasing=True)
        # Flatten the resized map into a 1D array
        processed_maps.append(resized_map.flatten())
    return np.array(processed_maps)


# Step 3: Perform Spectral Clustering
def perform_spectral_clustering(processed_maps, num_clusters=5):
    """
    Apply spectral clustering to the processed relevance maps.

    Parameters:
        processed_maps: 2D array of processed relevance maps.
        num_clusters: Number of clusters to form.

    Returns:
        Array of cluster labels for each relevance map.
    """
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
    labels = clustering.fit_predict(processed_maps)
    return labels


# Step 5: Visualize Cluster Results (Optional)
def visualize_clusters(processed_maps, labels, num_samples=1000):
    """
    Visualize the clustering results using t-SNE.

    Parameters:
        processed_maps: Processed relevance maps as input to t-SNE.
        labels: Cluster labels for each relevance map.
        num_samples: Number of samples to visualize.

    Returns:
        None. Displays the t-SNE plot.
    """
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_maps = tsne.fit_transform(processed_maps[:num_samples])  # Reduce to 2D
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        plt.scatter(
            reduced_maps[labels[:num_samples] == label, 0],
            reduced_maps[labels[:num_samples] == label, 1],
            label=f'Cluster {label}'
        )
    plt.legend()
    plt.title("t-SNE Clustering Visualization of Relevance Maps")
    plt.show()


def save_cluster_info(image_ids, labels, output_file="cluster_info.csv"):
    """
    Saves the cluster information to a CSV file.

    Parameters:
        image_ids (list): List of IDs corresponding to the images.
        labels (list): List of cluster labels for each image.
        output_file (str): Path to save the CSV file.

    Returns:
        None
    """
    # Combine image IDs and labels into a DataFrame
    cluster_data = pd.DataFrame({
        "Image_ID": image_ids,
        "Cluster_Label": labels
    })

    # Save the DataFrame to a CSV file
    cluster_data.to_csv(output_file, index=False)
    print(f"Cluster information saved to: {output_file}")

# Main function that reproduces SpRAy pipeline
def main(data_source):
    """
    Main function to reproduce SpRAy as described in the paper.
    Includes steps: LRP generation (relevance maps), preprocessing,
    clustering, and visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(gan_facial_ckpt_path, device)
    sut = load_facial_classifier(sut_facial_path, device)

    # Step 1: Generate relevance maps for the dataset
    relevance_maps = []
    image_ids = []
    if data_source == "generated":
        for torch_seed in range(100):
            torch.manual_seed(torch_seed)
            # generate one random seed from z latent space
            z = torch.randn([1, generator.z_dim], device=device)
            img_tensor =  generator(z, c=None, truncation_psi=1, noise_mode='const')
            img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            if sut(img_tensor).squeeze()[15] > 0:
                relevance_np = generate_relevance_maps(sut, img_tensor,
                                                       target_logit=15,
                                                       composite_config={"low": -3.0, "high": 3.0}
                                                       )
                relevance_maps.append(relevance_np)
                image_ids.append(torch_seed)
    elif data_source == "dataset":

        dataset = CelebADataset(root=dataset_path,
                              transform=None,  # use default
                              split='test'
                              )
        for id, (img_tensor, _ )in enumerate(dataset):

            img_tensor = img_tensor.to(device)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            if sut(img_tensor).squeeze()[15] > 0:
                relevance_np = generate_relevance_maps(sut, img_tensor,
                                                       target_logit=15,
                                                       composite_config={"low": -3.0, "high": 3.0}
                                                       )
                relevance_maps.append(relevance_np)
                image_ids.append(id)
    # Step 2: Preprocess relevance maps (normalization and downsizing)
    processed_maps = preprocess_relevance_maps(relevance_maps)

    # Step 3: Perform Spectral Clustering on processed relevance maps
    labels = perform_spectral_clustering(processed_maps, num_clusters=5)

    save_cluster_info(image_ids, labels, output_file="cluster_info.csv")

    # Step 4 (Optional): Visualize clustering with t-SNE
    visualize_clusters(processed_maps, labels)


if __name__ == "__main__":
    main(data_source="dataset")