import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import torch
from numpy.typing import NDArray
from typing import Union, Any


def softmax(x: Union[list, NDArray]) -> NDArray:
    """Apply a softmax operation on a list or numpy array."""
    if isinstance(x, list):
        x = np.array(x)
    return (np.e ** x) / np.sum(np.e ** x)

def transform_image(x: Union[str, NDArray]) -> NDArray:
    """Transform an image into standard config."""
    if isinstance(x, str):
        x = literal_eval(x)
    x = np.array(x).squeeze()
    x = x.transpose(1, 2, 0)
    return x

def to_tensor(x: str) -> torch.Tensor:
    """Convert a string to tensor."""
    x = np.array(literal_eval(x))
    return torch.Tensor(x)

def load_and_combine_dfs(path: str, filters: list[str]) -> pd.DataFrame:
    """
    Load and combined experiment dfs from a path.

    :param path: Path to experiment dfs.
    :param filters: Filers to select dfs.
    :returns: Combined experiment dfs.
    """
    maybe_slash = "" if path[-1] == "/" else "/"
    all_dfs = glob(f"{path}{maybe_slash}*.csv")
    if len(all_dfs) == 0:
        raise FileNotFoundError(f"No dfs in directory {path}")
    combined_df = pd.DataFrame()
    for elem in all_dfs:
        if all(kw in elem for kw in filters):
            tmp_df = pd.read_csv(elem)
            combined_df = pd.concat([tmp_df, combined_df], ignore_index=True)
            cols = tmp_df.columns

    combined_df.columns = cols
    return combined_df

def get_tsne_from_values(*values: list[pd.Series], components: int = 2,  random_state: int = 0) -> list[Any]:
    """Combine multiple data points and compute TSNE."""
    tsne = TSNE(n_components=components, random_state=random_state)

    values = [v.to_list() for v in values]  # Concert pd.Series to list
    cl = [0] + [len(v) for v in values]  # Component lengths for extraction
    total = np.vstack([np.array(v) for v in values])

    emb_total = tsne.fit_transform(total)
    return [emb_total[cl[i]:cl[i]+cl[i+1]] for i in range(len(cl)-1)]

def filter_for_classes(*elements: list[pd.Series], class_information: pd.Series, classes: list[int],
                       filter_class_information: bool = True) -> list[pd.Series]:
    """
    Filter elements based on classes, last element is used to filter.

    :param elements: Elements to filter.
    :param class_information: Classes information for elements.
    :param classes: Classes to filter for.
    :param filter_class_information: Whether to filter class information elements aswell.
    :returns: The filtered elements.
    """
    assert all((len(e) == len(class_information) for e in elements)), "Error, series provided have different lengths."
    mask = class_information.isin(classes)

    elements = [e[mask] for e in elements]
    elements = elements + [class_information[mask]] if filter_class_information else elements
    return elements


"""Plotting Functions."""

def plot_compare_images_with_confidences(im1, im2, y, yp) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    classes = range(len(y))
    ax[1, 0].bar(classes,y)
    ax[1, 1].bar(classes, yp)
    plt.show()

def plot_compare_image_differences(im1, im2) -> None:
    diff = im1 - im2

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(diff[:, :, 0], cmap="seismic", vmin=-1, vmax=1)
    ax[1].imshow(diff[:, :, 1], cmap="seismic", vmin=-1, vmax=1)
    h = ax[2].imshow(diff[:, :, 2], cmap="seismic", vmin=-1, vmax=1)
    fig.colorbar(h)
    plt.show()

def plot_manifold_analysis(
        emb_dataset,
        emb_orig,
        emb_targets,
        dataset_classes: list[int],
        orig_classes: list[int],
        targets_classes: list[int],
        cmap,
        mask_thresh: float = 0.02,
        show_arrows: bool = False,
) -> None:
    """
    Plot manifold analysis using.

    :param emb_dataset: Embedded dataset points.
    :param emb_orig: Embedded seed for optimization.
    :param emb_targets: Embedded results from optimization.
    :param dataset_classes: List of classes for points in dataset.
    :param orig_classes: List of classes for points in original dataset.
    :param targets_classes: List of classes for points in generated targets.
    :param mask_thresh: Mask threshold for density maps.
    :param show_arrows: Whether to show arrows of data transformations.
    """
    print("Plotting... This takes some time :)")
    x,y = emb_dataset.max(axis=0)
    x,y = int(x*1.1), int(y*1.1)

    xx, yy = np.meshgrid(np.linspace(-x, x, 300), np.linspace(-y, y, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    dataset_classes = np.array(dataset_classes)
    unique_classes = np.unique(dataset_classes)

    """Generate density maps for each class and standardize to [0,1]."""
    density_maps = {}
    for cls in unique_classes:
        class_points = emb_dataset[dataset_classes == cls]
        kde = gaussian_kde(class_points.T)
        density = kde(grid_points.T)
        density /= density.max()
        density_maps[cls] = density.reshape(xx.shape)

    combined_density = np.zeros(xx.shape)
    dominant_class = np.full(xx.shape, -1)

    """Combine density masks such that dominant density is visible at single point."""
    for cls, density in density_maps.items():
        mask = density > combined_density
        combined_density[mask] = density[mask]
        dominant_class[mask] = cls

    """Filter density masks to allow for unmasked regions."""
    filtered = dominant_class.copy()
    filtered[combined_density < mask_thresh] = -1
    filtered = np.ma.masked_where(filtered == -1, filtered)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.contourf(xx, yy, filtered, cmap=cmap, alpha=0.3, levels=np.arange(-1, len(unique_classes)))
    ax.scatter(x=emb_dataset[:, 0], y=emb_dataset[:, 1], c=dataset_classes, cmap=cmap, alpha=0.1, s=1)
    ax.scatter(x=emb_orig[:, 0], y=emb_orig[:, 1], c=orig_classes, cmap=cmap, s=15, marker="x")
    ax.scatter(x=emb_targets[:, 0], y=emb_targets[:, 1], c=targets_classes, cmap=cmap, s=30, marker="*")
    if show_arrows:
        for source, target in zip(emb_orig, emb_targets):
            ax.annotate("", target, source, arrowprops=dict(arrowstyle="->"))
    plt.show()
