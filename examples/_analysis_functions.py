import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from scipy.sparse.csgraph import laplacian
import torch
from numpy.typing import NDArray
from typing import Union, Any

from src.criteria.image_comparison import CFrobeniusDistance, SSIMD2


class ExperimentMetrics:
    """A class to extract metrics from df."""
    image_distance: list[float]  # Frobenius Distance between images.
    boundary_distance: list[float]  # Euclidean Distance towards boundary.
    ssim: list[float]  # SSIM between images.
    lap_variance: list[float]  # Laplacian variance on image differences.

    escape_ratios: list[float]  # Escape ratio of boundary search.
    coverage: list[float]  # Boundary coverage metric.

    def __init__(self, df:pd.DataFrame, xs: list[str], ys: list[str]) -> None:
        """
        Initialize the object and extract metrics from df.

        :param df: The dataframe to extract metrics from.
        :param xs: The names of the x columns to extract.
        :param ys: The names of the y columns to extract.
        """
        self.df = format_cols(df.copy(), reduce_channels=True)

        """Extract image distances."""
        im_comp = CFrobeniusDistance()._frob
        diffs = [(self.df["X"] - self.df[x]) for x in xs]
        f_dist = [d.apply(im_comp) for d in diffs]
        f_dist = self._apply_strategy(pd.DataFrame(f_dist), "min")
        self.image_distance = f_dist.tolist()

        """Extract SSIM between images."""
        #ssim = SSIMD2()

        #ssims = [df[["X", x]].apply(lambda row: ssim._ssim_d2(row.iloc[0][:,:, None], row.iloc[1][:,:, None]), axis=1) for x in xs]
        #ssims = self._apply_strategy(pd.DataFrame(ssims), "max")
        #self.ssim = ssims.tolist()

        """Extract boundary distance."""
        b_dists = [df[y].apply(distance_to_boundary) for y in ys]
        b_dists = self._apply_strategy(pd.DataFrame(b_dists), "min")
        self.boundary_distance = b_dists.tolist()

        """Extract laplacian variance."""
        lap_var = [d.apply(laplacian_variance) for d in diffs]
        lap_var = self._apply_strategy(pd.DataFrame(lap_var), "max")
        self.lap_variance = lap_var.tolist()

        """Get boundary stats."""
        cov_esc = [get_boundary_stats(df["y"], df[y]) for y in ys]
        cov, esc = tuple(zip(*cov_esc))

        esc = self._apply_strategy(pd.DataFrame(esc), "min")
        self.escape_ratios = esc.tolist()

        cov = self._apply_strategy(pd.DataFrame(cov), "max")
        self.coverage = cov.tolist()


    def _apply_strategy(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Apply strategy for merging df columns."""
        if strategy == "min":
            return df.min(axis=0)
        elif strategy == "max":
            return df.min(axis=0)
        elif strategy == "mean":
            return df.mean(axis=0)
        else:
            raise NotImplementedError(f"Strategy {strategy} not implemented.")



def softmax(x: Union[list, NDArray]) -> NDArray:
    """Apply a softmax operation on a list or numpy array."""
    if isinstance(x, list):
        x = np.array(x)
    return (np.e**x) / np.sum(np.e**x)


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
        split_file = elem.split(".")[0].split("_")
        res_size = 4 if "wrn" in split_file else 3
        if all((kw in split_file for kw in filters)) and (
            len(split_file) - len(filters) == res_size
        ):
            tmp_df = pd.read_csv(elem)
            combined_df = pd.concat([tmp_df, combined_df], ignore_index=True)
            cols = tmp_df.columns

    combined_df.columns = cols
    return combined_df


def get_tsne_from_values(
    *values: list[pd.Series], components: int = 2, random_state: int = 0
) -> list[Any]:
    """Combine multiple data points and compute TSNE."""
    tsne = TSNE(n_components=components, random_state=random_state)

    values = [v.to_list() for v in values]  # Concert pd.Series to list
    cl = [0] + [len(v) for v in values]  # Component lengths for extraction
    total = np.vstack([np.array(v) for v in values])

    emb_total = tsne.fit_transform(total)
    return [emb_total[cl[i] : cl[i] + cl[i + 1]] for i in range(len(cl) - 1)]


def filter_for_classes(
    *elements: list[pd.Series],
    class_information: pd.Series,
    classes: list[int],
    filter_class_information: bool = True,
) -> list[pd.Series]:
    """
    Filter elements based on classes, last element is used to filter.

    :param elements: Elements to filter.
    :param class_information: Classes information for elements.
    :param classes: Classes to filter for.
    :param filter_class_information: Whether to filter class information elements aswell.
    :returns: The filtered elements.
    """
    assert all(
        (len(e) == len(class_information) for e in elements)
    ), "Error, series provided have different lengths."
    mask = class_information.isin(classes)

    elements = [e[mask] for e in elements]
    elements = elements + [class_information[mask]] if filter_class_information else elements
    return elements


def get_boundary_stats(y1: pd.Series, y2: pd.Series) -> tuple[NDArray, float]:
    """
    Get boundary coverage measures based on Kolmogorov-Smirnov distance.

    Get boundary vicinity.

    :param y1: First series of confidence values.
    :param y2: Second series of confidence values.
    :returns: Mean boundary coverage across classes and std.
    """
    boundary_distribution = {i: [] for i in range(10)}
    escaped_boundaries = {i: [] for i in range(10)}
    esc = 0
    for y, yp in zip(y1, y2):
        yp = np.array(yp)
        label = np.argmax(y)
        boundary_location = yp.argsort()[::-1][:2]
        if label not in boundary_location:
            escaped_boundaries[label].append(boundary_location)
            esc += 1
        else:
            bl = list(boundary_location)
            bl.remove(label)
            boundary_distribution[label] += bl

    unif = np.full(9, 1 / 9)
    distances = []
    for label, dist in boundary_distribution.items():
        hist, _ = np.histogram(dist, bins=10, range=(0, 10))
        hist = np.delete(hist, label)
        hist = hist / 10

        dist = 0 if hist.sum() == 0 else (2 * np.sum(np.minimum(unif, hist)) / 2) * hist.sum()

        distances.append(dist)
    distances = np.array(distances)
    return distances, esc / len(y1)


def laplacian_variance(arr: NDArray) -> float:
    """
    Calculate the laplacian variance.

    :param arr: Array to calculate laplacian variance for.
    :returns: The variance value.
    """
    arr = arr.squeeze()
    if len(arr.shape) == 3:
        arr = arr.sum(axis=np.argmin(arr.shape))

    filtered = laplacian(arr)
    return filtered.var()


def reduce_dim(arr: NDArray, reduce_channels: bool) -> NDArray:
    arr = arr.squeeze()
    shape = arr.shape
    if len(shape) == 3 and reduce_channels:
        arr = arr.sum(axis=np.argmin(shape)) / min(shape)
    return arr


def format_cols(df: pd.DataFrame, reduce_channels: bool = False) -> None:
    for c in df.columns:
        if "X" in c:
            df[c] = df[c].apply(lambda x: reduce_dim(np.array(literal_eval(x)), reduce_channels))
        if "y" in c:
            df[c] = df[c].apply(lambda x: np.array(literal_eval(x)))
    return df

def distance_to_boundary(arr: NDArray) -> float:
    arr = arr.squeeze()
    boundary_indices = np.argsort(arr)[::-1][:2]
    ideal = np.zeros_like(arr)
    ideal[boundary_indices] = 0.5
    return np.linalg.norm(ideal - arr)


"""Plotting Functions."""


def plot_compare_images_with_confidences(im1, im2, y, yp) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    classes = range(len(y))
    ax[1, 0].bar(classes, y, color="darkgreen")
    ax[1, 1].bar(classes, yp, color="darkgreen")

    for i in range(2):
        ax[0, i].axis('off')

        ax[1, i].grid()
        ax[1, i].set_ylim([0,1])
        ax[1, i].set_xticks(range(10))
        ax[1, i].set_xlabel("Classes")
        ax[1, i].set_xticklabels(range(10))
    ax[1,0].set_ylabel("SUT Confidence")
    ax[1,1].set_yticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0)
    plt.show()


def plot_compare_image_differences(im1, im2, greyscale: bool = False) -> None:
    diff = im1 - im2

    h = 1 if greyscale else 3
    diff = diff.sum(axis=-1, keepdims=True) if greyscale else diff

    fig, ax = plt.subplots(1, h+1, figsize=(4*h+0.2, 4), gridspec_kw={'width_ratios': [1]*h + [0.05]})
    for i in range(h):
        h = ax[i].imshow(diff[:, :, i], cmap="seismic", vmin=-1, vmax=1)
        ax[i].axis("off")

    cbar = plt.colorbar(h, cax=ax[-1])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.05)
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
    x, y = emb_dataset.max(axis=0)
    x, y = int(x * 1.1), int(y * 1.1)

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
    ax.scatter(
        x=emb_dataset[:, 0], y=emb_dataset[:, 1], c=dataset_classes, cmap=cmap, alpha=0.1, s=1
    )
    ax.scatter(x=emb_orig[:, 0], y=emb_orig[:, 1], c=orig_classes, cmap=cmap, s=15, marker="x")
    ax.scatter(
        x=emb_targets[:, 0], y=emb_targets[:, 1], c=targets_classes, cmap=cmap, s=30, marker="*"
    )
    if show_arrows:
        for source, target in zip(emb_orig, emb_targets):
            ax.annotate("", target, source, arrowprops=dict(arrowstyle="->"))
    plt.show()


def plot_measures(exps: list[str], dfs: list[list[pd.DataFrame]]) -> None:
    num_methods = len(dfs)

    for i, exp in enumerate(exps):
        target_dfs = [df[i] for df in dfs]

        image_dists = []
        boundary_dists = []
        for df in target_dfs:
            format_cols(df)
            x = df["X"]
            xps_cols = [c for c in df.columns if ("X" in c) and ("p" in c)]
            res = pd.Series(0, index=x.index)
            for c in xps_cols:
                res += x - df[c].apply(CFrobeniusDistance._frob)
            res /= len(xps_cols)
            image_dists.append(res)
    pass
