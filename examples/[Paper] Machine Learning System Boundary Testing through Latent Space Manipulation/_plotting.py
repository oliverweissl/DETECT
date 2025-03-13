from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_compare_images_with_confidences(im1, im2, y, yp, save_as: str = None) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    classes = range(len(y))
    ax[1, 0].bar(classes, y, color="darkgreen")
    ax[1, 1].bar(classes, yp, color="darkgreen")

    for i in range(2):
        ax[0, i].axis("off")

        ax[1, i].grid()
        ax[1, i].set_ylim([0, 1])
        ax[1, i].set_xlim([-0.5, 9.5])
        ax[1, i].set_xticks(range(10))
        ax[1, i].set_xticklabels(range(10))

    ax[1, 0].set_xlabel(r"Pred Classes $X$")
    ax[1, 1].set_xlabel(r"Pred Classes $X'$")
    ax[1, 0].set_ylabel("SUT Confidence")
    ax[1, 1].set_yticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0)
    if save_as is not None:
        plt.savefig(f"{save_as}.pdf", bbox_inches="tight", dpi=200)
    else:
        plt.show()


def plot_compare_images(im1, im2, save_as: str = None) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    ax[0].imshow(im1)
    ax[1].imshow(im2)

    for i in range(2):
        ax[i].axis("off")

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    if save_as is not None:
        plt.savefig(f"{save_as}.pdf", bbox_inches="tight", dpi=200)
    else:
        plt.tight_layout()
        plt.show()


def plot_compare_image_differences(im1, im2, greyscale: bool = False, save_as: str = None) -> None:
    diff = im1 - im2

    h = 1 if greyscale else 3
    diff = diff.sum(axis=-1, keepdims=True) if greyscale else diff

    fig, ax = plt.subplots(
        1,
        h + 1,
        figsize=(4 * h + 0.2, 4),
        gridspec_kw={"width_ratios": [1] * h + [0.05], "wspace": -0.1},
    )
    cspec = "RGB"
    for i in range(h):
        h = ax[i].imshow(diff[:, :, i], cmap="seismic", vmin=-1, vmax=1)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xlabel(cspec[i], fontsize=24)

    cbar = plt.colorbar(h, cax=ax[-1])
    cbar.ax.tick_params(labelsize=24)

    if save_as is not None:
        plt.savefig(f"{save_as}.pdf", bbox_inches="tight", dpi=200)
    else:
        plt.tight_layout()
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


def plot_compare_metric(
    *experiments: list[float],
    exp_labels: list[str],
    tick_labels: list[str],
    name: str,
    log: bool = False,
    save_as: Optional[str] = None,
) -> None:
    """
    Compare metric across experiments and datasets.

    :param experiments: The experiments data for comparison.
    :param exp_labels: The labels for experiments.
    :param tick_labels: The labeles for the x_ticks (i.e Datasets).
    :param name: The name of the metric (used as y_label).
    :param log: Whether to log scale on the y axis.
    :param save_as: Where to save the figure.
    """
    colors = ["green", "brown", "blue"]

    max_len = max([len(e) for e in experiments])
    positions = [[] for _ in range(len(experiments))]
    middle = []

    prev_base, base = 1, 1
    for i in range(max_len):
        for j, exp in enumerate(experiments):
            if len(exp) > i:
                positions[j].append(base)
                base += 0.5
        middle.append((prev_base + base) / 2 - 0.25)
        base, prev_base = base + 1, base + 1

    boxpl_args = dict(widths=0.495, patch_artist=True, medianprops=dict(color="red"))
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, exp in enumerate(experiments):
        ax.boxplot(
            exp,
            positions=positions[i],
            boxprops=dict(facecolor=colors[i], color=colors[i]),
            **boxpl_args,
        )
        ax.plot([], color=colors[i], label=exp_labels[i], linewidth=8)

    ax.set_ylabel(name)
    ax.set_xticks(middle)
    ax.set_xticklabels(tick_labels, rotation=25)
    if log:
        ax.set_yscale("log")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax.grid()
    if save_as:
        plt.savefig(f"{save_as}.pdf", bbox_inches="tight")
    else:
        plt.show()
