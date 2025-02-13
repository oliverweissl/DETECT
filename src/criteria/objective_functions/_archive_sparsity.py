from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from .._criterion import Criterion


class ArchiveSparsity(Criterion):
    """Implements sparsity measure wrt an Archive."""

    _name: str = "ArchiveSparsity"
    _head: Callable
    _on_genomes: bool

    metric: Criterion

    def __init__(self, metric: Criterion, regime: str = "min", on_genomes: bool = False) -> None:
        """
        Initialize the Archive Sparsity measure.

        :param metric: The metric used in the measure calculation.
        :param regime: How to decide on the metric value.
        :param on_genomes: If true, the metric value will be computed on genomes.
        :raises NotImplementedError: If the regime parsed is not implemented.
        """
        super().__init__()
        self._on_genomes = on_genomes
        self.metric = metric
        if regime == "min":
            self._head = np.min
        elif regime == "max":
            self._head = np.max
        elif regime == "mean":
            self._head = np.mean
        else:
            raise NotImplementedError(f"Regime {regime} not implemented.")

    def evaluate(
        self,
        *,
        images: list[Tensor],
        solution_archive: list[Tensor],
        genome_target: NDArray,
        genome_archive: list[NDArray],
        **_: Any,
    ) -> float:
        """
        Get archive sparsity between an element and a set of elements (archive).

        :param images: The images to compare sparsity against (the first image is the candidate).
        :param solution_archive: The archive to compare sparsity against.
        :param genome_target: The target genome to compute sparsity for.
        :param genome_archive: The genome archive to compare sparsity against.
        :param _: Additional unused args.
        :return: The distance measure [0,1].
        """
        # TODO this is a bit ugly, target being images[1]
        target = images[1] if not self._on_genomes else genome_target
        archive = solution_archive if not self._on_genomes else genome_archive

        distances = [self.metric.evaluate(images=[target, elem]) for elem in archive]
        dist = self._head(distances)
        # A bigger distance is desired if we want to maximize sparsity
        return dist if self._inverse else 1 - dist
