from typing import Any, Callable

import numpy as np
from torch import Tensor

from .._criterion import Criterion


class ArchiveSparsity(Criterion):
    """Implements sparsity measure wrt an Archive."""

    _name: str = "ArchiveSparsity"
    metric: Criterion
    head: Callable

    def __init__(self, metric: Criterion, regime: str = "min") -> None:
        """
        Initialize the Archive Sparsity measure.

        :param metric: The metric used in the measure calculation.
        :param regime: How to decide on the metric value.
        :raises NotImplementedError: If the regieme parsed is not implemented.
        """
        super().__init__()
        self.metric = metric
        if regime == "min":
            self.head = np.min
        elif regime == "max":
            self.head = np.max
        elif regime == "mean":
            self.head = np.mean
        else:
            raise NotImplementedError(f"Regime {regime} not implemented.")

    def evaluate(self, *, images: list[Tensor], **_: Any) -> float:
        """
        Get archive sparsity between an element and a set of elements (archive).

        :param images: The images to compare sparsity against (the first image is the candidate).
        :param _: Additional unused args.
        :return: The distance measure [0,1].
        """
        target, archive = images[0], images[1:]
        distances = [self.metric.evaluate(images=[target, elem]) for elem in archive]

        return self.head(distances)
