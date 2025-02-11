from typing import Any

from torch import Tensor

from .._criterion import Criterion


class PenalizedDistance(Criterion):
    """Implements the penalized distance measure."""

    _name: str = "PenalizedDistance"
    metric: Criterion

    def __init__(self, metric: Criterion) -> None:
        """
        Initialize the Penalized Distance measure.

        :param metric: The metric used in the measure calculation.
        """
        super().__init__()
        self.metric = metric

    def evaluate(
        self, *, images: list[Tensor], logits: Tensor, label_targets: list[int], **_: Any
    ) -> float:
        """
        Get penalized distance between two images using their labels.

        :param images: The images used to compute the penalized distance.
        :param logits: The logits used to compute the penalized distance.
        :param label_targets: The labels used to compute the penalized distance.
        :param _: Additional unused args.
        :return: The distance measure [0,1].
        """
        y1p, y2p = logits[label_targets[0]], logits[label_targets[1]]
        score = self.metric.evaluate(images=images, logits=logits, label_targets=label_targets)
        distance = (1 - score) ** (0 if y2p < y1p else 1)
        return distance
