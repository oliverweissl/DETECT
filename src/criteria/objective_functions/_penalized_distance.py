from torch import Tensor

from .._criterion import Criterion


class PenalizedDistance(Criterion):
    """Implements the penalized distance measure."""

    metric: Criterion

    def __init__(self, metric: Criterion) -> None:
        """
        Initialize the Penalized Distance measure.

        :param metric: The metric used in the measure calculation.
        """
        self.metric = metric

    def evaluate(self, *, i1: Tensor, i2: Tensor, y1: int, y2: int) -> float:
        """
        Get penalized distance between two images using their labels.

        :param i1: The original image.
        :param i2: The perturbed image.
        :param y1: The original images label.
        :param y2: The perturbed images label.
        :return: The distance measure [0,1].
        """
        score = self.metric.evaluate(i1=i1, i2=i2)
        distance = (1 - score) ** (0 if y2 < y1 else 1)
        return distance
