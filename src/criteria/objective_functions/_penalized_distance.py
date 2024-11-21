from typing import Any

from .._criterion import Criterion
from .._default_arguments import DefaultArguments


class PenalizedDistance(Criterion):
    """Implements the penalized distance measure."""

    _name: str = "PenalizedDistance"
    metric: Criterion

    def __init__(self, metric: Criterion) -> None:
        """
        Initialize the Penalized Distance measure.

        :param metric: The metric used in the measure calculation.
        """
        self.metric = metric

    def evaluate(self, *, default_args: DefaultArguments, **_: Any) -> float:
        """
        Get penalized distance between two images using their labels.

        :param default_args: The default args parsed by the NeuralTester.
        :param _: Additional unused args.
        :return: The distance measure [0,1].
        """
        score = self.metric.evaluate(default_args=default_args)
        distance = (1 - score) ** (0 if default_args.y2p < default_args.y1p else 1)
        return distance
