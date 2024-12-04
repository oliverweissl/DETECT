from typing import Any

from .._criterion import Criterion
from .._default_arguments import DefaultArguments


class IsMisclassified(Criterion):
    """Implements a criterion to check if a prediction is incorrect."""

    _name: str = "IsMisclassified"

    def evaluate(self, *, default_args: DefaultArguments, **_: Any) -> float:
        """
        Check if a prediction is incorrect.

        This functions assumes input range of [0, 1].

        :param default_args: The default args parsed by the NeuralTester.
        :param _: Unused kwargs.
        :returns: The value.
        """
        result = (
            default_args.y1 == default_args.y2
            if self._inverse
            else default_args.y1 != default_args.y2
        )
        return result
