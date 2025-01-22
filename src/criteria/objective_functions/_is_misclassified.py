from typing import Any

from .._criterion import Criterion
from .._criteria_arguments import CriteriaArguments


class IsMisclassified(Criterion):
    """Implements a criterion to check if a prediction is incorrect."""

    _name: str = "IsMisclassified"

    def evaluate(self, *, default_args: CriteriaArguments, **_: Any) -> float:
        """
        Check if a prediction is incorrect.

        This functions assumes input range of [0, 1].

        :param default_args: The default args parsed by the NeuralTester.
        :param _: Unused kwargs.
        :returns: The value.
        """
        result = (
            default_args.c1 == default_args.yp.argmax().item()
            if self._inverse
            else default_args.c1 != default_args.yp.argmax().item()
        )
        return result
