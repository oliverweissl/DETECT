from typing import Any, Optional

import numpy as np

from .._criterion import Criterion
from .._criteria_arguments import CriteriaArguments


class DynamicConfidenceBalance(Criterion):
    """Implements a dynamic confidence balance measure."""

    _name: str = "DynCB"
    _target_primary: bool

    def __init__(self, inverse: bool = False, target_primary: Optional[bool] = None) -> None:
        """
        Initialize the criterion.

        :param inverse: Whether the measure should be inverted.
        :param target_primary: Whether y1 is focus of the measure or yp, if none neither is in focus.
        """
        super().__init__(inverse=inverse)
        self._target_primary = target_primary

    def evaluate(self, *, default_args: CriteriaArguments, **_: Any) -> float:
        """
        Calculate the confidence balance of 2 confidence values.

        This functions assumes input range of [0, 1].

        :param default_args: The default args parsed by the NeuralTester.
        :param _: Unused kwargs.
        :returns: The value.
        """
        yp_arr = default_args.yp.detach().cpu().numpy()
        y = np.delete(yp_arr, default_args.c1)
        s = default_args.y1p + y.max()
        d = abs(default_args.y1p - y.max())
        if self._target_primary is None:
            return abs(self._inverse.real - d / s)
        else:
            return abs(
                self._inverse.imag
                - (y.max() if self._target_primary else default_args.y1p)
                - d / s
            )
