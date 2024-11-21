from typing import Any

import numpy as np

from .._criterion import Criterion
from .._default_arguments import DefaultArguments


class DynamicConfidenceBalance(Criterion):
    """Implements a dynamic confidence balance measure."""

    _name: str = "DynCB"

    def evaluate(self, *, default_args: DefaultArguments, **_: Any) -> float:
        """
        Calculate the confidence balance of 2 confidence values.

        This functions assumes input range of [0, 1].

        :param default_args: The default args parsed by the NeuralTester.
        :param _: Unused kwargs.
        :returns: The value.
        """
        yp_arr = default_args.yp.detach().cpu().numpy()
        y = np.ma.array(yp_arr, mask=False)
        y.mask[default_args.y1] = True
        s = default_args.y1p + y.max()
        d = default_args.y1p - y.max()
        return abs(d) / s
