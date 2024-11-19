from typing import Any

import numpy as np

from .._criterion import Criterion
from .._default_arguments import DefaultArguments
from ._prepare_tensor import prepare_tensor


class CFrobeniusDistance(Criterion):
    """Implements a channel-wise Frobenius Distance measure."""

    def evaluate(self, *, default_args: DefaultArguments, **_: Any) -> float:
        """
        Calculate the normalized frobenius distance between two tensors [0,1].

        :param default_args: The default arguments parsed by the NeuralTester-
        :param _: Additional unused kwargs.
        :returns: The distance.
        """
        i1, i2 = prepare_tensor(default_args.i1), prepare_tensor(default_args.i2)
        ub = np.sqrt((255**2) * np.prod(i1.shape[:-1]))  # Distance supremum.
        return np.mean(np.linalg.norm(i1 - i2, ord="fro", axis=(0, 1))) / ub
