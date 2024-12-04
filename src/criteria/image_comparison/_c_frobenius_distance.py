from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._criterion import Criterion
from .._default_arguments import DefaultArguments
from ._prepare_tensor import prepare_tensor


class CFrobeniusDistance(Criterion):
    """Implements a channel-wise Frobenius Distance measure."""

    _name: str = "CFrobDistance"

    def evaluate(self, *, default_args: DefaultArguments, **_: Any) -> float:
        """
        Calculate the normalized frobenius distance between two tensors that range [0,1].

        :param default_args: The default arguments parsed by the NeuralTester-
        :param _: Additional unused kwargs.
        :returns: The distance.
        """
        i1, i2 = prepare_tensor(default_args.i1), prepare_tensor(default_args.i2)
        ub = self._frob(np.ones(i1.shape[:-1]))
        mean_fn = (
            sum([self._frob(i1[..., j] - i2[..., j]) for j in range(i1.shape[-1])]) / i1.shape[-1]
        )
        return abs(self._inverse.real - mean_fn / ub)

    @staticmethod
    def _frob(matrix: NDArray) -> float:
        """
        Calculate the frobenius norm for a NxN matrix.

        :param matrix: The matrix to calculate with.
        :returns: The norm.
        """
        return np.sqrt(np.trace(matrix.T @ matrix))
