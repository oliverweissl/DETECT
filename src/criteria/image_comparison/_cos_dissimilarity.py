from typing import Any

import numpy as np

from .._criterion import Criterion
from .._criteria_arguments import CriteriaArguments
from ._prepare_tensor import prepare_tensor


class CosDissimilarity(Criterion):
    """Implements cos dissimilarity measure."""

    _name: str = "CosDissim"

    def evaluate(self, *, default_args: CriteriaArguments, **_: Any) -> float:
        """
        Get the cosine-dissimilarity between two images.

        Range [0,1] with 0 being the same image.

        :param default_args: The default arguments parsed by the NeuralTester.
        :param _: Additional unused kwargs.
        :returns: The score.
        """
        i1, i2 = prepare_tensor(default_args.i1), prepare_tensor(default_args.i2)
        value = np.dot(i1.flatten(), i2.flatten()) / (np.linalg.norm(i1) * np.linalg.norm(i2))
        return 1 - value
