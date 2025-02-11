from typing import Any, Optional

import numpy as np
from torch import Tensor

from .._criterion import Criterion


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

    def evaluate(self, *, logits: Tensor, label_targets: list[int], **_: Any) -> float:
        """
        Calculate the confidence balance of 2 confidence values.

        This functions assumes input range of [0, 1].

        :param logits: Logits tensor.
        :param label_targets: Label targets used to determine targets of balance.
        :param _: Unused kwargs.
        :returns: The value.
        """
        c1 = label_targets[0]  # The primary class

        yp_arr = logits.detach().cpu().numpy().copy()
        y = np.delete(yp_arr, c1)
        s = logits[c1] + y.max()
        d = abs(logits[c1] - y.max())
        if self._target_primary is None:
            return abs(self._inverse.real - d / s)
        else:
            return abs(
                self._inverse.imag - (y.max() if self._target_primary else logits[c1]) - d / s
            )
