from typing import Union

import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def prepare_tensor(tensor: Union[Tensor, NDArray]) -> NDArray:
    """
    Prepare torch Tensor into numpy NDArray with correct dimension order.

    :param tensor: The tensor to prepare.
    :returns: The numpy array.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    ndarray = tensor.detach().cpu().numpy()
    ndarray = ndarray.transpose(1, 2, 0)
    return ndarray
