from numpy.typing import NDArray
from scipy import sparse
import numpy as np
#from scipy.stats import wasserstein_distance_nd

def get_w1_distance(u: NDArray, v: NDArray) -> float:
    """
    Get the wasserstein-1 distance from two distributions.

    :param u: Distribution 1.
    :param v: Distribution 2.
    :return: The distance [0,1].
    """
    m, n = len(u), len(v)


    u /= u.sum()
    v /= v.sum()

    A_u = sparse.block_diag((np.ones((1, n)),) * m)
    A_l = sparse.hstack((sparse.eye(n),) * m)

    A = sparse.vstack((A_u, A_l))
    A = sparse.coo_array(A)

    u1, u2 = u.shape
    v1, v2 = v.shape
    # TODO: this is WIP

    distance = 1 #wasserstein_distance_nd(u, v)
    assert 1>= distance >= 0, "Distance does something weird."
    return distance
