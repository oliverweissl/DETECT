from skimage.metrics import structural_similarity as ssim
from numpy.typing import NDArray

def get_ssim(i1: NDArray, i2: NDArray) -> float:
    """
    Get structural similarity between two images.

    :param i1: The first image.
    :param i2: The second image.
    :returns: SSIM score.
    """
    score = ssim(i1, i2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255.)
    return score