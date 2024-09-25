from ._w1_distance import get_w1_distance


def get_penalized_distance(img, p_img, y_img, y_p_img) -> float:
    """
    Get penalized distance between two images using their labels.


    :param img: The original image.
    :param p_img: The perturbed image.
    :param y_img: The original images label.
    :param y_p_img: The perturbed images label.
    :return: The distance measure [0,1].
    """
    distance = get_w1_distance(img, p_img) ** (0 if y_p_img == y_img else 1)
    return distance
