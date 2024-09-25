"""A collection of objective functions."""

from ._w1_distance import get_w1_distance
from ._accuracy import get_accuracy
from ._penalized_distance import get_penalized_distance

__all__ = ["get_w1_distance", "get_accuracy", "get_penalized_distance"]