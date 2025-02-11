from typing import Callable
from torch import Tensor

DEFAULT_KWARGS = {
    "images": list[Tensor],  # A list of images for the Criteria calculation.
    "logits": Tensor,  # Predicted class probabilities.
    "label_targets": list[int],  # The labels expected in the criterion, could be [true label] or [primary label, secondary label, ...]
}


def criteria_kwargs(func: Callable) -> Callable:
    """
    A decorator to enforce uniform Kwargs across criteria -> easier debugging.

    :param func: The function to wrap around.
    :returns: The wrapped function.
    """
    def wrapper(*args, **kwargs) -> Callable:
        """
        Checking kwargs and types.

        :param args: The arguments to check.
        :param kwargs: The keyword arguments to check.
        :returns: The parameterized function.
        """
        ikw = set(kwargs.keys()) - DEFAULT_KWARGS.keys()
        if ikw:  # Invalid Keyword arguments.
            raise KeyError(f"Function {func.__name__} found invalid keyword arguments: {ikw}")

        for key, value in kwargs.items():
            expected_type = DEFAULT_KWARGS.get(key)
            if expected_type and not isinstance(value, expected_type):
                raise TypeError(f"Argument {key} in function {func.__name__} must be of type {expected_type}, found {type(value)}")
        return func(*args, **kwargs)
    return wrapper
