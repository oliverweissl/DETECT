from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class CriteriaArguments:
    """
    A class to store the default arguments used in the NeuralTester.

    This class is made to allow for simple refactoring and addition of new arguments.
    """

    i1: Tensor  # The primary image
    i2: Tensor  # The generated image

    """The classes do not have to be the most likely ones!"""
    c1: int  # The primary class
    c2: int  # The secondary class

    yp: Tensor  # Class probabilities

    y1p: float = field(init=False)
    y2p: float = field(init=False)

    def __post_init__(self) -> None:
        """Additional processing after init."""
        self.y1p = self.yp[self.c1].item()
        self.y2p = self.yp[self.c2].item()
