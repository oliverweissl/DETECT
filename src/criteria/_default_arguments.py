from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class DefaultArguments:
    """
    A class to store the default arguments used in the NeuralTester.

    This class is made to allow for simple refactoring and addition of new arguments.
    """

    i1: Tensor  # The primary image
    i2: Tensor  # The generated image

    y1: int  # The primary class
    y2: int  # The second most likely class

    yp: Tensor  # Class probabilities

    y1p: float = field(init=False)
    y2p: float = field(init=False)

    def __post_init__(self) -> None:
        """Additional processing after init."""
        self.y1p = self.yp[self.y1]
        self.y2p = self.yp[self.y2]
