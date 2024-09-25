from torch import nn
from torch.utils.data import Dataset

class NeuralTester:
    """A tester class for neural style mixing."""

    _predictor: nn.Module
    _generator: nn.Module

    def __init__(
            self,
            *,
            predictor: nn.Module,
            generator: nn.Module,
            ):
        """
        Initialize the Neural Tester.

        :param predictor: The predictor network to test boundaries for.
        :param generator: Thy style mixer, that generates new inputs.
        """
        self._predictor = predictor
        self._generator = generator

    def test(self, input: Dataset):
        """
        Testing the predictor for its desicion boundary using a set of Inputs.

        :param input: The dataset to test.
        """
          




