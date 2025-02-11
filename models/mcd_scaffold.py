from typing import Union

import torch
import torch.nn as nn


class MonteCarloDropoutScaffold(nn.Module):
    """Scaffold class to support Monte-Carlo dropout in inference."""

    def __init__(
        self,
        model: nn.Module,
        dropout_prob: float = 0.1,
        mc_iter: int = 20,
        return_variance: bool = False,
    ) -> None:
        """
        Initialize the model scaffold.

        :param model: The model to modify.
        :param dropout_prob: The dropout probability.
        :param mc_iter: The number of Monte-Carlo iterations.
        :param return_variance: If True, return variance after each Monte-Carlo iteration.
        """
        super(MonteCarloDropoutScaffold, self).__init__()

        layers = list(model.children())
        fc = layers[-1]

        self.features = nn.Sequential(*layers[:-1])
        self.dropout = nn.Dropout(p=dropout_prob)
        assert isinstance(fc, nn.Linear), f"ERROR: last layer is not a linear layer. {type(fc)}"
        self.fc: nn.Linear = fc

        self._mc_iter = mc_iter
        self._return_variance = return_variance

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict x with uncertainty measure.

        :param x: The input tensor.
        :returns: The prediction distribution.
        """
        pred: torch.Tensor = torch.zeros((self._mc_iter, x.size()[0], self.fc.out_features)).to(
            x.device
        )
        self.dropout.train(True)
        for i in range(self._mc_iter):
            mx = self.features(x)

            mx = mx.view(mx.size(dim=0), -1)
            mx = self.dropout(mx)
            pred[i] = self.fc(mx)

        res = (pred.mean(dim=0), pred.var(dim=0)) if self._return_variance else pred.mean(dim=0)
        return res
