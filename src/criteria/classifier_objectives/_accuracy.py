import torch
from torch import Tensor

from .._criterion import Criterion


class Accuracy(Criterion):
    """Implements Accuracy as a criterion."""

    def evaluate(self, y_true: Tensor, y_pred: Tensor) -> float:
        """
        Calculate the accuracy from prediction probabilities.

        :param y_true: The true labels.
        :param y_pred: The predicted probabilities.
        :return: The accuracy score.
        """
        _, y_pred_tags = torch.max(y_pred, dim=1) if len(y_pred.shape) > 1 else (None, y_pred)
        _, y_true_tags = torch.max(y_true, dim=1) if len(y_true.shape) > 1 else (None, y_true)

        corr_preds = (y_true_tags == y_pred_tags).float()
        acc = corr_preds.sum() / len(corr_preds)
        return acc
