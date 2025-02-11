import torch
from torch import Tensor

from .._criterion import Criterion


class Accuracy(Criterion):
    """Implements Accuracy as a criterion."""

    _name: str = "Accuracy"

    def evaluate(self, logits: Tensor, label_targets: list[int]) -> float:
        """
        Calculate the accuracy from prediction probabilities.

        :param logits: The predicted probabilities.
        :param label_targets: The ground truth labels.
        :return: The accuracy score.
        """
        _, y_pred_tags = torch.max(logits, dim=1) if len(logits.shape) > 1 else (None, logits)

        corr_preds = (label_targets == y_pred_tags).float()
        acc = corr_preds.sum() / len(corr_preds)
        return acc
