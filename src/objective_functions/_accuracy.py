import torch


def get_accuracy(y_true, y_pred) -> float:
    """
    Calculate the accuracy from prediction probabilities.


    :param y_true: The true labels.
    :param y_pred: The predicted probabilities.
    :return: The accuracy score.
    """
    _, y_pred_tags = torch.max(y_pred, dim=1)
    _, y_true_tags = torch.max(y_true, dim=1)

    corr_preds = (y_true_tags == y_pred_tags).float()
    acc = corr_preds.sum() / len(corr_preds)
    return acc
