import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Type

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import wandb
from src.criteria import Criterion
from src.criteria.classifier_objectives import Accuracy


@dataclass
class TrainConfig:
    """Dataclass for training configs."""

    lr: float
    max_lr: float
    epochs: int
    batch: int
    model: str
    dataset: str


class ClTrainer:
    """A trainer class to train classifier models."""

    _train_loader: DataLoader
    _val_loader: DataLoader

    _model: torch.nn.Module
    _criterion: torch.nn.Module
    _evaluator: Criterion
    _optimizer: torch.optim.Optimizer
    _scheduler: object

    _label_dict: dict[int, str]  # A map between labels and indices
    _device: torch.device
    _config: TrainConfig
    _is_trained: bool
    _log_inner: bool

    def __init__(
        self,
        *,
        train_dataset: Dataset,
        val_dataset: Dataset,
        model: torch.nn.Module,
        criterion: Type[torch.nn.Module],
        optimizer: Type[torch.optim.Optimizer],
        scheduler: Type[object],
        train_config: TrainConfig,
        out_features: int,
        workers: int = 2,
        log_inner: bool = False,
    ) -> None:
        """
        Initialize the Classifier Trainer object.

        :param train_dataset: The training Dataset.
        :param val_dataset: The validation Dataset.
        :param model: The model to be trained (initialized).
        :param criterion: The loss type to be used (not initialized).
        :param optimizer: The optimizer for training (not initialized).
        :param scheduler: The scheduler for lr adaptation (not initialized).
        :param train_config: The training config.
        :param out_features: Adapt the FC layaer at the end of the model to fit class info.
        :param workers: The number of workers for data loaders (default=2).
        :param log_inner: If wandb logging should log step in epoch.
        """
        self._label_dict = {y: x for x, y in train_dataset.class_to_idx.items()}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._config = train_config
        self._log_inner = log_inner

        # Make dataloaders.
        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_config.batch,
            shuffle=True,
            num_workers=workers,
        )
        self._val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=train_config.batch,
            shuffle=False,
            num_workers=workers,
        )

        # Set training components.
        self._model = model
        self._model.fc = torch.nn.Linear(model.fc.in_features, out_features)
        self._criterion = criterion()
        self._evaluator = Accuracy()
        self._optimizer = optimizer(model.parameters(), lr=train_config.lr)
        self._scheduler = scheduler(
            self._optimizer,
            max_lr=train_config.max_lr,
            steps_per_epoch=len(self._train_loader),
            epochs=train_config.epochs,
        )

        self._model.to(self._device)
        wandb.init(project="NeuralStyleClassifiers", config=train_config.__dict__)
        self._is_trained = False

    def train(self) -> None:
        """
        Start the training.

        :raises Exception: If the model is already trained.
        """
        if self._is_trained:
            raise Exception("The model is already Trained!")
        start = datetime.now()
        for _ in tqdm(range(self._config.epochs)):
            self._train_epoch()
            self._val_epoch()
        logging.info(f"Finished Training in {datetime.now()-start}!")
        self._is_trained = True

    def show_samples(self, size: int = 4) -> None:
        """
        Show samples from the training loader.

        :param size: The size of the grid (N x N).
        """
        images, labels = next(iter(self._train_loader))
        plt.figure(figsize=(12, 8))
        for i, (img, lab) in enumerate(zip(images, labels)):
            plt.subplot(size, size, i + 1)
            image = to_pil_image(img)
            plt.imshow(image)
            plt.title(self._label_dict[lab.item()])
            plt.axis("off")
            if i + 2 > size**2:
                break
        plt.show()

    def save_model(self, path: str) -> None:
        """
        Save the trained model.

        :param path: The model path.
        :raises Exception: If model is not trained.
        """
        if not self._is_trained:
            raise Exception("The model is not Trained!")
        torch.save(self._model, path)

    def _train_epoch(self) -> None:
        self._model.train()
        tacc, tloss = 0.0, 0.0
        for X, y in self._train_loader:
            X, y = X.to(self._device), y.to(self._device)

            pred = self._model(X)
            loss = self._criterion(pred, y)

            loss.backward()
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()

            acc = self._evaluator.evaluate(y_true=y, y_pred=pred)
            tacc += acc
            tloss += loss.item()
            if self._log_inner:
                wandb.log(
                    {
                        "acc": acc,
                        "loss": loss.item(),
                        "lr": self._scheduler.optimizer.param_groups[0]["lr"],
                    }
                )
        if not self._log_inner:
            wandb.log(
                {
                    "acc": tacc / len(self._train_loader),
                    "loss": tloss / len(self._train_loader),
                    "lr": self._scheduler.optimizer.param_groups[0]["lr"],
                }
            )

    def _val_epoch(self) -> None:
        self._model.eval()
        with torch.no_grad():
            tacc, tloss = 0.0, 0.0
            for X, y in self._val_loader:
                X, y = X.to(self._device), y.to(self._device)
                pred = self._model(X)

                acc = self._evaluator.evaluate(y_true=y, y_pred=pred)
                tacc += acc
                tloss += (loss := self._criterion(pred, y).item())
                if self._log_inner:
                    wandb.log(
                        {
                            "val_acc": acc,
                            "val_loss": loss,
                        }
                    )
            if not self._log_inner:
                wandb.log(
                    {
                        "val_acc": tacc / len(self._val_loader),
                        "val_loss": tloss / len(self._val_loader),
                    }
                )
