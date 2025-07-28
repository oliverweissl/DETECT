import torch
import torch.nn as nn
from torchvision import models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class AttributeClassifier(nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)

        # Replace last FC layer for 40 attributes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        #return torch.sigmoid(self.backbone(x))  # Multi-label output
        return self.backbone(x)  # Multi-label output

class CelebADataset(Dataset):
    def __init__(self, root, attr_path='list_attr_celeba.csv', image_dir='img_align_celeba/img_align_celeba',
                 transform=None, split='train'):
        """
        Args:
            root (str): Path to the root CelebA directory
            attr_path (str): Relative path to the attribute file
            image_dir (str): Directory name containing the images
            transform (callable, optional): Transformations to apply to each image
            split (str): One of 'train', 'val', or 'test'
        """
        self.root = root
        self.transform = transform
        self.image_dir = os.path.join(root, image_dir)
        attr_path = os.path.join(root, attr_path)

        # Load attribute list
        df = pd.read_csv(attr_path)

        # Line 0: number of images (can be ignored)
        # Line 1: attribute names
        # first column contains filenames and rest are attributes
        self.attr_names = df.columns[1:].tolist()

        # Split the dataset (hardcoded indices from official CelebA split)
        if split == 'train':
            df = df.iloc[:162770]
        elif split == 'val':
            df = df.iloc[162770:182637]
        elif split == 'test':
            df = df.iloc[182637:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples = []
        for _, row in df.iterrows():
            filename = row.iloc[0]  # First column contains filename
            labels = row.iloc[1:].astype(float).values  # Rest of columns are attributes
            # Convert to tensor (assuming labels are already 0/1)
            # Since labels are -1/1, convert into 0/1
            labels = (labels + 1) // 2
            self.samples.append((filename, torch.tensor(labels, dtype=torch.float)))

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def update(self, train_loss, train_acc, val_loss, val_acc):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)

    def plot(self, save_dir='plots'):
        os.makedirs(save_dir, exist_ok=True)

        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss_history.png'))
        plt.close()

        # Plot accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accs, label='Training Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'accuracy_history.png'))
        plt.close()


def multi_label_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).float().sum()
    return correct / (targets.numel())


def evaluate(model, data_loader, criterion, device, target_logit=None):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if target_logit is None:
                loss = criterion(outputs, labels)
                acc = multi_label_accuracy(outputs, labels)

                total_loss += loss.item()
                total_acc += acc.item()
            else:
                loss = criterion(outputs[:, target_logit], labels[:, target_logit])
                acc = multi_label_accuracy(outputs[:, target_logit], labels[:, target_logit])

                total_loss += loss.item()
                total_acc += acc.item()
                print(f"loss: {loss.item()}, acc: {acc.item()}")
    return total_loss / len(data_loader), total_acc / len(data_loader)

