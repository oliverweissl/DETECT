import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from local_models.classifiers.celeb_resnet_model import (AttributeClassifier, CelebADataset,
                                                         EarlyStopping, TrainingHistory, multi_label_accuracy, evaluate)
from local_models.classifiers.celeb_configs import dataset_path

#%% ----------- Dataset --------------
# uncomment the following line to download the dataset
"""import kagglehub
# Download latest version
path_dataset = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Path to dataset files:", path)"""
print(os.path.exists(dataset_path))
print(os.getcwd())

#%%  ----------- Config ------------
batch_size = 256
num_epochs = 10
num_classes = 40
lr = 1e-4
patience = 5
verbose = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = os.path.join("local_models/classifiers","checkpoints")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'resnet_celeb_40_parallel.pth')

#%%  ----------- Data --------------
train_dataset = CelebADataset(root=dataset_path,
                              transform=None,  # use default
                              split='train'
                              )
val_dataset = CelebADataset(root=dataset_path,
                            transform=None,  # use default
                            split='val'
                            )
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
model = AttributeClassifier(num_classes=num_classes).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
early_stopping = EarlyStopping(patience=patience, verbose=verbose)
history = TrainingHistory()

#%%  ----------- Training ----------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #acc = multi_label_accuracy(outputs, labels)
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            acc = multi_label_accuracy(probs, labels)
        running_loss += loss.item()
        running_acc += acc.item()

        if verbose and i % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}], "
                  f"Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

    # compute average loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_acc = running_acc / len(train_loader)

    # validation
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # update history
    history.update(train_loss, train_acc, val_loss, val_acc)

    if verbose:
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("-" * 50)

    # check early stopping
    early_stopping(val_loss, model, save_path)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

history.plot()
#%%
"""
Epoch [7/10], Step [560], Loss: 0.1080, Acc: 0.9543
Epoch [7/10], Step [570], Loss: 0.1108, Acc: 0.9516
Epoch [7/10], Step [580], Loss: 0.1081, Acc: 0.9562
Epoch [7/10], Step [590], Loss: 0.1045, Acc: 0.9558
Epoch [7/10], Step [600], Loss: 0.1021, Acc: 0.9591
Epoch [7/10], Step [610], Loss: 0.1112, Acc: 0.9551
Epoch [7/10], Step [620], Loss: 0.0976, Acc: 0.9583
Epoch [7/10], Step [630], Loss: 0.1068, Acc: 0.9544
Epoch 7/10
Train Loss: 0.1012 | Train Acc: 0.9587
Val Loss: 0.2284 | Val Acc: 0.9127
--------------------------------------------------
EarlyStopping counter: 5 out of 5
Early stopping triggered"""
