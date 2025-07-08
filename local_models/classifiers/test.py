import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from local_models.classifiers.celeb_resnet_model import (AttributeClassifier, CelebADataset, evaluate)
from local_models.classifiers.celeb_configs import dataset_path, model_path
from local_models.classifiers.celeb_swag_model import SWAGCelebAClassifier
print(os.getcwd())
print(os.path.exists(dataset_path))
# ----------- Config ------------
batch_size = 256
num_classes = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% ----------- Data --------------
model = SWAGCelebAClassifier(num_classes=num_classes).to(device)


test_dataset = CelebADataset(root=dataset_path,
                              transform=model.transform, # use default
                              split='test'
                              )
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#model = AttributeClassifier(num_classes=num_classes).to(device)
#model.load_state_dict(torch.load(model_path, weights_only=False))

model.load_state_dict(torch.load(os.path.join("swag_celeb_40_parallel.pth",), weights_only=False))



model.eval()
criterion = nn.BCEWithLogitsLoss()

with torch.no_grad():
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, target_logit=15)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Resnet
# Test Loss: 0.1897, Test Acc: 0.9127
# 15 glasses: Test Loss: 0.0132, Test Acc: 0.9970

# SWAG
# Test Loss: 0.2344, Test Acc: 0.8922
# 15 glasses: Test Loss: 0.0445, Test Acc: 0.9819 2 epochs
# 15 glasses: Test Loss: 0.0303, Test Acc: 0.9917 10 epochs