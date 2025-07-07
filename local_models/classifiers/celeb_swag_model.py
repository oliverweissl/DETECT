import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class SWAGCelebAClassifier(nn.Module):
    def __init__(self, num_classes=40, swag_model_name="vit_h14_in1k", freeze_backbone=True):
        super().__init__()

        # Load SWAG model as feature extractor
        self.backbone = torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")

        # Get the feature dimension from the backbone
        # For ViT local_models, we need to replace the head
        if hasattr(self.backbone, 'head'):
            head = self.backbone.head
            # Check if it's a standard Linear layer
            if hasattr(head, 'in_features'):
                in_features = head.in_features
                self.backbone.head = nn.Identity()
            # Check if it's a VisionTransformerHead or similar
            elif hasattr(head, 'layers') and hasattr(head.layers, 'head') and hasattr(head.layers.head, 'in_features'):
                in_features = head.layers.head.in_features

                # Replace the entire head with Identity
                self.backbone.head = nn.Identity()
            # Check if it has a different structure
            elif hasattr(head, 'weight'):
                in_features = head.weight.shape[1]
                self.backbone.head = nn.Identity()

        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            # Remove the original classification head
            self.backbone.fc = nn.Identity()
        else:
            # Default feature dimension for ViT-H/14
            in_features = 1280

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add custom classification head for CelebA attributes
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Resolution expected by SWAG model
        self.resolution = 518 if "h14" in swag_model_name else 224

        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(
                self.resolution,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def forward(self, x):
        # Extract features using SWAG backbone
        features = self.backbone(x)

        # Apply custom classification head
        output = self.classifier(features)

        return output

    def predict_attributes(self, image_path_or_tensor, threshold=0.5):
        """
        Predict attributes for a single image

        Args:
            image_path_or_tensor: Path to image file or preprocessed tensor
            threshold: Threshold for binary classification

        Returns:
            Dictionary with attribute predictions
        """
        self.eval()

        if isinstance(image_path_or_tensor, str):
            # Load and preprocess image
            image = Image.open(image_path_or_tensor).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
        else:
            image_tensor = image_path_or_tensor

        with torch.no_grad():
            logits = self(image_tensor)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()

        return {
            'logits': logits.squeeze().cpu().numpy(),
            'probabilities': probabilities.squeeze().cpu().numpy(),
            'predictions': predictions.squeeze().cpu().numpy()
        }


# CelebA attribute names for reference
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]


def create_swag_celeba_model(model_name="vit_h14_in1k", freeze_backbone=True):
    """
    Factory function to create SWAG-based CelebA classifier

    Args:
        model_name: SWAG model name (e.g., "vit_h14_in1k", "vit_b16_in1k")
        freeze_backbone: Whether to freeze the SWAG backbone

    Returns:
        SWAGCelebAClassifier instance
    """
    return SWAGCelebAClassifier(
        num_classes=40,
        swag_model_name=model_name,
        freeze_backbone=freeze_backbone
    )


# Example usage and testing functions
def test_model():
    """Test the model with a dummy input"""
    model = create_swag_celeba_model()
    dummy_input = torch.randn(1, 3, 518, 518)

    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Sample output: {output[0][:5]}")

    return model


if __name__ == "__main__":
    import os
    print(os.getcwd())
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from celeb_resnet_model import (AttributeClassifier, CelebADataset,
                                                             EarlyStopping, TrainingHistory, multi_label_accuracy, evaluate)
    from celeb_configs import dataset_path


    print(os.path.exists(dataset_path))
    print(os.getcwd())

        #%%  ----------- Config ------------
    batch_size = 64
    num_epochs = 10
    num_classes = 40
    lr = 1e-4
    patience = 5
    verbose = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join("local_models/classifiers","checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'swag_celeb_40_parallel.pth')

    #%%  ----------- Data --------------
    model = SWAGCelebAClassifier(num_classes=num_classes,
                                 swag_model_name="vit_h14_in1k",
                                 freeze_backbone=True).to(device)

    train_dataset = CelebADataset(root=dataset_path,
                                  transform=model.transform,
                                  split='train'
                                  )
    val_dataset = CelebADataset(root=dataset_path,
                                transform=model.transform,  # use default
                                split='val'
                                )
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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