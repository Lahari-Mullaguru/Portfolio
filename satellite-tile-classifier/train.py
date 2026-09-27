"""
Training script: fine-tune ResNet-18 on candidate tiles.

Usage:
    python train.py

Expects:
    data/candidate_tiles/Forest/Forest_1.png
    data/candidate_tiles/River/River_1.png
    (etc.)

Produces:
    model/classifier.pt
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from app.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, MODEL_PATH, CANDIDATE_TILES_DIR


def main():
    if not CANDIDATE_TILES_DIR.exists():
        print(f"ERROR: candidate tiles directory not found at {CANDIDATE_TILES_DIR}")
        print("Place your training images in data/candidate_tiles/<ClassName>/")
        sys.exit(1)

    # Data pipeline
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(
        root=str(CANDIDATE_TILES_DIR),
        transform=train_transform,
    )

    folder_classes = sorted(dataset.classes)
    if folder_classes != CLASS_NAMES:
        # Verify class names match our config
        print(f"WARNING: folder classes {folder_classes} != expected {CLASS_NAMES}")
        print("The model will use folder ordering. Update config.py if needed.")

    print(f"Training samples: {len(dataset)}")
    print(f"Classes (from folders): {dataset.classes}")
    print(f"Class-to-index mapping: {dataset.class_to_idx}")

    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # Freeze early layers, only fine-tune later layers + fc
    for name, param in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 30
    device = torch.device("cpu")
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct / total_samples
        print(f"Epoch {epoch:2d}/{num_epochs} loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"\nModel saved to {MODEL_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()