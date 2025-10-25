"""Lightweight CNN for produce quality grading."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


@dataclass
class TrainingArtifacts:
    config: Dict
    history: Dict[str, list]
    class_to_idx: Dict[str, int]


class QualityModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(data_root: Path, config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    train_transform, eval_transform = build_transforms(config.training.image_size)
    datasets = {}
    for split, transform in zip(["train", "val", "test"], [train_transform, eval_transform, eval_transform]):
        datasets[split] = ImageFolder(root=data_root / split, transform=transform)
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=config.training.batch_size,
            shuffle=(split == "train"),
            num_workers=config.training.num_workers,
            pin_memory=True,
        )
        for split in datasets
    }
    return (
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["test"],
        datasets["train"].class_to_idx,
    )


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def select_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(config: DictConfig, data_root: Path, output_dir: Path) -> TrainingArtifacts:
    device = select_device()
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(data_root, config)
    model = QualityModel(num_classes=len(class_to_idx), pretrained=config.model.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(config.training.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
        )

    duration_seconds = time.time() - start_time
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "training_log.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "config": OmegaConf.to_container(config, resolve=True),
                "class_to_idx": class_to_idx,
                "device": str(device),
                "duration_seconds": duration_seconds,
                "inference_seconds": 0.0,
            },
            handle,
            indent=2,
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"loss": test_loss, "accuracy": test_acc}, handle, indent=2)

    return TrainingArtifacts(
        config=OmegaConf.to_container(config, resolve=True),
        history=history,
        class_to_idx=class_to_idx,
    )
