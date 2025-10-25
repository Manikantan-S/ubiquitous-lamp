"""Run inference with the trained quality model and export predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import time

from src.models.quality_model import QualityModel


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, images_root: Path, split: str):
        with manifest_path.open("r", encoding="utf-8") as handle:
            items = json.load(handle)
        self.records = [item for item in items if item["split"] == split]
        if not self.records:
            raise ValueError(f"No records found for split '{split}' in {manifest_path}")
        self.images_root = images_root
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.class_names = sorted({item["label"] for item in self.records})
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.images_root / record["filepath"]
        with Image.open(image_path) as img:
            tensor = self.transform(img.convert("RGB"))
        label = self.class_to_idx[record["label"]]
        return tensor, label, record


def load_model(checkpoint: Path, num_classes: int, device: torch.device) -> QualityModel:
    model = QualityModel(num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: QualityModel,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> List[Dict]:
    predictions: List[Dict] = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for images, labels, records in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = softmax(logits).cpu()
            preds = probs.argmax(dim=1)
            for prob, pred, label, record in zip(probs, preds, labels, records):
                predictions.append(
                    {
                        "image_id": record["image_id"],
                        "predicted_label": class_names[int(pred.item())],
                        "true_label": class_names[int(label.item())],
                        "confidence": float(prob[pred].item()),
                        "spoilage_score": float(record["spoilage_score"]),
                    }
                )
    return predictions


def attach_locations(predictions: List[Dict], locations_path: Path | None) -> pd.DataFrame:
    df = pd.DataFrame(predictions)
    if locations_path and locations_path.exists():
        loc_df = pd.read_csv(locations_path)
        df = df.merge(loc_df, on="image_id", how="left")
    else:
        df["location_id"] = df["image_id"].apply(lambda x: f"loc_{x}")
        df["latitude"] = 0.0
        df["longitude"] = 0.0
        df["demand_kg"] = 100.0
        df["service_minutes"] = 10.0
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality inference")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--locations", type=Path, help="Optional CSV with location metadata")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary output path")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    dataset = ManifestDataset(args.manifest, args.images_root, split="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, num_classes=len(dataset.class_names), device=device)

    start_time = time.time()
    predictions = run_inference(model, dataloader, device, dataset.class_names)
    inference_seconds = time.time() - start_time
    df = attach_locations(predictions, args.locations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    if args.summary:
        summary = {
            "num_samples": len(predictions),
            "mean_confidence": float(df["confidence"].mean()) if not df.empty else 0.0,
            "inference_seconds": inference_seconds,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
