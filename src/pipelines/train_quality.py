"""Command-line entry point for training the quality model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.models.quality_model import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train produce quality model")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--data-root", type=Path, required=True, help="Processed dataset root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for training outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    artifacts = train_model(config=config, data_root=args.data_root, output_dir=args.output_dir)
    summary_path = args.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": artifacts.config,
                "history": artifacts.history,
                "class_to_idx": artifacts.class_to_idx,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
