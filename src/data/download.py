"""Utilities for downloading and validating produce-quality datasets."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

import requests


@dataclass
class DatasetResource:
    """Metadata describing a dataset resource."""

    name: str
    url: str
    description: str
    manual_download: bool = True
    archive_name: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


KNOWN_DATASETS: List[DatasetResource] = [
    DatasetResource(
        name="fruits_fresh_rotten",
        url="https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification",
        description=(
            "Images of apples, bananas, and oranges labelled as fresh or rotten."
            " Kaggle credentials are required for programmatic download."
        ),
        manual_download=True,
    ),
    DatasetResource(
        name="mendeley_quality",
        url="https://data.mendeley.com/datasets/9sxdyb86ph/2",
        description=(
            "Annotated images of multiple fruits and vegetables captured across"
            " progressive spoilage stages."
        ),
        manual_download=True,
    ),
    DatasetResource(
        name="wasteless_veggies",
        url="https://zenodo.org/records/7814156/files/wasteless_veggies.zip?download=1",
        description=(
            "Open-access dataset with bounding boxes and quality ratings for"
            " vegetables collected in a commercial distribution centre."
        ),
        manual_download=False,
        archive_name="wasteless_veggies.zip",
    ),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_datasets(resources: Iterable[DatasetResource]) -> str:
    rows = []
    for resource in resources:
        rows.append(
            {
                "name": resource.name,
                "url": resource.url,
                "manual_download": resource.manual_download,
                "description": resource.description,
            }
        )
    return json.dumps(rows, indent=2)


def download_file(url: str, output_path: Path, chunk_size: int = 1 << 20) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)


def maybe_download(resource: DatasetResource, dataset_root: Path, force: bool = False) -> Path:
    ensure_dir(dataset_root)
    archive_name = resource.archive_name or f"{resource.name}.zip"
    archive_path = dataset_root / archive_name
    if archive_path.exists() and not force:
        return archive_path
    if resource.manual_download:
        raise RuntimeError(
            "Dataset requires manual download due to licensing. Visit"
            f" {resource.url} and place the archive at {archive_path}"
        )
    download_file(resource.url, archive_path)
    return archive_path


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Produce dataset helper")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory where dataset archives should be stored",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print JSON metadata about known datasets",
    )
    parser.add_argument(
        "--download",
        choices=[resource.name for resource in KNOWN_DATASETS],
        help="Attempt to download an open dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if archives exist",
    )
    parsed = parser.parse_args(args=args)

    ensure_dir(parsed.dataset_root)

    if parsed.list:
        print(list_datasets(KNOWN_DATASETS))
        return

    if parsed.download:
        resource = next(res for res in KNOWN_DATASETS if res.name == parsed.download)
        archive_path = maybe_download(resource, parsed.dataset_root, force=parsed.force)
        print(json.dumps({"status": "downloaded", "path": str(archive_path)}))
        return

    print(
        json.dumps(
            {
                "status": "noop",
                "message": (
                    "No download requested. Use --list to view available datasets"
                    " or --download <name> for automated retrieval when permitted."
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
