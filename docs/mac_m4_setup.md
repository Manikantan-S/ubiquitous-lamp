# macOS M4 Setup and Experiment Playbook

This guide describes how to reproduce the food-waste reduction pipeline on an Apple Silicon Mac with an M4 chip. It highlights the macOS-specific tooling required to unlock Metal Performance Shader (MPS) acceleration and points out the metrics that the repository emits for research reporting.

## 1. System prerequisites

| Requirement | Recommendation |
|-------------|----------------|
| macOS version | macOS 14.4 or later (Metal 3 runtime and Python universal binaries) |
| Command line tools | `xcode-select --install` (once per machine) |
| Storage | ≥ 35 GB free (≈ 12 GB for datasets, 8 GB for processed data, 5 GB for checkpoints/logs) |
| Memory | ≥ 16 GB RAM (MobileNetV3 fine-tuning with batch size 64 stays within 10 GB) |

## 2. Install a native Python distribution

The most reliable path on Apple Silicon is Miniforge:

```bash
/bin/bash -c "$(curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)"
conda init zsh  # restart the shell afterwards
```

Create and activate a fresh environment with Python 3.11:

```bash
conda create -n foodwaste python=3.11 -y
conda activate foodwaste
```

## 3. Clone the repository

```bash
git clone https://github.com/your-org/ubiquitous-lamp.git
cd ubiquitous-lamp
```

## 4. Install dependencies with MPS support

Upgrade `pip` and install the core requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`torch` 2.1+ ships universal wheels that include MPS kernels. Confirm that the runtime sees the Apple GPU:

```bash
python - <<'PY'
import torch
print({
    "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    "mps_built": torch.backends.mps.is_built(),
    "version": torch.__version__,
})
PY
```

If `mps_available` is `False`, export `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow the training loop to fall back to CPU ops while still dispatching supported layers to the GPU.

## 5. Acquire and stage datasets

1. Download **Fruits Fresh and Rotten for Classification** from Kaggle and **Fruits and Vegetables Quality Image Dataset** from Mendeley Data.
2. Unzip the archives under `data/raw/fruits-fresh-and-rotten/` and `data/raw/mendeley-quality/`.
3. Verify the placement:

```bash
python -m src.data.download --dataset-root data/raw
```

For quick descriptive statistics after preprocessing, run:

```bash
python - <<'PY'
from collections import Counter
import json
from pathlib import Path
manifest = json.loads(Path("data/processed/spoilage_quality/annotations.json").read_text())
labels = Counter(item["label"] for item in manifest)
splits = Counter(item["split"] for item in manifest)
print({"total_images": len(manifest), "labels": labels, "splits": splits})
PY
```

## 6. Preprocess into the unified taxonomy

```bash
python -m src.data.preprocess \
    --raw-root data/raw \
    --output-root data/processed/spoilage_quality
```

The script normalises every image into RGB, maps raw labels into `fresh`, `early_spoilage`, and `spoiled`, and shuffles a 70/15/15 train/val/test split while writing an `annotations.json` manifest and ready-to-train folder structure.【F:src/data/preprocess.py†L22-L105】

## 7. Train the MobileNetV3 quality model on Apple Silicon

The training configuration defaults to 15 epochs, batch size 64, image resolution 224×224, AdamW with a cosine schedule, and uses the pre-trained MobileNetV3-Small backbone.【F:configs/experiment.yaml†L1-L11】【F:src/models/quality_model.py†L21-L87】

Enable GPU fallbacks and launch the pipeline:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m src.pipelines.train_quality \
    --config configs/experiment.yaml \
    --data-root data/processed/spoilage_quality \
    --output-dir outputs/quality_model
```

Key artefacts for your paper:

- `outputs/quality_model/training_log.json` – includes epoch-wise accuracy/loss, the resolved config, detected device (`mps`, `cuda`, or `cpu`), and total training duration in seconds.【F:src/models/quality_model.py†L89-L134】
- `outputs/quality_model/test_metrics.json` – final hold-out loss and accuracy for headline reporting.【F:src/models/quality_model.py†L129-L134】

## 8. Generate inference summaries on the held-out split

```bash
python -m src.pipelines.infer_quality \
    --manifest data/processed/spoilage_quality/annotations.json \
    --images-root data/processed/spoilage_quality \
    --checkpoint outputs/quality_model/model.pt \
    --locations docs/sample_locations.csv \
    --output outputs/quality_model/predictions.csv \
    --summary outputs/quality_model/inference_summary.json
```

The command reuses the same device selector as training, logs wall-clock inference seconds, and exports per-sample confidences alongside optional logistics metadata for routing.【F:src/pipelines/infer_quality.py†L18-L114】

## 9. Optimise spoilage-aware routing

```bash
python -m src.routing.routing_optimizer \
    --config configs/routing.yaml \
    --predictions outputs/quality_model/predictions.csv \
    --output outputs/routing_schedule.json
```

The solver blends spoilage scores, travel durations, and capacity constraints to prioritise at-risk loads, emitting energy-aware runtime statistics and estimated kilograms of avoided waste that feed into the sustainability analysis.【F:src/routing/routing_optimizer.py†L1-L203】

## 10. Quantify green computing impact

```bash
python -m src.energy.energy_metrics \
    --training-log outputs/quality_model/training_log.json \
    --routing-log outputs/routing_schedule.json \
    --inference-summary outputs/quality_model/inference_summary.json
```

The report combines assumed device power draws (e.g., 120 W GPU training, 15 W edge inference, 45 W routing CPU) with recorded runtimes to compute consumed kWh, kg CO₂e, avoided energy per kilogram of saved produce, and the resulting net energy savings.【F:docs/energy_methodology.md†L11-L33】【F:src/energy/energy_metrics.py†L1-L94】

## 11. Suggested figures for publication

| Metric | Source | How to cite |
|--------|--------|-------------|
| Training accuracy / loss curves | `outputs/quality_model/training_log.json` | Plot epochs vs. accuracy for deployment readiness. |
| Final test accuracy | `outputs/quality_model/test_metrics.json` | Report as overall classification accuracy on the held-out split. |
| Mean inference latency & throughput | `outputs/quality_model/inference_summary.json` | Convert `inference_seconds` to samples/sec for edge deployment discussion. |
| Avoided waste (kg) and routing solver time | `outputs/routing_schedule.json` | Compare to baseline routing or historical spoilage rates. |
| Energy consumed vs. saved | Output of `energy_metrics` command | Discuss net sustainability delta alongside carbon intensity assumption (default 0.35 kg CO₂e/kWh). |

Documenting the above artefacts alongside dataset provenance ensures that the Mac M4 reproduction is traceable and ready for inclusion in academic or industry reports.
