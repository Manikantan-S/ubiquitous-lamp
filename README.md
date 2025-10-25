# Food-Waste Reduction in Supply Chains using Deep Learning and Green Computing

This repository hosts an end-to-end research prototype for minimising post-harvest food loss through the combination of lightweight computer-vision quality assessment, spoilage-aware vehicle routing, and sustainability accounting.

## Overview

Food waste in fruit and vegetable supply chains is frequently caused by delayed detection of spoilage and suboptimal logistics routing. The goal of this project is to use deployable deep-learning (DL) models to assess produce quality in real time and integrate their predictions into green, energy-aware supply-chain optimisation.

Key pillars:

1. **Quality inspection at the edge** – Convolutional neural networks (CNNs) classify fruit/vegetable quality on-board low-power devices installed on trucks and at depots.
2. **Routing optimisation** – Spoilage-aware vehicle routing dynamically prioritises high-risk loads to shorten travel times and reduce loss.
3. **Sustainability metrics** – Energy consumption of the ML workloads is monitored and compared to the avoided embodied emissions from food saved.

## Repository structure

```
.
├── configs/                # Experiment configuration files
├── docs/                   # Research notes and experiment logs
├── src/
│   ├── data/               # Dataset download and preprocessing utilities
│   ├── energy/             # Energy and carbon accounting helpers
│   ├── models/             # Lightweight CNN model definitions and training loops
│   ├── pipelines/          # Command-line pipelines for training/inference
│   └── routing/            # Spoilage-aware logistics optimisation
├── requirements.txt        # Python dependencies
└── README.md
```

## Datasets

Two open datasets provide diverse coverage of fruit and vegetable spoilage states:

- **Kaggle – Fruits Fresh and Rotten for Classification** ([link](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification))
  - 3 fruit types (apple, banana, orange) with labelled fresh and rotten images.
- **Mendeley Data – Fruits and Vegetables Quality Image Dataset** ([link](https://data.mendeley.com/datasets/9sxdyb86ph/2))
  - Multiple classes captured at progressive spoilage stages, suitable for multi-class quality grading.

Download the raw image archives manually (Kaggle and Mendeley require authentication) and unpack them under `data/raw/`. The preprocessing script converts folder structures into a unified format with JSON metadata capturing spoilage labels and acquisition timestamps.

```
data/
├── raw/
│   ├── fruits-fresh-and-rotten/...
│   └── mendeley-quality/...
└── processed/
    └── spoilage_quality/
        ├── images/
        └── annotations.json
```

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare the dataset**

   ```bash
   python -m src.data.download --dataset-root data/raw
   python -m src.data.preprocess --raw-root data/raw --output-root data/processed/spoilage_quality
   ```

   - `download` reports the expected directory structure and optional automated downloads for open datasets.
   - `preprocess` cleans labels, performs quality grading (fresh / early_spoilage / spoiled), and stores manifest metadata.

3. **Train the lightweight quality model**

   ```bash
   python -m src.pipelines.train_quality \
       --config configs/experiment.yaml \
       --data-root data/processed/spoilage_quality \
       --output-dir outputs/quality_model
   ```

   The default configuration fine-tunes MobileNetV3-Small on 224×224 images with RandAugment, mixed precision, and knowledge distillation options for edge deployment.

4. **Generate spoilage predictions for routing**

   ```bash
   python -m src.pipelines.infer_quality \\
       --manifest data/processed/spoilage_quality/annotations.json \\
       --images-root data/processed/spoilage_quality \\
       --checkpoint outputs/quality_model/model.pt \\
       --locations docs/sample_locations.csv \\
       --output outputs/quality_model/predictions.csv \\
       --summary outputs/quality_model/inference_summary.json
   ```

   The script evaluates the trained network on the held-out split, joins optional logistics metadata, and exports per-location spoilage risk for downstream optimisation while logging inference runtime statistics for energy accounting.

5. **Simulate spoilage-aware routing**

   ```bash
   python -m src.routing.routing_optimizer \
       --config configs/routing.yaml \
       --predictions outputs/quality_model/predictions.csv \
       --output outputs/routing_schedule.json
   ```

   This solves a time-dependent vehicle routing problem (VRP) with perishability penalties, using the latest quality predictions to prioritise at-risk loads.

6. **Quantify sustainability impact**

   ```bash
   python -m src.energy.energy_metrics \
       --training-log outputs/quality_model/training_log.json \
       --routing-log outputs/routing_schedule.json \
       --inference-summary outputs/quality_model/inference_summary.json
   ```

   The script estimates electricity consumption, CO₂e, and avoided waste energy to compute net sustainability savings.

## Sustainability metrics

The `src/energy` module captures:

- GPU/CPU energy estimates via theoretical power profiles and training duration.
- Carbon intensity per kWh tailored to the regional grid mix (user configurable).
- Avoided embodied emissions of food saved (kWh/kg) based on literature values.

Outputs include per-experiment summaries and cumulative dashboards.

## Experiment tracking

Log files (JSON + CSV) are produced under `outputs/` to make experiments reproducible and auditable. The repository is structured for easy integration with MLflow or Weights & Biases if desired.

## Citation

If you use this project in academic work, please cite it as:

> Food-Waste Reduction in Supply Chains using Deep Learning and Green Computing, 2024. GitHub: https://github.com/your-org/ubiquitous-lamp

