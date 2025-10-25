# Research Plan: Food-Waste Reduction via DL and Green Computing

## Objectives

1. Build an edge-deployable CNN to grade produce quality (fresh, early spoilage, spoiled).
2. Integrate real-time quality scores into spoilage-aware routing optimisation.
3. Quantify net sustainability impact by comparing ML energy use with avoided food waste emissions.

## Key hypotheses

- Lightweight CNNs (≤5M parameters) can achieve ≥90% F1 score on multi-class quality grading when trained on combined Kaggle + Mendeley datasets.
- Routing that prioritises loads with predicted low remaining shelf life will reduce expected waste by ≥15% compared with distance-only VRP baselines.
- The embodied energy of saved produce outweighs compute energy by at least 3× under realistic operating conditions.

## Experimental timeline

| Phase | Description | Deliverables |
|-------|-------------|--------------|
| Data consolidation | Curate consistent labels and metadata across datasets | Processed dataset manifest |
| Modelling | Train/fine-tune lightweight CNNs (MobileNetV3, EfficientNet-Lite) | Model checkpoints, evaluation metrics |
| Deployment simulation | Run inference benchmarks on Jetson Nano / Raspberry Pi 4 | Latency & energy measurements |
| Routing optimisation | Simulate 7-day supply-chain scenarios with VRP solver | Routing schedules, waste reduction metrics |
| Sustainability accounting | Estimate energy use vs. avoided waste | Energy and CO₂e balance report |

## Evaluation metrics

- **Quality model**: Accuracy, F1 (macro), balanced accuracy, latency on ARM Cortex-A57 @ 1.4GHz.
- **Routing**: Total waste (kg), average delivery delay (minutes), vehicle kilometres, computational time.
- **Sustainability**: kWh consumed, kg CO₂e emitted/saved, net energy balance.

## Risks and mitigations

- *Dataset bias*: Combine multiple sources, augment with synthetic lighting/occlusion transformations.
- *Edge deployment constraints*: Use quantisation-aware training and knowledge distillation to smaller student networks.
- *Solver scalability*: Implement heuristics (savings algorithm) before OR-Tools VRP for large fleets.
- *Energy estimation accuracy*: Cross-validate theoretical estimates with power-meter readings when available.

