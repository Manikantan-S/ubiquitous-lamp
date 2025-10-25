# Energy and Sustainability Accounting Methodology

This document describes the assumptions underpinning `src/energy/energy_metrics.py`.

## Inputs

- `training_log.json`: produced by the training pipeline and expected to include `duration_seconds` and `inference_seconds` fields. `inference_seconds` can be overridden by the standalone inference summary JSON.
- `inference_summary.json`: optional runtime summary written by `src/pipelines/infer_quality.py`.
- `routing_schedule.json`: produced by the routing optimiser and enriched with `solver_seconds` and `avoided_waste_kg` for sustainability analysis.

## Device power profiles

| Component | Typical hardware | Power (W) |
|-----------|------------------|-----------|
| Training GPU | NVIDIA RTX 3060 (underclocked for efficiency) | 120 |
| Edge inference device | NVIDIA Jetson Xavier NX | 15 |
| Routing solver server | 12-core CPU node | 45 |

These values can be overridden via a JSON config passed to `energy_metrics.py`.

## Carbon intensity

Default grid intensity is set to 0.35 kg CO₂e / kWh, reflecting the 2023 average for the European Union. Users can override this constant to match their deployment locale.

## Embodied energy of produce

The model assumes 4 kWh per kilogram of fruits/vegetables, combining upstream agricultural energy, cold-chain electricity, and retail refrigeration requirements. Literature references include:

- FAO (2013): *Food Wastage Footprint: Impacts on Natural Resources*.
- G. Heller et al. (2019): *Energy use in food supply chains*.

## Output metrics

- **Total energy (kWh)** consumed by ML training, inference, and routing computation.
- **Total emissions (kg CO₂e)** using the configured grid intensity.
- **Avoided energy (kWh)** from prevented waste (predicted vs. baseline).
- **Net energy (kWh)** = avoided - consumed, highlighting the sustainability pay-off.

