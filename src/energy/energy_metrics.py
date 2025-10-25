"""Energy and sustainability accounting for the food-waste pipeline."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class EnergyConfig:
    training_power_w: float
    inference_power_w: float
    routing_power_w: float
    carbon_intensity_kg_per_kwh: float
    embodied_energy_kwh_per_kg: float


DEFAULT_CONFIG = EnergyConfig(
    training_power_w=120.0,
    inference_power_w=15.0,
    routing_power_w=45.0,
    carbon_intensity_kg_per_kwh=0.35,
    embodied_energy_kwh_per_kg=4.0,
)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def estimate_energy_seconds(power_w: float, duration_seconds: float) -> float:
    return power_w * duration_seconds / 3600.0


def compute_metrics(training_log: Dict, routing_log: Dict, config: EnergyConfig) -> Dict:
    training_seconds = float(training_log.get("duration_seconds", 0))
    inference_seconds = float(training_log.get("inference_seconds", 0))

    training_energy_kwh = estimate_energy_seconds(config.training_power_w, training_seconds)
    inference_energy_kwh = estimate_energy_seconds(config.inference_power_w, inference_seconds)
    routing_energy_kwh = estimate_energy_seconds(config.routing_power_w, routing_log.get("solver_seconds", 0))

    total_energy_kwh = training_energy_kwh + inference_energy_kwh + routing_energy_kwh
    total_emissions = total_energy_kwh * config.carbon_intensity_kg_per_kwh

    avoided_waste_kg = float(routing_log.get("avoided_waste_kg", 0))
    avoided_energy_kwh = avoided_waste_kg * config.embodied_energy_kwh_per_kg
    net_energy_kwh = avoided_energy_kwh - total_energy_kwh

    return {
        "training_energy_kwh": training_energy_kwh,
        "inference_energy_kwh": inference_energy_kwh,
        "routing_energy_kwh": routing_energy_kwh,
        "total_energy_kwh": total_energy_kwh,
        "total_emissions_kg": total_emissions,
        "avoided_waste_kg": avoided_waste_kg,
        "avoided_energy_kwh": avoided_energy_kwh,
        "net_energy_kwh": net_energy_kwh,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy accounting")
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--routing-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/energy_summary.json"))
    parser.add_argument("--config", type=Path, help="Optional JSON config overriding defaults")
    parser.add_argument("--inference-summary", type=Path, help="Optional JSON with inference runtime stats")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    if args.config:
        overrides = load_json(args.config)
        config = EnergyConfig(**{**DEFAULT_CONFIG.__dict__, **overrides})

    training_log = load_json(args.training_log)
    if args.inference_summary and args.inference_summary.exists():
        inference_log = load_json(args.inference_summary)
        training_log["inference_seconds"] = inference_log.get("inference_seconds", training_log.get("inference_seconds", 0))
    routing_log = load_json(args.routing_log)
    metrics = compute_metrics(training_log, routing_log, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
