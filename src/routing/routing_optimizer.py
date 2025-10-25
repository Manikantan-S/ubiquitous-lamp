"""Spoilage-aware vehicle routing optimisation."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


@dataclass
class Location:
    location_id: str
    latitude: float
    longitude: float
    demand_kg: float
    spoilage_score: float
    service_minutes: float


@dataclass
class RoutingResult:
    routes: List[List[str]]
    total_distance_km: float
    expected_waste_kg: float


def load_locations(predictions_path: Path) -> List[Location]:
    df = pd.read_csv(predictions_path)
    return [
        Location(
            location_id=str(row["location_id"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            demand_kg=float(row["demand_kg"]),
            spoilage_score=float(row["spoilage_score"]),
            service_minutes=float(row.get("service_minutes", 10.0)),
        )
        for _, row in df.iterrows()
    ]


def create_distance_matrix(locations: List[Location]) -> np.ndarray:
    coords = np.array([[loc.latitude, loc.longitude] for loc in locations])
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])
    diff_lat = lat_rad[:, None] - lat_rad[None, :]
    diff_lon = lon_rad[:, None] - lon_rad[None, :]
    a = (
        np.sin(diff_lat / 2) ** 2
        + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(diff_lon / 2) ** 2
    )
    earth_radius_km = 6371.0
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a + 1e-12))
    distances = earth_radius_km * c
    return distances


def build_data_model(locations: List[Location], config) -> Dict:
    distance_matrix = create_distance_matrix(locations)
    time_matrix = (distance_matrix / config.routing.average_speed_kmph * 60).astype(int)
    demands = [int(loc.demand_kg) for loc in locations]
    vehicle_capacities = [int(cap) for cap in config.routing.vehicle_capacities]
    depot_index = 0
    time_windows = []
    for loc in locations:
        base = int(config.routing.horizon_minutes * (1.0 - loc.spoilage_score))
        time_windows.append((0, max(base, 30)))
    return {
        "distance_matrix": distance_matrix,
        "time_matrix": time_matrix,
        "demands": demands,
        "vehicle_capacities": vehicle_capacities,
        "depot": depot_index,
        "time_windows": time_windows,
        "service_times": [int(loc.service_minutes) for loc in locations],
        "locations": locations,
    }


def solve_vrp(data: Dict, config) -> Tuple[RoutingResult, float, float]:
    start_time = time.time()
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), len(data["vehicle_capacities"]), data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    distance_callback = lambda from_index, to_index: int(
        data["distance_matrix"][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)] * 1000
    )
    transit_callback = lambda from_index, to_index: int(
        data["time_matrix"][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
        + data["service_times"][manager.IndexToNode(from_index)]
    )

    distance_callback_idx = routing.RegisterTransitCallback(distance_callback)
    time_callback_idx = routing.RegisterTransitCallback(transit_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_idx)
    routing.AddDimensionWithVehicleTransitAndCapacity(
        distance_callback_idx,
        0,
        [int(config.routing.max_distance_km * 1000)] * len(data["vehicle_capacities"]),
        True,
        "Distance",
    )

    routing.AddDimensionWithVehicleTransitAndCapacity(
        time_callback_idx,
        config.routing.time_buffer_minutes,
        [int(config.routing.horizon_minutes)] * len(data["vehicle_capacities"]),
        False,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    for node_idx, window in enumerate(data["time_windows"]):
        index = manager.NodeToIndex(node_idx)
        time_dimension.CumulVar(index).SetRange(window[0], window[1])
        time_dimension.SlackVar(index).SetValue(0)
        routing.AddToAssignment(time_dimension.SlackVar(index))

    demand_callback_idx = routing.RegisterUnaryTransitCallback(lambda index: data["demands"][manager.IndexToNode(index)])
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_idx,
        0,
        data["vehicle_capacities"],
        True,
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(config.routing.solver_time_limit_s)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("No routing solution found")

    total_distance = 0.0
    routes: List[List[str]] = []
    for vehicle_id in range(len(data["vehicle_capacities"])):
        index = routing.Start(vehicle_id)
        route_distance = 0.0
        vehicle_route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            vehicle_route.append(data["locations"][node_index].location_id)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += data["distance_matrix"][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
        routes.append(vehicle_route)
        total_distance += route_distance

    expected_waste = 0.0
    baseline_waste = 0.0
    for loc in data["locations"]:
        expected_waste += loc.demand_kg * loc.spoilage_score * config.routing.waste_factor
        baseline_waste += loc.demand_kg * config.routing.baseline_waste_factor

    avoided_waste = max(baseline_waste - expected_waste, 0.0)
    solver_seconds = time.time() - start_time

    return (
        RoutingResult(routes=routes, total_distance_km=total_distance, expected_waste_kg=expected_waste),
        avoided_waste,
        solver_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Spoilage-aware routing optimisation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    locations = load_locations(args.predictions)
    data = build_data_model(locations, config)
    result, avoided_waste, solver_seconds = solve_vrp(data, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "routes": result.routes,
                "total_distance_km": result.total_distance_km,
                "expected_waste_kg": result.expected_waste_kg,
                "avoided_waste_kg": avoided_waste,
                "solver_seconds": solver_seconds,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
