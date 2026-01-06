"""Ant Colony Optimization implementation."""

import random
from typing import Any

import numpy as np


class AntColonyOptimization:
    """Ant Colony Optimization for TSP and combinatorial problems."""

    def __init__(
        self,
        n_ants: int = 50,
        alpha: float = 1.0,
        beta: float = 5.0,
        evaporation_rate: float = 0.5,
        initial_pheromone: float = 1.0,
    ):
        """Initialize ACO parameters.

        Args:
            n_ants: Number of ants per iteration.
            alpha: Pheromone influence factor.
            beta: Heuristic information influence factor.
            evaporation_rate: Rate of pheromone evaporation.
            initial_pheromone: Initial pheromone level on edges.
        """
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.initial_pheromone = initial_pheromone

    def run(
        self,
        task: Any,
        seed: int,
        iterations: int = 100,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, list], dict[str, Any]]:
        """Run ACO on a task.

        Args:
            task: Task object with distance_matrix and n_cities.
            seed: Random seed for reproducibility.
            iterations: Number of iterations.
            **kwargs: Additional algorithm parameters.

        Returns:
            Tuple of (best_solution, history, metadata).
        """
        random.seed(seed)
        np.random.seed(seed)

        # Override defaults with kwargs
        n_ants = kwargs.get("n_ants", self.n_ants)
        alpha = kwargs.get("alpha", self.alpha)
        beta = kwargs.get("beta", self.beta)
        evap_rate = kwargs.get("evaporation_rate", self.evaporation_rate)
        init_pher = kwargs.get("initial_pheromone", self.initial_pheromone)

        n_cities = task.n_cities
        distance_matrix = task.distance_matrix

        # Initialize pheromone and heuristic matrices
        pheromone = [[init_pher for _ in range(n_cities)] for _ in range(n_cities)]
        heuristic = [
            [1.0 / distance_matrix[i][j] if i != j and distance_matrix[i][j] > 0 else 0
             for j in range(n_cities)]
            for i in range(n_cities)
        ]

        best_tour = None
        best_distance = float("inf")

        history: dict[str, list] = {"iter": [], "best_fitness": [], "mean_fitness": []}

        for iteration in range(iterations):
            all_tours = []
            all_distances = []

            for _ in range(n_ants):
                tour = self._construct_tour(
                    n_cities, pheromone, heuristic, alpha, beta
                )
                distance = self._calculate_tour_distance(tour, distance_matrix)
                all_tours.append(tour)
                all_distances.append(distance)

                if distance < best_distance:
                    best_distance = distance
                    best_tour = tour[:]

            # Update pheromones
            self._update_pheromones(
                pheromone, all_tours, all_distances, evap_rate, n_cities
            )

            mean_dist = np.mean(all_distances)
            history["iter"].append(iteration)
            history["best_fitness"].append(best_distance)
            history["mean_fitness"].append(mean_dist)

        meta = {
            "n_ants": n_ants,
            "alpha": alpha,
            "beta": beta,
            "evaporation_rate": evap_rate,
            "n_cities": n_cities,
        }

        return best_tour, history, meta

    def _construct_tour(
        self,
        n_cities: int,
        pheromone: list[list[float]],
        heuristic: list[list[float]],
        alpha: float,
        beta: float,
    ) -> list[int]:
        """Construct a tour for one ant."""
        tour = []
        unvisited = set(range(n_cities))
        current = random.choice(list(unvisited))
        tour.append(current)
        unvisited.remove(current)

        while unvisited:
            probabilities = []
            total = 0.0

            for node in unvisited:
                pher = pheromone[current][node] ** alpha
                heur = heuristic[current][node] ** beta
                prob = pher * heur
                probabilities.append((node, prob))
                total += prob

            if total == 0:
                next_node = random.choice(list(unvisited))
            else:
                # Roulette wheel selection
                r = random.uniform(0, total)
                cumulative = 0.0
                next_node = probabilities[-1][0]  # default
                for node, prob in probabilities:
                    cumulative += prob
                    if cumulative >= r:
                        next_node = node
                        break

            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        return tour

    def _calculate_tour_distance(
        self, tour: list[int], distance_matrix: list[list[float]]
    ) -> float:
        """Calculate total tour distance."""
        total = 0.0
        for i in range(len(tour) - 1):
            total += distance_matrix[tour[i]][tour[i + 1]]
        total += distance_matrix[tour[-1]][tour[0]]
        return total

    def _update_pheromones(
        self,
        pheromone: list[list[float]],
        all_tours: list[list[int]],
        all_distances: list[float],
        evap_rate: float,
        n_cities: int,
    ) -> None:
        """Update pheromone levels."""
        # Evaporation
        for i in range(n_cities):
            for j in range(n_cities):
                pheromone[i][j] *= 1 - evap_rate
                if pheromone[i][j] < 1e-10:
                    pheromone[i][j] = 1e-10

        # Add new pheromone
        for tour, distance in zip(all_tours, all_distances):
            contribution = 1.0 / distance
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i + 1]
                pheromone[a][b] += contribution
                pheromone[b][a] += contribution
            # Return edge
            a, b = tour[-1], tour[0]
            pheromone[a][b] += contribution
            pheromone[b][a] += contribution
