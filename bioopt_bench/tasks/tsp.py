"""TSP (Travelling Salesman Problem) task module."""

import random
from typing import Any

import numpy as np


class TSPTask:
    """TSP benchmark task.

    Supports random and grid-based city configurations.
    """

    task_type = "tsp"
    optimization_type = "min"
    name = "tsp"

    def __init__(
        self,
        nodes: list[tuple[float, float]] | None = None,
        n_cities: int = 50,
        graph_type: str = "random",
        area: float = 100.0,
        grid_spacing: float = 10.0,
        seed: int | None = None,
    ):
        """Initialize TSP task.

        Args:
            nodes: Pre-defined list of node coordinates.
            n_cities: Number of cities (used if nodes not provided).
            graph_type: Type of graph - 'random' or 'grid'.
            area: Area size for random graph generation.
            grid_spacing: Spacing between nodes for grid graph.
            seed: Random seed for node generation.
        """
        if nodes is not None:
            self.nodes = nodes
        elif graph_type == "grid":
            grid_size = int(np.sqrt(n_cities))
            self.nodes = self._generate_grid(grid_size, grid_spacing)
        else:
            if seed is not None:
                random.seed(seed)
            self.nodes = self._generate_random(n_cities, area)

        self.n_cities = len(self.nodes)
        self.graph_type = graph_type
        self.distance_matrix = self._calculate_distance_matrix()

    def _generate_random(
        self, n_cities: int, area: float
    ) -> list[tuple[float, float]]:
        """Generate random node positions."""
        return [(random.uniform(0, area), random.uniform(0, area)) for _ in range(n_cities)]

    def _generate_grid(
        self, grid_size: int, spacing: float
    ) -> list[tuple[float, float]]:
        """Generate grid-based node positions."""
        nodes = []
        for i in range(grid_size):
            for j in range(grid_size):
                nodes.append((i * spacing, j * spacing))
        return nodes

    def _calculate_distance_matrix(self) -> list[list[float]]:
        """Calculate pairwise distance matrix."""
        n = self.n_cities
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dx = self.nodes[i][0] - self.nodes[j][0]
                dy = self.nodes[i][1] - self.nodes[j][1]
                matrix[i][j] = np.sqrt(dx**2 + dy**2)
        return matrix

    def evaluate(self, tour: list[int]) -> float:
        """Evaluate a tour (route).

        Args:
            tour: List of city indices representing the tour order.

        Returns:
            Total tour distance.
        """
        total = 0.0
        for i in range(len(tour) - 1):
            total += self.distance_matrix[tour[i]][tour[i + 1]]
        total += self.distance_matrix[tour[-1]][tour[0]]
        return total

    def get_config(self) -> dict[str, Any]:
        """Get task configuration."""
        return {
            "name": self.name,
            "n_cities": self.n_cities,
            "graph_type": self.graph_type,
        }

    def get_nodes_as_dict(self) -> list[dict[str, Any]]:
        """Get nodes as list of dicts for serialization."""
        return [
            {"index": i, "x": x, "y": y}
            for i, (x, y) in enumerate(self.nodes)
        ]
