"""Benchmark functions for continuous optimization."""

from typing import Any

import numpy as np


class BealeFunction:
    """Beale benchmark function.

    2D function with global minimum at (3, 0.5) with f(x) = 0.
    Search domain: [-4.5, 4.5] x [-4.5, 4.5]
    """

    task_type = "continuous"
    optimization_type = "min"
    name = "beale"
    global_optimum = 0.0
    global_optimum_position = [3.0, 0.5]

    def __init__(self, bounds: list[tuple[float, float]] | None = None):
        """Initialize Beale function.

        Args:
            bounds: Search bounds for each dimension. Defaults to [-4.5, 4.5].
        """
        self.bounds = bounds or [(-4.5, 4.5), (-4.5, 4.5)]
        self.n_dims = 2

    def evaluate(self, x: list[float] | np.ndarray) -> float:
        """Evaluate the Beale function.

        Args:
            x: 2D point [x1, x2].

        Returns:
            Function value at x.
        """
        x1, x2 = x[0], x[1]
        term1 = (1.5 - x1 + x1 * x2) ** 2
        term2 = (2.25 - x1 + x1 * x2**2) ** 2
        term3 = (2.625 - x1 + x1 * x2**3) ** 2
        return term1 + term2 + term3

    def get_config(self) -> dict[str, Any]:
        """Get task configuration."""
        return {
            "name": self.name,
            "bounds": self.bounds,
            "n_dims": self.n_dims,
            "global_optimum": self.global_optimum,
        }


class EasomFunction:
    """Easom benchmark function.

    2D function with global minimum at (pi, pi) with f(x) = -1.
    Search domain: [-5, 5] x [-5, 5] (or larger)
    """

    task_type = "continuous"
    optimization_type = "min"
    name = "easom"
    global_optimum = -1.0
    global_optimum_position = [np.pi, np.pi]

    def __init__(self, bounds: list[tuple[float, float]] | None = None):
        """Initialize Easom function.

        Args:
            bounds: Search bounds for each dimension. Defaults to [-5, 5].
        """
        self.bounds = bounds or [(-5.0, 5.0), (-5.0, 5.0)]
        self.n_dims = 2

    def evaluate(self, x: list[float] | np.ndarray) -> float:
        """Evaluate the Easom function.

        Args:
            x: 2D point [x1, x2].

        Returns:
            Function value at x.
        """
        x1, x2 = x[0], x[1]
        return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2 + (x2 - np.pi) ** 2))

    def get_config(self) -> dict[str, Any]:
        """Get task configuration."""
        return {
            "name": self.name,
            "bounds": self.bounds,
            "n_dims": self.n_dims,
            "global_optimum": self.global_optimum,
        }


class RastriginFunction:
    """Rastrigin benchmark function.

    N-D function with global minimum at (0, 0, ..., 0) with f(x) = 0.
    Search domain: [-5.12, 5.12]^n
    """

    task_type = "continuous"
    optimization_type = "min"
    name = "rastrigin"
    global_optimum = 0.0

    def __init__(
        self,
        n_dims: int = 10,
        bounds: list[tuple[float, float]] | None = None,
    ):
        """Initialize Rastrigin function.

        Args:
            n_dims: Number of dimensions.
            bounds: Search bounds for each dimension. Defaults to [-5.12, 5.12].
        """
        self.n_dims = n_dims
        self.bounds = bounds or [(-5.12, 5.12)] * n_dims
        self.global_optimum_position = [0.0] * n_dims

    def evaluate(self, x: list[float] | np.ndarray) -> float:
        """Evaluate the Rastrigin function.

        Args:
            x: N-D point.

        Returns:
            Function value at x.
        """
        x = np.array(x)
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def get_config(self) -> dict[str, Any]:
        """Get task configuration."""
        return {
            "name": self.name,
            "bounds": self.bounds,
            "n_dims": self.n_dims,
            "global_optimum": self.global_optimum,
        }
