"""Type definitions for bioopt_bench."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlgorithmConfig:
    """Configuration for an optimization algorithm."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    """Configuration for a benchmark task."""

    name: str
    variant: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMetrics:
    """Metrics collected during a benchmark run."""

    best_fitness: float
    final_fitness: float
    runtime_s: float
    iterations: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result from a single benchmark run."""

    run_id: str
    algorithm: AlgorithmConfig
    task: TaskConfig
    seed: int
    metrics: BenchmarkMetrics
    history: dict[str, list[float]] = field(default_factory=dict)
    solution: Any = None
