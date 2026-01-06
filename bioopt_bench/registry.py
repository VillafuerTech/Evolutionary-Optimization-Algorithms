"""Algorithm and task registry."""

from typing import Any

from .algorithms import (
    GeneticAlgorithm,
    ParticleSwarmOptimization,
    AntColonyOptimization,
    GreyWolfOptimizer,
)
from .tasks import (
    BealeFunction,
    EasomFunction,
    RastriginFunction,
    TSPTask,
    SchedulingTask,
)


# Algorithm registry
ALGORITHMS: dict[str, type] = {
    "ga": GeneticAlgorithm,
    "pso": ParticleSwarmOptimization,
    "aco": AntColonyOptimization,
    "gwo": GreyWolfOptimizer,
}

# Task registry - organized by task type
TASKS: dict[str, dict[str, type]] = {
    "functions": {
        "beale": BealeFunction,
        "easom": EasomFunction,
        "rastrigin": RastriginFunction,
    },
    "tsp": {
        "random": TSPTask,
        "grid": TSPTask,
    },
    "scheduling": {
        "default": SchedulingTask,
    },
}

# Default algorithm-task compatibility
ALGORITHM_TASK_COMPATIBILITY: dict[str, list[str]] = {
    "ga": ["functions", "tsp"],
    "pso": ["functions"],
    "aco": ["tsp"],
    "gwo": ["scheduling", "functions"],
}

# Suite definitions
SUITES: dict[str, list[dict[str, Any]]] = {
    "default": [
        # Function benchmarks
        {"task": "functions", "variant": "beale", "algo": "ga", "iters": 200},
        {"task": "functions", "variant": "beale", "algo": "pso", "iters": 200},
        {"task": "functions", "variant": "easom", "algo": "ga", "iters": 200},
        {"task": "functions", "variant": "easom", "algo": "pso", "iters": 200},
        {"task": "functions", "variant": "rastrigin", "algo": "pso", "iters": 200, "params": {"dim": 10}},
        # TSP benchmarks
        {"task": "tsp", "variant": "random", "algo": "ga", "iters": 300, "params": {"n_cities": 50}},
        {"task": "tsp", "variant": "random", "algo": "aco", "iters": 100, "params": {"n_cities": 50}},
        {"task": "tsp", "variant": "grid", "algo": "ga", "iters": 300, "params": {"n_cities": 25}},
        {"task": "tsp", "variant": "grid", "algo": "aco", "iters": 100, "params": {"n_cities": 25}},
        # Scheduling benchmarks
        {"task": "scheduling", "variant": "default", "algo": "gwo", "iters": 300},
    ],
    "functions-only": [
        {"task": "functions", "variant": "beale", "algo": "ga", "iters": 200},
        {"task": "functions", "variant": "beale", "algo": "pso", "iters": 200},
        {"task": "functions", "variant": "easom", "algo": "ga", "iters": 200},
        {"task": "functions", "variant": "easom", "algo": "pso", "iters": 200},
        {"task": "functions", "variant": "rastrigin", "algo": "pso", "iters": 200, "params": {"dim": 10}},
        {"task": "functions", "variant": "rastrigin", "algo": "gwo", "iters": 200, "params": {"dim": 10}},
    ],
    "tsp-only": [
        {"task": "tsp", "variant": "random", "algo": "ga", "iters": 300, "params": {"n_cities": 50}},
        {"task": "tsp", "variant": "random", "algo": "aco", "iters": 100, "params": {"n_cities": 50}},
        {"task": "tsp", "variant": "grid", "algo": "ga", "iters": 300, "params": {"n_cities": 25}},
        {"task": "tsp", "variant": "grid", "algo": "aco", "iters": 100, "params": {"n_cities": 25}},
    ],
    "scheduling-only": [
        {"task": "scheduling", "variant": "default", "algo": "gwo", "iters": 500},
    ],
    "quick": [
        {"task": "functions", "variant": "beale", "algo": "pso", "iters": 50},
        {"task": "tsp", "variant": "random", "algo": "aco", "iters": 20, "params": {"n_cities": 20}},
        {"task": "scheduling", "variant": "default", "algo": "gwo", "iters": 50},
    ],
}


def get_algorithm(name: str, **kwargs: Any) -> Any:
    """Get an algorithm instance by name.

    Args:
        name: Algorithm name (ga, pso, aco, gwo).
        **kwargs: Algorithm initialization parameters.

    Returns:
        Algorithm instance.

    Raises:
        ValueError: If algorithm name is unknown.
    """
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[name](**kwargs)


def get_task(task_type: str, variant: str, **kwargs: Any) -> Any:
    """Get a task instance by type and variant.

    Args:
        task_type: Task type (functions, tsp, scheduling).
        variant: Task variant (e.g., beale, random, default).
        **kwargs: Task initialization parameters.

    Returns:
        Task instance.

    Raises:
        ValueError: If task type or variant is unknown.
    """
    if task_type not in TASKS:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(TASKS.keys())}")
    if variant not in TASKS[task_type]:
        raise ValueError(
            f"Unknown variant '{variant}' for task '{task_type}'. "
            f"Available: {list(TASKS[task_type].keys())}"
        )

    task_class = TASKS[task_type][variant]

    # Handle TSP variants
    if task_type == "tsp":
        kwargs["graph_type"] = variant

    return task_class(**kwargs)


def get_suite(name: str) -> list[dict[str, Any]]:
    """Get a benchmark suite by name.

    Args:
        name: Suite name.

    Returns:
        List of benchmark configurations.

    Raises:
        ValueError: If suite name is unknown.
    """
    if name not in SUITES:
        raise ValueError(f"Unknown suite: {name}. Available: {list(SUITES.keys())}")
    return SUITES[name]


def list_available() -> dict[str, Any]:
    """List all available algorithms, tasks, and suites.

    Returns:
        Dictionary with available options.
    """
    return {
        "algorithms": list(ALGORITHMS.keys()),
        "tasks": {k: list(v.keys()) for k, v in TASKS.items()},
        "suites": list(SUITES.keys()),
        "compatibility": ALGORITHM_TASK_COMPATIBILITY,
    }
