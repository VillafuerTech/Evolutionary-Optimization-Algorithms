"""Reporting and artifact saving utilities."""

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from .common.types import RunResult


def generate_run_id() -> str:
    """Generate a unique run ID.

    Format: YYYYMMDD_HHMMSS_<short_hash>

    Returns:
        Unique run ID string.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    hash_input = f"{timestamp}{os.getpid()}{id(now)}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
    return f"{timestamp}_{short_hash}"


def get_environment_info() -> dict[str, Any]:
    """Get current environment information.

    Returns:
        Dictionary with environment details.
    """
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }


def save_run_artifacts(
    result: RunResult,
    reports_dir: str = "reports",
) -> Path:
    """Save all artifacts for a single run.

    Creates:
    - reports/runs/<run_id>/config.json
    - reports/runs/<run_id>/metrics.json
    - reports/runs/<run_id>/curves.csv
    - reports/runs/<run_id>/artifacts/best_solution.json
    - reports/runs/<run_id>/figures/fitness_convergence.png

    Args:
        result: Run result to save.
        reports_dir: Base reports directory.

    Returns:
        Path to the run directory.
    """
    run_dir = Path(reports_dir) / "runs" / result.run_id
    artifacts_dir = run_dir / "artifacts"
    figures_dir = run_dir / "figures"

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Save config.json
    config = {
        "run_id": result.run_id,
        "timestamp": datetime.now().isoformat(),
        "algorithm": {
            "name": result.algorithm.name,
            "params": result.algorithm.params,
        },
        "task": {
            "name": result.task.name,
            "variant": result.task.variant,
            "params": result.task.params,
        },
        "seed": result.seed,
        "environment": get_environment_info(),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save metrics.json
    metrics = {
        "best_fitness": result.metrics.best_fitness,
        "final_fitness": result.metrics.final_fitness,
        "runtime_s": result.metrics.runtime_s,
        "iterations": result.metrics.iterations,
        "extra": result.metrics.extra,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save curves.csv
    if result.history:
        with open(run_dir / "curves.csv", "w", newline="") as f:
            writer = csv.writer(f)
            # Get all keys from history
            keys = list(result.history.keys())
            writer.writerow(keys)
            # Write rows
            n_rows = len(result.history[keys[0]]) if keys else 0
            for i in range(n_rows):
                row = [result.history[k][i] for k in keys]
                writer.writerow(row)

    # Save best_solution.json
    if result.solution is not None:
        solution_data = _serialize_solution(result.solution, result.task.name)
        with open(artifacts_dir / "best_solution.json", "w") as f:
            json.dump(solution_data, f, indent=2)

    # Save convergence plot
    if result.history and "best_fitness" in result.history:
        _save_convergence_plot(
            result.history,
            figures_dir / "fitness_convergence.png",
            title=f"{result.algorithm.name.upper()} on {result.task.variant}",
        )

    return run_dir


def _serialize_solution(solution: Any, task_name: str) -> dict[str, Any]:
    """Serialize solution for JSON storage."""
    if task_name == "scheduling":
        return {"schedule": solution}
    elif task_name == "tsp":
        return {"tour": solution}
    else:
        # Continuous optimization
        if hasattr(solution, "tolist"):
            solution = solution.tolist()
        return {"position": solution}


def _save_convergence_plot(
    history: dict[str, list],
    path: Path,
    title: str = "Fitness Convergence",
) -> None:
    """Save fitness convergence plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = history.get("iter", list(range(len(history["best_fitness"]))))
    best_fitness = history["best_fitness"]

    ax.plot(iterations, best_fitness, "b-", linewidth=2, label="Best Fitness")

    if "mean_fitness" in history:
        ax.plot(
            iterations,
            history["mean_fitness"],
            "r--",
            linewidth=1,
            alpha=0.7,
            label="Mean Fitness",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def append_to_results_csv(
    result: RunResult,
    reports_dir: str = "reports",
) -> Path:
    """Append run result to results.csv.

    Creates the file with headers if it doesn't exist.

    Args:
        result: Run result to append.
        reports_dir: Base reports directory.

    Returns:
        Path to results.csv.
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    csv_path = reports_path / "results.csv"

    # Check if file exists to determine if we need headers
    write_header = not csv_path.exists()

    fieldnames = [
        "timestamp",
        "run_id",
        "task",
        "task_variant",
        "algorithm",
        "seed",
        "iterations",
        "best_fitness",
        "final_fitness",
        "runtime_s",
        "extra_metrics_json",
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        row = {
            "timestamp": datetime.now().isoformat(),
            "run_id": result.run_id,
            "task": result.task.name,
            "task_variant": result.task.variant,
            "algorithm": result.algorithm.name,
            "seed": result.seed,
            "iterations": result.metrics.iterations,
            "best_fitness": result.metrics.best_fitness,
            "final_fitness": result.metrics.final_fitness,
            "runtime_s": result.metrics.runtime_s,
            "extra_metrics_json": json.dumps(result.metrics.extra),
        }
        writer.writerow(row)

    return csv_path
