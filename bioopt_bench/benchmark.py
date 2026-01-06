"""Benchmark orchestrator."""

import time
from typing import Any

from .common.seeding import set_seed
from .common.types import AlgorithmConfig, BenchmarkMetrics, RunResult, TaskConfig
from .registry import get_algorithm, get_suite, get_task
from .reporting import append_to_results_csv, generate_run_id, save_run_artifacts


def run_one(
    task_type: str,
    task_variant: str,
    algo_name: str,
    iterations: int = 100,
    seed: int = 42,
    task_params: dict[str, Any] | None = None,
    algo_params: dict[str, Any] | None = None,
    save: bool = False,
    reports_dir: str = "reports",
) -> RunResult:
    """Run a single benchmark.

    Args:
        task_type: Type of task (functions, tsp, scheduling).
        task_variant: Task variant (beale, random, default, etc.).
        algo_name: Algorithm name (ga, pso, aco, gwo).
        iterations: Number of iterations.
        seed: Random seed.
        task_params: Additional task parameters.
        algo_params: Additional algorithm parameters.
        save: Whether to save artifacts.
        reports_dir: Directory for saving reports.

    Returns:
        RunResult with all benchmark data.
    """
    task_params = task_params or {}
    algo_params = algo_params or {}

    # Set seed for reproducibility
    set_seed(seed)

    # Create task and algorithm
    task = get_task(task_type, task_variant, **task_params)
    algorithm = get_algorithm(algo_name, **algo_params)

    # Run benchmark
    start_time = time.perf_counter()
    solution, history, meta = algorithm.run(task, seed, iterations=iterations, **algo_params)
    end_time = time.perf_counter()
    runtime = end_time - start_time

    # Calculate final fitness
    if hasattr(task, "evaluate"):
        if task_type == "scheduling":
            final_fitness = task.evaluate(solution)
        else:
            final_fitness = task.evaluate(solution)
    else:
        final_fitness = history["best_fitness"][-1] if history.get("best_fitness") else 0.0

    # Build result
    run_id = generate_run_id()

    algo_config = AlgorithmConfig(name=algo_name, params=algo_params)
    task_config = TaskConfig(name=task_type, variant=task_variant, params=task_params)

    metrics = BenchmarkMetrics(
        best_fitness=history["best_fitness"][-1] if history.get("best_fitness") else final_fitness,
        final_fitness=final_fitness,
        runtime_s=runtime,
        iterations=iterations,
        extra=meta,
    )

    result = RunResult(
        run_id=run_id,
        algorithm=algo_config,
        task=task_config,
        seed=seed,
        metrics=metrics,
        history=history,
        solution=solution,
    )

    # Save artifacts if requested
    if save:
        save_run_artifacts(result, reports_dir)
        append_to_results_csv(result, reports_dir)

    return result


def run_suite(
    suite_name: str,
    repeat: int = 1,
    base_seed: int = 42,
    save: bool = True,
    reports_dir: str = "reports",
    verbose: bool = True,
) -> list[RunResult]:
    """Run a benchmark suite.

    Args:
        suite_name: Name of the suite to run.
        repeat: Number of repetitions per benchmark (with different seeds).
        base_seed: Starting seed value.
        save: Whether to save artifacts.
        reports_dir: Directory for saving reports.
        verbose: Whether to print progress.

    Returns:
        List of all run results.
    """
    suite = get_suite(suite_name)
    results = []

    total_runs = len(suite) * repeat
    run_count = 0

    for bench_config in suite:
        task_type = bench_config["task"]
        task_variant = bench_config["variant"]
        algo_name = bench_config["algo"]
        iterations = bench_config.get("iters", 100)
        task_params = bench_config.get("params", {})
        algo_params = bench_config.get("algo_params", {})

        for rep in range(repeat):
            seed = base_seed + rep
            run_count += 1

            if verbose:
                print(
                    f"[{run_count}/{total_runs}] "
                    f"{algo_name.upper()} on {task_variant} "
                    f"(seed={seed}, iters={iterations})"
                )

            result = run_one(
                task_type=task_type,
                task_variant=task_variant,
                algo_name=algo_name,
                iterations=iterations,
                seed=seed,
                task_params=task_params,
                algo_params=algo_params,
                save=save,
                reports_dir=reports_dir,
            )
            results.append(result)

            if verbose:
                print(
                    f"    -> best_fitness={result.metrics.best_fitness:.6f}, "
                    f"runtime={result.metrics.runtime_s:.2f}s"
                )

    if verbose:
        print(f"\nCompleted {len(results)} runs. Results saved to {reports_dir}/")

    return results
