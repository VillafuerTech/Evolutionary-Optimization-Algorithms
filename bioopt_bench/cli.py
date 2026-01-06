"""Command-line interface for bioopt_bench."""

import argparse
import json
import sys
from typing import Any

from . import __version__
from .benchmark import run_one, run_suite
from .registry import list_available


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="bioopt_bench",
        description="Bio-inspired Optimization Benchmark Runner",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"bioopt_bench {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List available algorithms, tasks, and suites",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single benchmark",
    )
    run_parser.add_argument(
        "--task",
        required=True,
        choices=["functions", "tsp", "scheduling"],
        help="Task type",
    )
    run_parser.add_argument(
        "--variant",
        help="Task variant (e.g., beale, random, default)",
    )
    run_parser.add_argument(
        "--fn",
        help="Function name (alias for --variant for functions task)",
    )
    run_parser.add_argument(
        "--graph",
        help="Graph type (alias for --variant for tsp task)",
    )
    run_parser.add_argument(
        "--algo",
        required=True,
        choices=["ga", "pso", "aco", "gwo"],
        help="Algorithm to use",
    )
    run_parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of iterations (default: 100)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    run_parser.add_argument(
        "--save",
        action="store_true",
        help="Save results and artifacts",
    )
    run_parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Reports directory (default: reports/)",
    )
    # Task-specific options
    run_parser.add_argument(
        "--dim",
        type=int,
        help="Dimension for Rastrigin function",
    )
    run_parser.add_argument(
        "--n-cities",
        type=int,
        default=50,
        help="Number of cities for TSP (default: 50)",
    )
    run_parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Generate demo input for scheduling",
    )
    # Algorithm-specific options
    run_parser.add_argument(
        "--population-size",
        type=int,
        help="Population size for GA/GWO",
    )
    run_parser.add_argument(
        "--mutation-rate",
        type=float,
        help="Mutation rate for GA",
    )
    run_parser.add_argument(
        "--n-particles",
        type=int,
        help="Number of particles for PSO",
    )
    run_parser.add_argument(
        "--n-ants",
        type=int,
        help="Number of ants for ACO",
    )
    run_parser.add_argument(
        "--n-wolves",
        type=int,
        help="Number of wolves for GWO",
    )

    # Suite command
    suite_parser = subparsers.add_parser(
        "suite",
        help="Run a benchmark suite",
    )
    suite_parser.add_argument(
        "--suite",
        required=True,
        help="Suite name (e.g., default, quick, functions-only)",
    )
    suite_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repetitions per benchmark (default: 1)",
    )
    suite_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    suite_parser.add_argument(
        "--save",
        action="store_true",
        help="Save results and artifacts",
    )
    suite_parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Reports directory (default: reports/)",
    )
    suite_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "list":
        return cmd_list(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "suite":
        return cmd_suite(args)

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Handle list command."""
    available = list_available()

    if args.json:
        print(json.dumps(available, indent=2))
    else:
        print("Bio-inspired Optimization Benchmark Runner")
        print("=" * 50)
        print()
        print("ALGORITHMS:")
        for algo in available["algorithms"]:
            compat = available["compatibility"].get(algo, [])
            print(f"  - {algo}: supports {', '.join(compat)}")
        print()
        print("TASKS:")
        for task_type, variants in available["tasks"].items():
            print(f"  {task_type}:")
            for variant in variants:
                print(f"    - {variant}")
        print()
        print("SUITES:")
        for suite in available["suites"]:
            print(f"  - {suite}")
        print()

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Handle run command."""
    # Resolve variant
    variant = args.variant
    if args.task == "functions" and args.fn:
        variant = args.fn
    elif args.task == "tsp" and args.graph:
        variant = args.graph
    elif args.task == "scheduling":
        variant = "default"

    if variant is None:
        print(f"Error: --variant (or --fn/--graph) is required for task '{args.task}'")
        return 1

    # Build task params
    task_params: dict[str, Any] = {}
    if args.task == "functions" and variant == "rastrigin" and args.dim:
        task_params["n_dims"] = args.dim
    elif args.task == "tsp":
        task_params["n_cities"] = args.n_cities

    # Build algo params
    algo_params: dict[str, Any] = {}
    if args.population_size:
        algo_params["population_size"] = args.population_size
    if args.mutation_rate:
        algo_params["mutation_rate"] = args.mutation_rate
    if args.n_particles:
        algo_params["n_particles"] = args.n_particles
    if args.n_ants:
        algo_params["n_ants"] = args.n_ants
    if args.n_wolves:
        algo_params["n_wolves"] = args.n_wolves

    print(f"Running {args.algo.upper()} on {variant}...")
    print(f"  Task: {args.task}")
    print(f"  Iterations: {args.iters}")
    print(f"  Seed: {args.seed}")
    if task_params:
        print(f"  Task params: {task_params}")
    if algo_params:
        print(f"  Algo params: {algo_params}")
    print()

    result = run_one(
        task_type=args.task,
        task_variant=variant,
        algo_name=args.algo,
        iterations=args.iters,
        seed=args.seed,
        task_params=task_params,
        algo_params=algo_params,
        save=args.save,
        reports_dir=args.reports_dir,
    )

    print("Results:")
    print(f"  Run ID: {result.run_id}")
    print(f"  Best Fitness: {result.metrics.best_fitness:.6f}")
    print(f"  Final Fitness: {result.metrics.final_fitness:.6f}")
    print(f"  Runtime: {result.metrics.runtime_s:.3f}s")

    if result.metrics.extra:
        print(f"  Extra: {json.dumps(result.metrics.extra, indent=4)}")

    if args.save:
        print(f"\nArtifacts saved to: {args.reports_dir}/runs/{result.run_id}/")

    return 0


def cmd_suite(args: argparse.Namespace) -> int:
    """Handle suite command."""
    print(f"Running suite: {args.suite}")
    print(f"  Repeat: {args.repeat}")
    print(f"  Base seed: {args.seed}")
    print()

    results = run_suite(
        suite_name=args.suite,
        repeat=args.repeat,
        base_seed=args.seed,
        save=args.save,
        reports_dir=args.reports_dir,
        verbose=not args.quiet,
    )

    print()
    print("=" * 50)
    print(f"Suite completed: {len(results)} runs")

    if args.save:
        print(f"Results saved to: {args.reports_dir}/results.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
