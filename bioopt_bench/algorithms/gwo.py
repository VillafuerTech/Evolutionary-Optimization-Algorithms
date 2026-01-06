"""Grey Wolf Optimizer implementation."""

import copy
import random
from typing import Any

import numpy as np


class GreyWolfOptimizer:
    """Grey Wolf Optimizer for scheduling and discrete optimization."""

    def __init__(
        self,
        n_wolves: int = 100,
    ):
        """Initialize GWO parameters.

        Args:
            n_wolves: Number of wolves (population size).
        """
        self.n_wolves = n_wolves

    def run(
        self,
        task: Any,
        seed: int,
        iterations: int = 500,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, list], dict[str, Any]]:
        """Run GWO on a task.

        Args:
            task: Task object with evaluate() method and task-specific properties.
            seed: Random seed for reproducibility.
            iterations: Maximum number of iterations.
            **kwargs: Additional algorithm parameters.

        Returns:
            Tuple of (best_solution, history, metadata).
        """
        random.seed(seed)
        np.random.seed(seed)

        n_wolves = kwargs.get("n_wolves", self.n_wolves)

        task_type = getattr(task, "task_type", "scheduling")

        if task_type == "scheduling":
            return self._run_scheduling(task, iterations, n_wolves)
        else:
            return self._run_continuous(task, iterations, n_wolves)

    def _run_scheduling(
        self,
        task: Any,
        max_iter: int,
        n_wolves: int,
    ) -> tuple[Any, dict[str, list], dict[str, Any]]:
        """Run GWO for class scheduling optimization."""
        courses = task.courses
        n_time_slots = task.n_time_slots
        n_classrooms = task.n_classrooms

        # Initialize population
        population = []
        for _ in range(n_wolves):
            solution = []
            for course in courses:
                assignment = {
                    "course": course["name"],
                    "professor": course["professor"],
                    "time_slot": random.randint(0, n_time_slots - 1),
                    "classroom": random.randint(0, n_classrooms - 1),
                }
                solution.append(assignment)
            population.append(solution)

        # Sort and get alpha, beta, delta
        population.sort(key=lambda sol: task.evaluate(sol))
        alpha = copy.deepcopy(population[0])
        beta = copy.deepcopy(population[1])
        delta = copy.deepcopy(population[2])

        history: dict[str, list] = {"iter": [], "best_fitness": [], "mean_fitness": []}

        for iteration in range(max_iter):
            a = 2 - iteration * (2 / max_iter)

            # Update positions
            new_population = []
            for solution in population:
                new_solution = []
                for i in range(len(solution)):
                    # Time slot update
                    choices_ts = [
                        alpha[i]["time_slot"],
                        beta[i]["time_slot"],
                        delta[i]["time_slot"],
                    ]
                    probs = [0.4, 0.3, 0.3]
                    if random.random() < a / 2:
                        new_ts = random.randint(0, n_time_slots - 1)
                    else:
                        new_ts = random.choices(choices_ts, probs)[0]

                    # Classroom update
                    choices_cr = [
                        alpha[i]["classroom"],
                        beta[i]["classroom"],
                        delta[i]["classroom"],
                    ]
                    if random.random() < a / 2:
                        new_cr = random.randint(0, n_classrooms - 1)
                    else:
                        new_cr = random.choices(choices_cr, probs)[0]

                    new_assignment = {
                        "course": solution[i]["course"],
                        "professor": solution[i]["professor"],
                        "time_slot": new_ts,
                        "classroom": new_cr,
                    }
                    new_solution.append(new_assignment)
                new_population.append(new_solution)

            population = new_population

            # Sort and update alpha, beta, delta
            population.sort(key=lambda sol: task.evaluate(sol))
            current_alpha_fitness = task.evaluate(population[0])

            if current_alpha_fitness < task.evaluate(alpha):
                alpha = copy.deepcopy(population[0])
            if task.evaluate(population[1]) < task.evaluate(beta):
                beta = copy.deepcopy(population[1])
            if task.evaluate(population[2]) < task.evaluate(delta):
                delta = copy.deepcopy(population[2])

            # Record history
            fitness_values = [task.evaluate(sol) for sol in population]
            mean_fit = np.mean(fitness_values)

            history["iter"].append(iteration)
            history["best_fitness"].append(task.evaluate(alpha))
            history["mean_fitness"].append(mean_fit)

            # Early termination if optimal found
            if task.evaluate(alpha) == 0:
                break

        # Get conflict information
        conflicts = task.check_conflicts(alpha) if hasattr(task, "check_conflicts") else []

        meta = {
            "n_wolves": n_wolves,
            "converged_at": iteration if task.evaluate(alpha) == 0 else None,
            "conflicts_count": len(conflicts),
            "conflicts": conflicts,
        }

        return alpha, history, meta

    def _run_continuous(
        self,
        task: Any,
        max_iter: int,
        n_wolves: int,
    ) -> tuple[Any, dict[str, list], dict[str, Any]]:
        """Run GWO for continuous optimization."""
        bounds = np.array(task.bounds)
        n_dims = len(bounds)

        # Initialize population
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=(n_wolves, n_dims)
        )

        # Evaluate and sort
        fitness = np.array([task.evaluate(x) for x in population])
        sorted_idx = np.argsort(fitness)

        alpha = population[sorted_idx[0]].copy()
        beta = population[sorted_idx[1]].copy()
        delta = population[sorted_idx[2]].copy()
        alpha_fitness = fitness[sorted_idx[0]]

        history: dict[str, list] = {"iter": [], "best_fitness": [], "mean_fitness": []}

        for iteration in range(max_iter):
            a = 2 - iteration * (2 / max_iter)

            for i in range(n_wolves):
                for d in range(n_dims):
                    # Alpha influence
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[d] - population[i, d])
                    X1 = alpha[d] - A1 * D_alpha

                    # Beta influence
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[d] - population[i, d])
                    X2 = beta[d] - A2 * D_beta

                    # Delta influence
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[d] - population[i, d])
                    X3 = delta[d] - A3 * D_delta

                    population[i, d] = (X1 + X2 + X3) / 3

                # Clip to bounds
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

            # Re-evaluate and update hierarchy
            fitness = np.array([task.evaluate(x) for x in population])
            sorted_idx = np.argsort(fitness)

            if fitness[sorted_idx[0]] < alpha_fitness:
                alpha = population[sorted_idx[0]].copy()
                alpha_fitness = fitness[sorted_idx[0]]
            if fitness[sorted_idx[1]] < task.evaluate(beta):
                beta = population[sorted_idx[1]].copy()
            if fitness[sorted_idx[2]] < task.evaluate(delta):
                delta = population[sorted_idx[2]].copy()

            history["iter"].append(iteration)
            history["best_fitness"].append(alpha_fitness)
            history["mean_fitness"].append(np.mean(fitness))

        meta = {"n_wolves": n_wolves}
        return alpha.tolist(), history, meta
