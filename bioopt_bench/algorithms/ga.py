"""Genetic Algorithm implementation."""

import random
from typing import Any, Callable

import numpy as np


class GeneticAlgorithm:
    """Genetic Algorithm for continuous and combinatorial optimization.

    Supports both continuous function optimization (binary encoding) and
    TSP (permutation encoding).
    """

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 10,
        num_bits: int = 10,
    ):
        """Initialize GA parameters.

        Args:
            population_size: Number of individuals in population.
            mutation_rate: Probability of mutation per gene.
            crossover_rate: Probability of crossover.
            elite_size: Number of elite individuals preserved.
            num_bits: Bits per variable for continuous optimization.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.num_bits = num_bits

    def run(
        self,
        task: Any,
        seed: int,
        iterations: int = 100,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, list], dict[str, Any]]:
        """Run the genetic algorithm on a task.

        Args:
            task: Task object with evaluate() method and task properties.
            seed: Random seed for reproducibility.
            iterations: Number of generations.
            **kwargs: Additional algorithm parameters.

        Returns:
            Tuple of (best_solution, history, metadata).
        """
        random.seed(seed)
        np.random.seed(seed)

        # Override defaults with kwargs
        pop_size = kwargs.get("population_size", self.population_size)
        mut_rate = kwargs.get("mutation_rate", self.mutation_rate)
        cross_rate = kwargs.get("crossover_rate", self.crossover_rate)
        elite = kwargs.get("elite_size", self.elite_size)

        task_type = getattr(task, "task_type", "continuous")

        if task_type == "tsp":
            return self._run_tsp(task, iterations, pop_size, mut_rate, cross_rate, elite)
        else:
            return self._run_continuous(task, iterations, pop_size, mut_rate, elite, self.num_bits)

    def _run_continuous(
        self,
        task: Any,
        generations: int,
        pop_size: int,
        mutation: float,
        elite: int,
        num_bits: int,
    ) -> tuple[Any, dict[str, list], dict[str, Any]]:
        """Run GA for continuous optimization with binary encoding."""
        bounds = task.bounds
        num_vars = len(bounds)
        total_bits = num_bits * num_vars
        optimization_type = getattr(task, "optimization_type", "min")

        # Initialize population
        population = []
        for _ in range(pop_size):
            individual = ""
            for j in range(num_vars):
                value = random.uniform(bounds[j][0], bounds[j][1])
                individual += self._real_to_binary(value, bounds[j], num_bits)
            population.append(individual)

        history: dict[str, list] = {"iter": [], "best_fitness": [], "mean_fitness": []}
        best_individual = None
        best_fitness = None

        for gen in range(generations):
            # Evaluate population
            evaluated = []
            for ind in population:
                values = []
                for i in range(num_vars):
                    start = i * num_bits
                    end = start + num_bits
                    binary_str = ind[start:end]
                    value = self._binary_to_real(binary_str, bounds[i], num_bits)
                    values.append(value)
                fitness = task.evaluate(values)
                evaluated.append((ind, fitness))

            # Track statistics
            fitness_values = [e[1] for e in evaluated]
            mean_fit = np.mean(fitness_values)

            # Selection
            if optimization_type == "min":
                evaluated = sorted(evaluated, key=lambda x: x[1])
            else:
                evaluated = sorted(evaluated, key=lambda x: x[1], reverse=True)

            parents = evaluated[:elite]

            # Update best
            if optimization_type == "min":
                curr_best_ind, curr_best_fit = min(evaluated, key=lambda x: x[1])
                if best_fitness is None or curr_best_fit < best_fitness:
                    best_individual = curr_best_ind
                    best_fitness = curr_best_fit
            else:
                curr_best_ind, curr_best_fit = max(evaluated, key=lambda x: x[1])
                if best_fitness is None or curr_best_fit > best_fitness:
                    best_individual = curr_best_ind
                    best_fitness = curr_best_fit

            history["iter"].append(gen)
            history["best_fitness"].append(best_fitness)
            history["mean_fitness"].append(mean_fit)

            # Crossover
            parent_individuals = [p[0] for p in parents]
            children = []
            for _ in range(pop_size - elite):
                parent1, parent2 = random.sample(parent_individuals, 2)
                crossover_point = random.randint(1, total_bits - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
                children.append(child)

            # Mutation
            for i in range(len(children)):
                child = list(children[i])
                for j in range(len(child)):
                    if random.random() < mutation:
                        child[j] = "1" if child[j] == "0" else "0"
                children[i] = "".join(child)

            population = [p[0] for p in parents] + children

        # Decode best solution
        best_values = []
        for i in range(num_vars):
            start = i * num_bits
            end = start + num_bits
            binary_str = best_individual[start:end]
            value = self._binary_to_real(binary_str, bounds[i], num_bits)
            best_values.append(value)

        meta = {
            "encoding": "binary",
            "num_bits": num_bits,
            "final_population_size": len(population),
        }

        return best_values, history, meta

    def _run_tsp(
        self,
        task: Any,
        generations: int,
        pop_size: int,
        mutation: float,
        crossover_rate: float,
        elite: int,
    ) -> tuple[Any, dict[str, list], dict[str, Any]]:
        """Run GA for TSP with permutation encoding."""
        n_cities = task.n_cities
        distance_matrix = task.distance_matrix

        # Initialize population
        base = list(range(n_cities))
        population = []
        for _ in range(pop_size):
            ind = base[:]
            random.shuffle(ind)
            population.append(ind)

        history: dict[str, list] = {"iter": [], "best_fitness": [], "mean_fitness": []}
        best_individual = None
        best_fitness = float("inf")

        for gen in range(generations):
            # Evaluate population
            fitness_list = []
            for ind in population:
                dist = self._calculate_tour_distance(ind, distance_matrix)
                fitness_list.append(dist)

            mean_fit = np.mean(fitness_list)

            # Selection (minimize distance)
            fitness_with_idx = list(enumerate(fitness_list))
            fitness_with_idx.sort(key=lambda x: x[1])
            elite_indices = [idx for idx, _ in fitness_with_idx[:elite]]
            elite_individuals = [population[idx] for idx in elite_indices]

            # Update best
            curr_best_idx = fitness_with_idx[0][0]
            curr_best_fit = fitness_with_idx[0][1]
            if curr_best_fit < best_fitness:
                best_fitness = curr_best_fit
                best_individual = population[curr_best_idx][:]

            history["iter"].append(gen)
            history["best_fitness"].append(best_fitness)
            history["mean_fitness"].append(mean_fit)

            # Create new population
            new_population = [ind[:] for ind in elite_individuals]
            while len(new_population) < pop_size:
                if random.random() < crossover_rate:
                    parent1 = random.choice(elite_individuals)
                    parent2 = random.choice(elite_individuals)
                    child = self._ordered_crossover(parent1, parent2)
                else:
                    child = random.choice(population)[:]

                child = self._swap_mutate(child, mutation)
                new_population.append(child)

            population = new_population

        meta = {"encoding": "permutation", "n_cities": n_cities}
        return best_individual, history, meta

    def _ordered_crossover(self, parent1: list, parent2: list) -> list:
        """Perform ordered crossover for TSP."""
        size = len(parent1)
        child = [-1] * size
        start = random.randint(0, size - 1)
        end = random.randint(start, size - 1)
        child[start : end + 1] = parent1[start : end + 1]

        p2_idx = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2[p2_idx]
                p2_idx += 1
        return child

    def _swap_mutate(self, individual: list, rate: float) -> list:
        """Apply swap mutation to TSP individual."""
        for i in range(len(individual)):
            if random.random() < rate:
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def _calculate_tour_distance(self, tour: list, distance_matrix: list[list[float]]) -> float:
        """Calculate total tour distance."""
        total = 0.0
        for i in range(len(tour) - 1):
            total += distance_matrix[tour[i]][tour[i + 1]]
        total += distance_matrix[tour[-1]][tour[0]]
        return total

    def _binary_to_real(
        self, binary: str, value_range: tuple[float, float], num_bits: int
    ) -> float:
        """Convert binary string to real value."""
        max_int = 2**num_bits - 1
        integer_value = int(binary, 2)
        real_value = value_range[0] + (integer_value / max_int) * (value_range[1] - value_range[0])
        return real_value

    def _real_to_binary(self, value: float, value_range: tuple[float, float], num_bits: int) -> str:
        """Convert real value to binary string."""
        max_int = 2**num_bits - 1
        normalized = int((value - value_range[0]) / (value_range[1] - value_range[0]) * max_int)
        normalized = max(0, min(max_int, normalized))
        return format(normalized, f"0{num_bits}b")
