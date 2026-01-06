"""Particle Swarm Optimization implementation."""

import random
from typing import Any

import numpy as np


class ParticleSwarmOptimization:
    """Particle Swarm Optimization for continuous function optimization."""

    def __init__(
        self,
        n_particles: int = 50,
        alpha1: float = 2.0,
        alpha2: float = 2.0,
        phi1: float = 0.5,
        phi2: float = 0.5,
    ):
        """Initialize PSO parameters.

        Args:
            n_particles: Number of particles in the swarm.
            alpha1: Cognitive acceleration coefficient.
            alpha2: Social acceleration coefficient.
            phi1: Personal best weight factor.
            phi2: Global best weight factor.
        """
        self.n_particles = n_particles
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.phi1 = phi1
        self.phi2 = phi2

    def run(
        self,
        task: Any,
        seed: int,
        iterations: int = 100,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, list], dict[str, Any]]:
        """Run PSO on a task.

        Args:
            task: Task object with evaluate() method and bounds.
            seed: Random seed for reproducibility.
            iterations: Number of iterations (epochs).
            **kwargs: Additional algorithm parameters.

        Returns:
            Tuple of (best_solution, history, metadata).
        """
        random.seed(seed)
        np.random.seed(seed)

        # Override defaults with kwargs
        n_particles = kwargs.get("n_particles", self.n_particles)
        alpha1 = kwargs.get("alpha1", self.alpha1)
        alpha2 = kwargs.get("alpha2", self.alpha2)
        phi1 = kwargs.get("phi1", self.phi1)
        phi2 = kwargs.get("phi2", self.phi2)

        bounds = np.array(task.bounds)
        n_dims = len(bounds)

        # Initialize particles
        particles_x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, n_dims))
        particles_v = np.random.uniform(
            0.1 * bounds[:, 0], 0.1 * bounds[:, 1], size=(n_particles, n_dims)
        )
        particles_best_x = particles_x.copy()

        # Find initial global best
        fitness_values = np.array([task.evaluate(x) for x in particles_x])
        gbest_idx = np.argmin(fitness_values)
        gbest = particles_best_x[gbest_idx].copy()
        gbest_fitness = fitness_values[gbest_idx]

        history: dict[str, list] = {"iter": [], "best_fitness": [], "mean_fitness": []}

        for epoch in range(iterations):
            for i in range(n_particles):
                r1 = np.random.rand(n_dims)
                r2 = np.random.rand(n_dims)

                # Update velocity
                new_v = (
                    particles_v[i]
                    + alpha1 * phi1 * r1 * (particles_best_x[i] - particles_x[i])
                    + alpha2 * phi2 * r2 * (gbest - particles_x[i])
                )

                # Update position
                new_x = particles_x[i] + new_v
                new_x = np.clip(new_x, bounds[:, 0], bounds[:, 1])

                new_fitness = task.evaluate(new_x)

                # Update personal best
                if new_fitness < task.evaluate(particles_best_x[i]):
                    particles_best_x[i] = new_x.copy()

                # Update global best
                if new_fitness < gbest_fitness:
                    gbest = new_x.copy()
                    gbest_fitness = new_fitness

                particles_v[i] = new_v
                particles_x[i] = new_x

            # Compute mean fitness for this epoch
            current_fitness = np.array([task.evaluate(x) for x in particles_x])
            mean_fit = np.mean(current_fitness)

            history["iter"].append(epoch)
            history["best_fitness"].append(gbest_fitness)
            history["mean_fitness"].append(mean_fit)

        meta = {
            "n_particles": n_particles,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "final_gbest_fitness": float(gbest_fitness),
        }

        return gbest.tolist(), history, meta
