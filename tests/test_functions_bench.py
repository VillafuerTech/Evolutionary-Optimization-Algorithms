"""Tests for benchmark functions."""

import numpy as np

from bioopt_bench.algorithms import GeneticAlgorithm, ParticleSwarmOptimization
from bioopt_bench.tasks.functions import BealeFunction, EasomFunction, RastriginFunction


class TestBenchmarkFunctions:
    """Tests for benchmark functions."""

    def test_beale_at_optimum(self):
        """Test Beale function at global optimum."""
        beale = BealeFunction()
        result = beale.evaluate([3.0, 0.5])
        assert result < 1e-10

    def test_beale_away_from_optimum(self):
        """Test Beale function away from optimum."""
        beale = BealeFunction()
        result = beale.evaluate([0.0, 0.0])
        assert result > 0

    def test_easom_at_optimum(self):
        """Test Easom function at global optimum."""
        easom = EasomFunction()
        result = easom.evaluate([np.pi, np.pi])
        assert abs(result - (-1.0)) < 1e-10

    def test_easom_away_from_optimum(self):
        """Test Easom function away from optimum."""
        easom = EasomFunction()
        result = easom.evaluate([0.0, 0.0])
        assert result > -1.0

    def test_rastrigin_at_optimum(self):
        """Test Rastrigin function at global optimum."""
        rastrigin = RastriginFunction(n_dims=5)
        result = rastrigin.evaluate([0.0, 0.0, 0.0, 0.0, 0.0])
        assert result < 1e-10

    def test_rastrigin_away_from_optimum(self):
        """Test Rastrigin function away from optimum."""
        rastrigin = RastriginFunction(n_dims=5)
        result = rastrigin.evaluate([1.0, 1.0, 1.0, 1.0, 1.0])
        assert result > 0


class TestAlgorithmsOnFunctions:
    """Test algorithms on benchmark functions."""

    def test_pso_on_beale(self):
        """Test PSO can optimize Beale function."""
        beale = BealeFunction()
        pso = ParticleSwarmOptimization(n_particles=20)
        solution, history, meta = pso.run(beale, seed=42, iterations=50)

        assert len(solution) == 2
        assert "best_fitness" in history
        assert len(history["best_fitness"]) == 50
        # Should improve over iterations
        assert history["best_fitness"][-1] <= history["best_fitness"][0]

    def test_ga_on_beale(self):
        """Test GA can optimize Beale function."""
        beale = BealeFunction()
        ga = GeneticAlgorithm(population_size=50, num_bits=12)
        solution, history, meta = ga.run(beale, seed=42, iterations=50)

        assert len(solution) == 2
        assert "best_fitness" in history
        # Should improve over iterations
        assert history["best_fitness"][-1] <= history["best_fitness"][0]

    def test_pso_on_rastrigin(self):
        """Test PSO on Rastrigin function."""
        rastrigin = RastriginFunction(n_dims=3)
        pso = ParticleSwarmOptimization(n_particles=30)
        solution, history, meta = pso.run(rastrigin, seed=42, iterations=50)

        assert len(solution) == 3
        assert "best_fitness" in history
