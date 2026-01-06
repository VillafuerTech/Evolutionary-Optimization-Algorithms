"""Tests for TSP benchmarks."""

from bioopt_bench.tasks.tsp import TSPTask
from bioopt_bench.algorithms import GeneticAlgorithm, AntColonyOptimization


class TestTSPTask:
    """Tests for TSP task."""

    def test_random_graph_generation(self):
        """Test random graph generation."""
        tsp = TSPTask(n_cities=10, graph_type="random", seed=42)
        assert tsp.n_cities == 10
        assert len(tsp.nodes) == 10
        assert len(tsp.distance_matrix) == 10
        assert len(tsp.distance_matrix[0]) == 10

    def test_grid_graph_generation(self):
        """Test grid graph generation."""
        tsp = TSPTask(n_cities=16, graph_type="grid")  # 4x4 grid
        assert len(tsp.nodes) == 16

    def test_tour_evaluation(self):
        """Test tour evaluation."""
        nodes = [(0, 0), (1, 0), (1, 1), (0, 1)]  # Unit square
        tsp = TSPTask(nodes=nodes)

        # Tour around the square should have distance 4
        tour = [0, 1, 2, 3]
        distance = tsp.evaluate(tour)
        assert abs(distance - 4.0) < 1e-10

    def test_distance_matrix_symmetry(self):
        """Test that distance matrix is symmetric."""
        tsp = TSPTask(n_cities=5, graph_type="random", seed=42)
        for i in range(5):
            for j in range(5):
                assert abs(tsp.distance_matrix[i][j] - tsp.distance_matrix[j][i]) < 1e-10


class TestAlgorithmsOnTSP:
    """Test algorithms on TSP."""

    def test_ga_on_small_tsp(self):
        """Test GA on small TSP."""
        tsp = TSPTask(n_cities=10, graph_type="random", seed=42)
        ga = GeneticAlgorithm(population_size=30, mutation_rate=0.05)
        solution, history, meta = ga.run(tsp, seed=42, iterations=30)

        assert len(solution) == 10
        assert set(solution) == set(range(10))  # Valid permutation
        assert "best_fitness" in history
        # Should improve over iterations
        assert history["best_fitness"][-1] <= history["best_fitness"][0]

    def test_aco_on_small_tsp(self):
        """Test ACO on small TSP."""
        tsp = TSPTask(n_cities=10, graph_type="random", seed=42)
        aco = AntColonyOptimization(n_ants=10)
        solution, history, meta = aco.run(tsp, seed=42, iterations=20)

        assert len(solution) == 10
        assert set(solution) == set(range(10))  # Valid permutation
        assert "best_fitness" in history
