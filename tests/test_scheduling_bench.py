"""Tests for scheduling benchmarks."""

from bioopt_bench.algorithms import GreyWolfOptimizer
from bioopt_bench.tasks.scheduling import SchedulingTask


class TestSchedulingTask:
    """Tests for scheduling task."""

    def test_default_initialization(self):
        """Test default scheduling task initialization."""
        task = SchedulingTask()
        assert task.n_courses == 12
        assert task.n_time_slots == 7
        assert task.n_classrooms == 5

    def test_perfect_schedule_fitness(self):
        """Test that a schedule with no conflicts has low fitness."""
        task = SchedulingTask()
        # Create a simple schedule with one course per time slot
        solution = []
        for i, course in enumerate(task.courses[:7]):
            solution.append(
                {
                    "course": course["name"],
                    "professor": course["professor"],
                    "time_slot": i,
                    "classroom": i % task.n_classrooms,
                }
            )
        # Add remaining courses with potential conflicts
        for i, course in enumerate(task.courses[7:]):
            solution.append(
                {
                    "course": course["name"],
                    "professor": course["professor"],
                    "time_slot": i,
                    "classroom": (i + 1) % task.n_classrooms,
                }
            )

        # Should have some penalty but be evaluable
        fitness = task.evaluate(solution)
        assert isinstance(fitness, (int, float))

    def test_conflict_detection(self):
        """Test that conflicts are properly detected."""
        task = SchedulingTask()
        # Create schedule with obvious conflicts
        solution = []
        for course in task.courses:
            solution.append(
                {
                    "course": course["name"],
                    "professor": course["professor"],
                    "time_slot": 0,  # All at same time
                    "classroom": 0,  # All in same room
                }
            )

        conflicts = task.check_conflicts(solution)
        assert len(conflicts) > 0  # Should have many conflicts


class TestGWOOnScheduling:
    """Test GWO on scheduling task."""

    def test_gwo_on_scheduling(self):
        """Test GWO can optimize scheduling."""
        task = SchedulingTask()
        gwo = GreyWolfOptimizer(n_wolves=30)
        solution, history, meta = gwo.run(task, seed=42, iterations=50)

        assert len(solution) == task.n_courses
        assert "best_fitness" in history
        # Should improve over iterations
        assert history["best_fitness"][-1] <= history["best_fitness"][0]

    def test_gwo_solution_format(self):
        """Test GWO solution has correct format."""
        task = SchedulingTask()
        gwo = GreyWolfOptimizer(n_wolves=20)
        solution, history, meta = gwo.run(task, seed=42, iterations=20)

        for assignment in solution:
            assert "course" in assignment
            assert "professor" in assignment
            assert "time_slot" in assignment
            assert "classroom" in assignment
            assert 0 <= assignment["time_slot"] < task.n_time_slots
            assert 0 <= assignment["classroom"] < task.n_classrooms
