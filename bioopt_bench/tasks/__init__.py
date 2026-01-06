"""Benchmark tasks for optimization algorithms."""

from .functions import BealeFunction, EasomFunction, RastriginFunction
from .scheduling import SchedulingTask
from .tsp import TSPTask

__all__ = [
    "BealeFunction",
    "EasomFunction",
    "RastriginFunction",
    "TSPTask",
    "SchedulingTask",
]
