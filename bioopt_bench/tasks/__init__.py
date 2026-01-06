"""Benchmark tasks for optimization algorithms."""

from .functions import BealeFunction, EasomFunction, RastriginFunction
from .tsp import TSPTask
from .scheduling import SchedulingTask

__all__ = [
    "BealeFunction",
    "EasomFunction",
    "RastriginFunction",
    "TSPTask",
    "SchedulingTask",
]
