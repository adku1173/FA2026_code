"""Optimization algorithms module.

Includes ISTA, FISTA, SFISTA, and callback infrastructure for sparse recovery algorithms.
"""

from .ista import FISTA, ISTA, SFISTA
from .callbacks import (
    IterationCallback,
    SolutionChangeStopper,
    StepParamCollector,
    IntermediateOutputCollector,
)

__all__ = [
    "ISTA",
    "FISTA",
    "SFISTA",
    "IterationCallback",
    "SolutionChangeStopper",
    "StepParamCollector",
    "IntermediateOutputCollector",
]
