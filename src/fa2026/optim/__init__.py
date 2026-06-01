"""Optimization algorithms module.

Includes ISTA, FISTA, and callback infrastructure for sparse recovery algorithms.
"""

from .ista import FISTA, ISTA
from .callbacks import (
    IterationCallback,
    SolutionChangeStopper,
    StepParamCollector,
    IntermediateOutputCollector,
)

__all__ = [
    "ISTA",
    "FISTA",
    "IterationCallback",
    "SolutionChangeStopper",
    "StepParamCollector",
    "IntermediateOutputCollector",
]
