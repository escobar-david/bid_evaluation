# Create __init__.py file
"""
Bid Evaluation Library
Open-source toolkit for procurement bid evaluation
"""

__version__ = "0.1.0-alpha"

from .criteria import (
    # Base classes
    CriterionBase,

    # Criterion types
    LinearCriterion,
    ThresholdCriterion,
    DirectScoreCriterion,
    MinimumRatioCriterion,
    GeometricMeanCriterion,
    InverseProportionalCriterion,
    CustomCriterion,

    # Evaluator
    Evaluator,
)

from . import custom_templates

__all__ = [
    "CriterionBase",
    "LinearCriterion",
    "ThresholdCriterion",
    "DirectScoreCriterion",
    "MinimumRatioCriterion",
    "GeometricMeanCriterion",
    "InverseProportionalCriterion",
    "CustomCriterion",
    "Evaluator",
    "custom_templates",
]
