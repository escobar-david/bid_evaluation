# Create __init__.py file
"""
Bid Evaluation Library
Open-source toolkit for procurement bid evaluation
"""

__version__ = "0.1.0"

from .criteria import (
    # Base classes
    CriterionBase,

    # Criterion types
    LinearCriterion,
    ThresholdCriterion,
    DirectScoreCriterion,
    MinimumRatioCriterion,
    FormulaCriterion,
    CustomCriterion,
)

from .evaluator import Evaluator

from .staged import StagedEvaluator

from . import custom_templates

__all__ = [
    "CriterionBase",
    "LinearCriterion",
    "ThresholdCriterion",
    "DirectScoreCriterion",
    "MinimumRatioCriterion",
    "FormulaCriterion",
    "CustomCriterion",
    "Evaluator",
    "StagedEvaluator",
    "custom_templates",
]
