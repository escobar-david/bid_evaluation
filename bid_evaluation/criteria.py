# criteria.py
"""Criterion classes for bid evaluation scoring."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Callable


class CriterionBase(ABC):
    """Base class for all evaluation criteria"""

    def __init__(self, name: str, weight: float, **kwargs):
        self.name = name
        self.weight = weight
        self.config = kwargs
        self._statistics = {}

    def calculate_statistics(self, values: pd.Series) -> Dict[str, Any]:
        """Automatically calculates necessary statistics"""
        return {
            'min': values.min(),
            'max': values.max(),
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'q25': values.quantile(0.25),
            'q75': values.quantile(0.75)
        }

    @abstractmethod
    def evaluate(self, values: pd.Series) -> pd.Series:
        """Abstract method to evaluate the criterion"""
        pass

    def normalize(self, scores: pd.Series, scale: float = 100.0) -> pd.Series:
        """Normalizes scores to a scale (default 0-100)"""
        if scores.max() == scores.min():
            return pd.Series(scale, index=scores.index)
        return ((scores - scores.min()) / (scores.max() - scores.min())) * scale


class LinearCriterion(CriterionBase):
    """Simple linear normalization"""

    def evaluate(self, values: pd.Series) -> pd.Series:
        self._statistics = self.calculate_statistics(values)

        # Determine if higher is better or lower is better
        higher_is_better = self.config.get('higher_is_better', True)

        if higher_is_better:
            return self.normalize(values) * self.weight
        else:
            # Invert: lower value = higher score
            return self.normalize(-values) * self.weight


class ThresholdCriterion(CriterionBase):
    """Evaluation by thresholds (score ranges)"""

    def evaluate(self, values: pd.Series) -> pd.Series:
        self._statistics = self.calculate_statistics(values)

        thresholds = self.config.get('thresholds', [])
        # thresholds: [(lower_limit, upper_limit, score), ...]

        scores = pd.Series(0.0, index=values.index)

        for lower_limit, upper_limit, score in thresholds:
            mask = (values >= lower_limit) & (values < upper_limit)
            scores[mask] = score

        return scores * self.weight


class DirectScoreCriterion(CriterionBase):
    """Score is already evaluated (e.g., evaluation committee)"""

    def evaluate(self, values: pd.Series) -> pd.Series:
        self._statistics = self.calculate_statistics(values)

        input_scale = self.config.get('input_scale', 100)
        output_scale = self.config.get('output_scale', 100)

        # Adjust scale if necessary
        if input_scale != output_scale:
            values = values * (output_scale / input_scale)

        return values * self.weight


class MinimumRatioCriterion(CriterionBase):
    """Score = (minimum_value / value) * 100"""

    def evaluate(self, values: pd.Series) -> pd.Series:
        self._statistics = self.calculate_statistics(values)

        min_value = values.min()
        scores = (min_value / values) * 100

        return scores * self.weight


class CustomCriterion(CriterionBase):
    """Allows custom evaluation function"""

    def __init__(self, name: str, weight: float,
                 evaluation_function: Callable[[pd.Series, Dict], pd.Series],
                 **kwargs):
        super().__init__(name, weight, **kwargs)
        self.evaluation_function = evaluation_function

    def evaluate(self, values: pd.Series) -> pd.Series:
        self._statistics = self.calculate_statistics(values)

        # Custom function receives values and statistics
        scores = self.evaluation_function(values, self._statistics)

        return scores * self.weight
