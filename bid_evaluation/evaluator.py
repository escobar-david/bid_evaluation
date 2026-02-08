# evaluator.py
"""Evaluation engine with fluent interface and config support."""

import pandas as pd
import numpy as np
from typing import Any, Dict, Callable
import yaml
import json

from .criteria import (
    CriterionBase,
    LinearCriterion,
    ThresholdCriterion,
    DirectScoreCriterion,
    MinimumRatioCriterion,
    FormulaCriterion,
    CustomCriterion,
)


class Evaluator:
    """Simplified evaluation engine with fluent interface and config support"""

    def __init__(self, normalize_weights: bool = True):
        """
        Args:
            normalize_weights: If True, automatically normalizes weights to sum to 1.0
        """
        self.criteria: Dict[str, CriterionBase] = {}
        self.normalize_weights = normalize_weights

    # === Factory methods (from config) ===

    @classmethod
    def from_config(cls, config: Dict[str, Dict[str, Any]],
                    normalize_weights: bool = True) -> 'Evaluator':
        """
        Create evaluator from configuration dictionary

        Args:
            config: Dictionary with criterion configurations
            normalize_weights: If True, normalizes weights to sum to 1.0

        Example:
            config = {
                'experience': {'type': 'linear', 'weight': 0.3, 'higher_is_better': True},
                'bid_amount': {'type': 'min_ratio', 'weight': 0.7}
            }
            evaluator = Evaluator.from_config(config)
        """
        evaluator = cls(normalize_weights=normalize_weights)

        for column, params in config.items():
            params = params.copy()  # Don't modify original
            criterion_type = params.pop('type')
            weight = params.pop('weight')
            name = params.pop('name', column)

            if criterion_type == 'linear':
                criterion = LinearCriterion(name, weight, **params)
            elif criterion_type == 'threshold':
                criterion = ThresholdCriterion(name, weight, **params)
            elif criterion_type == 'direct':
                criterion = DirectScoreCriterion(name, weight, **params)
            elif criterion_type == 'min_ratio':
                criterion = MinimumRatioCriterion(name, weight, **params)
            elif criterion_type == 'formula':
                formula_str = params.pop('formula', 'value')
                variables = params.pop('variables', {})
                criterion = FormulaCriterion(name, weight, formula=formula_str, variables=variables, **params)
            else:
                raise ValueError(f"Unknown criterion type: {criterion_type}")

            evaluator.add_criterion(column, criterion)

        return evaluator

    @classmethod
    def from_yaml(cls, filepath: str, normalize_weights: bool = True) -> 'Evaluator':
        """
        Create evaluator from YAML file

        Args:
            filepath: Path to YAML configuration file
            normalize_weights: If True, normalizes weights to sum to 1.0

        Example YAML:
            criteria:
              experience:
                type: linear
                weight: 0.3
                higher_is_better: true
              bid_amount:
                type: min_ratio
                weight: 0.7
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_config(data.get('criteria', {}), normalize_weights)

    @classmethod
    def from_json(cls, filepath: str, normalize_weights: bool = True) -> 'Evaluator':
        """
        Create evaluator from JSON file

        Args:
            filepath: Path to JSON configuration file
            normalize_weights: If True, normalizes weights to sum to 1.0
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_config(data.get('criteria', {}), normalize_weights)

    # === Fluent interface methods ===

    def linear(self, column: str, weight: float, name: str = None,
               higher_is_better: bool = True) -> 'Evaluator':
        """
        Add linear criterion (fluent interface)

        Args:
            column: Column name in DataFrame
            weight: Criterion weight
            name: Display name (default: column name)
            higher_is_better: If True, higher values get higher scores

        Returns:
            Self for method chaining
        """
        self.add_criterion(column,
                           LinearCriterion(name or column, weight, higher_is_better=higher_is_better))
        return self

    def threshold(self, column: str, weight: float, thresholds: list,
                  name: str = None) -> 'Evaluator':
        """
        Add threshold criterion (fluent interface)

        Args:
            column: Column name in DataFrame
            weight: Criterion weight
            thresholds: List of (lower, upper, score) tuples
            name: Display name (default: column name)

        Returns:
            Self for method chaining
        """
        self.add_criterion(column,
                           ThresholdCriterion(name or column, weight, thresholds=thresholds))
        return self

    def direct(self, column: str, weight: float, name: str = None,
               input_scale: float = 100) -> 'Evaluator':
        """
        Add direct score criterion (fluent interface)

        Args:
            column: Column name in DataFrame
            weight: Criterion weight
            name: Display name (default: column name)
            input_scale: Scale of input scores (default: 100)

        Returns:
            Self for method chaining
        """
        self.add_criterion(column,
                           DirectScoreCriterion(name or column, weight, input_scale=input_scale))
        return self

    def min_ratio(self, column: str, weight: float, name: str = None) -> 'Evaluator':
        """
        Add minimum ratio criterion (fluent interface)

        Args:
            column: Column name in DataFrame
            weight: Criterion weight
            name: Display name (default: column name)

        Returns:
            Self for method chaining
        """
        self.add_criterion(column,
                           MinimumRatioCriterion(name or column, weight))
        return self

    def formula(self, column: str, weight: float, formula: str = 'value',
                variables: dict = None, name: str = None) -> 'Evaluator':
        """
        Add formula criterion (fluent interface)

        Args:
            column: Column name in DataFrame
            weight: Criterion weight
            formula: Math expression string (default: 'value')
            variables: Dict of custom variables available in the formula
            name: Display name (default: column name)

        Returns:
            Self for method chaining
        """
        self.add_criterion(column,
                           FormulaCriterion(name or column, weight, formula=formula, variables=variables))
        return self

    def custom(self, column: str, weight: float, func: Callable = None,
               name: str = None, **kwargs) -> 'Evaluator':
        """
        Add custom criterion (fluent interface)

        Args:
            column: Column name in DataFrame
            weight: Criterion weight
            func: Evaluation function(values, stats) -> scores
                  Can also be a string for built-in functions
            name: Display name (default: column name)
            **kwargs: Additional config passed to custom function

        Returns:
            Self for method chaining

        Examples:
            # With function reference
            .custom('price', 0.3, my_custom_func)

            # With lambda
            .custom('price', 0.3, lambda v, s: (v / s['mean']) * 100)

            # With string (built-in)
            .custom('price', 0.3, 'proximity_to_mean')
        """
        # Handle string shortcuts for common custom functions
        if isinstance(func, str):
            func = self._get_builtin_custom(func)

        self.add_criterion(column,
                           CustomCriterion(name or column, weight, func, **kwargs))
        return self

    def _get_builtin_custom(self, func_name: str) -> Callable:
        """Built-in custom functions"""
        builtins = {
            'proximity_to_mean': lambda v, s: (100 - abs((v - s['mean']) / s['mean']) * 100).clip(0),
            'proximity_to_median': lambda v, s: (100 - abs((v - s['median']) / s['median']) * 100).clip(0),
            'log_scale': lambda v, s: (np.log(v + 1) / np.log(s['max'] + 1)) * 100,
            'inverse_squared': lambda v, s: ((s['min'] / v) ** 2) * 100,
        }

        if func_name not in builtins:
            raise ValueError(f"Unknown built-in function: {func_name}. Available: {list(builtins.keys())}")

        return builtins[func_name]

    # === Core methods ===

    def add_criterion(self, column: str, criterion: CriterionBase):
        """Adds an evaluation criterion"""
        self.criteria[column] = criterion

    def remove_criterion(self, column: str):
        """Removes a criterion"""
        if column in self.criteria:
            del self.criteria[column]

    def get_total_weight(self) -> float:
        """Returns the sum of all criterion weights"""
        return sum(c.weight for c in self.criteria.values())

    def get_normalized_weights(self) -> Dict[str, float]:
        """Returns normalized weights (sum = 1.0)"""
        total = self.get_total_weight()
        if total == 0:
            return {}
        return {name: c.weight / total for name, c in self.criteria.items()}

    def evaluate(self, bids_df: pd.DataFrame,
                 include_details: bool = True) -> pd.DataFrame:
        """
        Evaluates all bids

        Args:
            bids_df: DataFrame with bid data
            include_details: If True, includes individual criterion scores

        Returns:
            DataFrame with evaluation results
        """
        result = bids_df.copy()

        # Evaluate each criterion
        criterion_scores = []
        for column, criterion in self.criteria.items():
            score = criterion.evaluate(bids_df[column])

            if include_details:
                result[f'score_{criterion.name}'] = score

            criterion_scores.append(score)

        # Calculate final score
        if criterion_scores:
            if self.normalize_weights:
                # Normalize weights to sum to 1.0
                total_weight = self.get_total_weight()
                if total_weight > 0:
                    result['final_score'] = sum(criterion_scores) / total_weight
                else:
                    result['final_score'] = 0
            else:
                # Use weights as-is
                result['final_score'] = sum(criterion_scores)
        else:
            result['final_score'] = 0

        # Ranking
        result['ranking'] = result['final_score'].rank(
            ascending=False, method='min'
        ).astype(int)

        return result.sort_values('ranking')

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Gets calculated statistics from all criteria"""
        statistics = {}

        for column, criterion in self.criteria.items():
            if criterion._statistics:
                statistics[criterion.name] = criterion._statistics

        return statistics

    def summary(self) -> pd.DataFrame:
        """Returns a summary of all configured criteria"""
        data = []
        for column, criterion in self.criteria.items():
            data.append({
                'column': column,
                'criterion_name': criterion.name,
                'type': type(criterion).__name__,
                'weight': criterion.weight,
                'normalized_weight': criterion.weight / self.get_total_weight()
                if self.get_total_weight() > 0 else 0
            })

        return pd.DataFrame(data)
