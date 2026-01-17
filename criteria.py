# criteria.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Callable
import yaml
import json


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


class GeometricMeanCriterion(CriterionBase):
    """Evaluation using geometric mean (common for economic bids)"""

    def evaluate(self, values: pd.Series) -> pd.Series:
        self._statistics = self.calculate_statistics(values)

        # Calculate geometric mean
        positive_values = values[values > 0]
        geometric_mean = np.exp(np.log(positive_values).mean())
        self._statistics['geometric_mean'] = geometric_mean

        # Apply formula
        scores = np.where(
            values <= geometric_mean,
            100,
            100 - ((values - geometric_mean) / geometric_mean) * 100
        )

        scores = np.maximum(scores, 0)  # Don't allow negatives

        return pd.Series(scores, index=values.index) * self.weight


class MinimumRatioCriterion(CriterionBase):
    """Score = (minimum_value / value) * 100"""

    def evaluate(self, values: pd.Series) -> pd.Series:
        self._statistics = self.calculate_statistics(values)

        min_value = values.min()
        scores = (min_value / values) * 100

        return scores * self.weight


class InverseProportionalCriterion(CriterionBase):
    """Inversely proportional score"""

    def evaluate(self, values: pd.Series) -> pd.Series:
        self._statistics = self.calculate_statistics(values)

        inverses = 1 / values
        scores = (inverses / inverses.sum()) * 100

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


# evaluator.py
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
            elif criterion_type == 'geometric_mean':
                criterion = GeometricMeanCriterion(name, weight, **params)
            elif criterion_type == 'inverse':
                criterion = InverseProportionalCriterion(name, weight, **params)
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

    def geometric_mean(self, column: str, weight: float, name: str = None) -> 'Evaluator':
        """
        Add geometric mean criterion (fluent interface)

        Args:
            column: Column name in DataFrame
            weight: Criterion weight
            name: Display name (default: column name)

        Returns:
            Self for method chaining
        """
        self.add_criterion(column,
                           GeometricMeanCriterion(name or column, weight))
        return self

    def inverse(self, column: str, weight: float, name: str = None) -> 'Evaluator':
        """
        Add inverse proportional criterion (fluent interface)

        Args:
            column: Column name in DataFrame
            weight: Criterion weight
            name: Display name (default: column name)

        Returns:
            Self for method chaining
        """
        self.add_criterion(column,
                           InverseProportionalCriterion(name or column, weight))
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
                    result['final_score'] = sum(criterion_scores) / total_weight * 100
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