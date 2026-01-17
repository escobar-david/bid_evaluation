# criteria.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Callable


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


class StepCriterion(CriterionBase):
    """Stepped score according to ranges"""
    
    def evaluate(self, values: pd.Series) -> pd.Series:
        self._statistics = self.calculate_statistics(values)
        
        steps = self.config.get('steps', [])
        # steps: [(minimum_value, score), ...] ordered ascending
        
        scores = pd.Series(0.0, index=values.index)
        
        for i, (min_value, score) in enumerate(steps):
            mask = values >= min_value
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
    """Simplified evaluation engine"""
    
    def __init__(self, normalize_weights: bool = True):
        """
        Args:
            normalize_weights: If True, automatically normalizes weights to sum to 1.0
        """
        self.criteria: Dict[str, CriterionBase] = {}
        self.normalize_weights = normalize_weights
    
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
