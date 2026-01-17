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
class ModularEvaluator:
    """Modular evaluation engine"""
    
    def __init__(self):
        self.technical_criteria: Dict[str, CriterionBase] = {}
        self.economic_criteria: Dict[str, CriterionBase] = {}
        self.technical_weight: float = 0.0
        self.economic_weight: float = 0.0
    
    def add_technical_criterion(self, column: str, criterion: CriterionBase):
        """Adds a technical criterion"""
        self.technical_criteria[column] = criterion
        self._recalculate_weights()
    
    def add_economic_criterion(self, column: str, criterion: CriterionBase):
        """Adds an economic criterion"""
        self.economic_criteria[column] = criterion
        self._recalculate_weights()
    
    def _recalculate_weights(self):
        """Automatically recalculates weights"""
        tech_weight = sum(c.weight for c in self.technical_criteria.values())
        eco_weight = sum(c.weight for c in self.economic_criteria.values())
        
        total = tech_weight + eco_weight
        
        if total > 0:
            self.technical_weight = tech_weight / total
            self.economic_weight = eco_weight / total
    
    def evaluate(self, bids_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates all bids"""
        result = bids_df.copy()
        
        # Evaluate technical criteria
        technical_scores = []
        for column, criterion in self.technical_criteria.items():
            score = criterion.evaluate(bids_df[column])
            result[f'score_{criterion.name}'] = score
            technical_scores.append(score)
        
        if technical_scores:
            result['technical_score_total'] = sum(technical_scores)
        else:
            result['technical_score_total'] = 0
        
        # Evaluate economic criteria
        economic_scores = []
        for column, criterion in self.economic_criteria.items():
            score = criterion.evaluate(bids_df[column])
            result[f'score_{criterion.name}'] = score
            economic_scores.append(score)
        
        if economic_scores:
            result['economic_score_total'] = sum(economic_scores)
        else:
            result['economic_score_total'] = 0
        
        # Final score
        result['final_score'] = (
            result['technical_score_total'] * self.technical_weight +
            result['economic_score_total'] * self.economic_weight
        )
        
        # Ranking
        result['ranking'] = result['final_score'].rank(
            ascending=False, method='min'
        )
        
        return result.sort_values('ranking')
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Gets calculated statistics from all criteria"""
        statistics = {}
        
        for column, criterion in {**self.technical_criteria, 
                                   **self.economic_criteria}.items():
            if criterion._statistics:
                statistics[criterion.name] = criterion._statistics
        
        return statistics
