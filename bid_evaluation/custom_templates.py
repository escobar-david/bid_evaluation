"""
Pre-built custom criteria templates for common evaluation patterns.
These templates can be used directly in the web interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def budget_proximity(values: pd.Series, stats: Dict[str, Any], target: float = 100000) -> pd.Series:
    """
    Score based on closeness to a target budget.
    Values closer to the target get higher scores.

    Args:
        values: Series of bid values
        stats: Statistics dictionary (min, max, mean, etc.)
        target: Target budget value

    Returns:
        Series of scores (0-100)
    """
    deviation = abs(values - target) / target
    return ((1 - deviation) * 100).clip(0, 100)


def sweet_spot_range(values: pd.Series, stats: Dict[str, Any],
                     min_ideal: float = 30, max_ideal: float = 60) -> pd.Series:
    """
    Highest score within an ideal range, with scores dropping outside.
    Perfect for delivery time or quantity where you want a specific range.

    Args:
        values: Series of values
        stats: Statistics dictionary
        min_ideal: Lower bound of ideal range
        max_ideal: Upper bound of ideal range

    Returns:
        Series of scores (0-100)
    """
    scores = pd.Series(100.0, index=values.index)

    # Below minimum: reduce score based on distance
    below_mask = values < min_ideal
    if below_mask.any():
        distance_below = (min_ideal - values[below_mask]) / min_ideal * 100
        scores.loc[below_mask] = (100 - distance_below).clip(0, 100)

    # Above maximum: reduce score based on distance
    above_mask = values > max_ideal
    if above_mask.any():
        distance_above = (values[above_mask] - max_ideal) / max_ideal * 100
        scores.loc[above_mask] = (100 - distance_above).clip(0, 100)

    return scores


def penalty_function(values: pd.Series, stats: Dict[str, Any],
                     base_score: float = 100, threshold: float = 50,
                     penalty_per_unit: float = 2) -> pd.Series:
    """
    Base score minus penalties for exceeding a threshold.
    Useful for delivery time penalties or quantity overages.

    Args:
        values: Series of values
        stats: Statistics dictionary
        base_score: Starting score before penalties
        threshold: Value above which penalties apply
        penalty_per_unit: Points deducted per unit above threshold

    Returns:
        Series of scores (0-100)
    """
    scores = pd.Series(base_score, index=values.index)
    excess_mask = values > threshold
    if excess_mask.any():
        excess = values[excess_mask] - threshold
        scores.loc[excess_mask] = base_score - (excess * penalty_per_unit)
    return scores.clip(0, 100)


def bonus_tiers(values: pd.Series, stats: Dict[str, Any],
                base_score: float = 50,
                tier1_threshold: float = 5, tier1_bonus: float = 20,
                tier2_threshold: float = 10, tier2_bonus: float = 30) -> pd.Series:
    """
    Base score plus bonus points for meeting tier thresholds.
    Higher tiers get cumulative bonuses.

    Args:
        values: Series of values
        stats: Statistics dictionary
        base_score: Starting score
        tier1_threshold: First tier threshold
        tier1_bonus: Bonus for meeting tier 1
        tier2_threshold: Second tier threshold
        tier2_bonus: Additional bonus for meeting tier 2

    Returns:
        Series of scores (0-100)
    """
    scores = pd.Series(base_score, index=values.index)

    # Tier 1 bonus
    tier1_mask = values >= tier1_threshold
    scores.loc[tier1_mask] += tier1_bonus

    # Tier 2 bonus (cumulative)
    tier2_mask = values >= tier2_threshold
    scores.loc[tier2_mask] += tier2_bonus

    return scores.clip(0, 100)


def percentage_of_best(values: pd.Series, stats: Dict[str, Any],
                       higher_is_better: bool = True) -> pd.Series:
    """
    Score as percentage relative to the best value.

    Args:
        values: Series of values
        stats: Statistics dictionary
        higher_is_better: If True, highest value gets 100; if False, lowest gets 100

    Returns:
        Series of scores (0-100)
    """
    if higher_is_better:
        best = values.max()
        return (values / best * 100).clip(0, 100)
    else:
        best = values.min()
        return (best / values * 100).clip(0, 100)


def distance_from_mean(values: pd.Series, stats: Dict[str, Any],
                       prefer_above: bool = True) -> pd.Series:
    """
    Score based on distance from mean, preferring values above or below.

    Args:
        values: Series of values
        stats: Statistics dictionary
        prefer_above: If True, values above mean score higher

    Returns:
        Series of scores (0-100)
    """
    mean = stats.get('mean', values.mean())
    std = stats.get('std', values.std())

    if std == 0:
        return pd.Series(100.0, index=values.index)

    z_scores = (values - mean) / std

    if prefer_above:
        # Higher values get higher scores
        normalized = (z_scores + 3) / 6  # Assume most values within 3 std
    else:
        # Lower values get higher scores
        normalized = (-z_scores + 3) / 6

    return (normalized * 100).clip(0, 100)


# Registry of all available templates
TEMPLATES = {
    'budget_proximity': {
        'function': budget_proximity,
        'name': 'Budget Proximity',
        'description': 'Score based on closeness to target budget',
        'parameters': {
            'target': {'type': 'float', 'default': 100000, 'label': 'Target Value'}
        }
    },
    'sweet_spot_range': {
        'function': sweet_spot_range,
        'name': 'Sweet Spot Range',
        'description': 'Highest score within ideal range, drops outside',
        'parameters': {
            'min_ideal': {'type': 'float', 'default': 30, 'label': 'Minimum Ideal'},
            'max_ideal': {'type': 'float', 'default': 60, 'label': 'Maximum Ideal'}
        }
    },
    'penalty_function': {
        'function': penalty_function,
        'name': 'Penalty Function',
        'description': 'Base score minus penalties for exceeding threshold',
        'parameters': {
            'base_score': {'type': 'float', 'default': 100, 'label': 'Base Score'},
            'threshold': {'type': 'float', 'default': 50, 'label': 'Penalty Threshold'},
            'penalty_per_unit': {'type': 'float', 'default': 2, 'label': 'Penalty Per Unit'}
        }
    },
    'bonus_tiers': {
        'function': bonus_tiers,
        'name': 'Bonus Tiers',
        'description': 'Base score plus bonuses for meeting tier thresholds',
        'parameters': {
            'base_score': {'type': 'float', 'default': 50, 'label': 'Base Score'},
            'tier1_threshold': {'type': 'float', 'default': 5, 'label': 'Tier 1 Threshold'},
            'tier1_bonus': {'type': 'float', 'default': 20, 'label': 'Tier 1 Bonus'},
            'tier2_threshold': {'type': 'float', 'default': 10, 'label': 'Tier 2 Threshold'},
            'tier2_bonus': {'type': 'float', 'default': 30, 'label': 'Tier 2 Bonus'}
        }
    },
    'percentage_of_best': {
        'function': percentage_of_best,
        'name': 'Percentage of Best',
        'description': 'Score as percentage relative to the best value',
        'parameters': {
            'higher_is_better': {'type': 'bool', 'default': True, 'label': 'Higher is Better'}
        }
    },
    'distance_from_mean': {
        'function': distance_from_mean,
        'name': 'Distance from Mean',
        'description': 'Score based on distance from mean',
        'parameters': {
            'prefer_above': {'type': 'bool', 'default': True, 'label': 'Prefer Above Mean'}
        }
    }
}


def get_template_names() -> list:
    """Get list of available template names."""
    return list(TEMPLATES.keys())


def get_template_info(template_name: str) -> dict:
    """Get information about a specific template."""
    return TEMPLATES.get(template_name)


def apply_template(template_name: str, values: pd.Series, stats: dict, **kwargs) -> pd.Series:
    """
    Apply a template function with the given parameters.

    Args:
        template_name: Name of the template to apply
        values: Series of values to evaluate
        stats: Statistics dictionary
        **kwargs: Template-specific parameters

    Returns:
        Series of scores
    """
    template = TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Unknown template: {template_name}")

    func = template['function']
    return func(values, stats, **kwargs)
