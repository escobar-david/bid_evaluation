# example_custom.py
import pandas as pd
from typing import Dict
from bid_evaluation import Evaluator, CustomCriterion, MinimumRatioCriterion, LinearCriterion


# Custom function to evaluate "proximity to reference budget"
def evaluate_budget_proximity(values: pd.Series, stats: Dict) -> pd.Series:
    """
    Rewards bids close to reference budget
    Penalizes those too far below (suspicious) or too far above
    """
    reference_budget = 50_000_000

    percentage_difference = abs((values - reference_budget) / reference_budget) * 100

    # Score decreases with difference
    scores = 100 - (percentage_difference * 2)
    scores = scores.clip(lower=0)

    return scores


# Another custom function: delivery time with sweet spot
def evaluate_delivery_time(values: pd.Series, stats: Dict) -> pd.Series:
    """
    Evaluates delivery time with penalty for extremes
    Sweet spot: 30-45 days
    """
    ideal_min = 30
    ideal_max = 45

    scores = pd.Series(100.0, index=values.index)

    # Penalty for too fast (suspicious)
    too_fast = values < ideal_min
    scores[too_fast] = 100 - ((ideal_min - values[too_fast]) * 3)

    # Penalty for too slow
    too_slow = values > ideal_max
    scores[too_slow] = 100 - ((values[too_slow] - ideal_max) * 2)

    return scores.clip(lower=0)


# Bid data
bids = pd.DataFrame({
    'vendor': ['Company A', 'Company B', 'Company C', 'Company D'],
    'bid_amount': [50_000_000, 45_000_000, 52_000_000, 48_000_000],
    'quality_score': [8.5, 9.2, 7.8, 8.9],
    'delivery_days': [25, 35, 60, 40],
    'warranty_months': [12, 24, 12, 18],
})

# Create evaluator
evaluator = Evaluator(normalize_weights=True)

# Standard criteria
evaluator.add_criterion('bid_amount',
                        MinimumRatioCriterion('price', weight=0.30))

evaluator.add_criterion('quality_score',
                        LinearCriterion('quality', weight=0.20, higher_is_better=True))

evaluator.add_criterion('warranty_months',
                        LinearCriterion('warranty', weight=0.10, higher_is_better=True))

# Custom criteria
evaluator.add_criterion('bid_amount',
                        CustomCriterion('budget_proximity', weight=0.20,
                                        evaluation_function=evaluate_budget_proximity))

evaluator.add_criterion('delivery_days',
                        CustomCriterion('delivery', weight=0.20,
                                        evaluation_function=evaluate_delivery_time))

print("\n=== CONFIGURATION ===")
print(evaluator.summary().to_string(index=False))

# Evaluate
result = evaluator.evaluate(bids)

print("\n=== RESULTS ===")
print(result[['vendor', 'ranking', 'final_score']].to_string(index=False))

print("\n=== DETAILS ===")
detail_cols = [c for c in result.columns if c.startswith('score_')]
print(result[['vendor'] + detail_cols].to_string(index=False))

print("\n=== STATISTICS ===")
stats = evaluator.get_statistics()
for criterion, values in stats.items():
    print(f"\n{criterion}:")
    for stat, value in values.items():
        if isinstance(value, (int, float)):
            print(f"  {stat}: {value:.2f}")
        else:
            print(f"  {stat}: {value}")