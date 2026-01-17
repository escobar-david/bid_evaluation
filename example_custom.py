# example_flexible.py
import pandas as pd
import numpy as np
from criteria import Evaluator, LinearCriterion, CustomCriterion, MinimumRatioCriterion

# Custom evaluation function
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
    'quality_score': [8.5, 9.2, 7.8, 8.9],  # 0-10 scale
    'delivery_days': [25, 35, 60, 40],  # days
    'warranty_months': [12, 24, 12, 18],  # months
})

# Create evaluator WITHOUT weight normalization
# (useful if you want exact control over final scores)
evaluator = Evaluator(normalize_weights=False)

# Add criteria with explicit weights
evaluator.add_criterion('bid_amount',
    MinimumRatioCriterion('price', weight=40))

evaluator.add_criterion('quality_score',
    LinearCriterion('quality', weight=30, higher_is_better=True))

evaluator.add_criterion('delivery_days',
    CustomCriterion('delivery', weight=20, 
                   evaluation_function=evaluate_delivery_time))

evaluator.add_criterion('warranty_months',
    LinearCriterion('warranty', weight=10, higher_is_better=True))

print("\n=== CONFIGURATION ===")
print(evaluator.summary().to_string(index=False))

# Evaluate
result = evaluator.evaluate(bids)

print("\n=== RESULTS ===")
print(result[['vendor', 'ranking', 'final_score']].to_string(index=False))

print("\n=== DETAILS ===")
detail_cols = [c for c in result.columns if c.startswith('score_')]
print(result[['vendor'] + detail_cols].to_string(index=False))

# You can also evaluate without details for cleaner output
print("\n=== CLEAN RESULTS ===")
clean_result = evaluator.evaluate(bids, include_details=False)
print(clean_result[['vendor', 'ranking', 'final_score']].to_string(index=False))
