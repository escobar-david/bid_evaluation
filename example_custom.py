# example_custom.py
import pandas as pd
from criteria import ModularEvaluator, CustomCriterion, MinimumRatioCriterion
from typing import Dict

# Custom function to evaluate "proximity to reference budget"
def evaluate_budget_proximity(values: pd.Series, stats: Dict) -> pd.Series:
    """
    Rewards bids close to reference budget
    Penalizes those too far below (suspicious) or too far above
    """
    reference_budget = 50_000_000  # could come from stats or config
    
    percentage_difference = abs((values - reference_budget) / reference_budget) * 100
    
    # Score decreases with difference
    scores = 100 - (percentage_difference * 2)
    scores = scores.clip(lower=0)  # Minimum 0
    
    return scores


# Another custom function: experience with diminishing returns
def evaluate_experience_diminishing(values: pd.Series, stats: Dict) -> pd.Series:
    """
    Experience evaluation with diminishing returns
    First years count more than later years
    """
    # Logarithmic scale: log(years + 1)
    log_values = np.log(values + 1)
    
    # Normalize to 0-100
    max_log = log_values.max()
    if max_log > 0:
        scores = (log_values / max_log) * 100
    else:
        scores = pd.Series(0, index=values.index)
    
    return scores


# Bid data
bids = pd.DataFrame({
    'vendor': ['Company A', 'Company B', 'Company C', 'Company D'],
    'bid_amount': [50_000_000, 45_000_000, 52_000_000, 48_000_000],
    'experience': [2, 5, 10, 15],  # years
})

# Create evaluator
evaluator = ModularEvaluator()

# === TECHNICAL CRITERIA ===

# Experience with custom diminishing returns function
evaluator.add_technical_criterion(
    'experience',
    CustomCriterion(
        'experience_diminishing',
        weight=0.30,
        evaluation_function=evaluate_experience_diminishing
    )
)

# === ECONOMIC CRITERIA ===

# Standard economic evaluation: ratio to minimum
evaluator.add_economic_criterion(
    'bid_amount',
    MinimumRatioCriterion('economic_bid', weight=0.40)
)

# Custom: proximity to reference budget
evaluator.add_economic_criterion(
    'bid_amount',
    CustomCriterion(
        'budget_proximity',
        weight=0.30,
        evaluation_function=evaluate_budget_proximity
    )
)

# Evaluate
result = evaluator.evaluate(bids)

print("\n=== EVALUATION WITH CUSTOM CRITERIA ===")
print(result[[
    'vendor', 'ranking', 'final_score',
    'technical_score_total', 'economic_score_total'
]].to_string(index=False))

print("\n=== DETAILED BREAKDOWN ===")
detail_cols = [c for c in result.columns if c.startswith('score_')]
print(result[['vendor'] + detail_cols].to_string(index=False))

print("\n=== CALCULATED STATISTICS ===")
stats = evaluator.get_statistics()
for criterion, values in stats.items():
    print(f"\n{criterion}:")
    for stat, value in values.items():
        if isinstance(value, (int, float)):
            print(f"  {stat}: {value:.2f}")
        else:
            print(f"  {stat}: {value}")

# Additional analysis
print("\n=== WEIGHT DISTRIBUTION ===")
print(f"Technical weight: {evaluator.technical_weight:.2%}")
print(f"Economic weight: {evaluator.economic_weight:.2%}")

print("\n=== INDIVIDUAL CRITERION WEIGHTS ===")
print("Technical:")
for col, criterion in evaluator.technical_criteria.items():
    print(f"  {criterion.name}: {criterion.weight:.2%}")
print("Economic:")
for col, criterion in evaluator.economic_criteria.items():
    print(f"  {criterion.name}: {criterion.weight:.2%}")
