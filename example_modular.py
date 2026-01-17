# example_simple.py
import pandas as pd
from criteria import (
    Evaluator, LinearCriterion, DirectScoreCriterion,
    ThresholdCriterion, StepCriterion, MinimumRatioCriterion
)

# Bid data
bids = pd.DataFrame({
    'vendor': ['Company A', 'Company B', 'Company C', 'Company D'],
    'bid_amount': [50_000_000, 45_000_000, 52_000_000, 48_000_000],
    'experience': [8, 10, 6, 12],  # years
    'methodology': [85, 90, 75, 88],  # committee score (0-100)
    'team': [4, 5, 3, 6],  # professionals
    'certifications': [2, 4, 1, 3]  # quantity
})

# Create evaluator
evaluator = Evaluator(normalize_weights=True)

# Add all criteria (no distinction between technical/economic)
evaluator.add_criterion('experience', 
    LinearCriterion('experience', weight=0.15, higher_is_better=True))

evaluator.add_criterion('methodology',
    DirectScoreCriterion('methodology', weight=0.25, input_scale=100))

evaluator.add_criterion('team',
    ThresholdCriterion('team', weight=0.10, thresholds=[
        (0, 3, 60),
        (3, 5, 80),
        (5, float('inf'), 100)
    ]))

evaluator.add_criterion('certifications',
    StepCriterion('certifications', weight=0.10, steps=[
        (0, 50), (2, 75), (3, 90), (4, 100)
    ]))

evaluator.add_criterion('bid_amount',
    MinimumRatioCriterion('economic_bid', weight=0.40))

# Show configuration summary
print("\n=== EVALUATION CONFIGURATION ===")
print(evaluator.summary().to_string(index=False))
print(f"\nTotal weight: {evaluator.get_total_weight():.2f}")

# Evaluate
result = evaluator.evaluate(bids)

print("\n=== EVALUATION RESULTS ===")
print(result[['vendor', 'ranking', 'final_score']].to_string(index=False))

print("\n=== DETAILED BREAKDOWN ===")
detail_cols = [c for c in result.columns if c.startswith('score_')]
print(result[['vendor'] + detail_cols].to_string(index=False))

print("\n=== STATISTICS ===")
stats = evaluator.get_statistics()
for criterion, values in stats.items():
    print(f"\n{criterion}:")
    for stat, value in values.items():
        print(f"  {stat}: {value:.2f}")
