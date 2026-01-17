# example_hybrid.py

"""from_config doesn't support custom criteria. For custom criteria use
 fluent interface or evaluator.add_criterion methods instead"""

import pandas as pd
import numpy as np
from criteria import Evaluator

# Sample bid data
bids = pd.DataFrame({
    'vendor': ['Company A', 'Company B', 'Company C', 'Company D', 'Company E'],
    'bid_amount': [50_000_000, 45_000_000, 52_000_000, 48_000_000, 55_000_000],
    'methodology': [85, 90, 75, 88, 82],  # Committee score (0-100)
    'team': [4, 5, 3, 6, 2],  # Number of professionals
    'experience': [8, 10, 6, 12, 5],  # Years
    'delivery_days': [30, 25, 45, 35, 60],  # Delivery time in days
})


# Custom functions
def budget_proximity(values, stats):
    """
    Evaluates how close bids are to reference budget
    Penalizes bids too far from reference (suspicious if too low, expensive if too high)
    """
    reference_budget = 50_000_000
    percentage_difference = abs((values - reference_budget) / reference_budget) * 100
    scores = 100 - (percentage_difference * 2)
    return scores.clip(lower=0)


def delivery_sweet_spot(values, stats):
    """
    Evaluates delivery time with optimal range
    Sweet spot: 30-45 days
    Penalizes too fast (suspicious) or too slow
    """
    ideal_min = 30
    ideal_max = 45

    scores = pd.Series(100.0, index=values.index)

    # Penalty for too fast (suspicious quality)
    too_fast = values < ideal_min
    scores[too_fast] = 100 - ((ideal_min - values[too_fast]) * 3)

    # Penalty for too slow
    too_slow = values > ideal_max
    scores[too_slow] = 100 - ((values[too_slow] - ideal_max) * 2)

    return scores.clip(lower=0)


# Simple criteria configuration (dict)
simple_config = {
    'methodology': {
        'type': 'direct',
        'weight': 0.25,
        'name': 'methodology_score'
    },
    'team': {
        'type': 'threshold',
        'weight': 0.10,
        'name': 'team_size',
        'thresholds': [
            (0, 3, 60),  # 0-2 professionals: 60 points
            (3, 5, 80),  # 3-4 professionals: 80 points
            (5, float('inf'), 100)  # 5+ professionals: 100 points
        ]
    }
}

# Build evaluator: config + fluent + custom
print("=== BUILDING EVALUATOR ===")
evaluator = (Evaluator.from_config(simple_config)
             .linear('experience', 0.15, higher_is_better=True, name='experience')
             .min_ratio('bid_amount', 0.25, name='price')
             .custom('bid_amount', 0.15, budget_proximity, name='proximity')
             .custom('delivery_days', 0.10, delivery_sweet_spot, name='delivery'))

# Show configuration
print("\n=== CONFIGURATION SUMMARY ===")
print(evaluator.summary().to_string(index=False))
print(f"\nTotal weight: {evaluator.get_total_weight():.2f}")

# Evaluate
print("\n=== EVALUATING BIDS ===")
result = evaluator.evaluate(bids)

# Show results
print("\n=== FINAL RESULTS ===")
print(result[['vendor', 'ranking', 'final_score']].to_string(index=False))

print("\n=== DETAILED SCORES ===")
detail_cols = ['vendor'] + sorted([c for c in result.columns if c.startswith('score_')])
print(result[detail_cols].to_string(index=False))

# Show statistics
print("\n=== STATISTICS ===")
stats = evaluator.get_statistics()
for criterion, values in stats.items():
    print(f"\n{criterion}:")
    for stat, value in values.items():
        if isinstance(value, (int, float)):
            print(f"  {stat}: {value:,.2f}")
        else:
            print(f"  {stat}: {value}")

# Show input data for reference
print("\n=== INPUT DATA ===")
print(bids.to_string(index=False))

# Additional analysis
print("\n=== WEIGHT DISTRIBUTION ===")
normalized = evaluator.get_normalized_weights()
for criterion_col, weight in normalized.items():
    criterion_name = evaluator.criteria[criterion_col].name
    print(f"{criterion_name}: {weight:.1%}")

# Show breakdown by vendor - DYNAMIC VERSION
print("\n=== SCORE BREAKDOWN BY VENDOR ===")
score_columns = [col for col in result.columns if col.startswith('score_')]
for idx, row in result.iterrows():
    print(f"\n{row['vendor']} (Rank #{int(row['ranking'])}, Final: {row['final_score']:.2f})")
    for score_col in sorted(score_columns):
        criterion_name = score_col.replace('score_', '').replace('_', ' ').title()
        print(f"  {criterion_name:25s}: {row[score_col]:6.2f}")