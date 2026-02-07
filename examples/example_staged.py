# example_staged.py
"""Examples of multi-stage bid evaluation."""

import pandas as pd
from bid_evaluation import StagedEvaluator

# Sample bid data
bids = pd.DataFrame({
    'vendor': ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon'],
    'experience_years': [12, 4, 9, 2, 7],
    'quality_score': [85, 55, 92, 45, 70],
    'bid_amount': [100_000, 88_000, 115_000, 82_000, 105_000],
    'delivery_days': [30, 50, 20, 65, 35],
})

# ── Example 1: Fluent interface with score_threshold ──

print("=== Example 1: Fluent interface (score_threshold) ===\n")

result = (StagedEvaluator()
    .add_stage('Technical', filter_type='score_threshold', threshold=60)
        .linear('experience_years', 0.4, higher_is_better=True)
        .direct('quality_score', 0.6)
    .add_stage('Economic')
        .min_ratio('bid_amount', 0.6)
        .linear('delivery_days', 0.4, higher_is_better=False)
    .evaluate(bids))

print(result[['vendor', 'stage_technical_score', 'stage_economic_score',
              'eliminated_at_stage', 'final_score', 'ranking']])
print()

# ── Example 2: Config-based ──

print("=== Example 2: Config-based ===\n")

config = {
    'stages': [
        {
            'name': 'Technical',
            'filter': {'type': 'top_n', 'top_n': 3},
            'criteria': {
                'experience_years': {'type': 'linear', 'weight': 0.4, 'higher_is_better': True},
                'quality_score': {'type': 'direct', 'weight': 0.6},
            }
        },
        {
            'name': 'Economic',
            'criteria': {
                'bid_amount': {'type': 'min_ratio', 'weight': 1.0},
            }
        }
    ]
}

result = StagedEvaluator.from_config(config).evaluate(bids)
print(result[['vendor', 'stage_technical_score', 'stage_economic_score',
              'eliminated_at_stage', 'final_score', 'ranking']])
print()

# ── Example 3: Weighted combination ──

print("=== Example 3: Weighted combination ===\n")

result = (StagedEvaluator(final_score_mode='weighted_combination')
    .add_stage('Technical', filter_type='score_threshold', threshold=55, weight=0.6)
        .direct('quality_score', 1.0)
    .add_stage('Economic', weight=0.4)
        .min_ratio('bid_amount', 1.0)
    .evaluate(bids))

print(result[['vendor', 'stage_technical_score', 'stage_economic_score',
              'eliminated_at_stage', 'final_score', 'ranking']])
print()

# ── Summary ──

print("=== Stage Summary ===\n")
staged = (StagedEvaluator()
    .add_stage('Technical', filter_type='score_threshold', threshold=60)
        .linear('experience_years', 0.4, higher_is_better=True)
        .direct('quality_score', 0.6)
    .add_stage('Economic')
        .min_ratio('bid_amount', 1.0))

print(staged.summary())
