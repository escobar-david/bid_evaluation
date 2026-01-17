# example_fluent.py
import pandas as pd
from criteria import Evaluator

bids = pd.DataFrame({
    'vendor': ['Company A', 'Company B', 'Company C'],
    'bid_amount': [50_000_000, 45_000_000, 52_000_000],
    'experience': [8, 10, 6],
    'methodology': [85, 90, 75],
})

# Fluent interface
result = (Evaluator()
    .linear('experience', 0.20, higher_is_better=True)
    .direct('methodology', 0.40)
    .min_ratio('bid_amount', 0.40)
    .evaluate(bids))

print(result[['vendor', 'ranking', 'final_score']])
