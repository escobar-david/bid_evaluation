# example_config.py
"""Here we create an Evaluator object from a configuration dict. from_config doesn't support custom criteria"""


import pandas as pd
from criteria import Evaluator

bids = pd.DataFrame({
    'vendor': ['Company A', 'Company B', 'Company C'],
    'bid_amount': [50_000_000, 45_000_000, 52_000_000],
    'experience': [8, 10, 6],
    'methodology': [85, 90, 75],
})

# Config dict
config = {
    'experience': {'type': 'linear', 'weight': 0.20, 'higher_is_better': True},
    'methodology': {'type': 'direct', 'weight': 0.40},
    'bid_amount': {'type': 'min_ratio', 'weight': 0.40}
}

evaluator = Evaluator.from_config(config)
result = evaluator.evaluate(bids)
print(result[['vendor', 'ranking', 'final_score']])