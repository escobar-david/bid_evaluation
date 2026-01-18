# Bid Evaluation

A flexible Python library for evaluating competitive bids using multiple weighted criteria. Designed to help procurement professionals, project managers, and organizations systematically score and rank vendor bids based on various evaluation factors.

## Features

- **Multiple Evaluation Strategies**: Linear normalization, threshold-based scoring, geometric mean, ratio-based scoring, and custom functions
- **Flexible Configuration**: Dictionary, YAML, JSON, or fluent interface
- **Automatic Weight Normalization**: Optional scaling of weights to sum to 1.0
- **Built-in Statistics**: Automatic calculation of min, max, mean, median, std dev, and quartiles
- **Pandas Integration**: Works seamlessly with DataFrames for input and output

## Installation

```bash
pip install pandas numpy PyYAML
```

## Quick Start

### Fluent Interface (Recommended)

```python
from criteria import Evaluator
import pandas as pd

bids = pd.DataFrame({
    'vendor': ['A', 'B', 'C'],
    'experience': [10, 5, 8],
    'methodology': [85, 90, 75],
    'bid_amount': [100000, 95000, 110000]
})

result = (Evaluator()
    .linear('experience', 0.20, higher_is_better=True)
    .direct('methodology', 0.40)
    .min_ratio('bid_amount', 0.40)
    .evaluate(bids))

print(result)
```

### Configuration Dictionary

```python
from criteria import Evaluator

config = {
    'experience': {'type': 'linear', 'weight': 0.20, 'higher_is_better': True},
    'methodology': {'type': 'direct', 'weight': 0.40},
    'bid_amount': {'type': 'min_ratio', 'weight': 0.40}
}

evaluator = Evaluator.from_config(config)
result = evaluator.evaluate(bids)
```

## Criterion Types

| Type | Description | Use Case |
|------|-------------|----------|
| `linear` | Linear normalization (0-100 scale) | Experience, ratings |
| `threshold` | Score ranges based on value thresholds | Team size, certifications |
| `direct` | Uses pre-scored values directly | Committee scores |
| `geometric_mean` | Geometric mean normalization | Economic evaluations |
| `min_ratio` | Ratio to minimum value | Price comparison |
| `inverse` | Inversely proportional scoring | Delivery time |
| `custom` | Custom evaluation function | Complex business logic |

## Detailed Usage

### Linear Criterion

Normalizes values to a 0-100 scale. Set `higher_is_better=True` for metrics where larger values are desirable (e.g., experience years).

```python
evaluator.linear('experience', weight=0.20, higher_is_better=True)
```

### Threshold Criterion

Assigns scores based on value ranges:

```python
evaluator.threshold('team_size', weight=0.10, thresholds=[
    (10, 100),  # >= 10 team members: 100 points
    (5, 75),    # >= 5 team members: 75 points
    (3, 50),    # >= 3 team members: 50 points
    (0, 25)     # < 3 team members: 25 points
])
```

### Minimum Ratio Criterion

Scores based on ratio to the minimum value (common for price evaluation):

```python
evaluator.min_ratio('bid_amount', weight=0.40)
```

### Custom Criterion

Define your own evaluation logic:

```python
def budget_proximity(values, stats):
    target_budget = 100000
    deviation = abs(values - target_budget) / target_budget
    return (1 - deviation).clip(0, 1) * 100

evaluator.custom('bid_amount', weight=0.15, func=budget_proximity)
```

## Output

The `evaluate()` method returns a DataFrame with:

- `final_score`: Weighted sum of all criterion scores
- `rank`: Overall ranking (1 = best)
- Individual criterion scores (e.g., `experience_score`, `bid_amount_score`)

```
   vendor  experience  bid_amount  final_score  rank  experience_score  bid_amount_score
0       A          10      100000        82.50     1            100.00             95.00
1       B           5       95000        79.00     2             50.00            100.00
2       C           8      110000        71.36     3             80.00             86.36
```

## Statistics

Get detailed statistics for each criterion:

```python
stats = evaluator.get_statistics(bids)
for name, criterion_stats in stats.items():
    print(f"{name}: min={criterion_stats['min']}, max={criterion_stats['max']}")
```

## Examples

The repository includes several example files:

- `example_simple.py` - Traditional API with `add_criterion()`
- `example_fluent.py` - Concise fluent interface
- `example_config.py` - Configuration dictionary approach
- `example_custom.py` - Custom evaluation functions
- `example_hybrid.py` - Combined approach with full output

Run any example:

```bash
python examples/example_hybrid.py
```

## Web Interface (Streamlit Demo)

A full-featured web interface is included in the `demos/` directory. The Streamlit app provides:

- **Excel file upload** with multi-sheet support
- **Interactive criteria configuration** with all criterion types
- **Formula builder** for custom scoring expressions
- **Real-time evaluation** with score breakdown charts
- **Export results** to Excel with statistics and configuration
- **Save/load configurations** for reuse

### Running the Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run demos/streamlit_demo.py
```

### Screenshot

The web interface allows you to:
1. Upload your bid data (Excel format)
2. Add evaluation criteria with weights
3. Configure criterion-specific parameters
4. Run evaluation and view results
5. Export results and save configurations

## Use Cases

- Procurement bid evaluation for government/enterprise contracts
- Vendor selection with multiple criteria
- RFP (Request for Proposal) scoring
- Project bidding systems
- Multi-factor decision making with standardized scoring
