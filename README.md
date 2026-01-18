# Bid Evaluation

![Status](https://img.shields.io/badge/status-alpha-orange)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bidevaluation.streamlit.app)

A flexible Python library for evaluating competitive bids using multiple weighted criteria. Designed to help procurement professionals, project managers, and organizations systematically score and rank vendor bids based on various evaluation factors.

> ⚠️ **Alpha Stage**: This library is in early development. APIs may change. Feedback welcome!

## Features

- **Multiple Evaluation Strategies**: Linear normalization, threshold-based scoring, geometric mean, ratio-based scoring, and custom functions
- **Flexible Configuration**: Dictionary, YAML, JSON, or fluent interface
- **Automatic Weight Normalization**: Optional scaling of weights to sum to 1.0
- **Built-in Statistics**: Automatic calculation of min, max, mean, median, std dev, and quartiles
- **Pandas Integration**: Works seamlessly with DataFrames for input and output

## 🚀 Quick Start

### Try Online (Fastest)
**[Launch Demo →](https://your-app.streamlit.app)** - Try it in your browser, no setup needed!



### Installation
```bash
git clone https://github.com/escobar-david/bid_evaluation.git
cd bid_evaluation
pip install -r requirements.txt
```

Or install directly from GitHub:
```bash
pip install git+https://github.com/escobar-david/bid_evaluation.git
```


### Basic Usage
```python
from bid_evaluation import Evaluator
import pandas as pd

# Load your bids
bids = pd.DataFrame({
    'vendor': ['Company A', 'Company B', 'Company C'],
    'bid_amount': [50_000_000, 45_000_000, 52_000_000],
    'experience': [8, 10, 6],
})

# Configure and evaluate
result = (Evaluator()
    .min_ratio('bid_amount', weight=0.6)
    .linear('experience', weight=0.4, higher_is_better=True)
    .evaluate(bids))

# View results
print(result[['vendor', 'ranking', 'final_score']])

# Export to Excel
result.to_excel('evaluation_results.xlsx')
```

**Output:**
```
     vendor  ranking  final_score
1  Company B        1        88.33
0  Company A        2        66.67
2  Company C        3        40.00
```

## 📚 Examples

- **[Simple evaluation](examples/example_simple.py)** - Basic usage with common criteria
- **[Hybrid approach](examples/example_hybrid.py)** - Config + fluent + custom functions
- **[Custom criteria](examples/example_custom.py)** - Write your own evaluation logic

## 🎨 Interactive Demo


### Online Demo
**🔗 [Live Demo](https://your-app.streamlit.app)** - Use it right now in your browser!


### Run Locally
```bash
streamlit run demos/streamlit_demo.py
```

IF you have more than 1 python version (streamlit could be associated with a different python installation):
```bash
python -m streamlit run demos/streamlit_demo.py
```

**Demo features:**
- 📤 Export/import evaluation configurations (local files)
- 📊 Interactive criteria configuration
- 📥 Download results as CSV or Excel
- 📈 View detailed statistics

## 📖 Documentation

### Available Criteria

#### LinearCriterion
Simple linear normalization (0-100)
```python
evaluator.linear('experience', weight=0.3, higher_is_better=True)
```

**Parameters:**
- `column`: Column name to evaluate
- `weight`: Criterion weight (0-1)
- `higher_is_better`: If True, higher values score better (default: True)

---

#### ThresholdCriterion
Assign scores based on value ranges
```python
evaluator.threshold('team_size', weight=0.2, thresholds=[
    (0, 5, 60),           # 0-4 people: 60 points
    (5, 10, 80),          # 5-9 people: 80 points
    (10, float('inf'), 100)  # 10+ people: 100 points
])
```

**Parameters:**
- `column`: Column name to evaluate
- `weight`: Criterion weight
- `thresholds`: List of `(lower, upper, score)` tuples

---

#### MinimumRatioCriterion
Score based on ratio to minimum value (common for prices)
```python
evaluator.min_ratio('bid_amount', weight=0.5)
```

**Formula:** `score = (min_value / value) * 100`

Best for: Price evaluation where lower is better

---

#### DirectScoreCriterion
Use pre-evaluated scores (e.g., from evaluation committee)
```python
evaluator.direct('committee_score', weight=0.3, input_scale=10)
```

**Parameters:**
- `input_scale`: Original scale of scores (default: 100)
- Automatically converts to 0-100 scale

---

#### GeometricMeanCriterion
Evaluation using geometric mean
```python
evaluator.geometric_mean('bid_amount', weight=0.4)
```

**Formula:**
- If `value <= geometric_mean`: 100 points
- If `value > geometric_mean`: Decreasing score

---

#### CustomCriterion
Define your own evaluation logic
```python
def proximity_to_budget(values, stats):
    """Penalize bids far from reference budget"""
    reference = 50_000_000
    deviation = abs((values - reference) / reference) * 100
    return (100 - deviation * 2).clip(lower=0)

evaluator.custom('bid_amount', weight=0.2, func=proximity_to_budget)
```

**Function signature:**
```python
def my_function(values: pd.Series, stats: dict) -> pd.Series:
    """
    Args:
        values: Column values to evaluate
        stats: Auto-calculated statistics (min, max, mean, median, std, q25, q75)
    
    Returns:
        Series of scores (0-100)
    """
    return scores
```

---

### Configuration Methods

#### Fluent Interface (Recommended)
```python
result = (Evaluator()
    .min_ratio('price', 0.4)
    .linear('experience', 0.3, higher_is_better=True)
    .direct('quality', 0.3)
    .evaluate(bids_df))
```

#### Dictionary Configuration
```python
config = {
    'price': {'type': 'min_ratio', 'weight': 0.4},
    'experience': {'type': 'linear', 'weight': 0.3, 'higher_is_better': True},
    'quality': {'type': 'direct', 'weight': 0.3}
}

evaluator = Evaluator.from_config(config)
result = evaluator.evaluate(bids_df)
```

#### YAML Configuration
```yaml
# config.yaml
criteria:
  price:
    type: min_ratio
    weight: 0.4
  
  experience:
    type: linear
    weight: 0.3
    higher_is_better: true
  
  quality:
    type: direct
    weight: 0.3
```
```python
evaluator = Evaluator.from_yaml('config.yaml')
result = evaluator.evaluate(bids_df)
```

---

### Working with Results
```python
# Evaluate
result = evaluator.evaluate(bids_df)

# Access results
print(result[['vendor', 'ranking', 'final_score']])

# Detailed scores
score_cols = [c for c in result.columns if c.startswith('score_')]
print(result[['vendor'] + score_cols])

# Get statistics
stats = evaluator.get_statistics()
for criterion, values in stats.items():
    print(f"{criterion}: min={values['min']}, max={values['max']}")

# Export
result.to_excel('results.xlsx', index=False)
result.to_csv('results.csv', index=False)
```

---

### Weight Normalization
```python
# Automatic normalization (default)
evaluator = Evaluator(normalize_weights=True)
evaluator.linear('price', 0.6)
evaluator.linear('quality', 0.4)
# Weights sum to 1.0 automatically

# Manual weights (sum must equal desired total)
evaluator = Evaluator(normalize_weights=False)
evaluator.linear('price', 60)
evaluator.linear('quality', 40)
# Final score = sum of weighted scores
```

## 🛣️ Roadmap

Planned features (vote with 👍 on issues):

- [ ] **Admissibility checks** - Required fields, min/max validation, document verification
- [ ] **Report generation** - PDF/Excel reports with charts and detailed breakdowns
- [ ] **More criterion types** - Percentile-based, exponential, custom formulas
- [ ] **Template library** - Pre-configured setups for common procurement types
- [ ] **Better documentation** - Video tutorials, comprehensive guides
- [ ] **Unit tests** - Full test coverage
- [ ] **Performance optimization** - Handle larger datasets efficiently

## 💡 Use Cases

This library is useful for:

- 🏛️ **Government procurement evaluation committees**
- 🏢 **Companies bidding on public contracts**
- 💼 **Procurement consultants and advisors**
- 🔬 **Researchers studying procurement processes**
- 📊 **Anyone needing objective, transparent bid evaluation**


## 📧 Contact
- **Email**: davesc78@gmail.com


## 📄 License

MIT License


## ⭐ Star History

If you find this useful, give it a star! ⭐

It helps others discover the project and motivates continued development.

---
