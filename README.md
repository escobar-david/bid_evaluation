# Bid Evaluation

![Status](https://img.shields.io/badge/status-alpha-orange)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A flexible Python library for evaluating competitive bids using multiple weighted criteria. Designed to help procurement professionals, project managers, and organizations systematically score and rank vendor bids based on various evaluation factors.

> ⚠️ **Alpha Stage**: This library is in early development. APIs may change. Feedback welcome!

> **🆕 Multi-Stage Evaluation** — Evaluate bids in sequential stages with automatic filtering between them. Eliminate unqualified bids at the technical stage before scoring economics. Supports score thresholds, top-N filters, tie-breaking rules, and weighted stage combinations. [Jump to docs →](#multi-stage-evaluation)

## Features

- **Multiple Evaluation Strategies**: Linear normalization, threshold-based scoring, ratio-based scoring, formula expressions, and custom functions
- **Multi-Stage Evaluation**: Sequential stages with filtering between them — eliminate bids that don't meet technical requirements before scoring economics
- **Flexible Configuration**: Dictionary, YAML, JSON, or fluent interface
- **Automatic Weight Normalization**: Optional scaling of weights to sum to 1.0
- **Built-in Statistics**: Automatic calculation of min, max, mean, median, std dev, and quartiles
- **Pandas Integration**: Works seamlessly with DataFrames for input and output

## 🚀 Quick Start

### Try Online
**Library Demo** (hosted): https://bidevaluation.streamlit.app/



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

### Staged Evaluation (Technical → Economic)

```python
from bid_evaluation import StagedEvaluator

result = (StagedEvaluator()
    # Stage 1: Technical — bids scoring below 60 are eliminated
    .add_stage('Technical', filter_type='score_threshold', threshold=60)
        .linear('experience', 0.4, higher_is_better=True)
        .direct('quality_score', 0.6)
    # Stage 2: Economic — only surviving bids are ranked
    .add_stage('Economic')
        .min_ratio('bid_amount', 1.0)
    .evaluate(bids))
```

Eliminated bids are marked in the `eliminated_at_stage` column and excluded from the final ranking. [Full staged evaluation docs →](#multi-stage-evaluation)

## 📚 Examples

- **[Simple evaluation](examples/example_simple.py)** - Basic usage with common criteria
- **[Hybrid approach](examples/example_hybrid.py)** - Config + fluent + custom functions
- **[Custom criteria](examples/example_custom.py)** - Write your own evaluation logic
- **[Staged evaluation](examples/example_staged.py)** - Multi-stage evaluation with filtering

## 🎨 Streamlit Demos (Open Source)

**Single-stage demo** (basic evaluation):
```bash
streamlit run demos/streamlit_demo.py
```

**Staged evaluation demo** (multi-stage with filtering):
```bash
streamlit run demos/streamlit_staged_demo.py
```

If you have more than 1 python version (streamlit could be associated with a different python installation):
```bash
python -m streamlit run demos/streamlit_demo.py
```

These demos are provided as open-source examples for library users.

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

#### FormulaCriterion
Score bids using a math expression. Uses `simpleeval` for safe evaluation.
```python
evaluator.formula('bid_amount', weight=0.4,
                  formula='100 - abs(value - target) / target * 100',
                  variables={'target': 50_000_000})
```

**Available in formulas:**
- `value` — the current bid value
- `min`, `max`, `mean`, `median`, `std` — statistics from all values
- Custom variables passed via `variables` dict
- Functions: `abs`, `min`, `max`, `sqrt`, `log`, `log10`, `exp`, `clip(x, lo, hi)`

**Config-based:**
```python
config = {
    'bid_amount': {
        'type': 'formula',
        'weight': 0.4,
        'formula': '100 - abs(value - target) / target * 100',
        'variables': {'target': 50_000_000}
    }
}
```

Scores are automatically clipped to 0–100. Invalid expressions return 0.

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

## Multi-Stage Evaluation

Real-world procurement often evaluates bids in stages: a technical stage eliminates unqualified bids, then an economic stage ranks the survivors. `StagedEvaluator` supports this pattern.

### Quick Start

```python
from bid_evaluation import StagedEvaluator
import pandas as pd

bids = pd.DataFrame({
    'vendor': ['Alpha', 'Beta', 'Gamma', 'Delta'],
    'experience': [15, 3, 10, 7],
    'quality_score': [88, 45, 92, 65],
    'bid_amount': [120_000, 85_000, 145_000, 95_000],
})

result = (StagedEvaluator()
    .add_stage('Technical', filter_type='score_threshold', threshold=60)
        .linear('experience', 0.4, higher_is_better=True)
        .direct('quality_score', 0.6)
    .add_stage('Economic')
        .min_ratio('bid_amount', 1.0)
    .evaluate(bids))

print(result[['vendor', 'technical_score', 'eliminated_at_stage', 'final_score', 'ranking']])
```

### How It Works

1. Bids are evaluated in **sequential stages**, each with its own criteria
2. After each stage (except the last), a **filter** can eliminate bids:
   - `score_threshold` — bids must score at or above a minimum
   - `top_n` — only the top N bids advance (with configurable tie-breaking)
3. Only surviving bids advance to the next stage
4. Eliminated bids are marked with the stage where they were removed

### Filter Types

```python
# Score threshold: bids must score >= 60 to advance
.add_stage('Technical', filter_type='score_threshold', threshold=60)

# Top N: only the best 5 bids advance
.add_stage('Shortlist', filter_type='top_n', top_n=5)

# Top N with tie-breaking: exclude tied bids at the cutoff
.add_stage('Shortlist', filter_type='top_n', top_n=5, on_tie='exclude')
```

### Final Score Modes

```python
# Default: ranking based on the last stage's score only
staged = StagedEvaluator(final_score_mode='last_stage')

# Weighted combination: weighted average of all stage scores
staged = StagedEvaluator(final_score_mode='weighted_combination')
```

### Config-Based Setup

```python
config = {
    'final_score_mode': 'last_stage',
    'stages': [
        {
            'name': 'Technical',
            'weight': 0.6,
            'filter': {'type': 'score_threshold', 'threshold': 60},
            'criteria': {
                'experience': {'type': 'linear', 'weight': 0.4, 'higher_is_better': True},
                'quality_score': {'type': 'direct', 'weight': 0.6}
            }
        },
        {
            'name': 'Economic',
            'weight': 0.4,
            'criteria': {
                'bid_amount': {'type': 'min_ratio', 'weight': 1.0}
            }
        }
    ]
}

result = StagedEvaluator.from_config(config).evaluate(bids)

# Also available: from_yaml() and from_json()
```

### Output Columns

The result DataFrame includes:
- `{name}_score` — score per stage
- `{name}_ranking` — ranking within each stage
- `eliminated_at_stage` — stage name where the bid was eliminated, or `None`
- `final_score` — overall score (from last stage or weighted combination)
- `ranking` — final ranking (`NaN` for eliminated bids)

### Inspection

```python
# Summary of all stages, criteria, and filters
staged.summary()

# Per-stage statistics (after evaluation)
staged.get_statistics()

# Detailed stage results (advanced/eliminated indices)
staged.get_stage_results()
```

For full documentation, see [README_STAGED.md](README_STAGED.md).

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

- [x] **Multi-stage evaluation** - Sequential stages with filtering between them
- [x] **Formula criterion** - User-defined math expressions via simpleeval
- [x] **Unit tests** - Test coverage for core and staged evaluation
- [ ] **Admissibility checks** - Required fields, min/max validation, document verification
- [ ] **Report generation** - PDF/Excel reports with charts and detailed breakdowns
- [ ] **Template library** - Pre-configured setups for common procurement types
- [ ] **Better documentation** - Video tutorials, comprehensive guides
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
