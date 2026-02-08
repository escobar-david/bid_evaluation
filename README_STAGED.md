# Multi-Stage Bid Evaluation

The `StagedEvaluator` extends the bid evaluation library with sequential, multi-stage evaluation. Bids are scored in stages, and a configurable filter between stages eliminates underperforming bids before the next round.

## Why Staged Evaluation?

Many real-world procurement processes work in phases:

1. **Technical screening** -- reject bids that don't meet a minimum quality bar.
2. **Shortlisting** -- narrow the field to the top N candidates.
3. **Economic evaluation** -- rank the survivors on price.

`StagedEvaluator` models this directly. Each stage has its own criteria and scoring, and an optional filter controls which bids advance.

## Quick Start

```python
import pandas as pd
from bid_evaluation import StagedEvaluator

bids = pd.DataFrame({
    'vendor': ['Alpha', 'Beta', 'Gamma', 'Delta'],
    'experience': [12, 4, 9, 2],
    'quality_score': [85, 55, 92, 45],
    'bid_amount': [100_000, 88_000, 115_000, 82_000],
})

result = (StagedEvaluator()
    .add_stage('Technical', filter_type='score_threshold', threshold=60)
        .linear('experience', 0.4, higher_is_better=True)
        .direct('quality_score', 0.6)
    .add_stage('Economic')
        .min_ratio('bid_amount', 1.0)
    .evaluate(bids))

print(result[['vendor', 'technical_score', 'economic_score',
              'eliminated_at_stage', 'final_score', 'ranking']])
```

## How It Works

```
Stage 1              Stage 2              Stage 3
+-----------+        +-----------+        +-----------+
| Evaluate  |        | Evaluate  |        | Evaluate  |
| all bids  |--filter-->surviving |--filter-->surviving |-->final ranking
|           |        | bids only |        | bids only |
+-----------+        +-----------+        +-----------+
     |                    |
     v                    v
  eliminated           eliminated
  (NaN ranking)        (NaN ranking)
```

1. Each stage evaluates only the bids that haven't been eliminated yet.
2. After scoring, the stage's filter decides who advances (except on the last stage).
3. Eliminated bids are tagged with the stage that removed them.
4. The final ranking is computed from the surviving bids only.

## API Reference

### Creating a StagedEvaluator

#### Fluent Interface

```python
staged = (StagedEvaluator()
    .add_stage('Technical', filter_type='score_threshold', threshold=60)
        .linear('experience', 0.4, higher_is_better=True)
        .direct('quality_score', 0.6)
    .add_stage('Economic')
        .min_ratio('bid_amount', 1.0))

result = staged.evaluate(bids_df)
```

After calling `.add_stage(...)`, all subsequent criterion methods (`.linear()`, `.direct()`, `.min_ratio()`, `.threshold()`, `.formula()`, `.custom()`) add criteria to **that** stage. Calling `.add_stage(...)` again starts a new stage.

#### Config Dictionary

```python
config = {
    'stages': [
        {
            'name': 'Technical',
            'filter': {'type': 'score_threshold', 'threshold': 60},
            'criteria': {
                'experience': {'type': 'linear', 'weight': 0.4, 'higher_is_better': True},
                'quality_score': {'type': 'direct', 'weight': 0.6}
            }
        },
        {
            'name': 'Economic',
            'criteria': {
                'bid_amount': {'type': 'min_ratio', 'weight': 1.0}
            }
        }
    ]
}

result = StagedEvaluator.from_config(config).evaluate(bids_df)
```

#### YAML / JSON Files

```python
result = StagedEvaluator.from_yaml('evaluation.yaml').evaluate(bids_df)
result = StagedEvaluator.from_json('evaluation.json').evaluate(bids_df)
```

Example YAML:

```yaml
stages:
  - name: Technical
    filter:
      type: top_n
      top_n: 5
    criteria:
      experience:
        type: linear
        weight: 0.4
        higher_is_better: true
      quality_score:
        type: direct
        weight: 0.6

  - name: Economic
    criteria:
      bid_amount:
        type: min_ratio
        weight: 1.0
```

### `add_stage()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *(required)* | Stage name, used in output column prefixes |
| `filter_type` | `str` or `None` | `None` | `'score_threshold'`, `'top_n'`, or `None` (no filter) |
| `threshold` | `float` | `None` | Minimum score to advance (required when `filter_type='score_threshold'`) |
| `top_n` | `int` | `None` | Number of top bids to advance (required when `filter_type='top_n'`) |
| `on_tie` | `str` | `'include'` | Tie-breaking at cutoff: `'include'` or `'exclude'` |
| `weight` | `float` | `1.0` | Stage weight (used only in `weighted_combination` mode) |

### Filter Types

**Score Threshold** -- bids must score at or above the threshold:

```python
.add_stage('Technical', filter_type='score_threshold', threshold=60)
```

**Top N** -- only the N highest-scoring bids advance:

```python
.add_stage('Shortlist', filter_type='top_n', top_n=5)
```

**No Filter** -- all bids advance (omit `filter_type` or set to `None`):

```python
.add_stage('Final')
```

### Tie-Breaking with `on_tie`

When using `top_n`, ties at the cutoff boundary need a policy:

- **`on_tie='include'`** (default) -- all tied bids at the Nth rank advance. The number of advancing bids may exceed N.
- **`on_tie='exclude'`** -- only bids strictly above the tied score advance. The number of advancing bids may be fewer than N.

```python
# Scores: A=90, B=80, C=80, D=70 with top_n=2
# include: A, B, C advance (3 bids — both rank-2 bids included)
# exclude: only A advances (1 bid — the tie at rank 2 is excluded)
```

### Final Score Modes

#### `last_stage` (default)

The final score equals the last stage's score. Earlier stages serve purely as filters.

```python
StagedEvaluator()  # defaults to final_score_mode='last_stage'
```

#### `weighted_combination`

The final score is a weighted average of all stage scores. Set per-stage weights via the `weight` parameter on `add_stage()`.

```python
result = (StagedEvaluator(final_score_mode='weighted_combination')
    .add_stage('Technical', filter_type='score_threshold', threshold=60, weight=0.6)
        .direct('quality_score', 1.0)
    .add_stage('Economic', weight=0.4)
        .min_ratio('bid_amount', 1.0)
    .evaluate(bids_df))

# final_score = technical_score * 0.6 + economic_score * 0.4
```

Weights are normalized to sum to 1.0.

### Output Columns

The result DataFrame contains all original columns plus:

| Column | Description |
|--------|-------------|
| `{name}_score` | Aggregate score for that stage |
| `{name}_ranking` | Ranking within that stage |
| `{name}_{criterion}` | Per-criterion score (when `include_details=True`) |
| `eliminated_at_stage` | Name of the stage that eliminated the bid, or `None` |
| `final_score` | Final score (from last stage or weighted combination) |
| `ranking` | Final ranking (`NaN` for eliminated bids) |

The output is sorted with ranked bids first (ascending by rank), followed by eliminated bids.

### Inspection Methods

```python
staged = (StagedEvaluator()
    .add_stage('Technical', filter_type='score_threshold', threshold=60)
        .linear('experience', 0.4, higher_is_better=True)
        .direct('quality_score', 0.6)
    .add_stage('Economic')
        .min_ratio('bid_amount', 1.0))

# Before evaluation: view configured stages and criteria
staged.summary()
#   stage  stage_weight         filter       column  criterion_type  criterion_weight
# Technical         1.0  score >= 60.0   experience  LinearCriterion             0.4
# Technical         1.0  score >= 60.0  quality_score  DirectScoreCriterion      0.6
# Economic          1.0           None   bid_amount  MinimumRatioCriterion       1.0

# After evaluation
result = staged.evaluate(bids_df)

# Per-stage statistics (min, max, mean, std, etc.)
staged.get_statistics()

# Detailed StageResult objects with advanced/eliminated indices
staged.get_stage_results()
```

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| All bids eliminated at a stage | A warning is issued. Subsequent stages are skipped. All bids get `NaN` for `final_score` and `ranking`. |
| Single stage, no filter | Behaves identically to the plain `Evaluator`. |
| Empty input DataFrame | Returns an empty DataFrame with all expected output columns. |
| Eliminated bids | Get `NaN` ranking and `NaN` for stages they never reached. |

## Running Tests

```bash
# Staged evaluator tests (29 tests)
pytest tests/test_staged.py -v

# All tests
pytest
```

## Example Script

```bash
python examples/example_staged.py
```

## Architecture

`StagedEvaluator` uses **composition**: it creates one `Evaluator` instance per stage internally. The existing `criteria.py` module is not modified. All stage-specific logic lives in `bid_evaluation/staged.py`.

```
StagedEvaluator
  |
  +-- StageDefinition("Technical")
  |     +-- Evaluator (with LinearCriterion, DirectScoreCriterion)
  |     +-- StageFilter(type='score_threshold', threshold=60)
  |
  +-- StageDefinition("Economic")
        +-- Evaluator (with MinimumRatioCriterion)
        +-- (no filter -- last stage)
```
