"""Tests for bid_evaluation criteria module."""

import pytest
import pandas as pd
import numpy as np

from bid_evaluation import Evaluator, FormulaCriterion


@pytest.fixture
def sample_bids():
    """Sample bid data for testing."""
    return pd.DataFrame({
        'vendor': ['A', 'B', 'C'],
        'experience': [10, 5, 8],
        'methodology': [85, 90, 75],
        'bid_amount': [100000, 95000, 110000]
    })


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_linear_criterion(self, sample_bids):
        """Test linear criterion evaluation."""
        result = (Evaluator()
            .linear('experience', 1.0, higher_is_better=True)
            .evaluate(sample_bids))

        assert 'score_experience' in result.columns
        assert 'final_score' in result.columns
        assert result.loc[result['vendor'] == 'A', 'score_experience'].iloc[0] == 100.0

    def test_min_ratio_criterion(self, sample_bids):
        """Test min_ratio criterion evaluation."""
        result = (Evaluator()
            .min_ratio('bid_amount', 1.0)
            .evaluate(sample_bids))

        assert 'score_bid_amount' in result.columns
        # Vendor B has lowest bid, should get 100
        assert result.loc[result['vendor'] == 'B', 'score_bid_amount'].iloc[0] == 100.0

    def test_direct_criterion(self, sample_bids):
        """Test direct score criterion."""
        result = (Evaluator()
            .direct('methodology', 1.0)
            .evaluate(sample_bids))

        assert 'score_methodology' in result.columns

    def test_multiple_criteria(self, sample_bids):
        """Test evaluation with multiple criteria."""
        result = (Evaluator()
            .linear('experience', 0.3, higher_is_better=True)
            .direct('methodology', 0.3)
            .min_ratio('bid_amount', 0.4)
            .evaluate(sample_bids))

        assert 'ranking' in result.columns
        assert len(result) == 3

    def test_weight_normalization(self, sample_bids):
        """Test that weights are normalized when option is set."""
        evaluator = Evaluator(normalize_weights=True)
        evaluator.linear('experience', 2.0, higher_is_better=True)
        evaluator.direct('methodology', 2.0)

        result = evaluator.evaluate(sample_bids)
        assert result is not None


class TestFormulaCriterion:
    """Tests for the FormulaCriterion class."""

    def test_basic_formula(self, sample_bids):
        """Test a basic formula that passes through the value."""
        result = (Evaluator()
            .formula('experience', 1.0, formula='value * 10')
            .evaluate(sample_bids))

        assert 'score_experience' in result.columns
        assert 'final_score' in result.columns
        # experience values: 10, 5, 8 → 100, 50, 80
        assert result.loc[result['vendor'] == 'A', 'score_experience'].iloc[0] == 100.0
        assert result.loc[result['vendor'] == 'B', 'score_experience'].iloc[0] == 50.0
        assert result.loc[result['vendor'] == 'C', 'score_experience'].iloc[0] == 80.0

    def test_formula_with_statistics(self, sample_bids):
        """Test formula using statistics (min, max, mean)."""
        result = (Evaluator()
            .formula('experience', 1.0, formula='(value - min) / (max - min) * 100')
            .evaluate(sample_bids))

        # experience: 10, 5, 8 → min=5, max=10
        # A: (10-5)/(10-5)*100 = 100
        # B: (5-5)/(10-5)*100 = 0
        # C: (8-5)/(10-5)*100 = 60
        assert result.loc[result['vendor'] == 'A', 'score_experience'].iloc[0] == 100.0
        assert result.loc[result['vendor'] == 'B', 'score_experience'].iloc[0] == 0.0
        assert result.loc[result['vendor'] == 'C', 'score_experience'].iloc[0] == 60.0

    def test_formula_with_variables(self, sample_bids):
        """Test formula with custom variables."""
        result = (Evaluator()
            .formula('bid_amount', 1.0,
                     formula='100 - abs(value - target) / target * 100',
                     variables={'target': 100000})
            .evaluate(sample_bids))

        # bid_amount: 100000, 95000, 110000, target=100000
        # A: 100 - |100000-100000|/100000*100 = 100
        # B: 100 - |95000-100000|/100000*100 = 95
        # C: 100 - |110000-100000|/100000*100 = 90
        assert result.loc[result['vendor'] == 'A', 'score_bid_amount'].iloc[0] == 100.0
        assert result.loc[result['vendor'] == 'B', 'score_bid_amount'].iloc[0] == 95.0
        assert result.loc[result['vendor'] == 'C', 'score_bid_amount'].iloc[0] == 90.0

    def test_formula_clipping(self, sample_bids):
        """Test that formula results are clipped to 0-100."""
        result = (Evaluator()
            .formula('experience', 1.0, formula='value * 100')
            .evaluate(sample_bids))

        # experience: 10, 5, 8 → 1000, 500, 800 → clipped to 100
        scores = result['score_experience']
        assert (scores <= 100.0).all()
        assert (scores >= 0.0).all()

    def test_formula_negative_clipping(self, sample_bids):
        """Test that negative formula results are clipped to 0."""
        result = (Evaluator()
            .formula('experience', 1.0, formula='value - 20')
            .evaluate(sample_bids))

        # experience: 10, 5, 8 → -10, -15, -12 → clipped to 0
        scores = result['score_experience']
        assert (scores >= 0.0).all()

    def test_formula_error_handling(self, sample_bids):
        """Test that invalid formulas return 0.0."""
        result = (Evaluator()
            .formula('experience', 1.0, formula='1 / 0')
            .evaluate(sample_bids))

        # Division by zero → 0.0 for all
        scores = result['score_experience']
        assert (scores == 0.0).all()

    def test_formula_math_functions(self, sample_bids):
        """Test math functions in formula (sqrt, clip)."""
        result = (Evaluator()
            .formula('experience', 1.0, formula='clip(sqrt(value) * 30, 0, 100)')
            .evaluate(sample_bids))

        assert 'score_experience' in result.columns
        scores = result['score_experience']
        assert (scores >= 0.0).all()
        assert (scores <= 100.0).all()

    def test_formula_from_config(self, sample_bids):
        """Test FormulaCriterion via from_config."""
        config = {
            'experience': {
                'type': 'formula',
                'weight': 1.0,
                'formula': 'value * 10',
            }
        }
        result = Evaluator.from_config(config).evaluate(sample_bids)

        assert 'score_experience' in result.columns
        assert result.loc[result['vendor'] == 'A', 'score_experience'].iloc[0] == 100.0

    def test_formula_config_with_variables(self, sample_bids):
        """Test FormulaCriterion via from_config with variables."""
        config = {
            'bid_amount': {
                'type': 'formula',
                'weight': 1.0,
                'formula': '100 - abs(value - target) / target * 100',
                'variables': {'target': 100000},
            }
        }
        result = Evaluator.from_config(config).evaluate(sample_bids)

        assert result.loc[result['vendor'] == 'A', 'score_bid_amount'].iloc[0] == 100.0

    def test_formula_with_other_criteria(self, sample_bids):
        """Test formula criterion alongside other criterion types."""
        result = (Evaluator()
            .linear('experience', 0.3, higher_is_better=True)
            .formula('methodology', 0.3, formula='value')
            .min_ratio('bid_amount', 0.4)
            .evaluate(sample_bids))

        assert 'ranking' in result.columns
        assert len(result) == 3
        assert 'score_experience' in result.columns
        assert 'score_methodology' in result.columns
        assert 'score_bid_amount' in result.columns

    def test_formula_criterion_direct(self):
        """Test FormulaCriterion class directly."""
        criterion = FormulaCriterion('test', 1.0, formula='value * 2')
        values = pd.Series([10, 20, 30])
        scores = criterion.evaluate(values)
        # 10*2=20, 20*2=40, 30*2=60, all within 0-100
        assert list(scores) == [20.0, 40.0, 60.0]

    def test_formula_default_identity(self):
        """Test default formula ('value') acts as identity."""
        criterion = FormulaCriterion('test', 1.0)
        values = pd.Series([50, 75, 100])
        scores = criterion.evaluate(values)
        assert list(scores) == [50.0, 75.0, 100.0]
