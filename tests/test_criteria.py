"""Tests for bid_evaluation criteria module."""

import pytest
import pandas as pd
import numpy as np

from bid_evaluation import Evaluator


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

        assert 'experience_score' in result.columns
        assert 'final_score' in result.columns
        assert result.loc[result['vendor'] == 'A', 'experience_score'].iloc[0] == 100.0

    def test_min_ratio_criterion(self, sample_bids):
        """Test min_ratio criterion evaluation."""
        result = (Evaluator()
            .min_ratio('bid_amount', 1.0)
            .evaluate(sample_bids))

        assert 'bid_amount_score' in result.columns
        # Vendor B has lowest bid, should get 100
        assert result.loc[result['vendor'] == 'B', 'bid_amount_score'].iloc[0] == 100.0

    def test_direct_criterion(self, sample_bids):
        """Test direct score criterion."""
        result = (Evaluator()
            .direct('methodology', 1.0)
            .evaluate(sample_bids))

        assert 'methodology_score' in result.columns

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
