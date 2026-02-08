"""Tests for multi-stage bid evaluation."""

import json
import warnings
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import yaml

from bid_evaluation import StagedEvaluator
from bid_evaluation.staged import StageFilter, StageResult


@pytest.fixture
def bids_5():
    """Five-vendor bid dataset with technical and economic columns."""
    return pd.DataFrame({
        'vendor': ['A', 'B', 'C', 'D', 'E'],
        'experience': [10, 5, 8, 3, 7],
        'quality_score': [80, 60, 90, 50, 70],
        'bid_amount': [100000, 95000, 110000, 90000, 105000],
        'delivery_days': [30, 45, 25, 60, 35],
    })


@pytest.fixture
def bids_3():
    """Three-vendor bid dataset."""
    return pd.DataFrame({
        'vendor': ['A', 'B', 'C'],
        'experience': [10, 5, 8],
        'methodology': [85, 90, 75],
        'bid_amount': [100000, 95000, 110000],
    })


class TestStagedEvaluatorFluent:
    """Tests for the fluent API."""

    def test_two_stage_threshold(self, bids_5):
        """Two stages with score_threshold filter — low scorers eliminated."""
        result = (StagedEvaluator()
            .add_stage('Technical', filter_type='score_threshold', threshold=60)
                .linear('experience', 0.4, higher_is_better=True)
                .direct('quality_score', 0.6)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        # All vendors should have technical stage scores
        assert 'technical_score' in result.columns
        assert 'economic_score' in result.columns
        assert 'eliminated_at_stage' in result.columns
        assert 'final_score' in result.columns
        assert 'ranking' in result.columns

        # Eliminated vendors should have NaN ranking
        eliminated = result[result['eliminated_at_stage'].notna()]
        assert eliminated['ranking'].isna().all()

        # Non-eliminated should have valid rankings
        surviving = result[result['eliminated_at_stage'].isna()]
        assert surviving['ranking'].notna().all()
        assert (surviving['ranking'] >= 1).all()

    def test_two_stage_top_n(self, bids_5):
        """Two stages with top_n filter — only top 3 advance."""
        result = (StagedEvaluator()
            .add_stage('Technical', filter_type='top_n', top_n=3)
                .direct('quality_score', 1.0)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        surviving = result[result['eliminated_at_stage'].isna()]
        eliminated = result[result['eliminated_at_stage'].notna()]

        # Top 3 by quality_score: C(90), A(80), E(70) survive; B(60), D(50) eliminated
        assert len(surviving) == 3
        assert len(eliminated) == 2
        assert set(eliminated['vendor']) == {'B', 'D'}

    def test_three_stages(self, bids_5):
        """Three stages with progressive filtering."""
        result = (StagedEvaluator()
            .add_stage('Screening', filter_type='score_threshold', threshold=50)
                .direct('quality_score', 1.0)
            .add_stage('Technical', filter_type='top_n', top_n=3)
                .linear('experience', 1.0, higher_is_better=True)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        assert 'screening_score' in result.columns
        assert 'technical_score' in result.columns
        assert 'economic_score' in result.columns

        surviving = result[result['eliminated_at_stage'].isna()]
        assert len(surviving) <= 3

    def test_weighted_combination(self, bids_5):
        """Weighted combination mode uses all stage scores."""
        result = (StagedEvaluator(final_score_mode='weighted_combination')
            .add_stage('Technical', filter_type='score_threshold', threshold=50, weight=0.6)
                .direct('quality_score', 1.0)
            .add_stage('Economic', weight=0.4)
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        surviving = result[result['eliminated_at_stage'].isna()]
        # Final score should be a blend of technical and economic scores
        for idx in surviving.index:
            tech = result.loc[idx, 'technical_score']
            econ = result.loc[idx, 'economic_score']
            expected = tech * 0.6 + econ * 0.4
            assert abs(result.loc[idx, 'final_score'] - expected) < 1e-9

    def test_single_stage_equivalence(self, bids_3):
        """Single stage should produce same ranking as plain Evaluator."""
        from bid_evaluation import Evaluator

        staged_result = (StagedEvaluator()
            .add_stage('Only')
                .linear('experience', 0.3, higher_is_better=True)
                .direct('methodology', 0.3)
                .min_ratio('bid_amount', 0.4)
            .evaluate(bids_3))

        plain_result = (Evaluator()
            .linear('experience', 0.3, higher_is_better=True)
            .direct('methodology', 0.3)
            .min_ratio('bid_amount', 0.4)
            .evaluate(bids_3))

        # Rankings should be identical
        staged_ranking = staged_result.set_index('vendor')['ranking'].sort_index()
        plain_ranking = plain_result.set_index('vendor')['ranking'].sort_index()
        pd.testing.assert_series_equal(
            staged_ranking.astype(int),
            plain_ranking.astype(int),
            check_names=False,
        )

    def test_no_filter_all_advance(self, bids_5):
        """Stage without a filter should advance all bids."""
        result = (StagedEvaluator()
            .add_stage('Technical')  # no filter
                .direct('quality_score', 1.0)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        assert result['eliminated_at_stage'].isna().all()
        assert result['ranking'].notna().all()


class TestStagedEvaluatorConfig:
    """Tests for config-based creation."""

    def test_from_config(self, bids_5):
        """Create StagedEvaluator from config dict."""
        config = {
            'stages': [
                {
                    'name': 'Technical',
                    'filter': {'type': 'score_threshold', 'threshold': 60},
                    'criteria': {
                        'experience': {'type': 'linear', 'weight': 0.4, 'higher_is_better': True},
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
        result = StagedEvaluator.from_config(config).evaluate(bids_5)
        assert 'technical_score' in result.columns
        assert 'economic_score' in result.columns

    def test_from_yaml(self, bids_5, tmp_path):
        """Create StagedEvaluator from YAML file."""
        config = {
            'stages': [
                {
                    'name': 'Technical',
                    'filter': {'type': 'top_n', 'top_n': 3},
                    'criteria': {
                        'quality_score': {'type': 'direct', 'weight': 1.0},
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
        yaml_path = tmp_path / 'staged.yaml'
        yaml_path.write_text(yaml.dump(config))

        result = StagedEvaluator.from_yaml(str(yaml_path)).evaluate(bids_5)
        surviving = result[result['eliminated_at_stage'].isna()]
        assert len(surviving) == 3

    def test_from_json(self, bids_5, tmp_path):
        """Create StagedEvaluator from JSON file."""
        config = {
            'stages': [
                {
                    'name': 'Technical',
                    'filter': {'type': 'score_threshold', 'threshold': 70},
                    'criteria': {
                        'quality_score': {'type': 'direct', 'weight': 1.0},
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
        json_path = tmp_path / 'staged.json'
        json_path.write_text(json.dumps(config))

        result = StagedEvaluator.from_json(str(json_path)).evaluate(bids_5)
        assert 'eliminated_at_stage' in result.columns

    def test_config_fluent_equivalence(self, bids_5):
        """Config-based and fluent-based should produce identical results."""
        config = {
            'stages': [
                {
                    'name': 'Technical',
                    'filter': {'type': 'score_threshold', 'threshold': 60},
                    'criteria': {
                        'quality_score': {'type': 'direct', 'weight': 1.0},
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
        config_result = StagedEvaluator.from_config(config).evaluate(bids_5)

        fluent_result = (StagedEvaluator()
            .add_stage('Technical', filter_type='score_threshold', threshold=60)
                .direct('quality_score', 1.0)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        # Same final rankings
        config_ranks = config_result.set_index('vendor')['ranking'].sort_index()
        fluent_ranks = fluent_result.set_index('vendor')['ranking'].sort_index()
        pd.testing.assert_series_equal(config_ranks, fluent_ranks, check_names=False)

    def test_weighted_combination_config(self, bids_5):
        """Weighted combination via config."""
        config = {
            'final_score_mode': 'weighted_combination',
            'stages': [
                {
                    'name': 'Technical',
                    'weight': 0.7,
                    'filter': {'type': 'score_threshold', 'threshold': 50},
                    'criteria': {
                        'quality_score': {'type': 'direct', 'weight': 1.0},
                    }
                },
                {
                    'name': 'Economic',
                    'weight': 0.3,
                    'criteria': {
                        'bid_amount': {'type': 'min_ratio', 'weight': 1.0},
                    }
                }
            ]
        }
        result = StagedEvaluator.from_config(config).evaluate(bids_5)
        assert result['final_score'].notna().any()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_eliminated(self, bids_5):
        """When all bids are eliminated, warn and set NaN scores."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = (StagedEvaluator()
                .add_stage('Technical', filter_type='score_threshold', threshold=99999)
                    .direct('quality_score', 1.0)
                .add_stage('Economic')
                    .min_ratio('bid_amount', 1.0)
                .evaluate(bids_5))

            # A warning should have been issued
            assert any("All bids were eliminated" in str(warning.message) for warning in w)

        assert result['eliminated_at_stage'].notna().all()
        assert result['ranking'].isna().all()

    def test_ties_include(self):
        """top_n with ties and on_tie='include' advances all tied bids."""
        bids = pd.DataFrame({
            'vendor': ['A', 'B', 'C', 'D'],
            'score': [90, 80, 80, 70],
            'price': [100, 200, 150, 120],
        })
        result = (StagedEvaluator()
            .add_stage('Round1', filter_type='top_n', top_n=2, on_tie='include')
                .direct('score', 1.0)
            .add_stage('Round2')
                .min_ratio('price', 1.0)
            .evaluate(bids))

        surviving = result[result['eliminated_at_stage'].isna()]
        # A(90), B(80), C(80) should all advance (tie at rank 2)
        assert len(surviving) == 3
        assert set(surviving['vendor']) == {'A', 'B', 'C'}

    def test_ties_exclude(self):
        """top_n with ties and on_tie='exclude' excludes tied bids at cutoff."""
        bids = pd.DataFrame({
            'vendor': ['A', 'B', 'C', 'D'],
            'score': [90, 80, 80, 70],
            'price': [100, 200, 150, 120],
        })
        result = (StagedEvaluator()
            .add_stage('Round1', filter_type='top_n', top_n=2, on_tie='exclude')
                .direct('score', 1.0)
            .add_stage('Round2')
                .min_ratio('price', 1.0)
            .evaluate(bids))

        surviving = result[result['eliminated_at_stage'].isna()]
        # Only A(90) is strictly above the tie at rank 2
        assert len(surviving) == 1
        assert surviving['vendor'].iloc[0] == 'A'

    def test_empty_dataframe(self, bids_5):
        """Empty DataFrame returns empty result with expected columns."""
        empty = bids_5.iloc[0:0]
        result = (StagedEvaluator()
            .add_stage('Technical', filter_type='score_threshold', threshold=60)
                .direct('quality_score', 1.0)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(empty))

        assert len(result) == 0
        assert 'eliminated_at_stage' in result.columns
        assert 'final_score' in result.columns
        assert 'ranking' in result.columns

    def test_eliminated_have_nan_ranking(self, bids_5):
        """Eliminated bids must have NaN ranking."""
        result = (StagedEvaluator()
            .add_stage('Technical', filter_type='top_n', top_n=2)
                .direct('quality_score', 1.0)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        eliminated = result[result['eliminated_at_stage'].notna()]
        assert eliminated['ranking'].isna().all()

    def test_eliminated_have_nan_later_stage_scores(self, bids_5):
        """Bids eliminated in stage 1 should have NaN in stage 2 scores."""
        result = (StagedEvaluator()
            .add_stage('Technical', filter_type='top_n', top_n=2)
                .direct('quality_score', 1.0)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        eliminated = result[result['eliminated_at_stage'].notna()]
        assert eliminated['economic_score'].isna().all()

    def test_output_columns_present(self, bids_5):
        """All expected output columns should be present."""
        result = (StagedEvaluator()
            .add_stage('Technical', filter_type='score_threshold', threshold=60)
                .linear('experience', 0.4, higher_is_better=True)
                .direct('quality_score', 0.6)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        expected_cols = [
            'technical_score', 'technical_ranking',
            'economic_score', 'economic_ranking',
            'eliminated_at_stage', 'final_score', 'ranking',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_detail_columns_with_prefix(self, bids_5):
        """Per-criterion detail columns should have stage prefix."""
        result = (StagedEvaluator()
            .add_stage('Technical')
                .linear('experience', 0.4, higher_is_better=True)
                .direct('quality_score', 0.6)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5, include_details=True))

        assert 'technical_experience' in result.columns
        assert 'technical_quality_score' in result.columns
        assert 'economic_bid_amount' in result.columns

    def test_no_details(self, bids_5):
        """include_details=False should omit per-criterion columns."""
        result = (StagedEvaluator()
            .add_stage('Technical')
                .linear('experience', 0.4, higher_is_better=True)
                .direct('quality_score', 0.6)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5, include_details=False))

        assert 'technical_experience' not in result.columns
        assert 'technical_score' in result.columns

    def test_no_stages_raises(self, bids_5):
        """Evaluating without any stages should raise."""
        with pytest.raises(RuntimeError, match="No stages defined"):
            StagedEvaluator().evaluate(bids_5)

    def test_criterion_before_stage_raises(self):
        """Adding a criterion before any stage should raise."""
        with pytest.raises(RuntimeError, match="No stages defined"):
            StagedEvaluator().linear('x', 1.0)

    def test_invalid_filter_type(self):
        """Invalid filter type should raise."""
        with pytest.raises(ValueError, match="Unknown filter type"):
            StageFilter(type='invalid')

    def test_invalid_final_score_mode(self):
        """Invalid final_score_mode should raise."""
        with pytest.raises(ValueError, match="final_score_mode"):
            StagedEvaluator(final_score_mode='invalid')


class TestInformational:
    """Tests for summary(), get_statistics(), get_stage_results()."""

    def test_summary(self, bids_5):
        """summary() should list all stages and criteria."""
        staged = (StagedEvaluator()
            .add_stage('Technical', filter_type='score_threshold', threshold=60)
                .linear('experience', 0.4, higher_is_better=True)
                .direct('quality_score', 0.6)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0))

        s = staged.summary()
        assert len(s) == 3  # 2 technical criteria + 1 economic
        assert set(s['stage']) == {'Technical', 'Economic'}
        assert 'filter' in s.columns
        assert 'criterion_type' in s.columns

    def test_get_statistics_after_evaluate(self, bids_5):
        """get_statistics() should return stats per stage after evaluation."""
        staged = (StagedEvaluator()
            .add_stage('Technical')
                .direct('quality_score', 1.0)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0))
        staged.evaluate(bids_5)

        stats = staged.get_statistics()
        assert 'Technical' in stats
        assert 'Economic' in stats

    def test_get_statistics_before_evaluate_raises(self):
        """get_statistics() before evaluate() should raise."""
        staged = (StagedEvaluator()
            .add_stage('Technical')
                .direct('quality_score', 1.0))
        with pytest.raises(RuntimeError, match="Call evaluate"):
            staged.get_statistics()

    def test_get_stage_results(self, bids_5):
        """get_stage_results() should return StageResult objects."""
        staged = (StagedEvaluator()
            .add_stage('Technical', filter_type='top_n', top_n=3)
                .direct('quality_score', 1.0)
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0))
        staged.evaluate(bids_5)

        results = staged.get_stage_results()
        assert len(results) == 2
        assert isinstance(results[0], StageResult)
        assert results[0].name == 'Technical'
        assert len(results[0].advanced_indices) == 3
        assert len(results[0].eliminated_indices) == 2

    def test_get_stage_results_before_evaluate_raises(self):
        """get_stage_results() before evaluate() should raise."""
        staged = (StagedEvaluator()
            .add_stage('Technical')
                .direct('quality_score', 1.0))
        with pytest.raises(RuntimeError, match="Call evaluate"):
            staged.get_stage_results()


class TestFormulaCriterionInStages:
    """Tests for formula criterion within staged evaluation."""

    def test_formula_in_stage_fluent(self, bids_5):
        """Formula criterion in a stage via fluent interface."""
        result = (StagedEvaluator()
            .add_stage('Technical', filter_type='score_threshold', threshold=50)
                .formula('quality_score', 1.0, formula='value')
            .add_stage('Economic')
                .min_ratio('bid_amount', 1.0)
            .evaluate(bids_5))

        assert 'technical_score' in result.columns
        assert 'economic_score' in result.columns
        surviving = result[result['eliminated_at_stage'].isna()]
        assert len(surviving) > 0

    def test_formula_with_variables_in_stage(self, bids_5):
        """Formula with custom variables in staged evaluation."""
        result = (StagedEvaluator()
            .add_stage('Economic')
                .formula('bid_amount', 1.0,
                         formula='100 - abs(value - target) / target * 100',
                         variables={'target': 100000})
            .evaluate(bids_5))

        assert 'economic_score' in result.columns
        assert result['final_score'].notna().all()

    def test_formula_via_config_in_stage(self, bids_5):
        """Formula criterion via config in staged evaluation."""
        config = {
            'stages': [
                {
                    'name': 'Scoring',
                    'criteria': {
                        'quality_score': {
                            'type': 'formula',
                            'weight': 1.0,
                            'formula': 'value',
                        }
                    }
                }
            ]
        }
        result = StagedEvaluator.from_config(config).evaluate(bids_5)
        assert 'scoring_score' in result.columns
        assert result['ranking'].notna().all()
