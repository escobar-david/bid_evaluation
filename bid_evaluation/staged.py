# staged.py
"""Multi-stage bid evaluation with filtering between stages."""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import json
import yaml
import numpy as np
import pandas as pd

from .evaluator import Evaluator


@dataclass
class StageFilter:
    """Configuration for filtering bids between stages."""

    type: str  # 'score_threshold' or 'top_n'
    threshold: Optional[float] = None  # for score_threshold
    top_n: Optional[int] = None  # for top_n
    on_tie: str = "include"  # 'include' or 'exclude'

    def __post_init__(self):
        if self.type not in ("score_threshold", "top_n"):
            raise ValueError(
                f"Unknown filter type: {self.type}. Use 'score_threshold' or 'top_n'."
            )
        if self.type == "score_threshold" and self.threshold is None:
            raise ValueError("threshold is required for score_threshold filter.")
        if self.type == "top_n" and self.top_n is None:
            raise ValueError("top_n is required for top_n filter.")
        if self.on_tie not in ("include", "exclude"):
            raise ValueError(
                f"on_tie must be 'include' or 'exclude', got: {self.on_tie}"
            )


@dataclass
class StageDefinition:
    """Definition of a single evaluation stage."""

    name: str
    evaluator: Evaluator
    filter: Optional[StageFilter] = None
    weight: float = 1.0


@dataclass
class StageResult:
    """Results from a single stage evaluation."""

    name: str
    result_df: pd.DataFrame
    advanced_indices: pd.Index
    eliminated_indices: pd.Index


class StagedEvaluator:
    """Orchestrates multi-stage bid evaluation with filtering between stages.

    Bids are evaluated in sequential stages. After each stage (except the last),
    a filter can eliminate bids that don't meet the criteria. Only surviving bids
    advance to the next stage.
    """

    def __init__(self, final_score_mode: str = "last_stage"):
        """
        Args:
            final_score_mode: How to compute the final score.
                'last_stage' — use the last stage's score (default).
                'weighted_combination' — weighted average of all stage scores.
        """
        if final_score_mode not in ("last_stage", "weighted_combination"):
            raise ValueError(
                f"final_score_mode must be 'last_stage' or 'weighted_combination', got: {final_score_mode}"
            )
        self.final_score_mode = final_score_mode
        self._stages: List[StageDefinition] = []
        self._stage_results: List[StageResult] = []
        self._evaluated = False

    # === Factory methods ===

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StagedEvaluator":
        """Create a StagedEvaluator from a configuration dictionary.

        Args:
            config: Dictionary with 'stages' key containing a list of stage configs.

        Example:
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
                        'criteria': {'bid_amount': {'type': 'min_ratio', 'weight': 1.0}}
                    }
                ]
            }
        """
        final_score_mode = config.get("final_score_mode", "last_stage")
        staged = cls(final_score_mode=final_score_mode)

        for stage_cfg in config.get("stages", []):
            name = stage_cfg["name"]
            weight = stage_cfg.get("weight", 1.0)

            # Build filter
            stage_filter = None
            if "filter" in stage_cfg:
                f = stage_cfg["filter"]
                stage_filter = StageFilter(
                    type=f["type"],
                    threshold=f.get("threshold"),
                    top_n=f.get("top_n"),
                    on_tie=f.get("on_tie", "include"),
                )

            # Build evaluator from criteria config
            criteria_config = stage_cfg.get("criteria", {})
            evaluator = Evaluator.from_config(criteria_config)

            staged._stages.append(
                StageDefinition(
                    name=name,
                    evaluator=evaluator,
                    filter=stage_filter,
                    weight=weight,
                )
            )

        return staged

    @classmethod
    def from_yaml(cls, filepath: str) -> "StagedEvaluator":
        """Create a StagedEvaluator from a YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_config(data)

    @classmethod
    def from_json(cls, filepath: str) -> "StagedEvaluator":
        """Create a StagedEvaluator from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_config(data)

    # === Fluent interface ===

    def add_stage(
        self,
        name: str,
        filter_type: Optional[str] = None,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        on_tie: str = "include",
        weight: float = 1.0,
    ) -> "StagedEvaluator":
        """Add a new evaluation stage.

        After calling add_stage, subsequent criterion calls (linear, direct, etc.)
        add criteria to this stage's internal Evaluator.

        Args:
            name: Stage name (used in column prefixes).
            filter_type: 'score_threshold' or 'top_n', or None for no filter.
            threshold: Score threshold (required if filter_type='score_threshold').
            top_n: Number of top bids to advance (required if filter_type='top_n').
            on_tie: 'include' (advance all tied) or 'exclude' (strict cutoff).
            weight: Stage weight for weighted_combination mode.

        Returns:
            Self for method chaining.
        """
        stage_filter = None
        if filter_type is not None:
            stage_filter = StageFilter(
                type=filter_type,
                threshold=threshold,
                top_n=top_n,
                on_tie=on_tie,
            )

        self._stages.append(
            StageDefinition(
                name=name,
                evaluator=Evaluator(),
                filter=stage_filter,
                weight=weight,
            )
        )
        return self

    def _current_evaluator(self) -> Evaluator:
        """Get the Evaluator for the current (last added) stage."""
        if not self._stages:
            raise RuntimeError("No stages defined. Call add_stage() first.")
        return self._stages[-1].evaluator

    def linear(
        self,
        column: str,
        weight: float,
        name: str = None,
        higher_is_better: bool = True,
    ) -> "StagedEvaluator":
        """Add linear criterion to the current stage."""
        self._current_evaluator().linear(column, weight, name, higher_is_better)
        return self

    def threshold(
        self, column: str, weight: float, thresholds: list, name: str = None
    ) -> "StagedEvaluator":
        """Add threshold criterion to the current stage."""
        self._current_evaluator().threshold(column, weight, thresholds, name)
        return self

    def direct(
        self, column: str, weight: float, name: str = None, input_scale: float = 100
    ) -> "StagedEvaluator":
        """Add direct score criterion to the current stage."""
        self._current_evaluator().direct(column, weight, name, input_scale)
        return self

    def min_ratio(
        self, column: str, weight: float, name: str = None
    ) -> "StagedEvaluator":
        """Add minimum ratio criterion to the current stage."""
        self._current_evaluator().min_ratio(column, weight, name)
        return self

    def formula(
        self,
        column: str,
        weight: float,
        formula: str = "value",
        variables: dict = None,
        name: str = None,
    ) -> "StagedEvaluator":
        """Add formula criterion to the current stage."""
        self._current_evaluator().formula(column, weight, formula, variables, name)
        return self

    def custom(
        self,
        column: str,
        weight: float,
        func: Callable = None,
        name: str = None,
        **kwargs,
    ) -> "StagedEvaluator":
        """Add custom criterion to the current stage."""
        self._current_evaluator().custom(column, weight, func, name, **kwargs)
        return self

    # === Evaluation ===

    def evaluate(
        self, bids_df: pd.DataFrame, include_details: bool = True
    ) -> pd.DataFrame:
        """Evaluate bids through all stages sequentially.

        Args:
            bids_df: DataFrame with bid data.
            include_details: If True, includes per-criterion score columns.

        Returns:
            DataFrame with all original columns plus stage scores, elimination
            info, final_score, and ranking.
        """
        if not self._stages:
            raise RuntimeError("No stages defined. Add stages before evaluating.")

        if bids_df.empty:
            return self._empty_result(bids_df)

        result = bids_df.copy()
        result["eliminated_at_stage"] = None
        active_mask = pd.Series(True, index=result.index)
        self._stage_results = []

        for i, stage in enumerate(self._stages):
            is_last = i == len(self._stages) - 1
            safe_name = self._safe_name(stage.name)
            active_indices = result.index[active_mask]

            if active_indices.empty:
                # All bids eliminated — skip remaining stages
                warnings.warn(
                    f"All bids were eliminated before stage '{stage.name}'. "
                    f"Skipping this and subsequent stages."
                )
                self._stage_results.append(
                    StageResult(
                        name=stage.name,
                        result_df=pd.DataFrame(),
                        advanced_indices=pd.Index([]),
                        eliminated_indices=pd.Index([]),
                    )
                )
                continue

            # Evaluate active bids using this stage's evaluator
            active_bids = bids_df.loc[active_indices]
            stage_result = stage.evaluator.evaluate(
                active_bids, include_details=include_details
            )

            # Copy stage score columns into main result with stage prefix
            for col in stage_result.columns:
                if col.startswith("score_"):
                    criterion_name = col[len("score_") :]
                    prefixed = f"{safe_name}_{criterion_name}"
                    result.loc[active_indices, prefixed] = stage_result[col]
                elif col == "final_score":
                    result.loc[active_indices, f"{safe_name}_score"] = stage_result[col]
                elif col == "ranking":
                    result.loc[active_indices, f"{safe_name}_ranking"] = stage_result[
                        col
                    ]

            # Apply filter (except on the last stage)
            if not is_last and stage.filter is not None:
                stage_scores = result.loc[active_indices, f"{safe_name}_score"]
                advanced, eliminated = self._apply_filter(stage_scores, stage.filter)
                result.loc[eliminated, "eliminated_at_stage"] = stage.name
                active_mask.loc[eliminated] = False
            else:
                advanced = active_indices
                eliminated = pd.Index([])

            self._stage_results.append(
                StageResult(
                    name=stage.name,
                    result_df=stage_result,
                    advanced_indices=advanced,
                    eliminated_indices=eliminated,
                )
            )

        # Compute final_score
        result = self._compute_final_score(result)

        # Compute ranking — only for non-eliminated bids
        non_eliminated = result["eliminated_at_stage"].isna()
        result["ranking"] = np.nan
        if non_eliminated.any():
            result.loc[non_eliminated, "ranking"] = (
                result.loc[non_eliminated, "final_score"]
                .rank(ascending=False, method="min")
                .astype(int)
            )

        # Sort: ranked bids first (by ranking), then eliminated bids
        result = result.sort_values(
            by=["ranking", "final_score"],
            ascending=[True, False],
            na_position="last",
        )

        self._evaluated = True
        return result

    def _apply_filter(self, scores: pd.Series, stage_filter: StageFilter):
        """Apply a filter to determine which bids advance.

        Returns:
            (advanced_indices, eliminated_indices)
        """
        if stage_filter.type == "score_threshold":
            advanced = scores.index[scores >= stage_filter.threshold]
            eliminated = scores.index[scores < stage_filter.threshold]

        elif stage_filter.type == "top_n":
            n = stage_filter.top_n
            rankings = scores.rank(ascending=False, method="min")

            if stage_filter.on_tie == "include":
                # Include all bids with rank <= n
                advanced = scores.index[rankings <= n]
            else:
                # Exclude: only bids strictly above the cutoff
                # Find the score at rank n, only include bids with score strictly higher
                # than the score of the bid at rank n+1
                sorted_scores = scores.sort_values(ascending=False)
                if len(sorted_scores) <= n:
                    advanced = scores.index
                else:
                    cutoff_score = sorted_scores.iloc[n - 1]
                    # Count how many bids have score >= cutoff_score
                    at_or_above = (scores >= cutoff_score).sum()
                    if at_or_above > n:
                        # Tie at the cutoff — exclude those tied at cutoff
                        advanced = scores.index[scores > cutoff_score]
                    else:
                        advanced = scores.index[scores >= cutoff_score]

            eliminated = scores.index.difference(advanced)

        return advanced, eliminated

    def _compute_final_score(self, result: pd.DataFrame) -> pd.DataFrame:
        """Compute the final_score column based on the configured mode."""
        if self.final_score_mode == "last_stage":
            # Use the last stage's score for non-eliminated bids
            last_stage_name = self._safe_name(self._stages[-1].name)
            score_col = f"{last_stage_name}_score"
            if score_col in result.columns:
                result["final_score"] = result[score_col]
            else:
                result["final_score"] = np.nan

        elif self.final_score_mode == "weighted_combination":
            total_weight = sum(s.weight for s in self._stages)
            if total_weight == 0:
                result["final_score"] = np.nan
                return result

            result["final_score"] = 0.0
            for stage in self._stages:
                safe_name = self._safe_name(stage.name)
                score_col = f"{safe_name}_score"
                if score_col in result.columns:
                    normalized_weight = stage.weight / total_weight
                    result["final_score"] = result["final_score"] + (
                        result[score_col].fillna(0) * normalized_weight
                    )

            # Set NaN for eliminated bids that never got scores in all stages
            all_nan = True
            for stage in self._stages:
                safe_name = self._safe_name(stage.name)
                score_col = f"{safe_name}_score"
                if score_col in result.columns:
                    all_nan = False
                    break
            if all_nan:
                result["final_score"] = np.nan

        return result

    def _safe_name(self, name: str) -> str:
        """Convert a stage name to a safe column-name fragment."""
        return name.lower().replace(" ", "_").replace("-", "_")

    def _empty_result(self, bids_df: pd.DataFrame) -> pd.DataFrame:
        """Return an empty DataFrame with expected columns."""
        result = bids_df.copy()
        result["eliminated_at_stage"] = pd.Series(dtype="object")
        result["final_score"] = pd.Series(dtype="float64")
        result["ranking"] = pd.Series(dtype="float64")
        return result

    # === Informational methods ===

    def summary(self) -> pd.DataFrame:
        """Returns a DataFrame summarizing all stages and their criteria."""
        rows = []
        for stage in self._stages:
            safe_name = self._safe_name(stage.name)
            filter_desc = "None"
            if stage.filter is not None:
                if stage.filter.type == "score_threshold":
                    filter_desc = f"score >= {stage.filter.threshold}"
                elif stage.filter.type == "top_n":
                    filter_desc = (
                        f"top {stage.filter.top_n} (on_tie={stage.filter.on_tie})"
                    )

            for column, criterion in stage.evaluator.criteria.items():
                rows.append(
                    {
                        "stage": stage.name,
                        "stage_weight": stage.weight,
                        "filter": filter_desc,
                        "column": column,
                        "criterion_name": criterion.name,
                        "criterion_type": type(criterion).__name__,
                        "criterion_weight": criterion.weight,
                    }
                )

        return pd.DataFrame(rows)

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Returns per-stage statistics after evaluation."""
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before get_statistics().")
        stats = {}
        for stage in self._stages:
            stats[stage.name] = stage.evaluator.get_statistics()
        return stats

    def get_stage_results(self) -> List[StageResult]:
        """Returns the list of StageResult objects from the last evaluation."""
        if not self._evaluated:
            raise RuntimeError("Call evaluate() before get_stage_results().")
        return list(self._stage_results)
