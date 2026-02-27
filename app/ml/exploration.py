"""
Exploration Policy - 80/20 exploit/explore decision logic.
Uses novelty and context uncertainty for exploration bonus.
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.structure_generator import GeneratedStructure


@dataclass
class ExplorationResult:
    """Result of exploration decision."""

    selected_structure: GeneratedStructure
    selected_score: Optional[Dict[str, float]]
    was_exploration: bool
    exploration_bonus: float
    novelty_score: float
    uncertainty_score: float


class ExplorationPolicy:
    """
    Implements 80/20 exploration policy.
    80% exploit (use ML scores), 20% explore (novelty bonus).
    """

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.exploration_ratio = self.settings.exploration_ratio  # 0.2

    def select_structure(
        self,
        candidates: List[GeneratedStructure],
        scores: Optional[List[Dict[str, float]]],
        context_hash: str,
    ) -> ExplorationResult:
        """
        Select best structure using 80/20 policy.

        Args:
            candidates: Generated candidate structures
            scores: ML scores for each candidate (None if no model)
            context_hash: Hash of current context for uncertainty lookup

        Returns:
            ExplorationResult with selected structure and metadata
        """
        if not candidates:
            raise ValueError("No candidates provided")

        # Decide explore vs exploit
        roll = random.random()
        is_exploration = roll < self.exploration_ratio

        if scores is None or is_exploration:
            # Exploration mode or no model available
            return self._explore_selection(candidates, context_hash)
        else:
            # Exploitation mode
            return self._exploit_selection(candidates, scores)

    def _exploit_selection(
        self,
        candidates: List[GeneratedStructure],
        scores: List[Dict[str, float]],
    ) -> ExplorationResult:
        """Select structure with highest ML score."""

        best_idx = 0
        best_score = scores[0]["final"]

        for i, score in enumerate(scores):
            if score["final"] > best_score:
                best_score = score["final"]
                best_idx = i

        return ExplorationResult(
            selected_structure=candidates[best_idx],
            selected_score=scores[best_idx],
            was_exploration=False,
            exploration_bonus=0.0,
            novelty_score=0.0,
            uncertainty_score=0.0,
        )

    def _explore_selection(
        self,
        candidates: List[GeneratedStructure],
        context_hash: str,
    ) -> ExplorationResult:
        """Select structure using exploration bonus."""

        # Get context uncertainty
        uncertainty = self._get_context_uncertainty(context_hash)

        # Calculate exploration scores for all candidates
        exploration_scores = []
        for candidate in candidates:
            novelty = self._get_structure_novelty(candidate.structure_hash)
            bonus = 0.2 * (novelty + uncertainty)
            exploration_scores.append(
                {
                    "bonus": bonus,
                    "novelty": novelty,
                }
            )

        # Select structure with highest exploration bonus
        # Add randomness to break ties and encourage diversity
        best_idx = 0
        best_bonus = exploration_scores[0]["bonus"]

        for i, exp_score in enumerate(exploration_scores):
            # Add small random factor for diversity
            adjusted_bonus = exp_score["bonus"] + random.uniform(0, 0.05)
            if adjusted_bonus > best_bonus:
                best_bonus = adjusted_bonus
                best_idx = i

        return ExplorationResult(
            selected_structure=candidates[best_idx],
            selected_score=None,
            was_exploration=True,
            exploration_bonus=exploration_scores[best_idx]["bonus"],
            novelty_score=exploration_scores[best_idx]["novelty"],
            uncertainty_score=uncertainty,
        )

    def _get_context_uncertainty(self, context_hash: str) -> float:
        """
        Get uncertainty score for context.
        Higher uncertainty = less data for this context.
        """
        query = text("""
            SELECT total_performances, std_attention, std_persuasion
            FROM context_stats
            WHERE context_hash = :context_hash
        """)

        result = self.db.execute(query, {"context_hash": context_hash})
        row = result.fetchone()

        if row is None or row.total_performances == 0:
            # No data = maximum uncertainty
            return 1.0

        # Uncertainty decreases with more data
        # Using inverse log scale
        import math

        data_factor = 1.0 / (1.0 + math.log1p(row.total_performances))

        # Add variance component if available
        variance_factor = 0.0
        if row.std_attention is not None and row.std_persuasion is not None:
            # Higher variance = higher uncertainty
            variance_factor = min(1.0, (row.std_attention + row.std_persuasion) / 2)

        # Combine factors
        uncertainty = 0.6 * data_factor + 0.4 * variance_factor
        return min(1.0, uncertainty)

    def _get_structure_novelty(self, structure_hash: str) -> float:
        """
        Get novelty score for structure.
        Higher novelty = less frequently used.
        """
        query = text("""
            SELECT total_uses
            FROM structure_stats
            WHERE structure_hash = :structure_hash
        """)

        result = self.db.execute(query, {"structure_hash": structure_hash})
        row = result.fetchone()

        if row is None or row.total_uses == 0:
            # Never used = maximum novelty
            return 1.0

        # Novelty decreases with usage
        # Using inverse log scale
        import math

        novelty = 1.0 / (1.0 + math.log1p(row.total_uses))
        return novelty


def create_exploration_policy(db: Session) -> ExplorationPolicy:
    """Factory function for exploration policy."""
    return ExplorationPolicy(db)
