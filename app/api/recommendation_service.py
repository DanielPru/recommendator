"""
Recommendation service - Core business logic for structure recommendation.
Uses asset_id as the primary lifecycle identifier.
"""

import hashlib
from typing import Dict, Optional
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.context_interpreter import get_context_interpreter
from app.core.structure_generator import get_structure_generator
from app.core.feature_schema import compute_structure_hash
from app.ml.model_manager import get_model_manager
from app.ml.exploration import ExplorationPolicy
from app.db.models import DecisionLog, StructureStats


class RecommendationService:
    """Service for generating structure recommendations."""

    def __init__(self, db: Session):
        self.db = db
        self.interpreter = get_context_interpreter()
        self.generator = get_structure_generator()
        self.model_manager = get_model_manager()
        self.exploration_policy = ExplorationPolicy(db)

    def check_asset_exists(self, asset_id: str) -> bool:
        """Check if asset_id already exists in decision_logs."""
        existing = (
            self.db.query(DecisionLog).filter(DecisionLog.asset_id == asset_id).first()
        )
        return existing is not None

    def recommend(
        self,
        asset_id: str,
        segment_strategy: str,
        channel: str,
        traffic_type: str,
        funnel_stage: str,
        content_type: str,
    ) -> Dict:
        """
        Generate a structure recommendation for an asset.

        Args:
            asset_id: Unique asset identifier (primary key)
            segment_strategy: Strategic text describing target segment
            channel: Distribution channel
            traffic_type: 'organic' or 'paid'
            funnel_stage: 'TOFU', 'MOFU', or 'BOFU'
            content_type: 'image' or 'video'

        Returns:
            Dictionary with recommendation result and metadata
        """
        # 1. Compute context hash
        context_hash = self._compute_context_hash(
            segment_strategy, channel, traffic_type, funnel_stage, content_type
        )

        # 2. Interpret context to get weight multipliers
        weights = self.interpreter.interpret(
            segment_strategy=segment_strategy,
            channel=channel,
            traffic_type=traffic_type,
            funnel_stage=funnel_stage,
            content_type=content_type,
        )

        # 3. Generate candidate structures
        candidates = self.generator.generate(
            weights=weights,
            content_type=content_type,
        )

        # 4. Score candidates with ML (if models available)
        scores = None
        if self.model_manager.is_ready:
            structures = [c.features for c in candidates]
            scores = self.model_manager.score_structures_batch(structures, traffic_type)

        # 5. Apply exploration policy
        result = self.exploration_policy.select_structure(
            candidates=candidates,
            scores=scores,
            context_hash=context_hash,
        )

        # 6. Log decision with asset_id as primary key
        decision_log = self._log_decision(
            asset_id=asset_id,
            segment_strategy=segment_strategy,
            channel=channel,
            traffic_type=traffic_type,
            funnel_stage=funnel_stage,
            content_type=content_type,
            context_hash=context_hash,
            structure=result.selected_structure,
            score=result.selected_score,
            was_exploration=result.was_exploration,
            exploration_bonus=result.exploration_bonus,
            uncertainty_score=result.uncertainty_score,
            candidates_count=len(candidates),
        )

        # 7. Update structure stats
        self._update_structure_stats(result.selected_structure)

        # Prepare response
        response = {
            "asset_id": asset_id,
            "structure_hash": result.selected_structure.structure_hash,
            "structure": result.selected_structure.features,
            "p_attention": None,
            "p_persuasion": None,
            "p_final": None,
            "mode": "exploration" if result.was_exploration else "exploitation",
            "context_confidence": 1.0 - result.uncertainty_score,
            "exploration_weight": result.exploration_bonus,
            "model_version": self.model_manager.current_version,
            "candidates_evaluated": len(candidates),
            "context_hash": context_hash,
        }

        # Add ML scores if available
        if result.selected_score:
            response["p_attention"] = result.selected_score.get("attention")
            response["p_persuasion"] = result.selected_score.get("persuasion")
            response["p_final"] = result.selected_score.get("final")

        return response

    def _compute_context_hash(
        self,
        segment_strategy: str,
        channel: str,
        traffic_type: str,
        funnel_stage: str,
        content_type: str,
    ) -> str:
        """Compute hash for context grouping."""
        # Use a simplified version of segment for grouping
        # (full segment too unique for useful aggregation)
        segment_keywords = self._extract_keywords(segment_strategy)

        hash_input = "|".join(
            [
                segment_keywords,
                channel.lower(),
                traffic_type.lower(),
                funnel_stage.upper(),
                content_type.lower(),
            ]
        )

        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> str:
        """Extract and sort significant keywords from text."""
        # Simple keyword extraction without NLP
        keywords = set()
        text_lower = text.lower()

        # Important marketing/visual keywords to look for
        important_terms = [
            "short-form",
            "authentic",
            "bold",
            "neon",
            "educational",
            "ugc",
            "product-focused",
            "lifestyle",
            "minimalist",
            "dynamic",
            "professional",
            "corporate",
            "b2b",
            "energy",
            "vibrant",
        ]

        for term in important_terms:
            if term in text_lower:
                keywords.add(term)

        # Sort and limit
        sorted_keywords = sorted(keywords)[:max_keywords]
        return ",".join(sorted_keywords) if sorted_keywords else "generic"

    def _log_decision(
        self,
        asset_id: str,
        segment_strategy: str,
        channel: str,
        traffic_type: str,
        funnel_stage: str,
        content_type: str,
        context_hash: str,
        structure,
        score: Optional[Dict],
        was_exploration: bool,
        exploration_bonus: float,
        uncertainty_score: float,
        candidates_count: int,
    ) -> DecisionLog:
        """Log the decision to database with asset_id as primary key."""

        # Extract scores if available
        p_attention = score["attention"] if score else None
        p_persuasion = score["persuasion"] if score else None
        p_final = score["final"] if score else None

        decision = DecisionLog(
            asset_id=asset_id,
            segment_strategy=segment_strategy,
            channel=channel,
            traffic_type=traffic_type,
            funnel_stage=funnel_stage,
            content_type=content_type,
            context_hash=context_hash,
            structure_hash=structure.structure_hash,
            structure_features=structure.features,
            p_attention=p_attention,
            p_persuasion=p_persuasion,
            p_final=p_final,
            mode="exploration" if was_exploration else "exploitation",
            context_confidence=1.0 - uncertainty_score,
            exploration_weight=exploration_bonus,
            model_version=self.model_manager.current_version,
            candidates_count=candidates_count,
        )

        self.db.add(decision)
        self.db.commit()
        self.db.refresh(decision)

        return decision

    def _update_structure_stats(self, structure) -> None:
        """Update or create structure statistics."""

        query = text("""
            INSERT INTO structure_stats (structure_hash, structure_features, total_uses, first_seen, last_updated)
            VALUES (:hash, :features, 1, NOW(), NOW())
            ON CONFLICT (structure_hash) DO UPDATE SET
                total_uses = structure_stats.total_uses + 1,
                last_updated = NOW()
        """)

        import json

        self.db.execute(
            query,
            {
                "hash": structure.structure_hash,
                "features": json.dumps(structure.features),
            },
        )
        self.db.commit()
