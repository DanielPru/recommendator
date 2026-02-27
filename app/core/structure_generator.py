"""
Structure Generator - Weighted sampling for candidate generation.
Generates 50-100 unique visual structure candidates.
"""

import random
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass

from app.core.feature_schema import (
    FEATURE_SCHEMA_V1,
    FEATURE_NAMES,
    get_allowed_values,
    compute_structure_hash,
)
from app.config import get_settings


@dataclass
class GeneratedStructure:
    """Container for a generated structure with metadata."""

    features: Dict[str, str]
    structure_hash: str
    is_diverse: bool  # True if generated via uniform sampling


class StructureGenerator:
    """
    Generates candidate visual structures using weighted sampling.
    Ensures uniqueness and diversity requirements.
    """

    def __init__(
        self,
        min_candidates: int = 50,
        max_candidates: int = 100,
        diversity_ratio: float = 0.2,
    ):
        """
        Initialize generator with configuration.

        Args:
            min_candidates: Minimum structures to generate
            max_candidates: Maximum structures to generate
            diversity_ratio: Fraction using uniform sampling (0.2 = 20%)
        """
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.diversity_ratio = diversity_ratio

    def generate(
        self,
        weights: Dict[str, Dict[str, float]],
        content_type: str,
        target_count: Optional[int] = None,
    ) -> List[GeneratedStructure]:
        """
        Generate unique candidate structures.

        Args:
            weights: Feature weight multipliers from context interpreter
            content_type: "image" or "video"
            target_count: Optional specific count (uses random in range if None)

        Returns:
            List of GeneratedStructure objects
        """
        if target_count is None:
            target_count = random.randint(self.min_candidates, self.max_candidates)

        # Calculate diversity vs heuristic counts
        diverse_count = int(target_count * self.diversity_ratio)
        heuristic_count = target_count - diverse_count

        # Get allowed values per feature based on content type
        allowed_values = self._get_allowed_values_map(content_type)

        # Normalize weights for allowed values only
        normalized_weights = self._normalize_weights(weights, allowed_values)

        # Generate structures
        seen_hashes: Set[str] = set()
        structures: List[GeneratedStructure] = []

        # Phase 1: Generate heuristic-biased structures (80%)
        heuristic_structures = self._generate_weighted(
            normalized_weights,
            allowed_values,
            heuristic_count,
            seen_hashes,
            is_diverse=False,
        )
        structures.extend(heuristic_structures)

        # Phase 2: Generate diverse structures with uniform sampling (20%)
        uniform_weights = self._create_uniform_weights(allowed_values)
        diverse_structures = self._generate_weighted(
            uniform_weights,
            allowed_values,
            diverse_count,
            seen_hashes,
            is_diverse=True,
        )
        structures.extend(diverse_structures)

        # Shuffle to avoid ordering bias
        random.shuffle(structures)

        return structures

    def _get_allowed_values_map(self, content_type: str) -> Dict[str, Tuple[str, ...]]:
        """Get allowed values for each feature based on content type."""
        return {
            feature: get_allowed_values(feature, content_type)
            for feature in FEATURE_NAMES
        }

    def _normalize_weights(
        self,
        weights: Dict[str, Dict[str, float]],
        allowed_values: Dict[str, Tuple[str, ...]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Normalize weights to probability distributions.
        Only includes allowed values per feature.
        """
        normalized = {}

        for feature, allowed in allowed_values.items():
            feature_weights = {}
            total = 0.0

            for value in allowed:
                w = weights.get(feature, {}).get(value, 1.0)
                feature_weights[value] = w
                total += w

            # Normalize to sum to 1.0
            if total > 0:
                normalized[feature] = {v: w / total for v, w in feature_weights.items()}
            else:
                # Fallback to uniform if all weights are zero
                uniform_w = 1.0 / len(allowed)
                normalized[feature] = {v: uniform_w for v in allowed}

        return normalized

    def _create_uniform_weights(
        self,
        allowed_values: Dict[str, Tuple[str, ...]],
    ) -> Dict[str, Dict[str, float]]:
        """Create uniform weight distribution for diversity sampling."""
        return {
            feature: {v: 1.0 / len(values) for v in values}
            for feature, values in allowed_values.items()
        }

    def _generate_weighted(
        self,
        weights: Dict[str, Dict[str, float]],
        allowed_values: Dict[str, Tuple[str, ...]],
        count: int,
        seen_hashes: Set[str],
        is_diverse: bool,
        max_attempts_multiplier: int = 10,
    ) -> List[GeneratedStructure]:
        """
        Generate structures using weighted sampling.

        Args:
            weights: Normalized probability distributions
            allowed_values: Allowed values per feature
            count: Number of structures to generate
            seen_hashes: Set of already-generated hashes (modified in place)
            is_diverse: Flag to mark diversity-sampled structures
            max_attempts_multiplier: Max attempts = count * multiplier

        Returns:
            List of unique GeneratedStructure objects
        """
        structures = []
        attempts = 0
        max_attempts = count * max_attempts_multiplier

        while len(structures) < count and attempts < max_attempts:
            attempts += 1

            # Sample each feature independently
            features = {}
            for feature in sorted(FEATURE_NAMES):
                values = list(weights[feature].keys())
                probs = [weights[feature][v] for v in values]

                # Weighted random choice
                features[feature] = random.choices(values, weights=probs, k=1)[0]

            # Check uniqueness
            structure_hash = compute_structure_hash(features)
            if structure_hash not in seen_hashes:
                seen_hashes.add(structure_hash)
                structures.append(
                    GeneratedStructure(
                        features=features,
                        structure_hash=structure_hash,
                        is_diverse=is_diverse,
                    )
                )

        return structures

    def get_feature_probabilities(
        self,
        weights: Dict[str, Dict[str, float]],
        content_type: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the actual sampling probabilities for debugging/inspection.
        """
        allowed_values = self._get_allowed_values_map(content_type)
        return self._normalize_weights(weights, allowed_values)


# Factory function
def create_structure_generator(
    min_candidates: int = 50,
    max_candidates: int = 100,
    diversity_ratio: float = 0.2,
) -> StructureGenerator:
    """Create generator with optional config override."""
    try:
        settings = get_settings()
        return StructureGenerator(
            min_candidates=settings.min_candidates,
            max_candidates=settings.max_candidates,
            diversity_ratio=settings.diversity_ratio,
        )
    except Exception:
        # Fallback to defaults if settings not available
        return StructureGenerator(
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            diversity_ratio=diversity_ratio,
        )


# Singleton instance
_generator_instance: Optional[StructureGenerator] = None


def get_structure_generator() -> StructureGenerator:
    """Get singleton generator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = create_structure_generator()
    return _generator_instance


def reset_generator() -> None:
    """Reset singleton (for testing)."""
    global _generator_instance
    _generator_instance = None
