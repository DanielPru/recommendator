"""
Context Interpreter - Deterministic heuristic engine.
Generates weight multipliers based on context inputs.
NO LLM usage - pure rule-based logic.
"""

from typing import Dict, Any
from copy import deepcopy

from app.core.feature_schema import FEATURE_SCHEMA_V1, FEATURE_NAMES


class ContextInterpreter:
    """
    Interprets request context to generate feature weight multipliers.
    All heuristics are deterministic and rule-based.
    """

    def __init__(self):
        """Initialize with default weights."""
        self._base_weights = self._create_base_weights()

    def _create_base_weights(self) -> Dict[str, Dict[str, float]]:
        """Create base weight structure with all values at 1.0."""
        return {
            feature: {value: 1.0 for value in values}
            for feature, values in FEATURE_SCHEMA_V1.items()
        }

    def interpret(
        self,
        segment_strategy: str,
        channel: str,
        traffic_type: str,
        funnel_stage: str,
        content_type: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate weight multipliers based on context.

        Returns:
            Dictionary mapping feature -> value -> weight_multiplier
        """
        weights = deepcopy(self._base_weights)

        # Apply heuristics in order
        self._apply_content_type_heuristics(weights, content_type)
        self._apply_channel_heuristics(weights, channel.lower())
        self._apply_traffic_type_heuristics(weights, traffic_type.lower())
        self._apply_funnel_stage_heuristics(weights, funnel_stage.upper())
        self._apply_segment_keyword_heuristics(weights, segment_strategy.lower())

        return weights

    def _apply_content_type_heuristics(
        self, weights: Dict[str, Dict[str, float]], content_type: str
    ) -> None:
        """Apply content type specific weight adjustments."""

        if content_type == "image":
            # For images: restrict motion to static/implied, reduce high motion weight
            weights["motion_intensity"]["high"] *= 0.1  # Effectively disable
            weights["motion_intensity"]["medium"] *= 0.1
            weights["motion_intensity"]["low"] *= 0.1
            weights["motion_intensity"]["static"] *= 1.5
            weights["motion_intensity"]["implied_motion"] *= 1.3

            # Bias toward simpler visual complexity
            weights["visual_complexity"]["minimal"] *= 1.2
            weights["visual_complexity"]["moderate"] *= 1.1
            weights["visual_complexity"]["busy"] *= 0.9

        elif content_type == "video":
            # For video: allow all motion, slight bias to medium/high
            weights["motion_intensity"]["medium"] *= 1.2
            weights["motion_intensity"]["high"] *= 1.1

    def _apply_channel_heuristics(
        self, weights: Dict[str, Dict[str, float]], channel: str
    ) -> None:
        """Apply channel-specific weight adjustments."""

        # TikTok / Reels optimization
        if "tiktok" in channel or "reels" in channel:
            weights["motion_intensity"]["high"] *= 1.5
            weights["figure_background_contrast"]["high"] *= 1.3
            weights["primary_subject_area_ratio"][">60"] *= 1.2
            weights["focal_point_count"]["1"] *= 1.1

        # Instagram Feed optimization
        if "instagram_feed" in channel or "instagram-feed" in channel:
            weights["focal_point_count"]["1"] *= 1.3
            weights["visual_complexity"]["minimal"] *= 1.2
            weights["figure_background_contrast"]["medium"] *= 1.1

        # YouTube optimization
        if "youtube" in channel:
            weights["text_coverage_ratio"]["<15"] *= 1.2
            weights["face_presence_scale"]["medium"] *= 1.2
            weights["face_presence_scale"]["dominant"] *= 1.1

        # LinkedIn optimization
        if "linkedin" in channel:
            weights["visual_complexity"]["minimal"] *= 1.3
            weights["figure_background_contrast"]["medium"] *= 1.2
            weights["text_coverage_ratio"]["15-30"] *= 1.1

        # Facebook optimization
        if "facebook" in channel:
            weights["primary_subject_area_ratio"]["40-60"] *= 1.2
            weights["text_coverage_ratio"]["<15"] *= 1.1

    def _apply_traffic_type_heuristics(
        self, weights: Dict[str, Dict[str, float]], traffic_type: str
    ) -> None:
        """Apply traffic type specific weight adjustments."""

        if traffic_type == "paid":
            # Paid traffic: emphasize product and offer visibility
            weights["product_visibility_ratio"]["clear"] *= 1.4
            weights["product_visibility_ratio"]["dominant"] *= 1.5
            weights["offer_visual_salience"]["clear"] *= 1.3
            weights["offer_visual_salience"]["dominant"] *= 1.4

            # Reduce subtle/none for key conversion features
            weights["product_visibility_ratio"]["none"] *= 0.7
            weights["offer_visual_salience"]["none"] *= 0.8

        elif traffic_type == "organic":
            # Organic: emphasize authenticity and human presence
            weights["face_presence_scale"]["medium"] *= 1.2
            weights["face_presence_scale"]["dominant"] *= 1.3
            weights["visual_complexity"]["minimal"] *= 1.1

            # Less aggressive product placement
            weights["product_visibility_ratio"]["subtle"] *= 1.2
            weights["offer_visual_salience"]["subtle"] *= 1.1

    def _apply_funnel_stage_heuristics(
        self, weights: Dict[str, Dict[str, float]], funnel_stage: str
    ) -> None:
        """Apply funnel stage specific weight adjustments."""

        if funnel_stage == "TOFU":
            # Top of funnel: grab attention, no hard sell
            weights["figure_background_contrast"]["high"] *= 1.4
            weights["offer_visual_salience"]["none"] *= 1.3
            weights["offer_visual_salience"]["subtle"] *= 1.1
            weights["motion_intensity"]["high"] *= 1.2

            # Reduce direct sales signals
            weights["offer_visual_salience"]["dominant"] *= 0.7
            weights["product_visibility_ratio"]["dominant"] *= 0.8

        elif funnel_stage == "MOFU":
            # Middle of funnel: educate and build interest
            weights["product_visibility_ratio"]["clear"] *= 1.3
            weights["product_visibility_ratio"]["subtle"] *= 1.2
            weights["text_coverage_ratio"]["15-30"] *= 1.2
            weights["face_presence_scale"]["medium"] *= 1.1

        elif funnel_stage == "BOFU":
            # Bottom of funnel: drive conversion
            weights["offer_visual_salience"]["dominant"] *= 1.5
            weights["offer_visual_salience"]["clear"] *= 1.3
            weights["product_visibility_ratio"]["dominant"] *= 1.4
            weights["product_visibility_ratio"]["clear"] *= 1.2
            weights["visual_complexity"]["minimal"] *= 1.2

            # Clear call to action, reduce ambiguity
            weights["focal_point_count"]["1"] *= 1.2
            weights["offer_visual_salience"]["none"] *= 0.5

    def _apply_segment_keyword_heuristics(
        self, weights: Dict[str, Dict[str, float]], segment_text: str
    ) -> None:
        """
        Apply keyword-based heuristics from segment strategy.
        Simple lowercase string matching - no NLP.
        """

        # Short-form content optimization
        if "short-form" in segment_text or "shortform" in segment_text:
            weights["motion_intensity"]["high"] *= 1.3
            weights["visual_complexity"]["minimal"] *= 1.2
            weights["focal_point_count"]["1"] *= 1.2

        # Authenticity signals
        if "authentic" in segment_text or "authenticity" in segment_text:
            weights["visual_complexity"]["minimal"] *= 1.2
            weights["face_presence_scale"]["medium"] *= 1.3
            weights["figure_background_contrast"]["medium"] *= 1.1

        # Bold/vibrant aesthetics
        if (
            "bold" in segment_text
            or "neon" in segment_text
            or "vibrant" in segment_text
        ):
            weights["figure_background_contrast"]["high"] *= 1.4
            weights["visual_complexity"]["moderate"] *= 1.2

        # Educational content
        if (
            "educational" in segment_text
            or "educate" in segment_text
            or "tutorial" in segment_text
        ):
            weights["text_coverage_ratio"]["15-30"] *= 1.3
            weights["text_coverage_ratio"]["<15"] *= 1.1
            weights["face_presence_scale"]["medium"] *= 1.2
            weights["visual_complexity"]["minimal"] *= 1.1

        # UGC style
        if "ugc" in segment_text or "user-generated" in segment_text:
            weights["face_presence_scale"]["medium"] *= 1.4
            weights["face_presence_scale"]["dominant"] *= 1.2
            weights["visual_complexity"]["minimal"] *= 1.2
            weights["figure_background_contrast"]["medium"] *= 1.1

        # Product focus
        if "product-focused" in segment_text or "product focused" in segment_text:
            weights["product_visibility_ratio"]["dominant"] *= 1.3
            weights["product_visibility_ratio"]["clear"] *= 1.2

        # Lifestyle content
        if "lifestyle" in segment_text:
            weights["face_presence_scale"]["medium"] *= 1.2
            weights["visual_complexity"]["moderate"] *= 1.1
            weights["product_visibility_ratio"]["subtle"] *= 1.2

        # Minimalist aesthetic
        if (
            "minimalist" in segment_text
            or "minimal" in segment_text
            or "clean" in segment_text
        ):
            weights["visual_complexity"]["minimal"] *= 1.4
            weights["focal_point_count"]["1"] *= 1.3
            weights["text_coverage_ratio"]["0"] *= 1.2

        # High energy / excitement
        if (
            "energy" in segment_text
            or "exciting" in segment_text
            or "dynamic" in segment_text
        ):
            weights["motion_intensity"]["high"] *= 1.3
            weights["figure_background_contrast"]["high"] *= 1.2

        # Professional / corporate
        if (
            "professional" in segment_text
            or "corporate" in segment_text
            or "b2b" in segment_text
        ):
            weights["visual_complexity"]["minimal"] *= 1.3
            weights["figure_background_contrast"]["medium"] *= 1.2
            weights["face_presence_scale"]["small"] *= 1.1


# Singleton instance
_interpreter_instance = None


def get_context_interpreter() -> ContextInterpreter:
    """Get singleton interpreter instance."""
    global _interpreter_instance
    if _interpreter_instance is None:
        _interpreter_instance = ContextInterpreter()
    return _interpreter_instance
