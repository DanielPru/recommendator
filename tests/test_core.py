"""
Unit tests for CVIE core modules.
Run with: pytest tests/ -v
"""

import pytest
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFeatureSchema:
    """Tests for feature_schema.py"""

    def test_schema_has_9_dimensions(self):
        from app.core.feature_schema import FEATURE_SCHEMA_V1

        assert len(FEATURE_SCHEMA_V1) == 9

    def test_schema_is_immutable(self):
        from app.core.feature_schema import FEATURE_SCHEMA_V1

        with pytest.raises(TypeError):
            FEATURE_SCHEMA_V1["new_feature"] = ("a", "b")

    def test_all_features_present(self):
        from app.core.feature_schema import FEATURE_SCHEMA_V1

        expected = {
            "primary_subject_area_ratio",
            "focal_point_count",
            "figure_background_contrast",
            "motion_intensity",
            "text_coverage_ratio",
            "face_presence_scale",
            "product_visibility_ratio",
            "offer_visual_salience",
            "visual_complexity",
        }
        assert set(FEATURE_SCHEMA_V1.keys()) == expected

    def test_motion_intensity_values(self):
        from app.core.feature_schema import FEATURE_SCHEMA_V1

        expected = ("low", "medium", "high", "static", "implied_motion")
        assert FEATURE_SCHEMA_V1["motion_intensity"] == expected

    def test_encode_structure(self):
        from app.core.feature_schema import encode_structure, FEATURE_SCHEMA_V1

        structure = {f: vals[0] for f, vals in FEATURE_SCHEMA_V1.items()}
        encoded = encode_structure(structure)

        assert len(encoded) == 9
        assert all(isinstance(v, int) for v in encoded.values())
        assert all(v == 0 for v in encoded.values())  # First values encode to 0

    def test_decode_structure(self):
        from app.core.feature_schema import (
            encode_structure,
            decode_structure,
            FEATURE_SCHEMA_V1,
        )

        original = {f: vals[0] for f, vals in FEATURE_SCHEMA_V1.items()}
        encoded = encode_structure(original)
        decoded = decode_structure(encoded)

        assert decoded == original

    def test_structure_to_vector(self):
        from app.core.feature_schema import (
            structure_to_vector,
            FEATURE_SCHEMA_V1,
            FEATURE_NAMES,
        )

        structure = {f: vals[0] for f, vals in FEATURE_SCHEMA_V1.items()}
        vector = structure_to_vector(structure)

        assert len(vector) == 9
        assert isinstance(vector, list)

    def test_compute_structure_hash_deterministic(self):
        from app.core.feature_schema import compute_structure_hash, FEATURE_SCHEMA_V1

        structure = {f: vals[0] for f, vals in FEATURE_SCHEMA_V1.items()}
        hash1 = compute_structure_hash(structure)
        hash2 = compute_structure_hash(structure)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_different_structures_different_hashes(self):
        from app.core.feature_schema import compute_structure_hash, FEATURE_SCHEMA_V1

        structure1 = {f: vals[0] for f, vals in FEATURE_SCHEMA_V1.items()}
        structure2 = {f: vals[-1] for f, vals in FEATURE_SCHEMA_V1.items()}

        assert compute_structure_hash(structure1) != compute_structure_hash(structure2)

    def test_get_allowed_values_image(self):
        from app.core.feature_schema import get_allowed_values

        motion_values = get_allowed_values("motion_intensity", "image")
        assert "static" in motion_values
        assert "implied_motion" in motion_values
        assert "high" not in motion_values

    def test_get_allowed_values_video(self):
        from app.core.feature_schema import get_allowed_values

        motion_values = get_allowed_values("motion_intensity", "video")
        assert "high" in motion_values
        assert "static" in motion_values


class TestContextInterpreter:
    """Tests for context_interpreter.py"""

    def test_interpreter_returns_weights(self):
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()
        weights = interpreter.interpret(
            segment_strategy="test strategy",
            channel="instagram_feed",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="image",
        )

        assert isinstance(weights, dict)
        assert len(weights) == 9

    def test_all_weights_positive(self):
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()
        weights = interpreter.interpret(
            segment_strategy="bold authentic ugc",
            channel="tiktok",
            traffic_type="paid",
            funnel_stage="BOFU",
            content_type="video",
        )

        for feature, value_weights in weights.items():
            for value, weight in value_weights.items():
                assert weight > 0, (
                    f"{feature}.{value} has non-positive weight: {weight}"
                )

    def test_tiktok_boosts_high_motion(self):
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()

        tiktok_weights = interpreter.interpret(
            segment_strategy="test",
            channel="tiktok",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        generic_weights = interpreter.interpret(
            segment_strategy="test",
            channel="generic",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        assert (
            tiktok_weights["motion_intensity"]["high"]
            > generic_weights["motion_intensity"]["high"]
        )

    def test_paid_boosts_product_visibility(self):
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()

        paid_weights = interpreter.interpret(
            segment_strategy="test",
            channel="generic",
            traffic_type="paid",
            funnel_stage="MOFU",
            content_type="image",
        )

        organic_weights = interpreter.interpret(
            segment_strategy="test",
            channel="generic",
            traffic_type="organic",
            funnel_stage="MOFU",
            content_type="image",
        )

        assert (
            paid_weights["product_visibility_ratio"]["dominant"]
            > organic_weights["product_visibility_ratio"]["dominant"]
        )

    def test_bofu_boosts_offer_salience(self):
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()

        bofu_weights = interpreter.interpret(
            segment_strategy="test",
            channel="generic",
            traffic_type="paid",
            funnel_stage="BOFU",
            content_type="image",
        )

        tofu_weights = interpreter.interpret(
            segment_strategy="test",
            channel="generic",
            traffic_type="paid",
            funnel_stage="TOFU",
            content_type="image",
        )

        assert (
            bofu_weights["offer_visual_salience"]["dominant"]
            > tofu_weights["offer_visual_salience"]["dominant"]
        )

    def test_keyword_ugc_boosts_face(self):
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()

        ugc_weights = interpreter.interpret(
            segment_strategy="we want ugc style content",
            channel="generic",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        no_ugc_weights = interpreter.interpret(
            segment_strategy="we want professional content",
            channel="generic",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        assert (
            ugc_weights["face_presence_scale"]["medium"]
            > no_ugc_weights["face_presence_scale"]["medium"]
        )

    def test_image_reduces_high_motion(self):
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()

        image_weights = interpreter.interpret(
            segment_strategy="test",
            channel="generic",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="image",
        )

        # High motion should be heavily reduced for images
        assert image_weights["motion_intensity"]["high"] < 0.5


class TestStructureGenerator:
    """Tests for structure_generator.py"""

    def test_generates_correct_count(self):
        from app.core.structure_generator import StructureGenerator
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()
        weights = interpreter.interpret(
            segment_strategy="test",
            channel="tiktok",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        generator = StructureGenerator(min_candidates=50, max_candidates=100)
        candidates = generator.generate(weights, "video", target_count=75)

        assert len(candidates) == 75

    def test_all_structures_unique(self):
        from app.core.structure_generator import StructureGenerator
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()
        weights = interpreter.interpret(
            segment_strategy="test",
            channel="tiktok",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        generator = StructureGenerator(min_candidates=50, max_candidates=100)
        candidates = generator.generate(weights, "video", target_count=80)

        hashes = [c.structure_hash for c in candidates]
        assert len(hashes) == len(set(hashes)), "Duplicate structures found"

    def test_diversity_ratio_respected(self):
        from app.core.structure_generator import StructureGenerator
        from app.core.context_interpreter import get_context_interpreter

        interpreter = get_context_interpreter()
        weights = interpreter.interpret(
            segment_strategy="test",
            channel="tiktok",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        generator = StructureGenerator(
            min_candidates=100,
            max_candidates=100,
            diversity_ratio=0.2,
        )
        candidates = generator.generate(weights, "video", target_count=100)

        diverse_count = sum(1 for c in candidates if c.is_diverse)
        # Should be approximately 20% (allow some variance)
        assert 15 <= diverse_count <= 25, (
            f"Diversity count {diverse_count} not in expected range"
        )

    def test_image_structures_have_valid_motion(self):
        from app.core.structure_generator import StructureGenerator
        from app.core.context_interpreter import get_context_interpreter
        from app.core.feature_schema import IMAGE_ALLOWED_MOTION

        interpreter = get_context_interpreter()
        weights = interpreter.interpret(
            segment_strategy="test",
            channel="instagram_feed",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="image",
        )

        generator = StructureGenerator(min_candidates=50, max_candidates=100)
        candidates = generator.generate(weights, "image", target_count=50)

        for c in candidates:
            motion = c.features["motion_intensity"]
            assert motion in IMAGE_ALLOWED_MOTION, f"Invalid motion for image: {motion}"

    def test_all_features_present_in_structure(self):
        from app.core.structure_generator import StructureGenerator
        from app.core.context_interpreter import get_context_interpreter
        from app.core.feature_schema import FEATURE_NAMES

        interpreter = get_context_interpreter()
        weights = interpreter.interpret(
            segment_strategy="test",
            channel="tiktok",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        generator = StructureGenerator(min_candidates=50, max_candidates=100)
        candidates = generator.generate(weights, "video", target_count=10)

        for c in candidates:
            assert set(c.features.keys()) == FEATURE_NAMES


class TestSchemas:
    """Tests for API schemas"""

    def test_recommend_request_validation(self):
        from app.api.schemas import RecommendRequest

        request = RecommendRequest(
            asset_id="asset_20240115_test123",
            segment_strategy="test strategy",
            channel="tiktok",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        assert request.asset_id == "asset_20240115_test123"
        assert request.traffic_type.value == "organic"
        assert request.funnel_stage.value == "TOFU"

    def test_recommend_request_requires_asset_id(self):
        from app.api.schemas import RecommendRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RecommendRequest(
                # Missing asset_id
                segment_strategy="test",
                channel="tiktok",
                traffic_type="organic",
                funnel_stage="TOFU",
                content_type="video",
            )

    def test_invalid_traffic_type_rejected(self):
        from app.api.schemas import RecommendRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RecommendRequest(
                asset_id="asset_123",
                segment_strategy="test",
                channel="tiktok",
                traffic_type="invalid",
                funnel_stage="TOFU",
                content_type="video",
            )

    def test_invalid_funnel_stage_rejected(self):
        from app.api.schemas import RecommendRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RecommendRequest(
                asset_id="asset_123",
                segment_strategy="test",
                channel="tiktok",
                traffic_type="organic",
                funnel_stage="INVALID",
                content_type="video",
            )

    def test_performance_request_uses_asset_id(self):
        from app.api.schemas import PerformanceRequest

        req = PerformanceRequest(
            asset_id="asset_20240115_abc123",
            attention_score=0.5,
            persuasion_score=0.8,
        )
        assert req.asset_id == "asset_20240115_abc123"

    def test_performance_score_bounds(self):
        from app.api.schemas import PerformanceRequest
        from pydantic import ValidationError

        # Valid scores
        req = PerformanceRequest(
            asset_id="asset_20240115_abc123",
            attention_score=0.5,
            persuasion_score=0.8,
        )
        assert req.attention_score == 0.5

        # Invalid score > 1
        with pytest.raises(ValidationError):
            PerformanceRequest(
                asset_id="asset_20240115_abc123",
                attention_score=1.5,
                persuasion_score=0.8,
            )

        # Invalid score < 0
        with pytest.raises(ValidationError):
            PerformanceRequest(
                asset_id="asset_20240115_abc123",
                attention_score=-0.1,
                persuasion_score=0.8,
            )

    def test_recommend_response_schema(self):
        from app.api.schemas import RecommendResponse, StructureFeatures

        response = RecommendResponse(
            asset_id="asset_20240115_test",
            structure_hash="abc123def456",
            structure=StructureFeatures(
                primary_subject_area_ratio=">60",
                focal_point_count="1",
                figure_background_contrast="high",
                motion_intensity="high",
                text_coverage_ratio="<15",
                face_presence_scale="medium",
                product_visibility_ratio="subtle",
                offer_visual_salience="none",
                visual_complexity="minimal",
            ),
            p_attention=0.75,
            p_persuasion=0.82,
            p_final=0.615,
            mode="exploitation",
            context_confidence=0.8,
            exploration_weight=0.0,
            model_version="v1_20240115",
            candidates_evaluated=75,
            context_hash="ctx_hash_123",
        )

        assert response.asset_id == "asset_20240115_test"
        assert response.mode == "exploitation"
        assert response.p_final == 0.615


class TestModelManager:
    """Tests for model_manager.py (without actual models)"""

    def test_model_manager_class_exists(self):
        """Test that ModelManager class is importable and has required methods"""
        from app.ml.model_manager import ModelManager, MODEL_TYPES

        # Verify class has required methods
        assert hasattr(ModelManager, "load_models")
        assert hasattr(ModelManager, "reload_models")
        assert hasattr(ModelManager, "get_model")
        assert hasattr(ModelManager, "score_structure")
        assert hasattr(ModelManager, "is_ready")

        # Verify model types
        assert len(MODEL_TYPES) == 4
        assert "organic_attention" in MODEL_TYPES
        assert "paid_persuasion" in MODEL_TYPES


class TestIntegration:
    """Integration tests for the full flow (without DB)"""

    def test_full_recommendation_flow_without_db(self):
        """Test that the core logic works end-to-end"""
        from app.core.context_interpreter import get_context_interpreter
        from app.core.structure_generator import StructureGenerator
        from app.core.feature_schema import FEATURE_NAMES

        # 1. Interpret context
        interpreter = get_context_interpreter()
        weights = interpreter.interpret(
            segment_strategy="Short-form authentic UGC content for Gen Z audience",
            channel="tiktok",
            traffic_type="organic",
            funnel_stage="TOFU",
            content_type="video",
        )

        assert len(weights) == 9

        # 2. Generate candidates (use class directly, not singleton)
        generator = StructureGenerator(min_candidates=50, max_candidates=100)
        candidates = generator.generate(weights, "video")

        assert 50 <= len(candidates) <= 100

        # 3. Verify structure integrity
        for c in candidates:
            assert len(c.features) == 9
            assert all(f in FEATURE_NAMES for f in c.features.keys())
            assert len(c.structure_hash) == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
