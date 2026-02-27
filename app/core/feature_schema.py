"""
Feature Schema V1 - Frozen visual feature definitions.
All encoding mappings are immutable and versioned.
"""

from typing import Dict, FrozenSet, Tuple
from types import MappingProxyType


SCHEMA_VERSION = "v1"


# Immutable feature schema definition
_FEATURE_SCHEMA_V1_RAW: Dict[str, Tuple[str, ...]] = {
    "primary_subject_area_ratio": ("<20", "20-40", "40-60", ">60"),
    "focal_point_count": ("1", "2", "3"),
    "figure_background_contrast": ("low", "medium", "high"),
    "motion_intensity": ("low", "medium", "high", "static", "implied_motion"),
    "text_coverage_ratio": ("0", "<15", "15-30", ">30"),
    "face_presence_scale": ("none", "small", "medium", "dominant"),
    "product_visibility_ratio": ("none", "subtle", "clear", "dominant"),
    "offer_visual_salience": ("none", "subtle", "clear", "dominant"),
    "visual_complexity": ("minimal", "moderate", "busy"),
}

# Freeze the schema
FEATURE_SCHEMA_V1: MappingProxyType = MappingProxyType(
    {k: tuple(v) for k, v in _FEATURE_SCHEMA_V1_RAW.items()}
)

# Feature names as frozen set
FEATURE_NAMES: FrozenSet[str] = frozenset(FEATURE_SCHEMA_V1.keys())


# Deterministic categorical encoding mappings (frozen)
def _create_encoding_map(values: Tuple[str, ...]) -> MappingProxyType:
    """Create frozen encoding map for categorical values."""
    return MappingProxyType({v: i for i, v in enumerate(values)})


def _create_decoding_map(values: Tuple[str, ...]) -> MappingProxyType:
    """Create frozen decoding map for categorical values."""
    return MappingProxyType({i: v for i, v in enumerate(values)})


# Frozen encoding maps per feature
FEATURE_ENCODINGS: MappingProxyType = MappingProxyType(
    {
        feature: _create_encoding_map(values)
        for feature, values in FEATURE_SCHEMA_V1.items()
    }
)

# Frozen decoding maps per feature
FEATURE_DECODINGS: MappingProxyType = MappingProxyType(
    {
        feature: _create_decoding_map(values)
        for feature, values in FEATURE_SCHEMA_V1.items()
    }
)


# Content type restrictions
IMAGE_ALLOWED_MOTION: FrozenSet[str] = frozenset({"static", "implied_motion"})
VIDEO_ALLOWED_MOTION: FrozenSet[str] = frozenset(
    {"low", "medium", "high", "static", "implied_motion"}
)


def get_allowed_values(feature: str, content_type: str) -> Tuple[str, ...]:
    """
    Get allowed values for a feature based on content type.
    Applies content-type restrictions.
    """
    base_values = FEATURE_SCHEMA_V1[feature]

    if feature == "motion_intensity":
        if content_type == "image":
            return tuple(v for v in base_values if v in IMAGE_ALLOWED_MOTION)
        elif content_type == "video":
            return tuple(v for v in base_values if v in VIDEO_ALLOWED_MOTION)

    return base_values


def encode_structure(structure: Dict[str, str]) -> Dict[str, int]:
    """
    Encode a structure dictionary to numeric values.
    Uses frozen encoding mappings for determinism.
    """
    encoded = {}
    for feature, value in structure.items():
        if feature in FEATURE_ENCODINGS:
            encoding_map = FEATURE_ENCODINGS[feature]
            if value in encoding_map:
                encoded[feature] = encoding_map[value]
            else:
                raise ValueError(f"Invalid value '{value}' for feature '{feature}'")
        else:
            raise ValueError(f"Unknown feature '{feature}'")
    return encoded


def decode_structure(encoded: Dict[str, int]) -> Dict[str, str]:
    """
    Decode a numeric structure back to categorical values.
    Uses frozen decoding mappings for determinism.
    """
    decoded = {}
    for feature, code in encoded.items():
        if feature in FEATURE_DECODINGS:
            decoding_map = FEATURE_DECODINGS[feature]
            if code in decoding_map:
                decoded[feature] = decoding_map[code]
            else:
                raise ValueError(f"Invalid code '{code}' for feature '{feature}'")
        else:
            raise ValueError(f"Unknown feature '{feature}'")
    return decoded


def structure_to_vector(structure: Dict[str, str]) -> list:
    """
    Convert structure to ordered feature vector for ML.
    Features are sorted alphabetically for consistency.
    """
    encoded = encode_structure(structure)
    return [encoded[f] for f in sorted(FEATURE_NAMES)]


def vector_to_structure(vector: list) -> Dict[str, str]:
    """
    Convert ordered feature vector back to structure dict.
    """
    sorted_features = sorted(FEATURE_NAMES)
    encoded = {f: v for f, v in zip(sorted_features, vector)}
    return decode_structure(encoded)


def compute_structure_hash(structure: Dict[str, str]) -> str:
    """
    Compute deterministic hash for structure uniqueness.
    Uses sorted feature-value pairs for consistency.
    """
    import hashlib

    sorted_items = sorted(structure.items())
    hash_input = "|".join(f"{k}:{v}" for k, v in sorted_items)
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
