"""
Model Manager - Thread-safe singleton for LightGBM model management.
Supports hot-reload after retraining.
"""

import os
import threading
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

import lightgbm as lgb
import numpy as np

from app.config import get_settings
from app.core.feature_schema import FEATURE_NAMES, structure_to_vector


# Model types
MODEL_TYPES = [
    "organic_attention",
    "organic_persuasion",
    "paid_attention",
    "paid_persuasion",
]


class ModelManager:
    """
    Thread-safe singleton manager for LightGBM models.
    Handles loading, scoring, and hot-reloading.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize model manager."""
        if self._initialized:
            return

        self._models: Dict[str, lgb.Booster] = {}
        self._model_lock = threading.RLock()
        self._current_version: Optional[str] = None
        self._models_loaded = False
        self._feature_names = sorted(FEATURE_NAMES)

        # Try to load models on init
        self.load_models()
        self._initialized = True

    @property
    def is_ready(self) -> bool:
        """Check if models are loaded and ready for scoring."""
        with self._model_lock:
            return self._models_loaded and len(self._models) == len(MODEL_TYPES)

    @property
    def current_version(self) -> Optional[str]:
        """Get current model version."""
        with self._model_lock:
            return self._current_version

    def load_models(self, version: Optional[str] = None) -> bool:
        """
        Load models from disk.

        Args:
            version: Specific version to load, or None for latest

        Returns:
            True if all models loaded successfully
        """
        settings = get_settings()
        model_dir = Path(settings.model_dir)

        if version is None:
            version = self._find_latest_version(model_dir)

        if version is None:
            return False

        version_dir = model_dir / version
        if not version_dir.exists():
            return False

        with self._model_lock:
            loaded_models = {}

            for model_type in MODEL_TYPES:
                model_path = version_dir / f"{model_type}.txt"
                if model_path.exists():
                    try:
                        model = lgb.Booster(model_file=str(model_path))
                        loaded_models[model_type] = model
                    except Exception as e:
                        print(f"Failed to load {model_type}: {e}")
                        return False
                else:
                    return False

            # All models loaded successfully
            self._models = loaded_models
            self._current_version = version
            self._models_loaded = True

            return True

    def reload_models(self, version: Optional[str] = None) -> bool:
        """
        Hot-reload models (thread-safe).

        Args:
            version: Specific version to reload, or None for latest

        Returns:
            True if reload successful
        """
        return self.load_models(version)

    def get_model(self, model_type: str) -> Optional[lgb.Booster]:
        """
        Get a specific model.

        Args:
            model_type: One of MODEL_TYPES

        Returns:
            LightGBM Booster or None
        """
        with self._model_lock:
            return self._models.get(model_type)

    def score_structure(
        self,
        structure: Dict[str, str],
        traffic_type: str,
    ) -> Optional[Dict[str, float]]:
        """
        Score a structure using appropriate models.

        Args:
            structure: Feature dictionary
            traffic_type: "organic" or "paid"

        Returns:
            Dictionary with attention, persuasion, and final scores
        """
        if not self.is_ready:
            return None

        # Convert structure to feature vector
        feature_vector = np.array([structure_to_vector(structure)])

        # Select models based on traffic type
        attention_model_type = f"{traffic_type}_attention"
        persuasion_model_type = f"{traffic_type}_persuasion"

        with self._model_lock:
            attention_model = self._models.get(attention_model_type)
            persuasion_model = self._models.get(persuasion_model_type)

            if attention_model is None or persuasion_model is None:
                return None

            # Predict probabilities
            p_attention = attention_model.predict(feature_vector)[0]
            p_persuasion = persuasion_model.predict(feature_vector)[0]

        # Clip to valid probability range
        p_attention = np.clip(p_attention, 0.0, 1.0)
        p_persuasion = np.clip(p_persuasion, 0.0, 1.0)

        # Combined score
        p_final = p_attention * p_persuasion

        return {
            "attention": float(p_attention),
            "persuasion": float(p_persuasion),
            "final": float(p_final),
        }

    def score_structures_batch(
        self,
        structures: List[Dict[str, str]],
        traffic_type: str,
    ) -> Optional[List[Dict[str, float]]]:
        """
        Score multiple structures in batch.

        Args:
            structures: List of feature dictionaries
            traffic_type: "organic" or "paid"

        Returns:
            List of score dictionaries
        """
        if not self.is_ready or not structures:
            return None

        # Convert all structures to feature matrix
        feature_matrix = np.array([structure_to_vector(s) for s in structures])

        attention_model_type = f"{traffic_type}_attention"
        persuasion_model_type = f"{traffic_type}_persuasion"

        with self._model_lock:
            attention_model = self._models.get(attention_model_type)
            persuasion_model = self._models.get(persuasion_model_type)

            if attention_model is None or persuasion_model is None:
                return None

            p_attention = attention_model.predict(feature_matrix)
            p_persuasion = persuasion_model.predict(feature_matrix)

        # Clip and combine
        p_attention = np.clip(p_attention, 0.0, 1.0)
        p_persuasion = np.clip(p_persuasion, 0.0, 1.0)
        p_final = p_attention * p_persuasion

        return [
            {
                "attention": float(att),
                "persuasion": float(per),
                "final": float(fin),
            }
            for att, per, fin in zip(p_attention, p_persuasion, p_final)
        ]

    def _find_latest_version(self, model_dir: Path) -> Optional[str]:
        """Find the latest model version in directory."""
        if not model_dir.exists():
            return None

        versions = []
        for item in model_dir.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                versions.append(item.name)

        if not versions:
            return None

        # Sort by version number
        versions.sort(key=lambda v: self._parse_version(v), reverse=True)
        return versions[0]

    def _parse_version(self, version: str) -> int:
        """Parse version string to sortable integer."""
        try:
            # Handle formats like "v1", "v2", "v1.0", etc.
            clean = version.lstrip("v").split(".")[0]
            return int(clean)
        except ValueError:
            return 0


# Singleton accessor
_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get singleton model manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelManager()
    return _manager_instance
