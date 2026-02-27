"""
Model Trainer - LightGBM training with 75th percentile labeling.
Produces 4 classifiers: organic_attention, organic_persuasion, paid_attention, paid_persuasion.
Uses asset_id for joining decision and performance logs.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import DecisionLog, PerformanceLog, ModelRegistry, ContextStats
from app.core.feature_schema import FEATURE_NAMES, structure_to_vector
from app.ml.model_manager import MODEL_TYPES, get_model_manager


# LightGBM parameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

NUM_BOOST_ROUNDS = 100
EARLY_STOPPING_ROUNDS = 10
MIN_TRAINING_SAMPLES = 50


class ModelTrainer:
    """Handles model training and versioning."""

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
        self.model_dir = Path(self.settings.model_dir)

    def train_all_models(self) -> Dict[str, any]:
        """
        Train all 4 model types.

        Returns:
            Dictionary with training results and new version
        """
        # Generate new version
        version = self._generate_version()
        version_dir = self.model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Fetch and prepare data
        data = self._fetch_training_data()
        if data is None or len(data["organic"]) < MIN_TRAINING_SAMPLES:
            return {
                "success": False,
                "error": f"Insufficient training data (need {MIN_TRAINING_SAMPLES}+)",
                "samples": len(data["organic"]) if data else 0,
            }

        # Compute 75th percentile thresholds
        thresholds = self._compute_thresholds(data)

        results = {
            "version": version,
            "models": {},
            "thresholds": thresholds,
        }

        # Train each model type
        for model_type in MODEL_TYPES:
            traffic_type, metric_type = model_type.rsplit("_", 1)

            # Get data for this traffic type
            traffic_data = data.get(traffic_type, [])
            if len(traffic_data) < MIN_TRAINING_SAMPLES:
                results["models"][model_type] = {
                    "success": False,
                    "error": f"Insufficient {traffic_type} data",
                }
                continue

            # Prepare features and labels
            X, y = self._prepare_dataset(
                traffic_data,
                metric_type,
                thresholds[traffic_type][metric_type],
            )

            # Train model
            model_result = self._train_single_model(X, y, model_type, version_dir)
            results["models"][model_type] = model_result

            # Register model
            if model_result["success"]:
                self._register_model(
                    version=version,
                    model_type=model_type,
                    model_path=str(version_dir / f"{model_type}.txt"),
                    training_samples=len(y),
                    auc_score=model_result.get("auc"),
                )

        # Update context stats
        self._update_context_stats()

        # Reload models
        manager = get_model_manager()
        reload_success = manager.reload_models(version)
        results["reload_success"] = reload_success
        results["success"] = all(
            m.get("success", False) for m in results["models"].values()
        )

        return results

    def _fetch_training_data(self) -> Optional[Dict[str, List]]:
        """
        Fetch training data by joining decision and performance logs via asset_id.
        Returns data grouped by traffic type.
        """
        query = text("""
            SELECT 
                d.structure_features,
                d.traffic_type,
                p.attention_score,
                p.persuasion_score
            FROM decision_logs d
            INNER JOIN performance_logs p ON d.asset_id = p.asset_id
            WHERE d.schema_version = 'v1'
        """)

        result = self.db.execute(query)
        rows = result.fetchall()

        if not rows:
            return None

        data = {"organic": [], "paid": []}

        for row in rows:
            traffic_type = row.traffic_type.lower()
            if traffic_type in data:
                data[traffic_type].append(
                    {
                        "features": row.structure_features,
                        "attention": row.attention_score,
                        "persuasion": row.persuasion_score,
                    }
                )

        return data

    def _compute_thresholds(self, data: Dict[str, List]) -> Dict[str, Dict[str, float]]:
        """
        Compute 75th percentile thresholds using SQL percentile_cont.
        """
        thresholds = {}

        for traffic_type in ["organic", "paid"]:
            query = text("""
                SELECT 
                    percentile_cont(0.75) WITHIN GROUP (ORDER BY p.attention_score) as p75_attention,
                    percentile_cont(0.75) WITHIN GROUP (ORDER BY p.persuasion_score) as p75_persuasion
                FROM decision_logs d
                INNER JOIN performance_logs p ON d.asset_id = p.asset_id
                WHERE d.traffic_type = :traffic_type
            """)

            result = self.db.execute(query, {"traffic_type": traffic_type})
            row = result.fetchone()

            if row and row.p75_attention is not None:
                thresholds[traffic_type] = {
                    "attention": float(row.p75_attention),
                    "persuasion": float(row.p75_persuasion),
                }
            else:
                # Fallback to in-memory calculation
                traffic_data = data.get(traffic_type, [])
                if traffic_data:
                    attention_scores = [d["attention"] for d in traffic_data]
                    persuasion_scores = [d["persuasion"] for d in traffic_data]
                    thresholds[traffic_type] = {
                        "attention": float(np.percentile(attention_scores, 75)),
                        "persuasion": float(np.percentile(persuasion_scores, 75)),
                    }
                else:
                    thresholds[traffic_type] = {
                        "attention": 0.5,
                        "persuasion": 0.5,
                    }

        return thresholds

    def _prepare_dataset(
        self,
        data: List[Dict],
        metric_type: str,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and binary labels.
        Labels are 1 if score >= 75th percentile threshold.
        """
        X = []
        y = []

        for item in data:
            features = item["features"]
            score = item[metric_type]

            # Convert to feature vector
            feature_vector = structure_to_vector(features)
            X.append(feature_vector)

            # Binary label: top 25% = 1
            label = 1 if score >= threshold else 0
            y.append(label)

        return np.array(X), np.array(y)

    def _train_single_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        version_dir: Path,
    ) -> Dict:
        """Train a single LightGBM model."""

        # Check class balance
        positive_ratio = y.mean()
        if positive_ratio < 0.05 or positive_ratio > 0.95:
            return {
                "success": False,
                "error": f"Severe class imbalance: {positive_ratio:.2%} positive",
            }

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=sorted(FEATURE_NAMES),
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
        )

        # Train
        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0),  # Suppress logging
        ]

        model = lgb.train(
            LGBM_PARAMS,
            train_data,
            num_boost_round=NUM_BOOST_ROUNDS,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Evaluate
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)

        # Binary predictions at 0.5 threshold
        y_pred_binary = (y_pred >= 0.5).astype(int)
        precision = precision_score(y_val, y_pred_binary, zero_division=0)

        # Save model
        model_path = version_dir / f"{model_type}.txt"
        model.save_model(str(model_path))

        return {
            "success": True,
            "auc": float(auc),
            "precision": float(precision),
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "positive_ratio": float(positive_ratio),
            "model_path": str(model_path),
        }

    def _register_model(
        self,
        version: str,
        model_type: str,
        model_path: str,
        training_samples: int,
        auc_score: Optional[float],
    ) -> None:
        """Register model in database."""

        # Deactivate previous versions of this model type
        self.db.execute(
            text("""
                UPDATE model_registry 
                SET is_active = false 
                WHERE model_type = :model_type AND is_active = true
            """),
            {"model_type": model_type},
        )

        # Insert new entry
        registry = ModelRegistry(
            version=version,
            model_type=model_type,
            model_path=model_path,
            training_samples=training_samples,
            auc_score=auc_score,
            is_active=True,
        )
        self.db.add(registry)
        self.db.commit()

    def _update_context_stats(self) -> None:
        """Update aggregated context statistics using asset_id joins."""

        query = text("""
            INSERT INTO context_stats (context_hash, total_decisions, total_performances, 
                                       avg_attention, avg_persuasion, std_attention, std_persuasion,
                                       first_seen, last_updated)
            SELECT 
                d.context_hash,
                COUNT(DISTINCT d.asset_id) as total_decisions,
                COUNT(p.id) as total_performances,
                AVG(p.attention_score) as avg_attention,
                AVG(p.persuasion_score) as avg_persuasion,
                STDDEV(p.attention_score) as std_attention,
                STDDEV(p.persuasion_score) as std_persuasion,
                MIN(d.created_at) as first_seen,
                NOW() as last_updated
            FROM decision_logs d
            LEFT JOIN performance_logs p ON d.asset_id = p.asset_id
            GROUP BY d.context_hash
            ON CONFLICT (context_hash) DO UPDATE SET
                total_decisions = EXCLUDED.total_decisions,
                total_performances = EXCLUDED.total_performances,
                avg_attention = EXCLUDED.avg_attention,
                avg_persuasion = EXCLUDED.avg_persuasion,
                std_attention = EXCLUDED.std_attention,
                std_persuasion = EXCLUDED.std_persuasion,
                last_updated = NOW()
        """)

        self.db.execute(query)
        self.db.commit()

    def _generate_version(self) -> str:
        """Generate new version string."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Find latest version number
        latest_num = 0
        if self.model_dir.exists():
            for item in self.model_dir.iterdir():
                if item.is_dir() and item.name.startswith("v"):
                    try:
                        num = int(item.name.split("_")[0].lstrip("v"))
                        latest_num = max(latest_num, num)
                    except (ValueError, IndexError):
                        pass

        return f"v{latest_num + 1}_{timestamp}"
