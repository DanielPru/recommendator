"""
Bulk import script for training CVIE with existing video data.

Usage:
    python scripts/bulk_import.py --input data/videos.csv
    python scripts/bulk_import.py --input data/videos.json

Expected CSV columns:
    asset_id, channel, traffic_type, funnel_stage, content_type, segment_strategy,
    attention_score, persuasion_score, [optional: raw_metrics as JSON string]

Expected JSON format:
    [
        {
            "asset_id": "video_001",
            "channel": "tiktok",
            "traffic_type": "organic",
            "funnel_stage": "TOFU",
            "content_type": "video",
            "segment_strategy": "UGC content for Gen Z",
            "attention_score": 0.72,
            "persuasion_score": 0.45,
            "raw_metrics": {"views": 15000, "ctr": 0.028}
        },
        ...
    ]
"""

import argparse
import json
import csv
import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from app.db.database import get_db_context
from app.db.models import DecisionLog, PerformanceLog
from app.core.context_interpreter import get_context_interpreter
from app.core.structure_generator import get_structure_generator
from app.core.feature_schema import compute_structure_hash
from app.ml.trainer import ModelTrainer
import hashlib


def compute_context_hash(
    segment_strategy: str,
    channel: str,
    traffic_type: str,
    funnel_stage: str,
    content_type: str,
) -> str:
    """Compute hash for context grouping."""
    keywords = extract_keywords(segment_strategy)
    hash_input = "|".join(
        [
            keywords,
            channel.lower(),
            traffic_type.lower(),
            funnel_stage.upper(),
            content_type.lower(),
        ]
    )
    return hashlib.sha256(hash_input.encode()).hexdigest()[:32]


def extract_keywords(text: str, max_keywords: int = 10) -> str:
    """Extract significant keywords from text."""
    keywords = set()
    text_lower = text.lower()

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

    sorted_keywords = sorted(keywords)[:max_keywords]
    return ",".join(sorted_keywords) if sorted_keywords else "generic"


def generate_structure_for_context(
    channel: str,
    traffic_type: str,
    funnel_stage: str,
    content_type: str,
    segment_strategy: str,
) -> dict:
    """Generate a plausible structure based on context."""
    interpreter = get_context_interpreter()
    generator = get_structure_generator()

    weights = interpreter.interpret(
        segment_strategy=segment_strategy,
        channel=channel,
        traffic_type=traffic_type,
        funnel_stage=funnel_stage,
        content_type=content_type,
    )

    # Generate one candidate
    candidates = generator.generate(weights, content_type, target_count=1)
    return candidates[0]


def load_csv(filepath: str) -> list:
    """Load data from CSV file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse raw_metrics if present
            if "raw_metrics" in row and row["raw_metrics"]:
                try:
                    row["raw_metrics"] = json.loads(row["raw_metrics"])
                except json.JSONDecodeError:
                    row["raw_metrics"] = None
            else:
                row["raw_metrics"] = None

            # Convert scores to float
            row["attention_score"] = float(row["attention_score"])
            row["persuasion_score"] = float(row["persuasion_score"])

            data.append(row)
    return data


def load_json(filepath: str) -> list:
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def import_videos(data: list, db: Session, batch_size: int = 100) -> dict:
    """Import videos into database."""
    stats = {"total": len(data), "imported": 0, "skipped": 0, "errors": []}

    for i, video in enumerate(data):
        try:
            # Check if already exists
            existing = (
                db.query(DecisionLog)
                .filter(DecisionLog.asset_id == video["asset_id"])
                .first()
            )

            if existing:
                stats["skipped"] += 1
                continue

            # Generate structure based on context
            structure = generate_structure_for_context(
                channel=video["channel"],
                traffic_type=video["traffic_type"],
                funnel_stage=video["funnel_stage"],
                content_type=video.get("content_type", "video"),
                segment_strategy=video.get("segment_strategy", "generic content"),
            )

            # Compute context hash
            context_hash = compute_context_hash(
                segment_strategy=video.get("segment_strategy", "generic content"),
                channel=video["channel"],
                traffic_type=video["traffic_type"],
                funnel_stage=video["funnel_stage"],
                content_type=video.get("content_type", "video"),
            )

            # Create decision log
            decision = DecisionLog(
                asset_id=video["asset_id"],
                segment_strategy=video.get(
                    "segment_strategy", "imported from historical data"
                ),
                channel=video["channel"],
                traffic_type=video["traffic_type"],
                funnel_stage=video["funnel_stage"],
                content_type=video.get("content_type", "video"),
                context_hash=context_hash,
                structure_hash=structure.structure_hash,
                structure_features=structure.features,
                p_attention=None,
                p_persuasion=None,
                p_final=None,
                mode="exploration",  # Historical data treated as exploration
                context_confidence=0.0,
                exploration_weight=0.0,
                model_version=None,
                candidates_count=1,
            )
            db.add(decision)

            # Create performance log
            performance = PerformanceLog(
                asset_id=video["asset_id"],
                context_hash=context_hash,
                structure_hash=structure.structure_hash,
                traffic_type=video["traffic_type"],
                attention_score=video["attention_score"],
                persuasion_score=video["persuasion_score"],
                raw_metrics=video.get("raw_metrics"),
            )
            db.add(performance)

            stats["imported"] += 1

            # Commit in batches
            if (i + 1) % batch_size == 0:
                db.commit()
                print(f"  Imported {i + 1}/{stats['total']} videos...")

        except Exception as e:
            stats["errors"].append({"asset_id": video.get("asset_id"), "error": str(e)})
            db.rollback()

    # Final commit
    db.commit()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Bulk import videos for CVIE training")
    parser.add_argument("--input", "-i", required=True, help="Input file (CSV or JSON)")
    parser.add_argument(
        "--train", "-t", action="store_true", help="Train models after import"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=100, help="Batch size for commits"
    )

    args = parser.parse_args()

    # Determine file type
    filepath = args.input
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"Loading data from {filepath}...")
    if filepath.endswith(".csv"):
        data = load_csv(filepath)
    elif filepath.endswith(".json"):
        data = load_json(filepath)
    else:
        print("Error: File must be .csv or .json")
        sys.exit(1)

    print(f"Found {len(data)} videos to import")

    # Import into database
    print("Importing videos...")
    with get_db_context() as db:
        stats = import_videos(data, db, batch_size=args.batch_size)

        print(f"\nImport complete:")
        print(f"  Total: {stats['total']}")
        print(f"  Imported: {stats['imported']}")
        print(f"  Skipped (already exists): {stats['skipped']}")
        print(f"  Errors: {len(stats['errors'])}")

        if stats["errors"]:
            print("\nFirst 5 errors:")
            for err in stats["errors"][:5]:
                print(f"  - {err['asset_id']}: {err['error']}")

        # Train models if requested
        if args.train and stats["imported"] > 0:
            print("\nTraining models...")
            trainer = ModelTrainer(db)
            result = trainer.train_all_models()

            if result.get("success"):
                print(f"\nTraining complete!")
                print(f"  Version: {result['version']}")
                print(f"  Models:")
                for model_type, model_result in result["models"].items():
                    if model_result.get("success"):
                        print(
                            f"    - {model_type}: AUC={model_result.get('auc', 0):.3f}"
                        )
                    else:
                        print(
                            f"    - {model_type}: FAILED - {model_result.get('error')}"
                        )
            else:
                print(f"\nTraining failed: {result.get('error')}")


if __name__ == "__main__":
    main()
