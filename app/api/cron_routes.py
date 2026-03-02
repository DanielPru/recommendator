"""
Scheduled tasks for CVIE service.
These endpoints can be triggered by Railway CRON or external schedulers.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta

from app.db.database import get_db
from app.ml.trainer import ModelTrainer

router = APIRouter(prefix="/cron", tags=["cron"])


@router.post("/check-retrain")
async def check_and_retrain(db: Session = Depends(get_db)):
    """
    Check if enough new data exists and trigger retraining if needed.

    Call this weekly via CRON.

    Conditions for retraining:
    - At least 50 new performance records since last training
    - At least 7 days since last model version
    """
    # Check last model training date
    result = db.execute(
        text("""
        SELECT MAX(created_at) as last_trained
        FROM model_registry
        WHERE is_active = true
    """)
    )
    row = result.fetchone()
    last_trained = row.last_trained if row else None

    # Check new performance records
    if last_trained:
        result = db.execute(
            text("""
            SELECT COUNT(*) as new_records
            FROM performance_logs
            WHERE created_at > :last_trained
        """),
            {"last_trained": last_trained},
        )
    else:
        result = db.execute(
            text("SELECT COUNT(*) as new_records FROM performance_logs")
        )

    new_records = result.fetchone().new_records

    # Decide if retraining is needed
    days_since_training = (
        (datetime.utcnow() - last_trained).days if last_trained else 999
    )
    should_retrain = new_records >= 50 or (
        new_records >= 20 and days_since_training >= 7
    )

    if not should_retrain:
        return {
            "retrained": False,
            "reason": f"Not enough data ({new_records} new records, {days_since_training} days since last training)",
            "new_records": new_records,
            "days_since_training": days_since_training,
        }

    # Trigger retraining
    trainer = ModelTrainer(db)
    result = trainer.train_all_models()

    return {"retrained": True, "new_records": new_records, "training_result": result}


@router.post("/cleanup-old-models")
async def cleanup_old_models(keep_versions: int = 5, db: Session = Depends(get_db)):
    """
    Remove old model versions from disk and database.

    Call this monthly via CRON.

    Args:
        keep_versions: Number of recent versions to keep
    """
    import os
    import shutil
    from app.config import get_settings
    from pathlib import Path

    settings = get_settings()
    model_dir = Path(settings.model_dir)

    # Get all versions ordered by date
    result = db.execute(
        text("""
        SELECT DISTINCT version, MIN(created_at) as created_at
        FROM model_registry
        GROUP BY version
        ORDER BY created_at DESC
    """)
    )
    versions = result.fetchall()

    if len(versions) <= keep_versions:
        return {
            "cleaned": False,
            "reason": f"Only {len(versions)} versions exist, keeping all",
            "versions_kept": len(versions),
        }

    # Versions to delete
    versions_to_delete = versions[keep_versions:]
    deleted = []

    for v in versions_to_delete:
        version = v.version

        # Delete from database
        db.execute(
            text("""
            DELETE FROM model_registry WHERE version = :version
        """),
            {"version": version},
        )

        # Delete from disk
        version_dir = model_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)

        deleted.append(version)

    db.commit()

    return {
        "cleaned": True,
        "deleted_versions": deleted,
        "versions_kept": keep_versions,
    }


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """
    Get current system statistics.

    Useful for monitoring dashboards.
    """
    # Total decisions
    result = db.execute(text("SELECT COUNT(*) as count FROM decision_logs"))
    total_decisions = result.fetchone().count

    # Total performance records
    result = db.execute(text("SELECT COUNT(*) as count FROM performance_logs"))
    total_performances = result.fetchone().count

    # Decisions by traffic type
    result = db.execute(
        text("""
        SELECT traffic_type, COUNT(*) as count
        FROM decision_logs
        GROUP BY traffic_type
    """)
    )
    by_traffic = {row.traffic_type: row.count for row in result.fetchall()}

    # Decisions by channel
    result = db.execute(
        text("""
        SELECT channel, COUNT(*) as count
        FROM decision_logs
        GROUP BY channel
        ORDER BY count DESC
        LIMIT 10
    """)
    )
    by_channel = {row.channel: row.count for row in result.fetchall()}

    # Recent activity (last 7 days)
    result = db.execute(
        text("""
        SELECT COUNT(*) as count
        FROM decision_logs
        WHERE created_at > NOW() - INTERVAL '7 days'
    """)
    )
    recent_decisions = result.fetchone().count

    result = db.execute(
        text("""
        SELECT COUNT(*) as count
        FROM performance_logs
        WHERE created_at > NOW() - INTERVAL '7 days'
    """)
    )
    recent_performances = result.fetchone().count

    # Current model info
    result = db.execute(
        text("""
        SELECT version, model_type, auc_score, created_at
        FROM model_registry
        WHERE is_active = true
        ORDER BY created_at DESC
    """)
    )
    active_models = [
        {
            "version": row.version,
            "model_type": row.model_type,
            "auc_score": row.auc_score,
        }
        for row in result.fetchall()
    ]

    return {
        "total_decisions": total_decisions,
        "total_performances": total_performances,
        "by_traffic_type": by_traffic,
        "by_channel": by_channel,
        "last_7_days": {
            "decisions": recent_decisions,
            "performances": recent_performances,
        },
        "active_models": active_models,
        "coverage_ratio": round(total_performances / total_decisions, 2)
        if total_decisions > 0
        else 0,
    }
