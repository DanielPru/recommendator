"""
API Routes for CVIE service.
Uses asset_id as the primary lifecycle identifier.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import PerformanceLog, DecisionLog
from app.ml.model_manager import get_model_manager
from app.ml.trainer import ModelTrainer
from app.api.schemas import (
    RecommendRequest,
    RecommendResponse,
    PerformanceRequest,
    PerformanceResponse,
    RetrainRequest,
    RetrainResponse,
    HealthResponse,
    StructureFeatures,
)
from app.api.recommendation_service import RecommendationService


router = APIRouter()


@router.post("/recommend-structure", response_model=RecommendResponse)
async def recommend_structure(
    request: RecommendRequest,
    db: Session = Depends(get_db),
):
    """
    Get a recommended visual structure for an asset.

    The asset_id must be unique - if it already exists, returns 409 Conflict.

    The service will:
    1. Check if asset_id already exists (return 409 if so)
    2. Interpret context using heuristic rules
    3. Generate 50-100 candidate structures
    4. Score with ML models (if available)
    5. Apply 80/20 exploration policy
    6. Log the decision under asset_id
    7. Return the recommended structure
    """
    service = RecommendationService(db)

    # Check for existing asset_id
    if service.check_asset_exists(request.asset_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Asset '{request.asset_id}' already has a recommendation. Each asset can only have one recommendation.",
        )

    result = service.recommend(
        asset_id=request.asset_id,
        segment_strategy=request.segment_strategy,
        channel=request.channel,
        traffic_type=request.traffic_type.value,
        funnel_stage=request.funnel_stage.value,
        content_type=request.content_type.value,
    )

    return RecommendResponse(
        asset_id=result["asset_id"],
        structure_hash=result["structure_hash"],
        structure=StructureFeatures(**result["structure"]),
        p_attention=result["p_attention"],
        p_persuasion=result["p_persuasion"],
        p_final=result["p_final"],
        mode=result["mode"],
        context_confidence=result["context_confidence"],
        exploration_weight=result["exploration_weight"],
        model_version=result["model_version"],
        candidates_evaluated=result["candidates_evaluated"],
        context_hash=result["context_hash"],
    )


@router.post("/ingest-performance", response_model=PerformanceResponse)
async def ingest_performance(
    request: PerformanceRequest,
    db: Session = Depends(get_db),
):
    """
    Ingest performance feedback for a previously recommended asset.

    This data is used to train/retrain ML models.
    """
    # Validate asset exists
    decision = (
        db.query(DecisionLog).filter(DecisionLog.asset_id == request.asset_id).first()
    )

    if not decision:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{request.asset_id}' not found. Ensure the asset has a recommendation before ingesting performance.",
        )

    # Create performance log linked via asset_id
    performance = PerformanceLog(
        asset_id=request.asset_id,
        context_hash=decision.context_hash,
        structure_hash=decision.structure_hash,
        traffic_type=decision.traffic_type,
        attention_score=request.attention_score,
        persuasion_score=request.persuasion_score,
        raw_metrics=request.raw_metrics,
    )

    db.add(performance)

    # Update structure stats with performance data
    update_query = text("""
        UPDATE structure_stats SET
            total_performances = total_performances + 1,
            avg_attention = COALESCE(
                (avg_attention * total_performances + :attention) / (total_performances + 1),
                :attention
            ),
            avg_persuasion = COALESCE(
                (avg_persuasion * total_performances + :persuasion) / (total_performances + 1),
                :persuasion
            ),
            last_updated = NOW()
        WHERE structure_hash = :hash
    """)

    db.execute(
        update_query,
        {
            "hash": decision.structure_hash,
            "attention": request.attention_score,
            "persuasion": request.persuasion_score,
        },
    )

    db.commit()
    db.refresh(performance)

    return PerformanceResponse(
        success=True,
        asset_id=request.asset_id,
        performance_id=str(performance.id),
        message="Performance data ingested successfully",
    )


@router.post("/retrain-model", response_model=RetrainResponse)
async def retrain_model(
    request: RetrainRequest = RetrainRequest(),
    db: Session = Depends(get_db),
):
    """
    Trigger model retraining.

    This will:
    1. Fetch all decision + performance data (joined via asset_id)
    2. Compute 75th percentile thresholds
    3. Train 4 LightGBM classifiers
    4. Save models to disk
    5. Register in model_registry
    6. Hot-reload models
    """
    trainer = ModelTrainer(db)
    result = trainer.train_all_models()

    if not result.get("success"):
        return RetrainResponse(
            success=False,
            error=result.get("error", "Training failed"),
        )

    return RetrainResponse(
        success=True,
        version=result["version"],
        models=result["models"],
        thresholds=result["thresholds"],
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.

    Returns 503 if database is not connected.
    Returns 'degraded' if models are not trained (uses heuristics only).
    """
    model_manager = get_model_manager()

    # Check database connection
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        pass

    models_loaded = model_manager.is_ready
    model_version = model_manager.current_version

    # Determine overall status
    if not db_connected:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        status_str = "unhealthy"
    elif not models_loaded:
        # Allow service to run without models (uses heuristics only)
        status_code = status.HTTP_200_OK
        status_str = "degraded"
    else:
        status_code = status.HTTP_200_OK
        status_str = "healthy"

    response = HealthResponse(
        status=status_str,
        models_loaded=models_loaded,
        model_version=model_version,
        database_connected=db_connected,
    )

    if status_code != status.HTTP_200_OK:
        raise HTTPException(status_code=status_code, detail=response.model_dump())

    return response
