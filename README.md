# Creative Visual Intelligence Engine (CVIE)

Production-grade microservice for recommending optimal visual creative structures before asset generation.

## Features

- **Deterministic Heuristic Engine**: Rule-based context interpretation without LLM dependency
- **Weighted Structure Generation**: 50-100 unique candidates per request with diversity preservation
- **4 LightGBM Classifiers**: organic_attention, organic_persuasion, paid_attention, paid_persuasion
- **80/20 Exploration Policy**: Balances exploitation with novelty-based exploration
- **Model Versioning**: Full registry with hot-reload capability
- **Supabase PostgreSQL**: Production-ready with connection pooling

## Architecture

```
cvie-service/
├── app/
│   ├── api/              # FastAPI routes and schemas
│   ├── core/             # Feature schema, interpreter, generator
│   ├── db/               # SQLAlchemy models and database config
│   └── ml/               # LightGBM models, trainer, exploration
├── alembic/              # Database migrations
├── models/               # Trained model storage
├── Dockerfile
└── requirements.txt
```

## Asset ID Lifecycle

CVIE uses `asset_id` as the primary lifecycle identifier. The asset ID is generated **upstream** (by your content management system) before calling CVIE:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ASSET LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Upstream System generates asset_id                          │
│     └─> asset_id = "asset_20240115_campaign_abc123"            │
│                                                                 │
│  2. POST /recommend-structure(asset_id, context...)            │
│     └─> Returns recommended visual structure                    │
│     └─> Stores decision with asset_id as PK                    │
│                                                                 │
│  3. Generate visual asset using structure                       │
│     └─> Your creative pipeline builds the asset                │
│                                                                 │
│  4. Publish asset to channel                                    │
│     └─> TikTok, Instagram, YouTube, etc.                       │
│                                                                 │
│  5. Collect performance metrics                                 │
│     └─> Cron job fetches metrics from ad platforms             │
│                                                                 │
│  6. POST /ingest-performance(asset_id, scores)                 │
│     └─> Links metrics back to original decision                │
│                                                                 │
│  7. POST /retrain-model                                         │
│     └─> Uses asset_id joins to train new models                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Feature Schema V1

9 visual dimensions with frozen categorical values:

| Feature | Values |
|---------|--------|
| primary_subject_area_ratio | <20, 20-40, 40-60, >60 |
| focal_point_count | 1, 2, 3 |
| figure_background_contrast | low, medium, high |
| motion_intensity | low, medium, high, static, implied_motion |
| text_coverage_ratio | 0, <15, 15-30, >30 |
| face_presence_scale | none, small, medium, dominant |
| product_visibility_ratio | none, subtle, clear, dominant |
| offer_visual_salience | none, subtle, clear, dominant |
| visual_complexity | minimal, moderate, busy |

## Setup

### Prerequisites

- Python 3.11+
- Supabase project with PostgreSQL
- Railway account (for deployment)

### Local Development

1. **Clone and setup environment**
```bash
git clone <repo>
cd cvie-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your Supabase credentials
```

3. **Enable pgcrypto in Supabase**

Run this SQL in your Supabase SQL editor:
```sql
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
```

4. **Run migrations**
```bash
alembic upgrade head
```

5. **Start server**
```bash
uvicorn app.main:app --reload
```

### Railway Deployment

1. Connect your GitHub repository to Railway
2. Set environment variables:
   - `DATABASE_URL` - Pooled connection (port 6543)
   - `DATABASE_URL_DIRECT` - Direct connection (port 5432)
   - `MODEL_DIR` - `/app/models`
3. Deploy

## API Endpoints

### POST /recommend-structure

Get a visual structure recommendation.

**Request:**
```json
{
  "asset_id": "asset_20240115_campaign_abc123",
  "segment_strategy": "Target short-form video consumers who respond to authentic UGC content",
  "channel": "tiktok",
  "traffic_type": "organic",
  "funnel_stage": "TOFU",
  "content_type": "video"
}
```

**Response:**
```json
{
  "asset_id": "asset_20240115_campaign_abc123",
  "structure_hash": "a1b2c3d4e5f6g7h8",
  "structure": {
    "primary_subject_area_ratio": ">60",
    "focal_point_count": "1",
    "figure_background_contrast": "high",
    "motion_intensity": "high",
    "text_coverage_ratio": "<15",
    "face_presence_scale": "medium",
    "product_visibility_ratio": "subtle",
    "offer_visual_salience": "none",
    "visual_complexity": "minimal"
  },
  "p_attention": null,
  "p_persuasion": null,
  "p_final": null,
  "mode": "exploration",
  "context_confidence": 0.0,
  "exploration_weight": 0.18,
  "model_version": null,
  "candidates_evaluated": 73,
  "context_hash": "abc123..."
}
```

**Error Response (409 Conflict):**
```json
{
  "detail": "asset_id 'asset_20240115_campaign_abc123' already has a recommendation"
}
```

### POST /ingest-performance

Log performance feedback for training data.

**Request:**
```json
{
  "asset_id": "asset_20240115_campaign_abc123",
  "attention_score": 0.75,
  "persuasion_score": 0.82,
  "raw_metrics": {
    "view_rate": 0.65,
    "ctr": 0.032
  }
}
```

**Response:**
```json
{
  "success": true,
  "asset_id": "asset_20240115_campaign_abc123",
  "performance_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Performance logged successfully"
}
```

### POST /retrain-model

Trigger model retraining with collected data.

**Response:**
```json
{
  "success": true,
  "version": "v1_20240115_143022",
  "models": {
    "organic_attention": {"success": true, "auc": 0.72},
    "organic_persuasion": {"success": true, "auc": 0.68},
    "paid_attention": {"success": true, "auc": 0.74},
    "paid_persuasion": {"success": true, "auc": 0.71}
  },
  "thresholds": {
    "organic": {"attention": 0.65, "persuasion": 0.58},
    "paid": {"attention": 0.71, "persuasion": 0.63}
  }
}
```

### GET /health

Health check endpoint.

## Heuristic Rules

### By Content Type
- **Image**: motion limited to static/implied_motion, bias toward minimal complexity
- **Video**: all motion levels allowed, bias toward medium/high motion

### By Channel
- **TikTok/Reels**: high motion, high contrast, large subject area
- **Instagram Feed**: single focal point, minimal complexity
- **YouTube**: moderate text, face presence
- **LinkedIn**: minimal complexity, professional aesthetic

### By Traffic Type
- **Paid**: emphasize product visibility and offer salience
- **Organic**: emphasize face presence and authenticity

### By Funnel Stage
- **TOFU**: high contrast, no hard sell
- **MOFU**: clear product visibility, educational text
- **BOFU**: dominant offers, clear CTAs

### By Segment Keywords
Keywords like "authentic", "bold", "ugc", "educational" trigger specific biases.

## Bootstrap Strategy

When starting with zero data:

1. Service runs using heuristics only (no ML scores)
2. 100% exploration mode due to maximum uncertainty
3. Collect performance feedback via `/ingest-performance`
4. After 50+ samples per traffic type, run `/retrain-model`
5. Service automatically hot-reloads new models

## Database Schema

- `decision_logs` - Every recommendation made (PK: `asset_id`)
- `performance_logs` - Outcome feedback (FK: `asset_id` -> decision_logs)
- `context_stats` - Aggregated context metrics
- `structure_stats` - Aggregated structure metrics
- `model_registry` - Trained model versions

## Model Scoring

```
p_final = p_attention x p_persuasion
```

Labels are top 25% (75th percentile) performers per traffic type.

## License

Proprietary
