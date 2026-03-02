# CVIE Integration Guide

## Overview

Integrate with the **Creative Visual Intelligence Engine (CVIE)** microservice to get AI-powered visual structure recommendations before creating marketing content.

**Base URL:** `https://YOUR-RAILWAY-URL` (replace with your actual Railway deployment URL)

---

## API Endpoints

### 1. POST /recommend-structure

Get a recommended visual structure before creating a video/image.

**Request:**
```json
{
  "asset_id": "string (required, unique)",
  "channel": "string (required)",
  "traffic_type": "string (required)",
  "funnel_stage": "string (required)",
  "content_type": "string (required)",
  "segment_strategy": "string (required)"
}
```

**Field Values:**

| Field | Allowed Values |
|-------|----------------|
| `asset_id` | Any unique string (e.g., `"video_20240301_abc123"`) |
| `channel` | `"tiktok"`, `"instagram_feed"`, `"instagram_reels"`, `"youtube"`, `"facebook"`, `"linkedin"` |
| `traffic_type` | `"organic"`, `"paid"` |
| `funnel_stage` | `"TOFU"`, `"MOFU"`, `"BOFU"` |
| `content_type` | `"video"`, `"image"` |
| `segment_strategy` | Free text describing target audience (e.g., `"Bold authentic UGC for Gen Z"`) |

**Response:**
```json
{
  "asset_id": "video_20240301_abc123",
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

**Error Responses:**
- `409 Conflict`: asset_id already exists
- `422 Validation Error`: Invalid request fields

---

### 2. POST /ingest-performance

Send performance metrics after content has been published (24-48h later).

**Request:**
```json
{
  "asset_id": "string (required, must match previous recommendation)",
  "attention_score": "float (required, 0.0-1.0)",
  "persuasion_score": "float (required, 0.0-1.0)",
  "raw_metrics": "object (optional)"
}
```

**Response:**
```json
{
  "success": true,
  "asset_id": "video_20240301_abc123",
  "performance_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Performance data ingested successfully"
}
```

**Error Responses:**
- `404 Not Found`: asset_id doesn't exist

---

### 3. POST /retrain-model

Trigger model retraining (call weekly after collecting 50+ samples).

**Request:**
```json
{
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "version": "v1_20240301_143022",
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

---

### 4. GET /health

Check service health.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_version": "v1_20240301_143022",
  "database_connected": true
}
```

---

## Structure Fields Explained

The `structure` object contains 9 visual dimensions to guide creative production:

| Field | Values | Meaning |
|-------|--------|---------|
| `primary_subject_area_ratio` | `"<20"`, `"20-40"`, `"40-60"`, `">60"` | How much of frame the main subject occupies |
| `focal_point_count` | `"1"`, `"2"`, `"3"` | Number of visual focal points |
| `figure_background_contrast` | `"low"`, `"medium"`, `"high"` | Contrast between subject and background |
| `motion_intensity` | `"static"`, `"implied_motion"`, `"low"`, `"medium"`, `"high"` | Amount of movement |
| `text_coverage_ratio` | `"0"`, `"<15"`, `"15-30"`, `">30"` | Percentage of frame with text overlay |
| `face_presence_scale` | `"none"`, `"small"`, `"medium"`, `"dominant"` | How prominent faces are |
| `product_visibility_ratio` | `"none"`, `"subtle"`, `"clear"`, `"dominant"` | How visible the product is |
| `offer_visual_salience` | `"none"`, `"subtle"`, `"clear"`, `"dominant"` | How prominent offers/CTAs are |
| `visual_complexity` | `"minimal"`, `"moderate"`, `"busy"` | Overall visual complexity |

---

## Integration Code

### Python

```python
import requests
from typing import Optional, Dict, Any

class CVIEClient:
    """Client for CVIE microservice."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
    
    def get_recommendation(
        self,
        asset_id: str,
        channel: str,
        traffic_type: str,
        funnel_stage: str,
        content_type: str,
        segment_strategy: str
    ) -> Dict[str, Any]:
        """
        Get visual structure recommendation for new content.
        
        Args:
            asset_id: Unique identifier for this asset
            channel: Platform (tiktok, instagram_feed, etc.)
            traffic_type: "organic" or "paid"
            funnel_stage: "TOFU", "MOFU", or "BOFU"
            content_type: "video" or "image"
            segment_strategy: Description of target audience
            
        Returns:
            Recommendation response with structure
        """
        response = requests.post(
            f"{self.base_url}/recommend-structure",
            json={
                "asset_id": asset_id,
                "channel": channel,
                "traffic_type": traffic_type,
                "funnel_stage": funnel_stage,
                "content_type": content_type,
                "segment_strategy": segment_strategy
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def send_performance(
        self,
        asset_id: str,
        attention_score: float,
        persuasion_score: float,
        raw_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send performance metrics for a previously recommended asset.
        
        Args:
            asset_id: Asset ID from original recommendation
            attention_score: Normalized attention metric (0.0-1.0)
            persuasion_score: Normalized persuasion metric (0.0-1.0)
            raw_metrics: Optional dict with raw platform metrics
            
        Returns:
            Confirmation response
        """
        payload = {
            "asset_id": asset_id,
            "attention_score": attention_score,
            "persuasion_score": persuasion_score
        }
        if raw_metrics:
            payload["raw_metrics"] = raw_metrics
            
        response = requests.post(
            f"{self.base_url}/ingest-performance",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def retrain_models(self, force: bool = False) -> Dict[str, Any]:
        """
        Trigger model retraining.
        
        Args:
            force: Force retrain even with limited data
            
        Returns:
            Training results with model metrics
        """
        response = requests.post(
            f"{self.base_url}/retrain-model",
            json={"force": force},
            timeout=300  # Training can take a few minutes
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()


# Usage example
if __name__ == "__main__":
    cvie = CVIEClient("https://YOUR-RAILWAY-URL")
    
    # Check health
    health = cvie.health_check()
    print(f"Service status: {health['status']}")
    
    # Get recommendation
    recommendation = cvie.get_recommendation(
        asset_id="video_20240301_001",
        channel="tiktok",
        traffic_type="organic",
        funnel_stage="TOFU",
        content_type="video",
        segment_strategy="Bold authentic UGC for Gen Z audience"
    )
    print(f"Recommended structure: {recommendation['structure']}")
    
    # Later, after collecting metrics...
    performance = cvie.send_performance(
        asset_id="video_20240301_001",
        attention_score=0.72,
        persuasion_score=0.45,
        raw_metrics={"views": 15000, "ctr": 0.028}
    )
    print(f"Performance logged: {performance['success']}")
```

---

### TypeScript/JavaScript

```typescript
interface RecommendRequest {
  asset_id: string;
  channel: string;
  traffic_type: "organic" | "paid";
  funnel_stage: "TOFU" | "MOFU" | "BOFU";
  content_type: "video" | "image";
  segment_strategy: string;
}

interface Structure {
  primary_subject_area_ratio: string;
  focal_point_count: string;
  figure_background_contrast: string;
  motion_intensity: string;
  text_coverage_ratio: string;
  face_presence_scale: string;
  product_visibility_ratio: string;
  offer_visual_salience: string;
  visual_complexity: string;
}

interface RecommendResponse {
  asset_id: string;
  structure_hash: string;
  structure: Structure;
  p_attention: number | null;
  p_persuasion: number | null;
  p_final: number | null;
  mode: "exploration" | "exploitation";
  context_confidence: number;
  exploration_weight: number;
  model_version: string | null;
  candidates_evaluated: number;
  context_hash: string;
}

interface PerformanceRequest {
  asset_id: string;
  attention_score: number;
  persuasion_score: number;
  raw_metrics?: Record<string, any>;
}

interface PerformanceResponse {
  success: boolean;
  asset_id: string;
  performance_id: string;
  message: string;
}

interface HealthResponse {
  status: string;
  models_loaded: boolean;
  model_version: string | null;
  database_connected: boolean;
}

class CVIEClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async getRecommendation(request: RecommendRequest): Promise<RecommendResponse> {
    const response = await fetch(`${this.baseUrl}/recommend-structure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to get recommendation");
    }

    return response.json();
  }

  async sendPerformance(request: PerformanceRequest): Promise<PerformanceResponse> {
    const response = await fetch(`${this.baseUrl}/ingest-performance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to send performance");
    }

    return response.json();
  }

  async retrainModels(force: boolean = false): Promise<any> {
    const response = await fetch(`${this.baseUrl}/retrain-model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to retrain models");
    }

    return response.json();
  }

  async healthCheck(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`);

    if (!response.ok) {
      throw new Error("Health check failed");
    }

    return response.json();
  }
}

// Usage example
const cvie = new CVIEClient("https://YOUR-RAILWAY-URL");

// Get recommendation
const recommendation = await cvie.getRecommendation({
  asset_id: "video_20240301_001",
  channel: "tiktok",
  traffic_type: "organic",
  funnel_stage: "TOFU",
  content_type: "video",
  segment_strategy: "Bold authentic UGC for Gen Z audience",
});

console.log("Recommended structure:", recommendation.structure);

// Later, send performance
const performance = await cvie.sendPerformance({
  asset_id: "video_20240301_001",
  attention_score: 0.72,
  persuasion_score: 0.45,
  raw_metrics: { views: 15000, ctr: 0.028 },
});

console.log("Performance logged:", performance.success);
```

---

## Score Normalization Guide

Before sending performance data, normalize your metrics to 0.0-1.0 scale:

### Attention Score

```python
# Option 1: View rate
attention_score = views / impressions

# Option 2: Completion rate (video)
attention_score = avg_watch_time / video_duration

# Option 3: Percentile within your dataset
attention_score = (views - min_views) / (max_views - min_views)
```

### Persuasion Score

```python
# Option 1: CTR
persuasion_score = clicks / impressions

# Option 2: Engagement rate
persuasion_score = (likes + comments + shares) / views

# Option 3: Conversion rate
persuasion_score = conversions / clicks
```

---

## Integration Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     INTEGRATION WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. User initiates content creation                             │
│     │                                                           │
│     ▼                                                           │
│  2. Generate unique asset_id                                    │
│     │   asset_id = f"video_{datetime}_{uuid}"                  │
│     │                                                           │
│     ▼                                                           │
│  3. POST /recommend-structure                                   │
│     │   → Get recommended visual structure                      │
│     │                                                           │
│     ▼                                                           │
│  4. Display structure guidelines to creator                     │
│     │   - "Use high motion"                                    │
│     │   - "Face should be medium prominence"                   │
│     │   - "Keep text coverage under 15%"                       │
│     │                                                           │
│     ▼                                                           │
│  5. Creator produces content following guidelines               │
│     │                                                           │
│     ▼                                                           │
│  6. Publish to platform (TikTok, Instagram, etc.)              │
│     │                                                           │
│     ▼                                                           │
│  7. Store asset_id with content record                         │
│     │                                                           │
│     ▼                                                           │
│  8. [CRON JOB] After 24-48h, fetch platform metrics            │
│     │                                                           │
│     ▼                                                           │
│  9. Normalize metrics to attention/persuasion scores           │
│     │                                                           │
│     ▼                                                           │
│  10. POST /ingest-performance                                   │
│     │                                                           │
│     ▼                                                           │
│  11. [WEEKLY] POST /retrain-model                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

```python
from requests.exceptions import HTTPError

try:
    recommendation = cvie.get_recommendation(...)
except HTTPError as e:
    if e.response.status_code == 409:
        # Asset already exists - retrieve existing recommendation
        print(f"Asset {asset_id} already has a recommendation")
    elif e.response.status_code == 422:
        # Validation error
        print(f"Invalid request: {e.response.json()}")
    else:
        raise
```

```typescript
try {
  const recommendation = await cvie.getRecommendation(request);
} catch (error) {
  if (error.message.includes("already has a recommendation")) {
    // Asset already exists
    console.log(`Asset ${assetId} already has a recommendation`);
  } else {
    throw error;
  }
}
```

---

## Environment Variables

Add to your app's environment:

```env
CVIE_API_URL=https://YOUR-RAILWAY-URL
```

---

## Testing

```bash
# Health check
curl https://YOUR-RAILWAY-URL/health

# Test recommendation
curl -X POST https://YOUR-RAILWAY-URL/recommend-structure \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "test_001",
    "channel": "tiktok",
    "traffic_type": "organic",
    "funnel_stage": "TOFU",
    "content_type": "video",
    "segment_strategy": "Test content"
  }'
```
