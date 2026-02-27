"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.ml.model_manager import get_model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"Starting CVIE service (env={settings.env})")

    # Initialize model manager
    model_manager = get_model_manager()
    if model_manager.is_ready:
        print(f"Models loaded: version={model_manager.current_version}")
    else:
        print("No models loaded - service will use heuristics only")

    yield

    # Shutdown
    print("Shutting down CVIE service")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Creative Visual Intelligence Engine",
        description="ML-powered visual structure recommendation service",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router, tags=["CVIE"])

    return app


# Application instance
app = create_app()
