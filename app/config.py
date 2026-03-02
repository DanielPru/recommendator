"""
Application configuration using pydantic-settings.
Handles Supabase connection strings and model paths.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",),  # Fix pydantic warning
    )

    # Database URLs (required - must be set in Railway)
    database_url: str = ""
    database_url_direct: Optional[str] = None

    # Model configuration (renamed to avoid pydantic namespace conflict)
    ml_model_dir: str = "./models"
    ml_model_version: str = "v1"

    # Server configuration
    port: int = 8000
    env: str = "production"

    # Exploration settings
    exploration_ratio: float = 0.2

    # Structure generation
    min_candidates: int = 50
    max_candidates: int = 100
    diversity_ratio: float = 0.2  # 20% uniform sampling for diversity

    @property
    def database_url_sync(self) -> str:
        """Return sync database URL for SQLAlchemy."""
        return self.database_url

    @property
    def migration_database_url(self) -> str:
        """Return direct connection URL for Alembic migrations."""
        return self.database_url_direct or self.database_url

    # Aliases for backward compatibility
    @property
    def model_dir(self) -> str:
        return self.ml_model_dir

    @property
    def model_version(self) -> str:
        return self.ml_model_version


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    settings = Settings()
    if not settings.database_url:
        raise ValueError(
            "DATABASE_URL environment variable is required. "
            "Set it in Railway: Settings -> Variables -> DATABASE_URL"
        )
    return settings
