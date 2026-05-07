"""
Application configuration using Pydantic Settings.
Reads from environment variables / .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All app configuration — loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Anthropic
    ANTHROPIC_API_KEY: str

    # App
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True

    # RAG Pipeline
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5
    MODEL_NAME: str = "claude-3-5-sonnet-20241022"

    # Storage paths
    VECTORSTORE_PATH: str = "./data/vectorstore"
    DATA_RAW_PATH: str = "./data/raw"

    # Logging
    LOG_LEVEL: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    """
    Return cached Settings instance.
    Uses lru_cache so the .env file is only read once per process.
    """
    return Settings()
