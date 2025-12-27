"""
Backend configuration for Medical Chatbot API.

Reuses the main settings.py from the project root and adds backend-specific settings.
"""

import sys
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Add parent directory to path to import main settings
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import get_settings as get_main_settings


class BackendSettings(BaseSettings):
    """
    Backend-specific settings that extend the main application settings.

    These settings are specific to the FastAPI backend service.
    """

    # Vector Database
    VECTOR_DB_PATH: str = "./vector_db"
    VECTOR_DB_COLLECTION_NAME: str = "medical_services"

    # Rate Limiting
    MAX_CONCURRENT_OPENAI_CALLS: int = 10

    # Conversation Settings
    MAX_CONVERSATION_HISTORY: int = 15  # Keep last 15 messages

    # Retrieval Settings
    RAG_TOP_K: int = 5  # Number of chunks to retrieve

    # API Settings
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["http://localhost:8501", "http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# Singleton instances
_main_settings: Optional[object] = None
_backend_settings: Optional[BackendSettings] = None


def get_settings():
    """
    Get the main application settings (from root settings.py).

    Returns:
        Settings instance with Azure credentials
    """
    global _main_settings
    if _main_settings is None:
        _main_settings = get_main_settings()
    return _main_settings


def get_backend_settings() -> BackendSettings:
    """
    Get backend-specific settings.

    Returns:
        BackendSettings instance
    """
    global _backend_settings
    if _backend_settings is None:
        _backend_settings = BackendSettings()
    return _backend_settings
