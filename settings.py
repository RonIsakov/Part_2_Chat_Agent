"""
Configuration management using Pydantic Settings.
Automatically loads Azure credentials from .env file in the project root.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings that automatically load from .env file.

    The .env file should contain:
        AZURE_DI_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
        AZURE_DI_KEY=your_key_here
        AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
        AZURE_OPENAI_KEY=your_key_here
    """

    # Azure Document Intelligence Configuration
    AZURE_DI_ENDPOINT: str
    AZURE_DI_KEY: str

    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_KEY: str
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4o"

    # Application Settings
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 10

    # Directory paths
    DATA_INPUT_DIR: str = "data/input"
    DATA_OUTPUT_DIR: str = "data/output"
    LOGS_DIR: str = "logs"

    model_config = SettingsConfigDict(
        env_file=".env",  # Automatically reads from .env file
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the settings singleton instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
