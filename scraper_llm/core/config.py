"""Configuration management for ScraperLLM."""
import os
from pathlib import Path
from typing import Optional, Literal

from pydantic import Field, HttpUrl, validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.types import DirectoryPath, FilePath


class Settings(BaseSettings):
    """Application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",  # Allow extra fields to be present in .env
    )

    # Application settings
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode",
        env="DEBUG"
    )
    APP_ENV: str = Field(
        default="development",
        description="Application environment (development, production, staging)",
        env="APP_ENV"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        env="LOG_LEVEL"
    )
    
    # File paths
    BASE_DIR: DirectoryPath = Path(__file__).parent.parent.parent
    DATA_DIR: DirectoryPath = Field(
        default=Path("data").absolute(),
        description="Base directory for data storage",
        env="DATA_DIR"
    )
    RAW_DATA_DIR: DirectoryPath = Field(
        default=Path("data/raw").absolute(),
        description="Directory for raw data",
        env="RAW_DATA_DIR"
    )
    PROCESSED_DATA_DIR: DirectoryPath = Field(
        default=Path("data/processed").absolute(),
        description="Directory for processed data",
        env="PROCESSED_DATA_DIR"
    )
    MODELS_DIR: DirectoryPath = Field(
        default=Path("data/models").absolute(),
        description="Directory for model files",
        env="MODELS_DIR"
    )
    LOGS_DIR: DirectoryPath = Field(
        default=Path("logs").absolute(),
        description="Directory for log files",
        env="LOGS_DIR"
    )
    
    # API Keys
    REDDIT_CLIENT_ID: Optional[str] = Field(
        default=None,
        description="Reddit API client ID",
        env="REDDIT_CLIENT_ID"
    )
    REDDIT_CLIENT_SECRET: Optional[str] = Field(
        default=None,
        description="Reddit API client secret",
        env="REDDIT_CLIENT_SECRET"
    )
    GOOGLE_API_KEY: Optional[str] = Field(
        default=None,
        description="Google API key",
        env="GOOGLE_API_KEY"
    )
    GETTY_IMAGES_API_KEY: Optional[str] = Field(
        default=None,
        description="Getty Images API key",
        env="GETTY_IMAGES_API_KEY"
    )
    SERPAPI_KEY: str = Field(
        default="",
        description="SerpAPI key for Google search results",
        env="SERPAPI_KEY"
    )
    
    # Model settings
    DEFAULT_MODEL_NAME: str = Field(
        default="all-mpnet-base-v2",
        description="Default model name for embeddings",
        env="DEFAULT_MODEL_NAME"
    )
    BATCH_SIZE: int = Field(
        default=32,
        description="Batch size for model inference",
        env="BATCH_SIZE"
    )
    MAX_SEQ_LENGTH: int = Field(
        default=512,
        description="Maximum sequence length for the model",
        env="MAX_SEQ_LENGTH"
    )
    
    # Search settings
    SEARCH_TIMEOUT: int = Field(
        default=30,
        description="Default timeout for search requests in seconds",
        env="SEARCH_TIMEOUT"
    )
    MAX_RESULTS: int = Field(
        default=50,
        description="Maximum number of results to return by default",
        env="MAX_RESULTS"
    )
    
    # NER settings
    NER_MODEL: str = Field(
        default="en_core_web_sm",
        description="SpaCy model to use for NER",
        env="NER_MODEL"
    )
    
    @field_validator("DATA_DIR", "LOGS_DIR", "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "MODELS_DIR", mode='before')
    @classmethod
    def ensure_directories_exist(cls, v):
        """Ensure that directories exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the settings instance.
    
    This function allows for dependency injection in FastAPI endpoints.
    """
    return settings
