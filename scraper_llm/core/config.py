"""Configuration management for ScraperLLM."""
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, HttpUrl, validator
from pydantic_settings import BaseSettings
from pydantic.types import DirectoryPath, FilePath


class Settings(BaseSettings):
    """Application settings."""

    # Application settings
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # File paths
    BASE_DIR: DirectoryPath = Path(__file__).parent.parent.parent
    DATA_DIR: DirectoryPath = Field(
        default=Path("data").absolute(),
        env="DATA_DIR"
    )
    LOGS_DIR: DirectoryPath = Field(
        default=Path("logs").absolute(),
        env="LOGS_DIR"
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
    
    class Config:
        """Pydantic config."""
        
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("DATA_DIR", "LOGS_DIR", pre=True)
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
