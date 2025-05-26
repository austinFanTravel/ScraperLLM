"""Logging configuration for ScraperLLM."""
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Configure logging for the application.

    Args:
        log_level: The log level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file. If None, logs will only go to stderr
        rotation: Log rotation configuration (e.g., "10 MB", "1 day")
        retention: Log retention period (e.g., "30 days")
    """
    # Remove default handler
    logger.remove()

    # Add stderr handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Add file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            rotation=rotation,
            retention=retention,
            level=log_level,
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}",
        )

    # Set log level from environment variable if not provided
    env_log_level = os.getenv("LOG_LEVEL")
    if env_log_level:
        log_level = env_log_level.upper()
    
    logger.info(f"Logging configured with level: {log_level}")
    if log_file:
        logger.info(f"Logging to file: {os.path.abspath(log_file)}")
