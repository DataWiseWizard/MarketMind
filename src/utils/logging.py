"""
MarketMind Logging Configuration

Centralized logging setup using loguru.
"""

import sys

from loguru import logger

from src.utils.config import settings


def setup_logging(level: str = "INFO"):
    """
    Configure application logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance.
    """
    # Remove default handler
    logger.remove()

    # Console handler (colored)
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler (rotated daily, compressed)
    log_dir = settings.ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        str(log_dir / "marketmind_{time:YYYY-MM-DD}.log"),
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level:<8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )

    return logger
