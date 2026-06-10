"""
Logger module for structured logging and tracing
"""
import logging
import sys
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance configured for stderr output

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only add handler if not already configured
    if not logger.handlers:
        # Configure to use stderr
        handler = logging.StreamHandler(sys.stderr)

        # Use simple format for stderr
        formatter = logging.Formatter(
            '[%(levelname)s] %(name)s - %(message)s'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
