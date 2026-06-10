"""
Logger module for structured logging and tracing
"""
import logging
import sys
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON Lines format."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


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


def get_trace_logger() -> logging.Logger:
    """
    Get a logger configured for JSON Lines trace output.

    Returns:
        Logger configured to write to logs/traces.jsonl
    """
    logger = logging.getLogger("trace_logger")

    # Only configure if not already done
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # File handler for JSON Lines
        file_handler = logging.FileHandler("logs/traces.jsonl", mode="a")
        json_formatter = JSONFormatter()
        file_handler.setFormatter(json_formatter)

        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    return logger


def write_trace(trace_dict: Dict[str, Any]) -> None:
    """
    Write a trace dictionary to logs/traces.jsonl.

    Args:
        trace_dict: Dictionary representation of a trace (from TraceContext.to_dict())
    """
    os.makedirs("logs", exist_ok=True)

    with open("logs/traces.jsonl", "a", encoding="utf-8") as f:
        json.dump(trace_dict, f, ensure_ascii=False)
        f.write("\n")

