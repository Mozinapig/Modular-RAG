"""
Logger module for structured logging and tracing
"""
import logging
import sys
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON Lines format."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)

        # Include all extra fields added to the record
        # (skip standard LogRecord attributes)
        standard_attrs = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName', 'levelname',
            'levelno', 'lineno', 'module', 'msecs', 'pathname', 'process', 'processName',
            'relativeCreated', 'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
            'getMessage', 'message'
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_'):
                try:
                    # Try to serialize the value to JSON
                    json.dumps(value)
                    log_obj[key] = value
                except (TypeError, ValueError):
                    # Skip non-JSON-serializable values
                    pass

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


def get_trace_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger configured for JSON Lines trace output.

    Args:
        log_file: Optional path to trace log file. Defaults to logs/traces.jsonl
        level: Logging level (default: logging.INFO)

    Returns:
        Logger configured to write JSON Lines
    """
    logger = logging.getLogger("trace_logger")

    # Only configure if not already done
    if not logger.handlers:
        if log_file is None:
            log_file = "logs/traces.jsonl"

        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file) or "logs"
        os.makedirs(log_dir, exist_ok=True)

        # File handler for JSON Lines
        file_handler = logging.FileHandler(log_file, mode="a")
        json_formatter = JSONFormatter()
        file_handler.setFormatter(json_formatter)

        logger.addHandler(file_handler)
        logger.setLevel(level)

    return logger


def write_trace(trace_dict: Dict[str, Any], filepath: Optional[str] = None) -> None:
    """
    Write a trace dictionary to trace log file in JSON Lines format.

    Args:
        trace_dict: Dictionary representation of a trace (from TraceContext.to_dict())
        filepath: Optional path to trace log file. Defaults to logs/traces.jsonl
    """
    if filepath is None:
        filepath = "logs/traces.jsonl"

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "a", encoding="utf-8") as f:
        json.dump(trace_dict, f, ensure_ascii=False)
        f.write("\n")

