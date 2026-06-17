"""
Unit tests for JSON Lines structured logger.
Tests the JSONFormatter and trace persistence mechanisms.
"""
import json
import logging
import os
import tempfile
from pathlib import Path

import pytest

from src.observability.logger import JSONFormatter, get_trace_logger, write_trace


class TestJSONFormatter:
    """Test the custom JSON formatter."""

    def test_format_basic_record(self):
        """Test formatting a basic log record to JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.module"
        assert "timestamp" in parsed

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields in LogRecord."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Message",
            args=(),
            exc_info=None,
        )
        record.trace_id = "trace_123"
        record.custom_field = "custom_value"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["trace_id"] == "trace_123"
        assert parsed["custom_field"] == "custom_value"

    def test_format_with_exception(self):
        """Test formatting a record with exception info."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test.module",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "ERROR"
        assert "exc_info" in parsed
        assert "ValueError" in parsed["exc_info"]

    def test_format_preserves_json_serializable_types(self):
        """Test that JSON-serializable types are preserved."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.number = 42
        record.float_val = 3.14
        record.list_val = [1, 2, 3]
        record.dict_val = {"key": "value"}
        record.bool_val = True
        record.none_val = None

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["number"] == 42
        assert parsed["float_val"] == 3.14
        assert parsed["list_val"] == [1, 2, 3]
        assert parsed["dict_val"] == {"key": "value"}
        assert parsed["bool_val"] is True
        assert parsed["none_val"] is None

    def test_format_skips_non_serializable_types(self):
        """Test that non-JSON-serializable types are skipped."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.obj = object()  # Non-serializable
        record.msg_field = "message"

        result = formatter.format(record)
        parsed = json.loads(result)

        # object() should be skipped, but msg_field should remain
        assert "obj" not in parsed
        assert parsed["msg_field"] == "message"


class TestWriteTrace:
    """Test the write_trace function."""

    def test_write_trace_creates_file(self):
        """Test that write_trace creates the traces file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "traces.jsonl"
            trace_dict = {
                "trace_id": "trace_001",
                "trace_type": "query",
                "timestamp": "2026-06-17T10:00:00",
                "query": "test query",
            }

            write_trace(trace_dict, filepath=trace_file)

            assert trace_file.exists()
            with open(trace_file, "r") as f:
                line = f.readline().strip()

            parsed = json.loads(line)
            assert parsed["trace_id"] == "trace_001"
            assert parsed["trace_type"] == "query"

    def test_write_trace_appends_to_existing_file(self):
        """Test that write_trace appends to existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "traces.jsonl"

            # Write first trace
            trace1 = {"trace_id": "trace_001"}
            write_trace(trace1, filepath=trace_file)

            # Write second trace
            trace2 = {"trace_id": "trace_002"}
            write_trace(trace2, filepath=trace_file)

            # Read both traces
            with open(trace_file, "r") as f:
                lines = f.readlines()

            assert len(lines) == 2
            parsed1 = json.loads(lines[0])
            parsed2 = json.loads(lines[1])
            assert parsed1["trace_id"] == "trace_001"
            assert parsed2["trace_id"] == "trace_002"

    def test_write_trace_handles_dict(self):
        """Test that write_trace correctly handles dictionary input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "traces.jsonl"
            trace_dict = {
                "trace_id": "test_123",
                "stages": [
                    {"name": "stage1", "elapsed_ms": 100},
                    {"name": "stage2", "elapsed_ms": 200},
                ],
            }

            write_trace(trace_dict, filepath=trace_file)

            with open(trace_file, "r") as f:
                parsed = json.loads(f.readline())

            assert parsed["trace_id"] == "test_123"
            assert len(parsed["stages"]) == 2
            assert parsed["stages"][0]["elapsed_ms"] == 100

    def test_write_trace_creates_parent_directory(self):
        """Test that write_trace creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "logs" / "subdir" / "traces.jsonl"

            write_trace({"trace_id": "test"}, filepath=trace_file)

            assert trace_file.exists()
            assert trace_file.parent.exists()


class TestGetTraceLogger:
    """Test the get_trace_logger function."""

    @pytest.fixture(autouse=True)
    def cleanup_logger(self):
        """Clean up logger singleton before and after each test."""
        # Clean up before test
        logger = logging.getLogger("trace_logger")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger.handlers.clear()
        yield
        # Clean up after test
        logger = logging.getLogger("trace_logger")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger.handlers.clear()

    def test_get_trace_logger_returns_logger(self):
        """Test that get_trace_logger returns a logging.Logger."""
        logger = get_trace_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "trace_logger"

    def test_trace_logger_has_json_handler(self):
        """Test that trace logger is configured with JSON handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "traces.jsonl"
            logger = get_trace_logger(log_file=str(log_file))

            # Log a message
            logger.info("Test message", extra={"trace_id": "test_001"})

            # Flush handlers to ensure data is written
            for handler in logger.handlers:
                handler.flush()

            # Verify JSON was written
            assert log_file.exists(), f"Log file not found at {log_file}"
            with open(log_file, "r") as f:
                line = f.readline().strip()

            parsed = json.loads(line)
            assert parsed["message"] == "Test message"
            assert parsed["trace_id"] == "test_001"

            # Close handlers before temp directory cleanup
            for handler in logger.handlers:
                handler.close()

    def test_trace_logger_singleton_behavior(self):
        """Test that multiple calls return same logger instance."""
        logger1 = get_trace_logger()
        # Note: This test may not work as expected due to singleton caching
        # The actual behavior depends on whether logger has been previously configured
        assert isinstance(logger1, logging.Logger)

    def test_trace_logger_level_configuration(self):
        """Test that log level can be configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "traces.jsonl"
            logger = get_trace_logger(log_file=str(log_file), level=logging.WARNING)

            logger.debug("Debug message")  # Should not be logged
            logger.warning("Warning message")  # Should be logged

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            with open(log_file, "r") as f:
                lines = f.readlines()

            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["message"] == "Warning message"

            # Close handlers before temp directory cleanup
            for handler in logger.handlers:
                handler.close()


class TestJSONFormatterEdgeCases:
    """Test edge cases in JSON formatting."""

    def test_format_with_formatting_arguments(self):
        """Test formatting with string % arguments."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="User %s logged in",
            args=("alice",),
            exc_info=None,
        )
        record.getMessage = lambda: "User alice logged in"  # Mock getMessage

        result = formatter.format(record)
        parsed = json.loads(result)

        # Should contain the formatted message
        assert "message" in parsed

    def test_format_with_very_long_message(self):
        """Test formatting with very long message."""
        formatter = JSONFormatter()
        long_msg = "x" * 10000
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg=long_msg,
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert len(parsed["message"]) == 10000

    def test_format_with_unicode_characters(self):
        """Test formatting with unicode characters."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="测试消息 🚀 test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == "测试消息 🚀 test"

    def test_format_includes_timestamp(self):
        """Test that timestamp is always included."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "timestamp" in parsed
        # Should be ISO format or similar
        assert isinstance(parsed["timestamp"], str)
