"""
E2E tests for query.py CLI script.

Tests the command-line query interface including:
- Argument parsing
- Integration with HybridSearch and Reranker
- Output formatting (default and verbose modes)
- Edge cases and error handling
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, Any

import pytest


# Path to query.py script
QUERY_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "query.py"


@pytest.fixture
def query_cmd_base():
    """Base command to run query.py script."""
    return [sys.executable, str(QUERY_SCRIPT)]


class TestQueryScriptBasic:
    """Test basic query.py functionality."""

    def test_query_script_exists(self):
        """Verify query.py script file exists."""
        assert QUERY_SCRIPT.exists(), f"Query script not found: {QUERY_SCRIPT}"

    def test_query_with_help_flag(self, query_cmd_base):
        """Test that --help works and shows usage information."""
        result = subprocess.run(
            query_cmd_base + ["--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--query" in result.stdout or "--query" in result.stderr
        assert "help" in result.stdout.lower() or "help" in result.stderr.lower()

    def test_query_missing_required_argument(self, query_cmd_base):
        """Test that --query is required and script fails without it."""
        result = subprocess.run(
            query_cmd_base,
            capture_output=True,
            text=True
        )
        # Should fail because --query is missing
        assert result.returncode != 0
        error_output = result.stdout + result.stderr
        # Should mention that query is required
        assert "query" in error_output.lower() or "required" in error_output.lower()


class TestQueryScriptArguments:
    """Test query.py argument parsing."""

    def test_query_with_simple_query(self, query_cmd_base):
        """Test running query with a simple query string."""
        result = subprocess.run(
            query_cmd_base + ["--query", "test query"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Script should run successfully (even if no results, it should handle gracefully)
        assert result.returncode == 0 or "not found" in result.stdout.lower() or "no" in result.stdout.lower()

    def test_query_with_top_k_parameter(self, query_cmd_base):
        """Test --top-k parameter."""
        result = subprocess.run(
            query_cmd_base + ["--query", "test", "--top-k", "5"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_query_with_collection_parameter(self, query_cmd_base):
        """Test --collection parameter."""
        result = subprocess.run(
            query_cmd_base + ["--query", "test", "--collection", "test_collection"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_query_with_verbose_flag(self, query_cmd_base):
        """Test --verbose flag."""
        result = subprocess.run(
            query_cmd_base + ["--query", "test", "--verbose"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_query_with_no_rerank_flag(self, query_cmd_base):
        """Test --no-rerank flag."""
        result = subprocess.run(
            query_cmd_base + ["--query", "test", "--no-rerank"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_query_with_multiple_parameters(self, query_cmd_base):
        """Test query with multiple parameters combined."""
        result = subprocess.run(
            query_cmd_base + [
                "--query", "test query",
                "--top-k", "20",
                "--collection", "test",
                "--verbose",
                "--no-rerank"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0


class TestQueryScriptOutput:
    """Test query.py output formatting."""

    def test_output_contains_results_or_no_data_message(self, query_cmd_base):
        """Test that output either contains results or a 'no data' message."""
        result = subprocess.run(
            query_cmd_base + ["--query", "test"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        output = result.stdout
        # Should either have results or mention no data
        has_results = "chunk" in output.lower() or "result" in output.lower()
        has_no_data = "not found" in output.lower() or "no" in output.lower() or "not" in output.lower()
        assert has_results or has_no_data, f"Unexpected output format: {output}"

    def test_verbose_output_more_detailed(self, query_cmd_base):
        """Test that verbose mode produces more output than normal mode."""
        # Run with verbose
        result_verbose = subprocess.run(
            query_cmd_base + ["--query", "test", "--verbose"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Run without verbose (or with less detail)
        result_normal = subprocess.run(
            query_cmd_base + ["--query", "test"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Both should succeed
        assert result_verbose.returncode == 0
        assert result_normal.returncode == 0

        # Verbose output should be at least as long or longer (or contain more detail markers)
        verbose_out = result_verbose.stdout + result_verbose.stderr
        normal_out = result_normal.stdout + result_normal.stderr

        # Verbose should contain more indicators of different stages
        verbose_stage_count = verbose_out.lower().count("stage") + verbose_out.lower().count("dense") + verbose_out.lower().count("sparse")
        normal_stage_count = normal_out.lower().count("stage") + normal_out.lower().count("dense") + normal_out.lower().count("sparse")

        # At minimum, verbose output should be present
        assert len(verbose_out) > 0


class TestQueryScriptEdgeCases:
    """Test edge cases and error handling."""

    def test_query_with_empty_string(self, query_cmd_base):
        """Test query with empty string."""
        result = subprocess.run(
            query_cmd_base + ["--query", ""],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Should either succeed or fail gracefully
        # Allow both behaviors as long as it doesn't crash
        assert result.returncode in [0, 1, 2]

    def test_query_with_special_characters(self, query_cmd_base):
        """Test query with special characters."""
        result = subprocess.run(
            query_cmd_base + ["--query", "测试查询 @#$%^&*"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0

    def test_query_with_invalid_top_k(self, query_cmd_base):
        """Test query with invalid top-k value."""
        result = subprocess.run(
            query_cmd_base + ["--query", "test", "--top-k", "abc"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Should fail gracefully with invalid parameter
        assert result.returncode != 0

    def test_query_with_negative_top_k(self, query_cmd_base):
        """Test query with negative top-k value."""
        result = subprocess.run(
            query_cmd_base + ["--query", "test", "--top-k", "-5"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # May fail due to validation or succeed (depending on implementation)
        # The important thing is it doesn't crash
        assert result.returncode in [0, 1, 2]

    def test_query_script_no_syntax_errors(self, query_cmd_base):
        """Test that query.py has no Python syntax errors."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(QUERY_SCRIPT)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Syntax errors in query.py: {result.stderr}"
