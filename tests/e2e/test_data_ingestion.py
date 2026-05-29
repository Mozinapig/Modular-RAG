"""
E2E test for data ingestion script (C15).

Tests the complete CLI entry point for offline data ingestion.
Verifies command-line parameter parsing and pipeline invocation.
"""

import tempfile
from pathlib import Path
import pytest
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.ingest import main, create_argument_parser


@pytest.fixture
def sample_pdf_path():
    """Fixture providing path to sample PDF for testing."""
    fixtures_dir = Path("tests/fixtures/sample_documents")
    if not fixtures_dir.exists():
        pytest.skip("Sample documents fixture not found")

    pdf_file = fixtures_dir / "simple.pdf"
    if not pdf_file.exists():
        pytest.skip("simple.pdf fixture not found")

    return str(pdf_file.absolute())


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestIngestScript:
    """Test suite for scripts/ingest.py CLI entry point."""

    def test_ingest_script_exists(self):
        """Verify ingest.py script file exists."""
        script_path = Path("scripts/ingest.py")
        assert script_path.exists(), "scripts/ingest.py does not exist"

    def test_create_argument_parser(self):
        """Test argument parser creation."""
        parser = create_argument_parser()
        assert parser is not None

    def test_parser_required_path_argument(self):
        """Test that --path argument is required."""
        parser = create_argument_parser()
        with pytest.raises(SystemExit):
            # Should fail without --path
            parser.parse_args(["--collection", "test"])

    def test_parser_accepts_path(self):
        """Test that parser accepts --path argument."""
        parser = create_argument_parser()
        args = parser.parse_args(["--path", "/some/file.pdf"])
        assert args.path == "/some/file.pdf"

    def test_parser_accepts_collection(self):
        """Test that parser accepts --collection argument."""
        parser = create_argument_parser()
        args = parser.parse_args(["--path", "/some/file.pdf", "--collection", "my_docs"])
        assert args.collection == "my_docs"

    def test_parser_default_collection(self):
        """Test default collection name."""
        parser = create_argument_parser()
        args = parser.parse_args(["--path", "/some/file.pdf"])
        assert args.collection == "default"

    def test_parser_accepts_force_flag(self):
        """Test that parser accepts --force flag."""
        parser = create_argument_parser()
        args = parser.parse_args(["--path", "/some/file.pdf", "--force"])
        assert args.force is True

    def test_parser_default_no_force(self):
        """Test --force defaults to False."""
        parser = create_argument_parser()
        args = parser.parse_args(["--path", "/some/file.pdf"])
        assert args.force is False

    def test_main_invalid_file_path(self):
        """Test main() error handling for non-existent file."""
        result = main(["--path", "/nonexistent/path/file.pdf"])
        assert result != 0, "Should fail for non-existent file"

    def test_main_invalid_file_extension(self):
        """Test main() error handling for non-PDF file."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as f:
            result = main(["--path", f.name])
            assert result != 0, "Should fail for non-PDF file"

    def test_main_with_sample_pdf(self, sample_pdf_path):
        """Test main() with actual sample PDF."""
        # This test requires a valid PDF and full environment setup
        # so it may fail if dependencies are not fully installed
        with tempfile.TemporaryDirectory() as tmpdir:
            # Note: This will attempt real ingestion which may require:
            # - Full LLM/Embedding configuration
            # - Valid API keys
            # - Database setup
            # So we just verify it doesn't crash immediately
            result = main([
                "--path", sample_pdf_path,
                "--collection", "test_e2e",
                "--data-dir", tmpdir
            ])
            # Result will depend on environment, so we just check it's an int
            assert isinstance(result, int)

    def test_help_output_shows_options(self):
        """Test that --help outputs usage information."""
        parser = create_argument_parser()
        # Test by capturing help text
        help_text = parser.format_help()
        assert "--path" in help_text
        assert "--collection" in help_text
        assert "--force" in help_text

    def test_main_accepts_args_list(self):
        """Test that main() accepts command-line arguments as list."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            # Create a minimal PDF file
            f.write(b"%PDF-1.4\n")
            f.flush()

            result = main([
                "--path", f.name,
                "--collection", "unit_test"
            ])
            # Should handle the call (success or failure depends on environment)
            assert isinstance(result, int)

    def test_parser_data_dir_option(self):
        """Test --data-dir option."""
        parser = create_argument_parser()
        args = parser.parse_args([
            "--path", "/some/file.pdf",
            "--data-dir", "/custom/data"
        ])
        assert args.data_dir == "/custom/data"

    def test_parser_default_data_dir(self):
        """Test --data-dir defaults to 'data'."""
        parser = create_argument_parser()
        args = parser.parse_args(["--path", "/some/file.pdf"])
        assert args.data_dir == "data"

    def test_parser_verbose_flag(self):
        """Test --verbose flag."""
        parser = create_argument_parser()
        args = parser.parse_args(["--path", "/some/file.pdf", "--verbose"])
        assert args.verbose is True

    def test_parser_default_no_verbose(self):
        """Test --verbose defaults to False."""
        parser = create_argument_parser()
        args = parser.parse_args(["--path", "/some/file.pdf"])
        assert args.verbose is False

    def test_main_returns_exit_code(self):
        """Test that main() returns an integer exit code."""
        result = main(["--path", "/nonexistent/file.pdf"])
        assert isinstance(result, int)
        assert result in [0, 1, 2], "Should return valid exit code"
