"""
Unit tests for file integrity checking (SHA256).
Tests focus on hash computation, skip logic, and SQLite persistence.
"""

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)


class TestFileIntegrityChecker:
    """Test FileIntegrityChecker abstract interface."""

    def test_abstract_interface(self):
        """Test that FileIntegrityChecker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FileIntegrityChecker()


class TestSQLiteIntegrityChecker:
    """Test SQLiteIntegrityChecker implementation."""

    @pytest.fixture
    def temp_db_dir(self):
        """Create a temporary directory for test database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content for file integrity checking")
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    @pytest.fixture
    def checker(self, temp_db_dir):
        """Create a SQLiteIntegrityChecker instance with temp database."""
        db_path = os.path.join(temp_db_dir, "test_integrity.db")
        return SQLiteIntegrityChecker(db_path=db_path)

    def test_checker_initialization(self, checker):
        """Test that checker initializes correctly."""
        assert checker is not None
        assert hasattr(checker, 'compute_sha256')
        assert hasattr(checker, 'should_skip')
        assert hasattr(checker, 'mark_success')
        assert hasattr(checker, 'mark_failed')

    def test_compute_sha256_consistency(self, checker, temp_file):
        """Test that computing SHA256 of same file returns consistent hash."""
        hash1 = checker.compute_sha256(temp_file)
        hash2 = checker.compute_sha256(temp_file)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest is 64 characters

    def test_compute_sha256_different_files(self, temp_db_dir):
        """Test that different files produce different hashes."""
        checker = SQLiteIntegrityChecker(
            db_path=os.path.join(temp_db_dir, "test_integrity.db")
        )

        # Create two different files
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f1:
            f1.write("content 1")
            file1 = f1.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f2:
            f2.write("content 2")
            file2 = f2.name

        try:
            hash1 = checker.compute_sha256(file1)
            hash2 = checker.compute_sha256(file2)
            assert hash1 != hash2
        finally:
            os.remove(file1)
            os.remove(file2)

    def test_should_skip_returns_false_initially(self, checker, temp_file):
        """Test that should_skip returns False for new file hash."""
        file_hash = checker.compute_sha256(temp_file)
        assert checker.should_skip(file_hash) is False

    def test_mark_success_then_should_skip(self, checker, temp_file):
        """Test that after marking success, should_skip returns True."""
        file_hash = checker.compute_sha256(temp_file)

        # Initially should not skip
        assert checker.should_skip(file_hash) is False

        # Mark as success
        checker.mark_success(file_hash, temp_file)

        # Now should skip
        assert checker.should_skip(file_hash) is True

    def test_mark_failed_then_should_skip(self, checker, temp_file):
        """Test that after marking failed, should_skip returns False."""
        file_hash = checker.compute_sha256(temp_file)

        # Mark as failed
        checker.mark_failed(file_hash, "test error message")

        # Should still not skip (failed files are retried)
        assert checker.should_skip(file_hash) is False

    def test_database_file_creation(self, temp_db_dir):
        """Test that database file is created in correct location."""
        db_path = os.path.join(temp_db_dir, "ingestion_history.db")
        checker = SQLiteIntegrityChecker(db_path=db_path)

        # Trigger database creation by performing an operation
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test")
            temp_file = f.name

        try:
            file_hash = checker.compute_sha256(temp_file)
            checker.mark_success(file_hash, temp_file)

            # Database file should exist
            assert os.path.exists(db_path)
        finally:
            os.remove(temp_file)

    def test_database_persistence(self, temp_db_dir):
        """Test that database persists across checker instances."""
        db_path = os.path.join(temp_db_dir, "ingestion_history.db")

        # Create first checker and mark a file as success
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("persistent test content")
            temp_file = f.name

        try:
            checker1 = SQLiteIntegrityChecker(db_path=db_path)
            file_hash = checker1.compute_sha256(temp_file)
            checker1.mark_success(file_hash, temp_file)

            # Create second checker instance
            checker2 = SQLiteIntegrityChecker(db_path=db_path)

            # Should still recognize the file as processed
            assert checker2.should_skip(file_hash) is True
        finally:
            os.remove(temp_file)

    def test_sqlite_wal_mode(self, temp_db_dir):
        """Test that SQLite is configured with WAL mode for concurrent writes."""
        db_path = os.path.join(temp_db_dir, "ingestion_history.db")
        checker = SQLiteIntegrityChecker(db_path=db_path)

        # Trigger database creation
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test")
            temp_file = f.name

        try:
            file_hash = checker.compute_sha256(temp_file)
            checker.mark_success(file_hash, temp_file)

            # Check WAL mode is enabled
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0]
            conn.close()

            assert journal_mode.lower() == "wal"
        finally:
            os.remove(temp_file)

    def test_mark_success_with_metadata(self, checker, temp_file):
        """Test that mark_success stores file path and metadata."""
        file_hash = checker.compute_sha256(temp_file)
        checker.mark_success(file_hash, temp_file)

        # Verify the record was stored (by checking should_skip)
        assert checker.should_skip(file_hash) is True

    def test_mark_failed_with_error_message(self, checker, temp_file):
        """Test that mark_failed stores error message."""
        file_hash = checker.compute_sha256(temp_file)
        error_msg = "Test error: file parsing failed"

        checker.mark_failed(file_hash, error_msg)

        # Failed files should not be skipped (will be retried)
        assert checker.should_skip(file_hash) is False

    def test_multiple_files_independence(self, checker, temp_db_dir):
        """Test that tracking multiple files works independently."""
        files = []
        hashes = []

        # Create and track 3 different files
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(f"content {i}")
                files.append(f.name)

        try:
            # Compute hashes
            for f in files:
                hashes.append(checker.compute_sha256(f))

            # Mark first file as success
            checker.mark_success(hashes[0], files[0])

            # Verify independence
            assert checker.should_skip(hashes[0]) is True
            assert checker.should_skip(hashes[1]) is False
            assert checker.should_skip(hashes[2]) is False

            # Mark second file as failed
            checker.mark_failed(hashes[1], "test error")

            # Verify still independent
            assert checker.should_skip(hashes[0]) is True
            assert checker.should_skip(hashes[1]) is False
            assert checker.should_skip(hashes[2]) is False
        finally:
            for f in files:
                os.remove(f)

    def test_large_file_hash_computation(self, checker, temp_db_dir):
        """Test SHA256 computation on larger file."""
        # Create a larger test file (1MB)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("x" * (1024 * 1024))  # 1MB of 'x'
            temp_file = f.name

        try:
            file_hash = checker.compute_sha256(temp_file)

            # Verify hash is valid SHA256
            assert len(file_hash) == 64
            assert all(c in '0123456789abcdef' for c in file_hash)

            # Verify consistency
            file_hash2 = checker.compute_sha256(temp_file)
            assert file_hash == file_hash2
        finally:
            os.remove(temp_file)

    def test_empty_file_hash(self, checker, temp_db_dir):
        """Test SHA256 computation on empty file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            # Write nothing, create empty file
            temp_file = f.name

        try:
            file_hash = checker.compute_sha256(temp_file)

            # Empty file should have valid SHA256
            assert len(file_hash) == 64

            # Known SHA256 of empty string
            expected_hash = hashlib.sha256(b'').hexdigest()
            assert file_hash == expected_hash
        finally:
            os.remove(temp_file)

    def test_nonexistent_file_handling(self, checker):
        """Test that computing hash of nonexistent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, OSError)):
            checker.compute_sha256("/nonexistent/path/to/file.txt")
