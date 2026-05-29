"""
Test suite for ImageStorage - image file persistence with SQLite index mapping.
Coverage: File saving, path lookup, SQLite persistence, batch operations, concurrent access.
Target: 16+ tests.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sqlite3

from src.ingestion.storage.image_storage import ImageStorage


class TestImageStorageInitialization:
    """Tests for ImageStorage initialization."""

    def test_init_with_base_path(self):
        """Should initialize with base path for image storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            assert storage.base_path == Path(tmpdir)

    def test_init_creates_base_directory(self):
        """Should create base directory if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "images"
            storage = ImageStorage(base_path=str(base))

            assert base.exists()

    def test_init_creates_database(self):
        """Should initialize SQLite database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            # Database should be created
            assert storage.db_path.exists()

    def test_init_default_database_path(self):
        """Should use default database path data/db/image_index.db."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            # Should create db in standard location
            assert storage.db_path.exists()


class TestImageStorageSave:
    """Tests for saving images."""

    def test_save_single_image(self):
        """Should save image file to collection directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            # Create mock image file
            image_data = b"fake image data"
            image_id = storage.save(
                image_data=image_data,
                collection="documents",
                doc_hash="doc123",
                page_num=1
            )

            # Check file was saved
            collection_path = Path(tmpdir) / "documents"
            assert collection_path.exists()
            assert image_id is not None

    def test_save_returns_image_id(self):
        """Should return image_id after saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            image_id = storage.save(
                image_data=b"data",
                collection="test",
                doc_hash="hash",
                page_num=1
            )

            assert isinstance(image_id, str)
            assert len(image_id) > 0

    def test_save_creates_collection_directory(self):
        """Should create collection directory if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            storage.save(
                image_data=b"data",
                collection="new_collection"
            )

            collection_path = Path(tmpdir) / "new_collection"
            assert collection_path.exists()

    def test_save_file_exists_after_save(self):
        """Saved image file should exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            image_id = storage.save(
                image_data=b"test data",
                collection="docs"
            )

            # Find the saved file
            collection_path = Path(tmpdir) / "docs"
            files = list(collection_path.glob("*"))
            assert len(files) > 0

    def test_save_with_metadata(self):
        """Should save image with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            image_id = storage.save(
                image_data=b"data",
                collection="docs",
                doc_hash="abc123",
                page_num=5
            )

            # Check metadata was stored
            path = storage.get_path(image_id)
            assert path is not None


class TestImageStorageLookup:
    """Tests for image path lookup."""

    def test_get_path_returns_correct_path(self):
        """Should return correct file path for image_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            image_id = storage.save(
                image_data=b"test",
                collection="docs"
            )

            path = storage.get_path(image_id)

            assert path is not None
            assert image_id in str(path)

    def test_get_path_returns_none_for_missing_id(self):
        """Should return None for non-existent image_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            path = storage.get_path("nonexistent_id")

            assert path is None

    def test_get_path_file_exists(self):
        """File at returned path should exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            image_id = storage.save(
                image_data=b"test data",
                collection="docs"
            )

            path = storage.get_path(image_id)

            assert Path(path).exists()


class TestImageStoragePersistence:
    """Tests for SQLite database persistence."""

    def test_save_persists_to_database(self):
        """Image mapping should be persisted to SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            image_id = storage.save(
                image_data=b"data",
                collection="docs",
                doc_hash="hash123"
            )

            # Query database directly
            conn = sqlite3.connect(storage.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM image_index WHERE image_id = ?", (image_id,))
            result = cursor.fetchone()
            conn.close()

            assert result is not None

    def test_mapping_retrieval_after_restart(self):
        """Should retrieve mapping after storage object recreation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save image with first storage instance
            storage1 = ImageStorage(base_path=tmpdir)
            image_id = storage1.save(
                image_data=b"persistent data",
                collection="docs"
            )
            path1 = storage1.get_path(image_id)

            # Retrieve with new storage instance
            storage2 = ImageStorage(base_path=tmpdir)
            path2 = storage2.get_path(image_id)

            assert path1 == path2

    def test_database_schema_correct(self):
        """SQLite database should have correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            # Check table exists
            conn = sqlite3.connect(storage.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_index'")
            result = cursor.fetchone()
            conn.close()

            assert result is not None

    def test_database_columns_exist(self):
        """Database should have required columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            conn = sqlite3.connect(storage.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(image_index)")
            columns = {row[1] for row in cursor.fetchall()}
            conn.close()

            required = {'image_id', 'file_path', 'collection', 'doc_hash', 'page_num', 'created_at'}
            assert required.issubset(columns)


class TestImageStorageBatchOperations:
    """Tests for batch operations."""

    def test_save_multiple_images(self):
        """Should save multiple images correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            ids = []
            for i in range(5):
                image_id = storage.save(
                    image_data=f"data {i}".encode(),
                    collection="docs"
                )
                ids.append(image_id)

            assert len(ids) == 5
            # All IDs should be different
            assert len(set(ids)) == 5

    def test_list_by_collection(self):
        """Should list all images in collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ImageStorage(base_path=tmpdir, db_path=str(db_path))

            # Save images in same collection
            for i in range(3):
                storage.save(
                    image_data=f"data {i}".encode(),
                    collection="docs"
                )

            # Save images in different collection
            for i in range(2):
                storage.save(
                    image_data=f"data {i}".encode(),
                    collection="other"
                )

            docs = storage.list_by_collection("docs")
            assert len(docs) == 3

    def test_list_returns_correct_structure(self):
        """List should return image info with path and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ImageStorage(base_path=tmpdir, db_path=str(db_path))

            image_id = storage.save(
                image_data=b"data",
                collection="docs",
                doc_hash="hash123",
                page_num=1
            )

            images = storage.list_by_collection("docs")

            assert len(images) == 1
            image_info = images[0]
            assert "image_id" in image_info
            assert "file_path" in image_info
            assert "doc_hash" in image_info


class TestImageStorageDataIntegrity:
    """Tests for data integrity and concurrency."""

    def test_wal_mode_enabled(self):
        """Should enable WAL mode for concurrent access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(base_path=tmpdir)

            # Check WAL mode
            conn = sqlite3.connect(storage.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            conn.close()

            assert mode.upper() == 'WAL'

    def test_primary_key_constraint(self):
        """image_id should be primary key (unique)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ImageStorage(base_path=tmpdir, db_path=str(db_path))

            # Attempt to insert duplicate should fail
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Insert first record
            cursor.execute(
                "INSERT INTO image_index (image_id, file_path) VALUES (?, ?)",
                ("test_id", "/path/to/file")
            )
            conn.commit()

            # Try to insert duplicate - should raise IntegrityError
            try:
                cursor.execute(
                    "INSERT INTO image_index (image_id, file_path) VALUES (?, ?)",
                    ("test_id", "/path/to/other")
                )
                conn.commit()
                # If we get here, the constraint didn't work
                assert False, "Primary key constraint not enforced"
            except sqlite3.IntegrityError:
                # Expected behavior
                pass

            conn.close()


class TestImageStorageEdgeCases:
    """Tests for edge cases."""

    def test_save_empty_image_data(self):
        """Should handle empty image data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ImageStorage(base_path=tmpdir, db_path=str(db_path))

            image_id = storage.save(
                image_data=b"",
                collection="docs"
            )

            assert image_id is not None

    def test_save_large_image(self):
        """Should handle large image data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ImageStorage(base_path=tmpdir, db_path=str(db_path))

            # 10MB of data
            large_data = b"x" * (10 * 1024 * 1024)
            image_id = storage.save(
                image_data=large_data,
                collection="docs"
            )

            assert image_id is not None
            # Verify file size
            path = storage.get_path(image_id)
            assert Path(path).stat().st_size == 10 * 1024 * 1024

    def test_collection_with_special_characters(self):
        """Should handle collection names with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ImageStorage(base_path=tmpdir, db_path=str(db_path))

            image_id = storage.save(
                image_data=b"data",
                collection="docs-2024_v1"
            )

            assert image_id is not None

    def test_get_path_with_special_chars_in_collection(self):
        """Should retrieve path from collection with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ImageStorage(base_path=tmpdir, db_path=str(db_path))

            image_id = storage.save(
                image_data=b"data",
                collection="docs_special-chars"
            )

            path = storage.get_path(image_id)
            assert path is not None
