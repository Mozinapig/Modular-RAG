"""
File integrity checking using SHA256 hashing.
Provides abstract interface and SQLite-based default implementation.
"""

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


class FileIntegrityChecker(ABC):
    """Abstract base class for file integrity checking."""

    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash as hex string (64 characters)

        Raises:
            FileNotFoundError: If file does not exist
        """
        pass

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """
        Check if a file with given hash should be skipped (already processed).

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            True if file was previously processed successfully, False otherwise
        """
        pass

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str) -> None:
        """
        Mark a file as successfully processed.

        Args:
            file_hash: SHA256 hash of the file
            file_path: Path to the file
        """
        pass

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """
        Mark a file as failed processing.

        Args:
            file_hash: SHA256 hash of the file
            error_msg: Error message describing the failure
        """
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-based implementation of FileIntegrityChecker.

    Uses SQLite with WAL mode to support concurrent writes.
    Database schema:
        - file_hash: PRIMARY KEY, SHA256 hash of file
        - file_path: Path to the file
        - status: 'success' or 'failed'
        - error_msg: Error message if status is 'failed'
        - created_at: Timestamp of record creation
    """

    def __init__(self, db_path: str = "data/db/ingestion_history.db"):
        """
        Initialize SQLiteIntegrityChecker.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema and enable WAL mode."""
        # Create parent directories if needed
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable WAL mode for concurrent writes
        cursor.execute("PRAGMA journal_mode=WAL;")

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_integrity (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                error_msg TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        conn.close()

    def compute_sha256(self, file_path: str) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash as hex string

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """
        Check if a file with given hash should be skipped.

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            True if file was previously processed successfully, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT status FROM file_integrity WHERE file_hash = ?",
            (file_hash,)
        )
        result = cursor.fetchone()
        conn.close()

        if result is None:
            return False

        status = result[0]
        return status == 'success'

    def mark_success(self, file_hash: str, file_path: str) -> None:
        """
        Mark a file as successfully processed.

        Args:
            file_hash: SHA256 hash of the file
            file_path: Path to the file
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO file_integrity
            (file_hash, file_path, status, error_msg)
            VALUES (?, ?, 'success', NULL)
        """, (file_hash, file_path))

        conn.commit()
        conn.close()

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """
        Mark a file as failed processing.

        Args:
            file_hash: SHA256 hash of the file
            error_msg: Error message describing the failure
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO file_integrity
            (file_hash, file_path, status, error_msg)
            VALUES (?, '', 'failed', ?)
        """, (file_hash, error_msg))

        conn.commit()
        conn.close()
