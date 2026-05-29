"""
ImageStorage: Save images to disk and maintain SQLite index mapping.
Provides persistent storage for extracted images with metadata tracking.
"""

import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ImageStorage:
    """
    Store images with SQLite-backed index mapping.

    Features:
    - Image persistence: Save images to data/images/{collection}/
    - SQLite indexing: Maintain image_id → file_path mapping
    - Concurrent access: WAL mode for safe concurrent reads
    - Batch operations: List images by collection
    - Metadata tracking: Store doc_hash, page_num, creation time
    """

    def __init__(
        self,
        base_path: str = "data/images",
        db_path: str = "data/db/image_index.db"
    ):
        """
        Initialize ImageStorage.

        Args:
            base_path: Base directory for storing images
            db_path: SQLite database file path
        """
        self.base_path = Path(base_path)
        self.db_path = Path(db_path)

        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database with image_index table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable WAL mode for concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_index (
                image_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                collection TEXT,
                doc_hash TEXT,
                page_num INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)
        """)

        conn.commit()
        conn.close()

        logger.info(f"Initialized image index database at {self.db_path}")

    def save(
        self,
        image_data: bytes,
        collection: str = "default",
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None
    ) -> str:
        """
        Save image file and record mapping in SQLite.

        Args:
            image_data: Binary image data
            collection: Collection name (used as subdirectory)
            doc_hash: Hash of source document (for tracking)
            page_num: Page number in source document

        Returns:
            Generated image_id
        """
        # Generate unique image ID
        image_id = str(uuid.uuid4())

        # Create collection directory
        collection_path = self.base_path / collection
        collection_path.mkdir(parents=True, exist_ok=True)

        # Determine file extension (simplified)
        file_name = f"{image_id}.png"
        file_path = collection_path / file_name

        # Write image to disk
        with open(file_path, "wb") as f:
            f.write(image_data)

        # Record in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num)
            VALUES (?, ?, ?, ?, ?)
        """, (image_id, str(file_path), collection, doc_hash, page_num))

        conn.commit()
        conn.close()

        logger.info(f"Saved image {image_id} to {file_path}")

        return image_id

    def get_path(self, image_id: str) -> Optional[str]:
        """
        Get file path for image_id.

        Args:
            image_id: Image ID to look up

        Returns:
            File path or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT file_path FROM image_index WHERE image_id = ?",
            (image_id,)
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            file_path = result[0]
            # Verify file exists
            if Path(file_path).exists():
                return file_path

        return None

    def list_by_collection(self, collection: str) -> List[Dict]:
        """
        List all images in a collection.

        Args:
            collection: Collection name

        Returns:
            List of image info dicts with image_id, file_path, metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT image_id, file_path, doc_hash, page_num, created_at
            FROM image_index
            WHERE collection = ?
            ORDER BY created_at DESC
        """, (collection,))

        results = cursor.fetchall()
        conn.close()

        images = []
        for image_id, file_path, doc_hash, page_num, created_at in results:
            images.append({
                "image_id": image_id,
                "file_path": file_path,
                "doc_hash": doc_hash,
                "page_num": page_num,
                "created_at": created_at
            })

        return images

    def delete(self, image_id: str) -> bool:
        """
        Delete image and its index entry.

        Args:
            image_id: Image ID to delete

        Returns:
            True if deleted, False if not found
        """
        # Get file path
        file_path = self.get_path(image_id)
        if not file_path:
            return False

        # Delete file
        try:
            Path(file_path).unlink()
        except FileNotFoundError:
            pass

        # Delete database entry
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM image_index WHERE image_id = ?",
            (image_id,)
        )

        conn.commit()
        conn.close()

        logger.info(f"Deleted image {image_id}")

        return True
