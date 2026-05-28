"""
PDF document loader implementation.
Extracts text and images from PDF files using markitdown.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

from src.core.types import Document, ImageRef
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """PDF document loader using markitdown for text and image extraction."""

    def __init__(self, image_output_dir: str = "data/images"):
        """
        Initialize PdfLoader.

        Args:
            image_output_dir: Base directory for storing extracted images
        """
        self.image_output_dir = image_output_dir
        Path(self.image_output_dir).mkdir(parents=True, exist_ok=True)

    def load(self, path: str) -> Document:
        """
        Load a PDF document and extract text and images.

        Args:
            path: Path to the PDF file

        Returns:
            Document object with extracted text and image metadata

        Raises:
            FileNotFoundError: If PDF file does not exist
            ValueError: If PDF cannot be parsed
        """
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {path}")

        try:
            import markitdown
        except ImportError:
            raise ImportError("markitdown is required for PDF loading. Install with: pip install markitdown")

        # Compute document hash for deterministic ID
        doc_hash = self._compute_file_hash(path)
        doc_id = f"pdf_{doc_hash}"

        # Extract text and images using markitdown
        try:
            converter = markitdown.MarkItDown()
            result = converter.convert(path)
            text = result.markdown if result else ""
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {path}: {e}")
            raise ValueError(f"Failed to parse PDF: {e}")

        # Extract images from PDF
        images = []
        try:
            images = self._extract_images(path, doc_hash, text)
        except Exception as e:
            logger.warning(f"Failed to extract images from PDF {path}: {e}")
            # Don't raise - image extraction failure should not block text parsing

        # Build metadata
        metadata = {
            "source_path": path,
        }
        if images:
            metadata["images"] = images

        return Document(
            id=doc_id,
            text=text,
            metadata=metadata
        )

    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA256 hash of file (first 16 chars).

        Args:
            file_path: Path to the file

        Returns:
            First 16 characters of SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()[:16]

    def _extract_images(self, pdf_path: str, doc_hash: str, text: str) -> list:
        """
        Extract images from PDF and save to disk.

        Args:
            pdf_path: Path to the PDF file
            doc_hash: Document hash for directory naming
            text: Extracted text (to find image placeholders)

        Returns:
            List of ImageRef objects
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.debug("PyMuPDF not available, skipping image extraction")
            return []

        images = []
        image_dir = Path(self.image_output_dir) / doc_hash
        image_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
            image_counter = 0

            for page_num, page in enumerate(doc):
                # Extract images from page
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        pix = fitz.Pixmap(doc, xref)

                        # Convert to RGB if needed
                        if pix.n - pix.alpha < 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # Save image
                        image_id = f"{doc_hash}_{page_num}_{image_counter}"
                        image_path = image_dir / f"{image_id}.png"
                        pix.save(str(image_path))

                        # Create placeholder and insert into text
                        placeholder = f"[IMAGE: {image_id}]"
                        text = self._insert_image_placeholder(text, placeholder, page_num)

                        # Record image metadata
                        image_ref = ImageRef(
                            id=image_id,
                            path=str(image_path),
                            text_offset=self._find_placeholder_offset(text, placeholder),
                            text_length=len(placeholder),
                            page=page_num,
                            position={"page": page_num, "index": img_index}
                        )
                        images.append(image_ref)
                        image_counter += 1

                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                        continue

            doc.close()

        except Exception as e:
            logger.warning(f"Failed to open PDF with PyMuPDF: {e}")
            return []

        return images

    def _insert_image_placeholder(self, text: str, placeholder: str, page_num: int) -> str:
        """
        Insert image placeholder into text at appropriate location.

        Args:
            text: Original text
            placeholder: Placeholder string to insert
            page_num: Page number where image appears

        Returns:
            Text with placeholder inserted
        """
        # Simple strategy: insert at end of page section
        # This is a simplified approach - in production, would use PDF coordinates
        lines = text.split('\n')
        if lines:
            # Insert placeholder at end of text for now
            return text.rstrip() + '\n' + placeholder
        return placeholder

    def _find_placeholder_offset(self, text: str, placeholder: str) -> int:
        """
        Find the character offset of a placeholder in text.

        Args:
            text: Text containing the placeholder
            placeholder: Placeholder string to find

        Returns:
            Character offset (0-indexed)
        """
        offset = text.rfind(placeholder)
        return max(0, offset)
