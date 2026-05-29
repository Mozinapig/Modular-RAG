"""
SparseEncoder: Build BM25 statistics from chunks via term frequency analysis.
Produces sparse term weight vectors compatible with BM25 indexer.
"""

import logging
from typing import List, Optional, Dict
from collections import defaultdict
import re

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.core.settings import Settings


logger = logging.getLogger(__name__)

# Common English stopwords for filtering
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'to', 'was', 'will',
    'with', 'this', 'but', 'they', 'have', 'had', 'what', 'which', 'who', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'can', 'just', 'should', 'now', 'do', 'does', 'did',
    'if', 'am', 'been', 'being', 'you', 'your', 'yours', 'yourself', 'we', 'our',
    'ours', 'ourselves', 'me', 'my', 'myself', 'him', 'his', 'himself', 'her',
    'herself', 'them', 'theirs', 'themselves', 'herself', 'itself', 'ourselves',
    'there', 'here', 'then', 'them', 'they', 'those', 'these', 'above', 'after',
    'again', 'against', 'below', 'between', 'during', 'before', 'through', 'under',
    'while', 'because', 'down', 'up', 'out', 'off', 'over', 'such', 'itself'
}


class SparseEncoder:
    """
    Encode chunks into sparse term weight vectors for BM25 indexing.

    Features:
    - Tokenization: Break text into terms with case normalization
    - Stopword filtering: Remove common words
    - Term frequency extraction: Count term occurrences per chunk
    - Metadata preservation: Keep all chunk metadata and add sparse_embedding field
    """

    def __init__(self, settings: Settings, tokenizer=None):
        """
        Initialize SparseEncoder.

        Args:
            settings: Settings object (currently minimal, reserved for future config)
            tokenizer: Optional tokenizer for dependency injection (testing)
        """
        self.settings = settings
        self.tokenizer = tokenizer

    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Encode chunks into sparse term weight vectors.

        Args:
            chunks: List of chunks to encode
            trace: Optional trace context for tracking

        Returns:
            List of chunks with sparse_embedding added to metadata
        """
        if not chunks:
            return []

        # Extract term statistics for each chunk
        for chunk in chunks:
            term_weights = self._extract_term_weights(chunk.text)
            sparse_stats = {
                "term_weights": term_weights,
            }
            chunk.metadata["sparse_embedding"] = sparse_stats

        return chunks

    def _extract_term_weights(self, text: str) -> Dict[str, float]:
        """
        Extract term weights from text via tokenization and term frequency.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping term to weight (term frequency for now)
        """
        if not text or not text.strip():
            return {}

        # Tokenize: convert to lowercase, split on non-alphanumeric
        tokens = self._tokenize(text)

        # Filter stopwords and build frequency map
        term_counts = defaultdict(int)
        for token in tokens:
            if token not in STOPWORDS and len(token) > 0:
                term_counts[token] += 1

        # Convert counts to weights (using raw TF for now)
        term_weights = {}
        for term, count in term_counts.items():
            # Simple TF weight: just use the count
            term_weights[term] = count

        return term_weights

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text: split into words, normalize case, remove special chars.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (normalized to lowercase)
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Split on non-alphanumeric characters (keep underscores for robustness)
        tokens = re.findall(r'\b\w+\b', text)

        return tokens
