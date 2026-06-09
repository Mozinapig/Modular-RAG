"""Query processing module."""
from dataclasses import dataclass, field
from typing import Dict, List
import re


@dataclass
class ProcessedQuery:
    """Represents a processed query with keywords and filters."""

    keywords: List[str]
    filters: Dict[str, str] = field(default_factory=dict)


class QueryProcessor:
    """Processes raw queries into structured ProcessedQuery objects."""

    # Common English stopwords
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "must", "shall", "as", "if",
        "that", "this", "these", "those", "i", "you", "he", "she", "it", "we",
        "they", "what", "which", "who", "when", "where", "why", "how"
    }

    def __init__(self):
        """Initialize QueryProcessor."""
        pass

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a raw query string into keywords and filters.

        Args:
            query: Raw query string

        Returns:
            ProcessedQuery with extracted keywords and filters
        """
        if not query or not query.strip():
            return ProcessedQuery(keywords=[], filters={})

        query = query.strip()

        # Extract filters (format: key:value)
        filters = self._extract_filters(query)

        # Remove filter patterns from query for keyword extraction
        query_for_keywords = self._remove_filter_patterns(query)

        # Extract keywords
        keywords = self._extract_keywords(query_for_keywords)

        return ProcessedQuery(keywords=keywords, filters=filters)

    def _extract_filters(self, query: str) -> Dict[str, str]:
        """Extract key:value filter pairs from query."""
        filters = {}
        # Match patterns like "key:value"
        pattern = r'(\w+):(\w+)'
        matches = re.findall(pattern, query)
        for key, value in matches:
            filters[key] = value
        return filters

    def _remove_filter_patterns(self, query: str) -> str:
        """Remove filter patterns from query string."""
        # Remove key:value patterns
        pattern = r'\w+:\w+\s*'
        return re.sub(pattern, '', query)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text, removing stopwords and punctuation."""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation but preserve some meaningful characters like +, -
        # Split by whitespace and punctuation
        tokens = re.findall(r'\b[\w一-鿿]+\b', text)

        # Filter out stopwords
        keywords = [token for token in tokens if token not in self.STOPWORDS]

        return keywords
