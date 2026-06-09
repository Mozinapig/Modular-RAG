"""
BM25Indexer: Build inverted index from sparse embeddings and enable BM25 queries.
Calculates IDF, constructs posting lists, persists to disk.
"""

import logging
import math
import json
from typing import List, Dict, Optional
from pathlib import Path

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


logger = logging.getLogger(__name__)


class BM25Indexer:
    """
    Build and query BM25 inverted index from sparse embeddings.

    Features:
    - IDF calculation: IDF(term) = log((N - df + 0.5) / (df + 0.5))
    - Inverted index structure: {term: {idf, postings: [{chunk_id, tf, doc_length}]}}
    - Index persistence: save/load from JSON files
    - Query support: retrieve documents by term with optional ranking
    - Incremental updates: add or rebuild index
    """

    def __init__(self):
        """Initialize BM25Indexer with empty index."""
        self.index: Dict = {}

    def build(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> None:
        """
        Build inverted index from chunks with sparse embeddings.

        Args:
            chunks: List of chunks with sparse_embedding metadata
            trace: Optional trace context for tracking
        """
        if not chunks:
            self.index = {}
            return

        # Clear existing index
        self.index = {}

        # Collect all terms and their document frequencies
        term_doc_freq = {}
        chunk_data = {}

        for chunk in chunks:
            sparse_embedding = chunk.metadata.get("sparse_embedding", {})
            term_weights = sparse_embedding.get("term_weights", {})

            if not term_weights:
                continue

            chunk_data[chunk.id] = {
                "terms": term_weights,
                "doc_length": len(term_weights)
            }

            # Count document frequency for each term
            for term in term_weights:
                if term not in term_doc_freq:
                    term_doc_freq[term] = 0
                term_doc_freq[term] += 1

        # Calculate total document count
        N = len(chunk_data)

        # Build index with IDF and postings
        for term, df in term_doc_freq.items():
            idf = math.log((N - df + 0.5) / (df + 0.5))

            self.index[term] = {
                "idf": idf,
                "postings": []
            }

        # Add postings for each chunk
        for chunk_id, data in chunk_data.items():
            for term, tf in data["terms"].items():
                posting = {
                    "chunk_id": chunk_id,
                    "tf": tf,
                    "doc_length": data["doc_length"]
                }
                self.index[term]["postings"].append(posting)

    def update(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> None:
        """
        Add chunks to existing index (incremental update).

        Args:
            chunks: List of new chunks to add
            trace: Optional trace context for tracking
        """
        if not chunks:
            return

        # Collect new terms and frequencies
        new_terms = {}
        chunk_data = {}

        for chunk in chunks:
            sparse_embedding = chunk.metadata.get("sparse_embedding", {})
            term_weights = sparse_embedding.get("term_weights", {})

            if not term_weights:
                continue

            chunk_data[chunk.id] = {
                "terms": term_weights,
                "doc_length": len(term_weights)
            }

            for term in term_weights:
                if term not in new_terms:
                    new_terms[term] = 0
                new_terms[term] += 1

        # Update document frequencies and add new postings
        for chunk_id, data in chunk_data.items():
            for term, tf in data["terms"].items():
                if term not in self.index:
                    self.index[term] = {
                        "idf": 0,  # Will recalculate
                        "postings": []
                    }

                posting = {
                    "chunk_id": chunk_id,
                    "tf": tf,
                    "doc_length": data["doc_length"]
                }
                self.index[term]["postings"].append(posting)

    def query(
        self,
        term: str,
        top_k: int = 10,
        trace: Optional[TraceContext] = None
    ) -> List[Dict]:
        """
        Query index by term.

        Args:
            term: Term to search for
            top_k: Maximum number of results to return
            trace: Optional trace context for tracking

        Returns:
            List of {chunk_id, tf, idf, score} dicts, ranked by score
        """
        if term not in self.index:
            return []

        term_entry = self.index[term]
        idf = term_entry["idf"]
        postings = term_entry["postings"]

        # Calculate BM25 scores for each posting
        results = []
        for posting in postings:
            tf = posting["tf"]
            doc_length = posting["doc_length"]

            # Simple BM25 score: tf * idf
            # (simplified version without k1, b parameters)
            score = tf * idf

            results.append({
                "chunk_id": posting["chunk_id"],
                "tf": tf,
                "idf": idf,
                "score": score
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        # Return top k
        return results[:top_k]

    def save(
        self,
        path: str,
        trace: Optional[TraceContext] = None
    ) -> None:
        """
        Serialize index to JSON file.

        Args:
            path: File path to save to
            trace: Optional trace context for tracking
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert index to JSON-serializable format
        serializable_index = {}
        for term, entry in self.index.items():
            serializable_index[term] = {
                "idf": entry["idf"],
                "postings": entry["postings"]
            }

        with open(path, "w") as f:
            json.dump(serializable_index, f, indent=2)

        logger.info(f"Saved BM25 index to {path}")

    def load(
        self,
        path: str,
        trace: Optional[TraceContext] = None
    ) -> None:
        """
        Load index from JSON file.

        Args:
            path: File path or directory path to load from (if directory, loads default.json)
            trace: Optional trace context for tracking
        """
        path = Path(path)

        # If path is directory, load default.json from it
        if path.is_dir():
            path = path / "default.json"

        if not path.exists():
            logger.warning(f"Index file not found: {path}")
            self.index = {}
            return

        with open(path, "r", encoding="utf-8") as f:
            loaded_index = json.load(f)

        self.index = loaded_index
        logger.info(f"Loaded BM25 index from {path}")
