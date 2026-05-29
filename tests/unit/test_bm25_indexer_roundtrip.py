"""
Test suite for BM25Indexer - inverted index construction and persistence.
Coverage: Index building, IDF calculation, serialization, query, incremental updates.
Target: 20+ tests.
"""

import pytest
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.core.types import Chunk
from src.ingestion.storage.bm25_indexer import BM25Indexer


class TestBM25IndexerInitialization:
    """Tests for BM25Indexer initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default settings."""
        indexer = BM25Indexer()

        assert indexer.index == {}
        assert isinstance(indexer.index, dict)

    def test_init_creates_empty_index(self):
        """Should create empty index structure."""
        indexer = BM25Indexer()

        assert len(indexer.index) == 0
        assert indexer.index == {}


class TestBM25IndexerBuildFromChunks:
    """Tests for building index from sparse embeddings."""

    def test_build_index_from_single_chunk(self):
        """Should build index from a single chunk with sparse embedding."""
        indexer = BM25Indexer()

        # Create chunk with sparse embedding
        chunk = Chunk(
            id="chunk_001",
            text="hello world",
            metadata={
                "sparse_embedding": {
                    "term_weights": {"hello": 1, "world": 1}
                }
            }
        )

        indexer.build([chunk])

        # Check index was built
        assert len(indexer.index) > 0
        assert "hello" in indexer.index
        assert "world" in indexer.index

    def test_build_index_structure_contains_idf(self):
        """Index should contain IDF for each term."""
        indexer = BM25Indexer()

        chunks = [
            Chunk(
                id="chunk_001",
                text="cat dog",
                metadata={
                    "sparse_embedding": {"term_weights": {"cat": 1, "dog": 1}}
                }
            )
        ]

        indexer.build(chunks)

        # Each term entry should have idf and postings
        assert "idf" in indexer.index["cat"]
        assert "postings" in indexer.index["cat"]
        assert isinstance(indexer.index["cat"]["idf"], float)
        assert isinstance(indexer.index["cat"]["postings"], list)

    def test_build_index_postings_structure(self):
        """Postings should contain chunk_id, tf, and doc_length."""
        indexer = BM25Indexer()

        chunk = Chunk(
            id="chunk_001",
            text="hello world test",
            metadata={
                "sparse_embedding": {"term_weights": {"hello": 2, "world": 1, "test": 1}}
            }
        )

        indexer.build([chunk])

        # Check postings structure for "hello" (tf=2)
        hello_postings = indexer.index["hello"]["postings"]
        assert len(hello_postings) == 1
        posting = hello_postings[0]
        assert posting["chunk_id"] == "chunk_001"
        assert posting["tf"] == 2
        assert posting["doc_length"] == 3  # 3 unique terms: hello, world, test

    def test_build_index_from_multiple_chunks(self):
        """Should build index from multiple chunks."""
        indexer = BM25Indexer()

        chunks = [
            Chunk(
                id="chunk_001",
                text="apple banana",
                metadata={
                    "sparse_embedding": {"term_weights": {"apple": 1, "banana": 1}}
                }
            ),
            Chunk(
                id="chunk_002",
                text="apple cherry",
                metadata={
                    "sparse_embedding": {"term_weights": {"apple": 1, "cherry": 1}}
                }
            )
        ]

        indexer.build(chunks)

        # "apple" appears in both documents
        assert "apple" in indexer.index
        assert len(indexer.index["apple"]["postings"]) == 2


class TestBM25IndexerIDFCalculation:
    """Tests for IDF calculation accuracy."""

    def test_idf_single_document(self):
        """IDF with single document should match formula."""
        indexer = BM25Indexer()

        chunks = [
            Chunk(
                id="chunk_001",
                text="term",
                metadata={
                    "sparse_embedding": {"term_weights": {"term": 1}}
                }
            )
        ]

        indexer.build(chunks)

        # IDF = log((N - df + 0.5) / (df + 0.5))
        # N = 1, df = 1
        # IDF = log((1 - 1 + 0.5) / (1 + 0.5)) = log(0.5 / 1.5) = log(1/3)
        expected_idf = math.log((1 - 1 + 0.5) / (1 + 0.5))
        actual_idf = indexer.index["term"]["idf"]

        assert abs(actual_idf - expected_idf) < 1e-6

    def test_idf_term_in_multiple_documents(self):
        """IDF should decrease as term appears in more documents."""
        indexer = BM25Indexer()

        chunks = [
            Chunk(
                id=f"chunk_{i:03d}",
                text=f"common word_{i}",
                metadata={
                    "sparse_embedding": {
                        "term_weights": {"common": 1, f"word_{i}": 1}
                    }
                }
            )
            for i in range(5)
        ]

        indexer.build(chunks)

        # "common" appears in all 5 documents
        N = 5
        df = 5
        expected_idf = math.log((N - df + 0.5) / (df + 0.5))
        actual_idf = indexer.index["common"]["idf"]

        assert abs(actual_idf - expected_idf) < 1e-6
        # IDF should be small since term is very common
        assert actual_idf < 1.0

    def test_idf_rare_term(self):
        """IDF should be high for rare terms."""
        indexer = BM25Indexer()

        chunks = [
            Chunk(
                id="chunk_001",
                text="rare special",
                metadata={
                    "sparse_embedding": {
                        "term_weights": {"rare": 1, "special": 1}
                    }
                }
            ),
            Chunk(
                id="chunk_002",
                text="common word",
                metadata={
                    "sparse_embedding": {
                        "term_weights": {"common": 2, "word": 1}
                    }
                }
            )
        ]

        indexer.build(chunks)

        # "rare" appears in 1 document out of 2
        rare_idf = indexer.index["rare"]["idf"]
        # "common" appears in 1 document out of 2
        common_idf = indexer.index["common"]["idf"]

        # Both should have same IDF since both appear in 1 doc
        assert abs(rare_idf - common_idf) < 1e-6


class TestBM25IndexerSerialization:
    """Tests for index serialization and deserialization."""

    def test_save_index_creates_file(self):
        """Should save index to file."""
        indexer = BM25Indexer()

        chunk = Chunk(
            id="chunk_001",
            text="hello world",
            metadata={
                "sparse_embedding": {"term_weights": {"hello": 1, "world": 1}}
            }
        )
        indexer.build([chunk])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index.json"
            indexer.save(str(path))

            assert path.exists()
            assert path.stat().st_size > 0

    def test_load_index_restores_data(self):
        """Should load index from file."""
        indexer1 = BM25Indexer()

        chunk = Chunk(
            id="chunk_001",
            text="hello world",
            metadata={
                "sparse_embedding": {"term_weights": {"hello": 1, "world": 1}}
            }
        )
        indexer1.build([chunk])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index.json"
            indexer1.save(str(path))

            # Load into new indexer
            indexer2 = BM25Indexer()
            indexer2.load(str(path))

            # Should have same index
            assert len(indexer2.index) == len(indexer1.index)
            assert "hello" in indexer2.index
            assert "world" in indexer2.index

    def test_roundtrip_preserves_idf(self):
        """Roundtrip save/load should preserve IDF values."""
        indexer1 = BM25Indexer()

        chunks = [
            Chunk(
                id="chunk_001",
                text="apple banana",
                metadata={
                    "sparse_embedding": {"term_weights": {"apple": 1, "banana": 1}}
                }
            ),
            Chunk(
                id="chunk_002",
                text="apple cherry",
                metadata={
                    "sparse_embedding": {"term_weights": {"apple": 1, "cherry": 1}}
                }
            )
        ]
        indexer1.build(chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index.json"
            indexer1.save(str(path))

            indexer2 = BM25Indexer()
            indexer2.load(str(path))

            # IDF values should match
            for term in ["apple", "banana", "cherry"]:
                assert abs(indexer1.index[term]["idf"] - indexer2.index[term]["idf"]) < 1e-6


class TestBM25IndexerQuery:
    """Tests for index query functionality."""

    def test_query_returns_results_for_existing_term(self):
        """Should return documents containing query term."""
        indexer = BM25Indexer()

        chunks = [
            Chunk(
                id="chunk_001",
                text="python programming",
                metadata={
                    "sparse_embedding": {"term_weights": {"python": 1, "programming": 1}}
                }
            ),
            Chunk(
                id="chunk_002",
                text="java programming",
                metadata={
                    "sparse_embedding": {"term_weights": {"java": 1, "programming": 1}}
                }
            )
        ]
        indexer.build(chunks)

        results = indexer.query("programming", top_k=10)

        # Should find both documents
        result_ids = [r["chunk_id"] for r in results]
        assert len(results) == 2
        assert "chunk_001" in result_ids
        assert "chunk_002" in result_ids

    def test_query_returns_empty_for_nonexistent_term(self):
        """Should return empty results for term not in index."""
        indexer = BM25Indexer()

        chunk = Chunk(
            id="chunk_001",
            text="hello",
            metadata={
                "sparse_embedding": {"term_weights": {"hello": 1}}
            }
        )
        indexer.build([chunk])

        results = indexer.query("nonexistent", top_k=10)

        assert results == []

    def test_query_respects_top_k(self):
        """Should limit results to top_k."""
        indexer = BM25Indexer()

        chunks = [
            Chunk(
                id=f"chunk_{i:03d}",
                text="common",
                metadata={
                    "sparse_embedding": {"term_weights": {"common": 1}}
                }
            )
            for i in range(10)
        ]
        indexer.build(chunks)

        results = indexer.query("common", top_k=3)

        assert len(results) == 3

    def test_query_stable_top_ids(self):
        """Query should return stable top IDs across multiple calls."""
        indexer = BM25Indexer()

        chunks = [
            Chunk(
                id="chunk_001",
                text="data science machine learning",
                metadata={
                    "sparse_embedding": {
                        "term_weights": {"data": 2, "science": 1, "machine": 1, "learning": 1}
                    }
                }
            ),
            Chunk(
                id="chunk_002",
                text="machine learning algorithms",
                metadata={
                    "sparse_embedding": {
                        "term_weights": {"machine": 1, "learning": 2, "algorithms": 1}
                    }
                }
            )
        ]
        indexer.build(chunks)

        results1 = indexer.query("machine", top_k=10)
        results2 = indexer.query("machine", top_k=10)

        ids1 = [r["chunk_id"] for r in results1]
        ids2 = [r["chunk_id"] for r in results2]

        # Should get same order
        assert ids1 == ids2


class TestBM25IndexerIncrementalUpdates:
    """Tests for incremental index updates."""

    def test_add_chunks_to_existing_index(self):
        """Should add new chunks to existing index."""
        indexer = BM25Indexer()

        # Initial chunk
        chunk1 = Chunk(
            id="chunk_001",
            text="apple",
            metadata={
                "sparse_embedding": {"term_weights": {"apple": 1}}
            }
        )
        indexer.build([chunk1])

        # Add more chunks
        chunk2 = Chunk(
            id="chunk_002",
            text="apple banana",
            metadata={
                "sparse_embedding": {"term_weights": {"apple": 1, "banana": 1}}
            }
        )
        indexer.update([chunk2])

        # Index should have both terms
        assert "apple" in indexer.index
        assert "banana" in indexer.index
        # "apple" should have 2 postings now
        assert len(indexer.index["apple"]["postings"]) == 2

    def test_rebuild_recalculates_idf(self):
        """Rebuild should recalculate IDF for all terms."""
        indexer = BM25Indexer()

        # Initial build with 1 chunk
        chunk1 = Chunk(
            id="chunk_001",
            text="common",
            metadata={
                "sparse_embedding": {"term_weights": {"common": 1}}
            }
        )
        indexer.build([chunk1])
        idf_after_1 = indexer.index["common"]["idf"]

        # Rebuild with 5 chunks
        chunks = [
            Chunk(
                id=f"chunk_{i:03d}",
                text="common",
                metadata={
                    "sparse_embedding": {"term_weights": {"common": 1}}
                }
            )
            for i in range(5)
        ]
        indexer.build(chunks)
        idf_after_5 = indexer.index["common"]["idf"]

        # IDF should be lower with more documents
        assert idf_after_5 < idf_after_1


class TestBM25IndexerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_chunk_list(self):
        """Should handle empty chunk list."""
        indexer = BM25Indexer()
        indexer.build([])

        assert indexer.index == {}

    def test_chunk_with_empty_sparse_embedding(self):
        """Should handle chunk with empty sparse embedding."""
        indexer = BM25Indexer()

        chunk = Chunk(
            id="chunk_001",
            text="",
            metadata={
                "sparse_embedding": {"term_weights": {}}
            }
        )
        indexer.build([chunk])

        # Should complete without error
        assert indexer.index == {}

    def test_term_frequency_accumulation(self):
        """Same term appearing multiple times should accumulate TF."""
        indexer = BM25Indexer()

        chunk = Chunk(
            id="chunk_001",
            text="python python python",
            metadata={
                "sparse_embedding": {"term_weights": {"python": 3}}
            }
        )
        indexer.build([chunk])

        postings = indexer.index["python"]["postings"]
        assert len(postings) == 1
        assert postings[0]["tf"] == 3
