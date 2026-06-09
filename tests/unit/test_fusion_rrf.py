"""
Unit tests for RRF (Reciprocal Rank Fusion) implementation.

Tests verify:
- RRF deterministic merging of dense/sparse rankings
- Configurable k parameter
- Proper score calculation and ranking
- Edge cases (empty inputs, duplicates, missing items)
"""

import pytest
from src.core.types import RetrievalResult
from src.core.query_engine.fusion import RRFFusion


class TestRRFFusionBasic:
    """Basic RRF functionality tests."""

    def test_fusion_with_identical_rankings(self):
        """When dense and sparse return same ranking, order preserved."""
        dense_results = [
            RetrievalResult(chunk_id="chunk_1", score=0.9, text="text1"),
            RetrievalResult(chunk_id="chunk_2", score=0.8, text="text2"),
            RetrievalResult(chunk_id="chunk_3", score=0.7, text="text3"),
        ]
        sparse_results = [
            RetrievalResult(chunk_id="chunk_1", score=0.9, text="text1"),
            RetrievalResult(chunk_id="chunk_2", score=0.8, text="text2"),
            RetrievalResult(chunk_id="chunk_3", score=0.7, text="text3"),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse(dense_results, sparse_results)

        assert len(fused) == 3
        assert fused[0].chunk_id == "chunk_1"
        assert fused[1].chunk_id == "chunk_2"
        assert fused[2].chunk_id == "chunk_3"
        # Scores should be updated to RRF scores
        assert fused[0].score > fused[1].score > fused[2].score

    def test_fusion_with_different_rankings(self):
        """RRF should rerank based on combined scores."""
        dense_results = [
            RetrievalResult(chunk_id="chunk_A", score=0.9, text="textA"),
            RetrievalResult(chunk_id="chunk_B", score=0.8, text="textB"),
            RetrievalResult(chunk_id="chunk_C", score=0.7, text="textC"),
        ]
        sparse_results = [
            RetrievalResult(chunk_id="chunk_B", score=0.95, text="textB"),
            RetrievalResult(chunk_id="chunk_A", score=0.80, text="textA"),
            RetrievalResult(chunk_id="chunk_D", score=0.7, text="textD"),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse(dense_results, sparse_results)

        # chunk_A and chunk_B appear in both, chunk_C and chunk_D in one each
        assert len(fused) == 4
        # chunk_A has rank 1 in dense (1/61) and rank 2 in sparse (1/62) = highest combined
        # chunk_B has rank 2 in dense (1/62) and rank 1 in sparse (1/61) = second highest
        assert fused[0].chunk_id == "chunk_A"
        assert fused[1].chunk_id == "chunk_B"

    def test_fusion_deterministic_ordering(self):
        """Multiple fusions with same inputs should produce same output."""
        dense_results = [
            RetrievalResult(chunk_id="chunk_1", score=0.9, text="text1"),
            RetrievalResult(chunk_id="chunk_2", score=0.8, text="text2"),
        ]
        sparse_results = [
            RetrievalResult(chunk_id="chunk_2", score=0.85, text="text2"),
            RetrievalResult(chunk_id="chunk_1", score=0.75, text="text1"),
        ]

        fusion = RRFFusion(k=60)
        result1 = fusion.fuse(dense_results, sparse_results)
        result2 = fusion.fuse(dense_results, sparse_results)

        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert r1.chunk_id == r2.chunk_id
            assert abs(r1.score - r2.score) < 1e-9

    def test_fusion_empty_dense(self):
        """Fusion with empty dense results should return sparse results."""
        sparse_results = [
            RetrievalResult(chunk_id="chunk_1", score=0.9, text="text1"),
            RetrievalResult(chunk_id="chunk_2", score=0.8, text="text2"),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse([], sparse_results)

        assert len(fused) == 2
        assert fused[0].chunk_id == "chunk_1"
        assert fused[1].chunk_id == "chunk_2"

    def test_fusion_empty_sparse(self):
        """Fusion with empty sparse results should return dense results."""
        dense_results = [
            RetrievalResult(chunk_id="chunk_1", score=0.9, text="text1"),
            RetrievalResult(chunk_id="chunk_2", score=0.8, text="text2"),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse(dense_results, [])

        assert len(fused) == 2
        assert fused[0].chunk_id == "chunk_1"
        assert fused[1].chunk_id == "chunk_2"

    def test_fusion_both_empty(self):
        """Fusion with both empty should return empty list."""
        fusion = RRFFusion(k=60)
        fused = fusion.fuse([], [])

        assert len(fused) == 0

    def test_fusion_with_configurable_k(self):
        """Different k values should produce different fusion scores."""
        dense_results = [
            RetrievalResult(chunk_id="chunk_A", score=0.9, text="textA"),
            RetrievalResult(chunk_id="chunk_B", score=0.8, text="textB"),
        ]
        sparse_results = [
            RetrievalResult(chunk_id="chunk_B", score=0.95, text="textB"),
            RetrievalResult(chunk_id="chunk_C", score=0.7, text="textC"),
        ]

        fusion_k60 = RRFFusion(k=60)
        fusion_k10 = RRFFusion(k=10)

        fused_k60 = fusion_k60.fuse(dense_results, sparse_results)
        fused_k10 = fusion_k10.fuse(dense_results, sparse_results)

        # Both should have same order but different scores
        assert fused_k60[0].chunk_id == fused_k10[0].chunk_id
        assert abs(fused_k60[0].score - fused_k10[0].score) > 1e-6


class TestRRFFusionScoreCalculation:
    """Tests for RRF score calculation correctness."""

    def test_rrf_score_formula(self):
        """Verify RRF score follows: sum(1/(k+rank)) formula."""
        dense_results = [
            RetrievalResult(chunk_id="chunk_A", score=0.9, text="textA"),
        ]
        sparse_results = [
            RetrievalResult(chunk_id="chunk_A", score=0.85, text="textA"),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse(dense_results, sparse_results)

        # chunk_A should have score = 1/(60+1) + 1/(60+1) = 2/61
        expected_score = 2.0 / 61.0
        assert abs(fused[0].score - expected_score) < 1e-9

    def test_rrf_score_single_source(self):
        """Items appearing in only one source should have proportional scores."""
        dense_results = [
            RetrievalResult(chunk_id="chunk_A", score=0.9, text="textA"),
            RetrievalResult(chunk_id="chunk_B", score=0.8, text="textB"),
        ]
        sparse_results = [
            RetrievalResult(chunk_id="chunk_A", score=0.85, text="textA"),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse(dense_results, sparse_results)

        # chunk_A: 1/61 + 1/61 = 2/61
        # chunk_B: 1/62 (only in dense, rank 2)
        assert fused[0].chunk_id == "chunk_A"
        assert fused[1].chunk_id == "chunk_B"
        # chunk_A score should be greater
        assert fused[0].score > fused[1].score


class TestRRFFusionEdgeCases:
    """Edge case and boundary tests."""

    def test_fusion_preserves_metadata(self):
        """Fusion should preserve chunk metadata."""
        dense_results = [
            RetrievalResult(
                chunk_id="chunk_1",
                score=0.9,
                text="text1",
                metadata={"source": "doc1.pdf", "page": 5},
            ),
        ]
        sparse_results = [
            RetrievalResult(
                chunk_id="chunk_1",
                score=0.85,
                text="text1",
                metadata={"source": "doc1.pdf", "page": 5},
            ),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse(dense_results, sparse_results)

        assert fused[0].metadata["source"] == "doc1.pdf"
        assert fused[0].metadata["page"] == 5

    def test_fusion_with_single_result(self):
        """Fusion with single result in each list should work."""
        dense_results = [
            RetrievalResult(chunk_id="chunk_1", score=0.9, text="text1"),
        ]
        sparse_results = [
            RetrievalResult(chunk_id="chunk_2", score=0.85, text="text2"),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse(dense_results, sparse_results)

        assert len(fused) == 2
        # chunk_1 has rank 1 in dense (1/61), not in sparse
        # chunk_2 has rank 1 in sparse (1/61), not in dense
        # They have equal scores, order by appearance (dense first)
        assert fused[0].chunk_id == "chunk_1"
        assert fused[1].chunk_id == "chunk_2"

    def test_fusion_with_large_result_sets(self):
        """Fusion should handle larger result sets correctly."""
        dense_results = [
            RetrievalResult(chunk_id=f"chunk_{i}", score=1.0 - i*0.01, text=f"text{i}")
            for i in range(20)
        ]
        sparse_results = [
            RetrievalResult(chunk_id=f"chunk_{i}", score=0.9 - i*0.01, text=f"text{i}")
            for i in range(15, 35)
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse(dense_results, sparse_results)

        # Should have 35 unique chunks
        assert len(fused) == 35
        # Chunks in both lists should rank higher
        chunk_ids = [r.chunk_id for r in fused]
        # chunk_15 to chunk_19 appear in both, should be in top ranks
        common_chunks = [f"chunk_{i}" for i in range(15, 20)]
        common_positions = [chunk_ids.index(c) for c in common_chunks]
        assert all(pos < 10 for pos in common_positions), \
            "Chunks appearing in both lists should rank high"


class TestRRFFusionIntegration:
    """Integration-like tests with realistic scenarios."""

    def test_fusion_realistic_scenario(self):
        """Test with realistic dense/sparse retrieval results."""
        # Dense retrieval focuses on semantic similarity
        dense_results = [
            RetrievalResult(chunk_id="config_azure_001", score=0.92, text="Azure configuration..."),
            RetrievalResult(chunk_id="setup_guide_005", score=0.85, text="Setup guide..."),
            RetrievalResult(chunk_id="api_keys_003", score=0.78, text="API keys..."),
            RetrievalResult(chunk_id="troubleshoot_001", score=0.71, text="Troubleshooting..."),
        ]

        # Sparse retrieval focuses on keyword matching
        sparse_results = [
            RetrievalResult(chunk_id="api_keys_003", score=0.88, text="API keys..."),
            RetrievalResult(chunk_id="config_azure_001", score=0.84, text="Azure configuration..."),
            RetrievalResult(chunk_id="auth_guide_002", score=0.76, text="Authentication guide..."),
            RetrievalResult(chunk_id="setup_guide_005", score=0.72, text="Setup guide..."),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse(dense_results, sparse_results)

        # All unique chunks should be present
        assert len(fused) == 5
        # Chunks appearing in both should rank high
        top_3 = [r.chunk_id for r in fused[:3]]
        assert "config_azure_001" in top_3
        assert "setup_guide_005" in top_3
        assert "api_keys_003" in top_3
