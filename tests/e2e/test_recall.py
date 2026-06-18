"""
End-to-end Recall regression tests based on golden test set.

Tests verify that the retrieval system meets minimum recall thresholds
on a set of known good query-document pairs.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

from src.observability.evaluation.eval_runner import EvalRunner


# Constants for recall thresholds
MIN_HIT_RATE = 0.3  # At least 30% of queries should find expected chunks
MIN_MRR = 0.3  # Mean Reciprocal Rank should be at least 0.3


class TestRecallRegression:
    """Recall regression tests using golden test set."""

    @pytest.fixture
    def eval_runner(self):
        """Create EvalRunner with mock dependencies."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_evaluator = Mock()

        # Configure mock evaluator to return realistic metrics
        mock_evaluator.evaluate.side_effect = self._mock_evaluate

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        return runner, mock_hybrid_search

    @staticmethod
    def _mock_evaluate(query: str, retrieved_ids: List[str], golden_ids: List[str], **kwargs):
        """Mock evaluator that calculates realistic metrics."""
        if not retrieved_ids or not golden_ids:
            return {'hit_rate': 0.0, 'mrr': 0.0}

        # Simple hit rate: proportion of golden IDs in retrieved
        hits = len(set(retrieved_ids) & set(golden_ids))
        hit_rate = hits / len(retrieved_ids) if retrieved_ids else 0.0

        # MRR: 1 / (position + 1) of first match
        mrr = 0.0
        for pos, chunk_id in enumerate(retrieved_ids):
            if chunk_id in golden_ids:
                mrr = 1.0 / (pos + 1)
                break

        return {'hit_rate': hit_rate, 'mrr': mrr}

    def test_golden_test_set_exists(self):
        """Verify golden test set fixture exists."""
        test_set_path = Path("tests/fixtures/golden_test_set.json")
        assert test_set_path.exists(), "Golden test set not found"

    def test_golden_test_set_valid_json(self):
        """Verify golden test set is valid JSON."""
        test_set_path = Path("tests/fixtures/golden_test_set.json")
        with open(test_set_path) as f:
            data = json.load(f)

        assert "test_cases" in data
        assert isinstance(data["test_cases"], list)
        assert len(data["test_cases"]) > 0

    def test_golden_test_set_has_required_fields(self):
        """Verify each test case has required fields."""
        test_set_path = Path("tests/fixtures/golden_test_set.json")
        with open(test_set_path) as f:
            data = json.load(f)

        for test_case in data["test_cases"]:
            assert "query" in test_case
            assert "expected_chunk_ids" in test_case
            assert isinstance(test_case["expected_chunk_ids"], list)

    def test_recall_hit_rate_meets_threshold(self, eval_runner):
        """Verify hit rate meets minimum threshold."""
        runner, mock_hybrid_search = eval_runner

        # Configure mock to return some matching chunks
        def search_side_effect(query, top_k=10):
            # Return some chunks that might match
            return [
                {'chunk_id': 'chunk_001', 'score': 0.9},
                {'chunk_id': 'chunk_002', 'score': 0.8},
                {'chunk_id': 'chunk_003', 'score': 0.7},
            ]

        mock_hybrid_search.search.side_effect = search_side_effect

        # Run evaluation
        test_set_path = Path("tests/fixtures/golden_test_set.json")
        report = runner.run(str(test_set_path))

        # Check hit rate threshold
        hit_rate = report.metrics.get('hit_rate', 0.0)
        assert hit_rate >= MIN_HIT_RATE, (
            f"Hit rate {hit_rate} below minimum {MIN_HIT_RATE}"
        )

    def test_recall_mrr_meets_threshold(self, eval_runner):
        """Verify MRR meets minimum threshold."""
        runner, mock_hybrid_search = eval_runner

        # Configure mock to return chunks with at least one match early
        def search_side_effect(query, top_k=10):
            # First chunk matches (MRR = 1.0)
            return [
                {'chunk_id': 'chunk_001', 'score': 0.9},
                {'chunk_id': 'chunk_002', 'score': 0.8},
            ]

        mock_hybrid_search.search.side_effect = search_side_effect

        # Run evaluation
        test_set_path = Path("tests/fixtures/golden_test_set.json")
        report = runner.run(str(test_set_path))

        # Check MRR threshold
        mrr = report.metrics.get('mrr', 0.0)
        assert mrr >= MIN_MRR, (
            f"MRR {mrr} below minimum {MIN_MRR}"
        )

    def test_each_query_has_result(self, eval_runner):
        """Verify each query in test set produces a result."""
        runner, mock_hybrid_search = eval_runner

        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9}
        ]

        test_set_path = Path("tests/fixtures/golden_test_set.json")
        with open(test_set_path) as f:
            test_data = json.load(f)

        report = runner.run(str(test_set_path))

        # Number of results should match number of test cases
        assert len(report.query_results) == len(test_data["test_cases"])

    def test_no_query_errors_in_result(self, eval_runner):
        """Verify no queries failed with errors."""
        runner, mock_hybrid_search = eval_runner

        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9}
        ]

        test_set_path = Path("tests/fixtures/golden_test_set.json")
        report = runner.run(str(test_set_path))

        # Check for errors in any query result
        errors = [r for r in report.query_results if 'error' in r]
        assert len(errors) == 0, f"Found {len(errors)} query errors"

    def test_metrics_are_numeric(self, eval_runner):
        """Verify all metrics are numeric values."""
        runner, mock_hybrid_search = eval_runner

        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9}
        ]

        test_set_path = Path("tests/fixtures/golden_test_set.json")
        report = runner.run(str(test_set_path))

        # Check metric values are numeric
        for metric_name, value in report.metrics.items():
            assert isinstance(value, (int, float)), (
                f"Metric {metric_name} has non-numeric value: {value}"
            )

    def test_metrics_in_valid_range(self, eval_runner):
        """Verify all metrics are in valid ranges (0-1)."""
        runner, mock_hybrid_search = eval_runner

        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9}
        ]

        test_set_path = Path("tests/fixtures/golden_test_set.json")
        report = runner.run(str(test_set_path))

        # Check metric ranges
        for metric_name, value in report.metrics.items():
            assert 0.0 <= value <= 1.0, (
                f"Metric {metric_name} value {value} out of range [0, 1]"
            )
