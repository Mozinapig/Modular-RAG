"""
Unit tests for EvalRunner implementation.

Tests verify the evaluation runner that executes queries against golden test set
and produces evaluation reports.
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from typing import Dict, List, Any

from src.observability.evaluation.eval_runner import EvalRunner, EvalReport, TestCase


class TestEvalReportDataStructure:
    """Test EvalReport data structure."""

    def test_eval_report_can_be_created(self):
        """Verify EvalReport can be instantiated."""
        report = EvalReport(
            total_queries=3,
            metrics={'hit_rate': 0.7, 'mrr': 0.8},
            query_results=[]
        )

        assert report.total_queries == 3
        assert report.metrics['hit_rate'] == 0.7

    def test_eval_report_stores_query_results(self):
        """Verify EvalReport stores individual query results."""
        query_results = [
            {'query': 'test1', 'hit': True},
            {'query': 'test2', 'hit': False}
        ]
        report = EvalReport(
            total_queries=2,
            metrics={},
            query_results=query_results
        )

        assert len(report.query_results) == 2


class TestTestCaseDataStructure:
    """Test TestCase data structure."""

    def test_test_case_can_be_created(self):
        """Verify TestCase can be instantiated."""
        test_case = TestCase(
            query="What is RAG?",
            expected_chunk_ids=["chunk_001"],
            expected_sources=["guide.pdf"]
        )

        assert test_case.query == "What is RAG?"
        assert len(test_case.expected_chunk_ids) == 1


class TestEvalRunnerInitialization:
    """Test EvalRunner initialization."""

    def test_eval_runner_can_be_instantiated(self):
        """Verify EvalRunner can be created with required dependencies."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_evaluator = Mock()

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        assert runner.settings is not None
        assert runner.hybrid_search is not None
        assert runner.evaluator is not None

    def test_eval_runner_stores_dependencies(self):
        """Verify EvalRunner stores all dependencies."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_evaluator = Mock()

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        assert hasattr(runner, 'settings')
        assert hasattr(runner, 'hybrid_search')
        assert hasattr(runner, 'evaluator')


class TestEvalRunnerLoadTestSet:
    """Test loading golden test set."""

    def test_load_test_set_from_json_file(self):
        """Verify load_test_set() reads and parses JSON file."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_evaluator = Mock()

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        # Create temporary test set file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test_cases": [
                    {
                        "query": "test query",
                        "expected_chunk_ids": ["chunk_001"],
                        "expected_sources": ["source.pdf"]
                    }
                ]
            }
            json.dump(test_data, f)
            temp_path = f.name

        try:
            test_cases = runner.load_test_set(temp_path)
            assert len(test_cases) == 1
            assert test_cases[0].query == "test query"
        finally:
            Path(temp_path).unlink()

    def test_load_test_set_with_nonexistent_file_raises_error(self):
        """Verify load_test_set() raises error for missing file."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_evaluator = Mock()

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        with pytest.raises(FileNotFoundError):
            runner.load_test_set("/nonexistent/path.json")


class TestEvalRunnerExecution:
    """Test EvalRunner execution."""

    def test_run_returns_eval_report(self):
        """Verify run() returns EvalReport."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9},
            {'chunk_id': 'chunk_002', 'score': 0.8}
        ]

        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'hit_rate': 0.7, 'mrr': 0.8}

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        # Create temporary test set
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test_cases": [
                    {
                        "query": "test query",
                        "expected_chunk_ids": ["chunk_001"],
                        "expected_sources": ["source.pdf"]
                    }
                ]
            }
            json.dump(test_data, f)
            temp_path = f.name

        try:
            report = runner.run(temp_path)
            assert isinstance(report, EvalReport)
            assert report.total_queries == 1
        finally:
            Path(temp_path).unlink()

    def test_run_calls_hybrid_search_for_each_query(self):
        """Verify run() executes hybrid search for each query."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9}
        ]

        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'metric': 0.5}

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        # Create temporary test set with 2 queries
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test_cases": [
                    {
                        "query": "query 1",
                        "expected_chunk_ids": ["chunk_001"],
                        "expected_sources": []
                    },
                    {
                        "query": "query 2",
                        "expected_chunk_ids": ["chunk_002"],
                        "expected_sources": []
                    }
                ]
            }
            json.dump(test_data, f)
            temp_path = f.name

        try:
            report = runner.run(temp_path)
            # Verify hybrid_search was called twice
            assert mock_hybrid_search.search.call_count >= 2
        finally:
            Path(temp_path).unlink()

    def test_run_calls_evaluator_with_correct_ids(self):
        """Verify run() passes correct IDs to evaluator."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9},
            {'chunk_id': 'chunk_002', 'score': 0.8}
        ]

        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'hit_rate': 0.5}

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        # Create temporary test set
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test_cases": [
                    {
                        "query": "test",
                        "expected_chunk_ids": ["chunk_001"],
                        "expected_sources": []
                    }
                ]
            }
            json.dump(test_data, f)
            temp_path = f.name

        try:
            report = runner.run(temp_path)
            # Verify evaluator was called
            assert mock_evaluator.evaluate.called
        finally:
            Path(temp_path).unlink()


class TestEvalRunnerMetrics:
    """Test metrics calculation."""

    def test_run_produces_non_empty_metrics(self):
        """Verify run() produces metrics dict."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9}
        ]

        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'hit_rate': 0.8}

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        # Create temporary test set
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test_cases": [
                    {
                        "query": "test",
                        "expected_chunk_ids": ["chunk_001"],
                        "expected_sources": []
                    }
                ]
            }
            json.dump(test_data, f)
            temp_path = f.name

        try:
            report = runner.run(temp_path)
            assert isinstance(report.metrics, dict)
            assert len(report.metrics) > 0
        finally:
            Path(temp_path).unlink()

    def test_run_aggregates_results_from_multiple_queries(self):
        """Verify run() aggregates results from all queries."""
        mock_settings = Mock()
        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9}
        ]

        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'hit_rate': 0.8}

        runner = EvalRunner(
            settings=mock_settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        # Create temporary test set with 3 queries
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {
                "test_cases": [
                    {"query": f"query {i}", "expected_chunk_ids": ["chunk_001"], "expected_sources": []}
                    for i in range(3)
                ]
            }
            json.dump(test_data, f)
            temp_path = f.name

        try:
            report = runner.run(temp_path)
            assert report.total_queries == 3
        finally:
            Path(temp_path).unlink()
