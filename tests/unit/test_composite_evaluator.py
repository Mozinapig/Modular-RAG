"""
Unit tests for CompositeEvaluator implementation.

Tests verify the composite evaluator that combines multiple evaluators
and aggregates their results.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.observability.evaluation.composite_evaluator import CompositeEvaluator


class TestCompositeEvaluatorInitialization:
    """Test CompositeEvaluator initialization."""

    def test_composite_evaluator_can_be_instantiated_with_single_evaluator(self):
        """Verify CompositeEvaluator can be created with a single evaluator."""
        mock_evaluator = Mock(spec=BaseEvaluator)
        composite = CompositeEvaluator([mock_evaluator])

        assert isinstance(composite, BaseEvaluator)
        assert isinstance(composite, CompositeEvaluator)

    def test_composite_evaluator_can_be_instantiated_with_multiple_evaluators(self):
        """Verify CompositeEvaluator can be created with multiple evaluators."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator2 = Mock(spec=BaseEvaluator)
        mock_evaluator3 = Mock(spec=BaseEvaluator)

        composite = CompositeEvaluator([mock_evaluator1, mock_evaluator2, mock_evaluator3])
        assert isinstance(composite, BaseEvaluator)

    def test_composite_evaluator_stores_evaluators(self):
        """Verify CompositeEvaluator stores the evaluators list."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator2 = Mock(spec=BaseEvaluator)
        evaluators = [mock_evaluator1, mock_evaluator2]

        composite = CompositeEvaluator(evaluators)
        assert hasattr(composite, 'evaluators')


class TestCompositeEvaluatorEvaluation:
    """Test CompositeEvaluator evaluation logic."""

    def test_evaluate_calls_all_evaluators(self):
        """Verify evaluate() calls all registered evaluators."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator1.evaluate.return_value = {'hit_rate': 0.8, 'mrr': 0.9}

        mock_evaluator2 = Mock(spec=BaseEvaluator)
        mock_evaluator2.evaluate.return_value = {'faithfulness': 0.85}

        composite = CompositeEvaluator([mock_evaluator1, mock_evaluator2])

        result = composite.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"]
        )

        # Verify all evaluators were called
        mock_evaluator1.evaluate.assert_called_once()
        mock_evaluator2.evaluate.assert_called_once()

    def test_evaluate_returns_dict(self):
        """Verify evaluate() returns a dictionary."""
        mock_evaluator = Mock(spec=BaseEvaluator)
        mock_evaluator.evaluate.return_value = {'hit_rate': 0.8}

        composite = CompositeEvaluator([mock_evaluator])
        result = composite.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"]
        )

        assert isinstance(result, dict)

    def test_evaluate_merges_metrics_from_multiple_evaluators(self):
        """Verify evaluate() merges metrics from all evaluators."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator1.evaluate.return_value = {'hit_rate': 0.8, 'mrr': 0.9}

        mock_evaluator2 = Mock(spec=BaseEvaluator)
        mock_evaluator2.evaluate.return_value = {'faithfulness': 0.85, 'answer_relevancy': 0.90}

        composite = CompositeEvaluator([mock_evaluator1, mock_evaluator2])
        result = composite.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"]
        )

        # All metrics should be in result
        assert 'hit_rate' in result
        assert 'mrr' in result
        assert 'faithfulness' in result
        assert 'answer_relevancy' in result

    def test_evaluate_with_overlapping_metric_names_handles_conflicts(self):
        """Verify evaluate() handles overlapping metric names gracefully."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator1.evaluate.return_value = {'score': 0.8, 'mrr': 0.9}

        mock_evaluator2 = Mock(spec=BaseEvaluator)
        mock_evaluator2.evaluate.return_value = {'score': 0.85, 'other': 0.90}

        composite = CompositeEvaluator([mock_evaluator1, mock_evaluator2])
        result = composite.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"]
        )

        # Result should handle overlapping keys (either last value wins or namespaced)
        assert isinstance(result, dict)


class TestCompositeEvaluatorErrorHandling:
    """Test error handling in CompositeEvaluator."""

    def test_evaluate_with_empty_evaluators_list_raises_on_init(self):
        """Verify CompositeEvaluator raises ValueError on init with empty list."""
        with pytest.raises(ValueError):
            CompositeEvaluator([])

    def test_evaluate_with_failing_evaluator_continues_with_others(self):
        """Verify evaluate() continues if one evaluator fails."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator1.evaluate.side_effect = RuntimeError("Evaluator failed")

        mock_evaluator2 = Mock(spec=BaseEvaluator)
        mock_evaluator2.evaluate.return_value = {'hit_rate': 0.8}

        composite = CompositeEvaluator([mock_evaluator1, mock_evaluator2])

        # Should not raise, should continue with second evaluator
        result = composite.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"]
        )

        # Second evaluator's result should be present
        assert 'hit_rate' in result

    def test_evaluate_passes_parameters_to_all_evaluators(self):
        """Verify evaluate() passes all parameters to each evaluator."""
        mock_evaluator = Mock(spec=BaseEvaluator)
        mock_evaluator.evaluate.return_value = {'metric': 0.5}

        composite = CompositeEvaluator([mock_evaluator])

        query = "test query"
        retrieved_ids = ["chunk_001", "chunk_002"]
        golden_ids = ["chunk_001"]
        mock_trace = Mock()

        composite.evaluate(
            query=query,
            retrieved_ids=retrieved_ids,
            golden_ids=golden_ids,
            trace=mock_trace
        )

        # Verify evaluator was called with correct parameters
        mock_evaluator.evaluate.assert_called_once_with(
            query=query,
            retrieved_ids=retrieved_ids,
            golden_ids=golden_ids,
            trace=mock_trace
        )


class TestCompositeEvaluatorInterfaceContract:
    """Test CompositeEvaluator adheres to BaseEvaluator interface."""

    def test_composite_evaluator_inherits_from_base_evaluator(self):
        """Verify CompositeEvaluator is a BaseEvaluator."""
        mock_evaluator = Mock(spec=BaseEvaluator)
        composite = CompositeEvaluator([mock_evaluator])

        assert isinstance(composite, BaseEvaluator)

    def test_evaluate_method_exists(self):
        """Verify evaluate() method exists and is callable."""
        mock_evaluator = Mock(spec=BaseEvaluator)
        composite = CompositeEvaluator([mock_evaluator])

        assert hasattr(composite, 'evaluate')
        assert callable(composite.evaluate)

    def test_validate_config_method_exists(self):
        """Verify validate_config() method exists and is callable."""
        mock_evaluator = Mock(spec=BaseEvaluator)
        composite = CompositeEvaluator([mock_evaluator])

        assert hasattr(composite, 'validate_config')
        assert callable(composite.validate_config)


class TestCompositeEvaluatorConfiguration:
    """Test CompositeEvaluator configuration handling."""

    def test_validate_config_with_empty_evaluators_raises_error_on_init(self):
        """Verify CompositeEvaluator raises ValueError on init with empty evaluators."""
        with pytest.raises(ValueError):
            CompositeEvaluator([])

    def test_validate_config_with_valid_evaluators_passes(self):
        """Verify validate_config() passes with valid evaluators."""
        mock_evaluator = Mock(spec=BaseEvaluator)
        composite = CompositeEvaluator([mock_evaluator])

        # Should not raise
        composite.validate_config()

    def test_validate_config_calls_validate_config_on_all_evaluators(self):
        """Verify validate_config() validates all child evaluators."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator2 = Mock(spec=BaseEvaluator)

        composite = CompositeEvaluator([mock_evaluator1, mock_evaluator2])
        composite.validate_config()

        # Both should be called
        mock_evaluator1.validate_config.assert_called_once()
        mock_evaluator2.validate_config.assert_called_once()


class TestCompositeEvaluatorMetrics:
    """Test metric aggregation in CompositeEvaluator."""

    def test_all_metrics_are_numeric(self):
        """Verify all returned metrics are numeric."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator1.evaluate.return_value = {'metric1': 0.8}

        mock_evaluator2 = Mock(spec=BaseEvaluator)
        mock_evaluator2.evaluate.return_value = {'metric2': 0.9}

        composite = CompositeEvaluator([mock_evaluator1, mock_evaluator2])
        result = composite.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"]
        )

        for value in result.values():
            assert isinstance(value, (int, float))

    def test_composite_evaluator_with_three_evaluators(self):
        """Verify CompositeEvaluator works with three evaluators."""
        mock_evaluator1 = Mock(spec=BaseEvaluator)
        mock_evaluator1.evaluate.return_value = {'m1': 0.7}

        mock_evaluator2 = Mock(spec=BaseEvaluator)
        mock_evaluator2.evaluate.return_value = {'m2': 0.8}

        mock_evaluator3 = Mock(spec=BaseEvaluator)
        mock_evaluator3.evaluate.return_value = {'m3': 0.9}

        composite = CompositeEvaluator([mock_evaluator1, mock_evaluator2, mock_evaluator3])
        result = composite.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"]
        )

        assert 'm1' in result
        assert 'm2' in result
        assert 'm3' in result
