"""
Unit tests for RagasEvaluator implementation.

Tests verify the Ragas framework integration for RAG evaluation metrics.
Uses mocking to avoid requiring ragas library in unit tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from src.libs.evaluator.base_evaluator import BaseEvaluator


class TestRagasEvaluatorInitialization:
    """Test RagasEvaluator initialization and configuration."""

    def test_ragas_evaluator_can_be_instantiated(self):
        """Verify RagasEvaluator can be created with settings."""
        mock_settings = Mock()
        mock_settings.provider = "ragas"

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)
        assert isinstance(evaluator, BaseEvaluator)
        assert isinstance(evaluator, RagasEvaluator)

    def test_ragas_evaluator_inherits_from_base_evaluator(self):
        """Verify RagasEvaluator correctly inherits from BaseEvaluator."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(evaluator, 'validate_config')


class TestRagasEvaluatorMetrics:
    """Test Ragas evaluation metrics calculation."""

    def test_evaluate_returns_dict_with_required_metrics(self):
        """Verify evaluate() returns dict with faithfulness and answer_relevancy metrics."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        # Prepare test data with chunk IDs format
        query = "What is RAG?"
        retrieved_ids = ["chunk_001", "chunk_002", "chunk_003"]
        golden_ids = ["chunk_001", "chunk_003"]

        # Execute evaluation
        result = evaluator.evaluate(
            query=query,
            retrieved_ids=retrieved_ids,
            golden_ids=golden_ids
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert 'faithfulness' in result
        assert 'answer_relevancy' in result

    def test_evaluate_returns_numeric_metric_values(self):
        """Verify all returned metrics are numeric."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        result = evaluator.evaluate(
            query="test query",
            retrieved_ids=["chunk_001", "chunk_002"],
            golden_ids=["chunk_001"]
        )

        # Verify all values are numeric
        for key, value in result.items():
            assert isinstance(value, (int, float))

    def test_evaluate_with_multiple_retrieved_ids(self):
        """Verify evaluate() handles multiple retrieved IDs correctly."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        result = evaluator.evaluate(
            query="test query",
            retrieved_ids=[
                "chunk_001",
                "chunk_002",
                "chunk_003",
                "chunk_004",
                "chunk_005"
            ],
            golden_ids=["chunk_001", "chunk_003", "chunk_005"]
        )

        assert isinstance(result, dict)
        assert len(result) > 0


class TestRagasEvaluatorErrorHandling:
    """Test error handling in RagasEvaluator."""

    def test_validate_config_with_no_settings_raises_error(self):
        """Verify validate_config() raises ValueError when settings is None."""
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(None)
        with pytest.raises(ValueError):
            evaluator.validate_config()

    def test_evaluate_with_empty_retrieved_ids_returns_zero_metrics(self):
        """Verify evaluate() handles empty retrieved IDs gracefully."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        result = evaluator.evaluate(
            query="test query",
            retrieved_ids=[],
            golden_ids=["chunk_001"]
        )

        assert isinstance(result, dict)
        assert result.get('faithfulness', 0) == 0.0
        assert result.get('answer_relevancy', 0) == 0.0

    def test_evaluate_with_empty_golden_ids_returns_zero_metrics(self):
        """Verify evaluate() handles empty golden IDs gracefully."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        result = evaluator.evaluate(
            query="test query",
            retrieved_ids=["chunk_001"],
            golden_ids=[]
        )

        assert isinstance(result, dict)
        assert result.get('faithfulness', 0) == 0.0
        assert result.get('answer_relevancy', 0) == 0.0


class TestRagasEvaluatorInterfaceContract:
    """Test RagasEvaluator adheres to BaseEvaluator interface contract."""

    def test_evaluate_method_signature_matches_base(self):
        """Verify evaluate() signature matches BaseEvaluator contract."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        # Verify method exists and is callable
        assert hasattr(evaluator, 'evaluate')
        assert callable(evaluator.evaluate)

    def test_evaluate_accepts_required_parameters(self):
        """Verify evaluate() accepts all required parameters."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        # Verify parameter acceptance
        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"]
        )

        assert result is not None

    def test_evaluate_accepts_optional_trace_parameter(self):
        """Verify evaluate() accepts optional trace parameter."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        mock_trace = Mock()
        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"],
            trace=mock_trace
        )

        assert result is not None


class TestRagasEvaluatorMetricRanges:
    """Test that returned metrics are within expected ranges."""

    def test_faithfulness_metric_in_valid_range(self):
        """Verify faithfulness metric is between 0 and 1."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["chunk_001", "chunk_002"],
            golden_ids=["chunk_001"]
        )

        faithfulness = result.get('faithfulness')
        assert 0 <= faithfulness <= 1

    def test_answer_relevancy_metric_in_valid_range(self):
        """Verify answer_relevancy metric is between 0 and 1."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["chunk_001"],
            golden_ids=["chunk_001"]
        )

        answer_relevancy = result.get('answer_relevancy')
        assert 0 <= answer_relevancy <= 1

    def test_all_metrics_in_valid_range(self):
        """Verify all returned metrics are in valid range (0-1)."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["chunk_001", "chunk_002", "chunk_003"],
            golden_ids=["chunk_001"]
        )

        for metric_name, value in result.items():
            assert 0 <= value <= 1, f"{metric_name} value {value} out of range"


class TestRagasEvaluatorDeterminism:
    """Test determinism of RagasEvaluator results."""

    def test_same_input_produces_deterministic_results(self):
        """Verify same input produces same or similar output."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        # Execute twice with same inputs
        result1 = evaluator.evaluate(
            query="test query",
            retrieved_ids=["chunk_001", "chunk_002"],
            golden_ids=["chunk_001"]
        )

        result2 = evaluator.evaluate(
            query="test query",
            retrieved_ids=["chunk_001", "chunk_002"],
            golden_ids=["chunk_001"]
        )

        # Results should be identical
        assert result1 == result2


class TestRagasEvaluatorValidateConfig:
    """Test configuration validation."""

    def test_validate_config_with_valid_settings(self):
        """Verify validate_config() passes with valid settings."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        # Should not raise
        evaluator.validate_config()

    def test_validate_config_stores_settings(self):
        """Verify validate_config() validates that settings are stored."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        assert evaluator.settings is not None


class TestRagasEvaluatorIntegration:
    """Integration tests for RagasEvaluator with factory."""

    def test_ragas_evaluator_can_be_created_and_used(self):
        """Verify RagasEvaluator can be instantiated and used."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)
        assert isinstance(evaluator, BaseEvaluator)

    def test_ragas_evaluator_provider_name(self):
        """Verify RagasEvaluator has identifiable provider name."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        assert hasattr(evaluator, '__class__')
        assert evaluator.__class__.__name__ == 'RagasEvaluator'

    def test_evaluate_perfect_recall(self):
        """Verify evaluate() with perfect recall returns appropriate metrics."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["chunk_001", "chunk_002", "chunk_003"],
            golden_ids=["chunk_001", "chunk_002", "chunk_003"]
        )

        # Perfect recall should give high metrics
        assert result['faithfulness'] >= 0.8
        assert result['answer_relevancy'] >= 0.8

    def test_evaluate_no_recall(self):
        """Verify evaluate() with no matching IDs returns low metrics."""
        mock_settings = Mock()

        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(mock_settings)

        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["chunk_001", "chunk_002"],
            golden_ids=["chunk_003", "chunk_004"]
        )

        # No matching IDs should give low metrics
        assert result['faithfulness'] <= 0.2
        assert result['answer_relevancy'] <= 0.2
