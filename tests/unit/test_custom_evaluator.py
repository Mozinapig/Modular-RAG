"""Unit tests for CustomEvaluator and EvaluatorFactory."""

import pytest
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory, CustomEvaluator
from src.core.settings import Settings


@pytest.fixture
def evaluator_settings():
    """Create evaluator settings."""
    class MockEvaluatorSettings:
        provider = "custom"
    return MockEvaluatorSettings()


@pytest.fixture
def custom_evaluator(evaluator_settings):
    """Create CustomEvaluator instance."""
    return CustomEvaluator(evaluator_settings)


class TestCustomEvaluator:
    """Test cases for CustomEvaluator."""

    def test_evaluate_perfect_hit(self, custom_evaluator):
        """Test evaluation when all retrieved IDs match golden IDs."""
        query = "test query"
        retrieved_ids = ["chunk_1", "chunk_2", "chunk_3"]
        golden_ids = ["chunk_1", "chunk_2", "chunk_3"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert "hit_rate" in metrics
        assert "mrr" in metrics
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_evaluate_partial_hit(self, custom_evaluator):
        """Test evaluation when some retrieved IDs match golden IDs."""
        query = "test query"
        retrieved_ids = ["chunk_1", "chunk_2", "chunk_3"]
        golden_ids = ["chunk_1", "chunk_4", "chunk_5"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 1/3  # 1 out of 3 retrieved
        assert metrics["mrr"] > 0  # First match at position 0

    def test_evaluate_mrr_position(self, custom_evaluator):
        """Test MRR calculation with match at different positions."""
        query = "test query"
        retrieved_ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
        golden_ids = ["chunk_3"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        # chunk_3 is at position 2 (0-indexed), so MRR = 1/(2+1) = 1/3
        assert metrics["mrr"] == pytest.approx(1/3)
        assert metrics["hit_rate"] == 1/4

    def test_evaluate_no_hit(self, custom_evaluator):
        """Test evaluation when no retrieved IDs match golden IDs."""
        query = "test query"
        retrieved_ids = ["chunk_1", "chunk_2", "chunk_3"]
        golden_ids = ["chunk_4", "chunk_5"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_evaluate_empty_retrieved(self, custom_evaluator):
        """Test evaluation with empty retrieved list."""
        query = "test query"
        retrieved_ids = []
        golden_ids = ["chunk_1", "chunk_2"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_evaluate_empty_golden(self, custom_evaluator):
        """Test evaluation with empty golden list."""
        query = "test query"
        retrieved_ids = ["chunk_1", "chunk_2"]
        golden_ids = []

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_evaluate_multiple_hits_same_position(self, custom_evaluator):
        """Test evaluation with multiple golden IDs matching at different positions."""
        query = "test query"
        retrieved_ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        golden_ids = ["chunk_2", "chunk_4"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        # hit_rate: 2 out of 5 = 0.4
        assert metrics["hit_rate"] == 2/5
        # MRR: first match at position 1 (chunk_2), so 1/(1+1) = 0.5
        assert metrics["mrr"] == 0.5

    def test_validate_config_success(self, custom_evaluator):
        """Test config validation succeeds."""
        custom_evaluator.validate_config()  # Should not raise

    def test_validate_config_no_settings(self):
        """Test config validation fails without settings."""
        evaluator = CustomEvaluator(None)
        with pytest.raises(ValueError):
            evaluator.validate_config()

    def test_is_base_evaluator_subclass(self, custom_evaluator):
        """Test CustomEvaluator is a BaseEvaluator subclass."""
        assert isinstance(custom_evaluator, BaseEvaluator)

    def test_evaluate_with_trace(self, custom_evaluator):
        """Test evaluation accepts optional trace parameter."""
        query = "test query"
        retrieved_ids = ["chunk_1"]
        golden_ids = ["chunk_1"]
        trace = {"dummy": "trace"}

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids, trace=trace)

        assert metrics["hit_rate"] == 1.0


class TestEvaluatorFactory:
    """Test cases for EvaluatorFactory."""

    def test_create_custom_evaluator(self, evaluator_settings):
        """Test factory creates CustomEvaluator correctly."""
        factory = EvaluatorFactory()
        evaluator = factory.create(evaluator_settings)

        assert isinstance(evaluator, CustomEvaluator)
        assert isinstance(evaluator, BaseEvaluator)

    def test_create_unknown_provider(self):
        """Test factory raises error for unknown provider."""
        class MockSettings:
            provider = "unknown_provider"

        factory = EvaluatorFactory()
        with pytest.raises(ValueError, match="Unknown Evaluator provider"):
            factory.create(MockSettings())

    def test_create_no_provider(self):
        """Test factory raises error when provider is not specified."""
        class MockSettings:
            provider = None

        factory = EvaluatorFactory()
        with pytest.raises(ValueError, match="provider is required"):
            factory.create(MockSettings())

    def test_create_provider_case_insensitive(self, evaluator_settings):
        """Test factory provider lookup is case insensitive."""
        evaluator_settings.provider = "CUSTOM"
        factory = EvaluatorFactory()
        evaluator = factory.create(evaluator_settings)

        assert isinstance(evaluator, CustomEvaluator)

    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        class DummyEvaluator(BaseEvaluator):
            def __init__(self, settings):
                self.settings = settings

            def evaluate(self, query, retrieved_ids, golden_ids, trace=None):
                return {"dummy": 1.0}

            def validate_config(self):
                pass

        class MockSettings:
            provider = "dummy"

        factory = EvaluatorFactory()
        factory.register_provider("dummy", DummyEvaluator)
        evaluator = factory.create(MockSettings())

        assert isinstance(evaluator, DummyEvaluator)

    def test_list_providers(self, evaluator_settings):
        """Test getting list of available providers."""
        factory = EvaluatorFactory()
        providers = factory.list_providers()

        assert "custom" in providers
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_factory_validation_called(self, evaluator_settings):
        """Test factory calls validate_config on created instance."""
        factory = EvaluatorFactory()
        evaluator = factory.create(evaluator_settings)
        # If validate_config raised error, this would fail during creation
        assert evaluator is not None

    def test_evaluate_large_retrieved_list(self, custom_evaluator):
        """Test evaluation with large retrieved candidate list."""
        query = "large query"
        retrieved_ids = [f"chunk_{i}" for i in range(1000)]
        golden_ids = ["chunk_100", "chunk_500"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert "hit_rate" in metrics
        assert "mrr" in metrics
        assert 0 <= metrics["hit_rate"] <= 1
        assert 0 <= metrics["mrr"] <= 1

    def test_evaluate_duplicate_retrieved_ids(self, custom_evaluator):
        """Test evaluation handles duplicate retrieved IDs."""
        query = "test query"
        retrieved_ids = ["chunk_1", "chunk_1", "chunk_2", "chunk_2"]
        golden_ids = ["chunk_1"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 1/4
        assert metrics["mrr"] == 1.0

    def test_evaluate_duplicate_golden_ids(self, custom_evaluator):
        """Test evaluation handles duplicate golden IDs."""
        query = "test query"
        retrieved_ids = ["chunk_1", "chunk_2"]
        golden_ids = ["chunk_1", "chunk_1"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 1/2

    def test_evaluate_special_characters_in_ids(self, custom_evaluator):
        """Test evaluation with special characters in chunk IDs."""
        query = "special test"
        retrieved_ids = ["chunk_001-v2@latest", "chunk_002/draft"]
        golden_ids = ["chunk_001-v2@latest", "chunk_003.final"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 1/2
        assert metrics["mrr"] == 1.0

    def test_evaluate_very_long_query(self, custom_evaluator):
        """Test evaluation with very long query string."""
        query = "test " * 1000  # Very long query
        retrieved_ids = ["chunk_1", "chunk_2"]
        golden_ids = ["chunk_1"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert "hit_rate" in metrics
        assert metrics["hit_rate"] == 1/2

    def test_evaluate_unicode_in_query(self, custom_evaluator):
        """Test evaluation with unicode characters in query."""
        query = "测试查询 🎯 тест query"
        retrieved_ids = ["chunk_1", "chunk_2"]
        golden_ids = ["chunk_1"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 1/2

    def test_evaluate_whitespace_only_query(self, custom_evaluator):
        """Test evaluation with whitespace-only query."""
        query = "   \t\n  "
        retrieved_ids = ["chunk_1"]
        golden_ids = ["chunk_1"]

        metrics = custom_evaluator.evaluate(query, retrieved_ids, golden_ids)

        assert metrics["hit_rate"] == 1.0

    def test_factory_empty_provider_string(self):
        """Test factory rejects empty provider string."""
        class MockSettings:
            provider = ""

        factory = EvaluatorFactory()
        with pytest.raises(ValueError):
            factory.create(MockSettings())
