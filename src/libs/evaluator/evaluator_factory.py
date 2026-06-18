"""Evaluator Factory for provider routing and instantiation."""

from typing import Dict, Type, List, Any, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Custom lightweight evaluator for standard metrics (hit_rate, mrr)."""

    def __init__(self, settings):
        self.settings = settings

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results against golden standard.

        Calculates:
        - hit_rate: Number of golden IDs in top-k retrieved / total retrieved
        - mrr: Mean Reciprocal Rank (1/(position+1) of first match, or 0)

        Args:
            query: The query string (for context)
            retrieved_ids: List of IDs retrieved by the system
            golden_ids: List of IDs that are golden standard (expected correct)
            trace: Optional TraceContext for tracking

        Returns:
            Dictionary with hit_rate and mrr metrics

        Raises:
            ValueError: If input validation fails
        """
        if not retrieved_ids:
            return {"hit_rate": 0.0, "mrr": 0.0}

        if not golden_ids:
            return {"hit_rate": 0.0, "mrr": 0.0}

        # Convert to sets for intersection
        retrieved_set = set(retrieved_ids)
        golden_set = set(golden_ids)

        # Calculate hit_rate: how many golden IDs are in retrieved list
        hits = len(retrieved_set & golden_set)
        hit_rate = hits / len(retrieved_ids)

        # Calculate MRR: reciprocal rank of first match
        mrr = 0.0
        for position, chunk_id in enumerate(retrieved_ids):
            if chunk_id in golden_set:
                mrr = 1.0 / (position + 1)
                break

        return {"hit_rate": hit_rate, "mrr": mrr}

    def validate_config(self) -> None:
        """Validate provider configuration."""
        if not self.settings:
            raise ValueError("Settings object is required")


class EvaluatorFactory:
    """Factory for creating Evaluator instances based on settings."""

    # Provider registry mapping
    _providers: Dict[str, Type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
    }

    def create(self, settings) -> BaseEvaluator:
        """
        Create an Evaluator instance based on settings.

        Args:
            settings: EvaluatorSettings object with provider configuration

        Returns:
            BaseEvaluator instance

        Raises:
            ValueError: If provider is unknown or configuration is invalid
        """
        # Validate basic settings
        if not settings.provider:
            raise ValueError("Evaluator provider is required")

        provider = settings.provider.lower()

        if provider not in self._providers:
            raise ValueError(
                f"Unknown Evaluator provider: {provider}. "
                f"Supported providers: {', '.join(self._providers.keys())}"
            )

        provider_class = self._providers[provider]
        evaluator = provider_class(settings)

        # Validate provider-specific config
        evaluator.validate_config()

        return evaluator

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseEvaluator]) -> None:
        """
        Register a new Evaluator provider.

        Args:
            name: Provider name (lowercase)
            provider_class: Class that extends BaseEvaluator
        """
        cls._providers[name.lower()] = provider_class

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider names.

        Returns:
            List of provider names (lowercase)
        """
        return list(cls._providers.keys())


# Lazy registration of RagasEvaluator to avoid hard dependency
def _register_ragas():
    """Register RagasEvaluator if ragas library is available."""
    try:
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator
        EvaluatorFactory.register_provider("ragas", RagasEvaluator)
    except ImportError:
        # ragas not installed, skip registration
        pass


# Register ragas provider on module load
_register_ragas()
