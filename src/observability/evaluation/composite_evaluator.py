"""
CompositeEvaluator: Combines multiple evaluators and aggregates their results.

Implements BaseEvaluator interface to compose multiple evaluator instances,
executing them in parallel (when possible) and merging their metric outputs.
"""

from typing import Dict, List, Any, Optional
import logging

from src.libs.evaluator.base_evaluator import BaseEvaluator


logger = logging.getLogger(__name__)


class CompositeEvaluator(BaseEvaluator):
    """
    Composite evaluator that aggregates multiple evaluators.

    Combines results from multiple BaseEvaluator instances, enabling:
    - Multi-criteria evaluation (combine custom + Ragas + other evaluators)
    - Fault tolerance (continues if one evaluator fails)
    - Flexible metric aggregation
    """

    def __init__(self, evaluators: List[BaseEvaluator]):
        """
        Initialize CompositeEvaluator with a list of evaluators.

        Args:
            evaluators: List of BaseEvaluator instances to compose

        Raises:
            ValueError: If evaluators list is empty
        """
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator")

        self.evaluators = evaluators

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Evaluate using all child evaluators and aggregate results.

        Executes each evaluator and merges their metrics. If an evaluator
        fails, logs a warning but continues with others.

        Args:
            query: The query string
            retrieved_ids: List of IDs retrieved by the system
            golden_ids: List of IDs that are golden standard (expected correct)
            trace: Optional TraceContext for tracking

        Returns:
            Dictionary combining metrics from all evaluators:
            {
                'metric1_from_evaluator1': float,
                'metric2_from_evaluator1': float,
                'metric1_from_evaluator2': float,
                ...
            }

        Raises:
            ValueError: If input validation fails
        """
        aggregated_metrics: Dict[str, float] = {}

        # Execute each evaluator
        for i, evaluator in enumerate(self.evaluators):
            try:
                # Get metrics from this evaluator
                metrics = evaluator.evaluate(
                    query=query,
                    retrieved_ids=retrieved_ids,
                    golden_ids=golden_ids,
                    trace=trace
                )

                # Merge metrics with prefix to avoid collisions
                if metrics:
                    # Include evaluator index if there are name collisions
                    evaluator_class_name = evaluator.__class__.__name__
                    for metric_name, value in metrics.items():
                        # Check if metric already exists
                        if metric_name in aggregated_metrics:
                            # Add evaluator index prefix to distinguish
                            new_key = f"{metric_name}_{evaluator_class_name}_{i}"
                            aggregated_metrics[new_key] = value
                            logger.debug(
                                f"Metric name collision for '{metric_name}', "
                                f"renamed to '{new_key}'"
                            )
                        else:
                            aggregated_metrics[metric_name] = value

            except Exception as e:
                # Log error but continue with other evaluators
                evaluator_class_name = self.evaluators[i].__class__.__name__
                logger.warning(
                    f"Evaluator {evaluator_class_name} (index {i}) failed: {str(e)}. "
                    f"Continuing with other evaluators."
                )

        return aggregated_metrics

    def validate_config(self) -> None:
        """
        Validate configuration of composite and all child evaluators.

        Args:
            None

        Raises:
            ValueError: If configuration is invalid or no evaluators present
        """
        if not self.evaluators:
            raise ValueError(
                "CompositeEvaluator must have at least one evaluator configured"
            )

        # Validate each child evaluator
        for i, evaluator in enumerate(self.evaluators):
            try:
                evaluator.validate_config()
            except Exception as e:
                evaluator_class_name = evaluator.__class__.__name__
                raise ValueError(
                    f"Child evaluator {evaluator_class_name} (index {i}) validation failed: {str(e)}"
                ) from e
