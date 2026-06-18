"""
EvalRunner: Executes queries from golden test set and produces evaluation reports.

Implements evaluation workflow: loads test cases from JSON, executes queries
through HybridSearch, evaluates results against golden standard, and aggregates
metrics into EvalReport.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator


logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a single test case from golden test set."""

    query: str
    expected_chunk_ids: List[str]
    expected_sources: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create TestCase from dictionary."""
        return cls(
            query=data.get('query', ''),
            expected_chunk_ids=data.get('expected_chunk_ids', []),
            expected_sources=data.get('expected_sources', [])
        )


@dataclass
class EvalReport:
    """Evaluation report containing aggregated metrics and per-query results."""

    total_queries: int
    metrics: Dict[str, float]
    query_results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            'total_queries': self.total_queries,
            'metrics': self.metrics,
            'query_results': self.query_results
        }


class EvalRunner:
    """
    Evaluation runner that executes golden test set and produces metrics.

    Workflow:
    1. Load test cases from JSON file
    2. For each query: execute HybridSearch
    3. Evaluate results against expected chunk IDs
    4. Aggregate metrics across all queries
    5. Return EvalReport with results
    """

    def __init__(
        self,
        settings: Any,
        hybrid_search: Any,
        evaluator: BaseEvaluator
    ):
        """
        Initialize EvalRunner.

        Args:
            settings: Configuration settings object
            hybrid_search: HybridSearch instance for query execution
            evaluator: BaseEvaluator instance for metrics calculation
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator

    def load_test_set(self, test_set_path: str) -> List[TestCase]:
        """
        Load test cases from JSON file.

        Args:
            test_set_path: Path to golden_test_set.json file

        Returns:
            List of TestCase objects

        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If JSON is invalid
        """
        path = Path(test_set_path)

        if not path.exists():
            raise FileNotFoundError(f"Test set file not found: {test_set_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_cases = []
        for case_data in data.get('test_cases', []):
            test_case = TestCase.from_dict(case_data)
            test_cases.append(test_case)

        logger.info(f"Loaded {len(test_cases)} test cases from {test_set_path}")
        return test_cases

    def run(self, test_set_path: str) -> EvalReport:
        """
        Execute evaluation on golden test set.

        Args:
            test_set_path: Path to golden_test_set.json file

        Returns:
            EvalReport with aggregated metrics and per-query results

        Raises:
            FileNotFoundError: If test set file not found
            RuntimeError: If evaluation fails
        """
        # Load test cases
        test_cases = self.load_test_set(test_set_path)

        if not test_cases:
            logger.warning("No test cases loaded")
            return EvalReport(
                total_queries=0,
                metrics={},
                query_results=[]
            )

        # Execute queries and collect results
        query_results = []
        retrieved_ids_per_query = []
        all_evaluator_results = []

        for i, test_case in enumerate(test_cases):
            try:
                # Execute query through HybridSearch
                search_results = self.hybrid_search.search(
                    query=test_case.query,
                    top_k=len(test_case.expected_chunk_ids) + 5
                )

                # Extract chunk IDs from results
                retrieved_ids = [result.get('chunk_id') for result in search_results]
                retrieved_ids_per_query.append(retrieved_ids)

                # Evaluate this query
                query_metrics = self.evaluator.evaluate(
                    query=test_case.query,
                    retrieved_ids=retrieved_ids,
                    golden_ids=test_case.expected_chunk_ids
                )

                all_evaluator_results.append(query_metrics)

                # Record result
                query_result = {
                    'query': test_case.query,
                    'expected_chunk_ids': test_case.expected_chunk_ids,
                    'retrieved_ids': retrieved_ids,
                    'metrics': query_metrics
                }
                query_results.append(query_result)

            except Exception as e:
                logger.error(f"Error evaluating query {i}: {str(e)}")
                query_result = {
                    'query': test_case.query,
                    'expected_chunk_ids': test_case.expected_chunk_ids,
                    'error': str(e)
                }
                query_results.append(query_result)

        # Aggregate metrics across all queries
        aggregated_metrics = self._aggregate_metrics(all_evaluator_results)

        report = EvalReport(
            total_queries=len(test_cases),
            metrics=aggregated_metrics,
            query_results=query_results
        )

        logger.info(
            f"Evaluation complete. Total queries: {len(test_cases)}, "
            f"Aggregated metrics: {aggregated_metrics}"
        )

        return report

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics from multiple queries.

        Simple averaging strategy: for each metric, compute mean across queries.

        Args:
            metrics_list: List of metric dicts from individual queries

        Returns:
            Aggregated metrics dict
        """
        if not metrics_list:
            return {}

        aggregated = {}

        # Get all metric keys from first result
        if metrics_list:
            all_keys = set()
            for metrics in metrics_list:
                all_keys.update(metrics.keys())

            # Average each metric across queries
            for key in all_keys:
                values = []
                for metrics in metrics_list:
                    if key in metrics:
                        values.append(metrics[key])

                if values:
                    # Compute mean
                    mean_value = sum(values) / len(values)
                    aggregated[key] = round(mean_value, 4)

        return aggregated
