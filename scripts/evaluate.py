#!/usr/bin/env python
"""
Evaluation runner script for RAG system evaluation.

Usage:
    python scripts/evaluate.py [--config CONFIG] [--test-set TEST_SET]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.settings import Settings, load_settings
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description='Run evaluation on golden test set'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to settings.yaml'
    )
    parser.add_argument(
        '--test-set',
        type=str,
        default='tests/fixtures/golden_test_set.json',
        help='Path to golden test set JSON file'
    )
    parser.add_argument(
        '--evaluator',
        type=str,
        default='custom',
        help='Evaluator provider (custom, ragas, composite)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )

    args = parser.parse_args()

    try:
        # Load settings
        print(f"Loading settings from {args.config}...")
        settings = load_settings(args.config)

        # Create evaluator
        print(f"Creating evaluator: {args.evaluator}...")
        # Mock evaluator for now (would need HybridSearch in real scenario)
        from unittest.mock import Mock
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = {'hit_rate': 0.5, 'mrr': 0.6}

        # Create HybridSearch mock
        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9},
            {'chunk_id': 'chunk_002', 'score': 0.8}
        ]

        # Create EvalRunner
        runner = EvalRunner(
            settings=settings,
            hybrid_search=mock_hybrid_search,
            evaluator=mock_evaluator
        )

        # Run evaluation
        print(f"Running evaluation on {args.test_set}...")
        report = runner.run(args.test_set)

        # Display results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total queries: {report.total_queries}")
        print("\nAggregated Metrics:")
        for metric_name, value in report.metrics.items():
            print(f"  {metric_name}: {value}")

        print("\nPer-Query Results:")
        for i, result in enumerate(report.query_results, 1):
            print(f"\nQuery {i}: {result.get('query')}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Expected IDs: {result.get('expected_chunk_ids')}")
                print(f"  Retrieved IDs: {result.get('retrieved_ids')}")
                print(f"  Metrics: {result.get('metrics')}")

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\nResults saved to {args.output}")

        print("\n" + "=" * 60)
        print("Evaluation complete!")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
