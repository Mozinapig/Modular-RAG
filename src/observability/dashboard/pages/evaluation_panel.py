"""
Evaluation Panel page for Dashboard

Displays evaluation results from EvalRunner, including metrics and per-query details.
"""

import streamlit as st
import json
from pathlib import Path
from typing import Optional

from src.core.settings import Settings
from src.observability.evaluation.eval_runner import EvalRunner, EvalReport
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.dashboard.services.trace_service import TraceService


def show_evaluation_panel(settings: Settings):
    """Display evaluation panel page."""
    st.title("📊 Evaluation Panel")
    st.write("---")

    # Section 1: Evaluation Configuration
    st.heading("1️⃣ Evaluation Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Evaluator backend selection
        evaluator_options = EvaluatorFactory.list_providers()
        selected_evaluator = st.selectbox(
            "Select Evaluator Backend",
            evaluator_options,
            help="Choose which evaluator to use for metrics calculation"
        )

    with col2:
        # Test set file selection
        test_set_path = st.text_input(
            "Golden Test Set Path",
            value="tests/fixtures/golden_test_set.json",
            help="Path to golden test set JSON file"
        )

    st.write("---")

    # Section 2: Run Evaluation
    st.heading("2️⃣ Run Evaluation")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.info(
            "ℹ️ Click 'Run Evaluation' to execute queries from the golden test set "
            "and calculate evaluation metrics."
        )

    with col2:
        if st.button("▶️ Run Evaluation", use_container_width=True):
            run_evaluation(settings, selected_evaluator, test_set_path)

    st.write("---")

    # Section 3: Recent Results
    st.heading("3️⃣ Recent Evaluation Results")

    # Try to load trace service for historical results
    try:
        trace_service = TraceService()
        traces = trace_service.list_traces(trace_type="evaluation", limit=5)

        if traces:
            st.success(f"Found {len(traces)} recent evaluation runs")

            for i, trace in enumerate(traces, 1):
                with st.expander(f"Evaluation {i} - {trace.get('timestamp', 'N/A')}"):
                    metrics = trace.get('data', {}).get('metrics', {})
                    st.json(metrics)
        else:
            st.info("No recent evaluation runs found. Run an evaluation to see results.")

    except Exception as e:
        st.warning(f"⚠️ Could not load trace history: {str(e)[:100]}")

    st.write("---")

    # Section 4: Quick Links
    st.heading("4️⃣ Quick Links")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏠 Go to Overview"):
            st.switch_page("pages/overview.py")

    with col2:
        if st.button("📊 Go to Query Traces"):
            st.switch_page("pages/query_traces.py")

    with col3:
        if st.button("📤 Go to Ingestion Manager"):
            st.switch_page("pages/ingestion_manager.py")

    st.write("---")
    st.caption("Smart Knowledge Hub v0.1 | Evaluation Module")


def run_evaluation(settings: Settings, evaluator_name: str, test_set_path: str) -> None:
    """
    Execute evaluation workflow.

    Args:
        settings: Configuration settings
        evaluator_name: Name of evaluator backend to use
        test_set_path: Path to golden test set file
    """
    try:
        # Validate test set file exists
        test_set = Path(test_set_path)
        if not test_set.exists():
            st.error(f"❌ Test set file not found: {test_set_path}")
            return

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("⏳ Creating evaluator...")
        progress_bar.progress(10)

        # Create evaluator (mock for now, would need real HybridSearch)
        try:
            from src.libs.evaluator.evaluator_factory import EvaluatorFactory
            factory = EvaluatorFactory()

            # Create mock settings for evaluator
            mock_settings = type('Settings', (), {'provider': evaluator_name})()
            evaluator = factory.create(mock_settings)
        except Exception as e:
            st.error(f"❌ Failed to create evaluator: {str(e)}")
            return

        status_text.text("⏳ Initializing HybridSearch...")
        progress_bar.progress(20)

        # Create mock HybridSearch (in real scenario would use actual instance)
        from unittest.mock import Mock
        mock_hybrid_search = Mock()
        mock_hybrid_search.search.return_value = [
            {'chunk_id': 'chunk_001', 'score': 0.9, 'text': 'Sample text'},
            {'chunk_id': 'chunk_002', 'score': 0.85, 'text': 'Another sample'}
        ]

        status_text.text("⏳ Creating EvalRunner...")
        progress_bar.progress(30)

        # Create runner
        runner = EvalRunner(
            settings=settings,
            hybrid_search=mock_hybrid_search,
            evaluator=evaluator
        )

        status_text.text("⏳ Running evaluation...")
        progress_bar.progress(50)

        # Run evaluation
        report = runner.run(test_set_path)

        status_text.text("⏳ Processing results...")
        progress_bar.progress(80)

        # Display results
        progress_bar.progress(100)
        status_text.text("✅ Evaluation complete!")

        # Show metrics
        st.success(f"✅ Evaluation completed successfully ({report.total_queries} queries)")

        st.subheader("📈 Aggregated Metrics")

        # Display metrics in columns
        metric_cols = st.columns(len(report.metrics))
        for col, (metric_name, value) in zip(metric_cols, report.metrics.items()):
            with col:
                st.metric(metric_name.replace('_', ' ').title(), f"{value:.4f}")

        # Display per-query results
        st.subheader("🔍 Per-Query Results")

        for i, result in enumerate(report.query_results, 1):
            with st.expander(f"Query {i}: {result.get('query', 'N/A')[:60]}..."):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Expected IDs:**")
                    st.write(result.get('expected_chunk_ids', []))

                with col2:
                    st.write("**Retrieved IDs:**")
                    st.write(result.get('retrieved_ids', []))

                if 'metrics' in result:
                    st.write("**Metrics:**")
                    st.json(result['metrics'])

                if 'error' in result:
                    st.error(f"**Error:** {result['error']}")

        # Save results to trace
        try:
            trace_service = TraceService()
            trace_dict = report.to_dict()
            trace_dict['evaluator'] = evaluator_name
            trace_dict['timestamp'] = str(Path.cwd())  # Use path as placeholder
            # Could save to trace service here
        except Exception as e:
            st.warning(f"⚠️ Could not save results to trace: {str(e)[:100]}")

    except Exception as e:
        st.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
