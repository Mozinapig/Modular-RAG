"""
Query Traces Page - View query history with Dense/Sparse comparison and Rerank changes
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from src.observability.dashboard.services.trace_service import TraceService
from src.core.settings import Settings


def _render_query_traces_list(traces: List[Dict[str, Any]]) -> None:
    """Render list of query traces."""
    if not traces:
        st.info("No query traces found. Run some queries to see traces.")
        return

    # Prepare dataframe
    df_data = []
    for trace in traces:
        metadata = trace.get("metadata", {})
        stages = trace.get("stages", [])

        # Calculate total duration
        total_duration = 0
        if stages:
            total_duration = stages[-1].get("end_ms", 0)

        df_data.append({
            "📝 Trace ID": trace.get("trace_id", "")[:12],
            "❓ Query": trace.get("query", "")[:50],
            "⏰ Timestamp": trace.get("timestamp", "")[:19],
            "⏱️ Duration (ms)": total_duration,
            "🔍 Dense Recall": len(metadata.get("dense_results", [])),
            "📄 Sparse Recall": len(metadata.get("sparse_results", [])),
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_retrieval_comparison(trace: Dict[str, Any]) -> None:
    """Render Dense vs Sparse retrieval comparison."""
    metadata = trace.get("metadata", {})
    dense_results = metadata.get("dense_results", [])
    sparse_results = metadata.get("sparse_results", [])

    col1, col2 = st.columns(2)

    with col1:
        st.write("**🔍 Dense Retrieval**")
        if dense_results:
            dense_df = pd.DataFrame({
                "Rank": range(1, len(dense_results) + 1),
                "Chunk ID": [r.get("id", "")[:20] for r in dense_results],
                "Score": [round(r.get("score", 0), 3) for r in dense_results],
            })
            st.dataframe(dense_df, use_container_width=True, hide_index=True)
        else:
            st.info("No dense results")

    with col2:
        st.write("**📄 Sparse Retrieval**")
        if sparse_results:
            sparse_df = pd.DataFrame({
                "Rank": range(1, len(sparse_results) + 1),
                "Chunk ID": [r.get("id", "")[:20] for r in sparse_results],
                "Score": [round(r.get("score", 0), 3) for r in sparse_results],
            })
            st.dataframe(sparse_df, use_container_width=True, hide_index=True)
        else:
            st.info("No sparse results")

    # Show overlap statistics
    if dense_results and sparse_results:
        dense_ids = set([r.get("id") for r in dense_results])
        sparse_ids = set([r.get("id") for r in sparse_results])
        overlap = len(dense_ids & sparse_ids)

        st.write("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dense Results", len(dense_results))
        with col2:
            st.metric("Sparse Results", len(sparse_results))
        with col3:
            overlap_percent = (overlap / max(len(dense_results), 1)) * 100
            st.metric("Overlap", f"{overlap_percent:.0f}%")


def _render_rerank_changes(trace: Dict[str, Any]) -> None:
    """Render Rerank changes (before/after ranking)."""
    metadata = trace.get("metadata", {})
    before_rerank = metadata.get("before_rerank", [])
    after_rerank = metadata.get("after_rerank", [])

    if not before_rerank or not after_rerank:
        st.info("No rerank data available")
        return

    # Show if top result changed
    top_before = before_rerank[0] if before_rerank else {}
    top_after = after_rerank[0] if after_rerank else {}

    if top_before.get("id") != top_after.get("id"):
        st.warning(f"⚠️ Top result changed: {top_before.get('id', 'N/A')} → {top_after.get('id', 'N/A')}")
    else:
        st.success("✓ Top result unchanged by rerank")

    # Show detailed ranking changes
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Before Rerank**")
        before_df = pd.DataFrame({
            "Rank": [r.get("rank", 0) for r in before_rerank],
            "Chunk ID": [r.get("id", "")[:20] for r in before_rerank],
            "Score": [round(r.get("score", 0), 3) for r in before_rerank],
        })
        st.dataframe(before_df, use_container_width=True, hide_index=True)

    with col2:
        st.write("**After Rerank**")
        after_df = pd.DataFrame({
            "Rank": [r.get("rank", 0) for r in after_rerank],
            "Chunk ID": [r.get("id", "")[:20] for r in after_rerank],
            "Score": [round(r.get("score", 0), 3) for r in after_rerank],
        })
        st.dataframe(after_df, use_container_width=True, hide_index=True)


def show_query_traces(trace_service: TraceService) -> None:
    """
    Render Query Traces page.

    Args:
        trace_service: TraceService instance for trace access
    """
    st.title("❓ Query Traces")
    st.write("View query history, compare Dense/Sparse retrieval, and track Rerank impact.")

    # Get query traces
    traces = trace_service.get_query_traces()

    if not traces:
        st.warning("No query traces found. Run some queries to see traces.")
        return

    st.write(f"**Total Queries**: {len(traces)}")

    # Search/filter section
    with st.expander("🔎 Search & Filter", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            search_query = st.text_input(
                "Search by query text",
                placeholder="Search queries...",
                key="query_search"
            )

        with col2:
            search_doc = st.text_input(
                "Search by document",
                placeholder="Search by source_path...",
                key="doc_search"
            )

        # Apply filters
        if search_query:
            traces = [t for t in traces if search_query.lower() in t.get("query", "").lower()]

    st.write("---")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Timeline", "🔄 Retrieval Comparison", "📈 Rerank Impact"])

    with tab1:
        st.subheader("Query Timeline")
        _render_query_traces_list(traces)

        # Select trace for detail view
        st.write("---")
        trace_options = [f"{t.get('trace_id', '')[:12]} - {t.get('query', '')[:30]}" for t in traces]

        if trace_options:
            selected_option = st.selectbox(
                "Select a query to view details",
                options=trace_options,
                key="query_trace_selector"
            )

            if selected_option:
                selected_idx = trace_options.index(selected_option)
                selected_trace = traces[selected_idx]

                st.write(f"**Query**: {selected_trace.get('query', '')}")
                st.write(f"**Timestamp**: {selected_trace.get('timestamp', '')}")

                # Show stage timing
                stages = selected_trace.get("stages", [])
                if stages:
                    st.write("**Timing**:")
                    total_duration = stages[-1].get("end_ms", 0) if stages else 0
                    st.metric("Total Duration", f"{total_duration}ms")

    with tab2:
        st.subheader("Dense vs Sparse Comparison")
        st.info("Select a trace above to see retrieval comparison")

        trace_options = [f"{t.get('trace_id', '')[:12]}" for t in traces]
        if trace_options:
            selected_option = st.selectbox(
                "Select trace",
                options=trace_options,
                key="retrieval_trace_selector"
            )

            if selected_option:
                selected_idx = trace_options.index(selected_option)
                selected_trace = traces[selected_idx]

                _render_retrieval_comparison(selected_trace)

    with tab3:
        st.subheader("Rerank Impact")
        st.info("Visualize how reranking changes result ordering")

        trace_options = [f"{t.get('trace_id', '')[:12]}" for t in traces]
        if trace_options:
            selected_option = st.selectbox(
                "Select trace",
                options=trace_options,
                key="rerank_trace_selector"
            )

            if selected_option:
                selected_idx = trace_options.index(selected_option)
                selected_trace = traces[selected_idx]

                _render_rerank_changes(selected_trace)

        # Summary statistics
        st.write("---")
        st.subheader("Rerank Statistics")

        with_rerank = sum(1 for t in traces if t.get("metadata", {}).get("after_rerank"))
        st.metric("Queries with Rerank", with_rerank)


def show_page(settings: Settings) -> None:
    """Entry point for query traces page (called from dashboard app)."""
    try:
        # Create trace service
        logs_path = settings.observability.get("logs_path", "logs")
        trace_service = TraceService(log_path=logs_path)

        # Render page
        show_query_traces(trace_service)
    except Exception as e:
        st.error(f"Failed to initialize query traces: {e}")
