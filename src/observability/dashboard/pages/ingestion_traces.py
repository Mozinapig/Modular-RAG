"""
Ingestion Traces Page - View ingestion history with stage duration waterfall
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from src.observability.dashboard.services.trace_service import TraceService
from src.core.settings import Settings


def _render_traces_timeline(traces: List[Dict[str, Any]]) -> None:
    """Render timeline of ingestion traces."""
    if not traces:
        st.info("No ingestion traces found. Start ingesting documents to see traces.")
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
            "📄 Document": metadata.get("source_path", ""),
            "⏰ Timestamp": trace.get("timestamp", "")[:19],
            "⏱️ Duration (ms)": total_duration,
            "📊 Stages": len(stages),
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_stage_waterfall(trace: Dict[str, Any]) -> None:
    """Render stage duration waterfall chart."""
    stages = trace.get("stages", [])

    if not stages:
        st.info("No stages in this trace")
        return

    # Prepare data for bar chart
    stage_data = {}
    for stage in stages:
        name = stage.get("name", "")
        start_ms = stage.get("start_ms", 0)
        end_ms = stage.get("end_ms", 0)
        duration = end_ms - start_ms

        if name:
            stage_data[name] = duration

    # Display as dataframe first (for details)
    st.write("**Stage Durations:**")
    stage_df = pd.DataFrame({
        "Stage": list(stage_data.keys()),
        "Duration (ms)": list(stage_data.values()),
    })
    st.dataframe(stage_df, use_container_width=True, hide_index=True)

    # Display as bar chart
    if stage_data:
        st.bar_chart(data=stage_df.set_index("Stage"))


def _render_stage_details(trace: Dict[str, Any]) -> None:
    """Render detailed stage information."""
    stages = trace.get("stages", [])

    st.write("**Stage Details:**")

    if not stages:
        st.info("No stages")
        return

    # Calculate total duration
    total_duration = stages[-1].get("end_ms", 0) if stages else 0

    # Display each stage
    for idx, stage in enumerate(stages):
        name = stage.get("name", f"Stage {idx}")
        start_ms = stage.get("start_ms", 0)
        end_ms = stage.get("end_ms", 0)
        duration = end_ms - start_ms

        # Calculate percentage
        percentage = (duration / total_duration * 100) if total_duration > 0 else 0

        with st.expander(f"📍 {name} ({duration}ms, {percentage:.1f}%)", expanded=(idx == 0)):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Start", f"{start_ms}ms")
            with col2:
                st.metric("End", f"{end_ms}ms")
            with col3:
                st.metric("Duration", f"{duration}ms")


def show_ingestion_traces(trace_service: TraceService) -> None:
    """
    Render Ingestion Traces page.

    Args:
        trace_service: TraceService instance for trace access
    """
    st.title("🔍 Ingestion Traces")
    st.write("View ingestion history and stage-by-stage performance breakdown.")

    # Get ingestion traces
    traces = trace_service.get_ingestion_traces()

    if not traces:
        st.warning("No ingestion traces found. Start ingesting documents to see traces.")
        return

    st.write(f"**Total Ingestions**: {len(traces)}")
    st.write("---")

    # Create tabs
    tab1, tab2 = st.tabs(["📊 Timeline", "📈 Details"])

    with tab1:
        st.subheader("Ingestion Timeline")
        _render_traces_timeline(traces)

        # Allow selecting a trace for detailed view
        st.write("---")
        trace_options = [f"{t.get('trace_id', '')[:12]} - {t.get('timestamp', '')[:10]}" for t in traces]
        selected_option = st.selectbox(
            "Select a trace to view details",
            options=trace_options,
            key="trace_selector"
        )

        if selected_option:
            # Get selected trace
            selected_idx = trace_options.index(selected_option)
            selected_trace = traces[selected_idx]

            st.write(f"**Trace Details**: {selected_trace.get('trace_id', '')}")
            st.write(f"**Document**: {selected_trace.get('metadata', {}).get('source_path', '')}")
            st.write(f"**Timestamp**: {selected_trace.get('timestamp', '')}")

            st.subheader("Stage Duration Waterfall")
            _render_stage_waterfall(selected_trace)

    with tab2:
        st.subheader("Stage Breakdown")

        # Select trace for detailed breakdown
        trace_options_detail = [f"{t.get('trace_id', '')[:12]} - {t.get('timestamp', '')[:10]}" for t in traces]
        selected_option_detail = st.selectbox(
            "Select a trace for breakdown",
            options=trace_options_detail,
            key="trace_detail_selector"
        )

        if selected_option_detail:
            selected_idx = trace_options_detail.index(selected_option_detail)
            selected_trace = traces[selected_idx]

            _render_stage_details(selected_trace)

            # Summary statistics
            st.write("---")
            st.subheader("Summary")

            stages = selected_trace.get("stages", [])
            if stages:
                total_duration = stages[-1].get("end_ms", 0)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Duration", f"{total_duration}ms")
                with col2:
                    st.metric("Stage Count", len(stages))
                with col3:
                    if total_duration > 0:
                        avg_stage_duration = total_duration / len(stages)
                        st.metric("Avg/Stage", f"{avg_stage_duration:.0f}ms")
                with col4:
                    # Longest stage
                    longest_stage = max(
                        [(s.get("name", ""), s.get("end_ms", 0) - s.get("start_ms", 0))
                         for s in stages],
                        key=lambda x: x[1],
                        default=("N/A", 0)
                    )
                    st.metric("Longest Stage", f"{longest_stage[0]} ({longest_stage[1]}ms)")


def show_page(settings: Settings) -> None:
    """Entry point for ingestion traces page (called from dashboard app)."""
    try:
        # Create trace service
        logs_path = settings.observability.get("logs_path", "logs")
        trace_service = TraceService(log_path=logs_path)

        # Render page
        show_ingestion_traces(trace_service)
    except Exception as e:
        st.error(f"Failed to initialize ingestion traces: {e}")
