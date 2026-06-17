"""
Main Dashboard application - Multi-page Streamlit app
"""
import streamlit as st
from pathlib import Path
from src.core.settings import load_settings
from src.observability.dashboard.services.config_service import ConfigService


def _get_settings():
    """Load settings, cache them for the session."""
    if "settings" not in st.session_state:
        settings_path = Path("config/settings.yaml")
        try:
            st.session_state.settings = load_settings(str(settings_path))
        except Exception as e:
            st.error(f"Failed to load settings: {e}")
            st.stop()
    return st.session_state.settings


def main():
    """Main entry point for Dashboard."""
    st.set_page_config(
        page_title="Smart Knowledge Hub Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load settings
    settings = _get_settings()

    # Define pages
    pages = {
        "🏠 System Overview": "pages/overview",
        "📚 Data Browser": "pages/data_browser",
        "📤 Ingestion Manager": "pages/ingestion_manager",
        "🔍 Ingestion Traces": "pages/ingestion_traces",
        "❓ Query Traces": "pages/query_traces",
        "📊 Evaluation Panel": "pages/evaluation_panel",
    }

    with st.sidebar:
        st.title("Smart Knowledge Hub")
        st.write("---")

        selected_page = st.radio("Navigation", list(pages.keys()))

        st.write("---")
        st.write("### Configuration Summary")
        config_service = ConfigService(settings)
        llm_config = config_service.get_llm_config()
        embedding_config = config_service.get_embedding_config()
        vs_config = config_service.get_vector_store_config()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("LLM", llm_config.get("model", "N/A"))
        with col2:
            st.metric("Embedding", embedding_config.get("model", "N/A"))

        st.write(f"**VectorStore**: {vs_config.get('backend', 'N/A')}")

    # Route to selected page
    page_key = pages[selected_page]

    if page_key == "pages/overview":
        from src.observability.dashboard.pages.overview import show_overview
        show_overview(settings)
    elif page_key == "pages/data_browser":
        st.info("📚 Data Browser page - Coming soon")
    elif page_key == "pages/ingestion_manager":
        st.info("📤 Ingestion Manager page - Coming soon")
    elif page_key == "pages/ingestion_traces":
        st.info("🔍 Ingestion Traces page - Coming soon")
    elif page_key == "pages/query_traces":
        st.info("❓ Query Traces page - Coming soon")
    elif page_key == "pages/evaluation_panel":
        st.info("📊 Evaluation Panel page - Coming soon")


if __name__ == "__main__":
    main()
