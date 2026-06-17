"""
System Overview page for Dashboard
"""
import streamlit as st
from src.core.settings import Settings
from src.observability.dashboard.services.config_service import ConfigService


def show_overview(settings: Settings):
    """Display system overview page."""
    st.title("🏠 System Overview")
    st.write("---")

    # Get configuration service
    config_service = ConfigService(settings)

    # Section 1: Component Configuration
    st.heading("1️⃣ Component Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🤖 LLM Provider")
        llm_config = config_service.get_llm_config()
        st.write(f"**Provider**: {llm_config.get('provider', 'N/A')}")
        st.write(f"**Model**: {llm_config.get('model', 'N/A')}")
        st.write(f"**Temperature**: {llm_config.get('temperature', 'N/A')}")
        if llm_config.get('base_url'):
            st.write(f"**Base URL**: {llm_config.get('base_url')}")

    with col2:
        st.subheader("🔗 Embedding Provider")
        embedding_config = config_service.get_embedding_config()
        st.write(f"**Provider**: {embedding_config.get('provider', 'N/A')}")
        st.write(f"**Model**: {embedding_config.get('model', 'N/A')}")
        st.write(f"**Dimensions**: {embedding_config.get('dimensions', 'N/A')}")
        if embedding_config.get('base_url'):
            st.write(f"**Base URL**: {embedding_config.get('base_url')}")

    with col3:
        st.subheader("💾 Vector Store")
        vs_config = config_service.get_vector_store_config()
        st.write(f"**Backend**: {vs_config.get('backend', 'N/A')}")
        st.write(f"**Persist Path**: {vs_config.get('persist_path', 'N/A')}")

    st.write("---")

    # Section 2: Retrieval Configuration
    st.heading("2️⃣ Retrieval Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Hybrid Search")
        retrieval_config = config_service.get_retrieval_config()
        st.write(f"**Sparse Backend**: {retrieval_config.get('sparse_backend', 'N/A')}")
        st.write(f"**Fusion Algorithm**: {retrieval_config.get('fusion_algorithm', 'N/A')}")
        st.write(f"**Top K Dense**: {retrieval_config.get('top_k_dense', 'N/A')}")
        st.write(f"**Top K Sparse**: {retrieval_config.get('top_k_sparse', 'N/A')}")
        st.write(f"**Top K Final**: {retrieval_config.get('top_k_final', 'N/A')}")

    with col2:
        st.subheader("🔄 Reranker")
        reranker_config = config_service.get_reranker_config()
        st.write(f"**Backend**: {reranker_config.get('backend', 'N/A')}")
        if reranker_config.get('model'):
            st.write(f"**Model**: {reranker_config.get('model')}")
        st.write(f"**Top M**: {reranker_config.get('top_m', 'N/A')}")

    st.write("---")

    # Section 3: Data Statistics (placeholder for integration)
    st.heading("3️⃣ Data Statistics")

    try:
        from src.libs.vector_store.chroma_store import ChromaStore
        store = ChromaStore(settings)

        # Try to get collection stats
        try:
            stats = store.get_collection_stats()
            col1, col2, col3 = st.columns(3)

            with col1:
                total_chunks = stats.get("total_chunks", 0)
                st.metric("📦 Total Chunks", total_chunks)

            with col2:
                total_images = stats.get("total_images", 0)
                st.metric("🖼️ Total Images", total_images)

            with col3:
                collections = stats.get("collections", [])
                st.metric("📚 Collections", len(collections))

            # Collections detail
            if collections:
                st.subheader("Collection Details")
                for collection in collections:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{collection.get('name')}**")
                    with col2:
                        st.write(f"Chunks: {collection.get('chunk_count')}")
                    with col3:
                        st.write(f"Images: {collection.get('image_count')}")

        except Exception as e:
            st.info(f"ℹ️ No data yet. Run `ingest.py` to load documents. ({str(e)[:50]}...)")

    except Exception as e:
        st.warning(f"⚠️ Could not connect to vector store: {str(e)[:50]}...")

    st.write("---")

    # Section 4: Quick Links
    st.heading("4️⃣ Quick Links")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📤 Go to Ingestion Manager"):
            st.switch_page("pages/ingestion_manager.py")

    with col2:
        if st.button("📚 Go to Data Browser"):
            st.switch_page("pages/data_browser.py")

    with col3:
        if st.button("❓ Go to Query Traces"):
            st.switch_page("pages/query_traces.py")

    st.write("---")
    st.caption("Smart Knowledge Hub v0.1 | Dashboard Module")
