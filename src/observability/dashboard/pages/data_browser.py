"""
Data Browser Page - View documents, chunks, and associated images
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from src.observability.dashboard.services.data_service import DataService
from src.core.settings import Settings


def _render_documents_table(documents: List[Dict[str, Any]]) -> None:
    """Render documents list as table."""
    if not documents:
        st.info("No documents found in this collection.")
        return

    # Prepare dataframe
    df_data = []
    for doc in documents:
        df_data.append({
            "📄 Document": doc.get("source_path", ""),
            "📦 Chunks": doc.get("chunk_count", 0),
            "⏰ Ingestion Time": doc.get("ingestion_time", ""),
        })

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)


def _render_chunk_details(chunk: Dict[str, Any]) -> None:
    """Render chunk details with metadata."""
    metadata = chunk.get("metadata", {})

    with st.expander(f"📝 Chunk: {chunk.get('id', 'unknown')[:20]}...", expanded=False):
        # Display content
        content = chunk.get("content", "")
        if len(content) > 500:
            st.write("**Content (first 500 chars):**")
            st.code(content[:500] + "...", language="markdown")
            with st.expander("📖 Show full content"):
                st.code(content, language="markdown")
        else:
            st.write("**Content:**")
            st.code(content, language="markdown")

        # Display metadata
        st.write("**Metadata:**")
        meta_cols = st.columns(2)
        with meta_cols[0]:
            for key in ["source_path", "page", "chunk_index"]:
                if key in metadata:
                    st.write(f"- **{key}**: {metadata[key]}")

        with meta_cols[1]:
            for key in ["title", "tags"]:
                if key in metadata:
                    st.write(f"- **{key}**: {metadata[key]}")


def _render_chunk_images(images: List[Dict[str, Any]]) -> None:
    """Render images associated with a chunk."""
    if not images:
        st.info("No images associated with this chunk.")
        return

    st.write(f"**Associated Images ({len(images)}):**")

    # Display images in columns
    cols = st.columns(min(3, len(images)))
    for idx, image in enumerate(images):
        with cols[idx % len(cols)]:
            try:
                image_path = image.get("path", "")
                if image_path:
                    st.image(image_path, caption=image.get("image_id", ""), use_container_width=True)
            except Exception as e:
                st.warning(f"Failed to load image: {e}")


def show_data_browser(data_service: DataService) -> None:
    """
    Render Data Browser page.

    Args:
        data_service: DataService instance for data access
    """
    st.title("📚 Data Browser")
    st.write("Browse documents, chunks, and associated images in your knowledge base.")

    # Get available collections
    collections = data_service.list_collections()
    if not collections:
        st.warning("No collections found. Please ingest some documents first.")
        return

    # Collection selector
    selected_collection = st.selectbox(
        "Select Collection",
        options=collections,
        help="Choose a collection to browse"
    )

    # Get documents in collection
    documents = data_service.get_documents(collection=selected_collection)

    # Create tabs for different views
    tab1, tab2 = st.tabs(["📄 Documents", "🔍 Search"])

    with tab1:
        st.subheader("Documents List")

        if not documents:
            st.info("No documents found in this collection.")
        else:
            # Show documents summary
            st.write(f"**Total Documents**: {len(documents)}")
            st.write(f"**Total Chunks**: {sum(d.get('chunk_count', 0) for d in documents)}")

            # Documents table
            st.write("---")
            _render_documents_table(documents)

            # Document details
            st.write("---")
            st.subheader("Document Details")

            # Select document to view chunks
            doc_choices = [d.get("source_path", "") for d in documents]
            selected_doc = st.selectbox(
                "Select a document to view chunks",
                options=doc_choices,
                key="doc_selector"
            )

            if selected_doc:
                # Get chunks for selected document
                chunks = data_service.get_chunks(
                    source_path=selected_doc,
                    collection=selected_collection
                )

                if chunks:
                    st.write(f"**Chunks in {selected_doc}**: {len(chunks)}")
                    st.write("---")

                    # Render each chunk
                    for idx, chunk in enumerate(chunks):
                        # Create two columns: chunk details + images
                        col1, col2 = st.columns([3, 2])

                        with col1:
                            _render_chunk_details(chunk)

                        with col2:
                            # Get and render images for this chunk
                            images = data_service.get_chunk_images(chunk.get("id", ""))
                            if images:
                                _render_chunk_images(images)
                            else:
                                st.info("No images")

                        st.write("---")
                else:
                    st.info(f"No chunks found for {selected_doc}")

    with tab2:
        st.subheader("Search Chunks")
        st.info("🔍 Search functionality coming in G4+")

        # Placeholder for search
        search_query = st.text_input("Search query", placeholder="Enter keywords...")
        if search_query:
            st.info("Search implementation coming soon")


def show_page(settings: Settings) -> None:
    """Entry point for data browser page (called from dashboard app)."""
    # Initialize services
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.ingestion.storage.image_storage import ImageStorage
    from pathlib import Path

    try:
        # Create services
        chroma_store = VectorStoreFactory.create(settings)
        image_storage = ImageStorage(base_path=Path(settings.data_path) / "images")
        data_service = DataService(chroma_store=chroma_store, image_storage=image_storage)

        # Render page
        show_data_browser(data_service)
    except Exception as e:
        st.error(f"Failed to initialize data browser: {e}")
