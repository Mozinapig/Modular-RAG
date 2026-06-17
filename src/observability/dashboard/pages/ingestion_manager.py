"""
Ingestion Manager Page - Upload files, monitor progress, manage documents
"""
import streamlit as st
import io
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional
from src.observability.dashboard.services.data_service import DataService
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.document_manager import DocumentManager
from src.core.settings import Settings


def _render_file_uploader(
    collections: List[str],
    on_upload: Optional[Callable[[io.BytesIO, str, str], None]] = None
) -> None:
    """Render file uploader with collection selector."""
    st.subheader("📤 Upload Document")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Select PDF file",
            type=["pdf"],
            help="Upload a PDF document to ingest"
        )

    with col2:
        selected_collection = st.selectbox(
            "Collection",
            options=collections if collections else ["default"],
            help="Choose a collection for this document"
        )

    if uploaded_file:
        if st.button("Start Ingestion", key="ingest_btn"):
            if on_upload:
                on_upload(uploaded_file, selected_collection)


def _render_documents_list_with_delete(
    documents: List[Dict[str, Any]],
    collection: str,
    on_delete: Optional[Callable[[str, str], None]] = None
) -> None:
    """Render documents list with delete buttons."""
    if not documents:
        st.info("No documents in this collection yet.")
        return

    st.write(f"**Documents in {collection}**: {len(documents)}")

    # Create columns for display
    col1, col2, col3, col4 = st.columns([2, 1, 1, 0.8])

    with col1:
        st.write("**Document**")
    with col2:
        st.write("**Chunks**")
    with col3:
        st.write("**Ingestion Time**")
    with col4:
        st.write("**Action**")

    # Display each document with delete button
    for doc in documents:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 0.8])

        with col1:
            st.write(doc.get("source_path", ""))
        with col2:
            st.write(str(doc.get("chunk_count", 0)))
        with col3:
            st.write(doc.get("ingestion_time", "")[:10])  # Show date only
        with col4:
            if st.button("🗑️", key=f"delete_{doc.get('source_path', '')}"):
                if on_delete:
                    on_delete(doc.get("source_path", ""), collection)
                    st.success(f"Deleted {doc.get('source_path', '')}")
                    st.rerun()


def _render_ingestion_progress(
    file_path: str,
    collection: str,
    pipeline: IngestionPipeline,
    on_progress: Optional[Callable[[str, int, int], None]] = None
) -> None:
    """Render ingestion progress display."""
    st.subheader("⏳ Ingestion Progress")

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    details_placeholder = st.empty()

    # Track progress
    progress_data = {"stage": "", "current": 0, "total": 0}

    def progress_callback(stage: str, current: int, total: int):
        """Update progress display."""
        progress_data["stage"] = stage
        progress_data["current"] = current
        progress_data["total"] = total

        # Update progress bar
        if total > 0:
            progress_value = current / total
        else:
            progress_value = 0

        progress_placeholder.progress(progress_value)
        status_placeholder.write(f"Current stage: **{stage}** ({current}/{total})")

    try:
        # Run ingestion
        result = pipeline.run(
            source_path=file_path,
            collection=collection,
            on_progress=progress_callback
        )

        # Show results
        if result.success:
            st.success(f"✅ Ingestion successful!")
            st.json({
                "chunks_created": result.chunks_created,
                "images_processed": result.images_processed,
                "elapsed_ms": result.elapsed_ms,
            })
        else:
            st.error(f"❌ Ingestion failed!")
            if result.errors:
                for error in result.errors:
                    st.write(f"- {error}")

    except Exception as e:
        st.error(f"Error during ingestion: {str(e)}")


def show_ingestion_manager(
    data_service: DataService,
    pipeline: IngestionPipeline,
    doc_manager: DocumentManager
) -> None:
    """
    Render Ingestion Manager page.

    Args:
        data_service: DataService instance for data access
        pipeline: IngestionPipeline instance for document ingestion
        doc_manager: DocumentManager instance for document lifecycle
    """
    st.title("📤 Ingestion Manager")
    st.write("Upload documents, monitor ingestion progress, and manage your knowledge base.")

    # Get available collections
    collections = data_service.list_collections()
    if not collections:
        collections = ["default"]

    # Create tabs
    tab1, tab2 = st.tabs(["📥 Upload & Manage", "📊 Ingestion Status"])

    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            # File uploader
            uploaded_file = st.file_uploader(
                "Select PDF file",
                type=["pdf"],
                help="Upload a PDF document to ingest"
            )

            selected_collection = st.selectbox(
                "Collection",
                options=collections,
                help="Choose a collection for this document"
            )

            if uploaded_file and st.button("Start Ingestion", key="ingest_start"):
                # Save uploaded file temporarily
                temp_path = Path("temp") / uploaded_file.name
                temp_path.parent.mkdir(exist_ok=True)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Show progress
                with col2:
                    _render_ingestion_progress(
                        file_path=str(temp_path),
                        collection=selected_collection,
                        pipeline=pipeline
                    )

                # Clean up
                temp_path.unlink()

                # Refresh display
                st.rerun()

        # Document management
        with col2:
            st.subheader("📋 Manage Documents")

            # Select collection for management
            mgmt_collection = st.selectbox(
                "Select collection",
                options=collections,
                key="mgmt_collection"
            )

            # Get documents
            documents = data_service.get_documents(collection=mgmt_collection)

            if documents:
                # Document list with delete
                st.write(f"**Documents**: {len(documents)}")

                for doc in documents:
                    col_name, col_chunks, col_delete = st.columns([2, 1, 0.5])

                    with col_name:
                        st.write(doc.get("source_path", ""))
                    with col_chunks:
                        st.write(f"{doc.get('chunk_count', 0)} chunks")
                    with col_delete:
                        if st.button("🗑️", key=f"del_{doc.get('source_path', '')}_{mgmt_collection}"):
                            try:
                                doc_manager.delete_document(
                                    doc.get("source_path", ""),
                                    mgmt_collection
                                )
                                st.success("Deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {str(e)}")
            else:
                st.info("No documents in this collection")

    with tab2:
        st.subheader("📊 Ingestion Status")
        st.info("📈 Status monitoring coming soon - will show real-time ingestion statistics")


def show_page(settings: Settings) -> None:
    """Entry point for ingestion manager page (called from dashboard app)."""
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.ingestion.storage.image_storage import ImageStorage

    try:
        # Create services
        chroma_store = VectorStoreFactory.create(settings)
        image_storage = ImageStorage(base_path=Path(settings.data_path) / "images")
        data_service = DataService(chroma_store=chroma_store, image_storage=image_storage)

        # Create pipeline and doc manager
        pipeline = IngestionPipeline(settings)
        doc_manager = DocumentManager(
            chroma_store=chroma_store,
            bm25_indexer_path=Path(settings.data_path) / "bm25",
            image_storage=image_storage
        )

        # Render page
        show_ingestion_manager(
            data_service=data_service,
            pipeline=pipeline,
            doc_manager=doc_manager
        )
    except Exception as e:
        st.error(f"Failed to initialize ingestion manager: {e}")
