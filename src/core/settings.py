"""
Configuration loading and validation module.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


@dataclass
class LLMSettings:
    """LLM provider settings."""
    provider: str
    model: str
    api_key: str
    temperature: float = 0.7
    base_url: Optional[str] = None
    azure_endpoint: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Resolve environment variables
        self.api_key = os.path.expandvars(self.api_key)
        if self.base_url:
            self.base_url = os.path.expandvars(self.base_url)
        if self.azure_endpoint:
            self.azure_endpoint = os.path.expandvars(self.azure_endpoint)


@dataclass
class EmbeddingSettings:
    """Embedding provider settings."""
    provider: str
    model: str
    api_key: str
    dimensions: int = 1536
    base_url: Optional[str] = None
    azure_endpoint: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.api_key = os.path.expandvars(self.api_key)
        if self.base_url:
            self.base_url = os.path.expandvars(self.base_url)
        if self.azure_endpoint:
            self.azure_endpoint = os.path.expandvars(self.azure_endpoint)


@dataclass
class VectorStoreSettings:
    """Vector store settings."""
    backend: str
    persist_path: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionLLMSettings:
    """Vision LLM provider settings."""
    provider: str
    model: str
    api_key: str
    base_url: Optional[str] = None
    azure_endpoint: Optional[str] = None
    max_image_size: int = 2048
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Resolve environment variables
        self.api_key = os.path.expandvars(self.api_key)
        if self.base_url:
            self.base_url = os.path.expandvars(self.base_url)
        if self.azure_endpoint:
            self.azure_endpoint = os.path.expandvars(self.azure_endpoint)


@dataclass
class RetrievalSettings:
    """Retrieval settings."""
    sparse_backend: str = "bm25"
    fusion_algorithm: str = "rrf"
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10


@dataclass
class RerankerSettings:
    """Reranker settings."""
    backend: str = "none"
    model: Optional[str] = None
    top_m: int = 30


@dataclass
class ObservabilitySettings:
    """Observability settings."""
    enabled: bool = True
    log_file: str = "./logs/traces.jsonl"
    log_level: str = "INFO"
    detail_level: str = "standard"


@dataclass
class Settings:
    """Main settings dataclass."""
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    vision_llm: Optional[VisionLLMSettings] = None
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    rerank: RerankerSettings = field(default_factory=RerankerSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    extra: Dict[str, Any] = field(default_factory=dict)


def load_settings(path: str) -> Settings:
    """
    Load settings from YAML file.

    Args:
        path: Path to settings.yaml file

    Returns:
        Settings object

    Raises:
        FileNotFoundError: If settings file not found
        ValueError: If required fields are missing
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError("Settings file is empty")

    return _parse_settings(data)


def _parse_settings(data: Dict[str, Any]) -> Settings:
    """Parse settings dictionary into Settings object."""

    # Validate required sections
    required_sections = ['llm', 'embedding', 'vector_store']
    for section in required_sections:
        if section not in data:
            raise ValueError(f"Missing required section: {section}")

    # Parse LLM settings
    llm_data = data['llm']
    llm = LLMSettings(
        provider=llm_data.get('provider', 'openai'),
        model=llm_data.get('model', 'gpt-4o'),
        api_key=llm_data.get('api_key', '${OPENAI_API_KEY}'),
        temperature=llm_data.get('temperature', 0.7),
        base_url=llm_data.get('base_url'),
        azure_endpoint=llm_data.get('azure_endpoint'),
        extra={k: v for k, v in llm_data.items()
               if k not in ['provider', 'model', 'api_key', 'temperature', 'base_url', 'azure_endpoint']}
    )

    # Parse Embedding settings
    emb_data = data['embedding']
    embedding = EmbeddingSettings(
        provider=emb_data.get('provider', 'openai'),
        model=emb_data.get('model', 'text-embedding-3-small'),
        api_key=emb_data.get('api_key', '${OPENAI_API_KEY}'),
        dimensions=emb_data.get('dimensions', 1536),
        base_url=emb_data.get('base_url'),
        azure_endpoint=emb_data.get('azure_endpoint'),
        extra={k: v for k, v in emb_data.items()
               if k not in ['provider', 'model', 'api_key', 'dimensions', 'base_url', 'azure_endpoint']}
    )

    # Parse VectorStore settings
    vs_data = data['vector_store']
    vector_store = VectorStoreSettings(
        backend=vs_data.get('backend', 'chroma'),
        persist_path=vs_data.get('persist_path', './data/db/chroma'),
        extra={k: v for k, v in vs_data.items()
               if k not in ['backend', 'persist_path']}
    )

    # Parse Vision LLM settings (optional)
    vision_llm = None
    if 'vision_llm' in data:
        vision_data = data['vision_llm']
        vision_llm = VisionLLMSettings(
            provider=vision_data.get('provider', 'openai'),
            model=vision_data.get('model', 'gpt-4o-mini'),
            api_key=vision_data.get('api_key', '${OPENAI_API_KEY}'),
            base_url=vision_data.get('base_url'),
            azure_endpoint=vision_data.get('azure_endpoint'),
            max_image_size=vision_data.get('max_image_size', 2048),
            extra={k: v for k, v in vision_data.items()
                   if k not in ['provider', 'model', 'api_key', 'base_url', 'azure_endpoint', 'max_image_size']}
        )

    # Parse optional sections
    retr_data = data.get('retrieval', {})
    retrieval = RetrievalSettings(
        sparse_backend=retr_data.get('sparse_backend', 'bm25'),
        fusion_algorithm=retr_data.get('fusion_algorithm', 'rrf'),
        top_k_dense=retr_data.get('top_k_dense', 20),
        top_k_sparse=retr_data.get('top_k_sparse', 20),
        top_k_final=retr_data.get('top_k_final', 10),
    )

    rerank_data = data.get('rerank', {})
    rerank = RerankerSettings(
        backend=rerank_data.get('backend', 'none'),
        model=rerank_data.get('model'),
        top_m=rerank_data.get('top_m', 30),
    )

    obs_data = data.get('observability', {})
    observability = ObservabilitySettings(
        enabled=obs_data.get('enabled', True),
        log_file=obs_data.get('log_file', './logs/traces.jsonl'),
        log_level=obs_data.get('log_level', 'INFO'),
        detail_level=obs_data.get('detail_level', 'standard'),
    )

    extra = {k: v for k, v in data.items()
             if k not in ['llm', 'embedding', 'vector_store', 'vision_llm', 'retrieval',
                          'rerank', 'observability']}

    return Settings(
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        vision_llm=vision_llm,
        retrieval=retrieval,
        rerank=rerank,
        observability=observability,
        extra=extra
    )


def validate_settings(settings: Settings) -> None:
    """
    Validate settings object for required fields.

    Args:
        settings: Settings object to validate

    Raises:
        ValueError: If validation fails
    """
    # Check LLM provider
    if not settings.llm.provider:
        raise ValueError("llm.provider is required")
    if not settings.llm.model:
        raise ValueError("llm.model is required")
    if not settings.llm.api_key:
        raise ValueError("llm.api_key is required")

    # Check Embedding provider
    if not settings.embedding.provider:
        raise ValueError("embedding.provider is required")
    if not settings.embedding.model:
        raise ValueError("embedding.model is required")
    if not settings.embedding.api_key:
        raise ValueError("embedding.api_key is required")

    # Check Vector Store backend
    if not settings.vector_store.backend:
        raise ValueError("vector_store.backend is required")

    # Check Vision LLM if present
    if settings.vision_llm:
        if not settings.vision_llm.provider:
            raise ValueError("vision_llm.provider is required")
        if not settings.vision_llm.model:
            raise ValueError("vision_llm.model is required")
        if not settings.vision_llm.api_key:
            raise ValueError("vision_llm.api_key is required")
