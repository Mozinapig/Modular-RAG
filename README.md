# Modular RAG System

A modular, pluggable RAG (Retrieval-Augmented Generation) system with MCP (Model Context Protocol) server support.

## Features

- **Modular Architecture**: Pluggable components for LLM, Embedding, Vector Store, Reranker, and Evaluator
- **Multi-provider Support**: OpenAI, Azure OpenAI, Ollama, DeepSeek, and more
- **MCP Server Integration**: Seamless integration with Copilot and Claude Desktop
- **Hybrid Search**: Combined dense and sparse retrieval with RRF fusion
- **Multi-modal Support**: Image understanding and caption generation
- **Full Observability**: Trace collection and web-based dashboard
- **Zero External Dependencies**: Local-first design, runs offline

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e .

# Install dev dependencies (optional)
pip install -e ".[dev]"
```

### Configuration

1. Create local config:
```bash
cp config/settings.yaml config/settings.local.yaml
```

2. Set environment variables:
```bash
export AZURE_API_ENDPOINT="https://..."
export AZURE_API_KEY="..."
export OPENAI_API_KEY="..."
```

### Running Tests

```bash
# Run smoke tests
python -m compileall src
pytest tests/unit/test_smoke_imports.py

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Architecture

See [DEV_SPEC.md](./DEV_SPEC.md) for detailed architecture documentation.

## Development Status

Currently implementing Phase A: Engineering skeleton and test base
- ✅ Directory structure initialized
- ⏳ Pytest setup and smoke tests
- ⏳ Configuration loading and validation

## License

MIT
