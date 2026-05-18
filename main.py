#!/usr/bin/env python3
"""
MCP RAG System - Main Entry Point

This is the MCP Server entry point for the modular RAG system.
"""

import sys
import logging


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("MCP RAG System starting...")

    # Load settings
    try:
        from src.core.settings import load_settings, validate_settings
        settings = load_settings("config/settings.yaml")
        validate_settings(settings)
        logger.info(f"Configuration loaded and validated successfully")
        logger.info(f"LLM Provider: {settings.llm.provider} ({settings.llm.model})")
        logger.info(f"LLM Base URL: {settings.llm.base_url or 'default'}")
        logger.info(f"Embedding Provider: {settings.embedding.provider} ({settings.embedding.model})")
        logger.info(f"Embedding Base URL: {settings.embedding.base_url or 'default'}")
        if settings.vision_llm:
            logger.info(f"Vision LLM Provider: {settings.vision_llm.provider} ({settings.vision_llm.model})")
        logger.info(f"Vector Store Backend: {settings.vector_store.backend}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        sys.exit(1)

    # TODO: Initialize MCP Server in Phase E
    logger.info("MCP RAG System initialized successfully")


if __name__ == "__main__":
    main()

