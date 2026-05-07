"""
Victor RAG Document Q&A — app package

Exports:
    RAGPipeline: Core ingestion and query pipeline
    get_settings: Cached settings loader
"""

from app.rag_pipeline import RAGPipeline
from app.config import get_settings

__all__ = ["RAGPipeline", "get_settings"]
__version__ = "1.0.0"
__author__ = "Olusegun (Victor) Makanju"
