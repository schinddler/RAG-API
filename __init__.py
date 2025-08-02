"""
RAG-API: A modular, production-grade Retrieval-Augmented Generation backend.

This package provides high-performance document processing capabilities including:
- Document chunking with semantic awareness and OCR cleaning
- GPU-optimized embedding with thread-safe caching
- Unified chunk ID generation for cross-module consistency
- Centralized configuration management
"""

try:
    from .config import (
        SUPPORTED_MODELS,
        CACHE_BACKENDS,
        SplitStrategy,
        ChunkConfig,
        EmbeddingConfig,
        get_model_name,
        validate_config
    )
    from .utils.hashing import generate_chunk_id
except ImportError:
    # Handle case when running as script
    from config import (
        SUPPORTED_MODELS,
        CACHE_BACKENDS,
        SplitStrategy,
        ChunkConfig,
        EmbeddingConfig,
        get_model_name,
        validate_config
    )
    from utils.hashing import generate_chunk_id

# Import main modules (these will be available after implementation)
try:
    from .ingestion.chunker import create_chunker, Chunker
    from .embedder.embedder import create_embedder, Embedder
except ImportError:
    # Modules not yet implemented or moved
    pass

__version__ = "1.0.0"
__author__ = "HackRx6 RAG Team"

__all__ = [
    # Configuration
    'SUPPORTED_MODELS',
    'CACHE_BACKENDS', 
    'SplitStrategy',
    'ChunkConfig',
    'EmbeddingConfig',
    'get_model_name',
    'validate_config',
    
    # Utilities
    'generate_chunk_id',
    
    # Main modules (when available)
    'create_chunker',
    'Chunker', 
    'create_embedder',
    'Embedder'
] 