"""
Central configuration and model registry for the RAG system.
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class SplitStrategy(Enum):
    """Chunking strategies for document processing."""
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    FIXED = "fixed"


# Supported embedding models
SUPPORTED_MODELS = {
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge": "BAAI/bge-small-en-v1.5",
    "e5": "intfloat/e5-small-v2",
    "gte": "thenlper/gte-small"
}

# Cache backend options
CACHE_BACKENDS = {
    "file": "Thread-safe file-based cache (current)",
    "redis": "Future: Redis for distributed systems",
    "sqlite": "Future: SQLite for single-server"
}

# Default configurations
DEFAULT_CHUNK_CONFIG = {
    "strategy": SplitStrategy.RECURSIVE,
    "max_tokens": 256,
    "min_tokens": 100,
    "overlap_tokens": 30,
    "enable_dedup": True,
    "preclean_ocr": True,
    "max_chars": 1000,
    "min_chars": 200,
    "overlap_chars": 100,
    "min_chunk_quality": 0.5,
    "suspicious_patterns": [
        r'^[A-Z\s]+$',  # All caps
        r'(\w)\1{5,}',  # Repeated characters
        r'[^\w\s]{10,}',  # Too many special chars
    ]
}

DEFAULT_EMBEDDING_CONFIG = {
    "model_name": "miniLM",
    "batch_size": 32,
    "device": "auto",  # Will be set to "cuda" if available, else "cpu"
    "use_cache": True,
    "cache_dir": "./embedding_cache",
    "enable_deterministic": False,
    "max_memory_usage_gb": 8.0,
    "chunk_id_length": 16
}


@dataclass
class ChunkConfig:
    """Configuration for document chunking."""
    split_strategy: SplitStrategy = SplitStrategy.RECURSIVE
    max_tokens: int = 256
    min_tokens: int = 100
    overlap_tokens: int = 30
    enable_dedup: bool = True
    preclean_ocr: bool = True
    max_chars: int = 1000
    min_chars: int = 200
    overlap_chars: int = 100
    min_chunk_quality: float = 0.5
    suspicious_patterns: list = None
    
    def __post_init__(self):
        if self.suspicious_patterns is None:
            self.suspicious_patterns = DEFAULT_CHUNK_CONFIG["suspicious_patterns"]


@dataclass
class EmbeddingConfig:
    """Configuration for document embedding."""
    model_name: str = "miniLM"
    batch_size: int = 32
    device: str = "auto"
    use_cache: bool = True
    cache_dir: str = "./embedding_cache"
    enable_deterministic: bool = False
    max_memory_usage_gb: float = 8.0
    chunk_id_length: int = 16
    
    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_name(alias: str) -> str:
    """
    Get the full model name from an alias.
    
    Args:
        alias: Model alias (e.g., 'miniLM', 'bge')
    
    Returns:
        Full model name from HuggingFace
    
    Raises:
        ValueError: If alias is not supported
    """
    if alias not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model alias: {alias}. "
                        f"Supported: {list(SUPPORTED_MODELS.keys())}")
    return SUPPORTED_MODELS[alias]


def validate_config(config: Dict[str, Any], config_type: str) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        config_type: Type of config ('chunk' or 'embedding')
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If configuration is invalid
    """
    if config_type == "chunk":
        required_keys = ["strategy", "max_tokens", "min_tokens"]
        if not all(key in config for key in required_keys):
            raise ValueError(f"Missing required keys for chunk config: {required_keys}")
        
        if config["max_tokens"] <= config["min_tokens"]:
            raise ValueError("max_tokens must be greater than min_tokens")
            
    elif config_type == "embedding":
        required_keys = ["model_name", "batch_size"]
        if not all(key in config for key in required_keys):
            raise ValueError(f"Missing required keys for embedding config: {required_keys}")
        
        if config["batch_size"] <= 0:
            raise ValueError("batch_size must be positive")
    else:
        raise ValueError(f"Unknown config_type: {config_type}")
    
    return True 