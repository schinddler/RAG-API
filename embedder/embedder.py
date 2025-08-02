"""
Production-grade embedding module for RAG system.
Supports multiple sentence-transformers models with caching and batch processing.
"""

import os
import logging
import re
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import time
import json
import threading

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Import shared utilities and config
try:
    from ..utils.hashing import generate_chunk_id
    from ..config import EmbeddingConfig as BaseEmbeddingConfig, get_model_name
except ImportError:
    # Handle case when running as script
    from utils.hashing import generate_chunk_id
    from config import EmbeddingConfig as BaseEmbeddingConfig, get_model_name

# Optional imports for advanced features
try:
    import portalocker
    PORTALOCKER_AVAILABLE = True
except ImportError:
    PORTALOCKER_AVAILABLE = False
    portalocker = None

try:
    import fcntl
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False
    fcntl = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    retry = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for embedding models and settings."""
    # Inherits from BaseEmbeddingConfig in config.py
    # Additional embedder-specific config can be added here if needed
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    show_progress_bar: bool = False
    embedding_dim: Optional[int] = None


@dataclass
class EmbeddingResult:
    """Result container for embedding operations."""
    vectors: np.ndarray
    chunk_ids: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    cache_hits: int
    cache_misses: int


class ThreadSafeEmbeddingCache:
    """Thread-safe file-based cache for embeddings with file locking."""
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.lock_file = self.cache_dir / "cache.lock"
        self._load_metadata()
        self._lock = threading.RLock()  # Reentrant lock for thread safety
    
    def _acquire_file_lock(self, timeout: float = 10.0) -> bool:
        """Acquire file lock for thread-safe operations."""
        if not PORTALOCKER_AVAILABLE and not FCNTL_AVAILABLE:
            logger.warning("No file locking available - using thread lock only")
            return True
        
        try:
            if PORTALOCKER_AVAILABLE:
                # Use portalocker for cross-platform file locking
                self._file_lock = portalocker.Lock(
                    str(self.lock_file),
                    timeout=timeout,
                    flags=portalocker.LOCK_EX | portalocker.LOCK_NB
                )
                self._file_lock.acquire()
            elif FCNTL_AVAILABLE:
                # Use fcntl for Unix systems
                self._lock_fd = open(self.lock_file, 'w')
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except Exception as e:
            logger.warning(f"Failed to acquire file lock: {e}")
            return False
    
    def _release_file_lock(self):
        """Release file lock."""
        try:
            if hasattr(self, '_file_lock') and PORTALOCKER_AVAILABLE:
                self._file_lock.release()
            elif hasattr(self, '_lock_fd') and FCNTL_AVAILABLE:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                self._lock_fd.close()
        except Exception as e:
            logger.warning(f"Failed to release file lock: {e}")
    
    def _load_metadata(self) -> None:
        """Load cache metadata with thread safety."""
        with self._lock:
            if self.metadata_file.exists():
                try:
                    with open(self.metadata_file, 'r') as f:
                        self.metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache metadata: {e}")
                    self.metadata = {}
            else:
                self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata with thread safety and file locking."""
        with self._lock:
            if self._acquire_file_lock():
                try:
                    with open(self.metadata_file, 'w') as f:
                        json.dump(self.metadata, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save cache metadata: {e}")
                finally:
                    self._release_file_lock()
    
    def get(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get embedding from cache with thread safety."""
        with self._lock:
            if chunk_id not in self.metadata:
                return None
            
            cache_file = self.cache_dir / f"{chunk_id}.npy"
            if cache_file.exists():
                try:
                    embedding = np.load(cache_file)
                    return embedding
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding {chunk_id}: {e}")
                    return None
            return None
    
    def set(self, chunk_id: str, embedding: np.ndarray, model_name: str) -> None:
        """Store embedding in cache with thread safety and file locking."""
        with self._lock:
            if self._acquire_file_lock():
                try:
                    cache_file = self.cache_dir / f"{chunk_id}.npy"
                    np.save(cache_file, embedding)
                    
                    # Update metadata
                    self.metadata[chunk_id] = {
                        "model_name": model_name,
                        "embedding_dim": embedding.shape[0],
                        "created_at": time.time()
                    }
                    self._save_metadata()
                except Exception as e:
                    logger.error(f"Failed to cache embedding {chunk_id}: {e}")
                finally:
                    self._release_file_lock()
    
    def clear(self) -> None:
        """Clear all cached embeddings with thread safety."""
        with self._lock:
            if self._acquire_file_lock():
                try:
                    for file in self.cache_dir.glob("*.npy"):
                        file.unlink()
                    self.metadata = {}
                    self._save_metadata()
                    logger.info("Embedding cache cleared")
                except Exception as e:
                    logger.error(f"Failed to clear cache: {e}")
                finally:
                    self._release_file_lock()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            try:
                cache_size_mb = sum(f.stat().st_size for f in self.cache_dir.glob("*.npy")) / (1024 * 1024)
            except Exception:
                cache_size_mb = 0.0
            
            return {
                "total_embeddings": len(self.metadata),
                "cache_size_mb": cache_size_mb,
                "models_used": list(set(m["model_name"] for m in self.metadata.values())),
                "thread_safe": True,
                "file_locking": PORTALOCKER_AVAILABLE or FCNTL_AVAILABLE
            }


class Embedder:
    """
    Production-grade embedder for RAG system.
    
    Features:
    - Multiple sentence-transformers models
    - Batch processing for efficiency
    - Thread-safe persistent caching
    - Device optimization (CUDA/CPU)
    - Memory-efficient processing with safety limits
    - Comprehensive logging and metrics
    - Retry logic for model loading
    - Deterministic algorithms for reproducibility
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedder with configuration.
        
        Args:
            config: Embedding configuration. If None, uses defaults.
        """
        self.config = config or EmbeddingConfig()
        
        # Enable deterministic algorithms if requested
        if self.config.enable_deterministic:
            torch.use_deterministic_algorithms(True)
            logger.info("Deterministic algorithms enabled for reproducibility")
        
        # Set device
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing embedder with model: {self.config.model_name}")
        logger.info(f"Using device: {self.config.device}")
        
        # Initialize model with retry logic
        self._load_model_with_retry()
        
        # Initialize thread-safe cache
        self.cache = ThreadSafeEmbeddingCache(self.config.cache_dir) if self.config.use_cache else None
        
        # Performance tracking
        self.stats = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_processing_time": 0.0,
            "batch_resizes": 0
        }
    
    def _load_model_with_retry(self) -> None:
        """Load the sentence-transformers model with retry logic."""
        if TENACITY_AVAILABLE:
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                retry=retry_if_exception_type((Exception,)),
                before_sleep=lambda retry_state: logger.warning(
                    f"Model loading attempt {retry_state.attempt_number} failed, retrying..."
                )
            )
            def _load_model():
                self._load_model()
            
            _load_model()
        else:
            logger.warning("Tenacity not available - loading model without retry logic")
            self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence-transformers model."""
        try:
            self.model = SentenceTransformer(self.config.model_name)
            self.model.max_seq_length = self.config.max_seq_length
            
            # Move to device
            self.model = self.model.to(self.config.device)
            
            # Get embedding dimension
            if self.config.embedding_dim is None:
                # Get dimension from model
                test_embedding = self.model.encode("test", convert_to_numpy=True)
                self.config.embedding_dim = test_embedding.shape[0]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.config.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise
    
    def _generate_chunk_id(self, text: str) -> str:
        """
        Generate a stable, normalized ID using SHA256 hash for deduplication.
        Uses the shared generate_chunk_id function for consistency.
        
        Args:
            text: Input text to normalize and hash
            
        Returns:
            Truncated SHA256 hash (16 characters by default)
        """
        return generate_chunk_id(text, length=self.config.chunk_id_length)
    
    def _calculate_safe_batch_size(self, current_batch_size: int, text_length: int) -> int:
        """
        Calculate a safe batch size to avoid OOM errors.
        
        Args:
            current_batch_size: Current batch size
            text_length: Average text length in the batch
            
        Returns:
            Safe batch size
        """
        if self.config.device == "cpu":
            return current_batch_size
        
        try:
            # Get GPU memory info
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(0)
                total_memory_gb = gpu_props.total_memory / (1024**3)
                free_memory_gb = torch.cuda.memory_reserved(0) / (1024**3)
                available_memory_gb = total_memory_gb - free_memory_gb
                
                # Estimate memory usage per embedding
                # Rough estimate: embedding_dim * 4 bytes * batch_size * safety_factor
                estimated_memory_per_batch_gb = (
                    self.config.embedding_dim * 4 * current_batch_size * 2.5
                ) / (1024**3)
                
                # If estimated usage exceeds available memory, reduce batch size
                if estimated_memory_per_batch_gb > available_memory_gb * 0.8:  # 80% safety margin
                    safe_batch_size = int(current_batch_size * 0.5)
                    logger.warning(
                        f"Memory usage estimate ({estimated_memory_per_batch_gb:.2f}GB) exceeds "
                        f"available GPU memory ({available_memory_gb:.2f}GB). "
                        f"Reducing batch size from {current_batch_size} to {safe_batch_size}"
                    )
                    self.stats["batch_resizes"] += 1
                    return max(1, safe_batch_size)
                
        except Exception as e:
            logger.warning(f"Could not calculate safe batch size: {e}")
        
        return current_batch_size
    
    def _process_batch(self, chunks: List[str], chunk_ids: List[str]) -> Tuple[List[np.ndarray], int, int]:
        """
        Process a batch of chunks, handling cache lookups and memory safety.
        
        Args:
            chunks: List of text chunks
            chunk_ids: List of chunk IDs
            
        Returns:
            Tuple of (embeddings, cache_hits, cache_misses)
        """
        embeddings = []
        cache_hits = 0
        cache_misses = 0
        
        # Check cache first
        uncached_chunks = []
        uncached_ids = []
        
        for chunk, chunk_id in zip(chunks, chunk_ids):
            if self.cache:
                cached_embedding = self.cache.get(chunk_id)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    cache_hits += 1
                    continue
            
            uncached_chunks.append(chunk)
            uncached_ids.append(chunk_id)
            cache_misses += 1
        
        # Process uncached chunks in batch with memory safety
        if uncached_chunks:
            try:
                # Calculate safe batch size
                avg_text_length = sum(len(chunk) for chunk in uncached_chunks) / len(uncached_chunks)
                safe_batch_size = self._calculate_safe_batch_size(self.config.batch_size, avg_text_length)
                
                batch_embeddings = self.model.encode(
                    uncached_chunks,
                    batch_size=safe_batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=self.config.show_progress_bar,
                    normalize_embeddings=self.config.normalize_embeddings
                )
                
                # Cache the new embeddings
                if self.cache:
                    for embedding, chunk_id in zip(batch_embeddings, uncached_ids):
                        self.cache.set(chunk_id, embedding, self.config.model_name)
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                raise
        
        return embeddings, cache_hits, cache_misses
    
    def embed_chunks(self, chunks: List[str], metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        """
        Embed a list of text chunks with caching and batch processing.
        
        Args:
            chunks: List of text chunks to embed
            metadata: Optional metadata for tracking
            
        Returns:
            EmbeddingResult with vectors, chunk_ids, and processing info
        """
        if not chunks:
            return EmbeddingResult(
                vectors=np.array([]),
                chunk_ids=[],
                metadata=metadata or {},
                processing_time=0.0,
                cache_hits=0,
                cache_misses=0
            )
        
        start_time = time.time()
        
        # Generate chunk IDs with improved normalization
        chunk_ids = [self._generate_chunk_id(chunk) for chunk in chunks]
        
        # Process in batches
        all_embeddings = []
        total_cache_hits = 0
        total_cache_misses = 0
        
        for i in range(0, len(chunks), self.config.batch_size):
            batch_chunks = chunks[i:i + self.config.batch_size]
            batch_ids = chunk_ids[i:i + self.config.batch_size]
            
            batch_embeddings, cache_hits, cache_misses = self._process_batch(batch_chunks, batch_ids)
            all_embeddings.extend(batch_embeddings)
            total_cache_hits += cache_hits
            total_cache_misses += cache_misses
        
        processing_time = time.time() - start_time
        
        # Update stats
        self.stats["total_embeddings"] += len(chunks)
        self.stats["cache_hits"] += total_cache_hits
        self.stats["cache_misses"] += total_cache_misses
        self.stats["total_processing_time"] += processing_time
        
        # Log performance
        logger.info(f"Embedded {len(chunks)} chunks in {processing_time:.2f}s "
                   f"(cache: {total_cache_hits} hits, {total_cache_misses} misses)")
        
        return EmbeddingResult(
            vectors=np.array(all_embeddings),
            chunk_ids=chunk_ids,
            metadata=metadata or {},
            processing_time=processing_time,
            cache_hits=total_cache_hits,
            cache_misses=total_cache_misses
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for retrieval.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding vector
        """
        try:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def compute_similarity(self, query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and chunk embeddings.
        
        Args:
            query_embedding: Query embedding vector
            chunk_embeddings: Array of chunk embedding vectors
            
        Returns:
            Array of similarity scores
        """
        try:
            similarities = cos_sim(query_embedding, chunk_embeddings)
            return similarities.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Failed to compute similarities: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedder statistics and performance metrics."""
        cache_stats = self.cache.get_stats() if self.cache else {}
        
        return {
            "model_name": self.config.model_name,
            "embedding_dim": self.config.embedding_dim,
            "device": self.config.device,
            "total_embeddings": self.stats["total_embeddings"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": (self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])) * 100,
            "total_processing_time": self.stats["total_processing_time"],
            "avg_processing_time": self.stats["total_processing_time"] / max(1, self.stats["total_embeddings"]),
            "batch_resizes": self.stats["batch_resizes"],
            "deterministic_mode": self.config.enable_deterministic,
            "thread_safe_cache": self.cache is not None,
            "cache_stats": cache_stats
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
    
    def change_model(self, model_name: str) -> None:
        """
        Change the embedding model.
        
        Args:
            model_name: New model name
        """
        logger.info(f"Changing model from {self.config.model_name} to {model_name}")
        self.config.model_name = model_name
        self._load_model_with_retry()


# Factory function for easy model selection
def create_embedder(
    model_name: str = "miniLM",
    use_cache: bool = True,
    batch_size: int = 32,
    device: Optional[str] = None,
    enable_deterministic: bool = False
) -> Embedder:
    """
    Factory function to create an embedder with common configurations.
    
    Args:
        model_name: Model alias (e.g., 'miniLM', 'bge') or full model name
        use_cache: Whether to use caching
        batch_size: Batch size for processing
        device: Device to use (cuda/cpu)
        enable_deterministic: Enable deterministic algorithms for reproducibility
        
    Returns:
        Configured Embedder instance
    """
    # Convert alias to full model name if needed
    try:
        full_model_name = get_model_name(model_name)
    except ValueError:
        # If not an alias, use as-is (full model name)
        full_model_name = model_name
    
    config = EmbeddingConfig(
        model_name=full_model_name,
        use_cache=use_cache,
        batch_size=batch_size,
        device=device,
        enable_deterministic=enable_deterministic
    )
    return Embedder(config)


if __name__ == "__main__":
    # Example usage
    embedder = create_embedder(model_name="miniLM")
    
    # Test embedding
    chunks = [
        "This is a test document about insurance policies.",
        "Another document about legal requirements.",
        "Financial document with important information."
    ]
    
    result = embedder.embed_chunks(chunks)
    print(f"Embedded {len(result.vectors)} chunks")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Cache hits: {result.cache_hits}, misses: {result.cache_misses}")
    
    # Test query embedding
    query = "What are the insurance requirements?"
    query_embedding = embedder.embed_query(query)
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Test similarity
    similarities = embedder.compute_similarity(query_embedding, result.vectors)
    print(f"Similarities: {similarities}")
    
    # Print stats
    print("\nEmbedder Stats:")
    stats = embedder.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
