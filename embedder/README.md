# Embedder Module

A production-grade embedding module for the RAG system that provides efficient, cached, and batch-processed text embeddings using sentence-transformers models.

## Features

✅ **Multiple Model Support**: Compatible with all sentence-transformers models (BGE, MiniLM, MPNet, E5, GTE, etc.)

✅ **Persistent Caching**: File-based caching to avoid recomputing embeddings

✅ **Batch Processing**: Memory-efficient batch processing for large document sets

✅ **Device Optimization**: Automatic CUDA/CPU detection and optimization

✅ **Performance Tracking**: Comprehensive metrics and logging

✅ **Modular Design**: Easy to swap models and configurations

✅ **Production Ready**: Error handling, logging, and comprehensive testing

✅ **Thread Safety**: Thread-safe caching with file locking for concurrent access

✅ **Memory Safety**: Memory-aware batching to prevent OOM errors

✅ **Reproducibility**: Deterministic algorithms for consistent results

✅ **Advanced Hashing**: SHA256-based chunk IDs with improved normalization

✅ **Retry Logic**: Robust model loading with automatic retries

## Quick Start

### Basic Usage

```python
from embedder import create_embedder

# Create embedder with default settings
embedder = create_embedder()

# Embed document chunks
chunks = [
    "Insurance policies must be renewed annually.",
    "Legal compliance requires regular audits.",
    "Financial statements should be filed quarterly."
]

result = embedder.embed_chunks(chunks)
print(f"Embedded {len(result.vectors)} chunks")
print(f"Processing time: {result.processing_time:.2f}s")

# Embed a query
query = "What are the insurance requirements?"
query_embedding = embedder.embed_query(query)

# Compute similarities
similarities = embedder.compute_similarity(query_embedding, result.vectors)
```

### Advanced Configuration

```python
from embedder import EmbeddingConfig, Embedder

# Custom configuration
config = EmbeddingConfig(
    model_name="sentence-transformers/all-mpnet-base-v2",
    batch_size=64,
    max_seq_length=512,
    use_cache=True,
    cache_dir="./my_cache"
)

embedder = Embedder(config)
```

## Supported Models

The embedder supports all sentence-transformers models. Common options include:

```python
SUPPORTED_MODELS = {
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",      # Fast, 384d
    "mpnet": "sentence-transformers/all-mpnet-base-v2",      # Balanced, 768d
    "bge": "BAAI/bge-small-en-v1.5",                        # High quality, 384d
    "e5": "intfloat/e5-small-v2",                           # Good performance, 384d
    "gte": "thenlper/gte-small"                             # Multilingual, 384d
}
```

## API Reference

### EmbeddingConfig

Configuration class for embedding settings:

```python
@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    show_progress_bar: bool = False
    device: Optional[str] = None
    use_cache: bool = True
    cache_dir: str = "./embedding_cache"
    embedding_dim: Optional[int] = None
    # New configuration options
    enable_deterministic: bool = False  # For reproducibility in training/test mode
    max_memory_usage_gb: float = 8.0  # Memory safety limit for batching
    chunk_id_length: int = 16  # Length of truncated SHA256 hash
```

### Embedder Class

Main embedding class with the following methods:

#### `embed_chunks(chunks: List[str], metadata: Optional[Dict] = None) -> EmbeddingResult`

Embeds a list of text chunks with caching and batch processing.

**Parameters:**
- `chunks`: List of text chunks to embed
- `metadata`: Optional metadata for tracking

**Returns:**
- `EmbeddingResult` with vectors, chunk_ids, and processing info

#### `embed_query(query: str) -> np.ndarray`

Embeds a single query string for retrieval.

**Parameters:**
- `query`: Query string to embed

**Returns:**
- Query embedding vector

#### `compute_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray`

Computes cosine similarity between query and chunk embeddings.

**Parameters:**
- `query_embedding`: Query embedding vector
- `chunk_embeddings`: Array of chunk embedding vectors

**Returns:**
- Array of similarity scores

#### `get_stats() -> Dict[str, Any]`

Returns comprehensive statistics and performance metrics.

#### `clear_cache()`

Clears the embedding cache.

#### `change_model(model_name: str)`

Changes the embedding model.

### EmbeddingResult

Result container for embedding operations:

```python
@dataclass
class EmbeddingResult:
    vectors: np.ndarray              # Embedding vectors
    chunk_ids: List[str]            # Generated chunk IDs
    metadata: Dict[str, Any]        # Optional metadata
    processing_time: float          # Processing time in seconds
    cache_hits: int                 # Number of cache hits
    cache_misses: int               # Number of cache misses
```

## Caching System

The embedder includes a file-based caching system that:

- Stores embeddings as `.npy` files
- Maintains metadata in `metadata.json`
- Automatically handles cache hits/misses
- Provides cache statistics

### Cache Configuration

```python
# Enable caching (default)
embedder = create_embedder(use_cache=True)

# Disable caching
embedder = create_embedder(use_cache=False)

# Custom cache directory
config = EmbeddingConfig(cache_dir="./my_embeddings_cache")
embedder = Embedder(config)
```

### Cache Management

```python
# Get cache statistics
stats = embedder.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")

# Clear cache
embedder.clear_cache()
```

## Performance Optimization

### Batch Processing

The embedder automatically processes chunks in batches for optimal performance:

```python
# Large batch for faster processing
config = EmbeddingConfig(batch_size=64)
embedder = Embedder(config)

# Small batch for memory-constrained environments
config = EmbeddingConfig(batch_size=8)
embedder = Embedder(config)
```

### Device Selection

Automatic device detection with manual override:

```python
# Automatic (recommended)
embedder = create_embedder()

# Manual device selection
config = EmbeddingConfig(device="cuda")  # or "cpu"
embedder = Embedder(config)
```

## Integration with RAG Pipeline

### Ingestion Pipeline

```python
# In your ingestion service
from embedder import create_embedder
from vector_store import VectorStore

# Initialize components
embedder = create_embedder()
vector_store = VectorStore()

# Process documents
chunks = chunker.chunk_documents(documents)
embedding_result = embedder.embed_chunks(chunks)

# Index in vector store
vector_store.index(embedding_result.vectors, embedding_result.chunk_ids)
```

### Query Pipeline

```python
# In your query service
query = "What are the insurance requirements?"
query_embedding = embedder.embed_query(query)

# Search vector store
results = vector_store.search(query_embedding, top_k=5)
```

## Monitoring and Metrics

The embedder provides comprehensive metrics:

```python
stats = embedder.get_stats()
print(f"Model: {stats['model_name']}")
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
print(f"Total processing time: {stats['total_processing_time']:.2f}s")
```

## Advanced Features

### Thread Safety

The embedder now provides thread-safe caching with file locking:

```python
# Thread-safe cache operations
import threading

def worker_function():
    result = embedder.embed_chunks(["Document chunk"])
    return result

# Multiple threads can safely access the same embedder
threads = [threading.Thread(target=worker_function) for _ in range(3)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
```

### Memory-Aware Batching

Automatic batch size adjustment to prevent OOM errors:

```python
# The embedder automatically adjusts batch size based on available GPU memory
config = EmbeddingConfig(
    batch_size=64,  # Will be reduced if needed
    max_memory_usage_gb=8.0
)
embedder = Embedder(config)

# Check if batch resizing occurred
stats = embedder.get_stats()
print(f"Batch resizes: {stats['batch_resizes']}")
```

### Deterministic Mode

Enable reproducible results for training and testing:

```python
# Enable deterministic algorithms
config = EmbeddingConfig(enable_deterministic=True)
embedder = Embedder(config)

# Same input will always produce same output
embedding1 = embedder.embed_query("test")
embedding2 = embedder.embed_query("test")
assert np.array_equal(embedding1, embedding2)
```

### Advanced Chunk ID Generation

Improved SHA256-based chunk IDs with regex normalization:

```python
# Advanced text normalization removes punctuation and extra whitespace
text1 = "Hello, World!"
text2 = "hello world"
text3 = "  Hello,   World!  "

# All produce the same chunk ID
id1 = embedder._generate_chunk_id(text1)
id2 = embedder._generate_chunk_id(text2)
id3 = embedder._generate_chunk_id(text3)
assert id1 == id2 == id3
```

### Retry Logic

Robust model loading with automatic retries:

```python
# Model loading automatically retries on failure
# Uses exponential backoff with 3 attempts
embedder = create_embedder()  # Retry logic is built-in
```

### Cache Backend Options

The embedder supports multiple cache backends for scalability:

```python
# Current implementation: Thread-safe file-based cache
# Future options (can be implemented):
# - Redis for distributed systems
# - SQLite for single-server deployments
# - In-memory for development/testing

from embedder import CACHE_BACKENDS
print("Available cache backends:", list(CACHE_BACKENDS.keys()))
```

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest test_embedder.py -v

# Run with coverage
pytest test_embedder.py --cov=embedder --cov-report=html
```

## Installation

### Core Dependencies

```bash
# Install core dependencies
pip install sentence-transformers torch numpy

# Install from requirements file
pip install -r requirements.txt
```

### Optional Dependencies

For advanced features, install optional dependencies:

```bash
# For retry logic
pip install tenacity

# For cross-platform file locking
pip install portalocker

# For Redis-based caching (distributed systems)
pip install redis

# For development and testing
pip install pytest pytest-cov
```

### Dependencies Overview

- **Core**: `sentence-transformers`, `torch`, `numpy` - Required for embedding functionality
- **Optional**: `tenacity` - Retry logic for model loading
- **Optional**: `portalocker` - Cross-platform file locking for thread safety
- **Optional**: `redis` - Redis-based caching for distributed systems
- **Built-in**: `sqlite3` - SQLite-based caching (Python standard library)
- **Built-in**: `fcntl` - File locking on Unix systems (Python standard library)

## Environment Variables

You can configure the embedder using environment variables:

```bash
export EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
export EMBEDDING_BATCH_SIZE="64"
export EMBEDDING_CACHE_DIR="./embeddings_cache"
export EMBEDDING_USE_CACHE="true"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: The embedder automatically adjusts batch size, but you can also:
   ```python
   config = EmbeddingConfig(batch_size=8, max_memory_usage_gb=4.0)
   ```

2. **Slow Processing**: Increase batch size (if memory allows)
   ```python
   config = EmbeddingConfig(batch_size=64)
   ```

3. **Cache Corruption**: Clear cache and restart
   ```python
   embedder.clear_cache()
   ```

4. **Model Download Issues**: Check internet connection and model name. Retry logic will help with temporary failures.

5. **Thread Safety Issues**: Ensure you're using the thread-safe cache (default) and have `portalocker` installed for file locking.

6. **Non-reproducible Results**: Enable deterministic mode for consistent results:
   ```python
   config = EmbeddingConfig(enable_deterministic=True)
   ```

### Performance Tips

- Use appropriate batch sizes for your hardware
- Enable caching for repeated embeddings
- Use CUDA if available
- Consider model size vs. quality trade-offs
- Monitor cache hit rates for optimization

## Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## License

This module is part of the RAG system and follows the same license terms. 