# RAG-API: Production-Grade RAG Preprocessing System

A modular, high-performance Retrieval-Augmented Generation (RAG) preprocessing backend optimized for large document processing (>1,000 pages) with sub-30-second processing on standard GPUs.

## 🏗️ Project Structure

```
RAG-API/
├── chunker.py              # Document chunking with cleaning, quality, metadata
├── embedder.py             # Embedding with GPU, caching, batching, retry
├── indexer.py              # FAISS indexing with PostgreSQL metadata
├── utils/
│   ├── hashing.py          # Shared chunk ID generation
│   └── __init__.py
├── config.py               # Central config and model registry
├── __init__.py             # Main package exports
├── requirements.txt        # Dependencies
├── example_workflow.py     # End-to-end demonstration
├── example_complete_pipeline.py  # Complete pipeline demo
└── README.md               # This file
```

## 🔗 Shared Conventions

### Chunk ID Consistency
Both `chunker.py` and `embedder.py` use identical chunk ID generation via `utils.hashing.generate_chunk_id()`:

```python
from utils.hashing import generate_chunk_id

# Consistent SHA256-based chunk IDs across all modules
chunk_id = generate_chunk_id("sample text", length=16)
```

## 🧩 Core Modules

### 1. Document Chunker (`chunker.py`)

Semantic-aware, noise-resilient document chunking with:

- **Three Strategies**: `RECURSIVE` (section-based), `SENTENCE` (sentence-based), `FIXED` (character-based)
- **Token-Aware Sizing**: Uses `tiktoken` or regex fallback
- **OCR Cleaning**: Fixes common OCR artifacts (`0→O`, `|→I`)
- **Quality Assessment**: Flags suspicious chunks (all caps, repetition)
- **Deduplication**: Removes duplicates using shared chunk ID generation
- **Metadata Tracking**: Source, offsets, token count, quality scores

```python
from ingestion.chunker import create_chunker

chunker = create_chunker(
    split_strategy="recursive",
    max_tokens=256,
    min_tokens=100,
    overlap_tokens=30,
    enable_dedup=True,
    preclean_ocr=True
)

result = chunker.chunk_document(large_text, source="document.pdf")
print(f"Created {result.total_chunks} chunks in {result.processing_time:.2f}s")
```

### 2. Document Embedder (`embedder.py`)

GPU-optimized embedding with advanced features:

- **Multiple Models**: MiniLM, BGE, MPNet, E5, GTE via central registry
- **Thread-Safe Caching**: File-based with optional Redis/SQLite
- **Memory Safety**: Auto-batch sizing to prevent OOM
- **Retry Logic**: Robust model loading with `tenacity`
- **Deterministic Mode**: Reproducible results for testing
- **Performance Monitoring**: Cache hits, processing time, batch resizes

```python
from embedder.embedder import create_embedder

embedder = create_embedder(
    model_name="miniLM",  # Uses central model registry
    enable_deterministic=False,
    batch_size=32
)

result = embedder.embed_chunks(chunk_contents)
print(f"Embedded {len(result.vectors)} chunks")
print(f"Cache hits: {result.cache_hits}, misses: {result.cache_misses}")
```

### 3. Document Indexer (`indexer.py`)

FAISS vector indexing with PostgreSQL metadata storage:

- **Multiple Index Types**: Flat IP/L2, IVF, HNSW for different use cases
- **Compression Support**: Float16, Product Quantization for large-scale deployments
- **Document-Level Hashing**: Skip re-indexing if content unchanged
- **Metadata Persistence**: Full traceability with PostgreSQL storage
- **Hybrid Retrieval Ready**: Support for sparse vectors and keyword tags
- **Versioning**: FAISS index persistence with load/save utilities

```python
from ingestion.indexer import create_indexer

indexer = create_indexer(
    index_type="flat_ip",
    compression_type="none",
    db_host="localhost",
    db_name="rag_index"
)

result = indexer.index_document(
    doc_id="policy_001",
    chunks=chunk_contents,
    embeddings=embeddings,
    metadata_list=metadata,
    overwrite=False
)

# Search with metadata
results = indexer.search_similar(query_embedding, k=10)
for result in results:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Document: {result['metadata']['doc_id']}")
```

### 4. Central Configuration (`config.py`)

Unified configuration management:

```python
from config import SUPPORTED_MODELS, CACHE_BACKENDS, ChunkConfig, EmbeddingConfig

# Model registry
SUPPORTED_MODELS = {
    "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    # ... more models
}

# Cache backends
CACHE_BACKENDS = {
    "file": "Thread-safe file-based cache (current)",
    "redis": "Future: Redis for distributed systems",
    "sqlite": "Future: SQLite for single-server"
}
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd RAG-API

# Install dependencies
pip install -r requirements.txt

# Optional: Install spaCy model for better sentence splitting
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from ingestion.chunker import create_chunker
from embedder.embedder import create_embedder

# Step 1: Chunk a large document
chunker = create_chunker(strategy="recursive", max_tokens=256)
chunk_result = chunker.chunk_document(large_text, source="manual.pdf")

# Step 2: Embed chunks (cache handles deduplication)
embedder = create_embedder(model_name="miniLM")
embed_result = embedder.embed_chunks([c.content for c in chunk_result.chunks])

# Step 3: Query similarity
query = "What are the compliance requirements?"
query_emb = embedder.embed_query(query)
similarities = embedder.compute_similarity(query_emb, embed_result.vectors)

# Step 4: View performance
print(embedder.get_stats())
```

### End-to-End Examples

#### Basic Workflow
Run the basic chunking and embedding demonstration:

```bash
python example_workflow.py
```

This will:
1. Create a sample large document
2. Chunk it using recursive strategy
3. Embed chunks with caching
4. Test query similarity
5. Verify chunk ID consistency
6. Display performance metrics

#### Complete Pipeline (with Indexing)
Run the complete pipeline including FAISS indexing:

```bash
python example_complete_pipeline.py
```

This demonstrates:
1. Document chunking with semantic awareness
2. Embedding generation with caching
3. FAISS indexing with PostgreSQL metadata
4. Semantic search with traceable results
5. Document retrieval and statistics

**Note**: The complete pipeline requires PostgreSQL to be running. If PostgreSQL is not available, the demo will continue with chunking and embedding only.

## ⚙️ Configuration

### Chunking Configuration

```python
from config import ChunkConfig, SplitStrategy

config = ChunkConfig(
    strategy=SplitStrategy.RECURSIVE,
    max_tokens=256,
    min_tokens=100,
    overlap_tokens=30,
    enable_dedup=True,
    preclean_ocr=True,
    min_chunk_quality=0.5
)
```

### Embedding Configuration

```python
from config import EmbeddingConfig

config = EmbeddingConfig(
    model_name="miniLM",
    batch_size=32,
    device="auto",  # cuda if available, else cpu
    use_cache=True,
    enable_deterministic=False,
    max_memory_usage_gb=8.0
)
```

### Indexing Configuration

```python
from ingestion.indexer import IndexingConfig, IndexType, CompressionType

config = IndexingConfig(
    index_type=IndexType.FLAT_IP,  # Inner product for cosine similarity
    compression_type=CompressionType.NONE,  # No compression
    normalize_vectors=True,
    db_host="localhost",
    db_name="rag_index",
    db_user="postgres",
    db_password="your_password",
    enable_document_hashing=True,
    enable_sparse_vectors=False,  # For hybrid retrieval
    enable_keyword_tags=False
)
```

## 🎯 Performance Targets

The system is designed to meet these performance targets:

- ✅ **Sub-30-second processing** on RTX 3090 for 1,000+ page documents
- ✅ **Memory safety** with <80% GPU memory usage
- ✅ **No context/token limit violations**
- ✅ **Cross-module consistency** (chunk ID, caching, deduplication)
- ✅ **Thread-safe persistent caching**
- ✅ **Reproducibility** and observability

## 🔧 Advanced Features

### Memory-Aware Batching

The embedder automatically adjusts batch sizes based on GPU memory:

```python
# Automatically reduces batch size if GPU memory is tight
embedder = create_embedder(batch_size=64)  # May be reduced to 32 or 16
```

### Thread Safety

Both modules support concurrent processing:

```python
import threading

# Safe to use embedder in multiple threads
def process_chunks(chunks):
    result = embedder.embed_chunks(chunks)
    return result

threads = [threading.Thread(target=process_chunks, args=(chunk_batch,)) 
           for chunk_batch in chunk_batches]
```

### Deterministic Mode

For reproducible results in testing:

```python
embedder = create_embedder(enable_deterministic=True)
# Uses torch.use_deterministic_algorithms(True)
```

## 🧪 Testing

Run the comprehensive test suites:

```bash
# Test chunker
python -m pytest embedder/test_chunker.py -v

# Test embedder
python -m pytest embedder/test_embedder.py -v

# Test shared utilities
python -m pytest utils/ -v
```

## 📊 Monitoring

Both modules provide detailed statistics:

```python
# Chunker stats
chunk_stats = chunk_result.metadata
print(f"Avg tokens per chunk: {chunk_stats['avg_tokens_per_chunk']}")
print(f"Duplicate chunks removed: {chunk_result.duplicate_chunks_removed}")

# Embedder stats
embedder_stats = embedder.get_stats()
print(f"Cache hit rate: {embedder_stats['cache_hit_rate']:.2%}")
print(f"Batch resizes: {embedder_stats['batch_resizes']}")
```

## 🔄 Integration

### With the Indexer Module

```python
from ingestion.indexer import create_indexer

# Create indexer
indexer = create_indexer(
    index_type="flat_ip",
    db_host="localhost",
    db_name="rag_index"
)

# Index document
result = indexer.index_document(
    doc_id="policy_001",
    chunks=chunk_contents,
    embeddings=embeddings,
    metadata_list=metadata,
    overwrite=False
)

# Search with full metadata
results = indexer.search_similar(query_embedding, k=5)
for result in results:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Document: {result['metadata']['doc_id']}")
    print(f"Section: {result['metadata']['section_title']}")
```

### With Vector Stores (Direct FAISS)

```python
import faiss

# Create FAISS index
dimension = embed_result.vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embed_result.vectors.astype('float32'))

# Search
query_emb = embedder.embed_query("insurance requirements")
D, I = index.search(query_emb.reshape(1, -1).astype('float32'), k=5)
```

### With Hybrid Retrievers

```python
# Combine with BM25 for hybrid search
from rank_bm25 import BM25Okapi

# BM25 on chunk contents
bm25 = BM25Okapi([chunk.content for chunk in chunk_result.chunks])
bm25_scores = bm25.get_scores("insurance requirements")

# Combine with embedding similarities
hybrid_scores = 0.7 * similarities + 0.3 * bm25_scores
```

## 🛠️ Troubleshooting

### Common Issues

1. **OOM Errors**: Reduce `batch_size` or `max_memory_usage_gb`
2. **Slow Processing**: Enable caching, use GPU, increase batch size
3. **Non-reproducible Results**: Set `enable_deterministic=True`
4. **Import Errors**: Install optional dependencies (`spacy`, `nltk`, `tiktoken`)
5. **PostgreSQL Connection**: Ensure PostgreSQL is running and credentials are correct
6. **FAISS Index Issues**: Check index file permissions and disk space

### Performance Tuning

```python
# For maximum speed (if memory allows)
embedder = create_embedder(
    model_name="miniLM",  # Fastest model
    batch_size=64,        # Larger batches
    enable_deterministic=False  # Disable for speed
)

# For memory-constrained environments
embedder = create_embedder(
    model_name="miniLM",
    batch_size=16,        # Smaller batches
    max_memory_usage_gb=4.0  # Lower memory limit
)
```

## 📈 Future Enhancements

- **Redis Integration**: Distributed caching for multi-server deployments
- **SQLite Backend**: Lightweight persistent storage
- **Model Switching**: Hot-swappable embedding models
- **Streaming Processing**: Process documents in chunks without loading entire file
- **Advanced Quality Metrics**: More sophisticated text quality assessment
- **Hybrid Retrieval**: BM25 + dense vector combination
- **Multi-Modal Support**: Image and document embedding
- **Distributed Indexing**: Sharded FAISS indices for large-scale deployments

## 📄 License

This project is part of the HackRx6 RAG backend system.

## 🤝 Contributing

1. Follow the shared chunk ID convention
2. Add tests for new features
3. Update configuration in `config.py`
4. Document new functionality
5. Ensure thread safety for concurrent operations 