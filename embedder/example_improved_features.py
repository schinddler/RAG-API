#!/usr/bin/env python3
"""
Example script demonstrating the improved features of the embedder module.

This script showcases:
- SHA256 hashing with improved normalization
- Thread safety with concurrent access
- Memory-aware batching
- Deterministic mode for reproducibility
- Retry logic for model loading
- Advanced cache features
"""

import time
import threading
import numpy as np
from embedder import (
    Embedder, 
    EmbeddingConfig, 
    create_embedder, 
    SUPPORTED_MODELS,
    CACHE_BACKENDS
)


def demonstrate_sha256_hashing():
    """Demonstrate SHA256 hashing with improved normalization."""
    print("🔐 SHA256 Hashing with Improved Normalization")
    print("=" * 50)
    
    embedder = create_embedder()
    
    # Test various text formats
    test_cases = [
        ("Hello, World!", "hello world"),
        ("  Multiple    Spaces  ", "multiple spaces"),
        ("Punctuation!@#$%^&*()", "punctuation"),
        ("Mixed CASE and 123 numbers", "mixed case and 123 numbers"),
        ("\n\nNewlines\n\n", "newlines"),
        ("Tabs\t\t\there", "tabs here"),
    ]
    
    for input_text, expected_normalized in test_cases:
        chunk_id = embedder._generate_chunk_id(input_text)
        print(f"Input: '{input_text}'")
        print(f"  → Chunk ID: {chunk_id} (length: {len(chunk_id)})")
        print()
    
    # Test that similar texts produce same IDs
    text1 = "Hello, World!"
    text2 = "hello world"
    text3 = "  Hello,   World!  "
    
    id1 = embedder._generate_chunk_id(text1)
    id2 = embedder._generate_chunk_id(text2)
    id3 = embedder._generate_chunk_id(text3)
    
    print(f"Normalization test:")
    print(f"  '{text1}' → {id1}")
    print(f"  '{text2}' → {id2}")
    print(f"  '{text3}' → {id3}")
    print(f"  All identical: {id1 == id2 == id3}")
    print()


def demonstrate_thread_safety():
    """Demonstrate thread safety with concurrent access."""
    print("🔄 Thread Safety with Concurrent Access")
    print("=" * 50)
    
    embedder = create_embedder(batch_size=4)
    test_chunks = [f"Document chunk {i} for thread safety testing" for i in range(20)]
    results = []
    errors = []
    
    def worker_function(thread_id):
        """Worker function for threading test."""
        try:
            start_time = time.time()
            result = embedder.embed_chunks(test_chunks)
            end_time = time.time()
            results.append({
                'thread_id': thread_id,
                'result': result,
                'processing_time': end_time - start_time
            })
            print(f"  Thread {thread_id}: Completed in {end_time - start_time:.2f}s")
        except Exception as e:
            errors.append((thread_id, e))
            print(f"  Thread {thread_id}: Error - {e}")
    
    # Create multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker_function, args=(i,))
        threads.append(thread)
    
    print("Starting concurrent embedding operations...")
    start_time = time.time()
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    print(f"All threads completed in {end_time - start_time:.2f}s")
    print(f"Successful threads: {len(results)}")
    print(f"Errors: {len(errors)}")
    
    if results:
        # Verify all results are consistent
        first_result = results[0]['result']
        for result_data in results[1:]:
            result = result_data['result']
            assert len(result.vectors) == len(first_result.vectors)
            print(f"  Thread {result_data['thread_id']}: {len(result.vectors)} embeddings")
    
    # Check cache stats
    cache_stats = embedder.cache.get_stats()
    print(f"Cache stats: {cache_stats['total_embeddings']} total embeddings")
    print(f"Thread safe: {cache_stats['thread_safe']}")
    print(f"File locking: {cache_stats['file_locking']}")
    print()


def demonstrate_memory_aware_batching():
    """Demonstrate memory-aware batch size calculation."""
    print("💾 Memory-Aware Batching")
    print("=" * 50)
    
    embedder = create_embedder(batch_size=32)
    
    # Test with different batch sizes
    test_batch_sizes = [8, 16, 32, 64, 128]
    text_length = 100
    
    print("Testing safe batch size calculation:")
    for batch_size in test_batch_sizes:
        safe_size = embedder._calculate_safe_batch_size(batch_size, text_length)
        print(f"  Requested: {batch_size}, Safe: {safe_size}")
        if safe_size != batch_size:
            print(f"    → Batch size reduced due to memory constraints")
    
    # Test with large chunks
    large_chunks = ["This is a very long document chunk " * 50 for _ in range(10)]
    
    print(f"\nEmbedding {len(large_chunks)} large chunks...")
    start_time = time.time()
    result = embedder.embed_chunks(large_chunks)
    end_time = time.time()
    
    print(f"Completed in {end_time - start_time:.2f}s")
    print(f"Cache hits: {result.cache_hits}, misses: {result.cache_misses}")
    
    # Check if batch resizing occurred
    stats = embedder.get_stats()
    print(f"Batch resizes: {stats['batch_resizes']}")
    print()


def demonstrate_deterministic_mode():
    """Demonstrate deterministic mode for reproducibility."""
    print("🎯 Deterministic Mode for Reproducibility")
    print("=" * 50)
    
    # Create embedder with deterministic mode
    config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        enable_deterministic=True,
        batch_size=4
    )
    embedder = Embedder(config)
    
    # Test reproducible embeddings
    test_texts = [
        "Insurance policies must be renewed annually.",
        "Legal compliance requires regular audits.",
        "Financial statements should be filed quarterly."
    ]
    
    print("Testing deterministic embeddings...")
    
    # First run
    embeddings1 = []
    for text in test_texts:
        embedding = embedder.embed_query(text)
        embeddings1.append(embedding)
    
    # Second run
    embeddings2 = []
    for text in test_texts:
        embedding = embedder.embed_query(text)
        embeddings2.append(embedding)
    
    # Compare results
    all_identical = True
    for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
        identical = np.array_equal(emb1, emb2)
        print(f"  Text {i+1}: {'✅ Identical' if identical else '❌ Different'}")
        all_identical = all_identical and identical
    
    print(f"All embeddings identical: {'✅ Yes' if all_identical else '❌ No'}")
    
    # Check stats
    stats = embedder.get_stats()
    print(f"Deterministic mode: {stats['deterministic_mode']}")
    print()


def demonstrate_retry_logic():
    """Demonstrate retry logic for model loading."""
    print("🔄 Retry Logic for Model Loading")
    print("=" * 50)
    
    print("Creating embedder with retry logic...")
    start_time = time.time()
    
    # This will use retry logic if tenacity is available
    embedder = create_embedder()
    
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time:.2f}s")
    
    # Test that model works
    test_text = "Testing retry logic with model loading"
    embedding = embedder.embed_query(test_text)
    print(f"Model working: ✅ (embedding shape: {embedding.shape})")
    
    # Check if retry logic is available
    try:
        from tenacity import retry
        print("Tenacity available: ✅")
    except ImportError:
        print("Tenacity available: ❌ (retry logic disabled)")
    
    print()


def demonstrate_cache_backends():
    """Demonstrate cache backend options."""
    print("🗄️ Cache Backend Options")
    print("=" * 50)
    
    print("Available cache backends:")
    for backend, description in CACHE_BACKENDS.items():
        print(f"  {backend}: {description}")
    
    print("\nCurrent implementation: Thread-safe file-based cache")
    print("Future implementations can be added for:")
    print("  - Redis: Distributed systems")
    print("  - SQLite: Single-server deployments")
    print("  - Memory: Development/testing")
    print()


def demonstrate_performance_monitoring():
    """Demonstrate comprehensive performance monitoring."""
    print("📊 Performance Monitoring")
    print("=" * 50)
    
    embedder = create_embedder()
    
    # Run some operations
    test_chunks = [f"Performance test chunk {i}" for i in range(50)]
    
    print("Running performance test...")
    result = embedder.embed_chunks(test_chunks)
    
    # Get comprehensive stats
    stats = embedder.get_stats()
    
    print("Embedder Statistics:")
    print(f"  Model: {stats['model_name']}")
    print(f"  Device: {stats['device']}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print(f"  Total embeddings: {stats['total_embeddings']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"  Total processing time: {stats['total_processing_time']:.2f}s")
    print(f"  Average processing time: {stats['avg_processing_time']:.3f}s")
    print(f"  Batch resizes: {stats['batch_resizes']}")
    print(f"  Deterministic mode: {stats['deterministic_mode']}")
    print(f"  Thread safe cache: {stats['thread_safe_cache']}")
    
    print("\nCache Statistics:")
    cache_stats = stats['cache_stats']
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    print()


def main():
    """Run all demonstrations."""
    print("🚀 Embedder Improved Features Demonstration")
    print("=" * 60)
    print()
    
    try:
        demonstrate_sha256_hashing()
        demonstrate_thread_safety()
        demonstrate_memory_aware_batching()
        demonstrate_deterministic_mode()
        demonstrate_retry_logic()
        demonstrate_cache_backends()
        demonstrate_performance_monitoring()
        
        print("✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 