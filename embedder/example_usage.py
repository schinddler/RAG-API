"""
Example usage of the embedder module in a RAG system.
This script demonstrates embedding documents and queries, with caching and performance monitoring.
"""

import time
import numpy as np
from typing import List, Dict, Any

from embedder import create_embedder, SUPPORTED_MODELS, EmbeddingConfig


def load_sample_documents() -> List[str]:
    """Load sample insurance/legal documents for demonstration."""
    return [
        "Insurance policies must be renewed annually before the expiration date. Failure to renew may result in coverage gaps and potential legal liabilities.",
        
        "Legal compliance requires regular audits conducted by certified professionals. These audits must be completed quarterly and documented thoroughly.",
        
        "Financial statements should be filed quarterly with the regulatory authority. Late filings may incur penalties of up to $10,000 per day.",
        
        "Risk assessment is mandatory for all business operations. This includes evaluating potential hazards, implementing mitigation strategies, and maintaining detailed records.",
        
        "Documentation must be maintained for a minimum of 7 years as per regulatory requirements. This includes all correspondence, contracts, and financial records.",
        
        "Employee training programs must be completed annually and documented. Training should cover safety protocols, compliance procedures, and emergency response.",
        
        "Data protection measures must be implemented according to industry standards. This includes encryption, access controls, and regular security audits.",
        
        "Contract negotiations require approval from legal counsel before finalization. All terms must be reviewed for compliance and risk assessment.",
        
        "Environmental impact assessments are required for new projects exceeding $1 million in value. These assessments must be submitted 30 days before project commencement.",
        
        "Insurance claims must be filed within 30 days of the incident. Supporting documentation including photos, witness statements, and police reports should be included."
    ]


def demonstrate_basic_embedding():
    """Demonstrate basic embedding functionality."""
    print("=== Basic Embedding Demo ===")
    
    # Create embedder with default settings
    embedder = create_embedder()
    
    # Sample documents
    documents = load_sample_documents()[:5]
    
    print(f"Embedding {len(documents)} documents...")
    start_time = time.time()
    
    # Embed documents
    result = embedder.embed_chunks(documents)
    
    print(f"✅ Embedded {len(result.vectors)} documents in {result.processing_time:.2f}s")
    print(f"   Cache hits: {result.cache_hits}, misses: {result.cache_misses}")
    print(f"   Embedding dimension: {result.vectors.shape[1]}")
    
    # Test query embedding
    query = "What are the insurance renewal requirements?"
    query_embedding = embedder.embed_query(query)
    
    # Compute similarities
    similarities = embedder.compute_similarity(query_embedding, result.vectors)
    
    print(f"\nQuery: '{query}'")
    print("Top 3 most similar documents:")
    for i, (doc, sim) in enumerate(zip(documents, similarities)):
        if i < 3:
            print(f"  {i+1}. Similarity: {sim:.3f} - {doc[:80]}...")


def demonstrate_caching():
    """Demonstrate caching functionality."""
    print("\n=== Caching Demo ===")
    
    # Create embedder with caching enabled
    embedder = create_embedder(use_cache=True)
    
    documents = load_sample_documents()[:3]
    
    print("First embedding (should miss cache):")
    result1 = embedder.embed_chunks(documents)
    print(f"  Cache hits: {result1.cache_hits}, misses: {result1.cache_misses}")
    
    print("\nSecond embedding (should hit cache):")
    result2 = embedder.embed_chunks(documents)
    print(f"  Cache hits: {result2.cache_hits}, misses: {result2.cache_misses}")
    
    # Verify embeddings are identical
    if np.array_equal(result1.vectors, result2.vectors):
        print("  ✅ Cached embeddings are identical")
    else:
        print("  ❌ Cached embeddings differ")


def demonstrate_batch_processing():
    """Demonstrate batch processing with different batch sizes."""
    print("\n=== Batch Processing Demo ===")
    
    documents = load_sample_documents()
    
    batch_sizes = [8, 16, 32]
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=batch_size,
            use_cache=False  # Disable cache for fair comparison
        )
        embedder = create_embedder(config)
        
        start_time = time.time()
        result = embedder.embed_chunks(documents)
        end_time = time.time()
        
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Total time: {end_time - start_time:.2f}s")


def demonstrate_model_comparison():
    """Compare different embedding models."""
    print("\n=== Model Comparison Demo ===")
    
    documents = load_sample_documents()[:5]
    query = "What are the compliance requirements?"
    
    models_to_test = {
        "MiniLM": SUPPORTED_MODELS["miniLM"],
        "MPNet": SUPPORTED_MODELS["mpnet"],
        "BGE": SUPPORTED_MODELS["bge"]
    }
    
    results = {}
    
    for model_name, model_path in models_to_test.items():
        print(f"\nTesting {model_name}...")
        
        try:
            embedder = create_embedder(
                model_name=model_path,
                use_cache=False
            )
            
            # Embed documents
            doc_result = embedder.embed_chunks(documents)
            
            # Embed query
            query_embedding = embedder.embed_query(query)
            
            # Compute similarities
            similarities = embedder.compute_similarity(query_embedding, doc_result.vectors)
            
            results[model_name] = {
                "processing_time": doc_result.processing_time,
                "embedding_dim": doc_result.vectors.shape[1],
                "max_similarity": np.max(similarities),
                "avg_similarity": np.mean(similarities)
            }
            
            print(f"  ✅ Processing time: {doc_result.processing_time:.2f}s")
            print(f"  ✅ Embedding dimension: {doc_result.vectors.shape[1]}")
            print(f"  ✅ Max similarity: {np.max(similarities):.3f}")
            
        except Exception as e:
            print(f"  ❌ Error with {model_name}: {e}")
    
    # Summary
    print("\nModel Comparison Summary:")
    print("Model          | Time(s) | Dim | Max Sim | Avg Sim")
    print("-" * 50)
    for model_name, result in results.items():
        print(f"{model_name:14} | {result['processing_time']:6.2f} | {result['embedding_dim']:3d} | {result['max_similarity']:7.3f} | {result['avg_similarity']:7.3f}")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and statistics."""
    print("\n=== Performance Monitoring Demo ===")
    
    embedder = create_embedder()
    
    # Perform multiple embedding operations
    documents = load_sample_documents()
    
    print("Performing multiple embedding operations...")
    
    # First batch
    result1 = embedder.embed_chunks(documents[:5])
    
    # Second batch (should hit cache)
    result2 = embedder.embed_chunks(documents[:5])
    
    # Third batch (new documents)
    result3 = embedder.embed_chunks(documents[5:10])
    
    # Get comprehensive statistics
    stats = embedder.get_stats()
    
    print("\nPerformance Statistics:")
    print(f"  Model: {stats['model_name']}")
    print(f"  Device: {stats['device']}")
    print(f"  Total embeddings processed: {stats['total_embeddings']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"  Total processing time: {stats['total_processing_time']:.2f}s")
    print(f"  Average processing time per embedding: {stats['avg_processing_time']:.3f}s")
    
    print("\nCache Statistics:")
    cache_stats = stats['cache_stats']
    print(f"  Total cached embeddings: {cache_stats['total_embeddings']}")
    print(f"  Cache size: {cache_stats['cache_size_mb']:.2f} MB")
    print(f"  Models used in cache: {cache_stats['models_used']}")


def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n=== Error Handling Demo ===")
    
    embedder = create_embedder()
    
    # Test with empty input
    print("Testing empty input:")
    result = embedder.embed_chunks([])
    print(f"  Empty chunks result: {len(result.vectors)} vectors")
    
    # Test with invalid model (should fail gracefully)
    print("\nTesting invalid model:")
    try:
        invalid_embedder = create_embedder(model_name="invalid/model/name")
        print("  ❌ Should have failed")
    except Exception as e:
        print(f"  ✅ Properly handled invalid model: {type(e).__name__}")
    
    # Test with very long text
    print("\nTesting very long text:")
    long_text = "This is a very long document. " * 1000
    try:
        result = embedder.embed_chunks([long_text])
        print(f"  ✅ Successfully embedded long text ({len(long_text)} characters)")
    except Exception as e:
        print(f"  ❌ Failed to embed long text: {e}")


def main():
    """Run all demonstrations."""
    print("🚀 Embedder Module Demonstration")
    print("=" * 50)
    
    try:
        # Run all demos
        demonstrate_basic_embedding()
        demonstrate_caching()
        demonstrate_batch_processing()
        demonstrate_model_comparison()
        demonstrate_performance_monitoring()
        demonstrate_error_handling()
        
        print("\n✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 