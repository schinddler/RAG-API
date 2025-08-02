"""
Integration example showing how the embedder module connects with other RAG components.
This demonstrates a complete pipeline from document ingestion to query processing.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import other modules
sys.path.append(str(Path(__file__).parent.parent))

from embedder import create_embedder, EmbeddingConfig
import numpy as np
from typing import List, Dict, Any, Tuple


class SimpleChunker:
    """Simple text chunker for demonstration."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
        return all_chunks


class SimpleVectorStore:
    """Simple in-memory vector store for demonstration."""
    
    def __init__(self):
        self.vectors = []
        self.chunk_ids = []
        self.chunks = []
        self.metadata = []
    
    def index(self, vectors: np.ndarray, chunk_ids: List[str], 
              chunks: List[str], metadata: List[Dict[str, Any]] = None):
        """Index vectors and associated data."""
        self.vectors.extend(vectors)
        self.chunk_ids.extend(chunk_ids)
        self.chunks.extend(chunks)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(chunk_ids))
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self.vectors:
            return []
        
        # Convert to numpy array
        vectors_array = np.array(self.vectors)
        
        # Compute cosine similarities
        similarities = np.dot(vectors_array, query_vector) / (
            np.linalg.norm(vectors_array, axis=1) * np.linalg.norm(query_vector)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "chunk": self.chunks[idx],
                "similarity": similarities[idx],
                "metadata": self.metadata[idx]
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_vectors": len(self.vectors),
            "vector_dimension": len(self.vectors[0]) if self.vectors else 0,
            "memory_usage_mb": len(self.vectors) * len(self.vectors[0]) * 4 / (1024 * 1024) if self.vectors else 0
        }


class RAGPipeline:
    """Complete RAG pipeline demonstrating integration."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize components
        self.chunker = SimpleChunker()
        self.embedder = create_embedder(model_name=model_name)
        self.vector_store = SimpleVectorStore()
        
        print(f"🚀 RAG Pipeline initialized with model: {model_name}")
    
    def ingest_documents(self, documents: List[str], 
                        metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete document ingestion pipeline.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            
        Returns:
            Ingestion statistics
        """
        print(f"\n📚 Ingesting {len(documents)} documents...")
        
        # Step 1: Chunk documents
        print("  Step 1: Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        print(f"    Created {len(chunks)} chunks")
        
        # Step 2: Embed chunks
        print("  Step 2: Embedding chunks...")
        embedding_result = self.embedder.embed_chunks(chunks)
        print(f"    Embedded {len(embedding_result.vectors)} chunks")
        print(f"    Processing time: {embedding_result.processing_time:.2f}s")
        print(f"    Cache hits: {embedding_result.cache_hits}, misses: {embedding_result.cache_misses}")
        
        # Step 3: Index in vector store
        print("  Step 3: Indexing in vector store...")
        self.vector_store.index(
            vectors=embedding_result.vectors,
            chunk_ids=embedding_result.chunk_ids,
            chunks=chunks,
            metadata=metadata
        )
        
        # Return statistics
        stats = {
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "embeddings_generated": len(embedding_result.vectors),
            "processing_time": embedding_result.processing_time,
            "cache_hits": embedding_result.cache_hits,
            "cache_misses": embedding_result.cache_misses,
            "vector_store_stats": self.vector_store.get_stats()
        }
        
        print(f"✅ Ingestion completed successfully!")
        return stats
    
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the RAG system.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with similarity scores
        """
        print(f"\n🔍 Querying: '{query}'")
        
        # Step 1: Embed query
        print("  Step 1: Embedding query...")
        query_embedding = self.embedder.embed_query(query)
        
        # Step 2: Search vector store
        print("  Step 2: Searching vector store...")
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        print(f"✅ Found {len(results)} relevant results")
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        embedder_stats = self.embedder.get_stats()
        vector_store_stats = self.vector_store.get_stats()
        
        return {
            "embedder": embedder_stats,
            "vector_store": vector_store_stats,
            "chunker": {
                "chunk_size": self.chunker.chunk_size,
                "overlap": self.chunker.overlap
            }
        }


def load_sample_documents() -> List[str]:
    """Load sample insurance/legal documents."""
    return [
        "Insurance policies must be renewed annually before the expiration date. Failure to renew may result in coverage gaps and potential legal liabilities. All policyholders are required to submit renewal applications at least 30 days before the current policy expires.",
        
        "Legal compliance requires regular audits conducted by certified professionals. These audits must be completed quarterly and documented thoroughly. The audit process includes reviewing financial records, operational procedures, and compliance with regulatory requirements. Any findings must be addressed within 60 days.",
        
        "Financial statements should be filed quarterly with the regulatory authority. Late filings may incur penalties of up to $10,000 per day. The statements must include balance sheets, income statements, and cash flow statements. All financial data must be accurate and verified by certified accountants.",
        
        "Risk assessment is mandatory for all business operations. This includes evaluating potential hazards, implementing mitigation strategies, and maintaining detailed records. Risk assessments must be updated annually or whenever significant changes occur in business operations. The assessment process involves identifying risks, analyzing their impact, and developing response strategies.",
        
        "Documentation must be maintained for a minimum of 7 years as per regulatory requirements. This includes all correspondence, contracts, and financial records. Electronic records must be backed up regularly and stored securely. Access to documentation must be restricted to authorized personnel only."
    ]


def demonstrate_complete_pipeline():
    """Demonstrate the complete RAG pipeline."""
    print("🚀 Complete RAG Pipeline Demonstration")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Load sample documents
    documents = load_sample_documents()
    
    # Ingest documents
    ingestion_stats = pipeline.ingest_documents(documents)
    
    print("\n📊 Ingestion Statistics:")
    for key, value in ingestion_stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # Test queries
    test_queries = [
        "What are the insurance renewal requirements?",
        "How often should audits be conducted?",
        "What are the penalties for late financial filings?",
        "How long should documentation be kept?",
        "What is involved in risk assessment?"
    ]
    
    print("\n🔍 Query Results:")
    for query in test_queries:
        results = pipeline.query(query, top_k=3)
        
        print(f"\nQuery: '{query}'")
        print("Top results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Similarity: {result['similarity']:.3f}")
            print(f"     Chunk: {result['chunk'][:100]}...")
    
    # System statistics
    print("\n📈 System Statistics:")
    system_stats = pipeline.get_system_stats()
    
    print("Embedder Stats:")
    for key, value in system_stats['embedder'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    print("\nVector Store Stats:")
    for key, value in system_stats['vector_store'].items():
        print(f"  {key}: {value}")


def demonstrate_model_comparison():
    """Compare different models in the pipeline."""
    print("\n🔬 Model Comparison in RAG Pipeline")
    print("=" * 50)
    
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    
    documents = load_sample_documents()[:3]  # Use subset for faster comparison
    query = "What are the insurance requirements?"
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\nTesting model: {model_name}")
        
        try:
            pipeline = RAGPipeline(model_name)
            
            # Ingest documents
            ingestion_stats = pipeline.ingest_documents(documents)
            
            # Query
            query_results = pipeline.query(query, top_k=3)
            
            results[model_name] = {
                "ingestion_time": ingestion_stats["processing_time"],
                "top_similarity": query_results[0]["similarity"] if query_results else 0,
                "avg_similarity": np.mean([r["similarity"] for r in query_results]) if query_results else 0,
                "embedding_dim": ingestion_stats["vector_store_stats"]["vector_dimension"]
            }
            
            print(f"  ✅ Ingestion time: {ingestion_stats['processing_time']:.2f}s")
            print(f"  ✅ Top similarity: {query_results[0]['similarity']:.3f}")
            print(f"  ✅ Embedding dimension: {ingestion_stats['vector_store_stats']['vector_dimension']}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print("\nModel Comparison Summary:")
    print("Model                    | Time(s) | Top Sim | Avg Sim | Dim")
    print("-" * 65)
    for model_name, result in results.items():
        short_name = model_name.split("/")[-1]
        print(f"{short_name:24} | {result['ingestion_time']:6.2f} | {result['top_similarity']:7.3f} | {result['avg_similarity']:7.3f} | {result['embedding_dim']:3d}")


def main():
    """Run the integration demonstration."""
    try:
        # Run complete pipeline demo
        demonstrate_complete_pipeline()
        
        # Run model comparison
        demonstrate_model_comparison()
        
        print("\n✅ Integration demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 