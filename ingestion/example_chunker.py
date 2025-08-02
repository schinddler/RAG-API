#!/usr/bin/env python3
"""
Example script demonstrating the chunker module capabilities.

This script showcases:
- Different chunking strategies (recursive, sentence, fixed)
- OCR text cleaning and quality assessment
- Deduplication and metadata propagation
- Integration with the embedder module
- Performance monitoring and statistics
"""

import time
import json
from chunker import (
    Chunker, 
    ChunkConfig, 
    SplitStrategy, 
    create_chunker
)


def load_sample_documents():
    """Load sample documents for demonstration."""
    return {
        "legal_document": """
        INSURANCE POLICY AGREEMENT
        
        1. COVERAGE TERMS AND CONDITIONS
        This insurance policy provides comprehensive coverage for the insured party as described herein.
        All terms and conditions must be strictly followed to maintain coverage validity.
        The policy covers damages up to $1,000,000 per incident.
        
        2. PREMIUM PAYMENTS AND SCHEDULE
        Premium payments are due on the first day of each month without exception.
        Late payments may result in immediate policy cancellation and loss of coverage.
        Payment methods include bank transfer, credit card, and automatic deductions.
        
        3. CLAIMS PROCESS AND PROCEDURES
        Claims must be filed within 30 days of the incident occurrence.
        Documentation must be provided with all claims including photographs and receipts.
        Claims processing typically takes 5-10 business days for standard cases.
        
        4. EXCLUSIONS AND LIMITATIONS
        This policy does not cover intentional acts or criminal activities.
        Natural disasters are covered under separate provisions and endorsements.
        Pre-existing conditions are excluded from coverage under this policy.
        """,
        
        "ocr_dirty_text": """
        Th1s |s OCR text w1th 0rrors   and   extra   spaces.
        S0me characters are m1sread by the OCR s0ftware.
        Th|s text needs cleaning bef0re processing.
        MULTIPLE   SPACES   AND   ALL   CAPS   TEXT!!!
        """,
        
        "repetitive_content": """
        Section A: Introduction
        This is the introduction section with important information.
        
        Section B: Main Content
        This is the main content section with detailed explanations.
        
        Section A: Introduction
        This is the introduction section with important information.
        
        Section C: Conclusion
        This is the conclusion section with final remarks.
        """,
        
        "plain_text": """
        This is a simple plain text document without any special formatting.
        It contains multiple sentences that should be chunked appropriately.
        The content is straightforward and easy to process.
        Each sentence provides useful information for the reader.
        The text flows naturally from one topic to the next.
        """
    }


def demonstrate_chunking_strategies():
    """Demonstrate different chunking strategies."""
    print("🔪 Chunking Strategies Demonstration")
    print("=" * 50)
    
    documents = load_sample_documents()
    legal_doc = documents["legal_document"]
    
    strategies = [
        ("recursive", "Structured documents with sections"),
        ("sentence", "Plain text with natural sentence boundaries"),
        ("fixed", "Fixed-size chunks for consistent processing")
    ]
    
    for strategy_name, description in strategies:
        print(f"\n📋 Strategy: {strategy_name.upper()} - {description}")
        print("-" * 40)
        
        chunker = create_chunker(
            split_strategy=strategy_name,
            max_tokens=150,
            min_tokens=50,
            overlap_tokens=20
        )
        
        result = chunker.chunk_document(legal_doc, source=f"legal_doc_{strategy_name}")
        
        print(f"  Chunks created: {result.total_chunks}")
        print(f"  Processing time: {result.processing_time:.3f}s")
        print(f"  Avg tokens per chunk: {result.metadata['avg_tokens_per_chunk']:.1f}")
        print(f"  Duplicates removed: {result.duplicate_chunks_removed}")
        print(f"  Suspicious chunks: {result.suspicious_chunks}")
        
        # Show first chunk as example
        if result.chunks:
            first_chunk = result.chunks[0]
            print(f"  First chunk preview: {first_chunk.content[:80]}...")
            print(f"  Chunk ID: {first_chunk.metadata.chunk_id}")


def demonstrate_ocr_cleaning():
    """Demonstrate OCR text cleaning capabilities."""
    print("\n🧹 OCR Text Cleaning Demonstration")
    print("=" * 50)
    
    documents = load_sample_documents()
    dirty_text = documents["ocr_dirty_text"]
    
    print("Original OCR text:")
    print(dirty_text)
    print("\n" + "-" * 40)
    
    # Test with OCR cleaning enabled
    chunker_with_cleaning = create_chunker(
        split_strategy="sentence",
        preclean_ocr=True,
        max_tokens=100
    )
    
    result = chunker_with_cleaning.chunk_document(dirty_text, source="ocr_dirty.txt")
    
    print("After OCR cleaning:")
    for i, chunk in enumerate(result.chunks):
        print(f"  Chunk {i+1}: {chunk.content}")
        print(f"    Quality score: {chunk.metadata.quality_score:.2f}")
        print(f"    Suspicious: {chunk.metadata.is_suspicious}")
    
    # Test without OCR cleaning for comparison
    chunker_without_cleaning = create_chunker(
        split_strategy="sentence",
        preclean_ocr=False,
        max_tokens=100
    )
    
    result_no_clean = chunker_without_cleaning.chunk_document(dirty_text, source="ocr_dirty.txt")
    
    print(f"\nComparison:")
    print(f"  With cleaning: {len(result.chunks)} chunks, {result.suspicious_chunks} suspicious")
    print(f"  Without cleaning: {len(result_no_clean.chunks)} chunks, {result_no_clean.suspicious_chunks} suspicious")


def demonstrate_deduplication():
    """Demonstrate deduplication capabilities."""
    print("\n🔄 Deduplication Demonstration")
    print("=" * 50)
    
    documents = load_sample_documents()
    repetitive_text = documents["repetitive_content"]
    
    print("Text with repeated sections:")
    print(repetitive_text)
    print("\n" + "-" * 40)
    
    # Test with deduplication enabled
    chunker_with_dedup = create_chunker(
        split_strategy="sentence",
        enable_dedup=True,
        max_tokens=100
    )
    
    result_with_dedup = chunker_with_dedup.chunk_document(repetitive_text, source="repetitive.txt")
    
    print("With deduplication:")
    print(f"  Total chunks: {result_with_dedup.total_chunks}")
    print(f"  Duplicates removed: {result_with_dedup.duplicate_chunks_removed}")
    
    for i, chunk in enumerate(result_with_dedup.chunks):
        print(f"  Chunk {i+1}: {chunk.content[:60]}...")
        print(f"    Chunk ID: {chunk.metadata.chunk_id}")
    
    # Test without deduplication for comparison
    chunker_without_dedup = create_chunker(
        split_strategy="sentence",
        enable_dedup=False,
        max_tokens=100
    )
    
    result_without_dedup = chunker_without_dedup.chunk_document(repetitive_text, source="repetitive.txt")
    
    print(f"\nComparison:")
    print(f"  With deduplication: {result_with_dedup.total_chunks} unique chunks")
    print(f"  Without deduplication: {result_without_dedup.total_chunks} total chunks")


def demonstrate_quality_assessment():
    """Demonstrate text quality assessment."""
    print("\n📊 Quality Assessment Demonstration")
    print("=" * 50)
    
    # Create text with varying quality
    mixed_quality_text = """
    This is well-formed text with proper punctuation and formatting.
    
    ALL CAPS TEXT WITH MULTIPLE   SPACES AND EXCESSIVE PUNCTUATION!!!!!
    
    Another good sentence with normal formatting and reasonable length.
    
    Text with excessive repetition repetition repetition repetition repetition.
    
    Very short text.
    
    Text with many special characters: @#$%^&*()_+-=[]{}|;':",./<>?
    """
    
    chunker = create_chunker(
        split_strategy="sentence",
        max_tokens=50,
        min_tokens=5
    )
    
    result = chunker.chunk_document(mixed_quality_text, source="mixed_quality.txt")
    
    print("Quality assessment results:")
    for i, chunk in enumerate(result.chunks):
        print(f"\n  Chunk {i+1}:")
        print(f"    Content: {chunk.content[:50]}...")
        print(f"    Quality score: {chunk.metadata.quality_score:.3f}")
        print(f"    Suspicious: {chunk.metadata.is_suspicious}")
        print(f"    Token count: {chunk.metadata.token_count}")
        print(f"    Chunk ID: {chunk.metadata.chunk_id}")
    
    # Summary statistics
    quality_scores = [chunk.metadata.quality_score for chunk in result.chunks]
    suspicious_count = sum(1 for chunk in result.chunks if chunk.metadata.is_suspicious)
    
    print(f"\nSummary:")
    print(f"  Average quality score: {sum(quality_scores) / len(quality_scores):.3f}")
    print(f"  Suspicious chunks: {suspicious_count}/{len(result.chunks)}")
    print(f"  Quality range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")


def demonstrate_metadata_propagation():
    """Demonstrate metadata propagation capabilities."""
    print("\n📋 Metadata Propagation Demonstration")
    print("=" * 50)
    
    documents = load_sample_documents()
    legal_doc = documents["legal_document"]
    
    chunker = create_chunker(
        split_strategy="recursive",
        max_tokens=200,
        min_tokens=50
    )
    
    result = chunker.chunk_document(legal_doc, source="insurance_policy.pdf")
    
    print("Metadata for each chunk:")
    for i, chunk in enumerate(result.chunks):
        print(f"\n  Chunk {i+1}:")
        print(f"    Source: {chunk.metadata.source}")
        print(f"    Chunk index: {chunk.metadata.chunk_index}")
        print(f"    Character range: {chunk.metadata.char_start} - {chunk.metadata.char_end}")
        print(f"    Token count: {chunk.metadata.token_count}")
        print(f"    Quality score: {chunk.metadata.quality_score:.3f}")
        print(f"    Suspicious: {chunk.metadata.is_suspicious}")
        print(f"    Chunk ID: {chunk.metadata.chunk_id}")
        print(f"    Content preview: {chunk.content[:80]}...")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n⚡ Performance Monitoring Demonstration")
    print("=" * 50)
    
    documents = load_sample_documents()
    legal_doc = documents["legal_document"]
    
    # Test different configurations
    configurations = [
        ("Small chunks", {"max_tokens": 100, "min_tokens": 30}),
        ("Medium chunks", {"max_tokens": 200, "min_tokens": 50}),
        ("Large chunks", {"max_tokens": 400, "min_tokens": 100}),
    ]
    
    for config_name, params in configurations:
        print(f"\n📊 Configuration: {config_name}")
        print("-" * 30)
        
        chunker = create_chunker(
            split_strategy="sentence",
            **params
        )
        
        start_time = time.time()
        result = chunker.chunk_document(legal_doc, source="performance_test.txt")
        end_time = time.time()
        
        print(f"  Processing time: {result.processing_time:.3f}s")
        print(f"  Total chunks: {result.total_chunks}")
        print(f"  Avg tokens per chunk: {result.metadata['avg_tokens_per_chunk']:.1f}")
        print(f"  Total tokens: {result.metadata['total_tokens']}")
        print(f"  Duplicates removed: {result.duplicate_chunks_removed}")
        print(f"  Suspicious chunks: {result.suspicious_chunks}")
        
        # Calculate throughput
        tokens_per_second = result.metadata['total_tokens'] / result.processing_time
        print(f"  Throughput: {tokens_per_second:.1f} tokens/second")


def demonstrate_integration_with_embedder():
    """Demonstrate integration with the embedder module."""
    print("\n🔗 Integration with Embedder Demonstration")
    print("=" * 50)
    
    try:
        # Import embedder (assuming it's available)
        import sys
        sys.path.append('../embedder')
        from embedder import create_embedder
        
        documents = load_sample_documents()
        legal_doc = documents["legal_document"]
        
        # Create chunker and embedder
        chunker = create_chunker(
            split_strategy="sentence",
            max_tokens=150,
            min_tokens=50
        )
        
        embedder = create_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=8
        )
        
        # Chunk the document
        print("Step 1: Chunking document...")
        chunk_result = chunker.chunk_document(legal_doc, source="integration_test.txt")
        print(f"  Created {chunk_result.total_chunks} chunks")
        
        # Extract chunk contents for embedding
        chunk_contents = [chunk.content for chunk in chunk_result.chunks]
        
        # Embed the chunks
        print("Step 2: Embedding chunks...")
        embedding_result = embedder.embed_chunks(chunk_contents)
        print(f"  Embedded {len(embedding_result.vectors)} chunks")
        print(f"  Embedding time: {embedding_result.processing_time:.3f}s")
        print(f"  Cache hits: {embedding_result.cache_hits}")
        print(f"  Cache misses: {embedding_result.cache_misses}")
        
        # Verify chunk ID consistency
        print("Step 3: Verifying chunk ID consistency...")
        for i, (chunk, chunk_id) in enumerate(zip(chunk_result.chunks, embedding_result.chunk_ids)):
            embedder_chunk_id = embedder._generate_chunk_id(chunk.content)
            print(f"  Chunk {i+1}: {chunk_id} == {embedder_chunk_id} ({chunk_id == embedder_chunk_id})")
        
        # Test query embedding and similarity
        print("Step 4: Testing query similarity...")
        query = "What are the premium payment requirements?"
        query_embedding = embedder.embed_query(query)
        
        similarities = embedder.compute_similarity(query_embedding, embedding_result.vectors)
        
        print("  Top 3 most similar chunks:")
        top_indices = similarities.argsort()[-3:][::-1]
        for i, idx in enumerate(top_indices):
            similarity = similarities[idx]
            chunk_content = chunk_contents[idx][:80] + "..."
            print(f"    {i+1}. Similarity: {similarity:.3f} - {chunk_content}")
        
        print("\n✅ Integration test completed successfully!")
        
    except ImportError as e:
        print(f"⚠️  Embedder module not available: {e}")
        print("  Skipping integration test. Make sure embedder.py is in the correct path.")


def main():
    """Run all demonstrations."""
    print("🚀 Chunker Module Demonstration")
    print("=" * 60)
    print()
    
    try:
        demonstrate_chunking_strategies()
        demonstrate_ocr_cleaning()
        demonstrate_deduplication()
        demonstrate_quality_assessment()
        demonstrate_metadata_propagation()
        demonstrate_performance_monitoring()
        demonstrate_integration_with_embedder()
        
        print("\n✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 