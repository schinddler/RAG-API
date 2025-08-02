#!/usr/bin/env python3
"""
End-to-end RAG workflow example demonstrating chunking and embedding integration.
This script shows how to process large documents efficiently with the unified pipeline.
"""

import time
import logging
from pathlib import Path

# Import our modules
from ingestion.chunker import create_chunker
from embedder.embedder import create_embedder
from utils.hashing import generate_chunk_id

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_document() -> str:
    """Create a sample large document for testing."""
    sections = [
        "1. INTRODUCTION",
        "This document outlines the comprehensive insurance policy terms and conditions. "
        "All policyholders must understand these requirements before signing any agreements. "
        "The policy covers various scenarios including natural disasters, accidents, and liability claims.",
        
        "2. COVERAGE DETAILS",
        "The insurance policy provides coverage for property damage, personal injury, and legal expenses. "
        "Coverage limits vary based on the policy tier selected. "
        "Standard coverage includes up to $500,000 for property damage and $1,000,000 for liability claims.",
        
        "3. EXCLUSIONS",
        "This policy does not cover intentional acts, war-related damages, or nuclear incidents. "
        "Additionally, damages caused by wear and tear or lack of maintenance are excluded. "
        "Policyholders should review all exclusions carefully before filing claims.",
        
        "4. CLAIMS PROCESS",
        "To file a claim, policyholders must contact the claims department within 30 days of the incident. "
        "Required documentation includes incident reports, photographs, and repair estimates. "
        "Claims are typically processed within 15 business days of receiving complete documentation.",
        
        "5. PREMIUM CALCULATION",
        "Premiums are calculated based on risk factors including location, property value, and claims history. "
        "Discounts are available for security systems, multiple policies, and good payment history. "
        "Annual premium reviews ensure rates remain competitive and reflect current risk assessments.",
        
        "6. POLICY RENEWAL",
        "Policies automatically renew unless cancelled in writing 30 days before expiration. "
        "Renewal notices are sent 45 days before the policy expiration date. "
        "Changes to coverage or policy terms require written approval from the insurance company.",
        
        "7. DISPUTE RESOLUTION",
        "Disputes regarding claims or policy interpretation are resolved through mediation or arbitration. "
        "Legal proceedings may be initiated if mediation fails to reach a resolution. "
        "All disputes must be filed within one year of the incident or claim denial.",
        
        "8. CONTACT INFORMATION",
        "For questions about this policy, contact our customer service department. "
        "Phone support is available 24/7 for emergency claims and urgent matters. "
        "Email support is available during business hours for general inquiries and policy changes."
    ]
    
    return "\n\n".join(sections)


def main():
    """Demonstrate the complete RAG preprocessing pipeline."""
    logger.info("Starting RAG preprocessing pipeline demonstration")
    
    # Step 1: Create a large document (simulating 1,500+ pages)
    logger.info("Creating sample document...")
    document_text = create_sample_document()
    logger.info(f"Document created: {len(document_text)} characters")
    
    # Step 2: Chunk the document
    logger.info("Step 1: Chunking document...")
    start_time = time.time()
    
    chunker = create_chunker(
        split_strategy="recursive",
        max_tokens=256,
        min_tokens=100,
        overlap_tokens=30,
        enable_dedup=True,
        preclean_ocr=True
    )
    
    chunk_result = chunker.chunk_document(document_text, source="sample_insurance_policy.txt")
    
    chunking_time = time.time() - start_time
    logger.info(f"Chunking completed in {chunking_time:.2f}s")
    logger.info(f"  - Total chunks: {chunk_result.total_chunks}")
    logger.info(f"  - Duplicates removed: {chunk_result.duplicate_chunks_removed}")
    logger.info(f"  - Suspicious chunks: {chunk_result.suspicious_chunks}")
    
    # Step 3: Embed chunks
    logger.info("Step 2: Embedding chunks...")
    start_time = time.time()
    
    embedder = create_embedder(
        model_name="miniLM",
        enable_deterministic=False,
        batch_size=32
    )
    
    # Extract chunk contents for embedding
    chunk_contents = [chunk.content for chunk in chunk_result.chunks]
    
    embed_result = embedder.embed_chunks(chunk_contents)
    
    embedding_time = time.time() - start_time
    logger.info(f"Embedding completed in {embedding_time:.2f}s")
    logger.info(f"  - Vectors shape: {embed_result.vectors.shape}")
    logger.info(f"  - Cache hits: {embed_result.cache_hits}")
    logger.info(f"  - Cache misses: {embed_result.cache_misses}")
    
    # Step 4: Test query similarity
    logger.info("Step 3: Testing query similarity...")
    
    query = "What are the compliance requirements for filing insurance claims?"
    query_embedding = embedder.embed_query(query)
    
    similarities = embedder.compute_similarity(query_embedding, embed_result.vectors)
    
    # Find top matches
    top_indices = similarities.argsort()[-3:][::-1]
    logger.info("Top 3 most similar chunks:")
    for i, idx in enumerate(top_indices):
        similarity = similarities[idx]
        chunk_content = chunk_contents[idx][:100] + "..."
        logger.info(f"  {i+1}. Similarity: {similarity:.3f} - {chunk_content}")
    
    # Step 5: Verify chunk ID consistency
    logger.info("Step 4: Verifying chunk ID consistency...")
    
    # Test that both modules generate the same chunk IDs
    test_text = "This is a test chunk for ID consistency verification."
    chunker_id = chunker._generate_chunk_id(test_text)
    embedder_id = embedder._generate_chunk_id(test_text)
    shared_id = generate_chunk_id(test_text)
    
    consistency_check = (chunker_id == embedder_id == shared_id)
    logger.info(f"Chunk ID consistency: {'✓ PASS' if consistency_check else '✗ FAIL'}")
    logger.info(f"  - Chunker ID: {chunker_id}")
    logger.info(f"  - Embedder ID: {embedder_id}")
    logger.info(f"  - Shared ID: {shared_id}")
    
    # Step 6: Performance summary
    total_time = chunking_time + embedding_time
    logger.info("Step 5: Performance summary...")
    logger.info(f"  - Total processing time: {total_time:.2f}s")
    logger.info(f"  - Chunking time: {chunking_time:.2f}s ({chunking_time/total_time*100:.1f}%)")
    logger.info(f"  - Embedding time: {embedding_time:.2f}s ({embedding_time/total_time*100:.1f}%)")
    logger.info(f"  - Processing rate: {len(chunk_contents)/total_time:.1f} chunks/second")
    
    # Step 7: System stats
    logger.info("Step 6: System statistics...")
    
    chunker_stats = chunk_result.metadata
    embedder_stats = embedder.get_stats()
    
    logger.info("Chunker stats:")
    for key, value in chunker_stats.items():
        logger.info(f"  - {key}: {value}")
    
    logger.info("Embedder stats:")
    for key, value in embedder_stats.items():
        logger.info(f"  - {key}: {value}")
    
    logger.info("Pipeline demonstration completed successfully!")


if __name__ == "__main__":
    main() 