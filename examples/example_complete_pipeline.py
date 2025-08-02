#!/usr/bin/env python3
"""
Complete RAG Pipeline Example: Chunking → Embedding → Indexing → Search

This example demonstrates the full pipeline integration:
1. Document chunking with semantic awareness
2. Embedding generation with caching
3. FAISS indexing with PostgreSQL metadata
4. Semantic search with traceable results
"""

import time
import logging
import numpy as np
from pathlib import Path

# Import our modules
from ingestion.chunker import create_chunker, ChunkMetadata
from embedder.embedder import create_embedder
from ingestion.indexer import create_indexer, ChunkMetadata as IndexerChunkMetadata
from utils.hashing import generate_chunk_id

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_documents() -> dict:
    """Create sample documents for testing the complete pipeline."""
    documents = {
        "insurance_policy_001": {
            "content": """
            INSURANCE POLICY TERMS AND CONDITIONS
            
            1. COVERAGE DETAILS
            This policy provides comprehensive coverage for property damage, personal injury, and legal expenses.
            Coverage limits vary based on the policy tier selected. Standard coverage includes up to $500,000
            for property damage and $1,000,000 for liability claims. Additional riders are available for
            specialized coverage including flood, earthquake, and cyber liability protection.
            
            2. CLAIMS PROCESS
            To file a claim, policyholders must contact the claims department within 30 days of the incident.
            Required documentation includes incident reports, photographs, repair estimates, and any relevant
            police reports. Claims are typically processed within 15 business days of receiving complete
            documentation. Emergency claims may be expedited for urgent situations.
            
            3. PREMIUM CALCULATION
            Premiums are calculated based on risk factors including location, property value, claims history,
            and coverage limits. Discounts are available for security systems, multiple policies, and good
            payment history. Annual premium reviews ensure rates remain competitive and reflect current risk
            assessments. Payment plans are available for qualified policyholders.
            
            4. EXCLUSIONS
            This policy does not cover intentional acts, war-related damages, nuclear incidents, or damages
            caused by wear and tear or lack of maintenance. Additionally, certain high-risk activities and
            pre-existing conditions may be excluded. Policyholders should review all exclusions carefully
            before filing claims and consult with their agent for clarification.
            """,
            "filename": "insurance_policy_001.pdf"
        },
        "legal_contract_002": {
            "content": """
            SOFTWARE LICENSE AGREEMENT
            
            PARTIES AND PURPOSE
            This agreement is entered into between TechCorp Inc. ("Licensor") and Client Solutions LLC
            ("Licensee") for the licensing of proprietary software solutions. The agreement governs the
            use, distribution, and maintenance of the licensed software and related services.
            
            LICENSE TERMS
            The Licensor grants the Licensee a non-exclusive, non-transferable license to use the software
            for internal business purposes only. The license is valid for a period of 24 months from the
            effective date and may be renewed upon mutual agreement. Unauthorized copying, distribution,
            or reverse engineering is strictly prohibited and may result in immediate termination.
            
            PAYMENT TERMS
            License fees are due quarterly in advance. Late payments may result in service suspension
            and additional penalties. All fees are non-refundable except as explicitly stated in this
            agreement. The Licensor reserves the right to adjust pricing with 30 days written notice.
            
            SUPPORT AND MAINTENANCE
            The Licensor provides technical support during business hours and emergency support for
            critical issues. Software updates and patches are included in the license fee. Custom
            development and consulting services are available at additional cost.
            """,
            "filename": "legal_contract_002.pdf"
        },
        "financial_report_003": {
            "content": """
            QUARTERLY FINANCIAL REPORT - Q3 2024
            
            EXECUTIVE SUMMARY
            The company achieved strong financial performance in Q3 2024 with revenue growth of 15%
            compared to Q2 2024. Net profit margins improved to 12.5% due to operational efficiency
            gains and cost control measures. Cash flow from operations increased by 22% year-over-year.
            
            REVENUE ANALYSIS
            Product sales accounted for 65% of total revenue, with services contributing 35%.
            International markets showed the strongest growth at 25%, while domestic markets grew
            by 8%. New product launches contributed 12% to overall revenue growth.
            
            EXPENSE MANAGEMENT
            Operating expenses were well controlled, increasing only 8% compared to revenue growth
            of 15%. Marketing expenses increased by 20% to support new product launches, while
            administrative costs decreased by 5% due to process automation initiatives.
            
            OUTLOOK AND PROJECTIONS
            Based on current trends and market conditions, the company projects full-year revenue
            growth of 18-20% and improved profitability margins. Key growth drivers include
            expanding international presence, new product development, and strategic partnerships.
            """,
            "filename": "financial_report_003.pdf"
        }
    }
    
    return documents


def convert_chunker_metadata_to_indexer_metadata(chunk_metadata, doc_id: str) -> IndexerChunkMetadata:
    """Convert chunker metadata to indexer metadata format."""
    return IndexerChunkMetadata(
        doc_id=doc_id,
        chunk_id=chunk_metadata.chunk_id,
        chunk_index=chunk_metadata.chunk_index,
        char_start=chunk_metadata.char_start,
        char_end=chunk_metadata.char_end,
        section_title=chunk_metadata.section_title,
        source_filename=chunk_metadata.source_filename,
        token_count=chunk_metadata.token_count,
        quality_score=chunk_metadata.quality_score,
        is_suspicious=chunk_metadata.is_suspicious,
        extra_metadata={
            "original_metadata": {
                "section_title": chunk_metadata.section_title,
                "quality_score": chunk_metadata.quality_score,
                "is_suspicious": chunk_metadata.is_suspicious
            }
        }
    )


def main():
    """Demonstrate the complete RAG pipeline."""
    logger.info("Starting Complete RAG Pipeline Demonstration")
    
    # Step 1: Create sample documents
    logger.info("Step 1: Creating sample documents...")
    documents = create_sample_documents()
    logger.info(f"Created {len(documents)} sample documents")
    
    # Step 2: Initialize pipeline components
    logger.info("Step 2: Initializing pipeline components...")
    
    # Chunker
    chunker = create_chunker(
        split_strategy="recursive",
        max_tokens=256,
        min_tokens=100,
        overlap_tokens=30,
        enable_dedup=True,
        preclean_ocr=True
    )
    
    # Embedder
    embedder = create_embedder(
        model_name="miniLM",
        enable_deterministic=False,
        batch_size=32
    )
    
    # Indexer (Note: Requires PostgreSQL to be running)
    try:
        indexer = create_indexer(
            index_type="flat_ip",
            compression_type="none",
            db_host="localhost",
            db_name="rag_index",
            db_user="postgres",
            db_password="",  # Set your password here
            index_dir="./faiss_indexes"
        )
        postgres_available = True
        logger.info("PostgreSQL connection successful")
    except Exception as e:
        logger.warning(f"PostgreSQL not available: {e}")
        logger.info("Continuing with chunking and embedding only")
        postgres_available = False
        indexer = None
    
    # Step 3: Process each document through the pipeline
    all_chunks = []
    all_embeddings = []
    all_metadata = []
    
    for doc_id, doc_info in documents.items():
        logger.info(f"Processing document: {doc_id}")
        
        # Chunk the document
        chunk_result = chunker.chunk_document(
            doc_info["content"], 
            source=doc_info["filename"]
        )
        
        logger.info(f"  Chunked into {chunk_result.total_chunks} chunks")
        
        # Extract chunk contents for embedding
        chunk_contents = [chunk.content for chunk in chunk_result.chunks]
        
        # Generate embeddings
        embed_result = embedder.embed_chunks(chunk_contents)
        
        logger.info(f"  Generated {len(embed_result.vectors)} embeddings")
        
        # Convert metadata format for indexer
        indexer_metadata = []
        for chunk in chunk_result.chunks:
            indexer_meta = convert_chunker_metadata_to_indexer_metadata(
                chunk.metadata, doc_id
            )
            indexer_metadata.append(indexer_meta)
        
        # Index the document (if PostgreSQL is available)
        if postgres_available and indexer:
            try:
                index_result = indexer.index_document(
                    doc_id=doc_id,
                    chunks=chunk_contents,
                    embeddings=embed_result.vectors,
                    metadata_list=indexer_metadata,
                    overwrite=True,  # Allow re-indexing for demo
                    source_filename=doc_info["filename"]
                )
                
                logger.info(f"  Indexed successfully: {index_result.total_vectors} vectors")
                logger.info(f"  Processing time: {index_result.processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"  Failed to index document {doc_id}: {e}")
        
        # Store for batch processing
        all_chunks.extend(chunk_contents)
        all_embeddings.append(embed_result.vectors)
        all_metadata.extend(indexer_metadata)
    
    # Step 4: Demonstrate search capabilities
    if postgres_available and indexer:
        logger.info("Step 4: Demonstrating search capabilities...")
        
        # Test queries
        test_queries = [
            "What are the coverage limits for property damage?",
            "How long does it take to process claims?",
            "What are the payment terms for software licensing?",
            "What is the revenue growth projection?",
            "What are the policy exclusions?"
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: {query}")
            
            # Generate query embedding
            query_embedding = embedder.embed_query(query)
            
            # Search for similar chunks
            search_results = indexer.search_similar(query_embedding, k=3)
            
            # Display results
            for i, result in enumerate(search_results, 1):
                metadata = result["metadata"]
                similarity = result["similarity_score"]
                
                logger.info(f"  Result {i}:")
                logger.info(f"    Document: {metadata['doc_id']}")
                logger.info(f"    Section: {metadata['section_title'] or 'N/A'}")
                logger.info(f"    Similarity: {similarity:.3f}")
                logger.info(f"    Chunk Index: {metadata['chunk_index']}")
                
                # Show a snippet of the content
                chunk_content = all_chunks[metadata['chunk_index']] if metadata['chunk_index'] < len(all_chunks) else "Content not available"
                snippet = chunk_content[:150] + "..." if len(chunk_content) > 150 else chunk_content
                logger.info(f"    Content: {snippet}")
    
    # Step 5: Show system statistics
    logger.info("Step 5: System Statistics...")
    
    # Chunker stats
    logger.info("Chunker Statistics:")
    logger.info(f"  Total chunks processed: {len(all_chunks)}")
    
    # Embedder stats
    embedder_stats = embedder.get_stats()
    logger.info("Embedder Statistics:")
    logger.info(f"  Cache hit rate: {embedder_stats.get('cache_hit_rate', 0):.2%}")
    logger.info(f"  Total embeddings generated: {embedder_stats.get('total_embeddings', 0)}")
    logger.info(f"  Processing time: {embedder_stats.get('total_processing_time', 0):.2f}s")
    
    # Indexer stats (if available)
    if postgres_available and indexer:
        indexer_stats = indexer.get_index_stats()
        logger.info("Indexer Statistics:")
        logger.info(f"  FAISS vectors: {indexer_stats['faiss']['total_vectors']}")
        logger.info(f"  PostgreSQL documents: {indexer_stats['postgresql']['total_documents']}")
        logger.info(f"  PostgreSQL chunks: {indexer_stats['postgresql']['total_chunks']}")
        logger.info(f"  Index size: {indexer_stats['faiss'].get('index_size_mb', 0):.2f} MB")
    
    # Step 6: Demonstrate document retrieval
    if postgres_available and indexer:
        logger.info("Step 6: Document Retrieval Demo...")
        
        # Get all chunks for a specific document
        doc_id = "insurance_policy_001"
        document_chunks = indexer.get_document_chunks(doc_id)
        
        logger.info(f"Retrieved {len(document_chunks)} chunks for document {doc_id}")
        for i, chunk_meta in enumerate(document_chunks[:3]):  # Show first 3 chunks
            logger.info(f"  Chunk {i+1}: {chunk_meta['section_title'] or 'No title'}")
            logger.info(f"    Char range: {chunk_meta['char_start']}-{chunk_meta['char_end']}")
            logger.info(f"    Quality score: {chunk_meta['quality_score']:.2f}")
    
    # Cleanup
    if indexer:
        indexer.close()
    
    logger.info("Complete RAG Pipeline demonstration finished!")


if __name__ == "__main__":
    main() 