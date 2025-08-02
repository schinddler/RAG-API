"""
Example: Metadata Store Integration with Document Loader

This example demonstrates how to integrate the MetadataStore with the DocumentLoader
in a complete RAG pipeline workflow, showing document tracking, deduplication,
and audit logging throughout the ingestion process.
"""

import asyncio
import logging
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

from loader import DocumentLoader, DocumentLoadError
from metadata_store import (
    MetadataStore, 
    DocumentStatus, 
    DocumentType,
    create_metadata_store
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGIngestionPipeline:
    """
    Complete RAG ingestion pipeline that integrates document loading with metadata tracking.
    """
    
    def __init__(self, db_url: str = None, cache_dir: str = "./data/documents"):
        """
        Initialize the RAG ingestion pipeline.
        
        Args:
            db_url: Database URL (if None, uses SQLite in memory)
            cache_dir: Directory for document cache
        """
        if db_url is None:
            # Use SQLite in memory for this example
            db_url = "sqlite:///:memory:"
        
        self.db_url = db_url
        self.cache_dir = cache_dir
        
        # Initialize components
        self.loader = DocumentLoader(cache_dir=cache_dir)
        self.metadata_store = None  # Will be initialized in setup()
        
        # Pipeline statistics
        self.stats = {
            'documents_processed': 0,
            'documents_skipped': 0,
            'documents_failed': 0,
            'chunks_created': 0,
            'total_processing_time': 0
        }
    
    async def setup(self):
        """Initialize the metadata store."""
        self.metadata_store = await create_metadata_store(
            self.db_url,
            enable_audit_logging=True,
            max_retries=3
        )
        logger.info("RAG ingestion pipeline initialized")
    
    async def process_document_url(self, url: str, user_id: str = None) -> Dict[str, Any]:
        """
        Process a document from URL through the complete pipeline.
        
        Args:
            url: URL of the document to process
            user_id: Optional user ID for tracking
            
        Returns:
            Processing result with metadata and status
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Processing document from URL: {url}")
            
            # Step 1: Load document
            file_path, metadata = await self.loader.load_from_url_async(url)
            
            # Step 2: Check for duplicates using metadata store
            existing_doc = await self.metadata_store.get_doc_by_hash(metadata['doc_hash'])
            if existing_doc:
                logger.info(f"Document already processed: {metadata['filename']}")
                self.stats['documents_skipped'] += 1
                return {
                    'status': 'skipped',
                    'reason': 'Document already processed',
                    'metadata': metadata,
                    'processing_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
                }
            
            # Step 3: Save document metadata
            doc_id = await self.metadata_store.save_document_metadata({
                **metadata,
                'user_id': user_id,
                'source_type': 'url'
            })
            
            # Step 4: Update status to indicate parsing
            await self.metadata_store.mark_doc_status(doc_id, DocumentStatus.PARSED)
            
            # Step 5: Simulate text extraction (in real implementation, use parser)
            extracted_text = await self._extract_text(file_path, metadata)
            
            # Step 6: Create chunks (in real implementation, use chunker)
            chunks = await self._create_chunks(extracted_text, metadata)
            
            # Step 7: Save chunks to metadata store
            chunk_ids = await self.metadata_store.save_chunks(doc_id, chunks)
            
            # Step 8: Update status to indicate chunking
            await self.metadata_store.mark_doc_status(doc_id, DocumentStatus.CHUNKED)
            
            # Step 9: Simulate embedding (in real implementation, use embedder)
            if chunk_ids:
                await self._simulate_embedding(chunk_ids)
            
            # Step 10: Update status to indicate indexing
            await self.metadata_store.mark_doc_status(doc_id, DocumentStatus.INDEXED)
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            
            processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats['total_processing_time'] += processing_time_ms
            
            return {
                'status': 'success',
                'doc_id': str(doc_id),
                'metadata': metadata,
                'chunks_created': len(chunks),
                'processing_time_ms': processing_time_ms
            }
            
        except DocumentLoadError as e:
            logger.error(f"Document loading failed: {e}")
            self.stats['documents_failed'] += 1
            return {
                'status': 'failed',
                'reason': f"Loading error: {e}",
                'url': url,
                'processing_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
            }
        except Exception as e:
            logger.error(f"Unexpected error processing document: {e}")
            self.stats['documents_failed'] += 1
            return {
                'status': 'failed',
                'reason': f"Unexpected error: {e}",
                'url': url,
                'processing_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
            }
    
    async def process_multiple_documents(self, urls: List[str], user_id: str = None) -> List[Dict[str, Any]]:
        """
        Process multiple documents concurrently.
        
        Args:
            urls: List of document URLs to process
            user_id: Optional user ID for tracking
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing {len(urls)} documents concurrently")
        
        # Create tasks for concurrent processing
        tasks = [self.process_document_url(url, user_id) for url in urls]
        
        # Process all documents concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {urls[i]}: {result}")
                processed_results.append({
                    'status': 'failed',
                    'reason': f"Exception: {result}",
                    'url': urls[i]
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _extract_text(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Simulate text extraction from document.
        
        Args:
            file_path: Path to the document file
            metadata: Document metadata
            
        Returns:
            Extracted text
        """
        # In a real implementation, you would use the parser module
        # For this example, we'll simulate text extraction
        
        if metadata['extension'] == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Simulate extracted text for other formats
            return f"Extracted text from {metadata['filename']}. This is simulated content for demonstration purposes. The document contains important information that would be processed by the RAG pipeline."
    
    async def _create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks from extracted text.
        
        Args:
            text: Extracted text from document
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries
        """
        # In a real implementation, you would use the chunker module
        # For this example, we'll create simple chunks
        
        words = text.split()
        chunk_size = 50  # words per chunk
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'chunk_index': len(chunks),
                'content': chunk_text,
                'token_count': len(chunk_words),
                'page_number': 1,  # Simplified
                'section_title': f"Section {len(chunks) + 1}",
                'quality_score': 95,
                'is_suspicious': False
            })
        
        return chunks
    
    async def _simulate_embedding(self, chunk_ids: List[str]):
        """
        Simulate embedding generation for chunks.
        
        Args:
            chunk_ids: List of chunk UUIDs
        """
        # In a real implementation, you would use the embedder module
        # For this example, we'll simulate embedding generation
        
        for chunk_id in chunk_ids:
            if chunk_id:  # Check if chunk_id is not None
                # Simulate embedding ID
                embedding_id = f"emb_{str(chunk_id)[:8]}"
                await self.metadata_store.update_chunk_embedding(chunk_id, embedding_id)
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics.
        
        Returns:
            Pipeline statistics
        """
        # Get database statistics
        db_stats = await self.metadata_store.get_document_stats()
        
        # Combine with pipeline statistics
        combined_stats = {
            **self.stats,
            **db_stats,
            'avg_processing_time_ms': (
                self.stats['total_processing_time'] / max(self.stats['documents_processed'], 1)
            ),
            'success_rate': (
                self.stats['documents_processed'] / 
                max(self.stats['documents_processed'] + self.stats['documents_failed'], 1) * 100
            )
        }
        
        return combined_stats
    
    async def get_audit_trail(self, doc_id: str = None) -> List[Dict[str, Any]]:
        """
        Get audit trail for debugging and monitoring.
        
        Args:
            doc_id: Optional document ID to filter by
            
        Returns:
            List of audit log entries
        """
        return await self.metadata_store.get_audit_log(doc_id=doc_id)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.metadata_store:
            await self.metadata_store.close()
        logger.info("Pipeline cleanup completed")


async def main():
    """Example usage of the RAG ingestion pipeline."""
    
    # Initialize pipeline
    pipeline = RAGIngestionPipeline()
    await pipeline.setup()
    
    # Example document URLs (replace with real URLs)
    document_urls = [
        "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
        "https://raw.githubusercontent.com/microsoft/vscode/main/CONTRIBUTING.md",
        "https://raw.githubusercontent.com/microsoft/vscode/main/LICENSE"
    ]
    
    print("🚀 Starting RAG Ingestion Pipeline with Metadata Tracking")
    print("=" * 70)
    
    # Process documents
    results = await pipeline.process_multiple_documents(document_urls, user_id="example_user")
    
    # Display results
    print("\n📊 Processing Results:")
    print("-" * 50)
    
    for i, result in enumerate(results):
        print(f"\nDocument {i+1}: {document_urls[i]}")
        print(f"  Status: {result['status']}")
        
        if result['status'] == 'success':
            metadata = result['metadata']
            print(f"  Document ID: {result['doc_id']}")
            print(f"  Filename: {metadata['filename']}")
            print(f"  File size: {metadata['file_size']} bytes")
            print(f"  Document hash: {metadata['doc_hash'][:16]}...")
            print(f"  Chunks created: {result['chunks_created']}")
            print(f"  Processing time: {result['processing_time_ms']:.2f}ms")
        
        elif result['status'] == 'failed':
            print(f"  Error: {result['reason']}")
        
        elif result['status'] == 'skipped':
            print(f"  Reason: {result['reason']}")
    
    # Display statistics
    stats = await pipeline.get_pipeline_stats()
    print(f"\n📈 Pipeline Statistics:")
    print("-" * 50)
    print(f"Documents processed: {stats['documents_processed']}")
    print(f"Documents skipped: {stats['documents_skipped']}")
    print(f"Documents failed: {stats['documents_failed']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average chunks per document: {stats['avg_chunks_per_doc']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Average processing time: {stats['avg_processing_time_ms']:.2f}ms")
    
    # Display status breakdown
    print(f"\n📋 Status Breakdown:")
    print("-" * 30)
    for status, count in stats['status_breakdown'].items():
        print(f"  {status.value}: {count}")
    
    # Display recent audit trail
    audit_log = await pipeline.get_audit_trail()
    print(f"\n🔍 Recent Audit Trail (last 5 entries):")
    print("-" * 50)
    for entry in audit_log[:5]:
        print(f"  {entry['timestamp']} - {entry['step']} - {entry['status']} - {entry['message']}")
    
    # Cleanup
    await pipeline.cleanup()
    
    print("\n✅ RAG Ingestion Pipeline completed!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 