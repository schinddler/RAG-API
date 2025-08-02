"""
Example integration of parser with loader and metadata store.

This module demonstrates how the parser integrates with the DocumentLoader
and MetadataStore in a complete RAG ingestion pipeline.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

from loader import DocumentLoader, DocumentLoadError
from parser import FileParser, ParserError, parse_url, parse_file
from metadata_store import MetadataStore, DocumentStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGIngestionPipeline:
    """
    Complete RAG ingestion pipeline integrating loader, parser, and metadata store.
    
    This class demonstrates how to use all three components together
    for a production-grade document ingestion system.
    """
    
    def __init__(self, db_url: str = "sqlite:///rag_metadata.db"):
        """
        Initialize the RAG ingestion pipeline.
        
        Args:
            db_url: Database URL for metadata storage.
        """
        self.loader = DocumentLoader()
        self.parser = FileParser()
        self.metadata_store = MetadataStore(db_url)
        
    async def process_document_url(self, url: str, doc_type: str = "unknown") -> Dict:
        """
        Process a document from URL through the complete pipeline.
        
        Args:
            url: URL of the document to process.
            doc_type: Type of document (e.g., "legal", "financial", "insurance").
            
        Returns:
            Dictionary containing processing results and metadata.
        """
        try:
            logger.info(f"Starting processing of document from URL: {url}")
            
            # Step 1: Load document using DocumentLoader
            logger.info("Step 1: Loading document...")
            file_path, metadata = await self.loader.load_from_url(url)
            
            # Check if document already exists (deduplication)
            existing_doc = await self.metadata_store.get_doc_by_hash(metadata['doc_hash'])
            if existing_doc:
                logger.info(f"Document already exists with hash {metadata['doc_hash']}")
                return {
                    "status": "skipped",
                    "reason": "Document already processed",
                    "doc_id": existing_doc['doc_id'],
                    "metadata": existing_doc
                }
            
            # Step 2: Parse document using FileParser
            logger.info("Step 2: Parsing document...")
            parsed_result = await self.parser.parse_from_url(url)
            
            # Step 3: Save document metadata
            logger.info("Step 3: Saving document metadata...")
            doc_id = await self.metadata_store.save_document_metadata({
                **metadata,
                "doc_type": doc_type,
                "status": DocumentStatus.PARSED.value
            })
            
            # Step 4: Create chunks from parsed text
            logger.info("Step 4: Creating document chunks...")
            chunks = self._create_chunks(parsed_result["text"], doc_id)
            
            # Step 5: Save chunks
            logger.info("Step 5: Saving chunks...")
            chunk_ids = await self.metadata_store.save_chunks(doc_id, chunks)
            
            # Step 6: Mark document as processed
            await self.metadata_store.mark_doc_status(doc_id, DocumentStatus.PROCESSED.value)
            
            # Step 7: Log audit event
            await self.metadata_store._log_audit_event(
                "document_processed",
                "success",
                {
                    "doc_id": str(doc_id),
                    "chunks_created": len(chunks),
                    "text_length": parsed_result["length"]
                }
            )
            
            logger.info(f"Successfully processed document {doc_id} with {len(chunks)} chunks")
            
            return {
                "status": "success",
                "doc_id": str(doc_id),
                "chunks_created": len(chunks),
                "text_length": parsed_result["length"],
                "metadata": metadata,
                "parsed_result": parsed_result
            }
            
        except DocumentLoadError as e:
            logger.error(f"Document loading failed: {e}")
            return {"status": "failed", "reason": f"Loading failed: {str(e)}"}
            
        except ParserError as e:
            logger.error(f"Document parsing failed: {e}")
            return {"status": "failed", "reason": f"Parsing failed: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"status": "failed", "reason": f"Unexpected error: {str(e)}"}
    
    async def process_local_file(self, file_path: str, doc_type: str = "unknown") -> Dict:
        """
        Process a local file through the complete pipeline.
        
        Args:
            file_path: Path to the local file to process.
            doc_type: Type of document.
            
        Returns:
            Dictionary containing processing results and metadata.
        """
        try:
            logger.info(f"Starting processing of local file: {file_path}")
            
            # Step 1: Load document using DocumentLoader
            logger.info("Step 1: Loading document...")
            cached_path, metadata = self.loader.load_from_file(file_path)
            
            # Check if document already exists
            existing_doc = await self.metadata_store.get_doc_by_hash(metadata['doc_hash'])
            if existing_doc:
                logger.info(f"Document already exists with hash {metadata['doc_hash']}")
                return {
                    "status": "skipped",
                    "reason": "Document already processed",
                    "doc_id": existing_doc['doc_id'],
                    "metadata": existing_doc
                }
            
            # Step 2: Parse document using FileParser
            logger.info("Step 2: Parsing document...")
            parsed_result = self.parser.parse_from_file(file_path)
            
            # Step 3: Save document metadata
            logger.info("Step 3: Saving document metadata...")
            doc_id = await self.metadata_store.save_document_metadata({
                **metadata,
                "doc_type": doc_type,
                "status": DocumentStatus.PARSED.value
            })
            
            # Step 4: Create chunks from parsed text
            logger.info("Step 4: Creating document chunks...")
            chunks = self._create_chunks(parsed_result["text"], doc_id)
            
            # Step 5: Save chunks
            logger.info("Step 5: Saving chunks...")
            chunk_ids = await self.metadata_store.save_chunks(doc_id, chunks)
            
            # Step 6: Mark document as processed
            await self.metadata_store.mark_doc_status(doc_id, DocumentStatus.PROCESSED.value)
            
            # Step 7: Log audit event
            await self.metadata_store._log_audit_event(
                "document_processed",
                "success",
                {
                    "doc_id": str(doc_id),
                    "chunks_created": len(chunks),
                    "text_length": parsed_result["length"]
                }
            )
            
            logger.info(f"Successfully processed document {doc_id} with {len(chunks)} chunks")
            
            return {
                "status": "success",
                "doc_id": str(doc_id),
                "chunks_created": len(chunks),
                "text_length": parsed_result["length"],
                "metadata": metadata,
                "parsed_result": parsed_result
            }
            
        except DocumentLoadError as e:
            logger.error(f"Document loading failed: {e}")
            return {"status": "failed", "reason": f"Loading failed: {str(e)}"}
            
        except ParserError as e:
            logger.error(f"Document parsing failed: {e}")
            return {"status": "failed", "reason": f"Parsing failed: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"status": "failed", "reason": f"Unexpected error: {str(e)}"}
    
    def _create_chunks(self, text: str, doc_id: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Create chunks from document text.
        
        Args:
            text: The text to chunk.
            doc_id: Document ID.
            chunk_size: Maximum size of each chunk.
            overlap: Overlap between chunks.
            
        Returns:
            List of chunk dictionaries.
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    "chunk_text": chunk_text,
                    "chunk_index": len(chunks),
                    "token_count": len(chunk_words),
                    "page_number": None  # Could be extracted from text if available
                })
        
        return chunks
    
    async def get_processing_stats(self) -> Dict:
        """
        Get processing statistics from metadata store.
        
        Returns:
            Dictionary containing processing statistics.
        """
        try:
            stats = await self.metadata_store.get_document_stats()
            return stats
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close all resources."""
        await self.metadata_store.close()


async def example_usage():
    """Example usage of the RAG ingestion pipeline."""
    
    # Initialize pipeline
    pipeline = RAGIngestionPipeline()
    
    try:
        # Example 1: Process a document from URL
        print("=== Processing document from URL ===")
        result = await pipeline.process_document_url(
            "https://example.com/sample.pdf",
            doc_type="legal"
        )
        print(f"URL processing result: {result}")
        
        # Example 2: Process a local file
        print("\n=== Processing local file ===")
        
        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for the RAG pipeline.\n" * 10)
            test_file = f.name
        
        try:
            result = await pipeline.process_local_file(test_file, doc_type="test")
            print(f"Local file processing result: {result}")
        finally:
            os.unlink(test_file)
        
        # Example 3: Get processing statistics
        print("\n=== Processing Statistics ===")
        stats = await pipeline.get_processing_stats()
        print(f"Processing stats: {stats}")
        
        # Example 4: Demonstrate parser integration
        print("\n=== Parser Integration Demo ===")
        
        # Test parser with a simple text
        test_text = "This is a test document for parsing."
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_text)
            test_file = f.name
        
        try:
            # Use parser directly
            parsed = parse_file(test_file)
            print(f"Direct parser result: {parsed}")
            
            # Use parser through pipeline
            pipeline_result = await pipeline.process_local_file(test_file, "demo")
            print(f"Pipeline parser result: {pipeline_result}")
            
        finally:
            os.unlink(test_file)
        
    except Exception as e:
        print(f"Error in example usage: {e}")
    
    finally:
        await pipeline.close()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage()) 