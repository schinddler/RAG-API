"""
Example: Document Loader in RAG Pipeline

This example demonstrates how to integrate the DocumentLoader into a complete
RAG pipeline workflow, including document loading, parsing, chunking, and
metadata management.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from loader import DocumentLoader, DocumentLoadError, DocumentValidationError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Example RAG pipeline that demonstrates document loader integration.
    """
    
    def __init__(self, cache_dir: str = "./data/documents"):
        self.loader = DocumentLoader(
            cache_dir=cache_dir,
            max_file_size_mb=100,  # Allow larger files for legal documents
            timeout_seconds=60,    # Longer timeout for large files
            max_retries=5,         # More retries for reliability
            enable_async=True
        )
        self.processed_documents = {}  # Track processed documents
    
    async def process_document_url(self, url: str) -> Dict[str, Any]:
        """
        Process a document from URL through the complete RAG pipeline.
        
        Args:
            url: URL of the document to process
            
        Returns:
            Processing result with metadata and status
        """
        try:
            logger.info(f"Processing document from URL: {url}")
            
            # Step 1: Load document
            file_path, metadata = await self.loader.load_from_url_async(url)
            
            # Step 2: Check for duplicates
            if self.loader.check_duplicate(metadata['doc_hash']):
                logger.info(f"Document already processed: {metadata['filename']}")
                return {
                    'status': 'skipped',
                    'reason': 'Document already processed',
                    'metadata': metadata
                }
            
            # Step 3: Route to appropriate parser based on file type
            parser_result = await self._route_to_parser(file_path, metadata)
            
            # Step 4: Process with chunker (if text extraction successful)
            if parser_result['status'] == 'success':
                chunker_result = await self._process_chunks(parser_result['text'], metadata)
                result = {
                    'status': 'success',
                    'metadata': metadata,
                    'parser_result': parser_result,
                    'chunker_result': chunker_result
                }
            else:
                result = {
                    'status': 'failed',
                    'reason': f"Parser failed: {parser_result['error']}",
                    'metadata': metadata,
                    'parser_result': parser_result
                }
            
            # Step 5: Store processing result
            self.processed_documents[metadata['doc_hash']] = result
            
            return result
            
        except DocumentValidationError as e:
            logger.error(f"Document validation failed: {e}")
            return {
                'status': 'failed',
                'reason': f"Validation error: {e}",
                'url': url
            }
        except DocumentLoadError as e:
            logger.error(f"Document loading failed: {e}")
            return {
                'status': 'failed',
                'reason': f"Loading error: {e}",
                'url': url
            }
        except Exception as e:
            logger.error(f"Unexpected error processing document: {e}")
            return {
                'status': 'failed',
                'reason': f"Unexpected error: {e}",
                'url': url
            }
    
    async def process_multiple_documents(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents concurrently.
        
        Args:
            urls: List of document URLs to process
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing {len(urls)} documents concurrently")
        
        # Create tasks for concurrent processing
        tasks = [self.process_document_url(url) for url in urls]
        
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
    
    async def _route_to_parser(self, file_path: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route document to appropriate parser based on file type and metadata.
        
        Args:
            file_path: Path to the document file
            metadata: Document metadata
            
        Returns:
            Parser result with extracted text or error
        """
        try:
            extension = metadata['extension']
            
            if extension == '.pdf':
                if metadata['requires_ocr']:
                    logger.info(f"PDF requires OCR: {metadata['filename']}")
                    # In a real implementation, you would route to OCR pipeline
                    return {
                        'status': 'success',
                        'text': f"[OCR REQUIRED] {metadata['filename']} - This document needs OCR processing",
                        'requires_ocr': True
                    }
                else:
                    logger.info(f"Processing PDF with text extraction: {metadata['filename']}")
                    # In a real implementation, you would use a PDF parser
                    return {
                        'status': 'success',
                        'text': f"[PDF TEXT] {metadata['filename']} - Extracted text would go here",
                        'requires_ocr': False
                    }
            
            elif extension == '.docx':
                logger.info(f"Processing DOCX: {metadata['filename']}")
                # In a real implementation, you would use a DOCX parser
                return {
                    'status': 'success',
                    'text': f"[DOCX TEXT] {metadata['filename']} - Extracted text would go here",
                    'requires_ocr': False
                }
            
            elif extension == '.txt':
                logger.info(f"Processing TXT: {metadata['filename']}")
                # Read text file directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return {
                    'status': 'success',
                    'text': text,
                    'requires_ocr': False
                }
            
            elif extension == '.html':
                logger.info(f"Processing HTML: {metadata['filename']}")
                # In a real implementation, you would use an HTML parser
                return {
                    'status': 'success',
                    'text': f"[HTML TEXT] {metadata['filename']} - Extracted text would go here",
                    'requires_ocr': False
                }
            
            else:
                return {
                    'status': 'failed',
                    'error': f"Unsupported file type: {extension}"
                }
                
        except Exception as e:
            logger.error(f"Parser error for {metadata['filename']}: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _process_chunks(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process extracted text through chunking.
        
        Args:
            text: Extracted text from document
            metadata: Document metadata
            
        Returns:
            Chunking result
        """
        try:
            logger.info(f"Chunking text from {metadata['filename']}")
            
            # In a real implementation, you would use the chunker module
            # For now, we'll create simple chunks
            words = text.split()
            chunk_size = 100
            chunks = []
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                chunks.append({
                    'content': chunk_text,
                    'chunk_index': len(chunks),
                    'word_count': len(chunk_words)
                })
            
            return {
                'status': 'success',
                'total_chunks': len(chunks),
                'chunks': chunks,
                'total_words': len(words)
            }
            
        except Exception as e:
            logger.error(f"Chunking error for {metadata['filename']}: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Returns:
            Processing statistics
        """
        total_documents = len(self.processed_documents)
        successful = sum(1 for r in self.processed_documents.values() if r['status'] == 'success')
        failed = sum(1 for r in self.processed_documents.values() if r['status'] == 'failed')
        skipped = sum(1 for r in self.processed_documents.values() if r['status'] == 'skipped')
        
        # Calculate total chunks
        total_chunks = 0
        for result in self.processed_documents.values():
            if result['status'] == 'success' and 'chunker_result' in result:
                chunker_result = result['chunker_result']
                if chunker_result['status'] == 'success':
                    total_chunks += chunker_result['total_chunks']
        
        return {
            'total_documents': total_documents,
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'total_chunks': total_chunks,
            'success_rate': (successful / total_documents * 100) if total_documents > 0 else 0
        }
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """
        Clean up old files from cache.
        
        Args:
            max_age_hours: Maximum age of files to keep
        """
        cleaned_count = self.loader.cleanup_cache(max_age_hours)
        logger.info(f"Cleaned up {cleaned_count} old files from cache")


async def main():
    """Example usage of the RAG pipeline with document loader."""
    
    # Initialize pipeline
    pipeline = RAGPipeline(cache_dir="./data/documents")
    
    # Example document URLs (replace with real URLs)
    document_urls = [
        "https://raw.githubusercontent.com/microsoft/vscode/main/README.md",
        "https://raw.githubusercontent.com/microsoft/vscode/main/CONTRIBUTING.md",
        "https://raw.githubusercontent.com/microsoft/vscode/main/LICENSE"
    ]
    
    print("🚀 Starting RAG Pipeline with Document Loader")
    print("=" * 60)
    
    # Process documents
    results = await pipeline.process_multiple_documents(document_urls)
    
    # Display results
    print("\n📊 Processing Results:")
    print("-" * 40)
    
    for i, result in enumerate(results):
        print(f"\nDocument {i+1}: {document_urls[i]}")
        print(f"  Status: {result['status']}")
        
        if result['status'] == 'success':
            metadata = result['metadata']
            print(f"  Filename: {metadata['filename']}")
            print(f"  File size: {metadata['file_size']} bytes")
            print(f"  Document hash: {metadata['doc_hash'][:16]}...")
            print(f"  Requires OCR: {metadata['requires_ocr']}")
            
            if 'chunker_result' in result and result['chunker_result']['status'] == 'success':
                chunker_result = result['chunker_result']
                print(f"  Chunks created: {chunker_result['total_chunks']}")
                print(f"  Total words: {chunker_result['total_words']}")
        
        elif result['status'] == 'failed':
            print(f"  Error: {result['reason']}")
        
        elif result['status'] == 'skipped':
            print(f"  Reason: {result['reason']}")
    
    # Display statistics
    stats = pipeline.get_processing_stats()
    print(f"\n📈 Processing Statistics:")
    print("-" * 40)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    
    # Cleanup old files
    pipeline.cleanup_old_files()
    
    print("\n✅ RAG Pipeline completed!")


if __name__ == "__main__":
    # Create data directory
    Path("./data/documents").mkdir(parents=True, exist_ok=True)
    
    # Run the example
    asyncio.run(main()) 