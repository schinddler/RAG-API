#!/usr/bin/env python3
"""
Enhanced Parser Integration Example

This example demonstrates the complete enhanced parser functionality
including OCR, MIME validation, HTML parsing, table preservation,
chunking, and metadata extraction in a RAG pipeline.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import json

# Import our modules
from parser import FileParser, ParserError, parse_file, parse_url
from metadata_store import MetadataStore
from loader import DocumentLoader


class EnhancedRAGPipeline:
    """
    Complete RAG pipeline using enhanced parser features.
    
    Demonstrates:
    - OCR fallback for scanned PDFs
    - MIME type validation
    - HTML parsing in EML files
    - DOCX table structure preservation
    - Intelligent text chunking
    - Comprehensive metadata extraction
    """
    
    def __init__(self, db_url: str = "sqlite:///rag_pipeline.db"):
        """
        Initialize the enhanced RAG pipeline.
        
        Args:
            db_url: Database URL for metadata storage
        """
        self.metadata_store = MetadataStore(db_url)
        self.loader = DocumentLoader()
        self.parser = FileParser(enable_ocr=True)  # Enable OCR by default
        
    async def process_document_url(self, url: str, doc_type: str = "general") -> Dict[str, Any]:
        """
        Process a document from URL through the complete pipeline.
        
        Args:
            url: URL of the document to process
            doc_type: Type of document (e.g., "legal", "financial", "insurance")
            
        Returns:
            Dictionary with processing results
        """
        try:
            print(f"🔄 Processing document from URL: {url}")
            
            # Step 1: Load document
            print("📥 Loading document...")
            file_path, load_metadata = await self.loader.load_from_url(url)
            
            # Step 2: Parse document with enhanced features
            print("🔍 Parsing document with enhanced features...")
            parse_result = await self.parser.parse_from_url(url, max_chars=1000)
            
            # Step 3: Save document metadata
            print("💾 Saving document metadata...")
            doc_metadata = {
                "filename": load_metadata.get("filename", "unknown"),
                "source_url": url,
                "doc_type": doc_type,
                "extension": load_metadata.get("extension", ""),
                "file_size": load_metadata.get("file_size", 0),
                "doc_hash": load_metadata.get("doc_hash", ""),
                "requires_ocr": parse_result.get("metadata", {}).get("ocr_used", False),
                "status": "parsed",
                "upload_time": "2024-01-01T00:00:00Z"
            }
            
            doc_id = self.metadata_store.save_document_metadata(doc_metadata)
            
            # Step 4: Save chunks if available
            chunks_created = 0
            if "text_chunks" in parse_result:
                print(f"📝 Saving {len(parse_result['text_chunks'])} text chunks...")
                chunk_metadata = []
                for i, chunk in enumerate(parse_result["text_chunks"]):
                    chunk_metadata.append({
                        "chunk_index": i,
                        "content": chunk,
                        "token_count": len(chunk.split()),
                        "page_number": None  # Could be extracted from PDF metadata
                    })
                
                self.metadata_store.save_chunks(doc_id, chunk_metadata)
                chunks_created = len(parse_result["text_chunks"])
            
            # Step 5: Update document status
            self.metadata_store.mark_doc_status(doc_id, "indexed")
            
            return {
                "doc_id": str(doc_id),
                "file_type": parse_result["filetype"],
                "text_length": parse_result["length"],
                "chunks_created": chunks_created,
                "metadata": parse_result["metadata"],
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def process_local_file(self, file_path: str, doc_type: str = "general") -> Dict[str, Any]:
        """
        Process a local file through the complete pipeline.
        
        Args:
            file_path: Path to the local file
            doc_type: Type of document
            
        Returns:
            Dictionary with processing results
        """
        try:
            print(f"🔄 Processing local file: {file_path}")
            
            # Step 1: Load document
            print("📥 Loading document...")
            file_path_loaded, load_metadata = self.loader.load_from_file(file_path)
            
            # Step 2: Parse document with enhanced features
            print("🔍 Parsing document with enhanced features...")
            parse_result = self.parser.parse_from_file(file_path_loaded, max_chars=1000)
            
            # Step 3: Save document metadata
            print("💾 Saving document metadata...")
            doc_metadata = {
                "filename": load_metadata.get("filename", "unknown"),
                "source_url": file_path,
                "doc_type": doc_type,
                "extension": load_metadata.get("extension", ""),
                "file_size": load_metadata.get("file_size", 0),
                "doc_hash": load_metadata.get("doc_hash", ""),
                "requires_ocr": parse_result.get("metadata", {}).get("ocr_used", False),
                "status": "parsed",
                "upload_time": "2024-01-01T00:00:00Z"
            }
            
            doc_id = await self.metadata_store.save_document_metadata(doc_metadata)
            
            # Step 4: Save chunks if available
            chunks_created = 0
            if "text_chunks" in parse_result:
                print(f"📝 Saving {len(parse_result['text_chunks'])} text chunks...")
                chunk_metadata = []
                for i, chunk in enumerate(parse_result["text_chunks"]):
                    chunk_metadata.append({
                        "chunk_index": i,
                        "content": chunk,
                        "token_count": len(chunk.split()),
                        "page_number": None
                    })
                
                await self.metadata_store.save_chunks(doc_id, chunk_metadata)
                chunks_created = len(parse_result["text_chunks"])
            
            # Step 5: Update document status
            await self.metadata_store.mark_doc_status(doc_id, "indexed")
            
            return {
                "doc_id": str(doc_id),
                "file_type": parse_result["filetype"],
                "text_length": parse_result["length"],
                "chunks_created": chunks_created,
                "metadata": parse_result["metadata"],
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def demonstrate_enhanced_features(self):
        """
        Demonstrate all enhanced parser features with sample data.
        """
        print("\n🎯 Enhanced Parser Feature Demonstration")
        print("=" * 50)
        
        # Create temporary directory for demo files
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Demo 1: Text file with chunking
            print("\n1️⃣ Text File with Chunking")
            print("-" * 30)
            text_content = "This is a sample document for demonstration. " * 20
            text_file = Path(temp_dir) / "demo.txt"
            text_file.write_text(text_content)
            
            result = self.parser.parse_from_file(str(text_file), max_chars=100)
            print(f"✓ File type: {result['filetype']}")
            print(f"✓ Text length: {result['length']}")
            print(f"✓ Chunks created: {len(result.get('text_chunks', []))}")
            print(f"✓ Metadata: {result['metadata']}")
            
            # Demo 2: MIME validation
            print("\n2️⃣ MIME Type Validation")
            print("-" * 30)
            is_valid = self.parser._validate_mime_type(text_file, '.txt')
            print(f"✓ MIME validation: {is_valid}")
            
            # Demo 3: OCR availability check
            print("\n3️⃣ OCR Feature Check")
            print("-" * 30)
            print(f"✓ OCR enabled: {self.parser.enable_ocr}")
            print(f"✓ OCR available: {hasattr(self.parser, '_ocr_pdf_pages')}")
            
            # Demo 4: Text chunking demonstration
            print("\n4️⃣ Text Chunking Demonstration")
            print("-" * 30)
            sample_text = "This is a sample text that will be chunked into smaller pieces. " * 5
            chunks = self.parser._chunk_text(sample_text, max_chars=80, overlap=20)
            print(f"✓ Original text length: {len(sample_text)}")
            print(f"✓ Number of chunks: {len(chunks)}")
            print(f"✓ Sample chunk: {chunks[0][:50]}...")
            
            # Demo 5: Metadata extraction
            print("\n5️⃣ Metadata Extraction")
            print("-" * 30)
            print(f"✓ Text file metadata: {result['metadata']}")
            
        except Exception as e:
            print(f"❌ Demo failed: {str(e)}")
        
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Returns:
            Dictionary with processing statistics
        """
        try:
            stats = await self.metadata_store.get_document_stats()
            return {
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "documents_by_status": stats.get("documents_by_status", {}),
                "average_chunks_per_doc": stats.get("average_chunks_per_doc", 0)
            }
        except Exception as e:
            return {"error": str(e)}


async def main():
    """Main function to demonstrate the enhanced RAG pipeline."""
    print("🚀 Enhanced RAG Pipeline Demonstration")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = EnhancedRAGPipeline()
    
    # Demonstrate enhanced features
    pipeline.demonstrate_enhanced_features()
    
    # Create a sample text file for processing
    temp_dir = tempfile.mkdtemp()
    sample_file = Path(temp_dir) / "sample.txt"
    sample_content = "This is a sample document for the enhanced RAG pipeline. " * 10
    sample_file.write_text(sample_content)
    
    try:
        # Process the sample file
        print("\n🔄 Processing Sample Document")
        print("-" * 30)
        result = await pipeline.process_local_file(str(sample_file), "demo")
        
        if result["status"] == "success":
            print(f"✅ Document processed successfully!")
            print(f"📄 Document ID: {result['doc_id']}")
            print(f"📝 File type: {result['file_type']}")
            print(f"📏 Text length: {result['text_length']}")
            print(f"📦 Chunks created: {result['chunks_created']}")
            print(f"📋 Metadata: {json.dumps(result['metadata'], indent=2)}")
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Get processing statistics
        print("\n📊 Processing Statistics")
        print("-" * 30)
        stats = await pipeline.get_processing_stats()
        print(f"📈 Total documents: {stats.get('total_documents', 0)}")
        print(f"📊 Total chunks: {stats.get('total_chunks', 0)}")
        print(f"📋 Documents by status: {stats.get('documents_by_status', {})}")
        
    except Exception as e:
        print(f"❌ Error in main demonstration: {str(e)}")
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n🎉 Enhanced RAG Pipeline demonstration completed!")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 