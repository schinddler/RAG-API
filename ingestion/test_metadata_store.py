"""
Test script for the Metadata Store

This script demonstrates the various capabilities of the MetadataStore class
including document storage, chunk management, audit logging, and statistics.
"""

import asyncio
import logging
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from metadata_store import (
    MetadataStore, 
    DocumentStatus, 
    DocumentType, 
    MetadataStoreError,
    DuplicateDocumentError
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_sqlite_metadata_store():
    """Test metadata store with SQLite (for development)."""
    print("\n=== Testing SQLite Metadata Store ===")
    
    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        # Initialize metadata store with SQLite
        store = MetadataStore(f"sqlite:///{db_path}")
        
        # Test document metadata storage
        doc_metadata = {
            "filename": "test_document.pdf",
            "source_url": "https://example.com/test.pdf",
            "extension": ".pdf",
            "file_size": 1024000,
            "doc_type": "legal",
            "upload_time": datetime.utcnow().isoformat(),
            "doc_hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
            "requires_ocr": False,
            "language": "en",
            "source_type": "url",
            "user_id": "test_user_123"
        }
        
        # Save document metadata
        doc_id = await store.save_document_metadata(doc_metadata)
        print(f"✅ Saved document with ID: {doc_id}")
        
        # Test duplicate detection
        try:
            await store.save_document_metadata(doc_metadata)
            print("❌ Should have raised DuplicateDocumentError")
        except DuplicateDocumentError:
            print("✅ Correctly detected duplicate document")
        
        # Test getting document by hash
        retrieved_doc = await store.get_doc_by_hash(doc_metadata['doc_hash'])
        if retrieved_doc:
            print(f"✅ Retrieved document: {retrieved_doc['filename']}")
        else:
            print("❌ Failed to retrieve document by hash")
        
        # Test getting document by ID
        retrieved_doc = await store.get_document_by_id(doc_id)
        if retrieved_doc:
            print(f"✅ Retrieved document by ID: {retrieved_doc['filename']}")
        else:
            print("❌ Failed to retrieve document by ID")
        
        # Test status updates
        await store.mark_doc_status(doc_id, DocumentStatus.PARSED)
        await store.mark_doc_status(doc_id, DocumentStatus.CHUNKED)
        await store.mark_doc_status(doc_id, DocumentStatus.INDEXED)
        print("✅ Updated document status successfully")
        
        # Test chunk storage
        chunks = [
            {
                "chunk_index": 0,
                "content": "This is the first chunk of the legal document. It contains important information about the terms and conditions.",
                "token_count": 25,
                "page_number": 1,
                "section_title": "Introduction",
                "quality_score": 95,
                "is_suspicious": False
            },
            {
                "chunk_index": 1,
                "content": "This is the second chunk containing more detailed legal information and specific clauses that need to be understood carefully.",
                "token_count": 30,
                "page_number": 1,
                "section_title": "Terms and Conditions",
                "quality_score": 88,
                "is_suspicious": False
            },
            {
                "chunk_index": 2,
                "content": "The third chunk includes liability clauses and indemnification terms that are critical for legal compliance.",
                "token_count": 28,
                "page_number": 2,
                "section_title": "Liability Clauses",
                "quality_score": 92,
                "is_suspicious": False
            }
        ]
        
        chunk_ids = await store.save_chunks(doc_id, chunks)
        print(f"✅ Saved {len(chunk_ids)} chunks")
        
        # Test retrieving chunks
        retrieved_chunks = await store.get_chunks_by_doc(doc_id)
        print(f"✅ Retrieved {len(retrieved_chunks)} chunks")
        
        # Test document statistics
        stats = await store.get_document_stats()
        print(f"✅ Document stats: {stats}")
        
        # Test audit log
        audit_log = await store.get_audit_log(doc_id=doc_id)
        print(f"✅ Retrieved {len(audit_log)} audit log entries")
        
        # Test getting documents by status
        indexed_docs = await store.get_documents_by_status(DocumentStatus.INDEXED)
        print(f"✅ Found {len(indexed_docs)} indexed documents")
        
        # Close connections
        await store.close()
        
    except Exception as e:
        print(f"❌ Error testing SQLite metadata store: {e}")
        raise
    
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
            print("🧹 Cleaned up SQLite database")

async def test_multiple_documents():
    """Test handling multiple documents."""
    print("\n=== Testing Multiple Documents ===")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        store = MetadataStore(f"sqlite:///{db_path}")
        
        # Create multiple documents
        documents = [
            {
                "filename": "contract_2024.pdf",
                "source_url": "https://example.com/contracts/2024.pdf",
                "extension": ".pdf",
                "file_size": 2048000,
                "doc_type": "contract",
                "upload_time": datetime.utcnow().isoformat(),
                "doc_hash": "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456a1",
                "requires_ocr": False
            },
            {
                "filename": "insurance_policy.docx",
                "source_url": "https://example.com/policies/insurance.docx",
                "extension": ".docx",
                "file_size": 1536000,
                "doc_type": "insurance",
                "upload_time": datetime.utcnow().isoformat(),
                "doc_hash": "c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456a1b2",
                "requires_ocr": False
            },
            {
                "filename": "financial_report.pdf",
                "source_url": "https://example.com/reports/financial.pdf",
                "extension": ".pdf",
                "file_size": 3072000,
                "doc_type": "financial",
                "upload_time": datetime.utcnow().isoformat(),
                "doc_hash": "d4e5f6789012345678901234567890abcdef1234567890abcdef123456a1b2c3",
                "requires_ocr": True
            }
        ]
        
        doc_ids = []
        for doc_metadata in documents:
            doc_id = await store.save_document_metadata(doc_metadata)
            doc_ids.append(doc_id)
            print(f"✅ Saved document: {doc_metadata['filename']} -> {doc_id}")
        
        # Update statuses
        for i, doc_id in enumerate(doc_ids):
            status = DocumentStatus.INDEXED if i < 2 else DocumentStatus.ERROR
            await store.mark_doc_status(doc_id, status)
            print(f"✅ Updated {documents[i]['filename']} status to {status.value}")
        
        # Get statistics
        stats = await store.get_document_stats()
        print(f"✅ Final stats: {stats}")
        
        await store.close()
        
    except Exception as e:
        print(f"❌ Error testing multiple documents: {e}")
        raise
    
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

async def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        store = MetadataStore(f"sqlite:///{db_path}")
        
        # Test invalid metadata
        invalid_metadata = {
            "filename": "test.pdf",
            "extension": ".pdf",
            "file_size": 1000,
            # Missing required fields
        }
        
        try:
            await store.save_document_metadata(invalid_metadata)
            print("❌ Should have failed with invalid metadata")
        except Exception as e:
            print(f"✅ Correctly handled invalid metadata: {type(e).__name__}")
        
        # Test non-existent document retrieval
        from uuid import UUID
        non_existent_doc = await store.get_document_by_id(UUID("00000000-0000-0000-0000-000000000000"))
        if non_existent_doc is None:
            print("✅ Correctly handled non-existent document")
        else:
            print("❌ Should have returned None for non-existent document")
        
        # Test non-existent chunks
        chunks = await store.get_chunks_by_doc(UUID("00000000-0000-0000-0000-000000000000"))
        if len(chunks) == 0:
            print("✅ Correctly handled non-existent chunks")
        else:
            print("❌ Should have returned empty list for non-existent chunks")
        
        await store.close()
        
    except Exception as e:
        print(f"❌ Error testing error handling: {e}")
        raise
    
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

async def test_audit_logging():
    """Test comprehensive audit logging."""
    print("\n=== Testing Audit Logging ===")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        store = MetadataStore(f"sqlite:///{db_path}", enable_audit_logging=True)
        
        # Create a document
        doc_metadata = {
            "filename": "audit_test.pdf",
            "extension": ".pdf",
            "file_size": 500000,
            "doc_type": "test",
            "upload_time": datetime.utcnow().isoformat(),
            "doc_hash": "e5f6789012345678901234567890abcdef1234567890abcdef123456a1b2c3d4",
            "requires_ocr": False
        }
        
        doc_id = await store.save_document_metadata(doc_metadata)
        
        # Perform various operations to generate audit logs
        await store.mark_doc_status(doc_id, DocumentStatus.PARSED)
        await store.mark_doc_status(doc_id, DocumentStatus.CHUNKED)
        await store.mark_doc_status(doc_id, DocumentStatus.EMBEDDED)
        await store.mark_doc_status(doc_id, DocumentStatus.INDEXED)
        
        # Save chunks
        chunks = [
            {
                "chunk_index": 0,
                "content": "Audit test chunk 1",
                "token_count": 10
            },
            {
                "chunk_index": 1,
                "content": "Audit test chunk 2",
                "token_count": 12
            }
        ]
        
        await store.save_chunks(doc_id, chunks)
        
        # Get audit log
        audit_log = await store.get_audit_log(doc_id=doc_id)
        print(f"✅ Generated {len(audit_log)} audit log entries")
        
        # Test filtering by step
        document_save_logs = await store.get_audit_log(doc_id=doc_id, step="document_save")
        print(f"✅ Found {len(document_save_logs)} document save log entries")
        
        # Test cleanup
        await store.cleanup_old_logs(days_to_keep=0)  # Clean all logs
        remaining_logs = await store.get_audit_log(doc_id=doc_id)
        print(f"✅ Remaining logs after cleanup: {len(remaining_logs)}")
        
        await store.close()
        
    except Exception as e:
        print(f"❌ Error testing audit logging: {e}")
        raise
    
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

async def test_performance():
    """Test performance with larger datasets."""
    print("\n=== Testing Performance ===")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        store = MetadataStore(f"sqlite:///{db_path}")
        
        import time
        start_time = time.time()
        
        # Create multiple documents quickly
        for i in range(10):
            doc_metadata = {
                "filename": f"performance_test_{i}.pdf",
                "extension": ".pdf",
                "file_size": 1000000 + i * 100000,
                "doc_type": "test",
                "upload_time": datetime.utcnow().isoformat(),
                "doc_hash": f"perf{i:02d}6789012345678901234567890abcdef1234567890abcdef123456a1b2c3d4",
                "requires_ocr": False
            }
            
            doc_id = await store.save_document_metadata(doc_metadata)
            
            # Add chunks for each document
            chunks = []
            for j in range(5):
                chunks.append({
                    "chunk_index": j,
                    "content": f"Performance test chunk {j} for document {i}",
                    "token_count": 20 + j
                })
            
            await store.save_chunks(doc_id, chunks)
            await store.mark_doc_status(doc_id, DocumentStatus.INDEXED)
        
        end_time = time.time()
        print(f"✅ Processed 10 documents with 50 total chunks in {end_time - start_time:.2f} seconds")
        
        # Test statistics performance
        stats = await store.get_document_stats()
        print(f"✅ Generated statistics: {stats}")
        
        await store.close()
        
    except Exception as e:
        print(f"❌ Error testing performance: {e}")
        raise
    
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

async def main():
    """Run all metadata store tests."""
    print("🚀 Starting Metadata Store Tests")
    print("=" * 60)
    
    try:
        # Run all tests
        await test_sqlite_metadata_store()
        await test_multiple_documents()
        await test_error_handling()
        await test_audit_logging()
        await test_performance()
        
        print("\n" + "=" * 60)
        print("✅ All metadata store tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 