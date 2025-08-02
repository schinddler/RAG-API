# Metadata Store Module

A robust, production-grade metadata storage system for RAG backends that tracks every document, chunk, and transformation in the pipeline. Built with SQLAlchemy ORM, supporting PostgreSQL and SQLite with ACID compliance, audit logging, and comprehensive querying capabilities.

## 🚀 Features

### ✅ Core Capabilities
- **Document Tracking**: Complete metadata for every uploaded document
- **Chunk Management**: Track all document chunks with embeddings and quality scores
- **Audit Logging**: Full audit trail for debugging and compliance
- **Deduplication**: SHA-256 hash-based duplicate detection
- **Status Tracking**: Document processing status from upload to indexing
- **Statistics**: Comprehensive pipeline statistics and analytics

### ✅ Database Support
- **PostgreSQL**: Production-ready with async support
- **SQLite**: Development and testing with in-memory option
- **ACID Compliance**: All operations are atomic and consistent
- **Connection Pooling**: Optimized for high-concurrency workloads

### ✅ Advanced Features
- **Versioning**: Document version tracking for updates
- **User Tracking**: Associate documents with users
- **Quality Scoring**: Chunk quality and suspicious content detection
- **Performance Metrics**: Processing time and success rate tracking
- **Cleanup Utilities**: Automatic log cleanup and maintenance

## 📊 Database Schema

### Documents Table
| Field           | Type         | Description                                |
|----------------|--------------|--------------------------------------------|
| `doc_id`        | UUID (PK)     | Unique document identifier                 |
| `filename`      | TEXT          | Sanitized original filename               |
| `source_url`    | TEXT          | Optional: where file was loaded from     |
| `doc_type`      | TEXT          | Document classification (legal, insurance, etc.) |
| `extension`     | TEXT          | File extension (.pdf, .docx, etc.)       |
| `file_size`     | INT           | File size in bytes                        |
| `doc_hash`      | TEXT          | SHA-256 hash for deduplication           |
| `requires_ocr`  | BOOLEAN       | Flag for scanned documents                |
| `upload_time`   | TIMESTAMP     | UTC upload timestamp                      |
| `status`        | ENUM          | Processing status (uploaded, parsed, indexed, error) |
| `error_msg`     | TEXT          | Error message if processing failed        |
| `language`      | TEXT          | Document language code                     |
| `source_type`   | TEXT          | Source type (upload, url, s3, etc.)      |
| `user_id`       | TEXT          | Associated user ID                        |
| `version`       | INT           | Document version number                   |
| `created_at`    | TIMESTAMP     | Record creation time                      |
| `updated_at`    | TIMESTAMP     | Record last update time                   |

### Chunks Table
| Field           | Type         | Description                                |
|----------------|--------------|--------------------------------------------|
| `chunk_id`      | UUID (PK)     | Unique chunk identifier                   |
| `doc_id`        | UUID (FK)     | Foreign key to documents                  |
| `chunk_index`   | INT           | Position in document                      |
| `chunk_text`    | TEXT          | Raw text content                          |
| `embedding_id`  | TEXT          | External embedding identifier             |
| `token_count`   | INT           | Token count for pricing/analysis         |
| `page_number`   | INT           | Page number if known                      |
| `section_title` | TEXT          | Section or heading title                  |
| `quality_score` | INT           | Quality score (0-100)                     |
| `is_suspicious` | BOOLEAN       | Flag for suspicious content               |
| `created_at`    | TIMESTAMP     | Record creation time                      |

### Ingestion Log Table
| Field               | Type         | Description                                |
|--------------------|--------------|--------------------------------------------|
| `log_id`           | UUID (PK)     | Unique log entry identifier               |
| `doc_id`           | UUID (FK)     | Associated document (optional)            |
| `step`             | TEXT          | Processing step (load, parse, chunk, etc.) |
| `status`           | TEXT          | Event status (started, success, error)   |
| `message`          | TEXT          | Human-readable message                    |
| `metadata_json`    | TEXT          | Additional metadata (JSON)                |
| `timestamp`        | TIMESTAMP     | Event timestamp                           |
| `processing_time_ms`| INT          | Processing time in milliseconds           |

## 🛠️ Installation

```bash
pip install sqlalchemy asyncpg
```

## 📖 Usage

### Basic Setup

```python
import asyncio
from metadata_store import create_metadata_store, DocumentStatus

async def main():
    # Initialize metadata store
    store = await create_metadata_store("postgresql://user:pass@localhost/rag_db")
    
    # Your RAG pipeline code here...
    
    # Cleanup
    await store.close()

asyncio.run(main())
```

### Document Processing Workflow

```python
# 1. Save document metadata
doc_metadata = {
    "filename": "contract.pdf",
    "source_url": "https://example.com/contract.pdf",
    "extension": ".pdf",
    "file_size": 1024000,
    "doc_type": "legal",
    "upload_time": "2025-07-31T14:55:00Z",
    "doc_hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
    "requires_ocr": False,
    "user_id": "user123"
}

doc_id = await store.save_document_metadata(doc_metadata)

# 2. Update processing status
await store.mark_doc_status(doc_id, DocumentStatus.PARSED)

# 3. Save chunks
chunks = [
    {
        "chunk_index": 0,
        "content": "This is the first chunk...",
        "token_count": 150,
        "page_number": 1,
        "section_title": "Introduction"
    }
]

chunk_ids = await store.save_chunks(doc_id, chunks)

# 4. Update chunk embeddings
for chunk_id in chunk_ids:
    await store.update_chunk_embedding(chunk_id, f"emb_{chunk_id[:8]}")

# 5. Mark as indexed
await store.mark_doc_status(doc_id, DocumentStatus.INDEXED)
```

### Deduplication

```python
# Check if document already exists
existing_doc = await store.get_doc_by_hash(doc_hash)
if existing_doc:
    print(f"Document already processed: {existing_doc['filename']}")
    # Skip processing or handle accordingly
```

### Querying and Analytics

```python
# Get documents by status
indexed_docs = await store.get_documents_by_status(DocumentStatus.INDEXED)

# Get document statistics
stats = await store.get_document_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Success rate: {stats['success_rate']:.1f}%")

# Get audit trail
audit_log = await store.get_audit_log(doc_id=doc_id)
for entry in audit_log:
    print(f"{entry['timestamp']} - {entry['step']} - {entry['status']}")
```

## 🔧 Configuration

### Database URLs

```python
# PostgreSQL
db_url = "postgresql://user:password@localhost:5432/rag_db"

# SQLite (file)
db_url = "sqlite:///./data/metadata.db"

# SQLite (memory)
db_url = "sqlite:///:memory:"
```

### MetadataStore Options

```python
store = await create_metadata_store(
    db_url="postgresql://user:pass@localhost/rag_db",
    enable_audit_logging=True,    # Enable detailed audit logging
    max_retries=3,                # Database operation retries
    pool_size=10                  # Connection pool size
)
```

## 🔄 Integration with Document Loader

```python
from loader import DocumentLoader
from metadata_store import create_metadata_store

class RAGPipeline:
    def __init__(self):
        self.loader = DocumentLoader()
        self.metadata_store = None
    
    async def setup(self):
        self.metadata_store = await create_metadata_store("sqlite:///:memory:")
    
    async def process_document(self, url: str):
        # Load document
        file_path, metadata = await self.loader.load_from_url_async(url)
        
        # Check for duplicates
        existing = await self.metadata_store.get_doc_by_hash(metadata['doc_hash'])
        if existing:
            return {'status': 'skipped', 'reason': 'Already processed'}
        
        # Save metadata
        doc_id = await self.metadata_store.save_document_metadata(metadata)
        
        # Process through pipeline...
        await self.metadata_store.mark_doc_status(doc_id, DocumentStatus.INDEXED)
        
        return {'status': 'success', 'doc_id': str(doc_id)}
```

## 📈 Monitoring and Analytics

### Pipeline Statistics

```python
stats = await store.get_document_stats()

# Available metrics:
# - total_documents: Total documents in database
# - total_chunks: Total chunks across all documents
# - avg_chunks_per_doc: Average chunks per document
# - status_breakdown: Documents by processing status
```

### Audit Trail Analysis

```python
# Get recent activity
recent_logs = await store.get_audit_log(limit=100)

# Filter by step
parse_logs = await store.get_audit_log(step="parse")

# Filter by document
doc_logs = await store.get_audit_log(doc_id=doc_id)
```

### Performance Monitoring

```python
# Track processing times
for entry in audit_log:
    if entry['processing_time_ms']:
        print(f"Step {entry['step']}: {entry['processing_time_ms']}ms")

# Calculate success rates
success_count = len([e for e in audit_log if e['status'] == 'success'])
total_count = len(audit_log)
success_rate = (success_count / total_count) * 100
```

## 🧹 Maintenance

### Log Cleanup

```python
# Clean up old audit logs (keep last 30 days)
await store.cleanup_old_logs(days_to_keep=30)
```

### Database Maintenance

```python
# Close connections properly
await store.close()

# For PostgreSQL, consider periodic VACUUM
# For SQLite, consider periodic OPTIMIZE
```

## 🔒 Security Considerations

### Input Validation
- All user inputs are sanitized
- File paths are validated to prevent traversal attacks
- SQL injection protection through parameterized queries

### Access Control
- User ID tracking for multi-tenant environments
- Audit logging for compliance requirements
- Sensitive data handling best practices

## 🚨 Error Handling

```python
from metadata_store import MetadataStoreError, DuplicateDocumentError

try:
    doc_id = await store.save_document_metadata(metadata)
except DuplicateDocumentError:
    print("Document already exists")
except MetadataStoreError as e:
    print(f"Database error: {e}")
```

## 📋 Best Practices

### 1. Connection Management
```python
# Always close connections
async with store:  # If context manager is implemented
    # Your operations here
    pass
```

### 2. Batch Operations
```python
# For multiple documents, use batch processing
for doc_metadata in documents:
    await store.save_document_metadata(doc_metadata)
```

### 3. Error Recovery
```python
# Implement retry logic for transient failures
for attempt in range(3):
    try:
        await store.save_document_metadata(metadata)
        break
    except MetadataStoreError as e:
        if attempt == 2:
            raise
        await asyncio.sleep(0.1 * (attempt + 1))
```

### 4. Monitoring
```python
# Regular health checks
stats = await store.get_document_stats()
if stats['total_documents'] == 0:
    logger.warning("No documents in database")

# Monitor error rates
error_logs = await store.get_audit_log(step="error")
if len(error_logs) > threshold:
    alert_admin()
```

## 🔧 Development

### Running Tests

```bash
python ingestion/test_metadata_store.py
```

### Example Integration

```bash
python ingestion/example_metadata_integration.py
```

## 📚 API Reference

### Core Methods

- `save_document_metadata(metadata)`: Save document metadata
- `get_doc_by_hash(doc_hash)`: Get document by hash
- `get_document_by_id(doc_id)`: Get document by UUID
- `save_chunks(doc_id, chunks)`: Save chunk metadata
- `get_chunks_by_doc(doc_id)`: Get chunks for document
- `mark_doc_status(doc_id, status)`: Update document status
- `update_chunk_embedding(chunk_id, embedding_id)`: Update chunk embedding
- `get_documents_by_status(status)`: Get documents by status
- `get_document_stats()`: Get processing statistics
- `get_audit_log()`: Get audit trail
- `cleanup_old_logs(days_to_keep)`: Clean up old logs
- `close()`: Close database connections

### Enums

- `DocumentStatus`: UPLOADED, PARSED, CHUNKED, EMBEDDED, INDEXED, ERROR
- `DocumentType`: LEGAL, INSURANCE, FINANCIAL, CONTRACT, POLICY, REPORT, MANUAL, OTHER

### Exceptions

- `MetadataStoreError`: Base exception for metadata store errors
- `DocumentNotFoundError`: Document not found
- `DuplicateDocumentError`: Document already exists

## 🤝 Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## 📄 License

This module is part of the RAG-API project and follows the same license terms. 