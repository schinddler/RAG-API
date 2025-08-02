# Document Ingestion Module

This module provides a robust, production-grade document loading system for the RAG backend, designed to handle large legal, financial, and insurance documents with comprehensive validation, security, and metadata capture.

## Features

### 🚀 Multi-Format Support
- **PDF** (.pdf) - With OCR detection for scanned documents
- **Word Documents** (.docx) - Microsoft Word files
- **Text Files** (.txt) - Plain text documents
- **Email Files** (.eml) - Email messages
- **HTML Files** (.html, .htm) - Web pages and HTML documents
- **ZIP Archives** (.zip) - Extracts and processes all supported files within

### 🌐 Download Capabilities
- **HTTP/HTTPS** - Direct web downloads with retry logic
- **S3 URLs** - Amazon S3 bucket access (requires boto3 setup)
- **GCS URLs** - Google Cloud Storage access (requires google-cloud-storage setup)
- **Local Files** - File system access with `file://` protocol
- **Automatic Retry** - Configurable retry attempts with exponential backoff

### 🔒 Security & Validation
- **File Type Validation** - MIME type and extension verification
- **Dangerous File Rejection** - Blocks executable and script files
- **Filename Sanitization** - Prevents path traversal attacks
- **Size Limits** - Configurable maximum file size (default: 50MB)
- **Content Type Validation** - Verifies Content-Type headers

### 📊 Metadata Capture
- **Document Hash** - SHA-256 for deduplication
- **File Information** - Size, extension, upload time
- **Source Tracking** - URL or file path origin
- **OCR Detection** - Automatic detection of scanned PDFs
- **Processing Status** - Success/failure tracking

### 🔄 Deduplication
- **Hash-based Detection** - Prevents re-processing identical documents
- **Cache Management** - Automatic cleanup of old files
- **Duplicate Checking** - API to check if document already exists

### ⚡ Performance Features
- **Async Support** - Non-blocking operations for web servers
- **Thread Pool** - Efficient concurrent processing
- **Streaming Downloads** - Memory-efficient large file handling
- **Caching** - Local file storage with UUID-based naming

## Quick Start

### Basic Usage

```python
from ingestion.loader import DocumentLoader

# Initialize loader
loader = DocumentLoader(cache_dir="/tmp/docs", max_file_size_mb=50)

# Load from URL
file_path, metadata = loader.load_from_url("https://example.com/document.pdf")

# Load from local file
file_path, metadata = loader.load_from_file("/path/to/document.docx")

# Load from ZIP archive (returns list of files)
extracted_files = loader.load_from_file("/path/to/documents.zip")
for file_path, metadata in extracted_files:
    print(f"Extracted: {metadata['filename']}")
```

### Async Usage

```python
import asyncio
from ingestion.loader import DocumentLoader

async def load_documents():
    loader = DocumentLoader(enable_async=True)
    
    # Async URL download
    file_path, metadata = await loader.load_from_url_async("https://example.com/doc.pdf")
    
    # Process multiple URLs concurrently
    urls = ["https://example.com/doc1.pdf", "https://example.com/doc2.docx"]
    tasks = [loader.load_from_url_async(url) for url in urls]
    results = await asyncio.gather(*tasks)

# Run async function
asyncio.run(load_documents())
```

### Convenience Functions

```python
from ingestion.loader import load_document, load_document_async

# Automatic URL/file detection
file_path, metadata = load_document("https://example.com/document.pdf")
file_path, metadata = load_document("/local/path/document.docx")

# Async convenience
file_path, metadata = await load_document_async("https://example.com/document.pdf")
```

## Configuration

### DocumentLoader Parameters

```python
loader = DocumentLoader(
    cache_dir="/tmp/docs",           # Cache directory for downloaded files
    max_file_size_mb=50,            # Maximum file size in MB
    timeout_seconds=30,             # Download timeout
    max_retries=3,                  # Retry attempts for failed downloads
    enable_async=True               # Enable async capabilities
)
```

### Supported File Extensions

```python
SUPPORTED_EXTENSIONS = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain',
    '.eml': 'message/rfc822',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.zip': 'application/zip'
}
```

### Rejected File Extensions

```python
DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar',
    '.msi', '.dmg', '.app', '.sh', '.py', '.php', '.asp', '.aspx', '.jsp'
}
```

## Metadata Structure

Each loaded document returns comprehensive metadata:

```python
metadata = {
    'filename': 'document.pdf',                    # Original filename
    'source_url': 'https://example.com/doc.pdf',  # Source URL or path
    'doc_type': 'downloaded',                     # Type: downloaded, local_file, extracted_from_zip
    'extension': '.pdf',                          # File extension
    'file_size': 1024000,                         # File size in bytes
    'upload_time': '2024-01-15T10:30:00',        # ISO timestamp
    'doc_hash': 'abc123...',                      # SHA-256 hash
    'requires_ocr': False,                        # OCR requirement flag
    'content_type': 'application/pdf',            # MIME type (for downloads)
    'extracted_path': '/path/to/file'             # Path for ZIP extractions
}
```

## Error Handling

The loader provides comprehensive error handling:

```python
from ingestion.loader import DocumentLoadError, DocumentValidationError

try:
    file_path, metadata = loader.load_from_url("https://example.com/document.pdf")
except DocumentValidationError as e:
    print(f"Validation error: {e}")  # File type, size, or security issues
except DocumentLoadError as e:
    print(f"Loading error: {e}")     # Download, extraction, or processing issues
```

## OCR Detection

The loader automatically detects if PDFs require OCR:

```python
file_path, metadata = loader.load_from_file("scanned_document.pdf")

if metadata['requires_ocr']:
    print("This PDF needs OCR processing")
    # Route to OCR pipeline
else:
    print("This PDF has extractable text")
    # Route to text extraction pipeline
```

## Cache Management

```python
# Check for duplicates
is_duplicate = loader.check_duplicate(doc_hash)

# Clean up old files (older than 24 hours)
cleaned_count = loader.cleanup_cache(max_age_hours=24)
print(f"Cleaned up {cleaned_count} old files")
```

## Integration with RAG Pipeline

The loader is designed to integrate seamlessly with the RAG pipeline:

```python
# 1. Load document
file_path, metadata = loader.load_from_url(url)

# 2. Check for duplicates
if loader.check_duplicate(metadata['doc_hash']):
    print("Document already processed")
    return

# 3. Route to appropriate parser based on metadata
if metadata['extension'] == '.pdf' and metadata['requires_ocr']:
    # Route to OCR pipeline
    pass
elif metadata['extension'] == '.pdf':
    # Route to PDF parser
    pass
elif metadata['extension'] == '.docx':
    # Route to DOCX parser
    pass

# 4. Store metadata in database
# store_metadata(metadata)
```

## Testing

Run the comprehensive test suite:

```bash
python ingestion/test_loader.py
```

The test suite covers:
- Local file loading
- URL downloads
- ZIP extraction
- Security validation
- Async operations
- Convenience functions

## Dependencies

Required packages (already included in requirements.txt):
- `PyMuPDF` - PDF processing and OCR detection
- `pdfminer.six` - PDF text extraction
- `aiohttp` - Async HTTP client
- `aiofiles` - Async file operations
- `requests` - HTTP client
- `pathlib` - Path operations

## Security Considerations

1. **File Type Validation**: Only allows safe document formats
2. **Filename Sanitization**: Prevents path traversal attacks
3. **Size Limits**: Prevents DoS attacks via large files
4. **Content Type Verification**: Validates MIME types
5. **Isolated Cache**: Files stored in controlled directory

## Performance Tips

1. **Use Async**: For web applications, use async methods
2. **Configure Cache**: Set appropriate cache directory and cleanup
3. **Batch Processing**: Use `asyncio.gather()` for multiple downloads
4. **Monitor Size**: Adjust `max_file_size_mb` based on your needs
5. **Regular Cleanup**: Schedule cache cleanup to prevent disk space issues

## Troubleshooting

### Common Issues

1. **File Not Found**: Check URL accessibility and file existence
2. **Permission Denied**: Ensure cache directory is writable
3. **Timeout Errors**: Increase `timeout_seconds` for slow connections
4. **Memory Issues**: Reduce `max_file_size_mb` or use streaming
5. **Duplicate Files**: Use `check_duplicate()` before processing

### Logging

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] S3 and GCS integration with proper authentication
- [ ] Additional file format support (RTF, ODT, etc.)
- [ ] Parallel processing for large ZIP files
- [ ] Progress callbacks for long operations
- [ ] Integration with cloud storage providers
- [ ] Advanced OCR pipeline integration 