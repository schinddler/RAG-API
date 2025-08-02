# Enhanced Parser Module

A production-grade file parser for RAG backend with advanced features including OCR fallback, MIME validation, HTML parsing, table preservation, and intelligent chunking.

## Features

### 🔍 **Multi-Format Support**
- **PDF**: Text extraction with PyMuPDF, OCR fallback for scanned documents
- **DOCX**: Full text extraction with table structure preservation
- **TXT**: UTF-8 and fallback encoding support
- **EML**: Email parsing with HTML content conversion

### 🤖 **OCR Fallback for Scanned PDFs**
- Automatic detection of scanned PDFs (no extractable text)
- Tesseract OCR integration for image-based text extraction
- Configurable OCR enablement via `enable_ocr` parameter
- High-quality image processing with 2x zoom for better accuracy

### 🔒 **MIME Type Validation**
- `python-magic` integration for accurate file type detection
- Fallback to `mimetypes` module if magic is unavailable
- Prevents spoofed file extensions and security risks
- Graceful error handling with warnings for mismatches

### 📧 **Enhanced EML Parsing**
- Parses both `text/plain` and `text/html` parts
- HTML to plain text conversion using `html2text`
- Merges content in fallback order: plain → HTML
- Extracts email metadata (subject, from, to, date, attachments)

### 📊 **DOCX Table Structure Preservation**
- Detects and preserves table layouts
- Uses delimiters (`|`) to maintain table structure in text
- Stores structured table data in metadata
- Identifies headers and row data separately

### ✂️ **Intelligent Text Chunking**
- Optional text chunking with configurable size limits
- Smart word boundary detection to avoid cutting words
- Configurable overlap between chunks (default: 100 characters)
- Returns both full text and chunked segments

### 📋 **Comprehensive Metadata Extraction**
- **PDF**: Title, author, subject, creator, producer, creation/modification dates, page count
- **DOCX**: Paragraph count, table count, section count, structured table data
- **TXT**: Encoding, line count, word count
- **EML**: Subject, sender, recipient, date, content type, attachment flags

## Installation

### Core Dependencies
```bash
pip install PyMuPDF python-docx aiofiles aiohttp
```

### Enhanced Features (Optional)
```bash
# For OCR support
pip install pytesseract Pillow

# For MIME validation
pip install python-magic

# For HTML parsing in EML
pip install html2text
```

### System Dependencies
For OCR functionality, install Tesseract:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## Usage

### Basic Usage

```python
from parser import FileParser, parse_file, parse_url

# Initialize parser
parser = FileParser(enable_ocr=True)

# Parse local file
result = parser.parse_from_file("document.pdf", max_chars=1000)

# Parse from URL
result = await parser.parse_from_url("https://example.com/document.pdf", max_chars=1000)

# Use convenience functions
result = parse_file("document.docx", max_chars=500, enable_ocr=False)
result = await parse_url("https://example.com/document.txt", max_chars=200)
```

### Output Format

```python
{
    "filetype": "pdf",                    # File extension without dot
    "url": "https://example.com/doc.pdf", # Source URL or file path
    "full_text": "Extracted text...",     # Complete extracted text
    "text_chunks": [                      # Optional: chunked text segments
        "Chunk 1 content...",
        "Chunk 2 content..."
    ],
    "length": 1234,                       # Character count
    "metadata": {                         # File-specific metadata
        "title": "Document Title",
        "author": "Author Name",
        "page_count": 5,
        "ocr_used": True,                 # If OCR was used
        "encoding": "utf-8",              # For text files
        "tables": [...]                   # For DOCX files
    }
}
```

### Advanced Configuration

```python
# Initialize with custom settings
parser = FileParser(
    temp_dir="/custom/temp/dir",  # Custom temp directory
    enable_ocr=True               # Enable OCR for scanned PDFs
)

# Parse with chunking
result = parser.parse_from_file(
    "document.pdf",
    max_chars=1000,              # Maximum characters per chunk
    # overlap=100                 # Default overlap between chunks
)
```

## API Reference

### FileParser Class

#### Constructor
```python
FileParser(temp_dir: Optional[str] = None, enable_ocr: bool = False)
```

**Parameters:**
- `temp_dir`: Directory for temporary files (defaults to system temp)
- `enable_ocr`: Enable OCR fallback for scanned PDFs

#### Methods

##### `parse_from_file(file_path: str, max_chars: Optional[int] = None) -> Dict[str, Any]`
Parse a local file.

**Parameters:**
- `file_path`: Path to the file to parse
- `max_chars`: Optional chunking size for text output

**Returns:** Dictionary with parsed text and metadata

##### `parse_from_url(url: str, max_chars: Optional[int] = None) -> Dict[str, Any]`
Parse a file from URL.

**Parameters:**
- `url`: The URL pointing to the file to parse
- `max_chars`: Optional chunking size for text output

**Returns:** Dictionary with parsed text and metadata

### Convenience Functions

#### `parse_file(file_path: str, max_chars: Optional[int] = None, enable_ocr: bool = False) -> Dict[str, Any]`
Parse a local file with default settings.

#### `parse_url(url: str, max_chars: Optional[int] = None, enable_ocr: bool = False) -> Dict[str, Any]`
Parse a file from URL with default settings.

## Supported File Types

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | `.pdf` | Text extraction, OCR fallback, metadata extraction |
| DOCX | `.docx` | Text extraction, table preservation, document stats |
| TXT | `.txt` | Text extraction, encoding detection, line/word count |
| EML | `.eml` | Email parsing, HTML conversion, metadata extraction |

## Error Handling

The parser includes comprehensive error handling:

```python
from parser import ParserError

try:
    result = parse_file("document.pdf")
except ParserError as e:
    print(f"Parsing failed: {e}")
```

### Common Error Scenarios
- **Unsupported file type**: File extension not in supported list
- **File not found**: Local file doesn't exist
- **Download failed**: Network issues or invalid URL
- **OCR required**: Scanned PDF with OCR disabled
- **MIME mismatch**: File content doesn't match extension
- **Encoding issues**: Text file with unsupported encoding

## Performance Considerations

### Memory Usage
- Large files are processed in chunks to manage memory
- Temporary files are automatically cleaned up
- OCR processing can be memory-intensive for large PDFs

### Speed Optimizations
- Async I/O for network operations
- Efficient text chunking with word boundary detection
- Lazy loading of optional dependencies (OCR, MIME validation)

### Configuration Tips
- Disable OCR for faster processing if not needed
- Use appropriate chunk sizes for your use case
- Consider file size limits for your environment

## Security Features

### File Validation
- MIME type validation prevents spoofed extensions
- File size limits prevent memory exhaustion
- Sanitized file handling prevents path traversal

### Safe Defaults
- OCR disabled by default (requires explicit opt-in)
- Conservative file size limits (50MB default)
- Graceful fallbacks for missing optional dependencies

## Integration Examples

### With RAG Pipeline
```python
from parser import parse_file
from metadata_store import MetadataStore

# Parse document
result = parse_file("document.pdf", max_chars=1000)

# Store metadata
metadata_store = MetadataStore("postgresql://...")
doc_id = metadata_store.save_document_metadata({
    "filename": "document.pdf",
    "doc_hash": "sha256_hash",
    "metadata": result["metadata"]
})

# Store chunks
if "text_chunks" in result:
    metadata_store.save_chunks(doc_id, result["text_chunks"])
```

### With FastAPI Endpoint
```python
from fastapi import FastAPI, UploadFile
from parser import parse_file

app = FastAPI()

@app.post("/parse")
async def parse_document(file: UploadFile, max_chars: int = None):
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        result = parse_file(temp_path, max_chars=max_chars)
        return result
    finally:
        os.unlink(temp_path)
```

## Testing

### Run Basic Tests
```bash
cd ingestion
python test_parser_enhanced_simple.py
```

### Run Comprehensive Tests
```bash
cd ingestion
python test_parser_enhanced.py
```

### Test Specific Features
```python
# Test OCR functionality
parser = FileParser(enable_ocr=True)
result = parser.parse_from_file("scanned_document.pdf")

# Test chunking
result = parser.parse_from_file("long_document.txt", max_chars=500)

# Test MIME validation
is_valid = parser._validate_mime_type(Path("file.txt"), ".txt")
```

## Troubleshooting

### OCR Issues
- Ensure Tesseract is installed and in PATH
- Check that `pytesseract` and `Pillow` are installed
- Verify OCR is enabled with `enable_ocr=True`

### MIME Validation Issues
- Install `python-magic` for better accuracy
- Check file permissions for MIME detection
- Verify file extensions match content

### Performance Issues
- Disable OCR for faster processing
- Reduce chunk sizes for large documents
- Use appropriate temp directory with sufficient space

## Future Enhancements

- **Multi-language OCR**: Support for non-English text
- **Advanced table detection**: Better table structure recognition
- **Image extraction**: Extract and process embedded images
- **Form field extraction**: Parse PDF form fields
- **Digital signatures**: Validate document authenticity
- **Batch processing**: Process multiple files efficiently

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility when possible

## License

This module is part of the RAG-API project and follows the same licensing terms. 