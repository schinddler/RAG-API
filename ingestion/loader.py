"""
Document Loader Module for RAG Backend

This module provides a robust, production-grade document loading system that can:
- Download documents from various sources (HTTP, HTTPS, S3, GCS, local files)
- Support multiple file formats (PDF, DOCX, TXT, EML, HTML, ZIP)
- Handle file deduplication and validation
- Detect OCR requirements for scanned documents
- Capture comprehensive metadata
- Provide security validation and error handling
"""

import os
import hashlib
import uuid
import zipfile
import mimetypes
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from urllib.parse import urlparse, unquote
import tempfile
import shutil

# Standard library imports
import requests
from requests.exceptions import RequestException, Timeout, TooManyRedirects
import aiohttp
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Document processing imports
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from io import StringIO

# Custom exceptions
class DocumentLoadError(Exception):
    """Custom exception for document loading errors"""
    pass

class DocumentValidationError(Exception):
    """Custom exception for document validation errors"""
    pass

class DocumentLoader:
    """
    Production-grade document loader for RAG backend.
    
    Supports multiple file formats, download capabilities, deduplication,
    OCR detection, metadata capture, and security validation.
    """
    
    # Supported file extensions and their MIME types
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.txt': 'text/plain',
        '.eml': 'message/rfc822',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.zip': 'application/zip'
    }
    
    # Dangerous file extensions to reject
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js', '.jar',
        '.msi', '.dmg', '.app', '.sh', '.py', '.php', '.asp', '.aspx', '.jsp'
    }
    
    def __init__(
        self,
        cache_dir: str = "/tmp/docs",
        max_file_size_mb: int = 50,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        enable_async: bool = True
    ):
        """
        Initialize the document loader.
        
        Args:
            cache_dir: Directory to store downloaded files
            max_file_size_mb: Maximum file size in MB
            timeout_seconds: Download timeout in seconds
            max_retries: Maximum retry attempts for downloads
            enable_async: Enable async capabilities
        """
        self.cache_dir = Path(cache_dir)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.enable_async = enable_async
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _compute_file_hash(self, file_path: Union[str, Path]) -> str:
        """Compute SHA-256 hash of file content."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and injection."""
        # Remove or replace dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    def _validate_file_extension(self, filename: str) -> Tuple[str, str]:
        """Validate file extension and return extension and MIME type."""
        file_ext = Path(filename).suffix.lower()
        
        # Check for dangerous extensions
        if file_ext in self.DANGEROUS_EXTENSIONS:
            raise DocumentValidationError(f"Dangerous file extension not allowed: {file_ext}")
        
        # Check if extension is supported
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise DocumentValidationError(f"Unsupported file extension: {file_ext}")
        
        mime_type = self.SUPPORTED_EXTENSIONS[file_ext]
        return file_ext, mime_type
    
    def _validate_file_size(self, file_path: Union[str, Path]) -> None:
        """Validate file size against maximum allowed size."""
        file_size = Path(file_path).stat().st_size
        if file_size > self.max_file_size_bytes:
            raise DocumentValidationError(
                f"File size {file_size} bytes exceeds maximum allowed size "
                f"{self.max_file_size_bytes} bytes"
            )
    
    def _detect_pdf_ocr_requirement(self, pdf_path: Union[str, Path]) -> bool:
        """
        Detect if PDF requires OCR by checking text extraction.
        
        Returns:
            True if OCR is required, False otherwise
        """
        try:
            # Try PyMuPDF first
            doc = fitz.open(pdf_path)
            text_content = ""
            
            for page_num in range(min(3, len(doc))):  # Check first 3 pages
                page = doc.load_page(page_num)
                text_content += page.get_text()
            
            doc.close()
            
            # If text content is very short, likely needs OCR
            if len(text_content.strip()) < 100:
                return True
            
            # Additional check with pdfminer
            try:
                pdfminer_text = pdfminer_extract_text(pdf_path)
                if len(pdfminer_text.strip()) < 100:
                    return True
            except Exception:
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error detecting OCR requirement: {e}")
            return True  # Assume OCR needed if detection fails
    
    def _extract_zip_contents(self, zip_path: Union[str, Path]) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        Extract ZIP file and return list of extracted files with metadata.
        
        Returns:
            List of tuples (file_path, metadata)
        """
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Create extraction directory
                extract_dir = self.cache_dir / f"extracted_{uuid.uuid4().hex[:8]}"
                extract_dir.mkdir(exist_ok=True)
                
                # Extract all files
                zip_ref.extractall(extract_dir)
                
                # Flatten directory structure and collect files
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        file_path = Path(root) / file
                        
                        # Skip hidden files and directories
                        if any(part.startswith('.') for part in file_path.parts):
                            continue
                        
                        try:
                            # Validate file
                            self._validate_file_extension(file)
                            self._validate_file_size(file_path)
                            
                            # Compute metadata
                            doc_hash = self._compute_file_hash(file_path)
                            file_size = file_path.stat().st_size
                            
                            metadata = {
                                'filename': self._sanitize_filename(file),
                                'source_url': str(zip_path),
                                'doc_type': 'extracted_from_zip',
                                'extension': Path(file).suffix.lower(),
                                'file_size': file_size,
                                'upload_time': datetime.utcnow().isoformat(),
                                'doc_hash': doc_hash,
                                'requires_ocr': False,
                                'extracted_path': str(file_path)
                            }
                            
                            # Check OCR requirement for PDFs
                            if Path(file).suffix.lower() == '.pdf':
                                metadata['requires_ocr'] = self._detect_pdf_ocr_requirement(file_path)
                            
                            extracted_files.append((file_path, metadata))
                            
                        except DocumentValidationError as e:
                            self.logger.warning(f"Skipping invalid file {file}: {e}")
                            continue
                
        except zipfile.BadZipFile:
            raise DocumentLoadError("Invalid or corrupted ZIP file")
        except Exception as e:
            raise DocumentLoadError(f"Error extracting ZIP file: {e}")
        
        return extracted_files
    
    def _download_file_sync(self, url: str) -> Tuple[Path, Dict[str, Any]]:
        """Download file synchronously from URL."""
        parsed_url = urlparse(url)
        
        # Handle different URL schemes
        if parsed_url.scheme in ['http', 'https']:
            return self._download_http_file(url)
        elif parsed_url.scheme == 's3':
            return self._download_s3_file(url)
        elif parsed_url.scheme == 'gs':
            return self._download_gcs_file(url)
        elif parsed_url.scheme == 'file':
            return self._download_local_file(url)
        else:
            raise DocumentLoadError(f"Unsupported URL scheme: {parsed_url.scheme}")
    
    def _download_http_file(self, url: str) -> Tuple[Path, Dict[str, Any]]:
        """Download file from HTTP/HTTPS URL."""
        headers = {
            'User-Agent': 'RAG-Document-Loader/1.0'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout_seconds,
                    stream=True
                )
                response.raise_for_status()
                
                # Get filename from URL or Content-Disposition header
                filename = self._get_filename_from_response(response, url)
                filename = self._sanitize_filename(filename)
                
                # Validate content type
                content_type = response.headers.get('content-type', '').split(';')[0]
                self._validate_content_type(content_type, filename)
                
                # Download to temporary file first
                temp_file = self.cache_dir / f"temp_{uuid.uuid4().hex[:8]}_{filename}"
                
                with open(temp_file, 'wb') as f:
                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            downloaded_size += len(chunk)
                            if downloaded_size > self.max_file_size_bytes:
                                temp_file.unlink(missing_ok=True)
                                raise DocumentValidationError("File size exceeds maximum allowed size")
                            f.write(chunk)
                
                # Validate file size
                self._validate_file_size(temp_file)
                
                # Compute hash and create final filename
                doc_hash = self._compute_file_hash(temp_file)
                final_filename = f"{doc_hash[:8]}_{filename}"
                final_path = self.cache_dir / final_filename
                
                # Handle file collision by adding timestamp if needed
                counter = 1
                original_final_path = final_path
                while final_path.exists():
                    name, ext = os.path.splitext(original_final_path.name)
                    final_path = self.cache_dir / f"{name}_{counter}{ext}"
                    counter += 1
                
                # Move to final location
                temp_file.rename(final_path)
                
                # Create metadata
                metadata = {
                    'filename': filename,
                    'source_url': url,
                    'doc_type': 'downloaded',
                    'extension': Path(filename).suffix.lower(),
                    'file_size': final_path.stat().st_size,
                    'upload_time': datetime.utcnow().isoformat(),
                    'doc_hash': doc_hash,
                    'requires_ocr': False,
                    'content_type': content_type
                }
                
                # Check OCR requirement for PDFs
                if Path(filename).suffix.lower() == '.pdf':
                    metadata['requires_ocr'] = self._detect_pdf_ocr_requirement(final_path)
                
                return final_path, metadata
                
            except (RequestException, Timeout, TooManyRedirects) as e:
                if attempt == self.max_retries - 1:
                    raise DocumentLoadError(f"Failed to download file after {self.max_retries} attempts: {e}")
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                continue
    
    def _download_s3_file(self, url: str) -> Tuple[Path, Dict[str, Any]]:
        """Download file from S3 URL."""
        # This would require boto3 - implement based on your AWS setup
        raise DocumentLoadError("S3 download not implemented - requires boto3 setup")
    
    def _download_gcs_file(self, url: str) -> Tuple[Path, Dict[str, Any]]:
        """Download file from Google Cloud Storage URL."""
        # This would require google-cloud-storage - implement based on your GCS setup
        raise DocumentLoadError("GCS download not implemented - requires google-cloud-storage setup")
    
    def _download_local_file(self, url: str) -> Tuple[Path, Dict[str, Any]]:
        """Download file from local file URL."""
        file_path = Path(unquote(urlparse(url).path))
        
        if not file_path.exists():
            raise DocumentLoadError(f"Local file not found: {file_path}")
        
        # Validate file
        filename = file_path.name
        self._validate_file_extension(filename)
        self._validate_file_size(file_path)
        
        # Copy to cache directory
        doc_hash = self._compute_file_hash(file_path)
        cache_filename = f"{doc_hash[:8]}_{filename}"
        cache_path = self.cache_dir / cache_filename
        
        # Handle file collision by adding counter if needed
        counter = 1
        original_cache_path = cache_path
        while cache_path.exists():
            name, ext = os.path.splitext(original_cache_path.name)
            cache_path = self.cache_dir / f"{name}_{counter}{ext}"
            counter += 1
        
        shutil.copy2(file_path, cache_path)
        
        # Create metadata
        metadata = {
            'filename': filename,
            'source_url': url,
            'doc_type': 'local_file',
            'extension': file_path.suffix.lower(),
            'file_size': cache_path.stat().st_size,
            'upload_time': datetime.utcnow().isoformat(),
            'doc_hash': doc_hash,
            'requires_ocr': False
        }
        
        # Check OCR requirement for PDFs
        if file_path.suffix.lower() == '.pdf':
            metadata['requires_ocr'] = self._detect_pdf_ocr_requirement(cache_path)
        
        return cache_path, metadata
    
    def _get_filename_from_response(self, response: requests.Response, url: str) -> str:
        """Extract filename from response headers or URL."""
        # Try Content-Disposition header first
        content_disposition = response.headers.get('content-disposition', '')
        if 'filename=' in content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"\'')
            if filename:
                return filename
        
        # Fall back to URL
        parsed_url = urlparse(url)
        filename = Path(parsed_url.path).name
        
        # If no filename in URL, generate one
        if not filename or filename == '/':
            content_type = response.headers.get('content-type', '').split(';')[0]
            extension = mimetypes.guess_extension(content_type) or '.bin'
            filename = f"downloaded_file{extension}"
        
        return filename
    
    def _validate_content_type(self, content_type: str, filename: str) -> None:
        """Validate content type against file extension."""
        expected_mime = self.SUPPORTED_EXTENSIONS.get(Path(filename).suffix.lower())
        if expected_mime and content_type and content_type != expected_mime:
            self.logger.warning(f"Content type mismatch: expected {expected_mime}, got {content_type}")
    
    async def _download_file_async(self, url: str) -> Tuple[Path, Dict[str, Any]]:
        """Download file asynchronously from URL."""
        # Run sync download in thread pool for async compatibility
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._download_file_sync, url)
    
    def load_from_url(self, url: str) -> Union[Tuple[Path, Dict[str, Any]], List[Tuple[Path, Dict[str, Any]]]]:
        """
        Load document from URL.
        
        Args:
            url: URL to download document from
            
        Returns:
            Tuple of (file_path, metadata) or list of tuples for ZIP files
        """
        try:
            self.logger.info(f"Loading document from URL: {url}")
            
            file_path, metadata = self._download_file_sync(url)
            
            # Handle ZIP files
            if metadata['extension'] == '.zip':
                extracted_files = self._extract_zip_contents(file_path)
                # Clean up the original ZIP file
                file_path.unlink(missing_ok=True)
                return extracted_files
            
            return file_path, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading document from URL {url}: {e}")
            raise DocumentLoadError(f"Failed to load document from URL: {e}")
    
    async def load_from_url_async(self, url: str) -> Union[Tuple[Path, Dict[str, Any]], List[Tuple[Path, Dict[str, Any]]]]:
        """
        Load document from URL asynchronously.
        
        Args:
            url: URL to download document from
            
        Returns:
            Tuple of (file_path, metadata) or list of tuples for ZIP files
        """
        if not self.enable_async:
            raise DocumentLoadError("Async mode not enabled")
        
        try:
            self.logger.info(f"Loading document from URL (async): {url}")
            
            file_path, metadata = await self._download_file_async(url)
            
            # Handle ZIP files
            if metadata['extension'] == '.zip':
                # Run ZIP extraction in thread pool
                loop = asyncio.get_event_loop()
                extracted_files = await loop.run_in_executor(
                    self.executor, self._extract_zip_contents, file_path
                )
                # Clean up the original ZIP file
                file_path.unlink(missing_ok=True)
                return extracted_files
            
            return file_path, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading document from URL {url}: {e}")
            raise DocumentLoadError(f"Failed to load document from URL: {e}")
    
    def load_from_file(self, file_path: Union[str, Path]) -> Union[Tuple[Path, Dict[str, Any]], List[Tuple[Path, Dict[str, Any]]]]:
        """
        Load document from local file.
        
        Args:
            file_path: Path to local file
            
        Returns:
            Tuple of (file_path, metadata) or list of tuples for ZIP files
        """
        try:
            file_path = Path(file_path)
            self.logger.info(f"Loading document from file: {file_path}")
            
            if not file_path.exists():
                raise DocumentLoadError(f"File not found: {file_path}")
            
            # Validate file
            filename = file_path.name
            self._validate_file_extension(filename)
            self._validate_file_size(file_path)
            
            # Copy to cache directory
            doc_hash = self._compute_file_hash(file_path)
            cache_filename = f"{doc_hash[:8]}_{filename}"
            cache_path = self.cache_dir / cache_filename
            
            # Handle file collision by adding counter if needed
            counter = 1
            original_cache_path = cache_path
            while cache_path.exists():
                name, ext = os.path.splitext(original_cache_path.name)
                cache_path = self.cache_dir / f"{name}_{counter}{ext}"
                counter += 1
            
            shutil.copy2(file_path, cache_path)
            
            # Create metadata
            metadata = {
                'filename': filename,
                'source_url': str(file_path),
                'doc_type': 'local_file',
                'extension': file_path.suffix.lower(),
                'file_size': cache_path.stat().st_size,
                'upload_time': datetime.utcnow().isoformat(),
                'doc_hash': doc_hash,
                'requires_ocr': False
            }
            
            # Check OCR requirement for PDFs
            if file_path.suffix.lower() == '.pdf':
                metadata['requires_ocr'] = self._detect_pdf_ocr_requirement(cache_path)
            
            # Handle ZIP files
            if metadata['extension'] == '.zip':
                extracted_files = self._extract_zip_contents(cache_path)
                # Clean up the original ZIP file
                cache_path.unlink(missing_ok=True)
                return extracted_files
            
            return cache_path, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading document from file {file_path}: {e}")
            raise DocumentLoadError(f"Failed to load document from file: {e}")
    
    def check_duplicate(self, doc_hash: str) -> bool:
        """
        Check if document with given hash already exists.
        
        Args:
            doc_hash: SHA-256 hash of document content
            
        Returns:
            True if document exists, False otherwise
        """
        # Check if any file in cache starts with the hash
        for file_path in self.cache_dir.glob(f"{doc_hash[:8]}*"):
            if file_path.is_file():
                return True
        return False
    
    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """
        Clean up old files from cache.
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
            
        Returns:
            Number of files cleaned up
        """
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up {file_path}: {e}")
        
        self.logger.info(f"Cleaned up {cleaned_count} old files from cache")
        return cleaned_count
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Convenience functions for backward compatibility
def load_document(url_or_path: str) -> Union[Tuple[Path, Dict[str, Any]], List[Tuple[Path, Dict[str, Any]]]]:
    """
    Convenience function to load a document from URL or file path.
    
    Args:
        url_or_path: URL or file path to load document from
        
    Returns:
        Tuple of (file_path, metadata) or list of tuples for ZIP files
    """
    loader = DocumentLoader()
    
    if url_or_path.startswith(('http://', 'https://', 's3://', 'gs://', 'file://')):
        return loader.load_from_url(url_or_path)
    else:
        return loader.load_from_file(url_or_path)


async def load_document_async(url_or_path: str) -> Union[Tuple[Path, Dict[str, Any]], List[Tuple[Path, Dict[str, Any]]]]:
    """
    Convenience function to load a document asynchronously from URL or file path.
    
    Args:
        url_or_path: URL or file path to load document from
        
    Returns:
        Tuple of (file_path, metadata) or list of tuples for ZIP files
    """
    loader = DocumentLoader(enable_async=True)
    
    if url_or_path.startswith(('http://', 'https://', 's3://', 'gs://', 'file://')):
        return await loader.load_from_url_async(url_or_path)
    else:
        return loader.load_from_file(url_or_path)
