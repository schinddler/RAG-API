#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced parser.py features.

Tests OCR fallback, MIME validation, HTML parsing in EML,
DOCX table preservation, chunking, and metadata extraction.
"""

import asyncio
import tempfile
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import io

# Import the parser
from parser import FileParser, ParserError, parse_url, parse_file


class TestEnhancedParser:
    """Test suite for enhanced parser features."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.parser = FileParser(temp_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_mime_validation(self):
        """Test MIME type validation."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("This is a test file")
        
        # Test with python-magic available
        with patch('parser.MAGIC_AVAILABLE', True):
            with patch('parser.magic') as mock_magic:
                mock_magic.from_file.return_value = 'text/plain'
                assert self.parser._validate_mime_type(test_file, '.txt') == True
                
                # Test mismatch
                mock_magic.from_file.return_value = 'application/pdf'
                assert self.parser._validate_mime_type(test_file, '.txt') == False
        
        # Test with python-magic unavailable
        with patch('parser.MAGIC_AVAILABLE', False):
            with patch('parser.fallback_mimetypes') as mock_mimetypes:
                mock_mimetypes.guess_type.return_value = ('text/plain', None)
                assert self.parser._validate_mime_type(test_file, '.txt') == True
    
    def test_pdf_metadata_extraction(self):
        """Test PDF metadata extraction."""
        # Mock PDF document with metadata
        mock_doc = MagicMock()
        mock_doc.metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Test Subject',
            'creator': 'Test Creator',
            'producer': 'Test Producer',
            'creationDate': '2024-01-01',
            'modDate': '2024-01-02'
        }
        mock_doc.__len__.return_value = 5
        
        # Mock page
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Sample text content"
        
        with patch('parser.fitz.open', return_value=mock_doc):
            mock_doc.load_page.return_value = mock_page
            mock_doc.close = MagicMock()
            
            text, metadata = self.parser._parse_pdf(Path("test.pdf"))
            
            assert "Sample text content" in text
            assert metadata['title'] == 'Test Document'
            assert metadata['author'] == 'Test Author'
            assert metadata['page_count'] == 5
    
    def test_ocr_fallback(self):
        """Test OCR fallback for scanned PDFs."""
        # Mock PDF with no text (scanned)
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_doc.__len__.return_value = 1
        
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""  # No text extracted
        
        with patch('parser.fitz.open', return_value=mock_doc):
            mock_doc.load_page.return_value = mock_page
            mock_doc.close = MagicMock()
            
            # Test with OCR enabled
            parser_with_ocr = FileParser(enable_ocr=True)
            
            with patch('parser.OCR_AVAILABLE', True):
                with patch('parser.pytesseract') as mock_tesseract:
                    mock_tesseract.image_to_string.return_value = "OCR extracted text"
                    
                    with patch('parser.Image') as mock_pil:
                        mock_img = MagicMock()
                        mock_pil.open.return_value = mock_img
                        
                        with patch('parser.io.BytesIO') as mock_bytesio:
                            mock_bytesio.return_value = MagicMock()
                            
                            text, metadata = parser_with_ocr._parse_pdf(Path("test.pdf"))
                            
                            assert "OCR extracted text" in text
                            assert metadata.get('ocr_used') == True
    
    def test_docx_table_preservation(self):
        """Test DOCX table structure preservation."""
        # Mock DOCX document with tables
        mock_doc = MagicMock()
        mock_doc.paragraphs = []
        mock_doc.sections = []
        
        # Mock table
        mock_table = MagicMock()
        mock_row1 = MagicMock()
        mock_row2 = MagicMock()
        
        # Mock cells
        mock_cell1 = MagicMock()
        mock_cell1.text = "Header1"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Header2"
        mock_cell3 = MagicMock()
        mock_cell3.text = "Data1"
        mock_cell4 = MagicMock()
        mock_cell4.text = "Data2"
        
        mock_row1.cells = [mock_cell1, mock_cell2]
        mock_row2.cells = [mock_cell3, mock_cell4]
        mock_table.rows = [mock_row1, mock_row2]
        
        mock_doc.tables = [mock_table]
        mock_doc.__len__ = MagicMock(return_value=1)
        
        with patch('parser.DocxDocument', return_value=mock_doc):
            text, metadata = self.parser._parse_docx(Path("test.docx"))
            
            assert "Header1 | Header2" in text
            assert "Data1 | Data2" in text
            assert metadata['table_count'] == 1
            assert len(metadata['tables']) == 1
            assert metadata['tables'][0]['headers'] == ['Header1', 'Header2']
    
    def test_eml_html_parsing(self):
        """Test EML HTML content parsing."""
        # Create mock email message with HTML content
        mock_msg = MagicMock()
        mock_msg.get.return_value = "Test Subject"
        mock_msg.get_content_type.return_value = "text/html"
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content.return_value = "<html><body><p>Test HTML content</p></body></html>"
        
        with patch('parser.BytesParser') as mock_parser:
            mock_parser.return_value.parse.return_value = mock_msg
            
            with patch('parser.HTML2TEXT_AVAILABLE', True):
                with patch('parser.html2text') as mock_html2text:
                    mock_converter = MagicMock()
                    mock_converter.ignore_links = True
                    mock_converter.ignore_images = True
                    mock_converter.handle.return_value = "Test HTML content"
                    mock_html2text.HTML2Text.return_value = mock_converter
                    
                    text, metadata = self.parser._parse_eml(Path("test.eml"))
                    
                    assert "Test HTML content" in text
                    assert metadata['subject'] == "Test Subject"
    
    def test_text_chunking(self):
        """Test text chunking functionality."""
        # Create long text
        long_text = "This is a very long text that needs to be chunked. " * 50
        
        chunks = self.parser._chunk_text(long_text, max_chars=100, overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
        
        # Test that chunks overlap
        for i in range(len(chunks) - 1):
            # Check for some overlap (not exact due to word boundaries)
            assert len(set(chunks[i][-20:]) & set(chunks[i+1][:20])) > 0
    
    def test_txt_metadata(self):
        """Test TXT file metadata extraction."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "Line 1\nLine 2\nLine 3\nThis is a test document with multiple lines."
        test_file.write_text(test_content, encoding='utf-8')
        
        text, metadata = self.parser._parse_txt(test_file)
        
        assert text == test_content
        assert metadata['encoding'] == 'utf-8'
        assert metadata['line_count'] == 4
        assert metadata['word_count'] == 12
    
    def test_parser_integration(self):
        """Test complete parser integration with all features."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "This is a test document for integration testing. " * 20
        test_file.write_text(test_content)
        
        # Test with chunking
        result = self.parser.parse_from_file(str(test_file), max_chars=100)
        
        assert 'filetype' in result
        assert 'url' in result
        assert 'full_text' in result
        assert 'text_chunks' in result
        assert 'length' in result
        assert 'metadata' in result
        
        assert result['filetype'] == 'txt'
        assert len(result['text_chunks']) > 1
        assert result['metadata']['encoding'] == 'utf-8'


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_parse_file_convenience(self):
        """Test parse_file convenience function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            result = parse_file(temp_path, max_chars=50, enable_ocr=False)
            assert result['filetype'] == 'txt'
            assert 'Test content' in result['full_text']
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_parse_url_convenience(self):
        """Test parse_url convenience function."""
        # Mock the download and parsing
        with patch('parser.FileParser') as mock_parser_class:
            mock_parser = MagicMock()
            mock_parser.parse_from_url.return_value = {
                'filetype': 'pdf',
                'url': 'https://example.com/test.pdf',
                'full_text': 'Test PDF content',
                'length': 18,
                'metadata': {'title': 'Test PDF'}
            }
            mock_parser_class.return_value = mock_parser
            
            result = await parse_url("https://example.com/test.pdf", max_chars=100, enable_ocr=True)
            
            assert result['filetype'] == 'pdf'
            mock_parser.parse_from_url.assert_called_once_with("https://example.com/test.pdf", 100)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        parser = FileParser()
        
        with pytest.raises(ParserError, match="Unsupported file type"):
            parser.parse_from_file("test.xyz")
    
    def test_file_not_found(self):
        """Test handling of non-existent files."""
        parser = FileParser()
        
        with pytest.raises(ParserError, match="File not found"):
            parser.parse_from_file("nonexistent.txt")
    
    def test_ocr_not_available(self):
        """Test OCR when not available."""
        parser = FileParser(enable_ocr=True)
        
        with patch('parser.OCR_AVAILABLE', False):
            with pytest.raises(ParserError, match="OCR not available"):
                parser._ocr_pdf_pages(Path("test.pdf"))
    
    def test_mime_validation_error(self):
        """Test MIME validation error handling."""
        parser = FileParser()
        test_file = Path("test.txt")
        
        with patch('parser.MAGIC_AVAILABLE', True):
            with patch('parser.magic') as mock_magic:
                mock_magic.from_file.side_effect = Exception("Magic error")
                
                # Should continue processing even if validation fails
                assert parser._validate_mime_type(test_file, '.txt') == True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 