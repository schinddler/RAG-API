"""
Test suite for the parser module.

This module tests the FileParser class and convenience functions
with various file types and edge cases.
"""

import asyncio
import tempfile
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from parser import FileParser, ParserError, parse_url, parse_file


class TestFileParser:
    """Test cases for the FileParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = FileParser()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test files
        self.create_test_files()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_files(self):
        """Create test files for different formats."""
        # Create a simple text file
        self.txt_file = os.path.join(self.temp_dir, "test.txt")
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document.\nIt contains multiple lines.\n")
        
        # Create a simple PDF-like content (we'll mock the actual PDF parsing)
        self.pdf_file = os.path.join(self.temp_dir, "test.pdf")
        with open(self.pdf_file, 'w') as f:
            f.write("PDF content placeholder")
        
        # Create a simple DOCX-like content (we'll mock the actual DOCX parsing)
        self.docx_file = os.path.join(self.temp_dir, "test.docx")
        with open(self.docx_file, 'w') as f:
            f.write("DOCX content placeholder")
        
        # Create a simple EML file
        self.eml_file = os.path.join(self.temp_dir, "test.eml")
        eml_content = """From: sender@example.com
To: recipient@example.com
Subject: Test Email
Content-Type: text/plain

This is the body of the test email.
It contains multiple lines of text.
"""
        with open(self.eml_file, 'w', encoding='utf-8') as f:
            f.write(eml_content)
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = FileParser()
        assert parser.temp_dir == tempfile.gettempdir()
        assert parser.max_file_size == 50 * 1024 * 1024
        assert '.pdf' in parser.supported_extensions
        assert '.docx' in parser.supported_extensions
        assert '.txt' in parser.supported_extensions
        assert '.eml' in parser.supported_extensions
    
    def test_parser_with_custom_temp_dir(self):
        """Test parser with custom temp directory."""
        custom_temp = "/custom/temp"
        parser = FileParser(temp_dir=custom_temp)
        assert parser.temp_dir == custom_temp
    
    def test_get_extension_from_url(self):
        """Test extension extraction from URLs."""
        assert self.parser._get_extension_from_url("https://example.com/file.pdf") == ".pdf"
        assert self.parser._get_extension_from_url("https://example.com/document.docx") == ".docx"
        assert self.parser._get_extension_from_url("https://example.com/text.txt") == ".txt"
        assert self.parser._get_extension_from_url("https://example.com/email.eml") == ".eml"
        assert self.parser._get_extension_from_url("https://example.com/noextension") == ".txt"
    
    def test_parse_txt_file(self):
        """Test parsing of TXT files."""
        result = self.parser.parse_from_file(self.txt_file)
        
        assert result["filetype"] == "txt"
        assert result["url"] == self.txt_file
        assert "test document" in result["text"].lower()
        assert result["length"] > 0
    
    @patch('parser.fitz.open')
    def test_parse_pdf_file(self, mock_fitz_open):
        """Test parsing of PDF files."""
        # Mock PyMuPDF
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is PDF content."
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz_open.return_value = mock_doc
        
        result = self.parser.parse_from_file(self.pdf_file)
        
        assert result["filetype"] == "pdf"
        assert result["url"] == self.pdf_file
        assert "PDF content" in result["text"]
        assert result["length"] > 0
    
    @patch('parser.DocxDocument')
    def test_parse_docx_file(self, mock_docx):
        """Test parsing of DOCX files."""
        # Mock python-docx
        mock_doc = MagicMock()
        mock_paragraph = MagicMock()
        mock_paragraph.text = "This is DOCX content."
        mock_doc.paragraphs = [mock_paragraph]
        mock_doc.tables = []
        mock_docx.return_value = mock_doc
        
        result = self.parser.parse_from_file(self.docx_file)
        
        assert result["filetype"] == "docx"
        assert result["url"] == self.docx_file
        assert "DOCX content" in result["text"]
        assert result["length"] > 0
    
    def test_parse_eml_file(self):
        """Test parsing of EML files."""
        result = self.parser.parse_from_file(self.eml_file)
        
        assert result["filetype"] == "eml"
        assert result["url"] == self.eml_file
        assert "test email" in result["text"].lower()
        assert result["length"] > 0
    
    def test_parse_unsupported_file(self):
        """Test parsing of unsupported file types."""
        unsupported_file = os.path.join(self.temp_dir, "test.xyz")
        with open(unsupported_file, 'w') as f:
            f.write("content")
        
        with pytest.raises(ParserError, match="Unsupported file type"):
            self.parser.parse_from_file(unsupported_file)
    
    def test_parse_nonexistent_file(self):
        """Test parsing of non-existent file."""
        with pytest.raises(ParserError, match="File not found"):
            self.parser.parse_from_file("/nonexistent/file.txt")
    
    def test_parse_empty_file(self):
        """Test parsing of empty file."""
        empty_file = os.path.join(self.temp_dir, "empty.txt")
        with open(empty_file, 'w') as f:
            pass
        
        with pytest.raises(ParserError, match="No text content extracted"):
            self.parser.parse_from_file(empty_file)
    
    def test_parse_large_file(self):
        """Test parsing of file that exceeds size limit."""
        large_file = os.path.join(self.temp_dir, "large.txt")
        # Create a file larger than 50MB
        with open(large_file, 'w') as f:
            f.write("x" * (self.parser.max_file_size + 1024))
        
        with pytest.raises(ParserError, match="File too large"):
            self.parser.parse_from_file(large_file)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = """
        Page 1
        
        This is actual content.
        
        Page 2
        
        More content here.
        
        Confidential
        Draft
        """
        
        cleaned = self.parser._clean_text(dirty_text)
        assert "Page 1" not in cleaned
        assert "Page 2" not in cleaned
        assert "Confidential" not in cleaned
        assert "Draft" not in cleaned
        assert "actual content" in cleaned.lower()
        assert "More content" in cleaned
    
    @patch('parser.aiohttp.ClientSession')
    @patch('parser.aiofiles.open')
    async def test_parse_from_url_success(self, mock_aiofiles_open, mock_session):
        """Test successful URL parsing."""
        # Mock aiohttp response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {'content-length': '100'}
        mock_response.read.return_value = b"This is test content"
        
        mock_session_context = MagicMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.get.return_value.__aexit__.return_value = None
        
        # Mock aiofiles
        mock_file = MagicMock()
        mock_aiofiles_open.return_value.__aenter__.return_value = mock_file
        mock_aiofiles_open.return_value.__aexit__.return_value = None
        
        # Mock tempfile
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.name = "/tmp/test.txt"
            mock_temp.return_value.close.return_value = None
            
            result = await self.parser.parse_from_url("https://example.com/test.txt")
            
            assert result["filetype"] == "txt"
            assert result["url"] == "https://example.com/test.txt"
            assert result["length"] > 0
    
    @patch('parser.aiohttp.ClientSession')
    async def test_parse_from_url_http_error(self, mock_session):
        """Test URL parsing with HTTP error."""
        # Mock aiohttp response with error
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.reason = "Not Found"
        
        mock_session_context = MagicMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.get.return_value.__aexit__.return_value = None
        
        with pytest.raises(ParserError, match="HTTP 404"):
            await self.parser.parse_from_url("https://example.com/notfound.txt")
    
    @patch('parser.aiohttp.ClientSession')
    async def test_parse_from_url_large_file(self, mock_session):
        """Test URL parsing with file too large."""
        # Mock aiohttp response with large content length
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {'content-length': str(self.parser.max_file_size + 1024)}
        
        mock_session_context = MagicMock()
        mock_session_context.__aenter__.return_value = mock_session
        mock_session_context.__aexit__.return_value = None
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.get.return_value.__aexit__.return_value = None
        
        with pytest.raises(ParserError, match="File too large"):
            await self.parser.parse_from_url("https://example.com/large.txt")


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_parse_file_function(self):
        """Test the parse_file convenience function."""
        # Create a simple test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for convenience function.")
            temp_path = f.name
        
        try:
            result = parse_file(temp_path)
            assert result["filetype"] == "txt"
            assert "Test content" in result["text"]
            assert result["length"] > 0
        finally:
            os.unlink(temp_path)
    
    @patch('parser.FileParser.parse_from_url')
    async def test_parse_url_function(self, mock_parse_url):
        """Test the parse_url convenience function."""
        expected_result = {
            "text": "Test content",
            "filetype": "pdf",
            "url": "https://example.com/test.pdf",
            "length": 12
        }
        mock_parse_url.return_value = expected_result
        
        result = await parse_url("https://example.com/test.pdf")
        assert result == expected_result
        mock_parse_url.assert_called_once_with("https://example.com/test.pdf")


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_pdf_parsing_error(self):
        """Test PDF parsing error handling."""
        parser = FileParser()
        
        with patch('parser.fitz.open') as mock_fitz:
            mock_fitz.side_effect = Exception("PDF parsing failed")
            
            with pytest.raises(ParserError, match="PDF parsing failed"):
                parser._parse_pdf(Path("/test.pdf"))
    
    def test_docx_parsing_error(self):
        """Test DOCX parsing error handling."""
        parser = FileParser()
        
        with patch('parser.DocxDocument') as mock_docx:
            mock_docx.side_effect = Exception("DOCX parsing failed")
            
            with pytest.raises(ParserError, match="DOCX parsing failed"):
                parser._parse_docx(Path("/test.docx"))
    
    def test_txt_parsing_error(self):
        """Test TXT parsing error handling."""
        parser = FileParser()
        
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = Exception("File read failed")
            
            with pytest.raises(ParserError, match="TXT parsing failed"):
                parser._parse_txt(Path("/test.txt"))
    
    def test_eml_parsing_error(self):
        """Test EML parsing error handling."""
        parser = FileParser()
        
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = Exception("EML parsing failed")
            
            with pytest.raises(ParserError, match="EML parsing failed"):
                parser._parse_eml(Path("/test.eml"))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 