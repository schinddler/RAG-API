"""
Test suite for the chunker module.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path

from chunker import (
    Chunker, 
    ChunkConfig, 
    SplitStrategy, 
    Chunk, 
    ChunkMetadata,
    create_chunker,
    Tokenizer,
    TextCleaner,
    SentenceSplitter
)


class TestTokenizer:
    """Test the Tokenizer class."""
    
    def setup_method(self):
        """Set up test tokenizer."""
        self.tokenizer = Tokenizer()
    
    def test_token_counting(self):
        """Test token counting functionality."""
        text = "This is a test sentence with multiple words."
        token_count = self.tokenizer.count_tokens(text)
        
        # Should return a positive integer
        assert isinstance(token_count, int)
        assert token_count > 0
        
        # Should handle empty text
        empty_count = self.tokenizer.count_tokens("")
        assert empty_count == 0
    
    def test_approximate_token_count(self):
        """Test approximate token counting fallback."""
        text = "Hello world! This is a test."
        count = self.tokenizer._approximate_token_count(text)
        
        # Should count words correctly
        expected_words = ["Hello", "world", "This", "is", "a", "test"]
        assert count == len(expected_words)
    
    def test_different_texts(self):
        """Test token counting with different text types."""
        texts = [
            "Simple text.",
            "Text with punctuation, numbers 123, and symbols!",
            "Multiple\nlines\nof\ntext",
            "   Text   with   extra   spaces   ",
        ]
        
        for text in texts:
            count = self.tokenizer.count_tokens(text)
            assert isinstance(count, int)
            assert count >= 0


class TestTextCleaner:
    """Test the TextCleaner class."""
    
    def test_clean_ocr_text(self):
        """Test OCR text cleaning."""
        dirty_text = "This |s OCR text w1th 0rrors   and   extra   spaces."
        cleaned = TextCleaner.clean_ocr_text(dirty_text)
        
        # Should fix common OCR issues
        assert "|" not in cleaned
        assert "1" not in cleaned  # Should be converted to 'l'
        assert "0" not in cleaned  # Should be converted to 'O'
        assert "   " not in cleaned  # Should normalize spaces
    
    def test_assess_text_quality(self):
        """Test text quality assessment."""
        # Good quality text
        good_text = "This is a well-formed sentence with proper punctuation."
        quality, suspicious = TextCleaner.assess_text_quality(good_text, [])
        assert quality > 0.8
        assert not suspicious
        
        # Poor quality text
        bad_text = "TEXT WITH ALL CAPS AND MULTIPLE   SPACES!!!"
        quality, suspicious = TextCleaner.assess_text_quality(bad_text, [])
        assert quality < 1.0
        assert suspicious
        
        # Empty text
        quality, suspicious = TextCleaner.assess_text_quality("", [])
        assert quality == 0.0
        assert suspicious
    
    def test_suspicious_patterns(self):
        """Test suspicious pattern detection."""
        patterns = [
            r'[A-Z]{5,}',  # ALL CAPS
            r'\s{3,}',     # Multiple spaces
        ]
        
        # Text with suspicious patterns
        suspicious_text = "THIS TEXT HAS ALL CAPS AND   multiple   spaces"
        quality, is_suspicious = TextCleaner.assess_text_quality(suspicious_text, patterns)
        assert is_suspicious
        assert quality < 1.0
        
        # Clean text
        clean_text = "This is clean text with normal formatting."
        quality, is_suspicious = TextCleaner.assess_text_quality(clean_text, patterns)
        assert not is_suspicious
        assert quality > 0.8


class TestSentenceSplitter:
    """Test the SentenceSplitter class."""
    
    def setup_method(self):
        """Set up test sentence splitter."""
        self.splitter = SentenceSplitter()
    
    def test_split_sentences(self):
        """Test sentence splitting functionality."""
        text = "This is sentence one. This is sentence two! And this is sentence three?"
        sentences = self.splitter.split_sentences(text)
        
        assert len(sentences) == 3
        assert all(isinstance(s, str) for s in sentences)
        assert all(s.strip() for s in sentences)
    
    def test_regex_fallback(self):
        """Test regex-based sentence splitting fallback."""
        text = "Sentence one. Sentence two! Sentence three?"
        sentences = self.splitter._split_with_regex(text)
        
        assert len(sentences) == 3
        assert "Sentence one" in sentences[0]
        assert "Sentence two" in sentences[1]
        assert "Sentence three" in sentences[2]
    
    def test_empty_text(self):
        """Test sentence splitting with empty text."""
        sentences = self.splitter.split_sentences("")
        assert sentences == []
        
        sentences = self.splitter.split_sentences("   ")
        assert sentences == []
    
    def test_single_sentence(self):
        """Test sentence splitting with single sentence."""
        text = "This is a single sentence."
        sentences = self.splitter.split_sentences(text)
        
        assert len(sentences) == 1
        assert sentences[0] == text


class TestChunker:
    """Test the Chunker class."""
    
    def setup_method(self):
        """Set up test chunker."""
        self.config = ChunkConfig(
            split_strategy=SplitStrategy.SENTENCE,
            max_tokens=50,
            min_tokens=10,
            overlap_tokens=5,
            enable_dedup=True,
            preclean_ocr=True
        )
        self.chunker = Chunker(self.config)
    
    def test_chunker_initialization(self):
        """Test chunker initialization."""
        assert self.chunker.config.split_strategy == SplitStrategy.SENTENCE
        assert self.chunker.config.max_tokens == 50
        assert self.chunker.config.enable_dedup == True
        assert hasattr(self.chunker, 'tokenizer')
        assert hasattr(self.chunker, 'text_cleaner')
        assert hasattr(self.chunker, 'sentence_splitter')
    
    def test_chunk_document_basic(self):
        """Test basic document chunking."""
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        result = self.chunker.chunk_document(text, source="test.txt")
        
        assert isinstance(result.chunks, list)
        assert len(result.chunks) > 0
        assert result.total_chunks == len(result.chunks)
        assert result.processing_time > 0
        
        # Check chunk structure
        for chunk in result.chunks:
            assert isinstance(chunk, Chunk)
            assert isinstance(chunk.content, str)
            assert isinstance(chunk.metadata, ChunkMetadata)
            assert chunk.metadata.source == "test.txt"
            assert chunk.metadata.chunk_id is not None
    
    def test_chunk_by_sentences(self):
        """Test sentence-based chunking."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = self.chunker._chunk_by_sentences(text, "test.txt")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata.token_count is not None
            assert chunk.metadata.token_count >= self.config.min_tokens
    
    def test_chunk_fixed(self):
        """Test fixed-size chunking."""
        self.chunker.config.split_strategy = SplitStrategy.FIXED
        text = "This is a longer text that should be split into fixed-size chunks. " * 10
        chunks = self.chunker._chunk_fixed(text, "test.txt")
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.content) <= self.config.max_chars
    
    def test_chunk_recursive(self):
        """Test recursive chunking for structured documents."""
        self.chunker.config.split_strategy = SplitStrategy.RECURSIVE
        text = """
        1. First Section
        This is the content of the first section.
        
        2. Second Section
        This is the content of the second section.
        
        3. Third Section
        This is the content of the third section.
        """
        chunks = self.chunker._chunk_recursive(text, "test.txt")
        
        assert len(chunks) > 0
        # Should preserve section structure
    
    def test_deduplication(self):
        """Test chunk deduplication."""
        # Create chunks with duplicate content
        chunks = [
            Chunk("Same content", ChunkMetadata("test.txt", 0, 0, 10)),
            Chunk("Same content", ChunkMetadata("test.txt", 1, 10, 20)),
            Chunk("Different content", ChunkMetadata("test.txt", 2, 20, 35)),
        ]
        
        unique_chunks, duplicate_count = self.chunker._deduplicate_chunks(chunks)
        
        assert len(unique_chunks) == 2  # Should remove one duplicate
        assert duplicate_count == 1
        assert unique_chunks[0].content == "Same content"
        assert unique_chunks[1].content == "Different content"
    
    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        text1 = "Hello, World!"
        text2 = "hello world"
        text3 = "  Hello,   World!  "
        
        id1 = self.chunker._generate_chunk_id(text1)
        id2 = self.chunker._generate_chunk_id(text2)
        id3 = self.chunker._generate_chunk_id(text3)
        
        # Should be consistent with embedder.py
        assert id1 == id2 == id3
        assert len(id1) == 16  # Truncated SHA256
    
    def test_empty_document(self):
        """Test chunking empty document."""
        result = self.chunker.chunk_document("", "empty.txt")
        
        assert len(result.chunks) == 0
        assert result.total_chunks == 0
        assert result.duplicate_chunks_removed == 0
        assert result.suspicious_chunks == 0
    
    def test_ocr_cleaning(self):
        """Test OCR text cleaning during chunking."""
        dirty_text = "This |s OCR text w1th 0rrors.   Multiple   spaces."
        result = self.chunker.chunk_document(dirty_text, "ocr.txt")
        
        # Should clean the text before chunking
        for chunk in result.chunks:
            assert "|" not in chunk.content
            assert "1" not in chunk.content
            assert "0" not in chunk.content
            assert "   " not in chunk.content
    
    def test_quality_assessment(self):
        """Test quality assessment during chunking."""
        suspicious_text = "ALL CAPS TEXT WITH MULTIPLE   SPACES!!!"
        result = self.chunker.chunk_document(suspicious_text, "suspicious.txt")
        
        # Should flag suspicious chunks
        suspicious_found = any(chunk.metadata.is_suspicious for chunk in result.chunks)
        assert suspicious_found
        
        # Should have quality scores
        for chunk in result.chunks:
            assert hasattr(chunk.metadata, 'quality_score')
            assert 0.0 <= chunk.metadata.quality_score <= 1.0


class TestFactoryFunctions:
    """Test factory functions and utilities."""
    
    def test_create_chunker(self):
        """Test chunker factory function."""
        chunker = create_chunker(
            split_strategy="sentence",
            max_tokens=100,
            min_tokens=20,
            overlap_tokens=10,
            enable_dedup=False,
            preclean_ocr=False
        )
        
        assert isinstance(chunker, Chunker)
        assert chunker.config.split_strategy == SplitStrategy.SENTENCE
        assert chunker.config.max_tokens == 100
        assert chunker.config.min_tokens == 20
        assert chunker.config.overlap_tokens == 10
        assert chunker.config.enable_dedup == False
        assert chunker.config.preclean_ocr == False
    
    def test_create_chunker_invalid_strategy(self):
        """Test factory function with invalid strategy."""
        chunker = create_chunker(split_strategy="invalid")
        
        # Should fall back to recursive strategy
        assert chunker.config.split_strategy == SplitStrategy.RECURSIVE


class TestIntegration:
    """Integration tests for the chunker module."""
    
    def setup_method(self):
        """Set up integration test."""
        self.temp_dir = tempfile.mkdtemp()
        self.chunker = create_chunker(
            split_strategy="sentence",
            max_tokens=100,
            min_tokens=20,
            overlap_tokens=10
        )
    
    def teardown_method(self):
        """Clean up integration test."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self):
        """Test complete chunking workflow."""
        # Sample legal document text
        document_text = """
        INSURANCE POLICY AGREEMENT
        
        1. COVERAGE TERMS
        This insurance policy provides coverage for the insured party as described herein.
        All terms and conditions must be strictly followed.
        
        2. PREMIUM PAYMENTS
        Premium payments are due on the first day of each month.
        Late payments may result in policy cancellation.
        
        3. CLAIMS PROCESS
        Claims must be filed within 30 days of the incident.
        Documentation must be provided with all claims.
        
        4. EXCLUSIONS
        This policy does not cover intentional acts or criminal activities.
        Natural disasters are covered under separate provisions.
        """
        
        result = self.chunker.chunk_document(document_text, source="insurance_policy.txt")
        
        # Should create multiple chunks
        assert len(result.chunks) > 1
        
        # Should preserve document structure
        chunk_contents = [chunk.content for chunk in result.chunks]
        assert any("COVERAGE TERMS" in content for content in chunk_contents)
        assert any("PREMIUM PAYMENTS" in content for content in chunk_contents)
        
        # Should have proper metadata
        for i, chunk in enumerate(result.chunks):
            assert chunk.metadata.source == "insurance_policy.txt"
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.token_count is not None
            assert chunk.metadata.chunk_id is not None
        
        # Should have reasonable statistics
        assert result.processing_time > 0
        assert result.metadata['avg_tokens_per_chunk'] > 0
    
    def test_deduplication_workflow(self):
        """Test deduplication in full workflow."""
        # Text with repeated sections
        text = """
        Section A: Introduction
        This is the introduction section.
        
        Section B: Main Content
        This is the main content section.
        
        Section A: Introduction
        This is the introduction section.
        
        Section C: Conclusion
        This is the conclusion section.
        """
        
        result = self.chunker.chunk_document(text, source="duplicate_sections.txt")
        
        # Should remove duplicates
        assert result.duplicate_chunks_removed > 0
        
        # Should have unique chunks
        chunk_contents = [chunk.content for chunk in result.chunks]
        unique_contents = set(chunk_contents)
        assert len(chunk_contents) == len(unique_contents)
    
    def test_quality_assessment_workflow(self):
        """Test quality assessment in full workflow."""
        # Text with suspicious patterns
        text = """
        Normal text with good quality.
        
        ALL CAPS TEXT WITH MULTIPLE   SPACES!!!
        
        More normal text.
        
        Text with excessive punctuation!!!!!?????
        """
        
        result = self.chunker.chunk_document(text, source="mixed_quality.txt")
        
        # Should flag suspicious chunks
        suspicious_chunks = [chunk for chunk in result.chunks if chunk.metadata.is_suspicious]
        assert len(suspicious_chunks) > 0
        
        # Should have varying quality scores
        quality_scores = [chunk.metadata.quality_score for chunk in result.chunks]
        assert min(quality_scores) < max(quality_scores)  # Should have variation


class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_import(self):
        """Test that CLI can be imported and run."""
        # This test verifies the CLI interface is properly structured
        # Actual CLI testing would require subprocess calls
        
        # Check that main block exists
        import chunker
        assert hasattr(chunker, '__main__')


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 