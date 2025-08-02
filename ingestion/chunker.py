"""
Production-grade document chunking module for RAG system.
Supports semantic-aware chunking, adaptive sizing, deduplication, and metadata propagation.
"""

import os
import re
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import shared utilities and config
try:
    from ..utils.hashing import generate_chunk_id
    from ..config import SplitStrategy, ChunkConfig as BaseChunkConfig
except ImportError:
    # Handle case when running as script
    from utils.hashing import generate_chunk_id
    from config import SplitStrategy, ChunkConfig as BaseChunkConfig

# Optional imports for advanced features
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    sent_tokenize = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Use the SplitStrategy from central config
# SplitStrategy is imported from config.py


@dataclass
class ChunkConfig(BaseChunkConfig):
    """Configuration for document chunking."""
    # Inherits from BaseChunkConfig in config.py
    # Additional chunker-specific config can be added here if needed
    pass


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    source: str
    chunk_index: int
    char_start: int
    char_end: int
    section_title: Optional[str] = None
    is_suspicious: bool = False
    quality_score: float = 1.0
    token_count: Optional[int] = None
    chunk_id: Optional[str] = None


@dataclass
class Chunk:
    """A document chunk with content and metadata."""
    content: str
    metadata: ChunkMetadata


@dataclass
class ChunkingResult:
    """Result container for chunking operations."""
    chunks: List[Chunk]
    metadata: Dict[str, Any]
    processing_time: float
    total_chunks: int
    duplicate_chunks_removed: int
    suspicious_chunks: int


class Tokenizer:
    """Token counting utility with multiple backends."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self._tokenizer = None
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize the appropriate tokenizer."""
        if TIKTOKEN_AVAILABLE:
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.model_name)
                logger.info(f"Using tiktoken tokenizer for {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}")
                self._tokenizer = None
        
        if not self._tokenizer:
            logger.info("Using regex-based token approximation")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the best available method."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        else:
            # Fallback: approximate token count using word boundaries
            return self._approximate_token_count(text)
    
    def _approximate_token_count(self, text: str) -> int:
        """Approximate token count using word boundaries."""
        # Simple approximation: split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text)
        return len(words)


class TextCleaner:
    """Text cleaning utilities for OCR and noisy text."""
    
    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """Clean OCR'd text by removing artifacts and normalizing."""
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Fix common OCR issues
        text = re.sub(r'[|]', 'I', text)  # Common OCR mistake
        text = re.sub(r'[0]', 'O', text)  # Another common mistake
        text = re.sub(r'[1]', 'l', text)  # Another common mistake
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    @staticmethod
    def assess_text_quality(text: str, suspicious_patterns: List[str]) -> Tuple[float, bool]:
        """Assess text quality and flag suspicious content."""
        if not text.strip():
            return 0.0, True
        
        quality_score = 1.0
        is_suspicious = False
        
        # Check for suspicious patterns
        for pattern in suspicious_patterns:
            matches = len(re.findall(pattern, text))
            if matches > 0:
                # Reduce quality score based on pattern frequency
                pattern_ratio = matches / len(text)
                quality_score -= pattern_ratio * 0.3
                if pattern_ratio > 0.1:  # More than 10% suspicious content
                    is_suspicious = True
        
        # Check for reasonable text length
        if len(text) < 50:
            quality_score *= 0.5
            is_suspicious = True
        
        # Check for excessive repetition
        words = text.split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.5:  # More than 50% repetition
                quality_score *= 0.7
                is_suspicious = True
        
        return max(0.0, quality_score), is_suspicious


class SentenceSplitter:
    """Sentence splitting utilities with multiple backends."""
    
    def __init__(self):
        self._nlp = None
        self._initialize_spacy()
    
    def _initialize_spacy(self):
        """Initialize spaCy model if available."""
        if SPACY_AVAILABLE:
            try:
                # Try to load English model
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Using spaCy for sentence splitting")
            except OSError:
                try:
                    # Try to load any available model
                    self._nlp = spacy.load("en_core_web_sm")
                    logger.info("Using spaCy for sentence splitting")
                except Exception as e:
                    logger.warning(f"Failed to load spaCy model: {e}")
                    self._nlp = None
        else:
            logger.info("spaCy not available, using NLTK or regex fallback")
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using the best available method."""
        if self._nlp:
            return self._split_with_spacy(text)
        elif NLTK_AVAILABLE:
            return self._split_with_nltk(text)
        else:
            return self._split_with_regex(text)
    
    def _split_with_spacy(self, text: str) -> List[str]:
        """Split sentences using spaCy."""
        doc = self._nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return [s for s in sentences if s]
    
    def _split_with_nltk(self, text: str) -> List[str]:
        """Split sentences using NLTK."""
        try:
            # Download punkt if not available
            nltk.download('punkt', quiet=True)
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"NLTK sentence splitting failed: {e}")
            return self._split_with_regex(text)
    
    def _split_with_regex(self, text: str) -> List[str]:
        """Split sentences using regex fallback."""
        # Simple regex-based sentence splitting
        sentence_pattern = r'[^.!?]+[.!?]+'
        sentences = re.findall(sentence_pattern, text)
        
        # Handle remaining text
        remaining = re.sub(sentence_pattern, '', text).strip()
        if remaining:
            sentences.append(remaining)
        
        return [s.strip() for s in sentences if s.strip()]


class Chunker:
    """
    Production-grade document chunker for RAG system.
    
    Features:
    - Semantic-aware chunking with multiple strategies
    - Adaptive token-based sizing
    - Deduplication and quality assessment
    - Metadata propagation
    - OCR text cleaning
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize the chunker with configuration.
        
        Args:
            config: Chunking configuration. If None, uses defaults.
        """
        self.config = config or ChunkConfig()
        self.tokenizer = Tokenizer()
        self.text_cleaner = TextCleaner()
        self.sentence_splitter = SentenceSplitter()
        
        logger.info(f"Initialized chunker with strategy: {self.config.split_strategy.value}")
    
    def chunk_document(self, text: str, source: str = "unknown") -> ChunkingResult:
        """
        Chunk a document into semantic pieces.
        
        Args:
            text: Document text to chunk
            source: Source identifier for metadata
            
        Returns:
            ChunkingResult with chunks and processing info
        """
        start_time = time.time()
        
        # Pre-clean text if OCR cleaning is enabled
        if self.config.preclean_ocr:
            text = self.text_cleaner.clean_ocr_text(text)
        
        # Split into chunks based on strategy
        if self.config.split_strategy == SplitStrategy.RECURSIVE:
            chunks = self._chunk_recursive(text, source)
        elif self.config.split_strategy == SplitStrategy.SENTENCE:
            chunks = self._chunk_by_sentences(text, source)
        else:  # FIXED
            chunks = self._chunk_fixed(text, source)
        
        # Deduplicate if enabled
        duplicate_count = 0
        if self.config.enable_dedup:
            chunks, duplicate_count = self._deduplicate_chunks(chunks)
        
        # Assess quality and flag suspicious chunks
        suspicious_count = 0
        for chunk in chunks:
            quality_score, is_suspicious = self.text_cleaner.assess_text_quality(
                chunk.content, self.config.suspicious_patterns
            )
            chunk.metadata.quality_score = quality_score
            chunk.metadata.is_suspicious = is_suspicious
            if is_suspicious:
                suspicious_count += 1
        
        processing_time = time.time() - start_time
        
        # Generate chunk IDs using consistent method
        for chunk in chunks:
            chunk.metadata.chunk_id = self._generate_chunk_id(chunk.content)
        
        # Calculate statistics
        total_tokens = sum(chunk.metadata.token_count or 0 for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        result_metadata = {
            "source": source,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens,
            "duplicate_chunks_removed": duplicate_count,
            "suspicious_chunks": suspicious_count,
            "split_strategy": self.config.split_strategy.value,
            "config": {
                "max_tokens": self.config.max_tokens,
                "min_tokens": self.config.min_tokens,
                "overlap_tokens": self.config.overlap_tokens,
                "enable_dedup": self.config.enable_dedup,
                "preclean_ocr": self.config.preclean_ocr,
            }
        }
        
        logger.info(f"Chunked document into {len(chunks)} chunks "
                   f"(duplicates removed: {duplicate_count}, "
                   f"suspicious: {suspicious_count}) in {processing_time:.2f}s")
        
        return ChunkingResult(
            chunks=chunks,
            metadata=result_metadata,
            processing_time=processing_time,
            total_chunks=len(chunks),
            duplicate_chunks_removed=duplicate_count,
            suspicious_chunks=suspicious_count
        )
    
    def _chunk_recursive(self, text: str, source: str) -> List[Chunk]:
        """Chunk text using recursive splitting for structured documents."""
        chunks = []
        char_offset = 0
        
        # First, try to split by major section markers
        section_patterns = [
            r'\n\s*[0-9]+\.\s+[A-Z][^.\n]*\n',  # Numbered sections
            r'\n\s*[A-Z][A-Z\s]+\n',  # ALL CAPS headers
            r'\n\s*[A-Z][a-z\s]+\n',  # Title case headers
        ]
        
        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                if len(section) > self.config.max_chars:
                    # Split by pattern
                    splits = re.split(pattern, section)
                    new_sections.extend(splits)
                else:
                    new_sections.append(section)
            sections = new_sections
        
        # Process each section
        for section in sections:
            if not section.strip():
                continue
            
            # If section is still too large, split by sentences
            if len(section) > self.config.max_chars:
                sentence_chunks = self._chunk_by_sentences(section, source, char_offset)
                chunks.extend(sentence_chunks)
                char_offset += len(section)
            else:
                # Create chunk for this section
                token_count = self.tokenizer.count_tokens(section)
                if token_count >= self.config.min_tokens:
                    chunk = Chunk(
                        content=section.strip(),
                        metadata=ChunkMetadata(
                            source=source,
                            chunk_index=len(chunks),
                            char_start=char_offset,
                            char_end=char_offset + len(section),
                            token_count=token_count
                        )
                    )
                    chunks.append(chunk)
                char_offset += len(section)
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, source: str, char_offset: int = 0) -> List[Chunk]:
        """Chunk text by sentences with token-based sizing."""
        sentences = self.sentence_splitter.split_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_start = char_offset
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.count_tokens(sentence)
            
            # If adding this sentence would exceed max_tokens, finalize current chunk
            if current_tokens + sentence_tokens > self.config.max_tokens and current_chunk:
                # Create chunk
                chunk_content = ' '.join(current_chunk)
                if current_tokens >= self.config.min_tokens:
                    chunk = Chunk(
                        content=chunk_content,
                        metadata=ChunkMetadata(
                            source=source,
                            chunk_index=len(chunks),
                            char_start=chunk_start,
                            char_end=chunk_start + len(chunk_content),
                            token_count=current_tokens
                        )
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.config.overlap_tokens > 0:
                    # Keep some sentences for overlap
                    overlap_tokens = 0
                    overlap_sentences = []
                    for sent in reversed(current_chunk):
                        sent_tokens = self.tokenizer.count_tokens(sent)
                        if overlap_tokens + sent_tokens <= self.config.overlap_tokens:
                            overlap_sentences.insert(0, sent)
                            overlap_tokens += sent_tokens
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens
                    chunk_start = char_offset + len(text) - len(' '.join(current_chunk))
                else:
                    current_chunk = []
                    current_tokens = 0
                    chunk_start = char_offset + len(text) - len(sentence)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Handle remaining content
        if current_chunk and current_tokens >= self.config.min_tokens:
            chunk_content = ' '.join(current_chunk)
            chunk = Chunk(
                content=chunk_content,
                metadata=ChunkMetadata(
                    source=source,
                    chunk_index=len(chunks),
                    char_start=chunk_start,
                    char_end=chunk_start + len(chunk_content),
                    token_count=current_tokens
                )
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_fixed(self, text: str, source: str) -> List[Chunk]:
        """Chunk text using fixed character-based sizing."""
        chunks = []
        char_offset = 0
        
        while char_offset < len(text):
            # Calculate chunk boundaries
            chunk_end = min(char_offset + self.config.max_chars, len(text))
            
            # Try to break at word boundary
            if chunk_end < len(text):
                # Look for the last space before chunk_end
                last_space = text.rfind(' ', char_offset, chunk_end)
                if last_space > char_offset + self.config.min_chars:
                    chunk_end = last_space
            
            chunk_content = text[char_offset:chunk_end].strip()
            
            if len(chunk_content) >= self.config.min_chars:
                token_count = self.tokenizer.count_tokens(chunk_content)
                chunk = Chunk(
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        source=source,
                        chunk_index=len(chunks),
                        char_start=char_offset,
                        char_end=chunk_end,
                        token_count=token_count
                    )
                )
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            char_offset = chunk_end - self.config.overlap_chars
            if char_offset <= 0:
                break
        
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[Chunk]) -> Tuple[List[Chunk], int]:
        """Remove duplicate chunks while preserving order."""
        seen_hashes = set()
        unique_chunks = []
        duplicate_count = 0
        
        for chunk in chunks:
            # Use the same chunk ID generation as embedder for consistency
            chunk_hash = self._generate_chunk_id(chunk.content)
            
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)
            else:
                duplicate_count += 1
        
        return unique_chunks, duplicate_count
    
    def _generate_chunk_id(self, text: str) -> str:
        """
        Generate a stable, normalized ID using SHA256 hash for deduplication.
        Uses the shared generate_chunk_id function for consistency.
        Args:
            text: Input text to normalize and hash
        Returns:
            Truncated SHA256 hash (16 characters by default)
        """
        return generate_chunk_id(text, length=16)


# Factory function for easy chunker creation
def create_chunker(
    split_strategy: str = "recursive",
    max_tokens: int = 256,
    min_tokens: int = 100,
    overlap_tokens: int = 30,
    enable_dedup: bool = True,
    preclean_ocr: bool = True
) -> Chunker:
    """
    Factory function to create a chunker with common configurations.
    
    Args:
        split_strategy: Chunking strategy ("recursive", "sentence", "fixed")
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk
        overlap_tokens: Token overlap between chunks
        enable_dedup: Enable deduplication
        preclean_ocr: Enable OCR text cleaning
        
    Returns:
        Configured Chunker instance
    """
    try:
        strategy = SplitStrategy(split_strategy)
    except ValueError:
        logger.warning(f"Invalid split_strategy '{split_strategy}', using 'recursive'")
        strategy = SplitStrategy.RECURSIVE
    
    config = ChunkConfig(
        split_strategy=strategy,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens,
        enable_dedup=enable_dedup,
        preclean_ocr=preclean_ocr
    )
    
    return Chunker(config)


# CLI interface for debugging
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Document chunker CLI")
    parser.add_argument("input_file", help="Input text file to chunk")
    parser.add_argument("--strategy", choices=["recursive", "sentence", "fixed"], 
                       default="recursive", help="Chunking strategy")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens per chunk")
    parser.add_argument("--min-tokens", type=int, default=100, help="Minimum tokens per chunk")
    parser.add_argument("--overlap", type=int, default=30, help="Token overlap")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    parser.add_argument("--no-clean", action="store_true", help="Disable OCR cleaning")
    parser.add_argument("--output", help="Output file for chunks (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)
    
    # Create chunker and process
    chunker = create_chunker(
        split_strategy=args.strategy,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        overlap_tokens=args.overlap,
        enable_dedup=not args.no_dedup,
        preclean_ocr=not args.no_clean
    )
    
    result = chunker.chunk_document(text, source=args.input_file)
    
    # Print results
    print(f"\nChunking Results:")
    print(f"  Total chunks: {result.total_chunks}")
    print(f"  Processing time: {result.processing_time:.2f}s")
    print(f"  Duplicates removed: {result.duplicate_chunks_removed}")
    print(f"  Suspicious chunks: {result.suspicious_chunks}")
    print(f"  Average tokens per chunk: {result.metadata['avg_tokens_per_chunk']:.1f}")
    
    # Print chunks
    print(f"\nChunks:")
    for i, chunk in enumerate(result.chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Content: {chunk.content[:100]}...")
        print(f"  Tokens: {chunk.metadata.token_count}")
        print(f"  Quality: {chunk.metadata.quality_score:.2f}")
        print(f"  Suspicious: {chunk.metadata.is_suspicious}")
        print(f"  Chunk ID: {chunk.metadata.chunk_id}")
    
    # Save to file if requested
    if args.output:
        output_data = {
            "chunks": [
                {
                    "content": chunk.content,
                    "metadata": {
                        "source": chunk.metadata.source,
                        "chunk_index": chunk.metadata.chunk_index,
                        "char_start": chunk.metadata.char_start,
                        "char_end": chunk.metadata.char_end,
                        "token_count": chunk.metadata.token_count,
                        "quality_score": chunk.metadata.quality_score,
                        "is_suspicious": chunk.metadata.is_suspicious,
                        "chunk_id": chunk.metadata.chunk_id
                    }
                }
                for chunk in result.chunks
            ],
            "summary": result.metadata
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nChunks saved to: {args.output}")
