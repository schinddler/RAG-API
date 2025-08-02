"""
Text preprocessor for RAG pipeline.

This module cleans and normalizes raw parsed text from parser.py before it reaches chunkers.
Handles whitespace normalization, unicode fixes, bullet standardization, HTML cleanup,
table structure preservation, section detection, and header/footer removal.
"""

import re
import unicodedata
import html
from typing import List, Tuple, Optional, Dict, Any, Iterator
from dataclasses import dataclass
import logging

# HTML processing libraries
try:
    import html2text
    HTML2TEXT_AVAILABLE = True
except ImportError:
    HTML2TEXT_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

# Encoding fixes
try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    standardize_bullets: bool = True
    remove_headers_footers: bool = True
    structure_sections: bool = False
    enable_table_formatting: bool = False
    append_trailing_newline: bool = True
    preserve_indentation: bool = False


# Precompiled regex patterns for performance
WHITESPACE_REGEX = re.compile(r'[\t\r\f\v]+')
MULTI_SPACE_REGEX = re.compile(r'[ \u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]+')
EXCESSIVE_NEWLINES_REGEX = re.compile(r'\n{3,}')
WINDOWS_LINEBREAKS_REGEX = re.compile(r'\r\n')
MAC_LINEBREAKS_REGEX = re.compile(r'\r')
HTML_TAG_REGEX = re.compile(r'<[^>]+>')
TABLE_LINE_REGEX = re.compile(r'\S+\s{2,}\S+')
MULTI_SPACE_TO_PIPE_REGEX = re.compile(r'\s{2,}')

# Bullet and list patterns
BULLET_PATTERNS = [
    (re.compile(r'^[•‣◦⁃∙·]\s*'), '• '),
    (re.compile(r'^[-–—]\s*'), '• '),
    (re.compile(r'^\*\s*'), '• '),
    (re.compile(r'^o\s+'), '• '),
    (re.compile(r'^O\s+'), '• '),
]

NUMBERED_PATTERNS = [
    (re.compile(r'^(\d+)\.\s*'), r'\1. '),
    (re.compile(r'^(\d+)\)\s*'), r'\1. '),
    (re.compile(r'^([ivxlcdm]+)\.\s*', re.IGNORECASE), r'\1. '),
    (re.compile(r'^([a-z])\.\s*', re.IGNORECASE), r'\1. '),
]

# Section header patterns
SECTION_PATTERNS = [
    re.compile(r'^(\d+)\.\s+[A-Z][A-Za-z\s]+$'),
    re.compile(r'^([IVX]+)\.\s+[A-Z][A-Za-z\s]+$'),
    re.compile(r'^([A-Z][A-Za-z\s]+):$'),
    re.compile(r'^([A-Z][A-Za-z\s]+)$'),
]

# Header/footer patterns
HEADER_FOOTER_PATTERNS = [
    re.compile(r'^Page\s+\d+\s+of\s+\d+$'),
    re.compile(r'^Page\s+\d+$'),
    re.compile(r'^Confidential$'),
    re.compile(r'^Draft$'),
    re.compile(r'^Internal Use Only$'),
    re.compile(r'^For Review Only$'),
    re.compile(r'^\d+$'),
    re.compile(r'^[A-Z\s]+$'),
]


def preprocess_text(raw_text: str, config: Optional[PreprocessingConfig] = None) -> str:
    """
    Clean and normalize raw parsed text.
    
    Args:
        raw_text: Raw text from parser.py
        config: Preprocessing configuration. Uses defaults if None.
        
    Returns:
        Cleaned and normalized text
        
    This function is idempotent - running it multiple times won't break formatting.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""
    
    if config is None:
        config = PreprocessingConfig()
    
    # Step 1: Basic unicode normalization and encoding fixes
    text = _normalize_unicode(raw_text)
    
    # Step 2: HTML cleanup and conversion
    text = _clean_html(text)
    
    # Step 3: Whitespace normalization
    text = _normalize_whitespace(text, config.preserve_indentation)
    
    # Step 4: Bullet and list standardization
    if config.standardize_bullets:
        text = _standardize_bullets(text)
    
    # Step 5: Table structure preservation
    if config.enable_table_formatting:
        text = _preserve_table_structure(text)
    
    # Step 6: Section detection and header tagging
    text = _detect_sections(text, config.structure_sections)
    
    # Step 7: Remove redundant headers/footers
    if config.remove_headers_footers:
        text = _remove_redundant_headers_footers(text)
    
    # Step 8: Final cleanup
    text = _final_cleanup(text, config.append_trailing_newline)
    
    return text


def _normalize_unicode(text: str) -> str:
    """Normalize unicode characters and fix encoding issues using ftfy."""
    if not text:
        return ""
    
    # Use ftfy for robust encoding fixes if available
    if FTFY_AVAILABLE:
        text = ftfy.fix_text(text)
    else:
        # Fallback to basic unicode normalization
        text = unicodedata.normalize('NFKC', text)
    
    # Fix common unicode characters that might not be handled by ftfy
    unicode_replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201C': '"',  # Left double quotation mark
        '\u201D': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2022': '•',  # Bullet
        '\u2023': '‣',  # Triangular bullet
        '\u25E6': '◦',  # White bullet
        '\u2043': '⁃',  # Hyphen bullet
        '\u2219': '∙',  # Bullet operator
        '\u00B7': '·',  # Middle dot
    }
    
    for old, new in unicode_replacements.items():
        text = text.replace(old, new)
    
    return text


def _clean_html(text: str) -> str:
    """Clean HTML tags and convert to plain text."""
    if not text:
        return ""
    
    # Check if text contains HTML
    if '<' in text and '>' in text:
        try:
            if BEAUTIFULSOUP_AVAILABLE:
                # Use BeautifulSoup for better HTML parsing
                soup = BeautifulSoup(text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Convert to text with proper spacing
                text = soup.get_text(separator=' ', strip=True)
                
            elif HTML2TEXT_AVAILABLE:
                # Use html2text as fallback
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                h.body_width = 0  # Don't wrap text
                text = h.handle(text)
                
            else:
                # Basic HTML tag removal using regex
                text = HTML_TAG_REGEX.sub('', text)
                text = html.unescape(text)
                
        except Exception as e:
            logger.warning(f"HTML cleaning failed: {str(e)}")
            # Fallback to basic tag removal
            text = HTML_TAG_REGEX.sub('', text)
            text = html.unescape(text)
    
    return text


def _normalize_whitespace(text: str, preserve_indentation: bool = False) -> str:
    """Normalize whitespace characters."""
    if not text:
        return ""
    
    # Replace various whitespace characters with standard ones
    text = WHITESPACE_REGEX.sub(' ', text)
    text = MULTI_SPACE_REGEX.sub(' ', text)
    
    # Normalize line breaks
    text = WINDOWS_LINEBREAKS_REGEX.sub('\n', text)
    text = MAC_LINEBREAKS_REGEX.sub('\n', text)
    
    # Remove excessive newlines (more than 2 consecutive)
    text = EXCESSIVE_NEWLINES_REGEX.sub('\n\n', text)
    
    # Process lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if preserve_indentation:
            # Preserve leading whitespace for code blocks
            stripped = line.rstrip()
            if stripped:  # Keep non-empty lines
                cleaned_lines.append(stripped)
            elif cleaned_lines and cleaned_lines[-1]:  # Add single empty line between content
                cleaned_lines.append('')
        else:
            # Remove all leading/trailing whitespace
            stripped = line.strip()
            if stripped:  # Keep non-empty lines
                cleaned_lines.append(stripped)
            elif cleaned_lines and cleaned_lines[-1]:  # Add single empty line between content
                cleaned_lines.append('')
    
    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)


def _standardize_bullets(text: str) -> str:
    """Standardize bullet points and numbered lists."""
    if not text:
        return ""
    
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append(line)
            continue
        
        # Standardize bullet points
        for pattern, replacement in BULLET_PATTERNS:
            if pattern.match(line):
                line = pattern.sub(replacement, line)
                break
        
        # Standardize numbered lists
        for pattern, replacement in NUMBERED_PATTERNS:
            if pattern.match(line):
                line = pattern.sub(replacement, line)
                break
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def _preserve_table_structure(text: str) -> str:
    """Preserve table structure using delimiters - only for actual tables."""
    if not text:
        return ""
    
    lines = text.split('\n')
    if len(lines) < 2:
        return text
    
    # Find consecutive lines that match table pattern
    table_lines = []
    processed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            processed_lines.append(line)
            i += 1
            continue
        
        # Check if current line and next few lines form a table
        consecutive_table_lines = 0
        j = i
        while j < len(lines) and j < i + 5:  # Check up to 5 consecutive lines
            if lines[j].strip() and TABLE_LINE_REGEX.search(lines[j]):
                consecutive_table_lines += 1
            else:
                break
            j += 1
        
        # If we have at least 2 consecutive table-like lines, format them
        if consecutive_table_lines >= 2:
            for k in range(consecutive_table_lines):
                table_line = lines[i + k].strip()
                if table_line:
                    # Convert multiple spaces to pipe delimiters
                    formatted_line = MULTI_SPACE_TO_PIPE_REGEX.sub(' | ', table_line)
                    processed_lines.append(formatted_line.strip())
                else:
                    processed_lines.append('')
            i += consecutive_table_lines
        else:
            # Not a table, keep original line
            processed_lines.append(line)
            i += 1
    
    return '\n'.join(processed_lines)


def _detect_sections(text: str, structure_sections: bool = False) -> str:
    """Detect and optionally tag section headers."""
    if not text:
        return ""
    
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append(line)
            continue
        
        # Detect potential section headers
        is_section_header = False
        for pattern in SECTION_PATTERNS:
            if pattern.match(line):
                is_section_header = True
                break
        
        # If it's a section header, optionally tag it
        if is_section_header and structure_sections:
            # Determine if it's a main section or subsection
            if re.match(r'^(\d+)\.\s+[A-Z]', line):
                line = f"[SECTION] {line}"
            elif re.match(r'^([IVX]+)\.\s+[A-Z]', line):
                line = f"[SECTION] {line}"
            else:
                line = f"[SUBSECTION] {line}"
        elif is_section_header:
            # Ensure proper capitalization without tagging
            if line.isupper():
                line = line.title()
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def _remove_redundant_headers_footers(text: str) -> str:
    """Remove redundant headers and footers that appear repeatedly."""
    if not text:
        return ""
    
    lines = text.split('\n')
    if len(lines) < 3:
        return text
    
    # Count occurrences of potential headers/footers
    header_footer_counts: Dict[str, int] = {}
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        
        for pattern in HEADER_FOOTER_PATTERNS:
            if pattern.match(line_clean):
                header_footer_counts[line_clean] = header_footer_counts.get(line_clean, 0) + 1
                break
    
    # Remove lines that appear too frequently (likely headers/footers)
    processed_lines = []
    total_lines = len([l for l in lines if l.strip()])
    
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            processed_lines.append(line)
            continue
        
        # Check if this line is a frequent header/footer
        is_redundant = False
        for pattern in HEADER_FOOTER_PATTERNS:
            if pattern.match(line_clean):
                count = header_footer_counts.get(line_clean, 0)
                # Remove if it appears more than 10% of the time
                if count > max(2, total_lines * 0.1):
                    is_redundant = True
                break
        
        if not is_redundant:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def _final_cleanup(text: str, append_trailing_newline: bool = True) -> str:
    """Final cleanup and validation."""
    if not text:
        return ""
    
    # Remove any remaining excessive whitespace
    text = EXCESSIVE_NEWLINES_REGEX.sub('\n\n', text)
    
    # Ensure proper line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Conditionally append trailing newline
    if append_trailing_newline and text and not text.endswith('\n'):
        text += '\n'
    
    return text


def process_text_stream(text: str) -> Iterator[str]:
    """
    Process text as a stream to minimize memory usage for large documents.
    
    Args:
        text: Input text to process
        
    Yields:
        Processed lines one at a time
    """
    if not text:
        return
    
    lines = text.split('\n')
    for line in lines:
        # Basic processing for each line
        line = line.strip()
        if line:
            yield line
        else:
            yield ''


# Convenience function for testing
def test_preprocessor():
    """Test the preprocessor with sample text."""
    sample_text = """
    <html><body>
    <h1>Insurance Policy</h1>
    <p>This is a sample document with various formatting issues.</p>
    <ul>
        <li>• First bullet point</li>
        <li>* Second bullet point</li>
        <li>- Third bullet point</li>
    </ul>
    <table>
        <tr><td>Name</td><td>Value</td></tr>
        <tr><td>Policy</td><td>12345</td></tr>
    </table>
    <p>Page 1 of 5</p>
    </body></html>
    """
    
    # Test with default config
    cleaned = preprocess_text(sample_text)
    print("Original text:")
    print(sample_text)
    print("\nCleaned text (default config):")
    print(cleaned)
    
    # Test with custom config
    config = PreprocessingConfig(
        standardize_bullets=True,
        remove_headers_footers=True,
        structure_sections=True,
        enable_table_formatting=True,
        append_trailing_newline=False
    )
    
    cleaned_custom = preprocess_text(sample_text, config)
    print("\nCleaned text (custom config):")
    print(cleaned_custom)


if __name__ == "__main__":
    test_preprocessor()
