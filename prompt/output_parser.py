import re
import time
import logging
import json
import html
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

# Import dependencies (assuming these exist in the project)
try:
    from ..monitoring.logs import log_event, log_success, log_failure, log_latency
    from ..monitoring.alerting import get_alert_manager
    from ..llm.fallback_manager import get_fallback_manager
except ImportError:
    def log_event(event_type: str, message: str, metadata: dict = None, **kwargs):
        logging.info(f"Output Parser - {event_type}: {message}")
    
    def log_success(module: str, action: str, details: dict = None, **kwargs):
        logging.info(f"Output Parser Success - {module}.{action}")
    
    def log_failure(module: str, action: str, error: Exception, details: dict = None, **kwargs):
        logging.error(f"Output Parser Failure - {module}.{action}: {error}")
    
    def log_latency(module: str, operation: str, duration_ms: float, details: dict = None, **kwargs):
        logging.info(f"Output Parser Latency - {module}.{operation}: {duration_ms:.2f}ms")
    
    def get_alert_manager():
        return None
    
    def get_fallback_manager():
        return None


class OutputFormat(Enum):
    """Supported output formats"""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


@dataclass
class ParsedOutput:
    """Structured parsed output"""
    answer: str
    confidence_score: Optional[float] = None
    citations: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    format: OutputFormat = OutputFormat.TEXT
    is_truncated: bool = False
    has_citations: bool = False
    word_count: int = 0
    char_count: int = 0


class OutputParser:
    """
    High-performance output parser for LLM responses with validation,
    safety checks, and model-specific post-processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the output parser.
        
        Args:
            config: Configuration dictionary with parsing settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.alert_manager = get_alert_manager()
        
        # Parsing settings
        self.max_output_length = self.config.get("max_output_length", 10000)
        self.min_output_length = self.config.get("min_output_length", 10)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enable_citation_extraction = self.config.get("enable_citation_extraction", True)
        self.enable_safety_checks = self.config.get("enable_safety_checks", True)
        
        # Model-specific rules
        self.model_rules = self.config.get("model_rules", {
            "gpt-4": {
                "trim_preamble": True,
                "extract_citations": True,
                "markdown_cleanup": True
            },
            "claude": {
                "trim_preamble": True,
                "extract_citations": True,
                "markdown_cleanup": False
            },
            "mistral": {
                "trim_preamble": False,
                "extract_citations": False,
                "markdown_cleanup": True
            }
        })
        
        # Citation patterns
        self.citation_patterns = [
            r'\[(\d+)\]',
            r'\(([^)]+)\)',
            r'Source: ([^\n]+)',
            r'Reference: ([^\n]+)',
            r'Cited from: ([^\n]+)'
        ]
        
        # Confidence patterns
        self.confidence_patterns = [
            r'confidence[:\s]*(\d+(?:\.\d+)?)%',
            r'certainty[:\s]*(\d+(?:\.\d+)?)%',
            r'confidence[:\s]*(\d+(?:\.\d+)?)',
            r'certainty[:\s]*(\d+(?:\.\d+)?)'
        ]
        
        # Safety patterns (content to flag)
        self.safety_patterns = [
            r'\b(?:password|secret|key|token)\s*[:=]\s*\S+',
            r'\b(?:admin|root|sudo)\s+',
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html'
        ]
        
        self.logger.info("Output parser initialized")
    
    def parse_response(self, raw_output: str, model: str, query: str) -> Dict[str, Any]:
        """
        Parse and validate raw LLM output.
        
        Args:
            raw_output: Raw LLM response text
            model: Model name that generated the response
            query: Original query for context
            
        Returns:
            Dictionary with parsed and validated output
        """
        start_time = time.time()
        
        try:
            if not raw_output or not raw_output.strip():
                self._handle_empty_response(model, query)
                return self._create_empty_response()
            
            # Clean the output
            cleaned_output = self.clean_output(raw_output)
            
            # Apply model-specific rules
            processed_output = self._apply_model_rules(cleaned_output, model)
            
            # Extract structured elements
            parsed = self._extract_structured_elements(processed_output, model)
            
            # Validate the parsed output
            is_valid = self.validate_format(parsed)
            
            if not is_valid:
                self._handle_invalid_response(model, query, raw_output)
                return self._create_fallback_response(query)
            
            # Check for safety issues
            if self.enable_safety_checks:
                safety_issues = self._check_safety_issues(processed_output)
                if safety_issues:
                    self._handle_safety_issues(model, query, safety_issues)
                    processed_output = self._sanitize_output(processed_output)
            
            # Create final response
            response = {
                "answer": parsed.answer,
                "confidence_score": parsed.confidence_score,
                "citations": parsed.citations,
                "sources": parsed.sources,
                "metadata": {
                    "model": model,
                    "format": parsed.format.value,
                    "is_truncated": parsed.is_truncated,
                    "has_citations": parsed.has_citations,
                    "word_count": parsed.word_count,
                    "char_count": parsed.char_count,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            }
            
            # Log successful parsing
            latency_ms = (time.time() - start_time) * 1000
            log_success("output_parser", "parse_response", {
                "model": model,
                "word_count": parsed.word_count,
                "has_citations": parsed.has_citations,
                "confidence_score": parsed.confidence_score,
                "latency_ms": latency_ms
            })
            log_latency("output_parser", "parse_response", latency_ms, {"model": model})
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_failure("output_parser", "parse_response", e, {
                "model": model,
                "output_length": len(raw_output),
                "latency_ms": latency_ms
            })
            
            if self.alert_manager:
                self.alert_manager.alert_on_api_failure("output_parser", e)
            
            return self._create_fallback_response(query)
    
    def validate_format(self, parsed: ParsedOutput) -> bool:
        """
        Validate parsed output format and content.
        
        Args:
            parsed: Parsed output object
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check minimum length
            if len(parsed.answer.strip()) < self.min_output_length:
                self.logger.warning(f"Output too short: {len(parsed.answer)} chars")
                return False
            
            # Check maximum length
            if len(parsed.answer) > self.max_output_length:
                self.logger.warning(f"Output too long: {len(parsed.answer)} chars")
                parsed.is_truncated = True
            
            # Check for empty or whitespace-only content
            if not parsed.answer.strip():
                self.logger.warning("Output is empty or whitespace-only")
                return False
            
            # Check for common error patterns
            error_patterns = [
                r'^I apologize',
                r'^I cannot',
                r'^I don\'t have',
                r'^I\'m unable',
                r'^I\'m sorry',
                r'^As an AI',
                r'^I am an AI'
            ]
            
            for pattern in error_patterns:
                if re.search(pattern, parsed.answer, re.IGNORECASE):
                    self.logger.warning(f"Output contains error pattern: {pattern}")
                    return False
            
            # Check confidence score if present
            if parsed.confidence_score is not None:
                if not (0.0 <= parsed.confidence_score <= 1.0):
                    self.logger.warning(f"Invalid confidence score: {parsed.confidence_score}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def clean_output(self, text: str) -> str:
        """
        Clean and normalize output text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        try:
            if not text:
                return ""
            
            # Remove leading/trailing whitespace
            cleaned = text.strip()
            
            # Normalize line endings
            cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove excessive whitespace
            cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
            cleaned = re.sub(r' +', ' ', cleaned)
            
            # Remove common LLM artifacts
            artifacts = [
                r'^Here\'s the answer:',
                r'^Answer:',
                r'^Response:',
                r'^Based on the provided information:',
                r'^According to the context:',
                r'^The answer is:',
                r'^Here is the information:'
            ]
            
            for artifact in artifacts:
                cleaned = re.sub(artifact, '', cleaned, flags=re.IGNORECASE)
            
            # Clean up markdown if present
            cleaned = self._clean_markdown(cleaned)
            
            # Final trim
            cleaned = cleaned.strip()
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error cleaning output: {e}")
            return text.strip() if text else ""
    
    def _extract_structured_elements(self, text: str, model: str) -> ParsedOutput:
        """Extract structured elements from text."""
        try:
            # Extract citations
            citations = []
            if self.enable_citation_extraction:
                citations = self._extract_citations(text)
            
            # Extract confidence score
            confidence_score = self._extract_confidence_score(text)
            
            # Determine format
            output_format = self._detect_format(text)
            
            # Count words and characters
            word_count = len(text.split())
            char_count = len(text)
            
            # Create parsed output
            parsed = ParsedOutput(
                answer=text,
                confidence_score=confidence_score,
                citations=citations,
                sources=citations,  # Use citations as sources for now
                format=output_format,
                has_citations=len(citations) > 0,
                word_count=word_count,
                char_count=char_count
            )
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"Error extracting structured elements: {e}")
            return ParsedOutput(answer=text, word_count=len(text.split()), char_count=len(text))
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text."""
        citations = []
        
        try:
            for pattern in self.citation_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                citations.extend(matches)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_citations = []
            for citation in citations:
                if citation not in seen:
                    seen.add(citation)
                    unique_citations.append(citation)
            
            return unique_citations
            
        except Exception as e:
            self.logger.error(f"Error extracting citations: {e}")
            return []
    
    def _extract_confidence_score(self, text: str) -> Optional[float]:
        """Extract confidence score from text."""
        try:
            for pattern in self.confidence_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    # Normalize to 0-1 range
                    if score > 1.0:
                        score = score / 100.0
                    return min(max(score, 0.0), 1.0)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting confidence score: {e}")
            return None
    
    def _detect_format(self, text: str) -> OutputFormat:
        """Detect the format of the text."""
        try:
            # Check for JSON
            if text.strip().startswith('{') and text.strip().endswith('}'):
                try:
                    json.loads(text)
                    return OutputFormat.JSON
                except:
                    pass
            
            # Check for HTML
            if re.search(r'<[^>]+>', text):
                return OutputFormat.HTML
            
            # Check for Markdown
            if re.search(r'[#*_`\[\]]', text):
                return OutputFormat.MARKDOWN
            
            return OutputFormat.TEXT
            
        except Exception as e:
            self.logger.error(f"Error detecting format: {e}")
            return OutputFormat.TEXT
    
    def _apply_model_rules(self, text: str, model: str) -> str:
        """Apply model-specific processing rules."""
        try:
            rules = self.model_rules.get(model, {})
            
            # Trim preamble
            if rules.get("trim_preamble", False):
                text = self._trim_preamble(text)
            
            # Markdown cleanup
            if rules.get("markdown_cleanup", False):
                text = self._clean_markdown(text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error applying model rules: {e}")
            return text
    
    def _trim_preamble(self, text: str) -> str:
        """Remove common preamble text."""
        try:
            # Common preamble patterns
            preambles = [
                r'^Based on the information provided,?\s*',
                r'^According to the context,?\s*',
                r'^Based on the available information,?\s*',
                r'^Here\'s what I found:?\s*',
                r'^The information shows that?\s*'
            ]
            
            for preamble in preambles:
                text = re.sub(preamble, '', text, flags=re.IGNORECASE)
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error trimming preamble: {e}")
            return text
    
    def _clean_markdown(self, text: str) -> str:
        """Clean up markdown formatting."""
        try:
            # Remove excessive markdown
            text = re.sub(r'#{3,}', '##', text)  # Limit headers to ##
            text = re.sub(r'\*\*{3,}', '**', text)  # Limit bold
            text = re.sub(r'\*{3,}', '*', text)  # Limit italic
            
            # Clean up code blocks
            text = re.sub(r'```\s*\n', '```\n', text)
            text = re.sub(r'\n\s*```', '\n```', text)
            
            # Remove excessive line breaks
            text = re.sub(r'\n{4,}', '\n\n\n', text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error cleaning markdown: {e}")
            return text
    
    def _check_safety_issues(self, text: str) -> List[str]:
        """Check for safety issues in text."""
        issues = []
        
        try:
            for pattern in self.safety_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    issues.append(f"Safety pattern detected: {pattern}")
            
            return issues
            
        except Exception as e:
            self.logger.error(f"Error checking safety issues: {e}")
            return []
    
    def _sanitize_output(self, text: str) -> str:
        """Sanitize output for safety."""
        try:
            # HTML escape
            text = html.escape(text)
            
            # Remove script tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove dangerous protocols
            text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
            text = re.sub(r'data:text/html', '', text, flags=re.IGNORECASE)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error sanitizing output: {e}")
            return text
    
    def _handle_empty_response(self, model: str, query: str):
        """Handle empty response."""
        log_event("empty_response", f"Empty response from {model}", {
            "model": model,
            "query": query[:100]
        })
        
        if self.alert_manager:
            self.alert_manager.send_alert("empty_response", f"Empty response from {model}")
    
    def _handle_invalid_response(self, model: str, query: str, raw_output: str):
        """Handle invalid response."""
        log_event("invalid_response", f"Invalid response from {model}", {
            "model": model,
            "query": query[:100],
            "output_length": len(raw_output)
        })
        
        if self.alert_manager:
            self.alert_manager.send_alert("invalid_response", f"Invalid response from {model}")
    
    def _handle_safety_issues(self, model: str, query: str, issues: List[str]):
        """Handle safety issues."""
        log_event("safety_issues", f"Safety issues in {model} response", {
            "model": model,
            "query": query[:100],
            "issues": issues
        })
        
        if self.alert_manager:
            self.alert_manager.send_alert("safety_issues", f"Safety issues in {model} response")
    
    def _create_empty_response(self) -> Dict[str, Any]:
        """Create empty response."""
        return {
            "answer": "I apologize, but I couldn't generate a valid response. Please try again.",
            "confidence_score": 0.0,
            "citations": [],
            "sources": [],
            "metadata": {
                "model": "unknown",
                "format": "text",
                "is_truncated": False,
                "has_citations": False,
                "word_count": 0,
                "char_count": 0,
                "error": "empty_response"
            }
        }
    
    def _create_fallback_response(self, query: str) -> Dict[str, Any]:
        """Create fallback response."""
        return {
            "answer": f"I apologize, but I encountered an issue processing your request: '{query[:100]}...'. Please try rephrasing your question.",
            "confidence_score": 0.0,
            "citations": [],
            "sources": [],
            "metadata": {
                "model": "unknown",
                "format": "text",
                "is_truncated": False,
                "has_citations": False,
                "word_count": 0,
                "char_count": 0,
                "error": "parsing_failed"
            }
        }
    
    def get_parser_stats(self) -> Dict[str, Any]:
        """Get parser statistics."""
        return {
            "max_output_length": self.max_output_length,
            "min_output_length": self.min_output_length,
            "confidence_threshold": self.confidence_threshold,
            "enable_citation_extraction": self.enable_citation_extraction,
            "enable_safety_checks": self.enable_safety_checks,
            "model_rules": self.model_rules.copy(),
            "citation_patterns_count": len(self.citation_patterns),
            "confidence_patterns_count": len(self.confidence_patterns),
            "safety_patterns_count": len(self.safety_patterns)
        }


# Global instance for easy access
_output_parser = None


def get_output_parser(config: Optional[Dict[str, Any]] = None) -> OutputParser:
    """Get the global output parser instance."""
    global _output_parser
    if _output_parser is None:
        _output_parser = OutputParser(config)
    return _output_parser


# Convenience functions for easy access
def parse_response(raw_output: str, model: str, query: str) -> Dict[str, Any]:
    """Parse LLM response."""
    return get_output_parser().parse_response(raw_output, model, query)


def validate_format(parsed: ParsedOutput) -> bool:
    """Validate parsed output."""
    return get_output_parser().validate_format(parsed)


def clean_output(text: str) -> str:
    """Clean output text."""
    return get_output_parser().clean_output(text)


# Example usage and testing
if __name__ == "__main__":
    # Test the output parser
    parser = OutputParser({
        "max_output_length": 5000,
        "min_output_length": 10,
        "confidence_threshold": 0.7,
        "enable_citation_extraction": True,
        "enable_safety_checks": True
    })
    
    # Test cases
    test_cases = [
        {
            "raw_output": "Based on the information provided, the insurance policy covers medical expenses up to $50,000 per year. [1] This includes hospital stays, doctor visits, and prescription medications.",
            "model": "gpt-4",
            "query": "What does the insurance policy cover?"
        },
        {
            "raw_output": "I apologize, but I cannot provide that information.",
            "model": "claude",
            "query": "What is the password?"
        },
        {
            "raw_output": "The contract specifies payment terms of net 30 days from invoice date. Confidence: 85%",
            "model": "mistral",
            "query": "What are the payment terms?"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        result = parser.parse_response(
            test_case["raw_output"],
            test_case["model"],
            test_case["query"]
        )
        print(f"Answer: {result['answer'][:100]}...")
        print(f"Confidence: {result['confidence_score']}")
        print(f"Citations: {result['citations']}")
        print(f"Valid: {parser.validate_format(ParsedOutput(answer=result['answer']))}")
    
    # Print stats
    print(f"\nParser stats: {parser.get_parser_stats()}")  