"""
Centralized prompt builder for RAG pipeline.

This module provides token-optimized, structured prompt formatting for LLM interactions
with support for different task types, model-specific adaptations, and traceable outputs.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported task types for prompt building."""
    CLAIM_EVALUATION = "claim_evaluation"
    COVERAGE_CHECK = "coverage_check"
    COMPLIANCE_AUDIT = "compliance_audit"
    POLICY_ANALYSIS = "policy_analysis"
    DOCUMENT_QA = "document_qa"
    LEGAL_REVIEW = "legal_review"
    HR_ASSESSMENT = "hr_assessment"
    GENERAL_QA = "general_qa"


class ModelType(Enum):
    """Supported LLM model types."""
    CLAUDE = "claude"
    GPT4 = "gpt-4"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    LOCAL = "local"


@dataclass
class PromptConfig:
    """Configuration for prompt building."""
    # Token limits
    max_tokens: int = 8000
    max_context_tokens: int = 6000
    max_response_tokens: int = 2000
    
    # Formatting options
    include_metadata: bool = True
    include_source_references: bool = True
    include_confidence_scores: bool = True
    compact_formatting: bool = True
    
    # Task-specific settings
    task_type: TaskType = TaskType.GENERAL_QA
    model_type: ModelType = ModelType.CLAUDE
    
    # Output formatting
    require_structured_output: bool = True
    include_justification: bool = True
    include_confidence: bool = True
    include_sources: bool = True


@dataclass
class ContextChunk:
    """Structured representation of a context chunk."""
    text: str
    source_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None
    clause_number: Optional[str] = None
    section_title: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class PromptBuilder:
    """Centralized prompt builder with token optimization and structured formatting."""
    
    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize prompt builder.
        
        Args:
            config: Prompt configuration. Uses defaults if None.
        """
        self.config = config or PromptConfig()
        self._templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize prompt templates for different task types."""
        return {
            TaskType.CLAIM_EVALUATION.value: self._get_claim_evaluation_template(),
            TaskType.COVERAGE_CHECK.value: self._get_coverage_check_template(),
            TaskType.COMPLIANCE_AUDIT.value: self._get_compliance_audit_template(),
            TaskType.POLICY_ANALYSIS.value: self._get_policy_analysis_template(),
            TaskType.DOCUMENT_QA.value: self._get_document_qa_template(),
            TaskType.LEGAL_REVIEW.value: self._get_legal_review_template(),
            TaskType.HR_ASSESSMENT.value: self._get_hr_assessment_template(),
            TaskType.GENERAL_QA.value: self._get_general_qa_template(),
        }
    
    def build_prompt(
        self,
        query: str,
        context_chunks: List[ContextChunk],
        task_type: Optional[TaskType] = None,
        model_type: Optional[ModelType] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Build optimized prompt for LLM interaction.
        
        Args:
            query: User query or structured input
            context_chunks: Retrieved context chunks
            task_type: Type of task (overrides config)
            model_type: LLM model type (overrides config)
            max_tokens: Token limit (overrides config)
            
        Returns:
            Formatted prompt string
        """
        # Use provided parameters or fall back to config
        task_type = task_type or self.config.task_type
        model_type = model_type or self.config.model_type
        max_tokens = max_tokens or self.config.max_tokens
        
        # Format context chunks
        formatted_context = self._format_context_chunks(context_chunks, max_tokens)
        
        # Get task-specific template
        template = self._templates.get(task_type.value, self._templates[TaskType.GENERAL_QA.value])
        
        # Build prompt with model-specific adaptations
        prompt = self._build_structured_prompt(
            query=query,
            context=formatted_context,
            template=template,
            task_type=task_type,
            model_type=model_type
        )
        
        # Validate token count
        estimated_tokens = self._estimate_tokens(prompt)
        if estimated_tokens > max_tokens:
            logger.warning(f"Prompt estimated at {estimated_tokens} tokens, exceeding limit of {max_tokens}")
            prompt = self._truncate_prompt(prompt, max_tokens)
        
        return prompt
    
    def _format_context_chunks(self, chunks: List[ContextChunk], max_tokens: int) -> str:
        """Format context chunks with metadata and references."""
        if not chunks:
            return "No relevant context available."
        
        formatted_chunks = []
        current_tokens = 0
        
        for i, chunk in enumerate(chunks, 1):
            # Build chunk header with metadata
            header_parts = [f"[Source {i}"]
            
            if self.config.include_source_references and chunk.source_id:
                header_parts.append(f"Doc: {chunk.source_id}")
            
            if chunk.clause_number:
                header_parts.append(f"Clause: {chunk.clause_number}")
            
            if chunk.page_number:
                header_parts.append(f"Page: {chunk.page_number}")
            
            if self.config.include_confidence_scores and chunk.score:
                header_parts.append(f"Score: {chunk.score:.3f}")
            
            header = " - ".join(header_parts) + "]"
            
            # Format chunk text
            chunk_text = chunk.text.strip()
            if self.config.compact_formatting:
                # Remove excessive whitespace
                chunk_text = " ".join(chunk_text.split())
            
            formatted_chunk = f"{header}\n{chunk_text}"
            
            # Estimate tokens for this chunk
            chunk_tokens = self._estimate_tokens(formatted_chunk)
            
            # Check if adding this chunk would exceed limits
            if current_tokens + chunk_tokens > self.config.max_context_tokens:
                if formatted_chunks:  # Keep at least one chunk
                    break
                else:  # If no chunks yet, truncate this one
                    formatted_chunk = self._truncate_text(formatted_chunk, self.config.max_context_tokens)
                    chunk_tokens = self._estimate_tokens(formatted_chunk)
            
            formatted_chunks.append(formatted_chunk)
            current_tokens += chunk_tokens
        
        return "\n\n".join(formatted_chunks)
    
    def _build_structured_prompt(
        self,
        query: str,
        context: str,
        template: str,
        task_type: TaskType,
        model_type: ModelType
    ) -> str:
        """Build structured prompt with model-specific adaptations."""
        
        # Get model-specific system prompt
        system_prompt = self._get_system_prompt(task_type, model_type)
        
        # Get output format instructions
        output_format = self._get_output_format(task_type, model_type)
        
        # Build user prompt
        user_prompt = template.format(
            context=context,
            query=query,
            output_format=output_format
        )
        
        # Model-specific formatting
        if model_type == ModelType.CLAUDE:
            return f"{system_prompt}\n\n{user_prompt}"
        elif model_type == ModelType.GPT4:
            return f"System: {system_prompt}\n\nUser: {user_prompt}"
        elif model_type == ModelType.DEEPSEEK:
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>"
        else:
            return f"{system_prompt}\n\n{user_prompt}"
    
    def _get_system_prompt(self, task_type: TaskType, model_type: ModelType) -> str:
        """Get system prompt for task and model type."""
        base_prompts = {
            TaskType.CLAIM_EVALUATION: "You are an expert insurance claim evaluator. Analyze the provided policy clauses and user query to determine claim eligibility, coverage, and payout amounts.",
            TaskType.COVERAGE_CHECK: "You are an insurance coverage specialist. Review policy documents to determine if specific procedures, treatments, or events are covered under the policy.",
            TaskType.COMPLIANCE_AUDIT: "You are a compliance auditor. Examine documents for regulatory compliance, policy violations, and required corrective actions.",
            TaskType.POLICY_ANALYSIS: "You are a policy analyst. Provide detailed analysis of insurance policies, legal documents, or HR policies based on the provided context.",
            TaskType.DOCUMENT_QA: "You are a document Q&A assistant. Answer questions based on the provided document context with accurate references.",
            TaskType.LEGAL_REVIEW: "You are a legal document reviewer. Analyze legal documents, contracts, and policies for key terms, obligations, and implications.",
            TaskType.HR_ASSESSMENT: "You are an HR policy specialist. Review HR documents, policies, and procedures to provide guidance on employment matters.",
            TaskType.GENERAL_QA: "You are an expert assistant. Answer questions based on the provided context with accurate, well-reasoned responses."
        }
        
        base_prompt = base_prompts.get(task_type, base_prompts[TaskType.GENERAL_QA])
        
        # Add model-specific instructions
        if model_type == ModelType.CLAUDE:
            return f"{base_prompt} Provide your response in the specified JSON format with clear justifications and source references."
        elif model_type == ModelType.GPT4:
            return f"{base_prompt} Format your response as structured JSON with explanations and source citations."
        else:
            return f"{base_prompt} Respond in the requested format with proper citations."
    
    def _get_output_format(self, task_type: TaskType, model_type: ModelType) -> str:
        """Get output format instructions for task type."""
        formats = {
            TaskType.CLAIM_EVALUATION: """{
  "decision": "Approved|Rejected|Conditional|Pending",
  "coverage_amount": "USD amount or 'Not covered'",
  "justification": "Detailed reasoning with clause references",
  "conditions": ["List of conditions if conditional"],
  "sources": ["Source 1", "Source 2"],
  "confidence": 0.95
}""",
            TaskType.COVERAGE_CHECK: """{
  "covered": true|false,
  "coverage_details": "Specific coverage information",
  "limitations": ["List of limitations"],
  "requirements": ["List of requirements"],
  "justification": "Reasoning with clause references",
  "sources": ["Source 1", "Source 2"]
}""",
            TaskType.COMPLIANCE_AUDIT: """{
  "compliant": true|false,
  "violations": ["List of violations found"],
  "risk_level": "Low|Medium|High|Critical",
  "recommendations": ["List of recommendations"],
  "deadlines": ["List of deadlines"],
  "sources": ["Source 1", "Source 2"]
}""",
            TaskType.POLICY_ANALYSIS: """{
  "summary": "Key policy points",
  "key_terms": ["Important terms"],
  "obligations": ["List of obligations"],
  "rights": ["List of rights"],
  "exceptions": ["List of exceptions"],
  "sources": ["Source 1", "Source 2"]
}""",
            TaskType.DOCUMENT_QA: """{
  "answer": "Direct answer to the question",
  "confidence": 0.95,
  "sources": ["Source 1", "Source 2"],
  "additional_info": "Any additional relevant information"
}""",
            TaskType.LEGAL_REVIEW: """{
  "key_terms": ["Important legal terms"],
  "obligations": ["Legal obligations"],
  "risks": ["Identified risks"],
  "recommendations": ["Legal recommendations"],
  "sources": ["Source 1", "Source 2"]
}""",
            TaskType.HR_ASSESSMENT: """{
  "policy_applicable": true|false,
  "guidelines": ["HR guidelines"],
  "procedures": ["Required procedures"],
  "compliance_status": "Compliant|Non-compliant|Partial",
  "recommendations": ["HR recommendations"],
  "sources": ["Source 1", "Source 2"]
}""",
            TaskType.GENERAL_QA: """{
  "answer": "Direct answer to the question",
  "confidence": 0.95,
  "sources": ["Source 1", "Source 2"],
  "additional_context": "Any additional relevant information"
}"""
        }
        
        return formats.get(task_type, formats[TaskType.GENERAL_QA])
    
    def _get_claim_evaluation_template(self) -> str:
        """Get template for claim evaluation tasks."""
        return """Based on the following insurance policy clauses:

{context}

User Query: {query}

Please evaluate this claim and provide your response in the following JSON format:

{output_format}

Important:
- Reference specific clauses by number (e.g., "Clause 3.2")
- Provide clear justification for your decision
- Include any conditions or requirements
- Cite sources from the provided context"""

    def _get_coverage_check_template(self) -> str:
        """Get template for coverage check tasks."""
        return """Review the following policy documents:

{context}

Query: {query}

Determine if this is covered under the policy and provide your response in this format:

{output_format}

Focus on:
- Specific coverage details
- Any limitations or exclusions
- Required documentation or procedures
- Clear references to policy clauses"""

    def _get_compliance_audit_template(self) -> str:
        """Get template for compliance audit tasks."""
        return """Audit the following documents for compliance:

{context}

Audit Query: {query}

Provide your compliance assessment in this format:

{output_format}

Consider:
- Regulatory requirements
- Policy violations
- Risk assessment
- Required corrective actions
- Specific clause references"""

    def _get_policy_analysis_template(self) -> str:
        """Get template for policy analysis tasks."""
        return """Analyze the following policy documents:

{context}

Analysis Request: {query}

Provide your analysis in this format:

{output_format}

Focus on:
- Key policy terms and definitions
- Rights and obligations
- Important exceptions
- Practical implications
- Clear source references"""

    def _get_document_qa_template(self) -> str:
        """Get template for document Q&A tasks."""
        return """Answer the following question based on the provided documents:

{context}

Question: {query}

Provide your answer in this format:

{output_format}

Guidelines:
- Answer directly and accurately
- Reference specific sources
- Include confidence level
- Provide additional context if relevant"""

    def _get_legal_review_template(self) -> str:
        """Get template for legal review tasks."""
        return """Review the following legal documents:

{context}

Legal Review Request: {query}

Provide your legal analysis in this format:

{output_format}

Consider:
- Key legal terms and definitions
- Legal obligations and rights
- Potential risks and liabilities
- Legal recommendations
- Specific clause references"""

    def _get_hr_assessment_template(self) -> str:
        """Get template for HR assessment tasks."""
        return """Review the following HR policies and documents:

{context}

HR Assessment Request: {query}

Provide your HR assessment in this format:

{output_format}

Focus on:
- Policy applicability
- HR guidelines and procedures
- Compliance status
- Recommendations
- Clear policy references"""

    def _get_general_qa_template(self) -> str:
        """Get template for general Q&A tasks."""
        return """Answer the following question based on the provided context:

{context}

Question: {query}

Provide your answer in this format:

{output_format}

Guidelines:
- Answer accurately and completely
- Reference specific sources
- Include confidence level
- Provide additional context if helpful"""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """Truncate prompt to fit within token limits."""
        estimated_tokens = self._estimate_tokens(prompt)
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # Calculate how much to truncate
        excess_tokens = estimated_tokens - max_tokens
        excess_chars = excess_tokens * 4
        
        # Truncate from the middle (preserve system prompt and end)
        if len(prompt) > excess_chars:
            # Find a good truncation point (end of a sentence)
            truncate_point = len(prompt) - excess_chars - 100  # Leave some buffer
            
            # Try to find a sentence boundary
            for i in range(truncate_point, min(truncate_point + 200, len(prompt))):
                if prompt[i] in '.!?':
                    truncate_point = i + 1
                    break
            
            prompt = prompt[:truncate_point] + "\n\n[Context truncated due to length limits]"
        
        return prompt
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limits."""
        estimated_tokens = self._estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Calculate how much to truncate
        excess_tokens = estimated_tokens - max_tokens
        excess_chars = excess_tokens * 4
        
        if len(text) > excess_chars:
            # Find a good truncation point
            truncate_point = len(text) - excess_chars - 50  # Leave some buffer
            
            # Try to find a sentence boundary
            for i in range(truncate_point, min(truncate_point + 100, len(text))):
                if text[i] in '.!?':
                    truncate_point = i + 1
                    break
            
            text = text[:truncate_point] + " [truncated]"
        
        return text
    
    def create_context_hash(self, context_chunks: List[ContextChunk]) -> str:
        """Create a hash of context chunks for caching."""
        # Create a string representation of chunks
        chunk_data = []
        for chunk in context_chunks:
            chunk_data.append(f"{chunk.source_id}:{chunk.chunk_id}:{chunk.text[:100]}")
        
        context_string = "|".join(chunk_data)
        return hashlib.md5(context_string.encode()).hexdigest()


def create_context_chunks_from_dicts(chunks_data: List[Dict[str, Any]]) -> List[ContextChunk]:
    """Create ContextChunk objects from dictionary data."""
    chunks = []
    for chunk_data in chunks_data:
        chunk = ContextChunk(
            text=chunk_data.get('text', ''),
            source_id=chunk_data.get('source_id'),
            chunk_id=chunk_data.get('chunk_id'),
            page_number=chunk_data.get('page_number'),
            clause_number=chunk_data.get('clause_number'),
            section_title=chunk_data.get('section_title'),
            score=chunk_data.get('score'),
            metadata=chunk_data.get('metadata')
        )
        chunks.append(chunk)
    return chunks


def test_prompt_builder():
    """Test prompt builder functionality."""
    print("Testing prompt builder...")
    
    # Create test context chunks
    chunks = [
        ContextChunk(
            text="Medical expenses are covered up to $10,000 per year. Dental procedures are limited to $2,000 annually.",
            source_id="policy_123",
            clause_number="3.2",
            page_number=15,
            score=0.95
        ),
        ContextChunk(
            text="Pre-existing conditions are excluded from coverage for the first 12 months of the policy.",
            source_id="policy_123",
            clause_number="4.1",
            page_number=18,
            score=0.87
        )
    ]
    
    # Test different task types
    config = PromptConfig(
        task_type=TaskType.CLAIM_EVALUATION,
        model_type=ModelType.CLAUDE,
        max_tokens=4000
    )
    
    builder = PromptBuilder(config)
    
    # Test claim evaluation prompt
    prompt = builder.build_prompt(
        query="I need a root canal treatment that costs $3,500. Is this covered?",
        context_chunks=chunks,
        task_type=TaskType.CLAIM_EVALUATION
    )
    
    print(f"Generated prompt length: {len(prompt)} characters")
    print(f"Estimated tokens: {builder._estimate_tokens(prompt)}")
    print("\nPrompt preview:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    # Test context hash
    context_hash = builder.create_context_hash(chunks)
    print(f"\nContext hash: {context_hash}")
    
    print("Prompt builder test completed successfully!")


if __name__ == "__main__":
    test_prompt_builder()
