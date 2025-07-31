"""
High-Performance Query Expansion for RAG Systems

This module implements intelligent query expansion techniques to enhance
retrieval performance by generating semantically rich query variants
while maintaining efficiency and semantic integrity.

Author: RAG System Team
License: MIT
"""

import logging
import time
import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict
import re

# Optional imports with graceful fallbacks
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    """Represents an expanded query variant."""
    query: str
    expansion_type: str  # "synonym", "llm", "thesaurus", "abbreviation"
    confidence: float
    source_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'query': self.query,
            'expansion_type': self.expansion_type,
            'confidence': self.confidence,
            'source_terms': self.source_terms
        }


@dataclass
class QueryExpanderConfig:
    """Configuration for query expansion."""
    top_n: int = 3
    use_llm: bool = True
    use_synonyms: bool = True
    use_thesaurus: bool = True
    use_abbreviations: bool = True
    similarity_threshold: float = 0.85
    max_query_length: int = 200
    cache_ttl: int = 3600  # 1 hour
    enable_caching: bool = True
    async_expansion: bool = True
    debug_mode: bool = False


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using LLM."""
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text synchronously."""
        pass


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing and offline mode."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Mock async generation."""
        return self.generate_sync(prompt, max_tokens)
    
    def generate_sync(self, prompt: str, max_tokens: int = 100) -> str:
        """Mock synchronous generation with improved paraphrasing."""
        query = prompt.lower()
        
        # Enhanced rule-based paraphrasing for better demo results
        if "liability" in query and "coverage" in query:
            return "responsibility protection insurance\nobligation safeguard policy\nduty coverage terms"
        elif "claim" in query and "medical" in query:
            return "medical expense reimbursement request\nhealthcare cost application\nmedical treatment payment request"
        elif "premium" in query and "payment" in query:
            return "insurance cost payment terms\npolicy fee payment schedule\ncoverage payment requirements"
        elif "tpa" in query and "requirements" in query:
            return "Third Party Administrator requirements\nTPA processing guidelines\nadministrator requirements for claims"
        elif "insurance" in query:
            return "coverage policy protection\ninsurance terms and conditions\npolicy coverage details"
        elif "claim" in query:
            return "request reimbursement application\nclaim processing requirements\nreimbursement application process"
        elif "premium" in query:
            return "payment cost fee\ninsurance payment terms\npolicy cost requirements"
        else:
            return "related query variant\nsimilar question format\nalternative query phrasing"


class QueryExpander:
    """
    High-performance query expansion engine for RAG systems.
    
    Combines synonym-based, LLM-based, and thesaurus-based expansion
    techniques to enhance retrieval performance while maintaining efficiency.
    """
    
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        config: Optional[QueryExpanderConfig] = None,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        abbreviation_dict: Optional[Dict[str, str]] = None
    ):
        """
        Initialize query expander.
        
        Args:
            llm_client: LLM client for paraphrasing (optional)
            config: Configuration for expansion behavior
            synonym_dict: Domain-specific synonym dictionary
            abbreviation_dict: Domain-specific abbreviation mappings
        """
        self.llm_client = llm_client or MockLLMClient()
        self.config = config or QueryExpanderConfig()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize dictionaries
        self.synonym_dict = synonym_dict or self._load_default_synonyms()
        self.abbreviation_dict = abbreviation_dict or self._load_default_abbreviations()
        
        # Initialize similarity model for deduplication
        self.similarity_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.config.similarity_threshold > 0:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Loaded sentence similarity model for deduplication")
            except Exception as e:
                self.logger.warning(f"Failed to load similarity model: {e}")
        
        # Initialize cache
        self.cache = None
        if self.config.enable_caching and REDIS_AVAILABLE:
            try:
                self.cache = redis.Redis(host='localhost', port=6379, db=0)
                self.logger.info("Connected to Redis cache")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Redis: {e}")
        
        # Performance tracking
        self.stats = {
            'expansion_count': 0,
            'total_expansion_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def expand(
        self, 
        query: str, 
        top_n: Optional[int] = None,
        expansion_types: Optional[List[str]] = None
    ) -> List[ExpandedQuery]:
        """
        Expand a query into multiple semantically related variants.
        
        Args:
            query: Original query string
            top_n: Number of expansions to return (uses config if None)
            expansion_types: Types of expansion to apply (uses all if None)
            
        Returns:
            List[ExpandedQuery]: Ranked list of expanded queries
        """
        start_time = time.time()
        top_n = top_n or self.config.top_n
        expansion_types = expansion_types or ["synonym", "llm", "thesaurus", "abbreviation"]
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, top_n, expansion_types)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            self.logger.info(f"Expanding query: '{query[:100]}...'")
            
            # Generate expansions
            expansions = []
            
            if "synonym" in expansion_types and self.config.use_synonyms:
                synonym_expansions = self._apply_synonym_expansion(query)
                expansions.extend(synonym_expansions)
            
            if "abbreviation" in expansion_types and self.config.use_abbreviations:
                abbrev_expansions = self._apply_abbreviation_expansion(query)
                expansions.extend(abbrev_expansions)
            
            if "thesaurus" in expansion_types and self.config.use_thesaurus:
                thesaurus_expansions = self._apply_thesaurus_expansion(query)
                expansions.extend(thesaurus_expansions)
            
            if "llm" in expansion_types and self.config.use_llm:
                if self.config.async_expansion:
                    llm_expansions = asyncio.run(self._apply_llm_expansion(query))
                else:
                    llm_expansions = self._apply_llm_expansion_sync(query)
                expansions.extend(llm_expansions)
            
            # Deduplicate and rank expansions
            unique_expansions = self._deduplicate_expansions(expansions)
            ranked_expansions = self._rank_expansions(unique_expansions, query)
            
            # Return top_n results
            final_expansions = ranked_expansions[:top_n]
            
            # Cache results
            self._cache_result(cache_key, final_expansions)
            
            # Update stats
            expansion_time = time.time() - start_time
            self.stats['expansion_count'] += 1
            self.stats['total_expansion_time'] += expansion_time
            
            self.logger.info(f"Query expansion completed in {expansion_time:.3f}s, generated {len(final_expansions)} variants")
            
            if self.config.debug_mode:
                self._log_debug_info(query, final_expansions)
            
            return final_expansions
            
        except Exception as e:
            self.logger.error(f"Failed to expand query: {e}")
            # Return original query as fallback
            return [ExpandedQuery(query=query, expansion_type="original", confidence=1.0)]
    
    def _apply_synonym_expansion(self, query: str) -> List[ExpandedQuery]:
        """Apply synonym-based expansion."""
        expansions = []
        words = self._tokenize_query(query.lower())
        
        for word in words:
            if word in self.synonym_dict:
                synonyms = self.synonym_dict[word]
                for synonym in synonyms[:2]:  # Limit to 2 synonyms per word
                    # Replace word with synonym
                    new_words = words.copy()
                    new_words[new_words.index(word)] = synonym
                    expanded_query = " ".join(new_words)
                    
                    if expanded_query != query:
                        expansions.append(ExpandedQuery(
                            query=expanded_query,
                            expansion_type="synonym",
                            confidence=0.8,
                            source_terms=[word, synonym]
                        ))
        
        return expansions
    
    def _apply_abbreviation_expansion(self, query: str) -> List[ExpandedQuery]:
        """Apply abbreviation expansion."""
        expansions = []
        words = self._tokenize_query(query)
        
        for i, word in enumerate(words):
            # Check for abbreviations
            if word.upper() in self.abbreviation_dict:
                expansion = self.abbreviation_dict[word.upper()]
                new_words = words.copy()
                new_words[i] = expansion
                expanded_query = " ".join(new_words)
                
                expansions.append(ExpandedQuery(
                    query=expanded_query,
                    expansion_type="abbreviation",
                    confidence=0.9,
                    source_terms=[word, expansion]
                ))
        
        return expansions
    
    def _apply_thesaurus_expansion(self, query: str) -> List[ExpandedQuery]:
        """Apply thesaurus-based expansion using internal synonym dictionary."""
        expansions = []
        words = self._tokenize_query(query.lower())
        
        for word in words:
            # Get synonyms from internal dictionary
            if word in self.synonym_dict:
                synonyms = self.synonym_dict[word]
                for synonym in synonyms[:2]:  # Limit to 2 synonyms per word
                    # Replace word with synonym
                    new_words = words.copy()
                    if word in new_words:
                        new_words[new_words.index(word)] = synonym
                        expanded_query = " ".join(new_words)
                        
                        if expanded_query != query:
                            expansions.append(ExpandedQuery(
                                query=expanded_query,
                                expansion_type="thesaurus",
                                confidence=0.7,
                                source_terms=[word, synonym]
                            ))
        
        return expansions
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Simple but effective tokenizer that handles punctuation."""
        # Remove extra whitespace and split on word boundaries
        # This handles punctuation and special characters effectively
        tokens = re.findall(r'\b\w+\b', query.lower())
        return [token for token in tokens if len(token) > 1]  # Filter out single characters
    
    async def _apply_llm_expansion(self, query: str) -> List[ExpandedQuery]:
        """Apply LLM-based expansion asynchronously."""
        try:
            prompt = self._create_expansion_prompt(query)
            response = await self.llm_client.generate(prompt, max_tokens=150)
            
            # Parse LLM response
            expansions = self._parse_llm_response(response, query)
            return expansions
            
        except Exception as e:
            self.logger.error(f"Failed to apply LLM expansion: {e}")
            return []
    
    def _apply_llm_expansion_sync(self, query: str) -> List[ExpandedQuery]:
        """Apply LLM-based expansion synchronously."""
        try:
            prompt = self._create_expansion_prompt(query)
            response = self.llm_client.generate_sync(prompt, max_tokens=150)
            
            # Parse LLM response
            expansions = self._parse_llm_response(response, query)
            return expansions
            
        except Exception as e:
            self.logger.error(f"Failed to apply LLM expansion: {e}")
            return []
    
    def _create_expansion_prompt(self, query: str) -> str:
        """Create prompt for LLM-based expansion."""
        return f"""
        Generate 2-3 different ways to ask the same question, focusing on insurance and legal terminology.
        Keep each variant concise and semantically equivalent.
        
        Original query: "{query}"
        
        Generate variants that:
        1. Use different but related terms
        2. Maintain the same intent
        3. Are suitable for document retrieval
        4. Include domain-specific terminology when relevant
        
        Return only the variants, one per line:
        """
    
    def _parse_llm_response(self, response: str, original_query: str) -> List[ExpandedQuery]:
        """Parse LLM response into ExpandedQuery objects."""
        expansions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and line != original_query and len(line) < self.config.max_query_length:
                # Remove numbering if present
                line = re.sub(r'^\d+\.\s*', '', line)
                
                expansions.append(ExpandedQuery(
                    query=line,
                    expansion_type="llm",
                    confidence=0.75,
                    source_terms=[original_query]
                ))
        
        return expansions[:3]  # Limit to 3 LLM expansions
    
    def _deduplicate_expansions(self, expansions: List[ExpandedQuery]) -> List[ExpandedQuery]:
        """Remove duplicate or very similar expansions."""
        if not expansions:
            return []
        
        unique_expansions = [expansions[0]]
        
        for expansion in expansions[1:]:
            is_duplicate = False
            
            for existing in unique_expansions:
                if self._are_similar(expansion.query, existing.query):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_expansions.append(expansion)
        
        return unique_expansions
    
    def _are_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are semantically similar."""
        if query1.lower() == query2.lower():
            return True
        
        if self.similarity_model:
            try:
                embeddings = self.similarity_model.encode([query1, query2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return similarity > self.config.similarity_threshold
            except Exception as e:
                self.logger.debug(f"Failed to compute similarity: {e}")
        
        # Fallback: simple word overlap
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        return overlap > 0.8
    
    def _rank_expansions(self, expansions: List[ExpandedQuery], original_query: str) -> List[ExpandedQuery]:
        """Rank expansions by relevance and quality."""
        # Sort by confidence first, then by length (prefer shorter queries)
        ranked = sorted(expansions, key=lambda x: (x.confidence, -len(x.query)), reverse=True)
        return ranked
    
    def _generate_cache_key(self, query: str, top_n: int, expansion_types: List[str]) -> str:
        """Generate cache key for query expansion."""
        key_data = {
            'query': query.lower().strip(),
            'top_n': top_n,
            'expansion_types': sorted(expansion_types)
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"query_expansion:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[ExpandedQuery]]:
        """Get expansion result from cache."""
        if not self.cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return [ExpandedQuery(**item) for item in data]
        except Exception as e:
            self.logger.debug(f"Failed to get from cache: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, expansions: List[ExpandedQuery]) -> None:
        """Cache expansion result."""
        if not self.cache:
            return
        
        try:
            data = [exp.to_dict() for exp in expansions]
            self.cache.setex(
                cache_key,
                self.config.cache_ttl,
                json.dumps(data)
            )
        except Exception as e:
            self.logger.debug(f"Failed to cache result: {e}")
    
    def _load_default_synonyms(self) -> Dict[str, List[str]]:
        """Load comprehensive insurance/legal synonym dictionary."""
        return {
            # Insurance terms
            "insurance": ["coverage", "policy", "protection", "safeguard"],
            "claim": ["request", "application", "reimbursement", "petition"],
            "premium": ["payment", "cost", "fee", "charge"],
            "deductible": ["excess", "threshold", "minimum", "out-of-pocket"],
            "coverage": ["protection", "insurance", "safeguard", "shield"],
            "policy": ["contract", "agreement", "terms", "document"],
            "liability": ["responsibility", "obligation", "duty", "accountability"],
            "damage": ["loss", "harm", "injury", "destruction"],
            "accident": ["incident", "event", "occurrence", "mishap"],
            "benefit": ["advantage", "compensation", "payment", "entitlement"],
            "exclusion": ["exception", "omission", "limitation", "restriction"],
            "endorsement": ["amendment", "modification", "addition", "rider"],
            "underwriting": ["evaluation", "assessment", "review", "analysis"],
            "risk": ["hazard", "danger", "threat", "peril"],
            
            # Legal terms
            "contract": ["agreement", "document", "terms", "covenant"],
            "clause": ["provision", "section", "term", "stipulation"],
            "obligation": ["duty", "responsibility", "requirement", "commitment"],
            "breach": ["violation", "infringement", "non-compliance", "default"],
            "termination": ["ending", "cancellation", "expiration", "cessation"],
            "amendment": ["modification", "change", "revision", "alteration"],
            "enforcement": ["implementation", "application", "execution", "imposition"],
            "compliance": ["adherence", "conformity", "observance", "following"],
            "waiver": ["relinquishment", "surrender", "abandonment", "release"],
            "indemnification": ["compensation", "reimbursement", "protection", "security"],
            
            # Financial terms
            "payment": ["remittance", "settlement", "disbursement", "transfer"],
            "reimbursement": ["compensation", "repayment", "refund", "restoration"],
            "cost": ["expense", "charge", "fee", "price"],
            "amount": ["sum", "total", "figure", "quantity"],
            "rate": ["percentage", "ratio", "proportion", "scale"],
            "premium": ["payment", "cost", "fee", "charge"],
            "deductible": ["excess", "threshold", "minimum", "out-of-pocket"],
            "copay": ["copayment", "cost-sharing", "contribution", "portion"],
            "coinsurance": ["cost-sharing", "percentage", "portion", "split"],
            
            # Medical terms
            "medical": ["healthcare", "clinical", "therapeutic", "health"],
            "treatment": ["therapy", "care", "intervention", "procedure"],
            "diagnosis": ["assessment", "evaluation", "examination", "analysis"],
            "prescription": ["medication", "drug", "medicine", "remedy"],
            "hospital": ["medical center", "clinic", "facility", "institution"],
            "physician": ["doctor", "provider", "practitioner", "clinician"],
            "specialist": ["expert", "consultant", "professional", "practitioner"],
            "procedure": ["treatment", "operation", "intervention", "surgery"],
            "therapy": ["treatment", "rehabilitation", "intervention", "care"],
            
            # General terms for better coverage
            "provide": ["offer", "supply", "furnish", "deliver"],
            "require": ["need", "demand", "necessitate", "mandate"],
            "include": ["contain", "comprise", "encompass", "incorporate"],
            "exclude": ["omit", "remove", "eliminate", "bar"],
            "cover": ["protect", "insure", "safeguard", "shield"],
            "file": ["submit", "present", "lodge", "register"],
            "process": ["handle", "manage", "deal with", "administer"],
            "review": ["examine", "assess", "evaluate", "analyze"],
            "approve": ["authorize", "sanction", "endorse", "accept"],
            "deny": ["reject", "refuse", "decline", "disapprove"]
        }
    
    def _load_default_abbreviations(self) -> Dict[str, str]:
        """Load comprehensive abbreviation mappings."""
        return {
            # Insurance abbreviations
            "TPA": "Third Party Administrator",
            "HMO": "Health Maintenance Organization",
            "PPO": "Preferred Provider Organization",
            "POS": "Point of Service",
            "EPO": "Exclusive Provider Organization",
            "HDHP": "High Deductible Health Plan",
            "FSA": "Flexible Spending Account",
            "HSA": "Health Savings Account",
            "COBRA": "Consolidated Omnibus Budget Reconciliation Act",
            "HIPAA": "Health Insurance Portability and Accountability Act",
            "ACA": "Affordable Care Act",
            "PPACA": "Patient Protection and Affordable Care Act",
            "CMS": "Centers for Medicare and Medicaid Services",
            "FDA": "Food and Drug Administration",
            "CDC": "Centers for Disease Control and Prevention",
            
            # Legal abbreviations
            "LLC": "Limited Liability Company",
            "LLP": "Limited Liability Partnership",
            "Inc": "Incorporated",
            "Corp": "Corporation",
            "Ltd": "Limited",
            "Esq": "Esquire",
            "Atty": "Attorney",
            "CPA": "Certified Public Accountant",
            "JD": "Juris Doctor",
            "BA": "Bachelor of Arts",
            "BS": "Bachelor of Science",
            "MA": "Master of Arts",
            "MS": "Master of Science",
            "PhD": "Doctor of Philosophy",
            "MBA": "Master of Business Administration",
            
            # Medical abbreviations
            "ER": "Emergency Room",
            "ICU": "Intensive Care Unit",
            "MRI": "Magnetic Resonance Imaging",
            "CT": "Computed Tomography",
            "X-ray": "X-Ray",
            "Rx": "Prescription",
            "Dr": "Doctor",
            "MD": "Medical Doctor",
            "RN": "Registered Nurse",
            "LPN": "Licensed Practical Nurse",
            "PA": "Physician Assistant",
            "NP": "Nurse Practitioner",
            "CRNA": "Certified Registered Nurse Anesthetist",
            "PT": "Physical Therapy",
            "OT": "Occupational Therapy",
            "ST": "Speech Therapy",
            "CNA": "Certified Nursing Assistant",
            "EMT": "Emergency Medical Technician",
            "Paramedic": "Emergency Medical Technician Paramedic",
            
            # Financial abbreviations
            "APR": "Annual Percentage Rate",
            "APY": "Annual Percentage Yield",
            "IRA": "Individual Retirement Account",
            "401k": "401(k) Retirement Plan",
            "ROI": "Return on Investment",
            "APR": "Annual Percentage Rate",
            "EFT": "Electronic Funds Transfer",
            "ACH": "Automated Clearing House",
            "PIN": "Personal Identification Number",
            "SSN": "Social Security Number",
            "EIN": "Employer Identification Number",
            "TIN": "Taxpayer Identification Number"
        }
    
    def _log_debug_info(self, original_query: str, expansions: List[ExpandedQuery]):
        """Log debug information for fine-tuning."""
        self.logger.debug(f"Original query: {original_query}")
        self.logger.debug(f"Generated {len(expansions)} expansions:")
        for i, exp in enumerate(expansions):
            self.logger.debug(f"  {i+1}. {exp.query} ({exp.expansion_type}, conf: {exp.confidence:.2f})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        if stats['expansion_count'] > 0:
            stats['avg_expansion_time'] = stats['total_expansion_time'] / stats['expansion_count']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        return stats
    
    def clear_stats(self) -> None:
        """Clear performance statistics."""
        self.stats = {
            'expansion_count': 0,
            'total_expansion_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }


# Factory functions for easy usage
def create_query_expander(
    llm_client: Optional[BaseLLMClient] = None,
    use_llm: bool = True,
    top_n: int = 3,
    **kwargs
) -> QueryExpander:
    """
    Create a query expander with default configuration.
    
    Args:
        llm_client: LLM client for paraphrasing
        use_llm: Enable LLM-based expansion
        top_n: Number of expansions to generate
        **kwargs: Additional configuration parameters
        
    Returns:
        QueryExpander: Configured query expander instance
    """
    config = QueryExpanderConfig(
        use_llm=use_llm,
        top_n=top_n,
        **kwargs
    )
    
    return QueryExpander(llm_client=llm_client, config=config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create query expander
    expander = create_query_expander(
        use_llm=True,
        top_n=3,
        debug_mode=True
    )
    
    # Test queries
    test_queries = [
        "What is the liability coverage for auto insurance?",
        "How do I file a claim for medical expenses?",
        "What are the premium payment terms?",
        "TPA requirements for claims processing"
    ]
    
    for query in test_queries:
        print(f"\nExpanding query: '{query}'")
        expansions = expander.expand(query)
        
        print(f"Generated {len(expansions)} expansions:")
        for i, exp in enumerate(expansions):
            print(f"  {i+1}. {exp.query}")
            print(f"     Type: {exp.expansion_type}, Confidence: {exp.confidence:.2f}")
            if exp.source_terms:
                print(f"     Source terms: {exp.source_terms}")
        print()
    
    # Get stats
    print(f"Query expander stats: {expander.get_stats()}")
