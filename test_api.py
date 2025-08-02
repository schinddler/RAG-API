"""
Test script for HackRx6 RAG API.

This script provides examples of how to use the API endpoints
and can be used for testing the system functionality.
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any


class HackRxAPIClient:
    """Client for testing HackRx6 RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_token: str = "c742772b47bb55597517747abafcc3d472fa1c4403a1574461aa3f70ea2d9301"):
        self.base_url = base_url
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health", headers=self.headers)
            return response.json()
    
    async def get_api_info(self) -> Dict[str, Any]:
        """Get API information."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/api/v1/info", headers=self.headers)
            return response.json()
    
    async def process_document_and_questions(
        self, 
        document_url: str, 
        questions: list
    ) -> Dict[str, Any]:
        """
        Process document and answer questions.
        
        Args:
            document_url: Blob URL of the document
            questions: List of questions to answer
            
        Returns:
            API response with answers
        """
        payload = {
            "documents": document_url,
            "questions": questions
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/hackrx/run",
                headers=self.headers,
                json=payload,
                timeout=60.0  # 60 second timeout
            )
            return response.json()


async def test_api_functionality():
    """Test the API functionality."""
    print("🧪 Testing HackRx6 RAG API...")
    
    # Initialize client
    client = HackRxAPIClient()
    
    try:
        # Test 1: Health check
        print("\n1. Testing health check...")
        health_response = await client.health_check()
        print(f"✅ Health check: {health_response}")
        
        # Test 2: API info
        print("\n2. Testing API info...")
        info_response = await client.get_api_info()
        print(f"✅ API info: {info_response}")
        
        # Test 3: Document processing (with sample document URL)
        print("\n3. Testing document processing...")
        
        # Sample questions for testing
        test_questions = [
            "What is the grace period for premium payment under the policy?",
            "Does this policy cover maternity expenses?",
            "What are the coverage limits for dental procedures?"
        ]
        
        # Note: You'll need to provide a real document URL for this test
        # For now, we'll use a placeholder
        sample_document_url = "https://example.com/sample-policy.pdf"
        
        print(f"📄 Document URL: {sample_document_url}")
        print(f"❓ Questions: {test_questions}")
        
        try:
            response = await client.process_document_and_questions(
                sample_document_url, 
                test_questions
            )
            print(f"✅ Document processing response: {json.dumps(response, indent=2)}")
        except httpx.HTTPStatusError as e:
            print(f"⚠️ Expected error (no real document): {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        
        print("\n🎉 API testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")


async def test_with_real_document(document_url: str):
    """Test with a real document URL."""
    print(f"🧪 Testing with real document: {document_url}")
    
    client = HackRxAPIClient()
    
    # Real questions for insurance policy analysis
    questions = [
        "What is the grace period for premium payment?",
        "What medical procedures are covered under this policy?",
        "What are the coverage limits for dental procedures?",
        "Are pre-existing conditions covered?",
        "What is the deductible amount?"
    ]
    
    try:
        start_time = time.time()
        response = await client.process_document_and_questions(document_url, questions)
        processing_time = time.time() - start_time
        
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📊 Response: {json.dumps(response, indent=2)}")
        
        # Analyze results
        if 'answers' in response:
            print(f"\n📋 Results Summary:")
            print(f"   - Questions processed: {len(response['answers'])}")
            print(f"   - Total processing time: {response.get('total_processing_time_ms', 0)}ms")
            
            for i, answer in enumerate(response['answers'], 1):
                print(f"\n   {i}. Question: {answer['question']}")
                print(f"      Answer: {answer['answer'][:100]}...")
                print(f"      Confidence: {answer['confidence']}")
                print(f"      Sources: {len(answer['source_clauses'])} clauses")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def print_usage_examples():
    """Print usage examples for the API."""
    print("""
📚 HackRx6 RAG API Usage Examples:

1. Health Check:
   curl -X GET "http://localhost:8000/health"

2. API Information:
   curl -X GET "http://localhost:8000/api/v1/info" \\
        -H "Authorization: Bearer c742772b47bb55597517747abafcc3d472fa1c4403a1574461aa3f70ea2d9301"

3. Process Document and Answer Questions:
   curl -X POST "http://localhost:8000/api/v1/hackrx/run" \\
        -H "Authorization: Bearer c742772b47bb55597517747abafcc3d472fa1c4403a1574461aa3f70ea2d9301" \\
        -H "Content-Type: application/json" \\
        -d '{
          "documents": "https://example.com/policy.pdf",
          "questions": [
            "What is the grace period for premium payment?",
            "Does this policy cover maternity expenses?"
          ]
        }'

4. Interactive Documentation:
   Open http://localhost:8000/docs in your browser

5. ReDoc Documentation:
   Open http://localhost:8000/redoc in your browser
""")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print_usage_examples()
    elif len(sys.argv) > 1:
        # Test with real document URL
        document_url = sys.argv[1]
        asyncio.run(test_with_real_document(document_url))
    else:
        # Run basic tests
        asyncio.run(test_api_functionality())
        print_usage_examples() 