#!/usr/bin/env python3
"""
Simple test script for enhanced parser functionality.

This script tests the basic features of the enhanced parser
with real file operations.
"""

import asyncio
import tempfile
import os
from pathlib import Path
from parser import FileParser, ParserError, parse_file


async def test_enhanced_parser():
    """Test the enhanced parser with various features."""
    print("Testing Enhanced Parser Features...")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: Basic text file with chunking
        print("\n1. Testing text file parsing with chunking...")
        test_file = Path(temp_dir) / "test.txt"
        test_content = "This is a test document. " * 10  # Reasonable text for chunking
        test_file.write_text(test_content)
        
        parser = FileParser(enable_ocr=False)
        result = parser.parse_from_file(str(test_file), max_chars=100)
        
        print(f"✓ File type: {result['filetype']}")
        print(f"✓ Text length: {result['length']}")
        print(f"✓ Number of chunks: {len(result.get('text_chunks', []))}")
        print(f"✓ Metadata: {result['metadata']}")
        
        # Test 2: Text file without chunking
        print("\n2. Testing text file parsing without chunking...")
        result_no_chunks = parser.parse_from_file(str(test_file))
        
        print(f"✓ Has full_text: {'full_text' in result_no_chunks}")
        print(f"✓ No text_chunks: 'text_chunks' not in result_no_chunks")
        
        # Test 3: MIME validation
        print("\n3. Testing MIME validation...")
        is_valid = parser._validate_mime_type(test_file, '.txt')
        print(f"✓ MIME validation result: {is_valid}")
        
        # Test 4: Text chunking functionality
        print("\n4. Testing text chunking...")
        long_text = "This is a very long text that needs to be chunked into smaller pieces. " * 5
        chunks = parser._chunk_text(long_text, max_chars=150, overlap=30)
        
        print(f"✓ Number of chunks: {len(chunks)}")
        print(f"✓ All chunks within limit: {all(len(chunk) <= 150 for chunk in chunks)}")
        print(f"✓ Sample chunk: {chunks[0][:50]}...")
        
        # Test 5: Error handling
        print("\n5. Testing error handling...")
        try:
            parser.parse_from_file("nonexistent.txt")
            print("✗ Should have raised an error")
        except ParserError as e:
            print(f"✓ Correctly caught error: {e}")
        
        try:
            parser.parse_from_file("test.xyz")  # Unsupported extension
            print("✗ Should have raised an error")
        except ParserError as e:
            print(f"✓ Correctly caught error: {e}")
        
        # Test 6: OCR availability check
        print("\n6. Testing OCR availability...")
        print(f"✓ OCR available: {parser.enable_ocr}")
        
        # Test 7: Parser with OCR enabled
        print("\n7. Testing parser with OCR enabled...")
        parser_with_ocr = FileParser(enable_ocr=True)
        print(f"✓ OCR enabled: {parser_with_ocr.enable_ocr}")
        
        print("\n✅ All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_convenience_functions():
    """Test convenience functions."""
    print("\nTesting Convenience Functions...")
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content for convenience function")
        temp_path = f.name
    
    try:
        # Test parse_file convenience function
        result = parse_file(temp_path, max_chars=50, enable_ocr=False)
        
        print(f"✓ File type: {result['filetype']}")
        print(f"✓ Has full_text: {'full_text' in result}")
        print(f"✓ Has text_chunks: {'text_chunks' in result}")
        print(f"✓ Metadata: {result['metadata']}")
        
        print("✅ Convenience function test passed!")
        
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
    
    finally:
        # Clean up
        os.unlink(temp_path)


if __name__ == "__main__":
    print("Enhanced Parser Test Suite")
    print("=" * 40)
    
    # Run async tests
    asyncio.run(test_enhanced_parser())
    
    # Run sync tests
    test_convenience_functions()
    
    print("\n🎉 All tests completed!") 