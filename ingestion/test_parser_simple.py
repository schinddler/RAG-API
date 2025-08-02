"""
Simple test script for the parser module.
"""

import asyncio
import tempfile
import os
from pathlib import Path

from parser import FileParser, parse_file, ParserError


async def test_parser():
    """Test the parser with a simple text file."""
    
    print("=== Testing Parser Module ===")
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document for the parser module.\nIt contains multiple lines of text.\nThis should be parsed correctly.")
        test_file = f.name
    
    try:
        print(f"Created test file: {test_file}")
        
        # Test 1: Parse using convenience function
        print("\n--- Test 1: Convenience Function ---")
        result = parse_file(test_file)
        print(f"Result: {result}")
        print(f"Text length: {result['length']}")
        print(f"File type: {result['filetype']}")
        print(f"First 100 chars: {result['text'][:100]}")
        
        # Test 2: Parse using FileParser class
        print("\n--- Test 2: FileParser Class ---")
        parser = FileParser()
        result2 = parser.parse_from_file(test_file)
        print(f"Result: {result2}")
        
        # Test 3: Test error handling
        print("\n--- Test 3: Error Handling ---")
        try:
            parse_file("/nonexistent/file.txt")
        except ParserError as e:
            print(f"Expected error caught: {e}")
        
        # Test 4: Test unsupported file type
        print("\n--- Test 4: Unsupported File Type ---")
        unsupported_file = test_file.replace('.txt', '.xyz')
        with open(unsupported_file, 'w') as f:
            f.write("content")
        
        try:
            parse_file(unsupported_file)
        except ParserError as e:
            print(f"Expected error caught: {e}")
        finally:
            os.unlink(unsupported_file)
        
        print("\n=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == "__main__":
    asyncio.run(test_parser()) 