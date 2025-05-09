#!/usr/bin/env python3
"""Example script to test the LangChain server."""
import requests
import json
import sys

def test_server(text, url="http://localhost:8000/process"):
    """Test the LangChain server with a text query."""
    try:
        # Send a request to the server
        response = requests.post(
            url,
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        
        # Print the response
        result = response.json()
        print(f"Query: {text}")
        print(f"Response: {result['response']}")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Get the query from command line arguments or use a default
    query = sys.argv[1] if len(sys.argv) > 1 else "Turn on the living room lights"
    
    # Test the server
    test_server(query)