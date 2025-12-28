"""
Test script for RAG system
Validates all components work correctly
"""
import os
from rag_pipeline import RAGPipeline
import config


def test_environment():
    """Test environment setup"""
    print("=" * 60)
    print("TESTING ENVIRONMENT")
    print("=" * 60)
    
    # Check API key
    if config.GROQ_API_KEY:
        print("GROQ_API_KEY is set")
    else:
        print("GROQ_API_KEY not set")
        print("  Set it with: export GROQ_API_KEY='your_key'")
        return False
    
    # Check knowledge base
    if config.KNOWLEDGE_BASE_DIR.exists():
        files = list(config.KNOWLEDGE_BASE_DIR.glob("*.md"))
        print(f"Knowledge base directory exists ({len(files)} .md files)")
    else:
        print("Knowledge base directory not found")
        return False
    
    # Check vector store
    if config.VECTOR_STORE_DIR.exists():
        print("Vector store exists")
    else:
        print("Vector store not found")
        print("  Run: python rag_pipeline.py")
        return False
    
    print()
    return True


def test_rag_pipeline():
    """Test RAG pipeline components"""
    print("=" * 60)
    print("TESTING RAG PIPELINE")
    print("=" * 60)
    
    try:
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = RAGPipeline()
        print("Pipeline initialized")
        
        # Load vector store
        print("Loading vector store...")
        pipeline.load_vectorstore()
        print("Vector store loaded")
        
        # Setup retriever
        print("Setting up retriever...")
        pipeline.setup_retriever()
        print("Retriever ready")
        
        # Test retrieval
        print("\nTesting retrieval...")
        test_query = "vacation policy"
        results = pipeline.retrieve(test_query)
        print(f"Retrieved {len(results)} documents")
        
        # Test generation
        print("\nTesting answer generation...")
        result = pipeline.search_and_summarize(test_query)
        print(f"Generated answer ({len(result['answer'])} chars)")
        print(f"\nSample: {result['answer'][:100]}...")
        
        print("\nALL PIPELINE TESTS PASSED")
        return True
    
    except Exception as e:
        print(f"\nPipeline test failed: {e}")
        return False


def test_mcp_protocol():
    """Test MCP server protocol"""
    print("\n" + "=" * 60)
    print("TESTING MCP SERVER")
    print("=" * 60)
    
    import json
    import subprocess
    
    try:
        # Test initialize request
        print("Testing initialize request...")
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }
        
        proc = subprocess.Popen(
            ["python", "mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = proc.communicate(json.dumps(request) + "\n", timeout=10)
        
        if stdout:
            response = json.loads(stdout.strip())
            if "result" in response:
                print("MCP server responds correctly")
                return True
        
        print("MCP server response invalid")
        return False
    
    except Exception as e:
        print(f"MCP test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RAG SYSTEM TEST SUITE")
    print("=" * 60 + "\n")
    
    tests_passed = 0
    total_tests = 3
    
    # Test environment
    if test_environment():
        tests_passed += 1
    
    # Test RAG pipeline
    if test_rag_pipeline():
        tests_passed += 1
    
    # Test MCP protocol
    if test_mcp_protocol():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    print("=" * 60 + "\n")
    
    if tests_passed == total_tests:
        print("🎉 All systems operational!")
        print("\nNext steps:")
        print("  - Run agent: python agent.py")
        print("  - Test MCP: npx @modelcontextprotocol/inspector python mcp_server.py")
    else:
        print("⚠️  Some tests failed. Please fix issues above.")


if __name__ == "__main__":
    main()