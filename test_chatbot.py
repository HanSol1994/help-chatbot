#!/usr/bin/env python3
"""
Simple test script for the AI Documentation Assistant
"""

import os
import tempfile
from io import StringIO
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chatbot import Chatbot

def test_document_processing():
    """Test document processing functionality"""
    print("🧪 Testing document processing...")
    
    # Create a temporary text file
    test_content = """
    # Test Documentation
    
    This is a test document for the AI Documentation Assistant.
    
    ## Features
    - Document processing
    - Vector embeddings
    - Question answering
    
    ## Usage
    Upload your documents and ask questions about them.
    """
    
    # Create a mock uploaded file
    class MockFile:
        def __init__(self, content, name):
            self.content = content.encode()
            self.name = name
        
        def getvalue(self):
            return self.content
    
    mock_file = MockFile(test_content, "test_doc.txt")
    
    try:
        processor = DocumentProcessor()
        documents = processor.process_file(mock_file)
        
        assert len(documents) > 0, "No documents were created"
        assert all(doc.page_content for doc in documents), "Some documents have empty content"
        
        print(f"✅ Successfully processed {len(documents)} document chunks")
        return True
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        return False

def test_vector_store():
    """Test vector store functionality"""
    print("🧪 Testing vector store...")
    
    try:
        vector_store = VectorStore(db_path="./test_vector_db")
        
        # Test search (should return empty initially)
        results = vector_store.similarity_search("test query", k=1)
        
        info = vector_store.get_collection_info()
        print(f"✅ Vector store working. Collection has {info['count']} documents")
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_chatbot():
    """Test chatbot functionality"""
    print("🧪 Testing chatbot...")
    
    try:
        chatbot = Chatbot()
        stats = chatbot.get_stats()
        
        print(f"✅ Chatbot initialized. Model loaded: {stats['model_loaded']}")
        
        # Test response generation
        response = chatbot.get_response("What is this documentation about?")
        assert response, "Chatbot returned empty response"
        
        print(f"✅ Chatbot response generated: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_dirs = ["./test_vector_db"]
    for directory in test_dirs:
        if os.path.exists(directory):
            import shutil
            shutil.rmtree(directory)
            print(f"🧹 Cleaned up {directory}")

def main():
    """Run all tests"""
    print("🚀 Running AI Documentation Assistant tests...\n")
    
    tests = [
        test_vector_store,
        test_document_processing,
        test_chatbot
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    cleanup_test_files()

if __name__ == "__main__":
    main()