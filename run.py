#!/usr/bin/env python3
"""
Simple runner script for the AI Documentation Assistant
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import chromadb
        import sentence_transformers
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'documents', 'vector_db']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def main():
    print("🤖 Starting AI Documentation Assistant...")
    
    if not check_requirements():
        sys.exit(1)
    
    create_directories()
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("⚠️  No .env file found. Using default configuration.")
        print("💡 Copy .env.example to .env to customize settings.")
    
    print("🚀 Launching Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])

if __name__ == "__main__":
    main()