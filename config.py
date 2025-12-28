"""
Configuration file for RAG system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_STORE_DIR = KNOWLEDGE_BASE_DIR / "vector_store"

# Ensure directories exist
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# API Keys - Load from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model configurations
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq model (updated)

# Retrieval settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# MCP Server settings
MCP_HOST = "localhost"
MCP_PORT = 8000