"""
RAG Pipeline Implementation
Handles document loading, chunking, embedding generation, and retrieval
"""
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from groq import Groq

import config


class RAGPipeline:
    """
    RAG Pipeline for document ingestion, retrieval, and generation
    """
    
    def __init__(self, silent=False):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm_client = None
        self.silent = silent
        
        self._initialize_embeddings()
        self._initialize_llm()
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings model"""
        if not self.silent:
            print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        if not self.silent:
            print("Embeddings initialized successfully")
    
    def _initialize_llm(self):
        """Initialize Groq LLM client"""
        if not config.GROQ_API_KEY:
            if not self.silent:
                print("Warning: GROQ_API_KEY not set. LLM features will be limited.")
            return
        
        self.llm_client = Groq(api_key=config.GROQ_API_KEY)
        if not self.silent:
            print("Groq LLM client initialized")
    
    def load_documents(self) -> List[Document]:
        """Load documents from knowledge base directory"""
        print(f"Loading documents from {config.KNOWLEDGE_BASE_DIR}")
        
        # Load markdown and text files
        loader = DirectoryLoader(
            str(config.KNOWLEDGE_BASE_DIR),
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for embedding"""
        print("Chunking documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vector store from document chunks"""
        print("Creating vector store...")
        
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Save to disk
        self.vectorstore.save_local(str(config.VECTOR_STORE_DIR))
        print(f"Vector store saved to {config.VECTOR_STORE_DIR}")
        
        return self.vectorstore
    
    def load_vectorstore(self) -> FAISS:
        """Load existing vector store from disk"""
        if not self.silent:
            print("Loading vector store from disk...")
        
        self.vectorstore = FAISS.load_local(
            str(config.VECTOR_STORE_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        if not self.silent:
            print("Vector store loaded successfully")
        return self.vectorstore
    
    def setup_retriever(self):
        """Initialize retriever from vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_vectorstore first.")
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.TOP_K_RESULTS}
        )
        if not self.silent:
            print("Retriever initialized")
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if not self.retriever:
            self.setup_retriever()
        
        results = self.retriever.invoke(query)
        return results
    
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer using retrieved context and LLM"""
        if not self.llm_client:
            return "LLM not initialized. Please set GROQ_API_KEY environment variable."
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
        
        # Call Groq API
        try:
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def search_and_summarize(self, query: str) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve and generate"""
        print(f"\nProcessing query: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        
        # Generate answer using LLM
        answer = self.generate_answer(query, retrieved_docs)
        
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ]
        }


def initialize_knowledge_base():
    """Initialize the knowledge base (run once)"""
    pipeline = RAGPipeline()
    
    # Load and process documents
    documents = pipeline.load_documents()
    chunks = pipeline.chunk_documents(documents)
    
    # Create vector store
    pipeline.create_vectorstore(chunks)
    
    print("\nKnowledge base initialized successfully!")
    return pipeline


if __name__ == "__main__":
    # Example: Initialize knowledge base
    print("=== Initializing Knowledge Base ===")
    pipeline = initialize_knowledge_base()
    
    # Example: Test retrieval
    print("\n=== Testing Retrieval ===")
    test_query = "What is the company's vacation policy?"
    result = pipeline.search_and_summarize(test_query)
    
    print(f"\nQuery: {result['query']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} documents retrieved")