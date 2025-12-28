Perfect — this clarifies it very well.
Below is a **fresh, from-scratch README** written in a **high-level, professional format**, with:

* Clear **structure**
* **Appropriate, restrained emojis only in headers**
* A **clean, correct architecture diagram**
* Tone of a **junior engineer who understands the system deeply**
* Nothing flashy, nothing “marketing”
* Everything a **proper README is expected to contain**

You can directly use this.

---

# RAG System with MCP Integration

A Retrieval-Augmented Generation (RAG) system that enables users to query a local knowledge base through a conversational AI agent.
The system retrieves relevant document context using vector similarity search and generates grounded responses using a large language model.

This project focuses on **correct RAG architecture**, **clear separation of responsibilities**, and **transparent system behavior**.

---

## 📌 Project Overview

This system allows users to ask natural language questions about internal documents (such as company policies).
Instead of relying on the language model’s internal knowledge, the system retrieves relevant information from a vector database and uses it as context for answer generation.

The goal of this project is to demonstrate:

* A clean RAG pipeline
* Deterministic and explainable responses
* A production-aligned folder structure
* MCP-compatible design for future agent integrations

---

## 🧠 What the System Does

1. Loads documents from a local knowledge base
2. Splits documents into overlapping chunks
3. Converts chunks into vector embeddings
4. Stores embeddings in a FAISS vector index
5. Retrieves the most relevant chunks for a user query
6. Generates an answer strictly based on retrieved content
7. Maintains conversational continuity across turns

If the information is not present in the knowledge base, the system clearly states that.

---

## 🏗️ System Architecture

### High-Level Architecture Diagram

```
┌────────────────────────────────────────────┐
│                  User                      │
│        (Natural Language Questions)        │
└───────────────────────────┬────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────┐
│           Conversational Agent              │
│                (agent.py)                  │
│                                            │
│  • Handles user interaction                │
│  • Maintains conversation history          │
│  • Sends queries to the RAG pipeline       │
└───────────────────────────┬────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────┐
│             RAG Pipeline                   │
│           (rag_pipeline.py)                │
│                                            │
│  • Document loading                        │
│  • Text chunking                           │
│  • Embedding generation                   │
│  • Similarity-based retrieval              │
└───────────────────────────┬────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌────────────────────────────┐   ┌──────────────────────────┐
│        FAISS Vector DB      │   │        Groq LLM           │
│                            │   │  (Llama 3.3 70B)          │
│  • Stores embeddings       │   │  • Generates answers      │
│  • Similarity search       │   │  • Uses retrieved context │
└────────────────────────────┘   └──────────────────────────┘
```

---

## 🧩 Architectural Responsibilities

| Component            | Responsibility                                                      |
| -------------------- | ------------------------------------------------------------------- |
| Agent                | Manages conversation flow and user interaction                      |
| RAG Pipeline         | Loads documents, chunks text, creates embeddings, retrieves context |
| Vector Store (FAISS) | Performs semantic similarity search                                 |
| LLM (Groq)           | Generates responses using retrieved context only                    |
| Knowledge Base       | Source of truth for all answers                                     |

---

## 📂 Project Structure

```
task3-rag-system/
├── agent.py                 # Conversational RAG agent
├── rag_pipeline.py          # Core RAG logic
├── mcp_server.py            # MCP-compatible server (STDIO)
├── config.py                # Central configuration
├── test_system.py           # System validation
├── requirements.txt         # Dependencies
├── README.md                # Documentation
├── .env                     # Environment variables
└── knowledge_base/
    ├── company_policies.md  # Sample knowledge base
    └── vector_store/        # Auto-generated FAISS index
        ├── index.faiss
        └── index.pkl
```

---

## ⚙️ Configuration

All major parameters are defined in `config.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3
```

This keeps system behavior explicit and easy to modify.

---

## 🚀 Setup and Execution

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

### 3. Initialize the Knowledge Base

Add documents to `knowledge_base/` and run:

```bash
python rag_pipeline.py
```

This step:

* Loads documents
* Splits them into chunks
* Generates embeddings
* Builds the FAISS vector index

---

### 4. Run the Agent

```bash
python agent.py
```

---

## 💬 Example Usage

```
Initializing RAG system...
Knowledge base loaded
Ready!

You: What is the leave policy?

Agent: The company provides 12 days of casual leave,
10 days of sick leave, and 15 days of earned leave.
Only earned leave can be carried forward.
```

If a question is outside the knowledge base:

```
You: What is the cafeteria menu?

Agent: The company policies do not mention any information
about the cafeteria or its menu.
```

---

## 🔄 Retrieval and Answer Generation Flow

1. User query is converted into an embedding
2. FAISS retrieves the most similar document chunks
3. Retrieved chunks are combined with conversation history
4. LLM generates a response grounded in retrieved content

This ensures minimal hallucination and transparent behavior.

---

## 🧪 Testing

Run the system tests:

```bash
python test_system.py
```

The tests verify:

* Environment configuration
* Vector store availability
* RAG pipeline execution
* MCP server initialization

---

## 🔌 MCP Integration

The project includes an MCP-compatible server using STDIO transport.

```bash
python mcp_server.py
```

This allows the system to integrate with MCP inspectors or future agent frameworks.

---

## 🎯 Design Choices

* Exact vector search for transparency
* Small overlapping chunks to preserve context
* Low-temperature LLM for factual responses
* No speculative answers beyond retrieved data

These choices prioritize correctness and explainability.

---

## 🚧 Known Limitations

* Answers are limited to available documents
* No citation rendering (can be added later)
* Designed for local knowledge bases

---

## 🔮 Future Improvements

* PDF and web document loaders
* Re-ranking or hybrid retrieval strategies
* Source-level citations
* Integration with LangGraph-based planners
* UI or API layer on top of the agent

---

## 📚 Resources

- [LangChain Docs](https://python.langchain.com/)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [FAISS Documentation](https://faiss.ai/)
- [Groq API](https://console.groq.com/docs)
- [Sentence Transformers](https://www.sbert.net/)

---
