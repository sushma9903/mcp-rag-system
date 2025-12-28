# RAG System with MCP Integration

A production-ready Retrieval-Augmented Generation system implementing semantic search, LLM-powered question answering, and Model Context Protocol (MCP) integration.

## Features

- **Semantic Search**: FAISS vector store with HuggingFace embeddings
- **AI-Powered Answers**: Groq API (Llama 3.3 70B) for intelligent responses
- **MCP Protocol**: STDIO-based server for tool integration
- **Conversational Agent**: Interactive chat with context memory
- **Source Attribution**: Transparent citations for all answers

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from [console.groq.com](https://console.groq.com)

### 3. Add Knowledge Base

Create `knowledge_base/company_policies.md` with your content:

```markdown
# Company Policies

## Leave Policy
- 12 days of Casual Leave per year
- 10 days of Sick Leave per year
- 15 days of Earned Leave per year

## Remote Work Policy
Remote employees must be available during core hours (10 AM - 4 PM IST).
```

### 4. Initialize Vector Store

```bash
python rag_pipeline.py
```

### 5. Run the Agent

```bash
python agent.py
```

## Usage

### Interactive Mode

```bash
python agent.py
```

**Example Session:**

```
🤖 RAG AI Agent
Ask questions about the knowledge base. Type 'exit' to quit.

You: What is the leave policy?

Searching knowledge base...
Found 3 relevant sources
Generating response...

Agent: According to the company's Leave Policy, employees are entitled to:
- 12 days of Casual Leave per year
- 10 days of Sick Leave per year
- 15 days of Earned Leave per year

Sources: company_policies.md

You: exit
Goodbye!
```

### Single Query Mode

```bash
python agent.py "How many sick days do I get?"
```

### MCP Server

```bash
# Test with MCP Inspector
npx @modelcontextprotocol/inspector python mcp_server.py
```

**Available Tools:**
- `search_knowledge_base` - Returns relevant document chunks
- `answer_question` - Generates AI-powered answers with sources

## Architecture

```
User Query → Agent → RAG Pipeline
                         ↓
                    Vector Store (FAISS)
                         ↓
                    Context Retrieval
                         ↓
                    LLM (Groq) → Answer
```

### Components

| File | Purpose |
|------|---------|
| `config.py` | Configuration and environment variables |
| `rag_pipeline.py` | Core RAG logic (load, chunk, retrieve, generate) |
| `mcp_server.py` | MCP protocol server (JSON-RPC over STDIO) |
| `agent.py` | Conversational interface with memory |
| `test_system.py` | Automated testing suite |

### Document Processing

**Chunking Strategy:**
```python
RecursiveCharacterTextSplitter(
    chunk_size=500,      # Optimal for embedding models
    chunk_overlap=50,    # Preserves context at boundaries
    separators=["\n\n", "\n", " ", ""]  # Split intelligently
)
```

**Process Flow:**
1. Load markdown files from `knowledge_base/`
2. Split into 500-character chunks with 50-char overlap
3. Generate 384-dimensional embeddings
4. Store in FAISS index
5. Persist to disk for reuse

## Configuration

Edit `config.py` to customize:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3
```

## Project Structure

```
task3-rag-system/
├── .env                      # API keys (create this)
├── .env.example              # Template for .env
├── config.py                 # Configuration
├── rag_pipeline.py          # Core RAG implementation
├── mcp_server.py            # MCP server
├── agent.py                 # AI agent
├── test_system.py           # Tests
├── requirements.txt         # Dependencies
├── README.md                # This file
└── knowledge_base/
    ├── company_policies.md  # Your documents
    └── vector_store/        # FAISS index (auto-generated)
```

## How It Works

### 1. Embeddings
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Converts text to semantic vectors
- Enables similarity-based search

### 2. Vector Store
- FAISS IndexFlatL2 for exact search
- Cosine similarity for retrieval
- Persistent storage on disk

### 3. Retrieval
```python
Query → Embedding → FAISS Search → Top-K Chunks
```

### 4. Generation
```python
Query + Context → LLM Prompt → AI Answer
```

### 5. Memory
Agent maintains last 5 conversation turns for context-aware responses.

## Testing

```bash
python test_system.py
```

**Validates:**
- ✓ Environment (API keys, directories)
- ✓ RAG pipeline (load, retrieve, generate)
- ✓ MCP server (protocol compliance)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GROQ_API_KEY not set` | Create `.env` file with your API key |
| `Vector store not found` | Run `python rag_pipeline.py` |
| `Module not found` | Run `pip install -r requirements.txt` |
| No response from agent | Check `.env` file exists and has valid key |

## Adding More Documents

1. Add `.md` files to `knowledge_base/`
2. Re-run initialization:
   ```bash
   python rag_pipeline.py
   ```
3. Test with new questions:
   ```bash
   python agent.py
   ```

## Extending the System

### Add PDF Support
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
```

### Add Multiple File Types
```python
from langchain_community.document_loaders import DirectoryLoader

# Load all text files
loader = DirectoryLoader(
    "knowledge_base/",
    glob="**/*.{md,txt,pdf}",
    loader_cls=TextLoader
)
```

### Improve Retrieval
```python
# Use MMR for diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

## Technical Details

### Why This Implementation Works

**Clean Architecture**
- Modular design with separation of concerns
- Each file has a single responsibility
- Easy to test and maintain

**LangChain Best Practices**
- Uses official loaders and splitters
- Implements standard retriever interface
- Follows documented patterns

**Production Ready**
- Error handling throughout
- Type hints for safety
- Logging and validation
- Environment-based configuration

**MCP Compliance**
- Proper JSON-RPC 2.0 protocol
- STDIO transport without stdout pollution
- Standard tool definitions

## Requirements

- Python 3.8+
- 2GB RAM (for embeddings)
- Internet connection (for Groq API)
- Groq API key

## License

MIT License

## Resources

- [LangChain Docs](https://python.langchain.com/)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [FAISS Documentation](https://faiss.ai/)
- [Groq API](https://console.groq.com/docs)
- [Sentence Transformers](https://www.sbert.net/)