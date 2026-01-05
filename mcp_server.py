"""
MCP Server for RAG System
Exposes RAG capabilities as MCP tools using STDIO transport
"""

import asyncio
import sys
import logging
from typing import List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from rag_pipeline import RAGPipeline

# Logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# MCP Server Initialization
server = Server("rag-mcp-server")

# Initialize RAG pipeline ONCE at startup
try:
    rag_pipeline = RAGPipeline(silent=True)
    rag_pipeline.load_vectorstore()
    rag_pipeline.setup_retriever()
    logging.info("RAG pipeline initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize RAG pipeline: {e}")
    raise


# MCP: List Tools

@server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="search_knowledge_base",
            description="Search the knowledge base and return relevant document chunks",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="answer_question",
            description="Answer a question using the knowledge base and LLM",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to answer"
                    }
                },
                "required": ["question"]
            }
        )
    ]


# MCP: Tool Execution
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    try:
        if name == "search_knowledge_base":
            query = arguments["query"]
            top_k = arguments.get("top_k", 3)

            docs = rag_pipeline.retrieve(query)

            results = []
            for i, doc in enumerate(docs[:top_k], 1):
                results.append(
                    TextContent(
                        type="text",
                        text=(
                            f"Result {i}\n"
                            f"{doc.page_content}\n\n"
                            f"Source: {doc.metadata.get('source', 'Unknown')}"
                        )
                    )
                )

            return results

        if name == "answer_question":
            question = arguments["question"]

            result = rag_pipeline.search_and_summarize(question)

            answer_text = f"{result['answer']}\n\nSources:\n"
            for i, src in enumerate(result["sources"], 1):
                answer_text += f"{i}. {src['metadata'].get('source', 'Unknown')}\n"

            return [
                TextContent(
                    type="text",
                    text=answer_text
                )
            ]

        raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logging.error(f"Tool execution failed: {e}")
        return [
            TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )
        ]


# MCP: STDIO Server Entrypoint
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
