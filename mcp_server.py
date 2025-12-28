"""
MCP Server Implementation using STDIO protocol
Exposes RAG capabilities as MCP tools
"""
import sys
import json
import logging
from typing import Any, Dict, List
from rag_pipeline import RAGPipeline

# Suppress print statements to avoid JSON parsing errors
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

class MCPServer:
    """MCP Server implementing JSON-RPC 2.0 protocol via STDIO"""
    
    def __init__(self):
        self.pipeline = None
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialize RAG pipeline silently"""
        try:
            self.pipeline = RAGPipeline(silent=True)
            self.pipeline.load_vectorstore()
            self.pipeline.setup_retriever()
        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {e}")
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC request"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return self.handle_initialize(request_id, params)
            elif method == "tools/list":
                return self.handle_list_tools(request_id)
            elif method == "tools/call":
                return self.handle_tool_call(request_id, params)
            else:
                return self.error_response(request_id, -32601, "Method not found")
        
        except Exception as e:
            logging.error(f"Error handling request: {e}")
            return self.error_response(request_id, -32603, str(e))
    
    def handle_initialize(self, request_id: Any, params: Dict) -> Dict:
        """Handle initialize request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "rag-mcp-server",
                    "version": "1.0.0"
                }
            }
        }
    
    def handle_list_tools(self, request_id: Any) -> Dict:
        """Handle tools/list request"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "search_knowledge_base",
                        "description": "Search the knowledge base and return relevant document chunks",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of results to return",
                                    "default": 3
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "answer_question",
                        "description": "Answer a question using the knowledge base and LLM",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The question to answer"
                                }
                            },
                            "required": ["question"]
                        }
                    }
                ]
            }
        }
    
    def handle_tool_call(self, request_id: Any, params: Dict) -> Dict:
        """Handle tools/call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "search_knowledge_base":
            return self.search_tool(request_id, arguments)
        elif tool_name == "answer_question":
            return self.answer_tool(request_id, arguments)
        else:
            return self.error_response(request_id, -32602, f"Unknown tool: {tool_name}")
    
    def search_tool(self, request_id: Any, arguments: Dict) -> Dict:
        """Execute search tool"""
        query = arguments.get("query")
        top_k = arguments.get("top_k", 3)
        
        if not query:
            return self.error_response(request_id, -32602, "Missing required parameter: query")
        
        try:
            results = self.pipeline.retrieve(query)
            
            content = []
            for i, doc in enumerate(results[:top_k], 1):
                content.append({
                    "type": "text",
                    "text": f"Result {i}:\n{doc.page_content}\n\nSource: {doc.metadata.get('source', 'Unknown')}"
                })
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": content
                }
            }
        
        except Exception as e:
            return self.error_response(request_id, -32603, f"Search failed: {str(e)}")
    
    def answer_tool(self, request_id: Any, arguments: Dict) -> Dict:
        """Execute answer tool"""
        question = arguments.get("question")
        
        if not question:
            return self.error_response(request_id, -32602, "Missing required parameter: question")
        
        try:
            result = self.pipeline.search_and_summarize(question)
            
            answer_text = f"Answer: {result['answer']}\n\nSources:\n"
            for i, source in enumerate(result['sources'], 1):
                answer_text += f"{i}. {source['metadata'].get('source', 'Unknown')}\n"
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": answer_text
                        }
                    ]
                }
            }
        
        except Exception as e:
            return self.error_response(request_id, -32603, f"Answer generation failed: {str(e)}")
    
    def error_response(self, request_id: Any, code: int, message: str) -> Dict:
        """Create error response"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
    def run(self):
        """Run the MCP server loop"""
        for line in sys.stdin:
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON: {e}")
            except Exception as e:
                logging.error(f"Error processing request: {e}")


if __name__ == "__main__":
    server = MCPServer()
    server.run()