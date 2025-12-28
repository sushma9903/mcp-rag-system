"""
AI Agent for RAG System
Orchestrates retrieval and generation with conversational interface
"""
from typing import List, Dict, Any
from groq import Groq
from rag_pipeline import RAGPipeline
import config
import sys


class RAGAgent:
    """
    Intelligent agent that uses RAG pipeline to answer questions
    Maintains conversation history and context
    """
    
    def __init__(self, silent=False):
        self.pipeline = RAGPipeline(silent=True)
        self.conversation_history = []
        self.silent = silent
        
        if not self.silent:
            print("Initializing RAG system...")
        
        # Load vector store
        try:
            self.pipeline.load_vectorstore()
            self.pipeline.setup_retriever()
            if not self.silent:
                print("Knowledge base loaded")
        except Exception as e:
            if not self.silent:
                print(f"Error: Could not load knowledge base. Run 'python rag_pipeline.py' first.")
            sys.exit(1)
        
        # Check LLM availability
        if not self.pipeline.llm_client:
            if not self.silent:
                print("Error: GROQ_API_KEY not configured in .env file")
            sys.exit(1)
        
        if not self.silent:
            print("Ready!\n")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant context from knowledge base"""
        docs = self.pipeline.retrieve(query)
        
        context = []
        for doc in docs[:top_k]:
            context.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return context
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate AI response using retrieved context"""
        
        # Format context
        context_text = "\n\n".join([
            f"[Source: {c['source']}]\n{c['content']}"
            for c in context
        ])
        
        # Build messages with conversation history
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant answering questions about company policies. "
                    "Provide clear, direct answers based on the context provided. "
                    "Use natural language without markdown formatting or asterisks. "
                    "Answer conversationally without repeatedly mentioning sources. "
                    "If the question asks about something, use all relevant information from the context "
                    "to give a comprehensive answer, even if it doesn't explicitly mention the exact words. "
                    "For example, if asked about 'benefits', describe the leave policies, remote work options, etc. "
                    "Only say information is unavailable if the context truly has nothing relevant."
                )
            }
        ]
        
        # Add conversation history (last 5 turns)
        for entry in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": entry["query"]})
            messages.append({"role": "assistant", "content": entry["response"]})
        
        # Add current query with context
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}"
        })
        
        # Generate response
        try:
            response = self.pipeline.llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"I encountered an error processing your request."
    
    def chat(self, query: str) -> Dict[str, Any]:
        """
        Main chat interface
        Retrieves context and generates response
        """
        context = self.retrieve_context(query)
        response = self.generate_response(query, context)
        
        # Store in conversation history
        self.conversation_history.append({
            "query": query,
            "response": response,
            "context": context
        })
        
        return {
            "query": query,
            "response": response,
            "sources": context
        }
    
    def display_response(self, result: Dict[str, Any]):
        """Display formatted response"""
        # Get unique source filenames
        unique_sources = list(set([s['source'].split('\\')[-1].split('/')[-1] for s in result['sources']]))
        
        # Clean up response (remove markdown formatting)
        response = result['response']
        response = response.replace('**', '')  # Remove bold
        response = response.replace('*', '')   # Remove italics
        
        print(f"\nAgent: {response}\n")
    
    def run_interactive(self):
        """Run interactive chat loop"""
        print("🤖 RAG AI Assistant")
        print("Ask questions about your knowledge base. Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye!\n")
                    break
                
                result = self.chat(query)
                self.display_response(result)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                print(f"\nError: Unable to process request.\n")


def main():
    """Main entry point"""
    agent = RAGAgent()
    
    # Check if running in interactive mode or single query
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        result = agent.chat(query)
        agent.display_response(result)
    else:
        # Interactive mode
        agent.run_interactive()


if __name__ == "__main__":
    main()