"""
AI Agent with MongoDB Vector Retrieval

This module implements an intelligent agent that uses the MongoRetriever
to find relevant documents and provides context-aware responses using
an OpenAI-compatible LLM API.

The agent is designed to be extensible and can be enhanced with additional
tools and capabilities as needed.

Author: AI Assistant
Date: 2025
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import openai
from dotenv import load_dotenv

from retriever import MongoRetriever, RetrievalResult, SimilarityType

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Data class for agent responses."""
    query: str
    answer: str
    sources: List[RetrievalResult]
    retrieval_method: str
    response_timestamp: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            'query': self.query,
            'answer': self.answer,
            'sources': [source.to_dict() for source in self.sources],
            'retrieval_method': self.retrieval_method,
            'response_timestamp': self.response_timestamp,
            'confidence': self.confidence
        }


class InsuranceAgent:
    """
    AI Agent that combines document retrieval with language model reasoning.
    
    This agent uses the MongoRetriever to find relevant documents and then
    uses an LLM to generate context-aware responses based on the retrieved
    information.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        mongodb_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        max_context_length: int = 4000
    ):
        """
        Initialize the Insurance Agent.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model_name: Name of the OpenAI model to use
            mongodb_uri: MongoDB connection URI
            database_name: Database name
            max_context_length: Maximum length of context to send to LLM
        """
        # Initialize OpenAI client
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        

        self.model_name = model_name
        self.max_context_length = max_context_length
        
        # Initialize retriever
        self.retriever = MongoRetriever(
            mongodb_uri=mongodb_uri,
            database_name=database_name
        )
        
        # System prompt for the agent
        self.system_prompt = """You are an expert insurance policy assistant. Your role is to help users understand insurance policies, coverage details, claims processes, and related questions.

When answering questions:
1. Use the provided context documents to give accurate, specific information
2. If information is not available in the context, clearly state this limitation
3. Be concise but comprehensive in your explanations
4. Always cite which documents or sources support your answer
5. For policy-specific questions, mention relevant policy sections or clauses
6. If asked about coverage limits, deductibles, or specific terms, provide exact details when available

Context documents will be provided with each query. Base your responses primarily on this context."""
        
        logger.info(f"InsuranceAgent initialized with model: {model_name}")
    
    def _build_context(self, results: List[RetrievalResult]) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted context string for the LLM
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            # Format each result with metadata
            chunk_info = f"[Document {i} - Score: {result.score:.3f} - Method: {result.retrieval_method}]"
            chunk_text = f"{chunk_info}\n{result.text}\n"
            
            # Check if adding this chunk would exceed max context length
            if current_length + len(chunk_text) > self.max_context_length:
                logger.info(f"Context truncated at {i-1} documents due to length limit")
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n---\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str) -> tuple[str, float]:
        """
        Generate response using OpenAI API.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Tuple of (response_text, confidence_score)
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.1,  # Low temperature for more consistent responses
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on context quality and response length
            confidence = self._calculate_confidence(context, answer)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}", 0.0
    
    def _calculate_confidence(self, context: str, answer: str) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            context: Retrieved context
            answer: Generated answer
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.5
        
        # Boost confidence if we have good context
        if len(context) > 100:
            base_confidence += 0.2
        
        # Boost confidence if answer is substantial
        if len(answer) > 50:
            base_confidence += 0.1
        
        # Reduce confidence if answer mentions limitations
        limitation_phrases = [
            "not available in the context",
            "cannot find",
            "not specified",
            "unclear from the provided information"
        ]
        
        if any(phrase in answer.lower() for phrase in limitation_phrases):
            base_confidence -= 0.2
        
        return max(0.0, min(1.0, base_confidence))
    
    def query(
        self,
        user_query: str,
        top_k: int = 5,
        similarity_type: Union[SimilarityType, str] = SimilarityType.HYBRID,
        min_score: float = 0.1
    ) -> AgentResponse:
        """
        Process a user query and generate a context-aware response.
        
        Args:
            user_query: Natural language question from the user
            top_k: Number of documents to retrieve for context
            similarity_type: Type of similarity search to use
            min_score: Minimum similarity score for retrieved documents
            
        Returns:
            AgentResponse containing the answer and supporting information
        """
        logger.info(f"Processing query: '{user_query[:50]}...'")
        
        try:
            # Step 1: Retrieve relevant documents
            retrieval_results = self.retriever.retrieve(
                query=user_query,
                top_k=top_k,
                similarity_type=similarity_type,
                min_score=min_score
            )
            
            if not retrieval_results:
                logger.warning("No relevant documents found")
                return AgentResponse(
                    query=user_query,
                    answer="I couldn't find any relevant information in the policy documents to answer your question. Please try rephrasing your query or ask about a different topic.",
                    sources=[],
                    retrieval_method=str(similarity_type.value if isinstance(similarity_type, SimilarityType) else similarity_type),
                    response_timestamp=datetime.now().isoformat(),
                    confidence=0.1
                )
            
            # Step 2: Build context from retrieved documents
            context = self._build_context(retrieval_results)
            
            # Step 3: Generate response using LLM
            answer, confidence = self._generate_response(user_query, context)
            
            # Step 4: Create response object
            response = AgentResponse(
                query=user_query,
                answer=answer,
                sources=retrieval_results,
                retrieval_method=str(similarity_type.value if isinstance(similarity_type, SimilarityType) else similarity_type),
                response_timestamp=datetime.now().isoformat(),
                confidence=confidence
            )
            
            logger.info(f"Query processed successfully. Retrieved {len(retrieval_results)} documents, confidence: {confidence:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return AgentResponse(
                query=user_query,
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                retrieval_method="error",
                response_timestamp=datetime.now().isoformat(),
                confidence=0.0
            )
    
    def batch_query(self, queries: List[str], **kwargs) -> List[AgentResponse]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            **kwargs: Additional arguments passed to query method
            
        Returns:
            List of AgentResponse objects
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        
        responses = []
        for query in queries:
            response = self.query(query, **kwargs)
            responses.append(response)
        
        return responses
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the underlying retriever.
        
        Returns:
            Dictionary with retriever statistics
        """
        return self.retriever.get_collection_stats()
    
    def close(self):
        """Close connections and cleanup resources."""
        if self.retriever:
            self.retriever.close()
        logger.info("Agent connections closed")


class AgentCLI:
    """
    Command-line interface for the Insurance Agent.
    """
    
    def __init__(self, agent: InsuranceAgent):
        """
        Initialize CLI with an agent instance.
        
        Args:
            agent: InsuranceAgent instance
        """
        self.agent = agent
    
    def run_interactive(self):
        """Run interactive CLI session."""
        print("\n" + "="*60)
        print("🤖 Insurance Policy Assistant")
        print("="*60)
        print("Ask me anything about insurance policies, coverage, or claims!")
        print("Type 'quit', 'exit', or 'bye' to end the session.")
        print("Type 'stats' to see retriever statistics.")
        print("Type 'help' for more commands.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Have a great day!")
                    break
                
                if user_input.lower() == 'stats':
                    stats = self.agent.get_retriever_stats()
                    print(f"\n📊 Retriever Statistics:")
                    print(json.dumps(stats, indent=2))
                    continue
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Process the query
                print("\n🔍 Searching for relevant information...")
                response = self.agent.query(user_input)
                
                # Display response
                print(f"\n🤖 Assistant (Confidence: {response.confidence:.1%}):")
                print(f"{response.answer}")
                
                # Show sources if available
                if response.sources:
                    print(f"\n📚 Sources ({len(response.sources)} documents):")
                    for i, source in enumerate(response.sources[:3], 1):  # Show top 3 sources
                        print(f"  {i}. Score: {source.score:.3f} | Method: {source.retrieval_method}")
                        print(f"     {source.text[:100]}...")
                
                print("\n" + "-"*40)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"CLI error: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\n📋 Available Commands:")
        print("  • Ask any question about insurance policies")
        print("  • 'stats' - Show retriever statistics")
        print("  • 'help' - Show this help message")
        print("  • 'quit', 'exit', 'bye' - End the session")
        print("\n💡 Example questions:")
        print("  • What does my policy cover for knee surgery?")
        print("  • What are the deductible amounts?")
        print("  • How do I file a claim?")


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the Insurance Agent.
    """
    print("Insurance Agent - Example Usage")
    print("=" * 50)
    
    try:
        # Initialize agent
        agent = InsuranceAgent()
        
        # Get stats
        stats = agent.get_retriever_stats()
        print(f"Retriever Stats: {json.dumps(stats, indent=2)}")
        
        # Test queries
        test_queries = [
          
            "What is the AIF scheme and how does it benefit farmers?",  
            "list all the schemes introduced by the prime minister for farmers"
        ]
        
        print(f"\nTesting {len(test_queries)} sample queries:")
        print("-" * 30)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            response = agent.query(query, top_k=5)
            
            print(f"📝 Answer (Confidence: {response.confidence:.1%}):")
            print(f"{response.answer}")
            print(f"📚 Sources: {len(response.sources)} documents from collections:")
            
            # Show which collections the sources came from
            collections_used = {}
            for source in response.sources:
                collection = source.metadata.get('collection', 'unknown')
                collections_used[collection] = collections_used.get(collection, 0) + 1
            
            for collection, count in collections_used.items():
                print(f"  - {collection}: {count} documents")
        
        # Interactive mode (commented out for example)
        # cli = AgentCLI(agent)
        # cli.run_interactive()
        
        # Close connections
        agent.close()
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        logger.error(f"Example execution failed: {e}")
