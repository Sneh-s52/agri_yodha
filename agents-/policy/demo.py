#!/usr/bin/env python3
"""
Demo Script for MongoDB Vector Retriever and AI Agent

This script demonstrates how to use the retriever and agent components
with sample queries and different configuration options.

Usage:
    python demo.py

Make sure to set up your .env file before running this demo.

Author: AI Assistant
Date: 2025
"""

import os
import sys
import json
import logging
from typing import List

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from retriever import MongoRetriever, SimilarityType
from agent import InsuranceAgent, AgentCLI

# Load environment variables
load_dotenv()

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites() -> bool:
    """Check if all required environment variables are set."""
    print("🔧 Checking Prerequisites...")
    print("=" * 50)
    
    required_vars = ['MONGODB_URI', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            print(f"❌ {var}: Not set")
        else:
            print(f"✅ {var}: Set")
    
    if missing_vars:
        print(f"\n⚠️  Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file based on .env.template")
        return False
    
    print("✅ All prerequisites met!")
    return True


def demo_retriever():
    """Demonstrate the MongoRetriever functionality."""
    print("\n🔍 MongoDB Retriever Demo")
    print("=" * 50)
    
    try:
        # Initialize retriever
        print("Initializing MongoRetriever...")
        retriever = MongoRetriever()
        
        # Get collection statistics
        stats = retriever.get_collection_stats()
        print(f"\n📊 Collection Statistics:")
        print(json.dumps(stats, indent=2))
        
        if stats.get('total_documents', 0) == 0:
            print("\n⚠️  No documents found in the collection.")
            print("Please ensure your MongoDB collection has documents before running the demo.")
            retriever.close()
            return False
        
        # Demo queries
        demo_queries = [
            "insurance policy coverage",
            "medical procedures covered",
            "claim filing process",
            "deductible amounts"
        ]
        
        print(f"\n🔍 Testing Retriever with {len(demo_queries)} queries:")
        print("-" * 40)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\nQuery {i}: '{query}'")
            
            # Test different similarity types
            for sim_type in [SimilarityType.SPARSE, SimilarityType.HYBRID]:
                print(f"\n  {sim_type.value.upper()} SEARCH:")
                
                results = retriever.retrieve(
                    query=query,
                    top_k=3,
                    similarity_type=sim_type,
                    min_score=0.1
                )
                
                if results:
                    print(f"    ✅ Found {len(results)} results")
                    for j, result in enumerate(results[:2], 1):  # Show top 2
                        print(f"      {j}. Score: {result.score:.4f} | Method: {result.retrieval_method}")
                        print(f"         Text: {result.text[:80]}...")
                else:
                    print("    ❌ No results found")
        
        retriever.close()
        print("\n✅ Retriever demo completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Retriever demo failed: {e}")
        print(f"❌ Retriever demo failed: {e}")
        return False


def demo_agent():
    """Demonstrate the InsuranceAgent functionality."""
    print("\n🤖 AI Agent Demo")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("Initializing InsuranceAgent...")
        agent = InsuranceAgent()
        
        # Demo questions
        demo_questions = [
            "What medical procedures are covered under my insurance policy?",
            "How much is the deductible for surgical procedures?",
            "What documents do I need to file an insurance claim?",
            "Are prescription medications covered?",
            "What is the process for getting pre-authorization?"
        ]
        
        print(f"\n🔍 Testing Agent with {len(demo_questions)} questions:")
        print("-" * 60)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("-" * 40)
            
            # Process the question
            response = agent.query(question, top_k=3)
            
            # Display results
            print(f"🤖 Answer (Confidence: {response.confidence:.1%}):")
            print(f"{response.answer}")
            
            print(f"\n📚 Sources ({len(response.sources)} documents):")
            for j, source in enumerate(response.sources[:2], 1):  # Show top 2 sources
                print(f"  {j}. Score: {source.score:.3f} | Method: {source.retrieval_method}")
                print(f"     Chunk ID: {source.chunk_id}")
                print(f"     Preview: {source.text[:100]}...")
            
            print("\n" + "="*60)
        
        # Demo batch processing
        print(f"\n🔄 Testing Batch Processing:")
        print("-" * 30)
        
        batch_questions = demo_questions[:3]
        batch_responses = agent.batch_query(batch_questions, top_k=2)
        
        print(f"✅ Processed {len(batch_responses)} questions in batch")
        for i, response in enumerate(batch_responses, 1):
            print(f"  {i}. Query: {response.query[:50]}...")
            print(f"     Confidence: {response.confidence:.1%}")
            print(f"     Sources: {len(response.sources)} documents")
        
        agent.close()
        print("\n✅ Agent demo completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Agent demo failed: {e}")
        print(f"❌ Agent demo failed: {e}")
        return False


def demo_interactive_mode():
    """Demonstrate the interactive CLI mode."""
    print("\n💬 Interactive Mode Demo")
    print("=" * 50)
    
    try:
        agent = InsuranceAgent()
        cli = AgentCLI(agent)
        
        print("Starting interactive mode...")
        print("You can now ask questions about insurance policies!")
        print("(This will start an interactive session)")
        
        # Ask user if they want to run interactive mode
        response = input("\nWould you like to start interactive mode? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            cli.run_interactive()
        else:
            print("Skipping interactive mode.")
        
        agent.close()
        return True
        
    except Exception as e:
        logger.error(f"Interactive demo failed: {e}")
        print(f"❌ Interactive demo failed: {e}")
        return False


def main():
    """Run the complete demo."""
    print("🚀 MongoDB Vector Retriever and AI Agent Demo")
    print("=" * 60)
    print("This demo will showcase the retriever and agent capabilities.")
    print("Make sure you have set up your .env file with valid credentials.\n")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Demo cannot proceed without required environment variables.")
        print("Please set up your .env file and try again.")
        return
    
    # Run demos
    demos = [
        ("Retriever", demo_retriever),
        ("Agent", demo_agent),
        ("Interactive Mode", demo_interactive_mode)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} Demo {'='*20}")
            
            # Ask user if they want to run this demo
            if demo_name == "Interactive Mode":
                # Special handling for interactive mode
                results[demo_name] = demo_func()
            else:
                response = input(f"\nRun {demo_name} demo? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    results[demo_name] = demo_func()
                else:
                    print(f"Skipping {demo_name} demo.")
                    results[demo_name] = True  # Mark as successful skip
        
        except KeyboardInterrupt:
            print(f"\n\n⏸️  {demo_name} demo interrupted by user.")
            results[demo_name] = False
            break
        except Exception as e:
            logger.error(f"{demo_name} demo error: {e}")
            results[demo_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 Demo Summary")
    print("="*60)
    
    for demo_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {demo_name}: {status}")
    
    successful_demos = sum(1 for success in results.values() if success)
    total_demos = len(results)
    
    print(f"\n🎯 Results: {successful_demos}/{total_demos} demos completed successfully")
    
    if successful_demos == total_demos:
        print("🎉 All demos completed successfully!")
        print("\n💡 Next steps:")
        print("  • Use 'python agent.py' for interactive mode")
        print("  • Import the modules in your own Python scripts")
        print("  • Run 'python test_integration.py' for comprehensive testing")
    else:
        print("⚠️  Some demos had issues. Check the logs above for details.")
    
    print("\n📚 Documentation: See README.md for detailed usage instructions")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Demo script error: {e}")
        print(f"❌ Demo script failed: {e}")
        sys.exit(1)
