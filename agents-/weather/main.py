#!/usr/bin/env python3
"""
Weather Agent Main Entry Point

This script provides a command-line interface to the Weather Agent system.
It accepts user queries and returns structured JSON responses with weather analysis.

Usage:
    python main.py                          # Interactive mode
    python main.py "query here"             # Single query mode
    python main.py --batch queries.txt      # Batch processing mode
    python main.py --health-check           # System health check
    python main.py --tools                  # List available tools

Author: AI Assistant
Date: 2025
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from typing import List
from datetime import datetime

from dotenv import load_dotenv
from agent import WeatherAgent, WeatherAgentCLI
from tools import run_tool_health_check, get_available_tools, get_all_tool_schemas

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('weather_agent.log')
    ]
)
logger = logging.getLogger(__name__)


class WeatherAgentMain:
    """Main application class for the Weather Agent system."""
    
    def __init__(self):
        """Initialize the main application."""
        self.agent = None
        
    async def initialize_agent(self) -> bool:
        """
        Initialize the Weather Agent.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.agent = WeatherAgent()
            logger.info("Weather Agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Weather Agent: {e}")
            print(f"❌ Error: Failed to initialize Weather Agent: {e}")
            print("Please check your environment variables and API keys.")
            return False
    
    async def process_single_query(self, query: str) -> None:
        """
        Process a single query and print the result.
        
        Args:
            query: User query string
        """
        try:
            print(f"🔍 Processing query: {query}")
            print("⏳ Analyzing weather data...")
            
            response = await self.agent.process_query(query)
            
            # Print structured JSON response
            response_dict = response.to_dict()
            response_dict["query"] = query
            response_dict["timestamp"] = datetime.now().isoformat()
            
            print("\n📊 Weather Analysis Result:")
            print(json.dumps(response_dict, indent=2, default=str, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
            print(json.dumps(error_response, indent=2))
    
    async def process_batch_queries(self, file_path: str) -> None:
        """
        Process multiple queries from a file.
        
        Args:
            file_path: Path to file containing queries (one per line)
        """
        try:
            if not os.path.exists(file_path):
                print(f"❌ Error: File not found: {file_path}")
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            if not queries:
                print(f"❌ Error: No queries found in file: {file_path}")
                return
            
            print(f"📋 Processing {len(queries)} queries from {file_path}")
            
            batch_results = []
            for i, query in enumerate(queries, 1):
                print(f"\n⏳ Processing query {i}/{len(queries)}: {query[:50]}...")
                
                try:
                    response = await self.agent.process_query(query)
                    result = response.to_dict()
                    result["query"] = query
                    result["query_number"] = i
                    result["timestamp"] = datetime.now().isoformat()
                    batch_results.append(result)
                    
                except Exception as e:
                    error_result = {
                        "query": query,
                        "query_number": i,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "status": "failed"
                    }
                    batch_results.append(error_result)
                    logger.error(f"Error processing query {i}: {e}")
            
            # Save results to file
            output_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"\n✅ Batch processing completed!")
            print(f"📄 Results saved to: {output_file}")
            print(f"📊 Summary: {len(batch_results)} queries processed")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            print(f"❌ Error in batch processing: {e}")
    
    async def run_interactive_mode(self) -> None:
        """Run the interactive CLI mode."""
        try:
            cli = WeatherAgentCLI(self.agent)
            await cli.run_interactive()
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"❌ Error in interactive mode: {e}")
    
    async def run_health_check(self) -> None:
        """Run system health check."""
        print("🏥 Running Weather Agent System Health Check")
        print("=" * 60)
        
        # Check environment variables
        print("\n1. Environment Variables:")
        required_vars = ['OPENAI_API_KEY']
        optional_vars = [
            'IMD_API_KEY',
            'TIMESERIES_DB_URI',
            'GEE_SERVICE_ACCOUNT_EMAIL',
            'CLIMATE_MONGODB_URI'
        ]
        
        env_status = True
        for var in required_vars:
            if os.getenv(var):
                print(f"   ✅ {var}: Set")
            else:
                print(f"   ❌ {var}: Not set (REQUIRED)")
                env_status = False
        
        for var in optional_vars:
            if os.getenv(var):
                print(f"   ✅ {var}: Set")
            else:
                print(f"   ⚠️  {var}: Not set (optional)")
        
        # Check agent initialization
        print("\n2. Agent Initialization:")
        if await self.initialize_agent():
            print("   ✅ Weather Agent: Initialized successfully")
            agent_status = True
        else:
            print("   ❌ Weather Agent: Initialization failed")
            agent_status = False
        
        # Check tools health
        print("\n3. Weather Analysis Tools:")
        try:
            tool_health = await run_tool_health_check()
            tools_healthy = 0
            for tool_name, is_healthy in tool_health.items():
                status = "✅ HEALTHY" if is_healthy else "❌ UNHEALTHY"
                print(f"   {status} {tool_name}")
                if is_healthy:
                    tools_healthy += 1
            
            tools_status = tools_healthy > 0
            
        except Exception as e:
            print(f"   ❌ Tool health check failed: {e}")
            tools_status = False
        
        # Overall status
        print("\n4. Overall System Status:")
        if env_status and agent_status and tools_status:
            print("   🎉 System is HEALTHY and ready for use!")
            return_code = 0
        else:
            print("   ⚠️  System has issues that need attention:")
            if not env_status:
                print("      - Missing required environment variables")
            if not agent_status:
                print("      - Agent initialization failed")
            if not tools_status:
                print("      - Weather analysis tools are not functioning")
            return_code = 1
        
        print(f"\n📋 Health Check Summary:")
        print(f"   Environment: {'✅' if env_status else '❌'}")
        print(f"   Agent: {'✅' if agent_status else '❌'}")
        print(f"   Tools: {'✅' if tools_status else '❌'}")
        
        sys.exit(return_code)
    
    def show_available_tools(self) -> None:
        """Display available weather analysis tools."""
        print("🛠️  Available Weather Analysis Tools")
        print("=" * 60)
        
        try:
            tools = get_all_tool_schemas()
            
            for i, tool in enumerate(tools, 1):
                print(f"\n{i}. {tool['name']}")
                print(f"   Description: {tool['description']}")
                print(f"   Parameters:")
                
                properties = tool['parameters']['properties']
                required = tool['parameters'].get('required', [])
                
                for param_name, param_info in properties.items():
                    required_marker = " (required)" if param_name in required else " (optional)"
                    param_type = param_info.get('type', 'unknown')
                    param_desc = param_info.get('description', 'No description')
                    print(f"     • {param_name} ({param_type}){required_marker}: {param_desc}")
            
            print(f"\n📊 Total Tools Available: {len(tools)}")
            
        except Exception as e:
            logger.error(f"Error displaying tools: {e}")
            print(f"❌ Error displaying tools: {e}")


async def main():
    """Main entry point for the Weather Agent application."""
    parser = argparse.ArgumentParser(
        description="Weather Agent - Intelligent Weather Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py "What's the weather in Mumbai?"    # Single query
  python main.py --batch queries.txt                # Batch processing
  python main.py --health-check                     # System health check
  python main.py --tools                            # List available tools

For more information, visit: https://github.com/your-repo/weather-agent
        """
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Weather query to process (if not provided, starts interactive mode)'
    )
    
    parser.add_argument(
        '--batch',
        metavar='FILE',
        help='Process queries from file (one query per line)'
    )
    
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run system health check'
    )
    
    parser.add_argument(
        '--tools',
        action='store_true',
        help='List available weather analysis tools'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize main application
    app = WeatherAgentMain()
    
    try:
        # Handle different modes
        if args.health_check:
            await app.run_health_check()
            
        elif args.tools:
            app.show_available_tools()
            
        elif args.batch:
            if not await app.initialize_agent():
                sys.exit(1)
            await app.process_batch_queries(args.batch)
            
        elif args.query:
            if not await app.initialize_agent():
                sys.exit(1)
            await app.process_single_query(args.query)
            
        else:
            # Interactive mode
            if not await app.initialize_agent():
                sys.exit(1)
            await app.run_interactive_mode()
    
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user. Goodbye!")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main application
    asyncio.run(main())

