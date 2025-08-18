"""
Weather Agent Module

This module implements an intelligent Weather Agent that uses various weather analysis tools
to provide comprehensive weather insights. The agent can understand natural language queries,
select appropriate tools, and synthesize responses in structured JSON format.

Author: AI Assistant
Date: 2025
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

from tools import (
    weather_tools,
    execute_weather_tool,
    get_all_tool_schemas,
    get_available_tools
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Data class for structured agent responses."""
    summary: str
    data_points: List[Dict[str, Any]]
    analysis: str
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "summary": self.summary,
            "data_points": self.data_points,
            "analysis": self.analysis,
            "confidence_score": self.confidence_score
        }


class WeatherAgent:
    """
    Intelligent Weather Agent that combines multiple weather analysis tools
    to provide comprehensive weather insights and analysis.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        max_context_length: int = 8000
    ):
        """
        Initialize the Weather Agent.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model_name: Name of the OpenAI model to use
            max_context_length: Maximum length of context to send to LLM
        """
        # Initialize OpenAI client
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.max_context_length = max_context_length
        
        # Get available tools
        self.available_tools = get_all_tool_schemas()
        self.tool_names = get_available_tools()
        
        # System prompt for the agent
        self.system_prompt = self._create_system_prompt()
        
        logger.info(f"WeatherAgent initialized with model: {model_name}")
        logger.info(f"Available tools: {', '.join(self.tool_names)}")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the Weather Agent."""
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in self.available_tools
        ])
        
        return f"""You are an expert Weather Agent that helps users analyze weather patterns, climate data, and agricultural conditions. Your role is to:

1. **Understand user queries** about weather, climate, and agricultural conditions
2. **Select appropriate tools** from the available weather analysis tools
3. **Synthesize information** from multiple sources
4. **Provide structured JSON responses** with analysis and insights

## Available Tools:
{tools_description}

## Response Guidelines:
- Use ONLY the provided tools - do not hallucinate data
- Always cite sources (IMD, satellite data, historical records, climate reports)
- Provide confidence scores based on data quality and completeness
- Structure your response as JSON with these exact fields:
  {{
    "summary": "Brief summary of findings",
    "data_points": [
      {{"source": "Tool Name", "data": {{tool_result_data}}}},
      {{"source": "Tool Name", "data": {{tool_result_data}}}}
    ],
    "analysis": "Detailed analysis combining all data sources",
    "confidence_score": 0.0-1.0
  }}

## Tool Selection Strategy:
- For current conditions: use get_current_weather
- For predictions: use get_weather_forecast  
- For historical comparisons: use get_historical_weather_stats
- For crop/vegetation health: use get_satellite_crop_health_analysis
- For research context: use find_relevant_climate_reports

## Analysis Quality:
- High confidence (0.8-1.0): Multiple consistent data sources
- Medium confidence (0.5-0.7): Limited but reliable data
- Low confidence (0.2-0.4): Incomplete or conflicting data
- Very low confidence (0.0-0.1): Insufficient data

Always explain your reasoning and highlight any limitations in the available data."""
    
    async def _plan_tool_execution(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Plan which tools to execute based on the user query.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            List of tool execution plans
        """
        try:
            # Use LLM to determine which tools to use
            planning_prompt = f"""
Analyze this weather query and determine which tools to use: "{user_query}"

Available tools:
{json.dumps(self.available_tools, indent=2)}

Respond with a JSON list of tool execution plans. Each plan should have:
{{
  "tool_name": "exact_tool_name",
  "parameters": {{"param1": "value1", "param2": "value2"}},
  "reasoning": "why this tool is needed"
}}

Consider:
- Extract location information from the query
- Determine if current weather, forecast, or historical data is needed
- Check if satellite analysis or climate reports would be helpful
- Use reasonable defaults for missing parameters (e.g., 7 days for forecast)

Respond with ONLY the JSON array, no other text.
"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a tool planning assistant. Respond only with valid JSON."},
                    {"role": "user", "content": planning_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            try:
                execution_plan = json.loads(plan_text)
                logger.info(f"Generated execution plan with {len(execution_plan)} tools")
                return execution_plan
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool execution plan: {e}")
                logger.error(f"Raw response: {plan_text}")
                # Fallback to simple planning
                return self._fallback_planning(user_query)
                
        except Exception as e:
            logger.error(f"Error in tool planning: {e}")
            return self._fallback_planning(user_query)
    
    def _fallback_planning(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Fallback planning when LLM planning fails.
        
        Args:
            user_query: User query
            
        Returns:
            Basic tool execution plan
        """
        plan = []
        query_lower = user_query.lower()
        
        # Extract location (simple heuristic)
        location = "New Delhi, India"  # Default location
        if "mumbai" in query_lower:
            location = "Mumbai, India"
        elif "bangalore" in query_lower:
            location = "Bangalore, India"
        elif "chennai" in query_lower:
            location = "Chennai, India"
        
        # Basic tool selection based on keywords
        if any(word in query_lower for word in ["current", "now", "today"]):
            plan.append({
                "tool_name": "get_current_weather",
                "parameters": {"location": location},
                "reasoning": "Query asks for current weather conditions"
            })
        
        if any(word in query_lower for word in ["forecast", "tomorrow", "next", "will"]):
            plan.append({
                "tool_name": "get_weather_forecast",
                "parameters": {"location": location, "days": 7},
                "reasoning": "Query asks for weather forecast"
            })
        
        if any(word in query_lower for word in ["crop", "vegetation", "ndvi", "satellite"]):
            # Simple GeoJSON for the location area
            sample_geojson = '{"type": "Polygon", "coordinates": [[[77.0, 12.5], [78.0, 12.5], [78.0, 13.5], [77.0, 13.5], [77.0, 12.5]]]}'
            plan.append({
                "tool_name": "get_satellite_crop_health_analysis",
                "parameters": {"region_geojson": sample_geojson},
                "reasoning": "Query relates to crop or vegetation health"
            })
        
        # If no specific tools identified, use current weather as default
        if not plan:
            plan.append({
                "tool_name": "get_current_weather",
                "parameters": {"location": location},
                "reasoning": "Default weather information"
            })
        
        logger.info(f"Using fallback planning with {len(plan)} tools")
        return plan
    
    async def _execute_tools(self, execution_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute the planned tools and collect results.
        
        Args:
            execution_plan: List of tool execution plans
            
        Returns:
            List of tool execution results
        """
        results = []
        
        for plan in execution_plan:
            try:
                tool_name = plan["tool_name"]
                parameters = plan["parameters"]
                reasoning = plan.get("reasoning", "")
                
                logger.info(f"Executing tool: {tool_name} with params: {parameters}")
                
                # Execute the tool
                result = await execute_weather_tool(tool_name, **parameters)
                
                # Add metadata to result
                result["tool_name"] = tool_name
                result["reasoning"] = reasoning
                result["execution_success"] = "error" not in result
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing tool {plan.get('tool_name', 'unknown')}: {e}")
                results.append({
                    "tool_name": plan.get("tool_name", "unknown"),
                    "error": str(e),
                    "execution_success": False,
                    "reasoning": plan.get("reasoning", "")
                })
        
        return results
    
    def _synthesize_response(self, user_query: str, tool_results: List[Dict[str, Any]]) -> AgentResponse:
        """
        Synthesize tool results into a structured agent response.
        
        Args:
            user_query: Original user query
            tool_results: Results from tool execution
            
        Returns:
            Structured agent response
        """
        try:
            # Prepare context for LLM
            context = {
                "user_query": user_query,
                "tool_results": tool_results,
                "successful_tools": [r for r in tool_results if r.get("execution_success", False)],
                "failed_tools": [r for r in tool_results if not r.get("execution_success", False)]
            }
            
            synthesis_prompt = f"""
Analyze the weather data and provide a comprehensive response to the user's query.

User Query: "{user_query}"

Tool Results:
{json.dumps(tool_results, indent=2, default=str)}

Create a structured response with:
1. **Summary**: Brief overview of key findings
2. **Data Points**: Organize tool results by source
3. **Analysis**: Comprehensive analysis combining all data
4. **Confidence Score**: Based on data quality and completeness

Respond with ONLY valid JSON in this exact format:
{{
  "summary": "Brief summary of findings",
  "data_points": [
    {{"source": "Tool Name", "data": {{key_data_from_tool}}}},
    {{"source": "Tool Name", "data": {{key_data_from_tool}}}}
  ],
  "analysis": "Detailed analysis combining all data sources with insights and patterns",
  "confidence_score": 0.85
}}

Guidelines:
- Cite specific data sources (IMD, satellite, historical records)
- Highlight trends, patterns, and anomalies
- Mention any limitations or gaps in data
- Use confidence score: 0.8-1.0 (high), 0.5-0.7 (medium), 0.2-0.4 (low), 0.0-0.1 (very low)
"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=1500,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            try:
                response_data = json.loads(response_text)
                
                return AgentResponse(
                    summary=response_data.get("summary", "Weather analysis completed"),
                    data_points=response_data.get("data_points", []),
                    analysis=response_data.get("analysis", "Analysis not available"),
                    confidence_score=float(response_data.get("confidence_score", 0.5))
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse synthesis response: {e}")
                logger.error(f"Raw response: {response_text}")
                return self._fallback_response(user_query, tool_results)
                
        except Exception as e:
            logger.error(f"Error in response synthesis: {e}")
            return self._fallback_response(user_query, tool_results)
    
    def _fallback_response(self, user_query: str, tool_results: List[Dict[str, Any]]) -> AgentResponse:
        """
        Generate a fallback response when synthesis fails.
        
        Args:
            user_query: Original user query
            tool_results: Tool execution results
            
        Returns:
            Basic agent response
        """
        successful_tools = [r for r in tool_results if r.get("execution_success", False)]
        
        # Create basic data points
        data_points = []
        for result in successful_tools:
            data_points.append({
                "source": result.get("tool_name", "Unknown Tool"),
                "data": {k: v for k, v in result.items() if k not in ["tool_name", "reasoning", "execution_success"]}
            })
        
        # Calculate basic confidence
        confidence = 0.3 if successful_tools else 0.1
        if len(successful_tools) >= 2:
            confidence = 0.6
        
        return AgentResponse(
            summary=f"Weather analysis completed for query: {user_query}",
            data_points=data_points,
            analysis=f"Retrieved data from {len(successful_tools)} weather analysis tools. "
                    f"{'Some tools failed to execute. ' if len(tool_results) > len(successful_tools) else ''}"
                    f"Please review the data points for specific weather information.",
            confidence_score=confidence
        )
    
    async def process_query(self, user_query: str) -> AgentResponse:
        """
        Process a natural language weather query and return structured response.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Structured weather analysis response
        """
        logger.info(f"Processing weather query: '{user_query[:100]}...'")
        
        try:
            # Step 1: Plan tool execution
            execution_plan = await self._plan_tool_execution(user_query)
            
            # Step 2: Execute tools
            tool_results = await self._execute_tools(execution_plan)
            
            # Step 3: Synthesize response
            response = self._synthesize_response(user_query, tool_results)
            
            logger.info(f"Query processed successfully. Confidence: {response.confidence_score:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return AgentResponse(
                summary=f"Error processing weather query: {user_query}",
                data_points=[],
                analysis=f"An error occurred while processing your query: {str(e)}",
                confidence_score=0.0
            )
    
    async def batch_process(self, queries: List[str]) -> List[AgentResponse]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of agent responses
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        
        responses = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            response = await self.process_query(query)
            responses.append(response)
        
        return responses


class WeatherAgentCLI:
    """
    Command-line interface for the Weather Agent.
    """
    
    def __init__(self, agent: WeatherAgent):
        """
        Initialize CLI with an agent instance.
        
        Args:
            agent: WeatherAgent instance
        """
        self.agent = agent
    
    async def run_interactive(self):
        """Run interactive CLI session."""
        print("\n" + "="*60)
        print("🌦️  Weather Analysis Agent")
        print("="*60)
        print("Ask me about weather patterns, forecasts, climate data, and crop conditions!")
        print("Type 'quit', 'exit', or 'bye' to end the session.")
        print("Type 'tools' to see available analysis tools.")
        print("Type 'help' for more commands.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Stay weather-aware!")
                    break
                
                if user_input.lower() == 'tools':
                    self._show_tools()
                    continue
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                # Process the query
                print("\n🔍 Analyzing weather data...")
                response = await self.agent.process_query(user_input)
                
                # Display response
                print(f"\n🤖 Weather Agent (Confidence: {response.confidence_score:.1%}):")
                print(f"\n📋 Summary:")
                print(f"{response.summary}")
                
                print(f"\n📊 Data Points:")
                for i, data_point in enumerate(response.data_points, 1):
                    print(f"  {i}. Source: {data_point['source']}")
                    if isinstance(data_point['data'], dict):
                        for key, value in list(data_point['data'].items())[:3]:  # Show first 3 items
                            print(f"     {key}: {value}")
                        if len(data_point['data']) > 3:
                            print(f"     ... and {len(data_point['data']) - 3} more fields")
                    else:
                        print(f"     {data_point['data']}")
                
                print(f"\n🔬 Analysis:")
                print(f"{response.analysis}")
                
                print("\n" + "-"*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"CLI error: {e}")
    
    def _show_tools(self):
        """Show available weather analysis tools."""
        print("\n🛠️  Available Weather Analysis Tools:")
        for tool_name in self.agent.tool_names:
            tool_schema = next(t for t in self.agent.available_tools if t['name'] == tool_name)
            print(f"  • {tool_name}: {tool_schema['description']}")
    
    def _show_help(self):
        """Show help information."""
        print("\n📋 Available Commands:")
        print("  • Ask any weather-related question in natural language")
        print("  • 'tools' - Show available analysis tools")
        print("  • 'help' - Show this help message")
        print("  • 'quit', 'exit', 'bye' - End the session")
        print("\n💡 Example questions:")
        print("  • What's the current weather in Mumbai?")
        print("  • Give me a 7-day forecast for Delhi")
        print("  • How does this month's rainfall compare to historical averages?")
        print("  • Analyze crop health in Karnataka using satellite data")
        print("  • Find research on monsoon patterns in India")


if __name__ == "__main__":
    """
    Example usage of the Weather Agent.
    """
    async def main():
        print("Weather Agent - Example Usage")
        print("=" * 50)
        
        # Check API key requirements
        openai_key = os.getenv("OPENAI_API_KEY")
        weather_key = os.getenv("OPENAI_API_KEY_WEATHER") or openai_key
        
        if not weather_key or weather_key == "your_openai_api_key_for_weather_agent":
            print("❌ ERROR: OpenAI API key not found or not set properly.")
            print("\n📋 REQUIREMENTS TO RUN WEATHER AGENT:")
            print("="*60)
            print("1. OpenAI API Key (REQUIRED):")
            print("   - Get an API key from: https://platform.openai.com/api-keys")
            print("   - Add to .env file as: OPENAI_API_KEY=sk-your-actual-key-here")
            print("   - OR: OPENAI_API_KEY_WEATHER=sk-your-actual-key-here")
            print(f"   - Current value: {weather_key}")
            print("\n2. Dependencies (✅ Already installed):")
            print("   ✅ openai, aiohttp, python-dotenv")
            print("\n3. Internet connection for OpenAI API calls")
            print("\n📝 Note: Weather tools use mock/demo data.")
            print("No external weather APIs are required for testing.")
            print("="*60)
            return
        
        try:
            print("✅ OpenAI API key found. Initializing Weather Agent...")
            logger.info("Starting Weather Agent with OpenAI API key")
            
            # Initialize agent
            agent = WeatherAgent()
            
            # Test queries
            test_queries = [
                "What's the current weather in Mumbai and how does it look for the next 3 days?",
                "Analyze the crop health in Karnataka using satellite data",
                "Find climate research about monsoon patterns in India"
            ]
            
            print(f"\nTesting {len(test_queries)} sample queries:")
            print("-" * 50)
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n🔍 Query {i}: {query}")
                logger.info(f"Processing query {i}: {query[:50]}...")
                
                try:
                    response = await agent.process_query(query)
                    
                    print(f"\n📝 Response (Confidence: {response.confidence_score:.1%}):")
                    response_dict = response.to_dict()
                    print(json.dumps(response_dict, indent=2, default=str))
                    logger.info(f"Query {i} completed successfully with confidence {response.confidence_score:.2f}")
                    
                except Exception as query_error:
                    print(f"❌ Error processing query {i}: {query_error}")
                    logger.error(f"Query {i} failed: {query_error}")
                    
                    # Check if it's an API-related error
                    if "api" in str(query_error).lower() or "openai" in str(query_error).lower():
                        print("💡 This appears to be an API-related error:")
                        print("   - Check your OpenAI API key")
                        print("   - Verify internet connection")
                        print("   - Check API rate limits")
                    
                print("\n" + "="*50)
            
            # Interactive mode (commented out for example)
            # cli = WeatherAgentCLI(agent)
            # await cli.run_interactive()
            
        except Exception as e:
            print(f"❌ Weather Agent initialization failed: {e}")
            logger.error(f"Agent initialization failed: {e}")
            
            # Provide specific guidance based on error type
            if "api" in str(e).lower() or "openai" in str(e).lower():
                print("\n💡 API-related error detected:")
                print("   - Verify your OpenAI API key is correct")
                print("   - Check if you have sufficient API credits")
                print("   - Ensure stable internet connection")
            elif "import" in str(e).lower() or "module" in str(e).lower():
                print("\n💡 Dependency error detected:")
                print("   - Run: .\\agri-env\\Scripts\\pip.exe install aiohttp openai")
            else:
                print(f"\n💡 Unexpected error: {e}")
    
    # Run the example
    asyncio.run(main())
