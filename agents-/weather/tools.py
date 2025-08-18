"""
Weather Agent Tools Module

This module wraps retriever functions as callable tools with proper schemas.
Each tool has a name, description, input schema, and output schema for use
by the Weather Agent.

Author: AI Assistant
Date: 2025
"""

import json
import asyncio
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging

from retriever import (
    get_current_weather,
    get_weather_forecast,
    get_historical_weather_stats,
    get_satellite_crop_health_analysis,
    find_relevant_climate_reports
)

# Configure logging
logger = logging.getLogger(__name__)


class ToolInputSchema(BaseModel):
    """Base class for tool input schemas."""
    pass


class ToolOutputSchema(BaseModel):
    """Base class for tool output schemas."""
    pass


# Input Schemas
class CurrentWeatherInput(ToolInputSchema):
    """Input schema for current weather tool."""
    location: str = Field(..., description="Location name (city, state) or coordinates (lat,lon)")


class WeatherForecastInput(ToolInputSchema):
    """Input schema for weather forecast tool."""
    location: str = Field(..., description="Location name (city, state) or coordinates (lat,lon)")
    days: int = Field(default=7, ge=1, le=10, description="Number of forecast days (1-10)")


class HistoricalWeatherInput(ToolInputSchema):
    """Input schema for historical weather stats tool."""
    location: str = Field(..., description="Location name (city, state) or coordinates (lat,lon)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")


class SatelliteCropAnalysisInput(ToolInputSchema):
    """Input schema for satellite crop health analysis tool."""
    region_geojson: str = Field(..., description="GeoJSON string defining the region of interest")
    start_date: str = Field(default=None, description="Start date for analysis (YYYY-MM-DD), optional")
    end_date: str = Field(default=None, description="End date for analysis (YYYY-MM-DD), optional")


class ClimateReportsInput(ToolInputSchema):
    """Input schema for climate reports search tool."""
    topic: str = Field(..., description="Search topic or query for climate reports")
    max_results: int = Field(default=5, ge=1, le=10, description="Maximum number of results to return")


# Output Schemas
class WeatherDataOutput(ToolOutputSchema):
    """Output schema for weather data."""
    location: str
    timestamp: str
    temperature: float
    humidity: float
    rainfall: float
    wind_speed: float
    pressure: float
    weather_condition: str = None
    source: str


class ForecastOutput(ToolOutputSchema):
    """Output schema for weather forecast."""
    location: str
    forecast_days: int
    generated_at: str
    forecasts: List[Dict[str, Any]]
    source: str


class HistoricalStatsOutput(ToolOutputSchema):
    """Output schema for historical weather statistics."""
    location: str
    period: str
    statistics: Dict[str, float]
    source: str


class NDVIAnalysisOutput(ToolOutputSchema):
    """Output schema for NDVI satellite analysis."""
    region: str
    analysis_period: str
    ndvi_statistics: Dict[str, float]
    crop_health_assessment: Dict[str, Any]
    cloud_cover_percentage: float
    map_visualization_url: str
    source: str


class ClimateReportsOutput(ToolOutputSchema):
    """Output schema for climate reports search."""
    query: str
    total_results: int
    reports: List[Dict[str, Any]]
    search_timestamp: str
    source: str


@dataclass
class WeatherTool:
    """
    Data class representing a weather analysis tool.
    
    Each tool has a name, description, input/output schemas, and an async function.
    """
    name: str
    description: str
    input_schema: type[ToolInputSchema]
    output_schema: type[ToolOutputSchema]
    function: Callable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation for agent use."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
            "output_schema": self.output_schema.model_json_schema(),
            "parameters": {
                "type": "object",
                "properties": self.input_schema.model_json_schema()["properties"],
                "required": self.input_schema.model_json_schema().get("required", [])
            }
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            # Validate input parameters
            validated_input = self.input_schema(**kwargs)
            
            # Execute the tool function
            result = await self.function(**validated_input.model_dump())
            
            # Add execution metadata
            result["tool_name"] = self.name
            result["execution_timestamp"] = json.dumps(result, default=str)  # Handle datetime serialization
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}")
            return {
                "error": str(e),
                "tool_name": self.name,
                "status": "failed"
            }


class WeatherToolRegistry:
    """
    Registry for all weather analysis tools.
    Provides easy access and management of available tools.
    """
    
    def __init__(self):
        """Initialize the tool registry with all available tools."""
        self.tools = self._initialize_tools()
        logger.info(f"Initialized weather tool registry with {len(self.tools)} tools")
    
    def _initialize_tools(self) -> Dict[str, WeatherTool]:
        """Initialize and register all weather tools."""
        tools = {}
        
        # Current Weather Tool
        tools["get_current_weather"] = WeatherTool(
            name="get_current_weather",
            description="Get current weather conditions for a specific location including temperature, humidity, rainfall, wind speed, and atmospheric pressure",
            input_schema=CurrentWeatherInput,
            output_schema=WeatherDataOutput,
            function=get_current_weather
        )
        
        # Weather Forecast Tool
        tools["get_weather_forecast"] = WeatherTool(
            name="get_weather_forecast",
            description="Get weather forecast for a specific location for up to 10 days, including temperature ranges, rainfall probability, and weather conditions",
            input_schema=WeatherForecastInput,
            output_schema=ForecastOutput,
            function=get_weather_forecast
        )
        
        # Historical Weather Statistics Tool
        tools["get_historical_weather_stats"] = WeatherTool(
            name="get_historical_weather_stats",
            description="Get historical weather statistics for a location and date range, including average, minimum, maximum temperatures and total rainfall",
            input_schema=HistoricalWeatherInput,
            output_schema=HistoricalStatsOutput,
            function=get_historical_weather_stats
        )
        
        # Satellite Crop Health Analysis Tool
        tools["get_satellite_crop_health_analysis"] = WeatherTool(
            name="get_satellite_crop_health_analysis",
            description="Analyze crop health using satellite NDVI data for a specific region, providing vegetation health assessment and stress indicators",
            input_schema=SatelliteCropAnalysisInput,
            output_schema=NDVIAnalysisOutput,
            function=get_satellite_crop_health_analysis
        )
        
        # Climate Reports Search Tool
        tools["find_relevant_climate_reports"] = WeatherTool(
            name="find_relevant_climate_reports",
            description="Search for relevant climate reports and research documents on specific topics from scientific databases and publications",
            input_schema=ClimateReportsInput,
            output_schema=ClimateReportsOutput,
            function=find_relevant_climate_reports
        )
        
        return tools
    
    def get_tool(self, tool_name: str) -> WeatherTool:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            WeatherTool instance
            
        Raises:
            KeyError: If tool not found
        """
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")
        
        return self.tools[tool_name]
    
    def get_all_tools(self) -> Dict[str, WeatherTool]:
        """Get all available tools."""
        return self.tools.copy()
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all tools for agent use.
        
        Returns:
            List of tool descriptions with schemas
        """
        return [tool.to_dict() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            tool = self.get_tool(tool_name)
            return await tool.execute(**kwargs)
        except KeyError as e:
            logger.error(f"Tool not found: {e}")
            return {
                "error": str(e),
                "status": "tool_not_found"
            }
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "error": str(e),
                "tool_name": tool_name,
                "status": "execution_failed"
            }


# Global tool registry instance
weather_tools = WeatherToolRegistry()


# Convenience functions for direct tool access
async def execute_weather_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Execute a weather tool by name.
    
    Args:
        tool_name: Name of the tool to execute
        **kwargs: Tool parameters
        
    Returns:
        Tool execution result
    """
    return await weather_tools.execute_tool(tool_name, **kwargs)


def get_available_tools() -> List[str]:
    """Get list of available tool names."""
    return list(weather_tools.get_all_tools().keys())


def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """
    Get the schema for a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool schema dictionary
    """
    tool = weather_tools.get_tool(tool_name)
    return tool.to_dict()


def get_all_tool_schemas() -> List[Dict[str, Any]]:
    """Get schemas for all available tools."""
    return weather_tools.get_tool_descriptions()


# Tool validation and testing functions
async def validate_tool_execution(tool_name: str, test_params: Dict[str, Any]) -> bool:
    """
    Validate that a tool can be executed with given test parameters.
    
    Args:
        tool_name: Name of the tool to validate
        test_params: Test parameters for the tool
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        result = await execute_weather_tool(tool_name, **test_params)
        return "error" not in result
    except Exception as e:
        logger.error(f"Tool validation failed for {tool_name}: {e}")
        return False


async def run_tool_health_check() -> Dict[str, bool]:
    """
    Run health check on all weather tools.
    
    Returns:
        Dict mapping tool names to health status
    """
    health_status = {}
    
    # Test parameters for each tool
    test_params = {
        "get_current_weather": {"location": "New Delhi, India"},
        "get_weather_forecast": {"location": "Mumbai, India", "days": 3},
        "get_historical_weather_stats": {
            "location": "Bangalore, India",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07"
        },
        "get_satellite_crop_health_analysis": {
            "region_geojson": '{"type": "Polygon", "coordinates": [[[77.5, 12.9], [77.6, 12.9], [77.6, 13.0], [77.5, 13.0], [77.5, 12.9]]]}'
        },
        "find_relevant_climate_reports": {"topic": "monsoon patterns", "max_results": 2}
    }
    
    for tool_name in get_available_tools():
        if tool_name in test_params:
            health_status[tool_name] = await validate_tool_execution(tool_name, test_params[tool_name])
        else:
            health_status[tool_name] = False
            logger.warning(f"No test parameters defined for tool: {tool_name}")
    
    return health_status


if __name__ == "__main__":
    """
    Example usage and testing of weather tools.
    """
    async def main():
        print("Weather Tools - Example Usage")
        print("=" * 50)
        
        # List available tools
        print("\n1. Available Tools:")
        for tool_name in get_available_tools():
            print(f"  - {tool_name}")
        
        # Show tool schemas
        print("\n2. Tool Schemas:")
        for schema in get_all_tool_schemas():
            print(f"\nTool: {schema['name']}")
            print(f"Description: {schema['description']}")
            print(f"Parameters: {list(schema['parameters']['properties'].keys())}")
        
        # Test tool execution
        print("\n3. Testing Tool Execution:")
        
        # Test current weather
        print("\n3a. Current Weather:")
        result = await execute_weather_tool("get_current_weather", location="New Delhi, India")
        print(json.dumps(result, indent=2, default=str))
        
        # Test weather forecast
        print("\n3b. Weather Forecast:")
        result = await execute_weather_tool("get_weather_forecast", location="Mumbai, India", days=3)
        print(json.dumps(result, indent=2, default=str))
        
        # Run health check
        print("\n4. Tool Health Check:")
        health_status = await run_tool_health_check()
        for tool_name, is_healthy in health_status.items():
            status = "✅ HEALTHY" if is_healthy else "❌ UNHEALTHY"
            print(f"  {tool_name}: {status}")
    
    # Run the example
    asyncio.run(main())
