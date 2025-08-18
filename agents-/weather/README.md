# Weather Agent System

A comprehensive, production-ready Weather Agent that combines multiple data sources and analysis tools to provide intelligent weather insights and agricultural condition assessments.

## 🌦️ Features

- **Multi-Source Weather Data**: IMD API, historical databases, satellite imagery
- **Intelligent Analysis**: AI-powered query understanding and response synthesis
- **Modular Architecture**: Clean separation between data retrieval, tools, and agent logic
- **Structured Responses**: JSON-formatted outputs with confidence scores
- **Multiple Interfaces**: CLI, batch processing, and programmatic API
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📁 Project Structure

```
weather/
├── retriever.py          # Data retrieval from various weather sources
├── tools.py             # Tool definitions with input/output schemas
├── agent.py             # AI agent for query processing and synthesis
├── main.py              # CLI interface and main entry point
├── requirements.txt     # Python dependencies
├── .env.template        # Environment variable template
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env with your API keys and configurations
```

### 2. Configuration

Edit the `.env` file with your credentials:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional (system uses mock data if not configured)
IMD_API_KEY=your-imd-api-key-here
TIMESERIES_DB_URI=postgresql://username:password@localhost:5432/weather_db
CLIMATE_MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
GEE_SERVICE_ACCOUNT_EMAIL=your-service-account@project.iam.gserviceaccount.com
```

### 3. Usage

#### Interactive Mode
```bash
python main.py
```

#### Single Query
```bash
python main.py "What's the current weather in Mumbai and forecast for next 5 days?"
```

#### Batch Processing
```bash
# Create queries.txt with one query per line
echo "Current weather in Delhi" > queries.txt
echo "Crop health analysis for Karnataka" >> queries.txt
python main.py --batch queries.txt
```

#### System Health Check
```bash
python main.py --health-check
```

#### List Available Tools
```bash
python main.py --tools
```

## 🛠️ Available Weather Analysis Tools

### 1. Current Weather (`get_current_weather`)
- **Purpose**: Get real-time weather conditions
- **Parameters**: `location` (required)
- **Returns**: Temperature, humidity, rainfall, wind speed, pressure

### 2. Weather Forecast (`get_weather_forecast`)
- **Purpose**: Get weather predictions up to 10 days
- **Parameters**: `location` (required), `days` (1-10, default: 7)
- **Returns**: Daily forecasts with temperature ranges, rainfall probability

### 3. Historical Weather Stats (`get_historical_weather_stats`)
- **Purpose**: Analyze historical weather patterns
- **Parameters**: `location`, `start_date`, `end_date` (YYYY-MM-DD format)
- **Returns**: Statistical summaries, averages, extremes

### 4. Satellite Crop Health Analysis (`get_satellite_crop_health_analysis`)
- **Purpose**: Assess vegetation health using NDVI satellite data
- **Parameters**: `region_geojson` (required), `start_date`, `end_date` (optional)
- **Returns**: NDVI statistics, crop health assessment, vegetation coverage

### 5. Climate Reports Search (`find_relevant_climate_reports`)
- **Purpose**: Find relevant climate research and reports
- **Parameters**: `topic` (required), `max_results` (1-10, default: 5)
- **Returns**: Relevant research papers and reports with relevance scores

## 📊 Response Format

All agent responses follow this structured JSON format:

```json
{
  "summary": "Brief overview of key findings",
  "data_points": [
    {
      "source": "Tool Name",
      "data": {
        "key_metric": "value",
        "another_metric": "value"
      }
    }
  ],
  "analysis": "Comprehensive analysis combining all data sources",
  "confidence_score": 0.85
}
```

### Confidence Score Interpretation
- **0.8-1.0**: High confidence (multiple consistent data sources)
- **0.5-0.7**: Medium confidence (limited but reliable data)
- **0.2-0.4**: Low confidence (incomplete or conflicting data)
- **0.0-0.1**: Very low confidence (insufficient data)

## 🔧 Data Sources and Integrations

### Weather Data
- **IMD API**: Real-time weather and forecasts from Indian Meteorological Department
- **TimescaleDB/InfluxDB**: Historical weather data storage and analysis
- **Mock Data**: Fallback when external APIs are unavailable

### Satellite Data
- **Google Earth Engine**: NDVI analysis for crop health assessment
- **Sentinel-2/Landsat**: High-resolution satellite imagery processing

### Research Data
- **MongoDB Vector Database**: Climate reports and research papers
- **Qdrant**: Alternative vector database for document search

## 🏗️ Architecture

### Modular Design
1. **`retriever.py`**: Clean data retrieval functions for each source
2. **`tools.py`**: Tool definitions with schemas and validation
3. **`agent.py`**: AI agent for query planning and response synthesis
4. **`main.py`**: User interfaces and application entry points

### Async Operations
- All network operations are asynchronous for better performance
- Concurrent tool execution when possible
- Proper error handling and fallback mechanisms

### Extensibility
- Easy to add new data sources in `retriever.py`
- Simple tool registration in `tools.py`
- Agent automatically discovers and uses new tools

## 📝 Example Queries

### Weather Queries
```
"What's the current weather in Mumbai?"
"Give me a 7-day forecast for Delhi with rainfall predictions"
"How does this month's temperature compare to historical averages in Bangalore?"
```

### Agricultural Queries
```
"Analyze crop health in Punjab using satellite data"
"What are the NDVI trends for wheat growing regions in Haryana?"
"Show vegetation stress indicators for Karnataka"
```

### Research Queries
```
"Find research on monsoon patterns in Western India"
"Search for climate studies about drought in Maharashtra"
"What are the latest findings on urban heat islands in Indian cities?"
```

### Combined Analysis
```
"Analyze weather patterns and crop conditions in Rajasthan for the last month"
"Compare current rainfall with historical data and assess crop impact in Tamil Nadu"
```

## 🧪 Testing and Development

### Run Health Check
```bash
python main.py --health-check
```

### Test Individual Components
```python
# Test retriever functions
python -c "import asyncio; from retriever import *; asyncio.run(get_current_weather('Mumbai'))"

# Test tools
python -c "import asyncio; from tools import *; asyncio.run(run_tool_health_check())"

# Test agent
python -c "import asyncio; from agent import *; agent = WeatherAgent(); asyncio.run(agent.process_query('Weather in Delhi'))"
```

### Enable Debug Logging
```bash
python main.py --verbose "your query here"
```

## 🔒 Security and Best Practices

### API Keys
- Store all API keys in `.env` file
- Never commit `.env` to version control
- Rotate API keys regularly
- Use least-privilege access for service accounts

### Database Security
- Use SSL/TLS for all database connections
- Implement proper authentication and authorization
- Regular security updates for database systems

### Rate Limiting
- Respect API rate limits for external services
- Implement exponential backoff for failed requests
- Monitor API usage and costs

## 🚀 Production Deployment

### Environment Setup
1. Configure production environment variables
2. Set up monitoring and logging
3. Implement health checks and alerts
4. Configure load balancing if needed

### Database Setup
```sql
-- Example TimescaleDB setup
CREATE DATABASE weather_db;
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE weather_data (
    time TIMESTAMPTZ NOT NULL,
    location TEXT NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    rainfall DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    pressure DOUBLE PRECISION
);

SELECT create_hypertable('weather_data', 'time');
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## 🤝 Extending the System

### Adding New Data Sources
1. Add retrieval function in `retriever.py`
2. Create tool definition in `tools.py`
3. Agent automatically discovers new tools

### Adding New Analysis Types
1. Implement analysis logic in retriever functions
2. Define tool schema with proper input/output validation
3. Update documentation

### Integration with Other Agents
```python
from weather.agent import WeatherAgent
from soil.agent import SoilAgent
from market.agent import MarketAgent

# Combine multiple agents for comprehensive analysis
weather_agent = WeatherAgent()
soil_agent = SoilAgent()
market_agent = MarketAgent()
```

## 📞 Support and Contributing

### Common Issues
1. **API Key Errors**: Check `.env` configuration and key validity
2. **Database Connection**: Verify connection strings and network access
3. **Tool Failures**: Run health check to identify specific issues

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### Reporting Issues
Include the following in issue reports:
- Error messages and stack traces
- Environment configuration (without sensitive data)
- Steps to reproduce the problem
- Expected vs actual behavior

## 📜 License

This project is provided as-is for educational and development purposes.

---

**Weather Agent System** - Intelligent Weather Analysis for Agricultural Applications

