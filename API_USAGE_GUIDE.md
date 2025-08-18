# 🌾 Agricultural Advisory API - Usage Guide

A comprehensive REST API that provides agricultural advice using a multi-agent orchestrator system.

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
- [Request/Response Examples](#request-response-examples)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Configuration](#configuration)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install API requirements
pip install -r api_requirements.txt

# Or install specific packages
pip install fastapi uvicorn python-dotenv langchain langgraph
```

### 2. Set Environment Variables
Create a `.env` file:
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for enhanced functionality)
TAVILY_API_KEY=your_tavily_api_key
PINECONE_API_KEY=your_pinecone_api_key
MONGODB_URI=mongodb://localhost:27017/agri_database

# API Configuration (optional)
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true
```

### 3. Start the API Server
```bash
# Method 1: Direct execution
python agricultural_api.py

# Method 2: Using uvicorn
uvicorn agricultural_api:app --host 127.0.0.1 --port 8000 --reload

# Method 3: Background process
nohup python agricultural_api.py &
```

### 4. Access the API
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

## 📊 API Endpoints

### 1. Root Endpoint
```
GET /
```
Basic API information and status.

### 2. Health Check
```
GET /health
```
Check API health and agent availability.

### 3. Agent Status
```
GET /agents/status
```
Detailed status of all agricultural agents.

### 4. Quick Advice
```
POST /advice/quick?query=<your_question>
```
Simple endpoint for quick farming questions.

### 5. Full Agricultural Advice
```
POST /advice
```
Comprehensive advice with detailed farm information.

### 6. Active Requests
```
GET /requests/active
```
Check currently processing requests.

## 💡 Request/Response Examples

### Quick Advice Example
```bash
curl -X POST "http://127.0.0.1:8000/advice/quick?query=What%20is%20the%20best%20crop%20for%20small%20farms?"
```

### Full Advice Example
```bash
curl -X POST "http://127.0.0.1:8000/advice" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I have a 10-acre farm in Punjab and want to grow the most profitable crop for Kharif season",
    "farm_info": {
      "size_acres": 10.0,
      "location": "Punjab, India",
      "season": "Kharif",
      "soil_type": "Alluvial"
    },
    "include_policies": true,
    "include_market_analysis": true,
    "include_weather": true
  }'
```

### Response Format
```json
{
  "success": true,
  "request_id": "req_20250118_140532_0001",
  "timestamp": "2025-01-18T14:05:45.123456",
  "query": "Your farming question",
  "recommendation": "Comprehensive agricultural advice...",
  "agent_results": {
    "soil_agent": {
      "suitable_crops": ["Rice", "Wheat", "Maize"],
      "soil_health": {...}
    },
    "market_agent": {
      "profitability_ranking": [...],
      "market_analysis": {...}
    }
  },
  "execution_stats": {
    "execution_time_seconds": 12.45,
    "iterations": 6,
    "agents_called": 5,
    "timestamp": "2025-01-18T14:05:32.678901"
  }
}
```

## 🐍 Python Client Example

```python
import requests
import json

# API Configuration
API_URL = "http://127.0.0.1:8000"

def get_agricultural_advice(query, farm_info=None):
    """Get agricultural advice from the API"""
    
    # Prepare request
    request_data = {
        "query": query,
        "include_policies": True,
        "include_market_analysis": True,
        "include_weather": True
    }
    
    if farm_info:
        request_data["farm_info"] = farm_info
    
    # Make request
    response = requests.post(
        f"{API_URL}/advice",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

# Example usage
if __name__ == "__main__":
    # Farm information
    farm_info = {
        "size_acres": 15.0,
        "location": "Maharashtra, India",
        "season": "Kharif",
        "current_crop": "Cotton",
        "soil_type": "Black soil"
    }
    
    # Get advice
    query = "What is the most profitable crop rotation strategy for my farm?"
    
    try:
        result = get_agricultural_advice(query, farm_info)
        
        print(f"Request ID: {result['request_id']}")
        print(f"Execution Time: {result['execution_stats']['execution_time_seconds']:.2f}s")
        print(f"Agents Called: {result['execution_stats']['agents_called']}")
        print(f"\nRecommendation:\n{result['recommendation']}")
        
    except Exception as e:
        print(f"Error: {e}")
```

## 🧪 Testing

### Run the Test Suite
```bash
# Start the API server first
python agricultural_api.py

# In another terminal, run tests
python test_api.py
```

### Manual Testing with curl
```bash
# Health check
curl http://127.0.0.1:8000/health

# Quick advice
curl -X POST "http://127.0.0.1:8000/advice/quick?query=Best%20crops%20for%20Punjab"

# Agent status
curl http://127.0.0.1:8000/agents/status
```

## ⚙️ Configuration

### Environment Variables
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM reasoning | None | Yes* |
| `API_HOST` | API server host | 127.0.0.1 | No |
| `API_PORT` | API server port | 8000 | No |
| `API_RELOAD` | Enable auto-reload in development | true | No |
| `TAVILY_API_KEY` | Tavily API for web search | None | No |
| `PINECONE_API_KEY` | Pinecone for vector storage | None | No |
| `MONGODB_URI` | MongoDB connection string | None | No |

*Note: API works without OpenAI key but uses fallback reasoning

### Production Deployment
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn agricultural_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or with Docker
docker build -t agri-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key agri-api
```

## 🔧 Troubleshooting

### Common Issues

1. **API not starting**
   - Check if port 8000 is available
   - Verify Python dependencies are installed
   - Check for syntax errors in the code

2. **Slow responses**
   - Ensure OpenAI API key is configured
   - Check internet connection for API calls
   - Monitor system resources

3. **Agent failures**
   - Check environment variables
   - Verify API keys are valid
   - Review logs in `api.log`

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python agricultural_api.py
```

### Logs Location
- API logs: `api.log`
- Orchestrator logs: `orchestrator.log`

## 📚 Advanced Usage

### Custom Headers
```python
headers = {
    "Content-Type": "application/json",
    "User-Agent": "MyApp/1.0",
    "X-Request-ID": "custom-id-123"
}

response = requests.post(url, json=data, headers=headers)
```

### Async Client
```python
import aiohttp
import asyncio

async def get_advice_async(query):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://127.0.0.1:8000/advice/quick",
            params={"query": query}
        ) as response:
            return await response.json()

# Usage
result = asyncio.run(get_advice_async("Best crops for monsoon"))
```

### Batch Processing
```python
import concurrent.futures
import requests

def process_query(query):
    response = requests.post(
        "http://127.0.0.1:8000/advice/quick",
        params={"query": query}
    )
    return response.json()

# Process multiple queries in parallel
queries = [
    "Best crops for Punjab",
    "Fertilizer recommendations for wheat",
    "Market prices for rice"
]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_query, queries))
```

## 📞 Support

For issues, questions, or contributions:
1. Check the logs (`api.log`, `orchestrator.log`)
2. Review the interactive documentation at `/docs`
3. Test with the provided test script
4. Verify environment configuration

---

**Happy Farming! 🌱**
