# 🚀 Agri Yodha Market Agent

**Next Generation AI Chatbot for Policy Makers and Financiers**

A comprehensive market intelligence and forecasting system that combines real-time data analysis, price forecasting, news sentiment analysis, and policy recommendations for agricultural commodity markets.

## 🌟 Features

### 🔍 **Market Analysis & Intelligence**
- **Real-time Market Data**: Integration with MongoDB and Pinecone for comprehensive data access
- **Price Forecasting**: Multiple statistical models (ARIMA, Exponential Smoothing, Moving Average, Ensemble)
- **Trend Analysis**: Advanced pattern recognition and volatility assessment
- **Market Intelligence**: Comprehensive reports combining multiple data sources

### 📰 **News & Sentiment Analysis**
- **Real-time News Search**: Powered by Tavily API for market-related news
- **Sentiment Analysis**: AI-powered sentiment scoring for market conditions
- **Commodity-specific News**: Targeted news search for specific commodities and APMCs

### 🏛️ **Policy & Risk Management**
- **Risk Assessment**: Automated risk scoring and factor identification
- **Policy Recommendations**: Data-driven policy implications and suggestions
- **Strategic Insights**: Actionable recommendations for policy makers and financiers

### 📊 **Data Integration**
- **MongoDB**: Time-series data storage for historical price analysis
- **Pinecone**: Vector database for semantic search and similarity matching
- **Multi-source Analysis**: Combines data from multiple sources for robust insights

## 🏗️ Architecture

```
Market-Agent/
├── agent/                    # Core agent logic
│   ├── market_agent.py      # Main market agent orchestrator
│   └── __init__.py
├── data_layer/              # Data access layer
│   ├── mongo_client.py      # MongoDB time-series client
│   ├── pinecone_client.py   # Pinecone vector client
│   └── __init__.py
├── tools/                    # Utility tools
│   ├── news_search_tool.py  # News search and sentiment analysis
│   ├── embeddings.py        # OpenAI embeddings utility
│   └── __init__.py
├── forecasting/              # Price forecasting engine
│   └── price_forecasting.py # Statistical forecasting models
├── config.py                 # Configuration and environment variables
├── main.py                   # CLI interface and main entry point
├── test_system.py           # System testing and validation
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### 1. **Environment Setup**

```bash
# Clone the repository
git clone <repository-url>
cd Market-Agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configuration**

Create a `.env` file in the project root:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Database Configuration
MONGO_URI=your_mongodb_connection_string
PINECONE_INDEX_NAME=your_pinecone_index_name

# Database Names
MONGO_DB_NAME=mandi_prices
MONGO_COLLECTION_NAME=commodity_prices
```

### 3. **System Testing**

```bash
# Test system components
python3 test_system.py
```

### 4. **Run the System**

```bash
# Start the CLI interface
python3 main.py
```

## 📖 Usage Guide

### **Command Line Interface**

The system provides an interactive CLI with the following commands:

#### 🔍 **Market Analysis**
```bash
analyze wheat "Delhi APMC"     # Comprehensive market analysis
forecast rice "Mumbai APMC"    # Price forecasting (7 days)
intelligence wheat             # Market intelligence report
```

#### 📰 **News & Sentiment**
```bash
news "wheat price increase India"    # Search market news
sentiment "rice export policy"        # Market sentiment analysis
```

#### 📊 **Data & Reports**
```bash
export wheat "Delhi APMC"      # Export analysis report
history                         # View conversation history
clear                          # Clear conversation history
```

#### ⚙️ **System**
```bash
help                           # Show help message
status                         # Show system status
quit/exit                     # Exit the application
```

### **Example Workflow**

1. **Market Analysis**: `analyze wheat "Delhi APMC"`
2. **Price Forecasting**: `forecast wheat "Delhi APMC"`
3. **News Search**: `news "wheat market trends"`
4. **Export Report**: `export wheat "Delhi APMC"`

## 🔧 Technical Details

### **Forecasting Models**

The system uses multiple statistical models for robust price predictions:

- **Simple Moving Average**: Short-term trend analysis
- **Exponential Smoothing**: Holt-Winters method with seasonality
- **ARIMA**: Auto-regressive integrated moving average
- **Ensemble Methods**: Weighted combination of multiple models

### **Data Processing**

- **Time Series Preparation**: Automatic data cleaning and resampling
- **Missing Data Handling**: Interpolation and forward-filling strategies
- **Data Quality Assessment**: Automatic quality scoring and recommendations

### **Risk Assessment**

- **Volatility Analysis**: Price volatility scoring and classification
- **Trend Strength**: Statistical trend strength measurement
- **Multi-factor Risk**: Combined risk scoring from multiple sources

## 📊 Data Sources

### **MongoDB Time-Series**
- Historical commodity prices
- APMC market data
- Time-stamped price records

### **Pinecone Vector Database**
- Semantic search capabilities
- Similarity matching
- Metadata-rich price information

### **News APIs**
- Real-time market news
- Sentiment analysis
- Trend identification

## 🛠️ Development

### **Adding New Features**

1. **New Data Sources**: Extend the `data_layer` module
2. **New Forecasting Models**: Add to `forecasting/price_forecasting.py`
3. **New Tools**: Create in the `tools/` directory
4. **Agent Logic**: Extend `agent/market_agent.py`

### **Testing**

```bash
# Run system tests
python3 test_system.py

# Test specific components
python3 -m pytest tests/
```

### **Logging**

The system provides comprehensive logging:
- File logging: `logs/market_agent.log`
- Console logging: Real-time system status
- Error tracking: Detailed error logging and debugging

## 🔒 Security & Privacy

- **API Key Management**: Environment variable-based configuration
- **Data Encryption**: Secure database connections
- **Access Control**: Configurable data access permissions
- **Audit Logging**: Complete system activity tracking

## 📈 Performance

### **Optimization Features**
- **Connection Pooling**: Efficient database connections
- **Caching**: Intelligent result caching
- **Async Processing**: Non-blocking operations
- **Resource Management**: Automatic cleanup and optimization

### **Scalability**
- **Modular Architecture**: Easy component replacement
- **Configurable Limits**: Adjustable resource usage
- **Horizontal Scaling**: Support for multiple instances

## 🚨 Troubleshooting

### **Common Issues**

1. **MongoDB Connection Failed**
   - Check `MONGO_URI` in `.env`
   - Verify network connectivity
   - Check MongoDB service status

2. **Pinecone Connection Failed**
   - Verify `PINECONE_API_KEY`
   - Check `PINECONE_INDEX_NAME`
   - Ensure index exists and is accessible

3. **API Key Errors**
   - Verify all required API keys in `.env`
   - Check API key validity and quotas
   - Ensure proper environment variable loading

### **Debug Mode**

Enable debug logging by modifying the logging level in `config.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests for new functionality**
5. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI**: For advanced language models and embeddings
- **Tavily**: For comprehensive news search capabilities
- **Pinecone**: For vector database infrastructure
- **MongoDB**: For time-series data storage

## 📞 Support

For support and questions:
- **Issues**: Create an issue on GitHub
- **Documentation**: Check this README and inline code comments
- **Testing**: Use `test_system.py` for system validation

---

**Built with ❤️ for the Agri Yodha initiative**

*Empowering policy makers and financiers with AI-driven market intelligence*
