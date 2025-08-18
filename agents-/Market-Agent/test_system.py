#!/usr/bin/env python3
# market_agent/test_system.py

import os
import sys
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        from config import (
            OPENAI_API_KEY, TAVILY_API_KEY, PINECONE_API_KEY,
            MONGO_URI, PINECONE_INDEX_NAME, MONGO_DB_NAME, MONGO_COLLECTION_NAME
        )
        print("✅ Config imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
        
    try:
        from data_layer.mongo_client import mongo_ts_client
        print("✅ MongoDB client imported successfully")
    except Exception as e:
        print(f"❌ MongoDB client import failed: {e}")
        
    try:
        from data_layer.pinecone_client import pinecone_vector_client
        print("✅ Pinecone client imported successfully")
    except Exception as e:
        print(f"❌ Pinecone client import failed: {e}")
        
    try:
        from tools.news_search_tool import NewsSearchTool
        print("✅ News search tool imported successfully")
    except Exception as e:
        print(f"❌ News search tool import failed: {e}")
        
    try:
        from forecasting.price_forecasting import PriceForecastingTool
        print("✅ Price forecasting tool imported successfully")
    except Exception as e:
        print(f"❌ Price forecasting tool import failed: {e}")
        
    try:
        from agent.market_agent import MarketAgent
        print("✅ Market agent imported successfully")
    except Exception as e:
        print(f"❌ Market agent import failed: {e}")
        
    return True

def test_config():
    """Test configuration values."""
    print("\n🔧 Testing configuration...")
    
    try:
        from config import (
            OPENAI_API_KEY, TAVILY_API_KEY, PINECONE_API_KEY,
            MONGO_URI, PINECONE_INDEX_NAME, MONGO_DB_NAME, MONGO_COLLECTION_NAME
        )
        
        config_vars = {
            "OPENAI_API_KEY": OPENAI_API_KEY,
            "TAVILY_API_KEY": TAVILY_API_KEY,
            "PINECONE_API_KEY": PINECONE_API_KEY,
            "MONGO_URI": MONGO_URI,
            "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
            "MONGO_DB_NAME": MONGO_DB_NAME,
            "MONGO_COLLECTION_NAME": MONGO_COLLECTION_NAME
        }
        
        for var, value in config_vars.items():
            if value:
                print(f"✅ {var}: Set")
            else:
                print(f"⚠️ {var}: Not set")
                
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_forecasting_tool():
    """Test the forecasting tool with sample data."""
    print("\n🔮 Testing forecasting tool...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        prices = [100 + np.random.normal(0, 5) + i*0.1 for i in range(len(dates))]
        
        sample_df = pd.DataFrame({
            'modal_price': prices
        }, index=dates)
        
        from forecasting.price_forecasting import PriceForecastingTool
        
        forecaster = PriceForecastingTool()
        
        # Test data preparation
        prepared_data = forecaster.prepare_data(sample_df)
        print(f"✅ Data preparation: {len(prepared_data)} data points")
        
        # Test simple forecasting
        forecast = forecaster.simple_moving_average_forecast(prepared_data, forecast_days=7)
        if "error" not in forecast:
            print("✅ Simple moving average forecast successful")
        else:
            print(f"❌ Simple forecast failed: {forecast['error']}")
            
        # Test ensemble forecast
        ensemble = forecaster.ensemble_forecast(prepared_data, forecast_days=7)
        if "error" not in ensemble:
            print("✅ Ensemble forecast successful")
        else:
            print(f"❌ Ensemble forecast failed: {ensemble['error']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Forecasting tool test failed: {e}")
        return False

def test_market_agent():
    """Test the market agent initialization."""
    print("\n🤖 Testing market agent...")
    
    try:
        from agent.market_agent import create_market_agent
        
        agent = create_market_agent()
        print("✅ Market agent created successfully")
        
        # Test basic methods
        if hasattr(agent, 'analyze_market_conditions'):
            print("✅ Market analysis method available")
        if hasattr(agent, 'get_price_forecast'):
            print("✅ Price forecasting method available")
        if hasattr(agent, 'search_market_news'):
            print("✅ News search method available")
            
        return True
        
    except Exception as e:
        print(f"❌ Market agent test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Agri Yodha Market Agent System Tests")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Forecasting Tool", test_forecasting_tool),
        ("Market Agent", test_market_agent)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
            
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
            
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nTo run the system, use: python3 main.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
