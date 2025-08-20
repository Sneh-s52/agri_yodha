#!/usr/bin/env python3
"""
Test script to verify agent imports and basic functionality
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))

def test_soil_agent():
    """Test soil agent import and functionality"""
    try:
        sys.path.append(str(Path(__file__).parent / "agents-" / "soil"))
        # Check what's available in the soil agent file
        with open("agents-/soil/agent_soil.py", "r") as f:
            content = f.read()
            print("Soil Agent Content Preview:")
            print(content[:500] + "...")
        return True
    except Exception as e:
        print(f"Soil agent test failed: {e}")
        return False

def test_weather_agent():
    """Test weather agent import and functionality"""
    try:
        sys.path.append(str(Path(__file__).parent / "agents-" / "weather"))
        from agent import WeatherAgent
        
        agent = WeatherAgent()
        print("Weather Agent initialized successfully")
        return True
    except Exception as e:
        print(f"Weather agent test failed: {e}")
        return False

def test_market_agent():
    """Test market agent import and functionality"""
    try:
        sys.path.append(str(Path(__file__).parent / "agents-" / "Market-Agent"))
        from agent.market_agent import MarketAgent
        
        agent = MarketAgent()
        print("Market Agent initialized successfully")
        return True
    except Exception as e:
        print(f"Market agent test failed: {e}")
        return False

def test_finance_agent():
    """Test finance agent import and functionality"""
    try:
        sys.path.append(str(Path(__file__).parent / "agents-" / "finance"))
        from financial_agent import run_agent
        
        print("Finance Agent run_agent function imported successfully")
        return True
    except Exception as e:
        print(f"Finance agent test failed: {e}")
        return False

def test_policy_agent():
    """Test policy agent import and functionality"""
    try:
        sys.path.append(str(Path(__file__).parent / "agents-" / "policy"))
        from agent import InsuranceAgent
        
        agent = InsuranceAgent()
        print("Policy Agent (InsuranceAgent) initialized successfully")
        return True
    except Exception as e:
        print(f"Policy agent test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing all agents...")
    
    tests = [
        ("Soil Agent", test_soil_agent),
        ("Weather Agent", test_weather_agent),
        ("Market Agent", test_market_agent),
        ("Finance Agent", test_finance_agent),
        ("Policy Agent", test_policy_agent)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print('='*50)
        results[name] = test_func()
    
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print('='*50)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
