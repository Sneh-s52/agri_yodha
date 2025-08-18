#!/usr/bin/env python3
"""
Test script for Agricultural Advisory API

This script tests the API endpoints to ensure they work correctly.
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Agents: {sum(data['agents_available'].values())}/{len(data['agents_available'])} available")
            print(f"   API Keys: {sum(data['api_keys_configured'].values())}/{len(data['api_keys_configured'])} configured")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_agents_status():
    """Test the agents status endpoint"""
    print("\n🔍 Testing agents status endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/agents/status")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Agents status retrieved")
            print(f"   Available agents: {len(data['agents'])}")
            for agent, info in data['agents'].items():
                print(f"   - {agent}: {info['status']}")
            return True
        else:
            print(f"❌ Agents status failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Agents status error: {e}")
        return False

def test_quick_advice():
    """Test the quick advice endpoint"""
    print("\n🔍 Testing quick advice endpoint...")
    
    try:
        query = "What is the best crop for small farms?"
        response = requests.post(
            f"{API_BASE_URL}/advice/quick",
            params={"query": query}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Quick advice request successful")
            print(f"   Request ID: {data['request_id']}")
            print(f"   Execution time: {data['execution_stats']['execution_time_seconds']:.2f}s")
            print(f"   Agents called: {data['execution_stats']['agents_called']}")
            print(f"   Recommendation preview: {data['recommendation'][:100]}...")
            return True
        else:
            print(f"❌ Quick advice failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Quick advice error: {e}")
        return False

def test_full_advice():
    """Test the full advice endpoint with farm details"""
    print("\n🔍 Testing full advice endpoint...")
    
    try:
        request_data = {
            "query": "I have a farm in Punjab and want to grow the most profitable crop for Kharif season",
            "farm_info": {
                "size_acres": 10.0,
                "location": "Punjab, India",
                "season": "Kharif",
                "soil_type": "Alluvial"
            },
            "include_policies": True,
            "include_market_analysis": True,
            "include_weather": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/advice",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Full advice request successful")
            print(f"   Request ID: {data['request_id']}")
            print(f"   Execution time: {data['execution_stats']['execution_time_seconds']:.2f}s")
            print(f"   Iterations: {data['execution_stats']['iterations']}")
            print(f"   Agents called: {data['execution_stats']['agents_called']}")
            print(f"   Agent results: {list(data['agent_results'].keys())}")
            print(f"   Recommendation preview: {data['recommendation'][:200]}...")
            return True
        else:
            print(f"❌ Full advice failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Full advice error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🔍 Testing root endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint accessible")
            print(f"   Message: {data['message']}")
            print(f"   Version: {data['version']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Starting Agricultural Advisory API Tests")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ API is not responding. Please start the API server first:")
            print("   python agricultural_api.py")
            return
    except:
        print("❌ Cannot connect to API. Please start the API server first:")
        print("   python agricultural_api.py")
        return
    
    # Run tests
    tests = [
        test_root_endpoint,
        test_health_check,
        test_agents_status,
        test_quick_advice,
        test_full_advice
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(1)  # Small delay between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the API implementation.")
    
    print(f"\n📖 API Documentation: {API_BASE_URL}/docs")
    print(f"🔍 Interactive API: {API_BASE_URL}/redoc")

if __name__ == "__main__":
    main()
