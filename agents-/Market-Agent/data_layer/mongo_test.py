# market_agent/tests/test_mongo_connection.py

import os
import sys

# Add the project root to the Python path to allow imports from other directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_mongo_test():
    """
    A simple script to test the MongoDB connection and a basic query.
    """
    print("🚀 Attempting to connect to MongoDB...")

    # This will import the singleton instance created in mongo_client.py
    # The connection error will happen here if there's a problem.
    try:
        from data_layer.mongo_client import mongo_ts_client

        if mongo_ts_client is None:
            print("❌ mongo_ts_client is None. Connection likely failed during initialization.")
            return

        print("✅ Connection successful!")
    except Exception as e:
        print(f"❌ Failed to import and initialize mongo_ts_client. Error: {e}")
        return

    print("\n📦 Testing a query to fetch a list of distinct commodities...")
    try:
        # Use the imported client instance to run a query
        commodities = mongo_ts_client.get_distinct_values("commodity")

        if commodities:
            print(f"✅ Success! Found {len(commodities)} commodities.")
            print(f"   Sample: {commodities[:5]}")
        else:
            print("⚠️ Query ran successfully, but no commodities were found in the collection.")

    except Exception as e:
        print(f"❌ An error occurred during the query: {e}")
    finally:
        # Cleanly close the connection
        mongo_ts_client.close_connection()


if __name__ == "__main__":
    run_mongo_test()