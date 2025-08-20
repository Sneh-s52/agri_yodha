# market_agent/data_layer/mongo_client.py

import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import sys
import os

# Add the project root to the Python path to allow imports from other directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

# --- Best Practice: Centralized Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MongoTSClient:
    """
    A client to interact with the MongoDB time-series database for market data.
    This class is designed to be instantiated once and reused across the application.
    """

    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        """
        Initializes the MongoDB client using credentials from the config file.

        Args:
            mongo_uri (str): The connection string for MongoDB.
            db_name (str): The name of the database.
            collection_name (str): The name of the collection.
        """
        self.client = None
        self.collection = None
        try:
            if not mongo_uri:
                logger.warning("MONGO_URI is not set. MongoDB functionality will be disabled.")
                self.client = None
                self.db = None
                self.collection = None
                return

            # The pymongo driver handles SSL/TLS automatically with 'mongodb+srv://' URIs.
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]

            # The ping command is a robust way to verify a successful connection.
            self.client.admin.command("ping")
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.error(f"Could not connect to MongoDB: {e}. Continuing without MongoDB functionality.")
            # Don't raise exception, just continue without MongoDB
            self.client = None
            self.db = None
            self.collection = None

    def get_latest_price(self, commodity: str, apmc_name: str) -> Dict[str, Any]:
        """
        Fetches the most recent price document for a commodity in a specific APMC.
        Note: Fallback logic is best handled by the agent's tools, not the data layer.
        This function now performs a precise, case-insensitive search.
        """
        if not self.collection:
            logger.warning("MongoDB not available. Returning empty result.")
            return {}
        if self.collection is None:
            return {"error": "Database connection is not available."}

        try:
            # Use case-insensitive regex for more flexible matching on the whole string.
            latest_price = self.collection.find_one(
                {
                    "commodity": {"$regex": f"^{commodity}$", "$options": "i"},
                    "apmc": {"$regex": f"^{apmc_name}$", "$options": "i"}
                },
                sort=[("created_at", -1)]
            )

            if latest_price:
                latest_price['_id'] = str(latest_price['_id'])
                return latest_price

            return {"message": f"No exact price data found for {commodity} in {apmc_name}."}

        except Exception as e:
            logger.error(f"Error fetching latest price for {commodity} at {apmc_name}: {e}")
            return {"error": "An internal error occurred while fetching the latest price."}

    def get_price_history(self, commodity: str, apmc_name: str, days: int = 365) -> pd.DataFrame:
        if not self.collection:
            logger.warning("MongoDB not available. Returning empty DataFrame.")
            return pd.DataFrame()
        """
        Fetches price history for a given period and returns a Pandas DataFrame.
        """
        if self.collection is None:
            logger.error("Database connection not available for price history.")
            return pd.DataFrame()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        query = {
            "commodity": {"$regex": f"^{commodity}$", "$options": "i"},
            "apmc": {"$regex": f"^{apmc_name}$", "$options": "i"},
            "created_at": {"$gte": start_date, "$lte": end_date}
        }

        try:
            cursor = self.collection.find(query).sort("created_at", 1)
            df = pd.DataFrame(list(cursor))

            if not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df.set_index('created_at')
                # For forecasting, it's best to return only the target variable.
                if 'modal_price' in df.columns:
                    return df[['modal_price']]
                else:
                    logger.warning("'modal_price' column not found in historical data.")
                    return pd.DataFrame()

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching price history from MongoDB: {e}")
            return pd.DataFrame()

    def get_distinct_values(self, field: str, query: Dict = None) -> List[str]:
        if not self.collection:
            logger.warning("MongoDB not available. Returning empty list.")
            return []
        """
        Generic function to get a list of unique values for a given field.
        """
        if self.collection is None:
            return []
        try:
            values = self.collection.distinct(field, query or {})
            return [str(val) for val in values if val]
        except Exception as e:
            logger.error(f"Error fetching distinct values for field '{field}': {e}")
            return []

    def close_connection(self):
        if not self.client:
            return
        """
        Close the MongoDB connection.
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")


# --- Best Practice: Create a Singleton Instance ---
# This creates one single, reusable client instance for the entire application.
# Other modules (like your tools) should import this instance directly.
# This prevents repeated connections and the SSL handshake errors.
try:
    mongo_ts_client = MongoTSClient(
        mongo_uri=MONGO_URI,
        db_name=MONGO_DB_NAME,
        collection_name=MONGO_COLLECTION_NAME
    )
except Exception:
    mongo_ts_client = None  # Ensure the variable exists even if connection fails