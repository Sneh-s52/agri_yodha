# market_agent/data_layer/pinecone_client.py

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from tools.embeddings import get_embedding

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeVectorClient:
    """
    A client to interact with the Pinecone vector database using the gRPC interface.
    """

    def __init__(self, api_key: str, index_name: str):
        """
        Initializes the Pinecone client and connects to the specified index.

        Args:
            api_key (str): The API key for Pinecone.
            index_name (str): The name of the Pinecone index.
        """
        self.api_key = api_key
        self.index_name = index_name

        try:
            if not api_key or not index_name:
                raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set in the environment.")

            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index(index_name)
            # Verify connection by getting index stats
            self.index.describe_index_stats()
            logger.info("Successfully connected to Pinecone index.")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            self.index = None
            raise ConnectionError(f"Failed to connect to Pinecone: {e}")

    def get_latest_price(self, commodity: str, apmc_name: str) -> Dict[str, Any]:
        """
        Fetches the most recent price document for a commodity in a specific APMC using vector similarity.
        Includes fallback to nearby APMCs if exact match not found.

        Args:
            commodity (str): The name of the commodity.
            apmc_name (str): The name of the APMC market.

        Returns:
            dict: The latest price document or an error message.
        """
        if not self.index:
            return {"error": "Pinecone index not available."}

        try:
            # 1. Build prompt and embed
            query_text = f"latest price {commodity} at {apmc_name} APMC"
            query_vector = get_embedding(query_text)  # your embedding function

            if query_vector is None:
                return {"error": "Failed to generate embedding for query."}

            # 2. Query Pinecone for the most similar vector
            response = self.index.query(
                vector=query_vector,
                top_k=5,
                include_metadata=True
            )

            matches = response.matches
            if matches:
                match = matches[0]
                metadata = match.metadata or {}
                # Convert metadata fields as needed
                return {
                    "commodity": metadata.get("commodity"),
                    "apmc": metadata.get("apmc"),
                    "state": metadata.get("state"),
                    "modal_price": metadata.get("modal_price"),
                    "Commodity_Uom": metadata.get("Commodity_Uom"),
                    "created_at": metadata.get("created_at"),
                    "score": match.score,
                    "source": "vector_exact"
                }

            # 3. Fallback: vector search by commodity only (any APMC)
            query_text = f"latest price {commodity}"
            query_vector = get_embedding(query_text)
            if query_vector:
                response = self.index.query(
                    vector=query_vector,
                    top_k=1,
                    include_metadata=True
                )
                matches = response.matches
                if matches:
                    match = matches[0]
                    metadata = match.metadata or {}
                    return {
                        "commodity": metadata.get("commodity"),
                        "apmc": metadata.get("apmc"),
                        "state": metadata.get("state"),
                        "modal_price": metadata.get("modal_price"),
                        "Commodity_Uom": metadata.get("Commodity_Uom"),
                        "created_at": metadata.get("created_at"),
                        "score": match.score,
                        "source": "vector_commodity_fallback"
                    }

            return {"message": f"No vector-match data found for {commodity} at {apmc_name} or fallback."}

        except Exception as e:
            logger.error(f"Error fetching latest price from Pinecone: {e}")
            return {"error": f"An error occurred while fetching the latest price: {e}"}

    def _find_fallback_by_state(self, commodity: str, apmc_name: str) -> Optional[Dict[str, Any]]:
        """
        Helper method to find commodity price in the same state when exact APMC not found using vector similarity.
        """
        if not self.index:
            return None

        try:
            # 1. Get embedding for the fallback query
            query_text = f"latest {commodity} price in the state of {apmc_name}"
            query_vector = get_embedding(query_text)
            if not query_vector:
                return None

            # 2. Query Pinecone for top-1 result
            resp = self.index.query(
                vector=query_vector,
                top_k=1,
                include_metadata=True
            )
            matches = resp.matches
            if matches:
                m = matches[0]
                md = m.metadata or {}
                return {
                    "commodity": md.get("commodity"),
                    "apmc": md.get("apmc"),
                    "state": md.get("state"),
                    "modal_price": md.get("modal_price"),
                    "Commodity_Uom": md.get("Commodity_Uom"),
                    "created_at": md.get("created_at"),
                    "score": m.score,
                    "source": "vector_state_fallback"
                }
            return None

        except Exception as e:
            logger.error(f"Error in Pinecone state fallback search: {e}")
            return None

    def get_price_history(self, commodity: str, apmc_name: str, days: int = 365) -> pd.DataFrame:
        """
        Fetches price history for a given period using vector similarity search.
        """
        if not self.index:
            logger.error("Pinecone index not available.")
            return pd.DataFrame()

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # 1. Get embedding for history query
            query_text = f"{commodity} price history at {apmc_name} for last {days} days"
            query_vector = get_embedding(query_text)
            if not query_vector:
                return pd.DataFrame()

            # 2. Query Pinecone
            resp = self.index.query(
                vector=query_vector,
                top_k=days,  # assume one per day max
                include_metadata=True
            )

            # 3. Build DataFrame from metadata
            rows = []
            for m in resp.matches:
                md = m.metadata or {}
                created = md.get("created_at")
                price = md.get("modal_price")
                if created and price is not None:
                    rows.append({
                        "created_at": datetime.fromisoformat(created),
                        "modal_price": price
                    })

            if not rows:
                return self._get_price_history_fallback(commodity, apmc_name, days)

            df = pd.DataFrame(rows).sort_values("created_at").set_index("created_at")
            return df

        except Exception as e:
            logger.error(f"Error fetching price history from Pinecone: {e}")
            return pd.DataFrame()

    def _get_price_history_fallback(self, commodity: str, apmc_name: str, days: int) -> pd.DataFrame:
        """
        Fallback method to get price history from same state if exact APMC not found.
        """
        if not self.index:
            return pd.DataFrame()

        try:
            # Determine state-level fallback
            state_fallback = self._find_fallback_by_state(commodity, apmc_name)
            if not state_fallback:
                return pd.DataFrame()

            state = state_fallback.get("state")
            query_text = f"{commodity} price history in state {state} for last {days} days"
            query_vector = get_embedding(query_text)
            if not query_vector:
                return pd.DataFrame()

            resp = self.index.query(
                vector=query_vector,
                top_k=days,
                include_metadata=True
            )

            rows = []
            for m in resp.matches:
                md = m.metadata or {}
                created = md.get("created_at")
                price = md.get("modal_price")
                if created and price is not None:
                    rows.append({
                        "created_at": datetime.fromisoformat(created),
                        "modal_price": price
                    })

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows).sort_values("created_at").set_index("created_at")
            return df

        except Exception as e:
            logger.error(f"Error in Pinecone price history fallback: {e}")
            return pd.DataFrame()

    def get_all_commodities(self) -> List[str]:
        if not self.index:
            return []

        try:
            stats = self.index.describe_index_stats()
            dim = stats.get("dimension") or stats.get("dims") or 0
            if not dim:
                logger.warning("Index stats missing 'dimension'; cannot sample metadata.")
                return []

            # Query with a zero vector of correct length
            resp = self.index.query(
                vector=[0.0] * dim,
                top_k=1000,
                include_metadata=True
            )

            commodities = {m.metadata.get("commodity") for m in resp.matches if m.metadata}
            return [c for c in commodities if c]

        except Exception as e:
            logger.error(f"Error fetching commodities from Pinecone: {e}")
            return []

    def get_all_apmcs(self, state: str = None) -> List[str]:
        if not self.index:
            return []

        try:
            stats = self.index.describe_index_stats()
            dim = stats.get("dimension") or stats.get("dims") or 0
            if not dim:
                return []

            resp = self.index.query(
                vector=[0.0] * dim,
                top_k=1000,
                include_metadata=True
            )

            apmcs = set()
            for m in resp.matches:
                md = m.metadata or {}
                apmc = md.get("apmc")
                st = md.get("state")
                if apmc and (state is None or (st and st.lower() == state.lower())):
                    apmcs.add(apmc)
            return list(apmcs)

        except Exception as e:
            logger.error(f"Error fetching APMCs from Pinecone: {e}")
            return []

    def get_market_summary(self, commodity: str, days: int = 7) -> Dict[str, Any]:
        if not self.index:
            return {"error": "Pinecone index not available."}

        try:
            # Query embedding
            query_text = f"{commodity} market summary last {days} days"
            query_vector = get_embedding(query_text)
            if not query_vector:
                return {"error": "Failed to generate embedding."}

            resp = self.index.query(
                vector=query_vector,
                top_k=days,
                include_metadata=True
            )

            prices = []
            for m in resp.matches:
                md = m.metadata or {}
                price = md.get("modal_price")
                # Cast to float if it's string
                try:
                    price = float(price)
                except (TypeError, ValueError):
                    continue
                prices.append(price)

            if not prices:
                return {"message": f"No data for summary of {commodity}"}

            return {
                "commodity": commodity,
                "avg_price": sum(prices) / len(prices),
                "min_price": min(prices),
                "max_price": max(prices),
                "data_points": len(prices),
                "days": days
            }

        except Exception as e:
            logger.error(f"Error generating market summary from Pinecone: {e}")
            return {"error": f"Error generating market summary: {e}"}

    def vector_search_similar_events(self, query_vector: list, top_k: int = 5) -> list:
        if not self.index:
            return [{"error": "Pinecone index not available."}]

        try:
            resp = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            return resp.matches  # ensure this method exists on your Pinecone client
        except Exception as e:
            return [{"error": f"Pinecone vector search failed: {e}"}]

    def semantic_price_lookup(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.index:
            return [{"error": "Pinecone index not available."}]

        try:
            query_vector = get_embedding(query)
            if not query_vector:
                return [{"error": "Failed to generate embedding."}]

            resp = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )

            results = []
            for m in resp.matches:
                md = m.metadata or {}
                results.append({
                    "commodity": md.get("commodity"),
                    "apmc": md.get("apmc"),
                    "state": md.get("state"),
                    "modal_price": md.get("modal_price"),
                    "score": m.score
                })
            return results

        except Exception as e:
            logger.error(f"Error in semantic price lookup: {e}")
            return [{"error": f"Semantic search failed: {e}"}]
    def close_connection(self):
        """
        Close the Pinecone connection.
        """
        if self.pc:
            # Pinecone gRPC connections are automatically managed
            logger.info("Pinecone connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()


# Factory function to create client instance
def create_pinecone_client(api_key: str, index_name: str) -> PineconeVectorClient:
    """
    Factory function to create and return a PineconeVectorClient instance.

    Args:
        api_key (str): Pinecone API key.
        index_name (str): Pinecone index name.

    Returns:
        PineconeVectorClient: Configured client instance.
    """
    return PineconeVectorClient(api_key, index_name)


# Singleton instance for the application
try:
    pinecone_vector_client = PineconeVectorClient(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX_NAME
    )
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {e}")
    pinecone_vector_client = None