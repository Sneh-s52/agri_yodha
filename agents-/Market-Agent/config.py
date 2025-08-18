import os
from dotenv import load_dotenv
load_dotenv()

# --- API Keys ---
# Key for the OpenAI model (o3)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_MARKET")
# Key for the Pinecone vector database
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_MARKET")
# Key for the historical data API (2007-2022)
COMMODITY_DATA_GOV_KEY = os.getenv("COMMODITY_DATA_GOV_KEY_MARKET")
# Key for a web search tool, useful for the news capability
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY_MARKET")

MANDI_API_KEY = os.getenv("MANDI_API_KEY_MARKET")
MANDI_API_BASE_URL = os.getenv("MANDI_API_BASE_URL_MARKET")

# --- Pinecone Vector DB Configuration ---
# The name of your index in Pinecone
PINECONE_INDEX_NAME = "mandi-prices-vector"


# --- MongoDB Time-Series DB Configuration ---
# Your MongoDB connection string [1]
MONGO_URI = os.getenv("MONGO_URI_MARKET")
# The name of the database for time-series data
MONGO_DB_NAME = "mandi_prices"
# The name of the collection holding the commodity prices [2]
MONGO_COLLECTION_NAME = "commodity_prices"


# --- Model Configuration ---
# The main reasoning model for the agent
AGENT_MODEL = "gpt-o3"
# The model used for generating vector embeddings
EMBEDDING_MODEL = "text-embedding-3-small"


# --- Data File Paths ---
# Path to the local CSV file containing geocoded APMC data
APMC_GEO_DATA_PATH = "Market_agent/data/apmc_geo_data.csv"