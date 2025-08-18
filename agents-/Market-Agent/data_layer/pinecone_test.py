from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY
from data_layer.pinecone_client import PineconeVectorClient
from openai import OpenAI
from tools.embeddings import get_embedding

client = OpenAI(api_key=OPENAI_API_KEY)


def main():
    # Initialize the Pinecone client
    pc_client = PineconeVectorClient(api_key=PINECONE_API_KEY, index_host=PINECONE_INDEX_NAME)

    # Exact APMC lookup via vector similarity
    result = pc_client.get_latest_price("WHEAT", "Karnal")
    # e.g.
    # {
    #   "commodity": "WHEAT",
    #   "apmc": "Karnal APMC",
    #   "state": "Haryana",
    #   "modal_price": 2500,
    #   "Commodity_Uom": "Quintal",
    #   "created_at": "2025-08-17T00:00:00",
    #   "score": 0.92,
    #   "source": "vector_exact"
    # }
    print(result)

    # If exact APMC not found, find in same state
    fallback = pc_client._find_fallback_by_state("WHEAT", "NonExistentAPMC")
    # e.g.
    # {
    #   "commodity": "WHEAT",
    #   "apmc": "Ambala APMC",
    #   "state": "Haryana",
    #   "modal_price": 2480,
    #   "Commodity_Uom": "Quintal",
    #   "created_at": "2025-08-16T00:00:00",
    #   "score": 0.85,
    #   "source": "vector_state_fallback"
    # }
    print(fallback)

    # Get daily modal prices for last 30 days
    df = pc_client.get_price_history("RICE", "Delhi")
    # DataFrame indexed by date with one column 'modal_price'
    print(df.head())
    #                     modal_price
    # created_at
    # 2025-07-18             2200.0
    # 2025-07-19             2225.0
    # ...

    # Get historical prices for a commodity in a specific APMC
    # Get daily modal prices for last 30 days
    df = pc_client.get_price_history("RICE", "Delhi")
    # DataFrame indexed by date with one column 'modal_price'
    print(df.head())
    #                     modal_price
    # created_at
    # 2025-07-18             2200.0
    # 2025-07-19             2225.0
    # ...
    # If no history for that APMC, fallback to state-level history
    df_fb = pc_client._get_price_history_fallback("RICE", "UnknownAPMC", days=30)
    print(df_fb.tail())

    # List all unique commodities in the index
    commodities = pc_client.get_all_commodities()
    print(f"Available commodities ({len(commodities)}):", commodities[:10])
    # e.g. ['WHEAT', 'RICE', 'ONION', ...]
    # List all APMCs
    apmcs = pc_client.get_all_apmcs()
    print(f"All APMCs ({len(apmcs)}):", apmcs[:10])

    # Or filter by state
    har_apmcs = pc_client.get_all_apmcs(state="Haryana")
    print("Haryana APMCs:", har_apmcs)

    # Compute 7-day summary stats for WHEAT
    summary = pc_client.get_market_summary("WHEAT", days=7)
    print(summary)
    # e.g.
    # {
    #   "commodity": "WHEAT",
    #   "avg_price": 2490.3,
    #   "min_price": 2450,
    #   "max_price": 2525,
    #   "data_points": 7,
    #   "days": 7
    # }
    # Assuming you have some event embedding
    event_vec = get_embedding("monsoon effect on rice prices")
    matches = pc_client.vector_search_similar_events(event_vec, top_k=5)
    for m in matches:
        print(m.id, m.score, m.metadata)

    # Natural language query
    results = pc_client.semantic_price_lookup("What was the highest wheat price in Karnataka last month?", top_k=5)
    print(results)
    # e.g. list of messages or metadata dicts once embedding is implemented.


if __name__ == "__main__":
    main()