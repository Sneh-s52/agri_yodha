from openai import OpenAI
from config import OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text: str):
    """Get OpenAI embedding"""
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return None