import os
import json
import logging
from pymongo import MongoClient
from dotenv import load_dotenv

# LangChain components
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()

# Configure logging with UTF-8 encoding
file_handler = logging.FileHandler("rag_log.txt", encoding="utf-8")
stream_handler = logging.StreamHandler()
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s',
                    handlers=[file_handler, stream_handler])

# MongoDB Connection - UPDATED KEY NAME
MONGO_CONNECTION_STRING = os.getenv("MONGODB_URI_SOIL")
DB_NAME = "soil_and_health"
CLAUSE_COLLECTION_NAME = "clauses"

# Models - UPDATED KEY NAMES
# LangChain automatically looks for OPENAI_API_KEY and TAVILY_API_KEY.
# To use a custom key name, you must pass it to the constructor.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY_SOIL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY_SOIL")

OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'
LLM_MODEL = "gpt-4o"

# --- 2. DATA LOADING AND INDEXING ---
def load_data_from_mongo(collection):
    """Loads all documents from a MongoDB collection."""
    logging.info(f"Loading documents from '{collection.name}' collection...")
    return list(collection.find({}, {"clause_text": 1, "metadata": 1, "_id": 0}))

def setup_retrievers(documents):
    """Creates a hybrid retriever (Dense + Sparse) from the documents."""
    logging.info("Setting up retrievers...")
    
    for doc_dict in documents:
        doc_dict['page_content'] = doc_dict.pop('clause_text', '')

    langchain_documents = [Document(page_content=d['page_content'], metadata=d['metadata']) for d in documents]

    bm25_retriever = BM25Retriever.from_documents(langchain_documents)
    bm25_retriever.k = 5 

    embedding_function = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(langchain_documents, embedding_function)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5] 
    )
    logging.info("✅ Hybrid retriever is ready.")
    return ensemble_retriever

# --- 3. RAG CHAIN COMPONENTS ---

class RefinedQuery(BaseModel):
    db_query: str = Field(description="A detailed, rewritten query for a vector database search.")
    web_query: str = Field(description="A concise query for a web search engine, or an empty string if not needed.")

query_refiner_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at rewriting user queries for a vector database and a web search engine. "
               "Based on the user query, generate a database query and a web search query to retrieve relevant information.\n"
               "{format_instructions}"),
    ("user", "User Query: {query}")
])

web_search_tool = TavilySearch(k=3)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert agricultural assistant. Answer the user's question based *only* on the provided context from our database and web search results.\n"
               "If the context does not contain the answer, state that clearly. Be concise and helpful.\n\n"
               "--- CONTEXT --- \n{context}"),
    ("user", "Question: {query}")
])

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client[DB_NAME]
        clauses_collection = db[CLAUSE_COLLECTION_NAME]
        documents = load_data_from_mongo(clauses_collection)
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB or load data: {e}")
        exit()

    if not documents:
        logging.error("No documents loaded from the database. Halting.")
        exit()

    hybrid_retriever = setup_retrievers(documents)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    
    # --- Build the Full RAG Chain (Corrected Version) ---
    
    # 1. Define the query refiner chain
    query_parser = JsonOutputParser(pydantic_object=RefinedQuery)
    refiner_prompt_with_instructions = query_refiner_prompt.partial(format_instructions=query_parser.get_format_instructions())
    refiner_chain = refiner_prompt_with_instructions | llm | query_parser

    # 2. Define the retrieval function
    def retrieve_context(state: dict):
        """Combines DB retrieval and web search based on the refined queries."""
        db_query = state['refined_queries']['db_query']
        web_query = state['refined_queries']['web_query']
        
        # --- Database context ---
        db_context_docs = hybrid_retriever.invoke(db_query)
        db_context = "\n\n".join([doc.page_content for doc in db_context_docs])
        
        # --- Web context ---
        web_context = ""
        if web_query:
            logging.info(f"Performing web search for: '{web_query}'")
            web_results = web_search_tool.invoke(web_query)

            if web_results:
                # Case 1: Tavily returned a dict
                if isinstance(web_results, dict):
                    results_list = web_results.get("results", [])
                    web_context = "\n\n".join([item.get("content", str(item)) for item in results_list])
                # Case 2: Tavily returned a list
                elif isinstance(web_results, list):
                    if isinstance(web_results[0], str):
                        web_context = "\n\n".join(web_results)
                    else:
                        web_context = "\n\n".join([item.get("content", str(item)) for item in web_results])
                # Case 3: Unexpected type
                else:
                    web_context = str(web_results)

        return f"--- From Database ---\n{db_context}\n\n--- From Web Search ---\n{web_context}"




    # 3. Assemble the full chain using a more robust method
    full_rag_chain = (
        RunnablePassthrough.assign(refined_queries=refiner_chain)
        | RunnablePassthrough.assign(context=retrieve_context)
        | RunnableLambda(lambda x: logging.info(f"--- LOGGING CONTEXT ---\nQuery: {x['query']}\nContext: {x['context']}\n--- END LOG ---") or x)
        | final_prompt
        | llm
        | StrOutputParser()
    )

    # --- 5. RUN AN EXAMPLE ---
    logging.info("\n" + "="*50 + "\n🚀 RAG System Ready. Ask a question.\n" + "="*50)
    
    user_question = "What is Azotobacter used in? for what crops and where are those crops grown?"
    
    # The input to the chain is now a dictionary
    final_answer = full_rag_chain.invoke({"query": user_question})
    
    print("\n" + "="*50 + "\n✅ FINAL ANSWER:\n" + "="*50)
    print(final_answer)
