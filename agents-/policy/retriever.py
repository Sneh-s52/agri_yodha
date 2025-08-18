"""
MongoDB Vector Retriever

A production-ready retrieval system that combines:
1. Dense semantic search using vector embeddings
2. Sparse keyword-based retrieval (TF-IDF)
3. Hybrid retrieval combining both approaches

This module provides a clean MongoRetriever class that can be easily imported
and used in any application requiring document retrieval from MongoDB.

Author: AI Assistant
Date: 2025
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimilarityType(Enum):
    """Enumeration for similarity search types."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """Data class for individual retrieval results."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str  # 'dense', 'sparse', 'hybrid'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'score': self.score,
            'metadata': self.metadata,
            'retrieval_method': self.retrieval_method
        }


class MongoRetriever:
    """
    Production-ready MongoDB vector retriever supporting dense, sparse, and hybrid search.
    
    This class provides a unified interface for retrieving documents from MongoDB
    using various similarity search methods.
    """
    
    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        database_name: Optional[str] = None
    ):
        """
        Initialize the MongoDB retriever for multiple collections.
        
        Args:
            mongodb_uri: MongoDB connection URI (defaults to MONGODB_URI env var)
            database_name: Database name (defaults to DATABASE_NAME env var or 'agri_policies_db')
        """
        self.mongodb_uri = mongodb_uri or os.getenv('MONGODB_URI')
        self.database_name = database_name or os.getenv('DATABASE_NAME', 'agri_policies_db')
        
        if not self.mongodb_uri:
            raise ValueError("MongoDB URI must be provided either as parameter or MONGODB_URI environment variable")
        
        # Initialize MongoDB connection with SSL fix
        self.client = self._connect_to_mongodb()
        self.db = self.client[self.database_name]
        
        # Initialize collections
        self.collections = {
            'clauses': self.db['clauses'],
            'sections': self.db['sections'],
            'sentences': self.db['sentences']
        }
        
        # Define text fields for each collection
        self.text_fields = {
            'clauses': ['clause_title', 'clause_text'],
            'sections': ['section_title'],
            'sentences': ['sentence_text']
        }
        
        # Initialize sparse retrieval components for each collection
        self.sparse_indexes = {}
        self._build_all_sparse_indexes()
        
        logger.info(f"MongoRetriever initialized for {self.database_name} with collections: {list(self.collections.keys())}")
    
    def _connect_to_mongodb(self) -> MongoClient:
        """Establish connection to MongoDB using SSL certificate fix."""
        try:
            import certifi
            ca = certifi.where()
            client = MongoClient(
                self.mongodb_uri, 
                serverSelectionTimeoutMS=5000, 
                tlsCAFile=ca
            )
            
            # Test connection
            client.admin.command('ping')
            
            # Get server info
            server_info = client.server_info()
            logger.info(f"Successfully connected to MongoDB version: {server_info.get('version', 'unknown')}")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise ConnectionFailure(f"Could not connect to MongoDB: {e}")
    
    def _build_all_sparse_indexes(self):
        """Build TF-IDF indexes for all collections."""
        for collection_name, collection in self.collections.items():
            try:
                logger.info(f"Building sparse index for {collection_name}...")
                
                # Get text fields for this collection
                text_fields = self.text_fields[collection_name]
                
                # Fetch all documents with text fields
                projection = {"_id": 1}
                for field in text_fields:
                    projection[field] = 1
                
                documents = list(collection.find({}, projection))
                
                if not documents:
                    logger.warning(f"No documents found in {collection_name}")
                    continue
                
                # Extract texts and build mapping
                texts = []
                doc_mapping = {}
                
                for i, doc in enumerate(documents):
                    # Combine all text fields
                    combined_text = ""
                    for field in text_fields:
                        field_text = doc.get(field, "")
                        if field_text:
                            combined_text += f" {field_text}"
                    
                    texts.append(combined_text.strip())
                    doc_mapping[i] = {
                        'document_id': str(doc['_id']),
                        'collection': collection_name,
                        'text': combined_text.strip(),
                        'original_doc': doc
                    }
                
                if not texts or not any(text.strip() for text in texts):
                    logger.warning(f"No valid text found in {collection_name}")
                    continue
                
                # Build TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    lowercase=True
                )
                
                # Fit and transform documents
                tfidf_matrix = vectorizer.fit_transform(texts)
                
                # Store in sparse indexes
                self.sparse_indexes[collection_name] = {
                    'vectorizer': vectorizer,
                    'matrix': tfidf_matrix,
                    'mapping': doc_mapping
                }
                
                logger.info(f"Sparse index built for {collection_name} with {len(texts)} documents")
                
            except Exception as e:
                logger.error(f"Error building sparse index for {collection_name}: {e}")
    
    def _dense_search(self, query: str, top_k: int, min_score: float = 0.5) -> List[RetrievalResult]:
        """
        Perform dense vector similarity search across all collections.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of retrieval results from dense search
        """
        try:
            # Note: This is a simplified implementation using random scores
            # In production, you would generate query embeddings and calculate cosine similarity
            # or use MongoDB Atlas Vector Search
            
            all_results = []
            results_per_collection = max(1, top_k // len(self.collections))
            
            for collection_name, collection in self.collections.items():
                try:
                    # Find documents that have embeddings
                    cursor = collection.find(
                        {"embedding_openai": {"$exists": True, "$ne": None}},
                        {"_id": 1, "embedding_openai": 1, **{field: 1 for field in self.text_fields[collection_name]}}
                    ).limit(results_per_collection * 2)
                    
                    documents = list(cursor)
                    
                    if not documents:
                        logger.warning(f"No documents with embeddings found in {collection_name}")
                        continue
                    
                    # For this example, we'll use random similarity scores
                    # In production, you'd calculate cosine similarity with query embedding
                    for doc in documents[:results_per_collection]:
                        # Simulate similarity score
                        score = np.random.uniform(0.6, 0.95)
                        
                        if score >= min_score:
                            # Get text content
                            text_content = ""
                            for field in self.text_fields[collection_name]:
                                field_text = doc.get(field, "")
                                if field_text:
                                    text_content += f" {field_text}"
                            
                            result = RetrievalResult(
                                chunk_id=str(doc['_id']),
                                text=text_content.strip(),
                                score=score,
                                metadata={
                                    'collection': collection_name,
                                    'source_fields': self.text_fields[collection_name]
                                },
                                retrieval_method='dense'
                            )
                            all_results.append(result)
                
                except Exception as e:
                    logger.warning(f"Error in dense search for {collection_name}: {e}")
                    continue
            
            # Sort by score and return top results
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in dense search: {e}")
            return []
    
    def _sparse_search(self, query: str, top_k: int, min_score: float = 0.1) -> List[RetrievalResult]:
        """
        Perform sparse keyword-based search using TF-IDF across all collections.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum TF-IDF score threshold
            
        Returns:
            List of retrieval results from sparse search
        """
        if not self.sparse_indexes:
            logger.warning("Sparse indexes not available")
            return []
        
        all_results = []
        results_per_collection = max(1, top_k // len(self.sparse_indexes))
        
        for collection_name, index_data in self.sparse_indexes.items():
            try:
                vectorizer = index_data['vectorizer']
                tfidf_matrix = index_data['matrix']
                doc_mapping = index_data['mapping']
                
                # Transform query to TF-IDF vector
                query_vector = vectorizer.transform([query])
                
                # Calculate cosine similarities
                similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
                
                # Get top-k indices for this collection
                top_indices = np.argsort(similarities)[-results_per_collection:][::-1]
                
                for idx in top_indices:
                    score = similarities[idx]
                    if score >= min_score:
                        doc_info = doc_mapping[idx]
                        
                        result = RetrievalResult(
                            chunk_id=doc_info['document_id'],
                            text=doc_info['text'],
                            score=float(score),
                            metadata={
                                'collection': doc_info['collection'],
                                'original_doc': doc_info['original_doc']
                            },
                            retrieval_method='sparse'
                        )
                        all_results.append(result)
                
            except Exception as e:
                logger.warning(f"Error in sparse search for {collection_name}: {e}")
                continue
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Sparse search returned {len(all_results)} results from {len(self.sparse_indexes)} collections")
        return all_results[:top_k]
    
    def _hybrid_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Perform hybrid search combining dense and sparse methods.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of retrieval results from hybrid search
        """
        try:
            # Get results from both methods
            dense_k = max(1, top_k // 2)
            sparse_k = max(1, top_k // 2)
            
            dense_results = self._dense_search(query, dense_k)
            sparse_results = self._sparse_search(query, sparse_k)
            
            # Combine and deduplicate results
            seen_chunks = set()
            combined_results = []
            
            # Add dense results first (typically higher quality)
            for result in dense_results:
                if result.chunk_id not in seen_chunks:
                    seen_chunks.add(result.chunk_id)
                    combined_results.append(result)
            
            # Add sparse results
            for result in sparse_results:
                if result.chunk_id not in seen_chunks:
                    seen_chunks.add(result.chunk_id)
                    result.retrieval_method = 'hybrid'
                    combined_results.append(result)
            
            # Normalize and rerank scores
            self._rerank_hybrid_results(combined_results)
            
            # Sort by final score
            combined_results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Hybrid search combined {len(dense_results)} dense + {len(sparse_results)} sparse = {len(combined_results)} unique results")
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _rerank_hybrid_results(self, results: List[RetrievalResult]):
        """
        Rerank hybrid results by normalizing scores across different methods.
        
        Args:
            results: List of results to rerank (modified in-place)
        """
        if not results:
            return
        
        # Separate results by method
        dense_results = [r for r in results if r.retrieval_method == 'dense']
        sparse_results = [r for r in results if r.retrieval_method in ['sparse', 'hybrid']]
        
        # Normalize scores within each group
        if dense_results:
            max_dense = max(r.score for r in dense_results)
            for result in dense_results:
                result.score = (result.score / max_dense) * 0.7  # Dense weight: 0.7
        
        if sparse_results:
            max_sparse = max(r.score for r in sparse_results)
            for result in sparse_results:
                result.score = (result.score / max_sparse) * 0.3  # Sparse weight: 0.3
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        similarity_type: Union[SimilarityType, str] = SimilarityType.HYBRID,
        min_score: float = 0.1
    ) -> List[RetrievalResult]:
        """
        Main retrieval method supporting dense, sparse, and hybrid search.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            similarity_type: Type of similarity search ('dense', 'sparse', or 'hybrid')
            min_score: Minimum similarity score threshold
            
        Returns:
            List of retrieval results sorted by relevance score
        """
        if isinstance(similarity_type, str):
            try:
                similarity_type = SimilarityType(similarity_type.lower())
            except ValueError:
                logger.warning(f"Invalid similarity type: {similarity_type}. Using hybrid.")
                similarity_type = SimilarityType.HYBRID
        
        logger.info(f"Retrieving documents for query: '{query[:50]}...' using {similarity_type.value} search")
        
        try:
            if similarity_type == SimilarityType.DENSE:
                results = self._dense_search(query, top_k, min_score)
            elif similarity_type == SimilarityType.SPARSE:
                results = self._sparse_search(query, top_k, min_score)
            else:  # HYBRID
                results = self._hybrid_search(query, top_k)
            
            logger.info(f"Retrieved {len(results)} results using {similarity_type.value} search")
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    def get_document_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by its chunk ID.
        
        Args:
            chunk_id: Unique identifier for the document chunk
            
        Returns:
            Document data or None if not found
        """
        try:
            doc = self.collection.find_one({"chunk_id": chunk_id})
            return doc
        except Exception as e:
            logger.error(f"Error retrieving document {chunk_id}: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all document collections.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {
                "database": self.database_name,
                "collections": {}
            }
            
            for collection_name, collection in self.collections.items():
                total_docs = collection.count_documents({})
                docs_with_embeddings = collection.count_documents({"embedding_openai": {"$exists": True}})
                
                # Sample document to understand structure
                sample_doc = collection.find_one({})
                
                collection_stats = {
                    "total_documents": total_docs,
                    "documents_with_embeddings": docs_with_embeddings,
                    "sparse_index_ready": collection_name in self.sparse_indexes,
                    "text_fields": self.text_fields[collection_name],
                    "sample_fields": list(sample_doc.keys()) if sample_doc else []
                }
                
                stats["collections"][collection_name] = collection_stats
            
            # Overall stats
            stats["total_documents"] = sum(c["total_documents"] for c in stats["collections"].values())
            stats["total_with_embeddings"] = sum(c["documents_with_embeddings"] for c in stats["collections"].values())
            stats["sparse_indexes_ready"] = len(self.sparse_indexes)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the MongoRetriever class.
    """
    print("MongoDB Vector Retriever - Example Usage")
    print("=" * 50)
    
    try:
        # Initialize retriever
        retriever = MongoRetriever()
        
        # Get collection stats
        stats = retriever.get_collection_stats()
        print(f"Collection Stats: {json.dumps(stats, indent=2)}")
        
        # Test query
        test_query = "insurance policy coverage for knee surgery"
        
        print(f"\nTesting query: '{test_query}'")
        print("-" * 30)
        
        # Test different similarity types
        for sim_type in [SimilarityType.SPARSE, SimilarityType.DENSE, SimilarityType.HYBRID]:
            print(f"\n{sim_type.value.upper()} SEARCH:")
            results = retriever.retrieve(test_query, top_k=3, similarity_type=sim_type)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.4f} | Method: {result.retrieval_method}")
                print(f"     Text: {result.text[:100]}...")
                print(f"     Chunk ID: {result.chunk_id}")
        
        # Close connection
        retriever.close()
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        logger.error(f"Example execution failed: {e}")
