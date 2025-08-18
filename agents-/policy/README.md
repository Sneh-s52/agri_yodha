# MongoDB Vector Retriever and AI Agent

A production-ready system for document retrieval and intelligent question answering using MongoDB vector search and OpenAI's language models.

## 🚀 Features

- **Hybrid Retrieval**: Combines dense semantic search and sparse keyword-based retrieval
- **MongoDB Integration**: Seamless integration with MongoDB Atlas for vector storage
- **AI-Powered Responses**: Uses OpenAI's GPT models for context-aware answers
- **Modular Design**: Clean, extensible architecture for easy integration
- **Interactive CLI**: Command-line interface for testing and interaction
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 📁 Project Structure

```
policy/
├── retriever.py          # MongoDB vector retriever module
├── agent.py             # AI agent with LLM integration
├── test_integration.py  # Integration tests
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🛠 Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <your-repo-url>
   cd policy/
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the `policy/` directory:
   ```bash
   # Required
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
   OPENAI_API_KEY=sk-your-openai-api-key-here
   
   # Optional (will use defaults if not specified)
   DATABASE_NAME=insurance_rag
   COLLECTION_NAME=policy_chunks
   ```

## 🎯 Quick Start

### Using the Retriever Only

```python
from retriever import MongoRetriever, SimilarityType

# Initialize retriever
retriever = MongoRetriever()

# Perform different types of searches
results = retriever.retrieve(
    query="insurance coverage for surgery",
    top_k=5,
    similarity_type=SimilarityType.HYBRID
)

# Process results
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text[:100]}...")
    print(f"Method: {result.retrieval_method}")
    print("---")

# Clean up
retriever.close()
```

### Using the AI Agent

```python
from agent import InsuranceAgent

# Initialize agent
agent = InsuranceAgent()

# Ask a question
response = agent.query("What does my policy cover for knee surgery?")

# Get the answer
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.1%}")
print(f"Sources: {len(response.sources)} documents")

# Clean up
agent.close()
```

### Interactive Mode

Run the interactive CLI:

```bash
python agent.py
```

This will start an interactive session where you can ask questions and get responses in real-time.

## 📚 API Reference

### MongoRetriever Class

#### Constructor
```python
MongoRetriever(
    mongodb_uri: Optional[str] = None,
    database_name: Optional[str] = None,
    collection_name: Optional[str] = None
)
```

#### Main Methods

- **`retrieve(query, top_k, similarity_type, min_score)`**: Main retrieval method
  - `query`: Natural language query string
  - `top_k`: Number of results to return (default: 10)
  - `similarity_type`: 'dense', 'sparse', or 'hybrid' (default: 'hybrid')
  - `min_score`: Minimum similarity threshold (default: 0.1)

- **`get_collection_stats()`**: Get database collection statistics
- **`get_document_by_id(chunk_id)`**: Retrieve specific document by ID
- **`close()`**: Close database connection

### InsuranceAgent Class

#### Constructor
```python
InsuranceAgent(
    openai_api_key: Optional[str] = None,
    model_name: str = "gpt-3.5-turbo",
    mongodb_uri: Optional[str] = None,
    database_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    max_context_length: int = 4000
)
```

#### Main Methods

- **`query(user_query, top_k, similarity_type, min_score)`**: Process user query and generate response
- **`batch_query(queries, **kwargs)`**: Process multiple queries in batch
- **`get_retriever_stats()`**: Get retriever statistics
- **`close()`**: Close connections and cleanup

### Response Objects

#### RetrievalResult
```python
@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str
```

#### AgentResponse
```python
@dataclass
class AgentResponse:
    query: str
    answer: str
    sources: List[RetrievalResult]
    retrieval_method: str
    response_timestamp: str
    confidence: float
```

## 🧪 Testing

Run the integration tests to verify everything is working:

```bash
python test_integration.py
```

This will:
1. Check environment variables
2. Test the retriever independently
3. Test the agent independently
4. Test full integration between components

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MONGODB_URI` | Yes | - | MongoDB connection string |
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `DATABASE_NAME` | No | `insurance_rag` | MongoDB database name |
| `COLLECTION_NAME` | No | `policy_chunks` | MongoDB collection name |

### Similarity Types

- **`dense`**: Uses vector embeddings for semantic similarity
- **`sparse`**: Uses TF-IDF for keyword-based matching
- **`hybrid`**: Combines both methods (recommended)

## 📊 Performance Considerations

### Retrieval Performance
- **Dense search**: Best for semantic understanding, requires pre-computed embeddings
- **Sparse search**: Fast keyword matching, good for specific terms
- **Hybrid search**: Balanced approach, combines strengths of both methods

### Scaling Recommendations
- Use MongoDB Atlas Vector Search for production deployments
- Implement connection pooling for high-throughput scenarios
- Consider caching frequently accessed documents
- Monitor and optimize TF-IDF index size based on document corpus

## 🔍 Troubleshooting

### Common Issues

1. **"No documents found"**
   - Check MongoDB connection and collection name
   - Verify documents exist in the specified collection
   - Ensure documents have required fields (`text`, `chunk_id`)

2. **"Sparse index not available"**
   - Check if documents contain text content
   - Verify scikit-learn is installed correctly
   - Review TF-IDF vectorizer parameters

3. **"OpenAI API errors"**
   - Verify API key is correct and has sufficient credits
   - Check rate limits and quotas
   - Ensure model name is valid (e.g., "gpt-3.5-turbo")

4. **Low retrieval quality**
   - Adjust `min_score` threshold
   - Try different `similarity_type` options
   - Increase `top_k` for more context
   - Review document preprocessing and chunking strategy

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Extending the System

### Adding New Tools
The agent architecture supports extension with additional tools:

```python
class ExtendedAgent(InsuranceAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_tools = {}  # Add your tools here
    
    def query_with_tools(self, query: str):
        # Custom logic using additional tools
        pass
```

### Custom Retrieval Methods
Extend the retriever with custom similarity functions:

```python
class CustomRetriever(MongoRetriever):
    def custom_search(self, query: str, **kwargs):
        # Implement your custom retrieval logic
        pass
```

## 📝 License

This project is provided as-is for educational and development purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the integration tests
3. Enable debug logging for detailed information
4. Open an issue with detailed error logs and environment information

