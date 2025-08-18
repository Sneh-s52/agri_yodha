# market_agent/tools/news_search_tool.py

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from tavily import TavilyClient
from config import TAVILY_API_KEY
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSearchTool:
    """
    A tool for searching market-related news using Tavily search API.
    Provides real-time news search capabilities for market intelligence.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the Tavily news search client.

        Args:
            api_key (str, optional): Tavily API key. If None, reads from config.
        """
        self.api_key = api_key or TAVILY_API_KEY

        if not self.api_key:
            raise ValueError("TAVILY_API_KEY must be provided or set in environment variables")

        try:
            self.client = TavilyClient(api_key=self.api_key)
            logger.info("Successfully initialized Tavily news search client")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily client: {e}")
            raise

    def search_market_news(
            self,
            query: str,
            max_results: int = 10,
            days_back: int = 7,
            include_domains: List[str] = None,
            exclude_domains: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for market-related news based on user query.

        Args:
            query (str): The search query for news
            max_results (int): Maximum number of results to return
            days_back (int): How many days back to search for news
            include_domains (List[str], optional): Specific domains to include in search
            exclude_domains (List[str], optional): Domains to exclude from search

        Returns:
            List[Dict[str, Any]]: List of news articles with metadata
        """
        if not query.strip():
            return [{"error": "Query cannot be empty"}]

        try:
            # Enhance query with market-related terms
            enhanced_query = self._enhance_market_query(query)

            # Set up search parameters
            search_params = {
                "query": enhanced_query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_raw_content": True,
                "include_answer": True
            }

            # Add date filter
            if days_back > 0:
                start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                search_params["published_after"] = start_date

            # Add domain filters if specified
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains

            logger.info(f"Searching news for: {enhanced_query}")
            response = self.client.search(**search_params)

            # Process and format results
            news_articles = self._process_search_results(response, query)

            logger.info(f"Found {len(news_articles)} relevant news articles")
            return news_articles

        except Exception as e:
            logger.error(f"Error searching market news: {e}")
            return [{"error": f"Failed to search news: {str(e)}"}]

    def search_commodity_news(
            self,
            commodity: str,
            context: str = "",
            max_results: int = 8,
            days_back: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for news specifically related to a commodity.

        Args:
            commodity (str): The commodity name (e.g., "wheat", "rice", "onion")
            context (str): Additional context (e.g., "price", "export", "weather")
            max_results (int): Maximum number of results
            days_back (int): Days to look back for news

        Returns:
            List[Dict[str, Any]]: Commodity-specific news articles
        """
        # Build commodity-specific query
        query = f"{commodity} {context}".strip()

        # Use market-focused domains for commodity news
        market_domains = [
            "https://agriexchange.apeda.gov.in/Market/MarketNews?page="
            "reuters.com",
            "bloomberg.com",
            "cnbc.com",
            "marketwatch.com",
            "economictimes.indiatimes.com",
            "business-standard.com",
            "livemint.com"
        ]

        return self.search_market_news(
            query=query,
            max_results=max_results,
            days_back=days_back,
            include_domains=market_domains
        )

    def search_apmc_news(
            self,
            apmc: str,
            state: str = "",
            max_results: int = 5,
            days_back: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for news related to specific APMC markets.

        Args:
            apmc (str): APMC name
            state (str): State name for more specific results
            max_results (int): Maximum results to return
            days_back (int): Days to look back

        Returns:
            List[Dict[str, Any]]: APMC-related news articles
        """
        query = f"{apmc} APMC {state} market".strip()

        return self.search_market_news(
            query=query,
            max_results=max_results,
            days_back=days_back
        )

    def search_price_trend_news(
            self,
            commodity: str,
            trend: str = "price",
            max_results: int = 6,
            days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Search for news about price trends and market movements.

        Args:
            commodity (str): Commodity name
            trend (str): Type of trend ("price", "increase", "decrease", "volatile")
            max_results (int): Maximum results
            days_back (int): Days to look back

        Returns:
            List[Dict[str, Any]]: Price trend related news
        """
        query = f"{commodity} {trend} trend market analysis"

        return self.search_market_news(
            query=query,
            max_results=max_results,
            days_back=days_back
        )

    def get_market_sentiment_news(
            self,
            query: str,
            max_results: int = 15,
            days_back: int = 3
    ) -> Dict[str, Any]:
        """
        Get news articles and analyze overall market sentiment.

        Args:
            query (str): Market query
            max_results (int): Maximum results to analyze
            days_back (int): Recent days to consider

        Returns:
            Dict[str, Any]: News articles with sentiment analysis
        """
        articles = self.search_market_news(
            query=query,
            max_results=max_results,
            days_back=days_back
        )

        if not articles or "error" in articles[0]:
            return {"error": "No articles found for sentiment analysis"}

        # Basic sentiment analysis based on keywords
        sentiment_score = self._analyze_sentiment(articles)

        return {
            "query": query,
            "total_articles": len(articles),
            "sentiment_score": sentiment_score,
            "sentiment_label": self._get_sentiment_label(sentiment_score),
            "articles": articles[:5],  # Return top 5 for sentiment
            "summary": self._generate_news_summary(articles)
        }

    def _enhance_market_query(self, query: str) -> str:
        """
        Enhance user query with market-related keywords for better results.
        """
        market_keywords = [
            "market", "price", "trading", "commodity", "agriculture",
            "APMC", "mandi", "wholesale", "retail", "supply", "demand"
        ]

        query_lower = query.lower()
        has_market_context = any(keyword in query_lower for keyword in market_keywords)

        if not has_market_context:
            return f"{query} market news"

        return query

    def _process_search_results(self, response: Dict, original_query: str) -> List[Dict[str, Any]]:
        """
        Process and format Tavily search results.
        """
        if not response or "results" not in response:
            return []

        processed_articles = []

        for result in response.get("results", []):
            article = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "published_date": result.get("published_date", ""),
                "score": result.get("score", 0),
                "source": self._extract_domain(result.get("url", "")),
                "relevance": self._calculate_relevance(result, original_query)
            }

            # Only include articles with substantial content
            if len(article["content"]) > 50:
                processed_articles.append(article)

        # Sort by relevance and score
        processed_articles.sort(key=lambda x: (x["relevance"], x["score"]), reverse=True)

        return processed_articles

    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except:
            return "unknown"

    def _calculate_relevance(self, result: Dict, query: str) -> float:
        """Calculate relevance score based on query terms in title and content."""
        query_terms = query.lower().split()
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()

        title_matches = sum(1 for term in query_terms if term in title)
        content_matches = sum(1 for term in query_terms if term in content)

        # Weight title matches more heavily
        relevance = (title_matches * 2 + content_matches) / len(query_terms)
        return min(relevance, 1.0)  # Cap at 1.0

    def _analyze_sentiment(self, articles: List[Dict[str, Any]]) -> float:
        """
        Basic sentiment analysis of news articles.
        Returns score between -1 (very negative) and 1 (very positive).
        """
        positive_words = [
            "increase", "rise", "growth", "profit", "gain", "boost", "improve",
            "strong", "positive", "bullish", "rally", "surge", "uptick"
        ]

        negative_words = [
            "decrease", "fall", "decline", "loss", "drop", "weak", "negative",
            "bearish", "crash", "plunge", "downturn", "crisis", "shortage"
        ]

        total_score = 0
        total_articles = 0

        for article in articles:
            text = f"{article.get('title', '')} {article.get('content', '')}".lower()

            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            if pos_count + neg_count > 0:
                article_score = (pos_count - neg_count) / (pos_count + neg_count)
                total_score += article_score
                total_articles += 1

        return total_score / total_articles if total_articles > 0 else 0

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to human-readable label."""
        if score > 0.3:
            return "Positive"
        elif score < -0.3:
            return "Negative"
        else:
            return "Neutral"

    def _generate_news_summary(self, articles: List[Dict[str, Any]]) -> str:
        """Generate a brief summary of the news articles."""
        if not articles:
            return "No articles to summarize."

        # Extract key themes from titles
        all_titles = " ".join([article.get("title", "") for article in articles[:5]])

        return f"Recent news coverage includes {len(articles)} articles focusing on market developments and trends."


# Factory function for easy instantiation
def create_news_search_tool(api_key: str = None) -> NewsSearchTool:
    """
    Create and return a NewsSearchTool instance.

    Args:
        api_key (str, optional): Tavily API key

    Returns:
        NewsSearchTool: Configured news search tool
    """
    return NewsSearchTool(api_key=api_key)


# Example usage and testing
if __name__ == "__main__":
    # Example usage - uncomment and add your API key to test
    news_tool = NewsSearchTool(api_key=TAVILY_API_KEY)

    # Test general market news search
    results = news_tool.search_market_news("wheat price increase India", max_results=5)
    print("Market News Results:")
    for article in results:
        print(f"- {article['title']} ({article['source']})")

    # Test commodity-specific search
    commodity_news = news_tool.search_commodity_news("rice", "export policy", max_results=5)
    print("\nCommodity News Results:")
    for article in commodity_news:
        print(f"- {article['title']}")

    # Test sentiment analysis
    sentiment = news_tool.get_market_sentiment_news("onion price volatility")
    print(f"\nSentiment Analysis: {sentiment['sentiment_label']} ({sentiment['sentiment_score']:.2f})")
