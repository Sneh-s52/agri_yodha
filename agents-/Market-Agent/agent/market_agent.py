# market_agent/agent/market_agent.py

import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import pandas as pd
import openai
import requests
import math
from typing import List, Dict, Tuple, Optional
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_layer.mongo_client import mongo_ts_client
from data_layer.pinecone_client import pinecone_vector_client
from tools.news_search_tool import NewsSearchTool
from forecasting.price_forecasting import PriceForecastingTool
from config import OPENAI_API_KEY, TAVILY_API_KEY

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)





class MarketAgent:
    """
    AI-powered market agent for Agri Yodha using GPT-4o for intelligent reasoning,
    analysis, and decision-making for policy makers and financiers.
    """

    def __init__(self):
        """Initialize the AI Market Agent with all tools and data sources."""
        try:
            # Initialize data sources
            self.mongo_client = mongo_ts_client
            self.pinecone_client = pinecone_vector_client
            self.news_tool = NewsSearchTool()
            self.forecasting_tool = PriceForecastingTool()
            

            
            # Initialize OpenAI
            openai.api_key = OPENAI_API_KEY
            self.llm_model = "gpt-4o"
            
            # Initialize conversation history
            self.conversation_history = []
            self.last_analysis = {}
            
            logger.info(f"AI Market Agent initialized successfully with {self.llm_model}")
            
        except Exception as e:
            logger.error(f"Error initializing Market Agent: {e}")
            raise

    def _get_llm_insights(self, prompt: str, context: str = "", max_tokens: int = 1000) -> str:
        """
        Get insights from GPT-4o for intelligent analysis and reasoning.
        
        Args:
            prompt: The main prompt for the LLM
            context: Additional context or data
            max_tokens: Maximum tokens for response
            
        Returns:
            LLM-generated insights
        """
        try:
            # Get current market overview for context
            market_context = ""
            try:
                if self.pinecone_client and self.pinecone_client.index is not None:
                    commodities = self.pinecone_client.get_all_commodities()
                    apmcs = self.pinecone_client.get_all_apmcs()
                    market_context = f"""
                    Current Market Overview:
                    - Total Commodities Available: {len(commodities) if commodities else 'Unknown'}
                    - Total APMCs Available: {len(apmcs) if apmcs else 'Unknown'}
                    - Sample Commodities: {', '.join(commodities[:10]) if commodities else 'None'}
                    - Sample APMCs: {', '.join(apmcs[:10]) if apmcs else 'None'}
                    """
            except Exception as e:
                logger.warning(f"Could not get market context: {e}")
                market_context = "Market context unavailable"
            
            full_prompt = f"""
            You are an expert agricultural market analyst and policy advisor for Agri Yodha, 
            a next-generation AI system for policy makers and financiers in India.
            
            Your expertise includes:
            - Agricultural commodity markets and price analysis
            - APMC (Agricultural Produce Market Committee) operations
            - Weather impact on agricultural markets
            - Supply chain analysis and risk assessment
            - Policy recommendations for government officials
            - Investment strategies for financiers and traders
            - Regional market dynamics across Indian states
            
            {market_context}
            
            Context: {context}
            
            Task: {prompt}
            
            Please provide:
            1. Clear analysis and insights based on your expertise
            2. Actionable recommendations for different stakeholders
            3. Policy implications and government actions needed
            4. Risk assessment and mitigation strategies
            5. Market impact analysis and forecasting insights
            
            Be specific, data-driven, practical, and conversational in your response.
            If you need specific market data to answer a question, suggest using the 
            appropriate commands like 'analyze', 'forecast', or 'news'.
            
            Focus on Indian agricultural markets and APMC systems.
            """
            
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert agricultural market analyst and policy advisor for Agri Yodha, specializing in Indian agricultural markets, APMC systems, and policy recommendations."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error getting LLM insights: {e}")
            return f"LLM analysis failed: {str(e)}"

    def _safe_json_serialize(self, data: Any) -> str:
        """
        Safely serialize data to JSON, handling non-serializable types.
        """
        import json
        from datetime import datetime, date
        import numpy as np
        import pandas as pd
        
        def convert_to_serializable(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, date):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
                
        try:
            serializable_data = convert_to_serializable(data)
            return json.dumps(serializable_data, default=str, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON serialization failed: {e}")
            return str(data)

    def _analyze_market_data_with_llm(self, commodity: str, apmc: str, 
                                     market_data: Dict, news_data: Dict, 
                                     forecast_data: Dict) -> Dict[str, Any]:
        """
        Use GPT-4o to analyze market data and provide intelligent insights.
        """
        try:
            # Prepare context for LLM analysis with safe serialization
            context = f"""
            Commodity: {commodity}
            APMC: {apmc}
            
            Market Data: {self._safe_json_serialize(market_data)}
            News Data: {self._safe_json_serialize(news_data)}
            Forecast Data: {self._safe_json_serialize(forecast_data)}
            """
            
            prompt = f"""
            Analyze the market data for {commodity} at {apmc} and provide:
            
            1. **Market Intelligence Summary**: Key insights about current market conditions
            2. **Price Analysis**: Understanding of price trends and factors
            3. **Risk Assessment**: Identify potential risks and their severity
            4. **Policy Recommendations**: Specific policy suggestions for government officials
            5. **Investment Insights**: Strategic advice for financiers and traders
            6. **Supply Chain Analysis**: Insights about supply-demand dynamics
            7. **Future Outlook**: Predictions based on current data and trends
            
            Provide actionable, specific recommendations that policy makers can implement.
            """
            
            llm_analysis = self._get_llm_insights(prompt, context, max_tokens=1500)
            
            # Parse LLM response into structured format
            analysis_sections = self._parse_llm_response(llm_analysis)
            
            return {
                "llm_analysis": llm_analysis,
                "structured_insights": analysis_sections,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in LLM market analysis: {e}")
            return {"error": f"LLM analysis failed: {str(e)}"}

    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured insights.
        """
        try:
            # Simple parsing - in production, you might want more sophisticated parsing
            sections = {
                "market_intelligence": "",
                "price_analysis": "",
                "risk_assessment": "",
                "policy_recommendations": "",
                "investment_insights": "",
                "supply_chain_analysis": "",
                "future_outlook": ""
            }
            
            # Extract sections based on common patterns in LLM responses
            lines = llm_response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Identify section headers
                if any(keyword in line.lower() for keyword in ['market intelligence', 'market summary']):
                    current_section = 'market_intelligence'
                elif any(keyword in line.lower() for keyword in ['price analysis', 'price trends']):
                    current_section = 'price_analysis'
                elif any(keyword in line.lower() for keyword in ['risk assessment', 'risks']):
                    current_section = 'risk_assessment'
                elif any(keyword in line.lower() for keyword in ['policy recommendations', 'policy']):
                    current_section = 'policy_recommendations'
                elif any(keyword in line.lower() for keyword in ['investment insights', 'investment']):
                    current_section = 'investment_insights'
                elif any(keyword in line.lower() for keyword in ['supply chain', 'supply-demand']):
                    current_section = 'supply_chain_analysis'
                elif any(keyword in line.lower() for keyword in ['future outlook', 'predictions']):
                    current_section = 'future_outlook'
                elif current_section and line.startswith(('•', '-', '1.', '2.', '3.')):
                    # This is content for the current section
                    if sections[current_section]:
                        sections[current_section] += " " + line
                    else:
                        sections[current_section] = line
                        
            return sections
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {"error": f"Parsing failed: {str(e)}"}

    def _generate_intelligent_recommendations(self, analysis_data: Dict, 
                                            commodity: str, apmc: str) -> List[str]:
        """
        Use GPT-4o to generate intelligent, context-aware recommendations.
        """
        try:
            context = f"""
            Commodity: {commodity}
            APMC: {apmc}
            Analysis Data: {self._safe_json_serialize(analysis_data)}
            """
            
            prompt = f"""
            Based on the market analysis for {commodity} at {apmc}, generate 5-7 specific, actionable recommendations.
            
            Focus on:
            1. **Immediate Actions**: What should be done in the next 7-30 days?
            2. **Strategic Planning**: Medium-term strategies for the next 3-6 months?
            3. **Policy Interventions**: Specific government actions needed?
            4. **Risk Mitigation**: How to address identified risks?
            5. **Opportunity Capture**: How to leverage positive market conditions?
            
            Make each recommendation specific, actionable, and prioritized by impact and urgency.
            """
            
            llm_recommendations = self._get_llm_insights(prompt, context, max_tokens=800)
            
            # Parse recommendations into list
            recommendations = []
            for line in llm_recommendations.split('\n'):
                line = line.strip()
                if line and (line.startswith(('•', '-', '1.', '2.', '3.', '4.', '5.')) or 
                            any(keyword in line.lower() for keyword in ['recommend', 'should', 'need', 'implement'])):
                    recommendations.append(line)
                    
            return recommendations[:7]  # Limit to 7 recommendations
            
        except Exception as e:
            logger.error(f"Error generating intelligent recommendations: {e}")
            return ["Unable to generate specific recommendations due to technical limitations."]

    def _assess_risks_with_llm(self, market_data: Dict, commodity: str, apmc: str) -> Dict[str, Any]:
        """
        Use GPT-4o to assess market risks intelligently.
        """
        try:
            context = f"""
            Commodity: {commodity}
            APMC: {apmc}
            Market Data: {self._safe_json_serialize(market_data)}
            """
            
            prompt = f"""
            Assess the market risks for {commodity} at {apmc} based on the provided data.
            
            Provide:
            1. **Overall Risk Level**: Low/Medium/High with confidence score (0-100)
            2. **Key Risk Factors**: List specific risks identified
            3. **Risk Severity**: Rate each risk (Low/Medium/High)
            4. **Mitigation Strategies**: Specific actions to reduce each risk
            5. **Early Warning Indicators**: What to monitor for risk escalation
            
            Format your response in a structured way that can be easily parsed.
            """
            
            llm_risk_assessment = self._get_llm_insights(prompt, context, max_tokens=1000)
            
            # Parse risk assessment
            risk_analysis = self._parse_risk_assessment(llm_risk_assessment)
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error in LLM risk assessment: {e}")
            return {"error": f"Risk assessment failed: {str(e)}"}

    def _parse_risk_assessment(self, risk_text: str) -> Dict[str, Any]:
        """
        Parse LLM risk assessment into structured format.
        """
        try:
            risk_analysis = {
                "overall_risk_level": "medium",
                "risk_score": 50,
                "risk_factors": [],
                "mitigation_strategies": [],
                "early_warning_indicators": []
            }
            
            lines = risk_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Identify sections
                if 'risk level' in line.lower():
                    if 'high' in line.lower():
                        risk_analysis["overall_risk_level"] = "high"
                    elif 'low' in line.lower():
                        risk_analysis["overall_risk_level"] = "low"
                        
                elif 'risk factors' in line.lower() or 'risks' in line.lower():
                    current_section = 'risk_factors'
                elif 'mitigation' in line.lower():
                    current_section = 'mitigation_strategies'
                elif 'warning' in line.lower() or 'monitor' in line.lower():
                    current_section = 'early_warning_indicators'
                elif current_section and line.startswith(('•', '-', '1.', '2.', '3.')):
                    if current_section == 'risk_factors':
                        risk_analysis["risk_factors"].append(line)
                    elif current_section == 'mitigation_strategies':
                        risk_analysis["mitigation_strategies"].append(line)
                    elif current_section == 'early_warning_indicators':
                        risk_analysis["early_warning_indicators"].append(line)
                        
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error parsing risk assessment: {e}")
            return {"error": f"Risk parsing failed: {str(e)}"}

    def analyze_market_conditions(self, commodity: str, apmc: str, 
                                include_news: bool = True, 
                                include_forecast: bool = True) -> Dict[str, Any]:
        """
        AI-powered comprehensive market analysis using GPT-4o for intelligent insights.
        """
        try:
            logger.info(f"Starting AI-powered market analysis for {commodity} at {apmc}")
            
            analysis = {
                "commodity": commodity,
                "apmc": apmc,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_sources": [],
                "current_market_data": {},
                "news_analysis": {},
                "price_forecast": {},
                "ai_insights": {},
                "intelligent_recommendations": [],
                "risk_assessment": {},
                "policy_implications": []
            }
            
            # 1. Get current market data from MongoDB
            if self.mongo_client and self.mongo_client.collection is not None:
                try:
                    current_price = self.mongo_client.get_latest_price(commodity, apmc)
                    if "error" not in current_price:
                        analysis["current_market_data"]["mongodb"] = current_price
                        analysis["data_sources"].append("MongoDB")
                        logger.info(f"Retrieved MongoDB data for {commodity} at {apmc}")
                    else:
                        logger.warning(f"MongoDB data retrieval failed: {current_price['error']}")
                        
                    # Get price history for forecasting
                    price_history = self.mongo_client.get_price_history(commodity, apmc, days=90)
                    if not price_history.empty:
                        analysis["current_market_data"]["price_history"] = {
                            "data_points": len(price_history),
                            "date_range": {
                                "start": price_history.index[0].isoformat() if not price_history.empty else None,
                                "end": price_history.index[-1].isoformat() if not price_history.empty else None
                            }
                        }
                except Exception as e:
                    logger.error(f"Error retrieving MongoDB data: {e}")
                    
            # 2. Get vector search results from Pinecone
            if self.pinecone_client and self.pinecone_client.index is not None:
                try:
                    vector_price = self.pinecone_client.get_latest_price(commodity, apmc)
                    if "error" not in vector_price:
                        analysis["current_market_data"]["pinecone"] = vector_price
                        analysis["data_sources"].append("Pinecone")
                        logger.info(f"Retrieved Pinecone data for {commodity} at {apmc}")
                        
                    # Get market summary
                    market_summary = self.pinecone_client.get_market_summary(commodity, days=30)
                    if "error" not in market_summary:
                        analysis["current_market_data"]["market_summary"] = market_summary
                        
                except Exception as e:
                    logger.error(f"Error retrieving Pinecone data: {e}")
                    
            # 3. News analysis
            if include_news:
                try:
                    news_results = self.news_tool.search_commodity_news(
                        commodity=commodity,
                        context=f"price market {apmc}",
                        max_results=10,
                        days_back=7
                    )
                    
                    if news_results and "error" not in news_results[0]:
                        analysis["news_analysis"]["articles"] = news_results[:5]  # Top 5 articles
                        
                        # Get sentiment analysis
                        sentiment = self.news_tool.get_market_sentiment_news(
                            f"{commodity} {apmc} market price",
                            max_results=15,
                            days_back=3
                        )
                        if "error" not in sentiment:
                            analysis["news_analysis"]["sentiment"] = sentiment
                            
                        analysis["data_sources"].append("News API")
                        logger.info(f"Retrieved news data for {commodity} at {apmc}")
                        
                except Exception as e:
                    logger.error(f"Error retrieving news data: {e}")
                    
            # 4. Price forecasting
            if include_forecast:
                try:
                    # Use MongoDB data if available, otherwise use Pinecone
                    forecast_data = None
                    if (self.mongo_client and self.mongo_client.collection is not None and 
                        "price_history" in analysis["current_market_data"].get("mongodb", {})):
                        # Get full price history for forecasting
                        price_history = self.mongo_client.get_price_history(commodity, apmc, days=90)
                        if not price_history.empty:
                            forecast_data = price_history
                    elif self.pinecone_client and self.pinecone_client.index is not None:
                        # Fallback to Pinecone data
                        price_history = self.pinecone_client.get_price_history(commodity, apmc, days=90)
                        if not price_history.empty:
                            forecast_data = price_history
                            
                    if forecast_data is not None and not forecast_data.empty:
                        forecast_report = self.forecasting_tool.generate_forecast_report(
                            commodity, apmc, forecast_data, forecast_days=7
                        )
                        
                        if "error" not in forecast_report:
                            analysis["price_forecast"] = forecast_report
                            analysis["data_sources"].append("Forecasting Engine")
                            logger.info(f"Generated price forecast for {commodity} at {apmc}")
                            
                except Exception as e:
                    logger.error(f"Error generating price forecast: {e}")
                    
            # 5. AI-Powered Analysis using GPT-4o
            try:
                logger.info("Generating AI-powered insights using GPT-4o...")
                
                # Generate intelligent market analysis
                ai_analysis = self._analyze_market_data_with_llm(
                    commodity, apmc, 
                    analysis["current_market_data"],
                    analysis["news_analysis"],
                    analysis["price_forecast"]
                )
                
                if "error" not in ai_analysis:
                    analysis["ai_insights"] = ai_analysis
                    analysis["data_sources"].append("GPT-4o AI Analysis")
                    logger.info("AI analysis completed successfully")
                    
                # Generate intelligent recommendations
                intelligent_recommendations = self._generate_intelligent_recommendations(
                    analysis, commodity, apmc
                )
                analysis["intelligent_recommendations"] = intelligent_recommendations
                
                # AI-powered risk assessment
                risk_assessment = self._assess_risks_with_llm(analysis["current_market_data"], commodity, apmc)
                if "error" not in risk_assessment:
                    analysis["risk_assessment"] = risk_assessment
                    
            except Exception as e:
                logger.error(f"Error in AI-powered analysis: {e}")
                
            # Store analysis for future reference
            self.last_analysis[f"{commodity}_{apmc}"] = analysis
            
            logger.info(f"AI-powered market analysis completed for {commodity} at {apmc}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in AI-powered market analysis: {e}")
            return {"error": f"AI market analysis failed: {str(e)}"}

    def get_price_forecast(self, commodity: str, apmc: str, 
                          forecast_days: int = 7) -> Dict[str, Any]:
        """
        Get AI-enhanced price forecast for a specific commodity and APMC.
        """
        try:
            logger.info(f"Generating AI-enhanced price forecast for {commodity} at {apmc}")
            
            # Get price history
            forecast_data = None
            
            if self.mongo_client and self.mongo_client.collection is not None:
                forecast_data = self.mongo_client.get_price_history(commodity, apmc, days=90)
                
            if forecast_data is None or forecast_data.empty:
                if self.pinecone_client and self.pinecone_client.index is not None:
                    forecast_data = self.pinecone_client.get_price_history(commodity, apmc, days=90)
                    
            if forecast_data is None or forecast_data.empty:
                return {"error": "No price data available for forecasting"}
                
            # Generate forecast
            forecast_report = self.forecasting_tool.generate_forecast_report(
                commodity, apmc, forecast_data, forecast_days
            )
            
            # Enhance forecast with AI insights
            if "error" not in forecast_report:
                try:
                    context = f"""
                    Commodity: {commodity}
                    APMC: {apmc}
                    Forecast Data: {self._safe_json_serialize(forecast_report)}
                    """
                    
                    prompt = f"""
                    Analyze the price forecast for {commodity} at {apmc} and provide:
                    
                    1. **Forecast Interpretation**: What do these numbers mean in practical terms?
                    2. **Confidence Assessment**: How reliable are these predictions?
                    3. **Market Implications**: What do these forecasts suggest about market conditions?
                    4. **Strategic Advice**: What actions should stakeholders consider?
                    5. **Risk Factors**: What could make these forecasts inaccurate?
                    
                    Be specific and actionable in your analysis.
                    """
                    
                    ai_forecast_analysis = self._get_llm_insights(prompt, context, max_tokens=800)
                    forecast_report["ai_enhancement"] = ai_forecast_analysis
                    
                except Exception as e:
                    logger.error(f"Error enhancing forecast with AI: {e}")
                    
            return forecast_report
            
        except Exception as e:
            logger.error(f"Error in AI-enhanced price forecasting: {e}")
            return {"error": f"AI price forecasting failed: {str(e)}"}

    def search_market_news(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for market-related news with AI-powered analysis.
        """
        try:
            logger.info(f"Searching news for query: {query}")
            
            news_results = self.news_tool.search_market_news(
                query=query,
                max_results=max_results,
                days_back=7
            )
            
            if news_results and "error" not in news_results[0]:
                # Get sentiment analysis
                sentiment = self.news_tool.get_market_sentiment_news(
                    query, max_results=max_results, days_back=3
                )
                
                # AI-powered news analysis
                try:
                    context = f"""
                    Query: {query}
                    News Articles: {self._safe_json_serialize(news_results[:5])}
                    Sentiment: {self._safe_json_serialize(sentiment)}
                    """
                    
                    prompt = f"""
                    Analyze the news results for '{query}' and provide:
                    
                    1. **Key Themes**: What are the main topics and trends?
                    2. **Market Impact**: How might this news affect market conditions?
                    3. **Stakeholder Implications**: What does this mean for different market participants?
                    4. **Action Items**: What should stakeholders do based on this news?
                    5. **Risk Assessment**: Any potential risks or opportunities identified?
                    
                    Provide actionable insights based on the news analysis.
                    """
                    
                    ai_news_analysis = self._get_llm_insights(prompt, context, max_tokens=1000)
                    
                    return {
                        "query": query,
                        "articles": news_results,
                        "sentiment": sentiment,
                        "ai_analysis": ai_news_analysis,
                        "search_timestamp": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"Error in AI news analysis: {e}")
                    return {
                        "query": query,
                        "articles": news_results,
                        "sentiment": sentiment,
                        "search_timestamp": datetime.now().isoformat()
                    }
            else:
                return {"error": "News search failed", "query": query}
                
        except Exception as e:
            logger.error(f"Error in AI-enhanced news search: {e}")
            return {"error": f"AI news search failed: {str(e)}"}

    def get_market_intelligence(self, commodity: str = None, 
                              apmc: str = None, 
                              include_forecast: bool = True) -> Dict[str, Any]:
        """
        Get AI-powered comprehensive market intelligence.
        """
        try:
            logger.info("Generating AI-powered market intelligence report")
            
            intelligence = {
                "report_timestamp": datetime.now().isoformat(),
                "market_overview": {},
                "commodity_analysis": {},
                "ai_insights": {},
                "strategic_recommendations": [],
                "policy_framework": [],
                "risk_landscape": {}
            }
            
            # Get available commodities and APMCs
            if self.pinecone_client and self.pinecone_client.index is not None:
                try:
                    commodities = self.pinecone_client.get_all_commodities()
                    apmcs = self.pinecone_client.get_all_apmcs()
                    
                    intelligence["market_overview"]["total_commodities"] = len(commodities)
                    intelligence["market_overview"]["total_apmcs"] = len(apmcs)
                    intelligence["market_overview"]["sample_commodities"] = commodities[:10]
                    intelligence["market_overview"]["sample_apmcs"] = apmcs[:10]
                    
                except Exception as e:
                    logger.error(f"Error getting market overview: {e}")
                    
            # Analyze specific commodity if provided
            if commodity:
                if apmc:
                    # Single commodity-APMC analysis
                    analysis = self.analyze_market_conditions(commodity, apmc, include_forecast=include_forecast)
                    intelligence["commodity_analysis"][f"{commodity}_{apmc}"] = analysis
                else:
                    # Commodity analysis across all APMCs
                    if self.pinecone_client and self.pinecone_client.index is not None:
                        apmcs_for_commodity = self.pinecone_client.get_all_apmcs()
                        for apmc_name in apmcs_for_commodity[:5]:  # Limit to 5 APMCs
                            try:
                                analysis = self.analyze_market_conditions(commodity, apmc_name, include_forecast=False)
                                intelligence["commodity_analysis"][f"{commodity}_{apmc_name}"] = analysis
                            except Exception as e:
                                logger.warning(f"Failed to analyze {commodity} at {apmc_name}: {e}")
                                
            # Generate AI-powered strategic insights
            try:
                context = f"""
                Market Overview: {self._safe_json_serialize(intelligence["market_overview"])}
                Commodity Analysis: {self._safe_json_serialize(intelligence["commodity_analysis"])}
                """
                
                prompt = f"""
                Based on the market intelligence data, provide:
                
                1. **Strategic Market Insights**: Key patterns and trends across the market
                2. **Policy Framework Recommendations**: Structured policy approach for government
                3. **Risk Landscape Assessment**: Overall market risk profile
                4. **Strategic Opportunities**: Where should stakeholders focus attention?
                5. **Implementation Roadmap**: How to implement the recommendations?
                
                Focus on actionable, strategic insights for policy makers and financiers.
                """
                
                ai_strategic_insights = self._get_llm_insights(prompt, context, max_tokens=1200)
                intelligence["ai_insights"] = ai_strategic_insights
                
                # Parse strategic recommendations
                strategic_recommendations = self._parse_strategic_recommendations(ai_strategic_insights)
                intelligence["strategic_recommendations"] = strategic_recommendations
                
            except Exception as e:
                logger.error(f"Error generating AI strategic insights: {e}")
                
            return intelligence
            
        except Exception as e:
            logger.error(f"Error in AI-powered market intelligence: {e}")
            return {"error": f"AI market intelligence failed: {str(e)}"}

    def _parse_strategic_recommendations(self, ai_text: str) -> List[str]:
        """Parse AI strategic insights into actionable recommendations."""
        try:
            recommendations = []
            lines = ai_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('•', '-', '1.', '2.', '3.', '4.', '5.')) or 
                            any(keyword in line.lower() for keyword in ['recommend', 'should', 'need', 'implement', 'focus', 'develop'])):
                    recommendations.append(line)
                    
            return recommendations[:10]  # Limit to 10 recommendations
            
        except Exception as e:
            logger.error(f"Error parsing strategic recommendations: {e}")
            return ["Unable to parse strategic recommendations due to technical limitations."]

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for analysis."""
        return self.conversation_history

    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def export_analysis_report(self, commodity: str, apmc: str, 
                              format: str = "json") -> str:
        """
        Export AI-enhanced analysis report in specified format.
        """
        try:
            key = f"{commodity}_{apmc}"
            if key not in self.last_analysis:
                return "No analysis available for export"
                
            analysis = self.last_analysis[key]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_market_analysis_{commodity}_{apmc}_{timestamp}"
            
            if format.lower() == "json":
                filepath = f"logs/{filename}.json"
                with open(filepath, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                    
            elif format.lower() == "csv":
                # Convert relevant parts to CSV
                filepath = f"logs/{filename}.csv"
                # Implementation for CSV export
                pass
                
            else:
                filepath = f"logs/{filename}.txt"
                with open(filepath, 'w') as f:
                    f.write(f"AI-Enhanced Market Analysis Report\n")
                    f.write(f"Commodity: {commodity}\n")
                    f.write(f"APMC: {apmc}\n")
                    f.write(f"Generated: {timestamp}\n")
                    f.write(f"AI Model: {self.llm_model}\n\n")
                    f.write(json.dumps(analysis, indent=2, default=str))
                    
            logger.info(f"AI-enhanced analysis report exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting AI-enhanced analysis report: {e}")
            return f"Export failed: {str(e)}"

    def handle_conversational_query(self, query: str) -> Dict[str, Any]:
        """
        Handle conversational queries by intelligently calling relevant tools and data sources.
        This method analyzes the query and calls appropriate tools to gather real data.
        """
        try:
            logger.info(f"Processing conversational query: {query}")
            
            # Initialize response structure
            response_data = {
                "query": query,
                "data_sources_used": [],
                "market_data": {},
                "news_data": {},
                "forecast_data": {},
                "ai_analysis": "",
                "recommendations": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 1: Analyze query intent and extract entities
            query_analysis = self._analyze_query_intent(query)
            logger.info(f"Query analysis: {query_analysis}")
            
            # Step 2: Extract commodity and APMC from query if present
            commodity, apmc = self._extract_entities_from_query(query)
            
            # Step 3: Gather market data based on query type
            if query_analysis.get('needs_market_data', False):
                response_data = self._gather_market_data(commodity, apmc, response_data)
                
            # Step 4: Gather news data for market outlook queries
            if query_analysis.get('needs_news_data', False):
                response_data = self._gather_news_data(query, response_data)
                
            # Step 5: Generate forecasts if requested
            if query_analysis.get('needs_forecast', False):
                response_data = self._gather_forecast_data(commodity, apmc, response_data)
                
            # Step 6: Generate comprehensive AI analysis
            response_data = self._generate_comprehensive_analysis(query, response_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in conversational query handling: {e}")
            return {
                "error": f"Query processing failed: {str(e)}",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze the intent of a conversational query and determine what tools to call.
        """
        try:
            # Use LLM to analyze query intent
            intent_prompt = f"""
            Analyze this agricultural market query and determine what data sources and tools are needed:
            
            Query: "{query}"
            
            Determine if the query needs:
            1. Market data (current prices, historical data)
            2. News data (market trends, developments)
            3. Forecasting data (price predictions)
            4. General market intelligence
            
            Respond with a JSON object containing:
            {{
                "needs_market_data": boolean,
                "needs_news_data": boolean, 
                "needs_forecast": boolean,
                "query_type": "price_inquiry|news_inquiry|forecast_inquiry|general_analysis",
                "entities": {{
                    "commodity": "extracted commodity name or null",
                    "apmc": "extracted APMC/market name or null",
                    "location": "extracted location or null",
                    "time_period": "extracted time period or null"
                }},
                "search_terms": ["relevant search terms for news"]
            }}
            
            Be intelligent about entity extraction. For example:
            - "wheat prices in Delhi" -> commodity: "wheat", apmc: "delhi"
            - "rice market trends" -> commodity: "rice", needs_news_data: true
            - "forecast soybean prices" -> commodity: "soybean", needs_forecast: true
            """
            
            intent_response = self._get_llm_insights(intent_prompt, max_tokens=500)
            
            # Try to parse JSON response
            import json
            try:
                # Extract JSON from response
                start_idx = intent_response.find('{')
                end_idx = intent_response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = intent_response[start_idx:end_idx]
                    intent_analysis = json.loads(json_str)
                else:
                    # Fallback to pattern matching
                    intent_analysis = self._fallback_intent_analysis(query)
            except json.JSONDecodeError:
                # Fallback to pattern matching
                intent_analysis = self._fallback_intent_analysis(query)
                
            logger.info(f"LLM Query analysis: {intent_analysis}")
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Error in LLM intent analysis: {e}")
            return self._fallback_intent_analysis(query)

    def _fallback_intent_analysis(self, query: str) -> Dict[str, Any]:
        """
        Fallback intent analysis using pattern matching.
        """
        query_lower = query.lower()
        
        intent_analysis = {
            "needs_market_data": False,
            "needs_news_data": False,
            "needs_forecast": False,
            "query_type": "general",
            "entities": {
                "commodity": None,
                "apmc": None,
                "location": None,
                "time_period": None
            },
            "search_terms": []
        }
        
        # Market data patterns
        market_patterns = [
            "price", "modal price", "current price", "latest price", "market price",
            "what is the price", "how much", "cost", "rate", "value"
        ]
        
        # News patterns
        news_patterns = [
            "news", "latest news", "recent news", "market news", "trends", "outlook",
            "what's happening", "market situation", "current situation", "developments"
        ]
        
        # Forecast patterns
        forecast_patterns = [
            "forecast", "prediction", "future", "next", "upcoming", "trend",
            "what will happen", "expect", "anticipate", "projection"
        ]
        
        # Check patterns
        for pattern in market_patterns:
            if pattern in query_lower:
                intent_analysis["needs_market_data"] = True
                intent_analysis["query_type"] = "price_inquiry"
                break
                
        for pattern in news_patterns:
            if pattern in query_lower:
                intent_analysis["needs_news_data"] = True
                intent_analysis["query_type"] = "news_inquiry"
                break
                
        for pattern in forecast_patterns:
            if pattern in query_lower:
                intent_analysis["needs_forecast"] = True
                intent_analysis["query_type"] = "forecast_inquiry"
                break
                
        # If no specific pattern found, default to general market analysis
        if intent_analysis["query_type"] == "general":
            intent_analysis["needs_market_data"] = True
            intent_analysis["needs_news_data"] = True
            
        return intent_analysis

    def _extract_entities_from_query(self, query: str) -> Tuple[str, str]:
        """
        Use LLM to intelligently extract commodity and APMC names from natural language query.
        """
        try:
            # Use LLM for intelligent entity extraction
            entity_prompt = f"""
            Extract agricultural commodity and market/APMC names from this query:
            
            Query: "{query}"
            
            Respond with a JSON object:
            {{
                "commodity": "extracted commodity name or null",
                "apmc": "extracted APMC/market name or null"
            }}
            
            Be intelligent about extraction:
            - "wheat prices in Delhi" -> commodity: "wheat", apmc: "delhi"
            - "rice market trends in Mumbai" -> commodity: "rice", apmc: "mumbai"
            - "soybean forecast for Pune" -> commodity: "soybean", apmc: "pune"
            - "cotton prices" -> commodity: "cotton", apmc: null
            - "market trends" -> commodity: null, apmc: null
            
            Only extract if you're confident about the entity. Return null if uncertain.
            """
            
            entity_response = self._get_llm_insights(entity_prompt, max_tokens=300)
            
            # Try to parse JSON response
            import json
            try:
                # Extract JSON from response
                start_idx = entity_response.find('{')
                end_idx = entity_response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = entity_response[start_idx:end_idx]
                    entities = json.loads(json_str)
                    commodity = entities.get("commodity")
                    apmc = entities.get("apmc")
                else:
                    commodity, apmc = None, None
            except json.JSONDecodeError:
                commodity, apmc = None, None
                
            logger.info(f"LLM Extracted entities - Commodity: {commodity}, APMC: {apmc} from query: {query}")
            return commodity, apmc
            
        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {e}")
            return None, None

    def _gather_market_data(self, commodity: str, apmc: str, response_data: Dict) -> Dict:
        """
        Gather market data using intelligent vector search and semantic matching.
        """
        logger.info(f"Gathering market data for {commodity} at {apmc}")
        
        market_data = {}
        data_sources = []
        
        # Use semantic search for better matching
        if self.pinecone_client and self.pinecone_client.index is not None:
            try:
                # Build semantic query
                if commodity and apmc:
                    semantic_query = f"{commodity} price at {apmc} APMC"
                elif commodity:
                    semantic_query = f"{commodity} agricultural commodity price"
                elif apmc:
                    semantic_query = f"agricultural market prices at {apmc}"
                else:
                    semantic_query = "agricultural commodity market prices"
                
                # Use semantic search
                semantic_results = self.pinecone_client.semantic_price_lookup(semantic_query, top_k=10)
                
                if semantic_results and "error" not in semantic_results[0]:
                    market_data["semantic_search_results"] = semantic_results[:5]  # Top 5 results
                    data_sources.append("Pinecone Semantic Search")
                    
                    # Extract latest price from semantic results
                    for result in semantic_results:
                        if result.get("modal_price") and result.get("commodity") and result.get("apmc"):
                            market_data["pinecone_latest"] = result
                            break
                            
            except Exception as e:
                logger.warning(f"Pinecone semantic search failed: {e}")
                
        # Try MongoDB with fuzzy matching
        if self.mongo_client and self.mongo_client.collection is not None:
            try:
                if commodity and apmc:
                    # Try exact match first
                    latest_price = self.mongo_client.get_latest_price(commodity, apmc)
                    if latest_price and "error" not in latest_price:
                        market_data["mongodb_latest"] = latest_price
                        data_sources.append("MongoDB")
                        
                    # Get price history
                    price_history = self.mongo_client.get_price_history(commodity, apmc, days=30)
                    if not price_history.empty:
                        market_data["mongodb_history"] = {
                            "data_points": len(price_history),
                            "date_range": f"{price_history.index.min()} to {price_history.index.max()}",
                            "avg_price": price_history['modal_price'].mean() if 'modal_price' in price_history.columns else None
                        }
                        data_sources.append("MongoDB History")
                        
            except Exception as e:
                logger.warning(f"MongoDB data gathering failed: {e}")
                
        # Get market overview using vector search
        if not commodity and not apmc:
            try:
                if self.pinecone_client and self.pinecone_client.index is not None:
                    # Use semantic search for market overview
                    overview_query = "agricultural commodity market overview India"
                    overview_results = self.pinecone_client.semantic_price_lookup(overview_query, top_k=20)
                    
                    if overview_results and "error" not in overview_results[0]:
                        # Extract unique commodities and APMCs
                        commodities = set()
                        apmcs = set()
                        
                        for result in overview_results:
                            if result.get("commodity"):
                                commodities.add(result["commodity"])
                            if result.get("apmc"):
                                apmcs.add(result["apmc"])
                                
                        market_data["market_overview"] = {
                            "available_commodities": list(commodities)[:10],
                            "available_apmcs": list(apmcs)[:10],
                            "total_commodities": len(commodities),
                            "total_apmcs": len(apmcs),
                            "sample_data": overview_results[:5]
                        }
                        data_sources.append("Market Overview")
                        
            except Exception as e:
                logger.warning(f"Market overview gathering failed: {e}")
                
        response_data["market_data"] = market_data
        response_data["data_sources_used"].extend(data_sources)
        
        return response_data

    def _gather_news_data(self, query: str, response_data: Dict) -> Dict:
        """
        Gather relevant news data using intelligent search term extraction.
        """
        logger.info(f"Gathering news data for query: {query}")
        
        if not self.news_tool:
            logger.warning("News tool not available")
            return response_data
            
        try:
            # Use LLM to extract intelligent search terms
            search_terms = self._extract_intelligent_news_search_terms(query)
            
            # Search for news
            news_results = self.news_tool.search_market_news(search_terms)
            
            if news_results and "error" not in news_results[0]:
                response_data["news_data"] = {
                    "articles": news_results[:5],  # Top 5 articles
                    "search_terms": search_terms
                }
                response_data["data_sources_used"].append("News API")
                
        except Exception as e:
            logger.warning(f"News data gathering failed: {e}")
            
        return response_data

    def _extract_intelligent_news_search_terms(self, query: str) -> str:
        """
        Use LLM to extract intelligent search terms for news search.
        """
        try:
            # Use LLM for intelligent search term extraction
            search_prompt = f"""
            Extract relevant search terms for agricultural market news from this query:
            
            Query: "{query}"
            
            Generate 2-3 search terms that would find relevant agricultural market news.
            Focus on:
            - Commodity names (wheat, rice, cotton, etc.)
            - Market locations (India, Maharashtra, Delhi, etc.)
            - Market concepts (agricultural exports, APMC, price trends, etc.)
            
            Respond with just the search terms separated by spaces, no JSON or formatting.
            Example: "wheat agricultural markets India"
            """
            
            search_terms = self._get_llm_insights(search_prompt, max_tokens=100).strip()
            
            # Fallback if LLM fails
            if not search_terms or len(search_terms) < 5:
                search_terms = "agricultural markets India"
                
            logger.info(f"LLM extracted search terms: {search_terms}")
            return search_terms
            
        except Exception as e:
            logger.error(f"Error in LLM search term extraction: {e}")
            return "agricultural markets India"

    def _gather_forecast_data(self, commodity: str, apmc: str, response_data: Dict) -> Dict:
        """
        Generate price forecasts if commodity and APMC are available.
        """
        logger.info(f"Gathering forecast data for {commodity} at {apmc}")
        
        if not commodity or not apmc:
            logger.warning("Cannot generate forecast without commodity and APMC")
            return response_data
            
        if not self.forecasting_tool:
            logger.warning("Forecasting tool not available")
            return response_data
            
        try:
            forecast = self.get_price_forecast(commodity, apmc)
            
            if forecast and "error" not in forecast:
                response_data["forecast_data"] = {
                    "forecast_period": forecast.get("forecast_period", "7 days"),
                    "methodology": forecast.get("methodology", {}),
                    "forecasts": forecast.get("forecasts", {}),
                    "recommendations": forecast.get("recommendations", [])
                }
                response_data["data_sources_used"].append("Price Forecasting")
                
        except Exception as e:
            logger.warning(f"Forecast data gathering failed: {e}")
            
        return response_data

    def _generate_comprehensive_analysis(self, query: str, response_data: Dict) -> Dict:
        """
        Generate comprehensive AI analysis based on gathered data including geospatial insights.
        """
        try:
            # Prepare context with all gathered data
            context = f"""
            User Query: {query}
            
            Data Sources Used: {', '.join(response_data['data_sources_used'])}
            
            Market Data: {self._safe_json_serialize(response_data['market_data'])}
            News Data: {self._safe_json_serialize(response_data['news_data'])}
            Forecast Data: {self._safe_json_serialize(response_data['forecast_data'])}
            """
            
            prompt = f"""
            You are an expert agricultural market analyst for Agri Yodha.
            
            The user asked: "{query}"
            
            Based on the available data, provide a comprehensive analysis that includes:
            
            1. **Direct Answer**: Address the user's specific question using the available data
            2. **Data Summary**: Summarize the key data points found
            3. **Market Analysis**: Provide insights based on the data
            4. **Market Insights**: Analyze market dynamics and regional variations
            5. **Trends & Patterns**: Identify any trends or patterns in the data
            6. **Risk Assessment**: Identify potential risks or opportunities
            7. **Recommendations**: Provide actionable recommendations, especially regarding nearby markets if applicable
            8. **Policy Implications**: Suggest policy actions if relevant
            
            If market data is found, provide specific insights about:
            - Market dynamics and price trends
            - Regional variations and patterns
            - Supply and demand factors
            - Market accessibility and opportunities
            
            If specific data is not available, acknowledge this and provide general insights.
            Be specific, data-driven, and practical in your response.
            Focus on Indian agricultural markets and APMC systems.
            """
            
            ai_analysis = self._get_llm_insights(prompt, context, max_tokens=2000)
            
            response_data["ai_analysis"] = ai_analysis
            
            # Extract recommendations from AI analysis
            recommendations = self._extract_recommendations_from_analysis(ai_analysis)
            response_data["recommendations"] = recommendations
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {e}")
            response_data["ai_analysis"] = f"Analysis generation failed: {str(e)}"
            return response_data

    def _extract_recommendations_from_analysis(self, analysis: str) -> List[str]:
        """
        Extract actionable recommendations from AI analysis.
        """
        recommendations = []
        
        # Look for recommendation patterns in the analysis
        lines = analysis.split('\n')
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'consider', 'action']):
                if line.startswith(('•', '-', '1.', '2.', '3.', '4.', '5.')):
                    recommendations.append(line)
                elif len(line) > 20 and len(line) < 200:  # Reasonable length for a recommendation
                    recommendations.append(line)
                    
        return recommendations[:5]  # Limit to 5 recommendations


# Factory function for easy instantiation
def create_market_agent() -> MarketAgent:
    """Create and return an AI-powered MarketAgent instance."""
    return MarketAgent()


if __name__ == "__main__":
    # Test the AI market agent
    try:
        agent = create_market_agent()
        print("AI Market Agent created successfully with GPT-4o!")
        
        # Test basic functionality
        print("\nTesting AI-powered market intelligence...")
        intelligence = agent.get_market_intelligence()
        print(f"AI market intelligence generated: {len(intelligence)} sections")
        
    except Exception as e:
        print(f"Error testing AI market agent: {e}")
