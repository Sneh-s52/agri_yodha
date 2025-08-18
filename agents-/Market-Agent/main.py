#!/usr/bin/env python3
# market_agent/main.py

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agent.market_agent import create_market_agent
from config import OPENAI_API_KEY, TAVILY_API_KEY

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarketAgentCLI:
    """
    Command-line interface for the Market Agent.
    Provides an interactive terminal interface for users to query the market agent.
    """

    def __init__(self):
        """Initialize the CLI interface."""
        self.agent = None
        self.running = False
        
    def initialize_agent(self):
        """Initialize the market agent."""
        try:
            logger.info("Initializing Market Agent...")
            self.agent = create_market_agent()
            logger.info("Market Agent initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Market Agent: {e}")
            print(f"❌ Failed to initialize Market Agent: {e}")
            return False

    def display_banner(self):
        """Display the welcome banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🚀 AGRI YODHA AI MARKET AGENT 🚀         ║
║                                                              ║
║  Next Generation AI-Powered Chatbot for Policy Makers &     ║
║  Financiers - Powered by GPT-4o                             ║
║                                                              ║
║  AI Capabilities:                                            ║
║  • GPT-4o Powered Market Analysis & Intelligence            ║
║  • AI-Enhanced Price Forecasting & Risk Assessment          ║
║  • Intelligent Policy Recommendations & Strategic Insights   ║
║  • News Sentiment Analysis with AI Interpretation           ║
║  • Multi-source Data Integration (MongoDB, Pinecone)        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def display_help(self):
        """Display help information."""
        help_text = """
📚 AVAILABLE COMMANDS:

🔍 AI-POWERED MARKET ANALYSIS:
  analyze <commodity> <apmc>     - AI-powered comprehensive market analysis
  forecast <commodity> <apmc>    - AI-enhanced price forecasting (7 days)
  intelligence [commodity]       - AI market intelligence report
  
📰 AI NEWS & SENTIMENT:
  news <query>                   - AI-powered news search & analysis
  sentiment <query>              - AI market sentiment analysis
  
📊 AI DATA & REPORTS:
  export <commodity> <apmc>      - Export AI-enhanced analysis report
  history                        - View conversation history
  clear                          - Clear conversation history
  
⚙️ SYSTEM:
  help                           - Show this help message
  status                         - Show system status
  quit/exit                     - Exit the application

💬 CONVERSATIONAL AI:
  You can now ask questions in natural language! Examples:
  
  • "What will happen to wheat prices if there's a drought?"
  • "How should I prepare for monsoon season in Maharashtra?"
  • "What are the current market trends for rice?"
  • "If there's a hurricane coming, what should farmers do?"
  • "Hello, how can you help me with market analysis?"
  • "Tell me about the current agricultural market situation"
  
  The AI will understand your question and provide intelligent, 
  context-aware responses using GPT-4o!

💡 EXAMPLE USAGE:
  analyze wheat "Delhi APMC"     # AI-powered market analysis
  forecast rice "Mumbai APMC"    # AI-enhanced price forecasting
  news "wheat price increase India"  # AI news analysis
  intelligence wheat             # AI market intelligence
  export wheat "Delhi APMC"     # AI-enhanced report export

🤖 AI Features:
  • GPT-4o powered reasoning and analysis
  • Natural language understanding and conversation
  • Intelligent policy recommendations
  • AI risk assessment and mitigation strategies
  • Strategic market insights and forecasting
        """
        print(help_text)

    def display_status(self):
        """Display system status."""
        if not self.agent:
            print("❌ Market Agent not initialized")
            return
            
        status = {
            "Agent Status": "✅ AI-Powered Running",
            "AI Model": f"✅ {self.agent.llm_model}",
            "MongoDB": "✅ Connected" if self.agent.mongo_client else "❌ Not Available",
            "Pinecone": "✅ Connected" if self.agent.pinecone_client else "❌ Not Available",
            "News Tool": "✅ Available" if self.agent.news_tool else "❌ Not Available",
            "Forecasting": "✅ AI-Enhanced" if self.agent.forecasting_tool else "❌ Not Available",
            "Last Analysis": len(self.agent.last_analysis),
            "Conversation History": len(self.agent.conversation_history)
        }
        
        print("\n📊 SYSTEM STATUS:")
        print("=" * 50)
        for key, value in status.items():
            print(f"{key:<20}: {value}")
        print("=" * 50)

    def parse_command(self, command: str) -> List[str]:
        """Parse user command into tokens."""
        # Handle quoted strings properly
        import shlex
        try:
            return shlex.split(command.strip())
        except:
            # Fallback to simple split if shlex fails
            return command.strip().split()

    def handle_analyze(self, args: List[str]):
        """Handle market analysis command."""
        if len(args) < 2:
            print("❌ Usage: analyze <commodity> <apmc>")
            print("   Example: analyze wheat \"Delhi APMC\"")
            return
            
        commodity = args[0]
        apmc = args[1]
        
        print(f"\n🔍 Analyzing market conditions for {commodity} at {apmc}...")
        print("This may take a few moments...")
        
        try:
            analysis = self.agent.analyze_market_conditions(commodity, apmc)
            
            if "error" in analysis:
                print(f"❌ Analysis failed: {analysis['error']}")
                return
                
            # Display analysis results
            print(f"\n✅ AI-Powered Market Analysis Complete!")
            print("=" * 60)
            print(f"Commodity: {analysis['commodity']}")
            print(f"APMC: {analysis['apmc']}")
            print(f"Data Sources: {', '.join(analysis['data_sources'])}")
            
            # Current market data
            if analysis['current_market_data']:
                print(f"\n📊 Current Market Data:")
                for source, data in analysis['current_market_data'].items():
                    if source == "mongodb" and "modal_price" in data:
                        print(f"  MongoDB Price: ₹{data['modal_price']} per {data.get('Commodity_Uom', 'unit')}")
                    elif source == "pinecone" and "modal_price" in data:
                        print(f"  Pinecone Price: ₹{data['modal_price']} per {data.get('Commodity_Uom', 'unit')}")
                        
            # AI Insights
            if 'ai_insights' in analysis and 'structured_insights' in analysis['ai_insights']:
                insights = analysis['ai_insights']['structured_insights']
                print(f"\n🤖 AI-Powered Insights:")
                if 'market_intelligence' in insights and insights['market_intelligence']:
                    print(f"  Market Intelligence: {insights['market_intelligence'][:100]}...")
                if 'price_analysis' in insights and insights['price_analysis']:
                    print(f"  Price Analysis: {insights['price_analysis'][:100]}...")
                    
            # Price forecast
            if analysis['price_forecast'] and 'forecasts' in analysis['price_forecast']:
                print(f"\n🔮 AI-Enhanced Price Forecast (7 days):")
                ensemble = analysis['price_forecast'].get('ensemble', {})
                if 'forecast_values' in ensemble:
                    print(f"  Next 7 days: {[f'₹{p:.2f}' for p in ensemble['forecast_values']]}")
                    
                # Show AI enhancement if available
                if 'ai_enhancement' in analysis['price_forecast']:
                    print(f"  AI Analysis: {analysis['price_forecast']['ai_enhancement'][:150]}...")
                    
            # AI Recommendations
            if analysis['intelligent_recommendations']:
                print(f"\n💡 AI-Generated Recommendations:")
                for i, rec in enumerate(analysis['intelligent_recommendations'][:5], 1):
                    print(f"  {i}. {rec}")
                    
            # Risk assessment
            if analysis['risk_assessment']:
                risk = analysis['risk_assessment']
                print(f"\n⚠️ AI Risk Assessment:")
                print(f"  Overall Risk: {risk['overall_risk_level'].upper()}")
                if risk['risk_factors']:
                    print(f"  Risk Factors: {', '.join(risk['risk_factors'][:3])}")
                if risk['mitigation_strategies']:
                    print(f"  Mitigation: {risk['mitigation_strategies'][0]}")
                    
            # Policy implications
            if analysis['policy_implications']:
                print(f"\n🏛️ AI Policy Implications:")
                for i, policy in enumerate(analysis['policy_implications'][:3], 1):
                    print(f"  {i}. {policy}")
                    
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            print(f"❌ Analysis failed: {e}")

    def handle_forecast(self, args: List[str]):
        """Handle price forecasting command."""
        if len(args) < 2:
            print("❌ Usage: forecast <commodity> <apmc>")
            print("   Example: forecast wheat \"Delhi APMC\"")
            return
            
        commodity = args[0]
        apmc = args[1]
        
        print(f"\n🔮 Generating price forecast for {commodity} at {apmc}...")
        
        try:
            forecast = self.agent.get_price_forecast(commodity, apmc)
            
            if "error" in forecast:
                print(f"❌ Forecasting failed: {forecast['error']}")
                return
                
            print(f"\n✅ AI-Enhanced Price Forecast Complete!")
            print("=" * 50)
            print(f"Commodity: {forecast['commodity']}")
            print(f"APMC: {forecast['apmc']}")
            print(f"Forecast Period: {forecast['forecast_period']}")
            print(f"Data Quality: {forecast['methodology']['data_quality']}")
            
            # Display forecasts
            if 'forecasts' in forecast:
                print(f"\n📈 AI-Enhanced Forecast Results:")
                for method, result in forecast['forecasts'].items():
                    if 'forecast_values' in result:
                        print(f"  {method.replace('_', ' ').title()}: {[f'₹{p:.2f}' for p in result['forecast_values']]}")
                        
            # AI Enhancement
            if 'ai_enhancement' in forecast:
                print(f"\n🤖 AI Analysis:")
                print(f"  {forecast['ai_enhancement'][:200]}...")
                        
            # Recommendations
            if 'recommendations' in forecast:
                print(f"\n💡 AI-Generated Recommendations:")
                for i, rec in enumerate(forecast['recommendations'][:3], 1):
                    print(f"  {i}. {rec}")
                    
        except Exception as e:
            logger.error(f"Error in price forecasting: {e}")
            print(f"❌ Forecasting failed: {e}")

    def handle_intelligence(self, args: List[str]):
        """Handle market intelligence command."""
        commodity = args[0] if args else None
        
        print(f"\n🧠 Generating market intelligence report...")
        if commodity:
            print(f"Focusing on: {commodity}")
        
        try:
            intelligence = self.agent.get_market_intelligence(commodity=commodity)
            
            if "error" in intelligence:
                print(f"❌ Intelligence failed: {intelligence['error']}")
                return
                
            print(f"\n✅ AI-Powered Market Intelligence Complete!")
            print("=" * 60)
            print(f"Report Generated: {intelligence['report_timestamp']}")
            
            # Market overview
            if 'market_overview' in intelligence:
                overview = intelligence['market_overview']
                print(f"\n📊 Market Overview:")
                print(f"  Total Commodities: {overview.get('total_commodities', 'N/A')}")
                print(f"  Total APMCs: {overview.get('total_apmcs', 'N/A')}")
                
            # AI Insights
            if 'ai_insights' in intelligence:
                print(f"\n🤖 AI Strategic Insights:")
                print(f"  {intelligence['ai_insights'][:300]}...")
                
            # Strategic recommendations
            if 'strategic_recommendations' in intelligence:
                print(f"\n💡 AI Strategic Recommendations:")
                for i, rec in enumerate(intelligence['strategic_recommendations'][:5], 1):
                    print(f"  {i}. {rec}")
                    
            # Policy framework
            if 'policy_framework' in intelligence:
                print(f"\n🏛️ AI Policy Framework:")
                for i, policy in enumerate(intelligence['policy_framework'][:3], 1):
                    print(f"  {i}. {policy}")
                    
        except Exception as e:
            logger.error(f"Error in market intelligence: {e}")
            print(f"❌ Intelligence failed: {e}")

    def handle_news(self, args: List[str]):
        """Handle news search command."""
        if not args:
            print("❌ Usage: news <query>")
            return
            
        query = " ".join(args)
        print(f"\n📰 Searching news for: {query}")
        
        try:
            news_results = self.agent.search_market_news(query)
            
            if "error" in news_results:
                print(f"❌ News search failed: {news_results['error']}")
                return
                
            print(f"\n✅ AI-Powered News Search Complete!")
            print("=" * 50)
            print(f"Query: {news_results['query']}")
            print(f"Articles Found: {len(news_results['articles'])}")
            
            # Display articles
            if 'articles' in news_results:
                print(f"\n📄 Top Articles:")
                for i, article in enumerate(news_results['articles'][:5], 1):
                    print(f"  {i}. {article['title']}")
                    print(f"     Source: {article['source']}")
                    print(f"     URL: {article['url']}")
                    print()
                    
            # Sentiment analysis
            if 'sentiment' in news_results:
                sentiment = news_results['sentiment']
                if 'sentiment_label' in sentiment:
                    print(f"📊 Market Sentiment: {sentiment['sentiment_label']}")
                    if 'sentiment_score' in sentiment:
                        print(f"  Score: {sentiment['sentiment_score']:.2f}")
                        
            # AI Analysis
            if 'ai_analysis' in news_results:
                print(f"\n🤖 AI-Powered News Analysis:")
                print(f"  {news_results['ai_analysis'][:300]}...")
                        
        except Exception as e:
            logger.error(f"Error in news search: {e}")
            print(f"❌ News search failed: {e}")

    def handle_export(self, args: List[str]):
        """Handle export command."""
        if len(args) < 2:
            print("❌ Usage: export <commodity> <apmc>")
            return
            
        commodity = args[0]
        apmc = args[1]
        
        print(f"\n📤 Exporting analysis report for {commodity} at {apmc}...")
        
        try:
            filepath = self.agent.export_analysis_report(commodity, apmc, "json")
            
            if filepath.startswith("Export failed"):
                print(f"❌ Export failed: {filepath}")
                return
                
            print(f"✅ Report exported successfully!")
            print(f"📁 File: {filepath}")
            
        except Exception as e:
            logger.error(f"Error in export: {e}")
            print(f"❌ Export failed: {e}")

    def handle_history(self):
        """Handle history command."""
        if not self.agent:
            print("❌ Market Agent not initialized")
            return
            
        history = self.agent.get_conversation_history()
        
        if not history:
            print("📝 No conversation history found.")
            return
            
        print(f"\n📝 Conversation History ({len(history)} entries):")
        print("=" * 50)
        
        for i, entry in enumerate(history[-10:], 1):  # Show last 10 entries
            print(f"{i}. {entry.get('timestamp', 'N/A')} - {entry.get('query', 'N/A')}")

    def handle_clear(self):
        """Handle clear command."""
        if not self.agent:
            print("❌ Market Agent not initialized")
            return
            
        self.agent.clear_conversation_history()
        print("🧹 Conversation history cleared!")

    def handle_conversational_query(self, query: str):
        """
        Handle conversational queries using comprehensive tool integration.
        This method calls all relevant tools and data sources before providing an answer.
        """
        try:
            print(f"\n🤖 Processing your query: {query}")
            print("🔍 Gathering data from multiple sources...")
            
            # Use the agent's comprehensive conversational query handler
            response = self.agent.handle_conversational_query(query)
            
            if "error" in response:
                print(f"❌ Query processing failed: {response['error']}")
                return
                
            print(f"\n✅ Comprehensive Analysis Complete!")
            print("=" * 60)
            
            # Show data sources used
            if response.get('data_sources_used'):
                print(f"📊 Data Sources Used: {', '.join(response['data_sources_used'])}")
                print()
                
            # Show market data summary
            if response.get('market_data'):
                market_data = response['market_data']
                print("📈 Market Data Summary:")
                
                if 'mongodb_latest' in market_data:
                    latest = market_data['mongodb_latest']
                    if 'modal_price' in latest:
                        print(f"  • Latest Price (MongoDB): ₹{latest['modal_price']}")
                        
                if 'pinecone_latest' in market_data:
                    latest = market_data['pinecone_latest']
                    if 'modal_price' in latest:
                        print(f"  • Latest Price (Pinecone): ₹{latest['modal_price']}")
                        

                        
                if 'mongodb_history' in market_data:
                    history = market_data['mongodb_history']
                    print(f"  • Historical Data: {history['data_points']} data points")
                    if history.get('avg_price'):
                        print(f"  • Average Price: ₹{history['avg_price']:.2f}")
                        
                if 'market_overview' in market_data:
                    overview = market_data['market_overview']
                    print(f"  • Available Commodities: {overview['total_commodities']}")
                    print(f"  • Available APMCs: {overview['total_apmcs']}")
                    
                print()
                
            # Show news summary
            if response.get('news_data') and response['news_data'].get('articles'):
                news_data = response['news_data']
                print("📰 Recent News:")
                for i, article in enumerate(news_data['articles'][:3], 1):
                    print(f"  {i}. {article['title'][:80]}...")
                print()
                
            # Show forecast summary
            if response.get('forecast_data'):
                forecast_data = response['forecast_data']
                print("🔮 Price Forecast:")
                if 'forecasts' in forecast_data:
                    for method, result in forecast_data['forecasts'].items():
                        if 'forecast_values' in result:
                            print(f"  • {method.replace('_', ' ').title()}: {[f'₹{p:.2f}' for p in result['forecast_values'][:3]]}...")
                print()
                
            # Show AI analysis
            if response.get('ai_analysis'):
                print("🤖 AI-Powered Analysis:")
                print("=" * 60)
                print(response['ai_analysis'])
                print("=" * 60)
                
            # Show recommendations
            if response.get('recommendations'):
                print(f"\n💡 Key Recommendations:")
                for i, rec in enumerate(response['recommendations'], 1):
                    print(f"  {i}. {rec}")
                    
            # Store in conversation history
            if hasattr(self.agent, 'conversation_history'):
                self.agent.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'query': query,
                    'response': response.get('ai_analysis', ''),
                    'type': 'comprehensive_analysis',
                    'data_sources': response.get('data_sources_used', [])
                })
                
        except Exception as e:
            logger.error(f"Error in conversational query: {e}")
            print(f"❌ Sorry, I encountered an error processing your query: {e}")
            print("💡 Try using specific commands like 'analyze', 'forecast', or 'news'")

    def is_conversational_query(self, command: str) -> bool:
        """
        Determine if the input is a conversational query rather than a command.
        """
        # List of known commands
        known_commands = [
            'analyze', 'analysis', 'forecast', 'prediction', 'intelligence', 'intel',
            'news', 'search', 'export', 'save', 'history', 'hist', 'clear', 'clean',
            'help', 'h', '?', 'status', 'info', 'quit', 'exit', 'q'
        ]
        
        # Check if the first word is a known command
        first_word = command.strip().split()[0].lower()
        
        # If it's not a known command, treat it as conversational
        if first_word not in known_commands:
            return True
            
        # Also check for natural language patterns that indicate conversational queries
        conversational_indicators = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'suppose', 'imagine', 'if', 'can you', 'please', 'hello', 'hi',
            'tell me', 'explain', 'describe', 'compare', 'the', 'a', 'an',
            'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should'
        ]
        
        command_lower = command.lower()
        
        # Check for conversational patterns
        for indicator in conversational_indicators:
            if indicator in command_lower:
                return True
                
        # Check for longer queries that are likely conversational
        words = command.strip().split()
        if len(words) > 3:  # If query has more than 3 words, likely conversational
            return True
            
        # Check for specific conversational patterns even if they start with command words
        conversational_patterns = [
            'forecast the price of', 'analyze the market for', 'what is the price of',
            'how much is', 'tell me about', 'explain the', 'describe the'
        ]
        
        for pattern in conversational_patterns:
            if pattern in command_lower:
                return True
                
        return False

    def process_command(self, command: str):
        """Process user command with enhanced conversational AI support."""
        if not command.strip():
            return
            
        # Check if this is a conversational query
        if self.is_conversational_query(command):
            self.handle_conversational_query(command)
            return
            
        # Process as regular command
        tokens = self.parse_command(command)
        if not tokens:
            return
            
        cmd = tokens[0].lower()
        args = tokens[1:]
        
        try:
            if cmd in ['analyze', 'analysis']:
                self.handle_analyze(args)
            elif cmd in ['forecast', 'prediction']:
                self.handle_forecast(args)
            elif cmd in ['intelligence', 'intel']:
                self.handle_intelligence(args)
            elif cmd in ['news', 'search']:
                self.handle_news(args)
            elif cmd in ['export', 'save']:
                self.handle_export(args)
            elif cmd in ['history', 'hist']:
                self.handle_history()
            elif cmd in ['clear', 'clean']:
                self.handle_clear()
            elif cmd in ['help', 'h', '?']:
                self.display_help()
            elif cmd in ['status', 'info']:
                self.display_status()
            elif cmd in ['quit', 'exit', 'q']:
                self.running = False
                print("\n👋 Thank you for using Agri Yodha Market Agent!")
                return
            else:
                # If we get here, it might be a conversational query
                print(f"🤔 I'm not sure how to handle '{cmd}'. Let me try to help you conversationally...")
                self.handle_conversational_query(command)
                
        except Exception as e:
            logger.error(f"Error processing command '{cmd}': {e}")
            print(f"❌ Error processing command: {e}")
            print("💡 Try asking your question in natural language or use 'help' for available commands")

    def run(self):
        """Run the CLI interface."""
        print("🚀 Starting Agri Yodha Market Agent...")
        
        # Check API keys
        if not OPENAI_API_KEY:
            print("⚠️ Warning: OPENAI_API_KEY not set")
        if not TAVILY_API_KEY:
            print("⚠️ Warning: TAVILY_API_KEY not set")
            
        # Initialize agent
        if not self.initialize_agent():
            print("❌ Failed to initialize Market Agent. Exiting.")
            return
            
        # Display banner
        self.display_banner()
        
        # Display help
        self.display_help()
        
        # Main command loop
        self.running = True
        while self.running:
            try:
                command = input("\n🤖 Market Agent > ").strip()
                if command:
                    self.process_command(command)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                print(f"❌ Unexpected error: {e}")


def main():
    """Main entry point."""
    try:
        cli = MarketAgentCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
