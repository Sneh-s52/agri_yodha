"""
Financial Advisory Agent for Indian Farmers using LangGraph
AgriYodha v1 - A specialized agent for providing data-driven financial guidance
"""

from typing import Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the Agent's Tools with placeholder functions returning hardcoded data

@tool
def calculate_crop_profitability(crop: str, region: str, acres: float) -> dict:
    """Calculates the estimated cost of cultivation, potential revenue, and net profitability for a given 'crop' in a specific 'region' for a certain number of 'acres'. Returns a structured dictionary with all cost and revenue components."""
    
    # Hardcoded sample data for demonstration
    cost_per_acre = {
        'wheat': 12000,
        'rice': 15000,
        'sugarcane': 25000,
        'cotton': 18000,
        'maize': 10000
    }
    
    revenue_per_acre = {
        'wheat': 18000,
        'rice': 22000,
        'sugarcane': 35000,
        'cotton': 25000,
        'maize': 15000
    }
    
    crop_lower = crop.lower()
    base_cost = cost_per_acre.get(crop_lower, 12000)
    base_revenue = revenue_per_acre.get(crop_lower, 18000)
    
    # Regional adjustments
    regional_multiplier = {
        'punjab': 1.2,
        'haryana': 1.15,
        'uttar pradesh': 1.0,
        'bihar': 0.9,
        'west bengal': 0.95,
        'maharashtra': 1.1
    }
    
    multiplier = regional_multiplier.get(region.lower(), 1.0)
    
    estimated_cost = base_cost * acres * multiplier
    potential_revenue = base_revenue * acres * multiplier
    projected_profit = potential_revenue - estimated_cost
    
    return {
        'crop': crop,
        'region': region,
        'acres': acres,
        'estimated_cost': round(estimated_cost, 2),
        'potential_revenue': round(potential_revenue, 2),
        'projected_profit': round(projected_profit, 2),
        'cost_per_acre': round(base_cost * multiplier, 2),
        'revenue_per_acre': round(base_revenue * multiplier, 2),
        'profit_margin': round((projected_profit / potential_revenue) * 100, 2) if potential_revenue > 0 else 0
    }

@tool
def find_relevant_financial_products(query: str, farmer_profile: dict = None) -> list[dict]:
    """Searches the knowledge base for suitable financial products (loans, insurance, schemes) based on the user's 'query' and optional 'farmer_profile'. The profile can include details like land ownership, credit score, and location. Returns a list of relevant products with summaries and source links."""
    
    # Hardcoded financial products database
    products = [
        {
            'product_name': 'Kisan Credit Card (KCC)',
            'type': 'Loan',
            'summary': 'Low-interest loan for farming needs including crop cultivation, post-harvest expenses, and maintenance of farm assets. Interest rate: 7% per annum.',
            'eligibility': 'All farmers with land ownership or tenant farmers',
            'max_amount': '3 lakh for crop cultivation',
            'source': 'nabard.org'
        },
        {
            'product_name': 'PM-KISAN Scheme',
            'type': 'Direct Benefit Transfer',
            'summary': 'Direct income support of Rs. 6000 per year to all landholding farmer families in three equal installments.',
            'eligibility': 'All landholding farmer families',
            'max_amount': '6000 per year',
            'source': 'pmkisan.gov.in'
        },
        {
            'product_name': 'Pradhan Mantri Fasal Bima Yojana (PMFBY)',
            'type': 'Crop Insurance',
            'summary': 'Comprehensive crop insurance scheme covering pre-sowing to post-harvest losses due to natural calamities.',
            'eligibility': 'All farmers growing notified crops',
            'premium': '2% for Kharif, 1.5% for Rabi crops',
            'source': 'pmfby.gov.in'
        },
        {
            'product_name': 'Agriculture Infrastructure Fund',
            'type': 'Infrastructure Loan',
            'summary': 'Medium to long-term debt financing for agriculture infrastructure projects. Interest subvention of 3% per annum.',
            'eligibility': 'Farmers, FPOs, Agri-entrepreneurs, Startups',
            'max_amount': '2 crore',
            'source': 'agriinfra.dac.gov.in'
        },
        {
            'product_name': 'Stand Up India Scheme',
            'type': 'Loan',
            'summary': 'Bank loans between Rs. 10 lakh to Rs. 1 crore for SC/ST and women entrepreneurs in agriculture sector.',
            'eligibility': 'SC/ST and Women entrepreneurs',
            'max_amount': '1 crore',
            'source': 'standupmitra.in'
        }
    ]
    
    # Simple keyword matching for relevant products
    query_lower = query.lower()
    relevant_products = []
    
    for product in products:
        if (any(keyword in query_lower for keyword in ['loan', 'credit', 'kisan']) and product['type'] in ['Loan']) or \
           (any(keyword in query_lower for keyword in ['insurance', 'crop insurance']) and product['type'] == 'Crop Insurance') or \
           (any(keyword in query_lower for keyword in ['scheme', 'benefit', 'support']) and product['type'] == 'Direct Benefit Transfer') or \
           (any(keyword in query_lower for keyword in ['infrastructure']) and product['type'] == 'Infrastructure Loan'):
            relevant_products.append(product)
    
    # If no specific match, return KCC and PM-KISAN as default relevant products
    if not relevant_products:
        relevant_products = [products[0], products[1]]  # KCC and PM-KISAN
    
    return relevant_products

@tool
def assess_farmer_risk_profile(crop: str, region: str, credit_score: int, loan_amount: float) -> dict:
    """Assesses the financial risk associated with a farmer's plan. Considers the 'crop' type's market volatility, 'region'-specific risks, the farmer's 'credit_score', and the requested 'loan_amount'. Returns a risk rating (e.g., 'Low', 'Medium', 'High') and the reasons."""
    
    # Risk assessment logic based on various factors
    risk_factors = []
    risk_score = 0
    
    # Crop volatility assessment
    high_volatility_crops = ['cotton', 'sugarcane', 'chili', 'onion']
    medium_volatility_crops = ['rice', 'maize', 'soybean']
    low_volatility_crops = ['wheat', 'pulses']
    
    if crop.lower() in high_volatility_crops:
        risk_score += 30
        risk_factors.append(f"{crop} has high market price volatility")
    elif crop.lower() in medium_volatility_crops:
        risk_score += 20
        risk_factors.append(f"{crop} has moderate market price volatility")
    else:
        risk_score += 10
        risk_factors.append(f"{crop} has relatively stable market prices")
    
    # Regional risk assessment
    high_risk_regions = ['rajasthan', 'gujarat', 'maharashtra']  # Drought-prone
    medium_risk_regions = ['uttar pradesh', 'bihar', 'west bengal']
    
    if region.lower() in high_risk_regions:
        risk_score += 25
        risk_factors.append(f"{region} is prone to weather-related risks")
    elif region.lower() in medium_risk_regions:
        risk_score += 15
        risk_factors.append(f"{region} has moderate weather-related risks")
    else:
        risk_score += 5
        risk_factors.append(f"{region} has relatively lower weather risks")
    
    # Credit score assessment
    if credit_score >= 750:
        risk_score -= 10
        risk_factors.append("Excellent credit score indicates strong repayment capability")
    elif credit_score >= 650:
        risk_score += 5
        risk_factors.append("Good credit score with manageable repayment risk")
    else:
        risk_score += 20
        risk_factors.append("Credit score indicates higher repayment risk")
    
    # Loan amount assessment (assuming typical farm income)
    if loan_amount > 500000:
        risk_score += 15
        risk_factors.append("High loan amount relative to typical farm income")
    elif loan_amount > 200000:
        risk_score += 10
        risk_factors.append("Moderate loan amount")
    else:
        risk_score += 0
        risk_factors.append("Conservative loan amount")
    
    # Determine overall risk rating
    if risk_score <= 30:
        risk_rating = "Low"
    elif risk_score <= 60:
        risk_rating = "Medium"
    else:
        risk_rating = "High"
    
    recommendations = []
    if risk_rating == "High":
        recommendations.extend([
            "Consider crop insurance to mitigate weather risks",
            "Start with smaller loan amount and scale gradually",
            "Diversify crops to reduce market volatility impact"
        ])
    elif risk_rating == "Medium":
        recommendations.extend([
            "Crop insurance is recommended",
            "Monitor market prices closely",
            "Maintain emergency fund for unforeseen circumstances"
        ])
    else:
        recommendations.extend([
            "Consider optional crop insurance for additional security",
            "Good candidate for agricultural loans"
        ])
    
    return {
        'risk_rating': risk_rating,
        'risk_score': risk_score,
        'reasons': risk_factors,
        'recommendations': recommendations,
        'loan_approval_likelihood': 'High' if risk_score <= 40 else 'Medium' if risk_score <= 65 else 'Low'
    }

# Set up the Language Model and Tools
tools = [calculate_crop_profitability, find_relevant_financial_products, assess_farmer_risk_profile]
tool_node = ToolNode(tools)

# Initialize the language model (make sure to set your OpenAI API key)
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
model = model.bind_tools(tools)

# Define the Agent's State with proper message reduction
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define the Graph Nodes
def call_model(state: AgentState):
    """Node that calls the language model"""
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState):
    """Node that executes tools based on the model's tool calls"""
    return tool_node.invoke(state)

# Define Conditional Logic
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determines whether to continue with tool calls or end the conversation"""
    messages = state['messages']
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tool execution
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    else:
        return "end"

# Create the System Prompt
SYSTEM_PROMPT = """You are AgriYodha v1, a specialized Financial Advisory Agent for Indian farmers. Your primary goal is to provide clear, data-driven, and responsible financial guidance.

Your Mandate:
1. **Analyze & Calculate First**: When a user asks about costs or profitability, you MUST use the `calculate_crop_profitability` tool to get precise numbers.
2. **Find Relevant Products**: When a user asks for loans or schemes, you MUST use the `find_relevant_financial_products` tool to search your knowledge base.
3. **Assess Risk**: For any plan involving a loan, you MUST use the `assess_farmer_risk_profile` tool to provide a risk assessment.
4. **Synthesize, Don't Invent**: Your final answer MUST be a synthesis of the information returned by your tools. Do not add information from your general knowledge. If the tools don't provide an answer, state that the information is unavailable.
5. **Be Cautious**: Always include a disclaimer that your advice is based on available data and market conditions, which can change. Advise the user to always consult directly with a bank or government official before making a final decision.

Think step-by-step about which tools you need to call to answer the user's query completely. Then, call the necessary tools. Finally, synthesize the results from the tools into a comprehensive and responsible answer for the user."""

# Assemble the Graph
def create_agent():
    """Create and compile the LangGraph agent"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", call_tool)
    
    # Set entry point
    workflow.set_entry_point("call_model")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "continue": "call_tool",
            "end": END
        }
    )
    
    # Add edge from tool back to model
    workflow.add_edge("call_tool", "call_model")
    
    # Compile the graph
    app = workflow.compile()
    return app

# Main execution function
def run_agent(user_query: str):
    """Run the agent with a user query"""
    app = create_agent()
    
    # Create initial state with user query only (system prompt is handled by the model)
    initial_messages = [
        HumanMessage(content=f"{SYSTEM_PROMPT}\n\nUser Query: {user_query}")
    ]
    
    initial_state = {"messages": initial_messages}
    
    print("="*80)
    print("AGRIYODHA v1 - FINANCIAL ADVISORY AGENT")
    print("="*80)
    print(f"User Query: {user_query}")
    print("-"*80)
    
    # Stream the execution
    for step_output in app.stream(initial_state):
        for node_name, output in step_output.items():
            print(f"\n--- {node_name.upper()} ---")
            
            if node_name == "call_model":
                last_message = output["messages"][-1]
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    print("Model decided to call tools:")
                    for tool_call in last_message.tool_calls:
                        print(f"  - {tool_call['name']} with args: {tool_call['args']}")
                else:
                    print("Model Response:")
                    print(last_message.content)
            
            elif node_name == "call_tool":
                print("Tool Results:")
                for message in output["messages"]:
                    if isinstance(message, ToolMessage):
                        print(f"  - {message.name}: {message.content}")
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETED")
    print("="*80)

if __name__ == "__main__":
    # Sample user query
    sample_query = "I have 5 acres of land in Kanpur district and want to grow wheat. What will be my approximate cost, and which government loan can I get for it? My CIBIL score is 720."
    
    # Check if OpenAI API key is set
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("❌ ERROR: OPENAI_API_KEY not found or not set properly in environment variables.")
        print("\n📋 REQUIREMENTS TO RUN FINANCE AGENT:")
        print("="*60)
        print("1. OpenAI API Key:")
        print("   - Get an API key from: https://platform.openai.com/api-keys")
        print("   - Add it to your .env file as: OPENAI_API_KEY=sk-your-actual-key-here")
        print("   - Current value in .env:", openai_key)
        print("\n2. Dependencies (already installed):")
        print("   ✅ langchain, langchain-openai, langgraph, python-dotenv")
        print("\n3. Internet connection for OpenAI API calls")
        print("\n📝 Note: The agent uses hardcoded data for demonstrations.")
        print("No external databases or additional APIs are required.")
        print("="*60)
    else:
        print("✅ OpenAI API key found. Starting agent...")
        try:
            run_agent(sample_query)
        except Exception as e:
            print(f"❌ Error running agent: {e}")
            print("\n💡 Common issues:")
            print("- Invalid OpenAI API key")
            print("- No internet connection")
            print("- API rate limits exceeded")
