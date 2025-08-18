# AgriYodha v1 - Financial Advisory Agent

A specialized AI agent built with LangGraph to provide data-driven financial guidance for Indian farmers.

## Features

The agent provides three core functionalities:

1. **Crop Profitability Analysis** - Calculates estimated costs, potential revenue, and net profitability for crops in specific regions
2. **Financial Product Search** - Finds relevant government schemes, loans, and insurance products
3. **Risk Assessment** - Evaluates financial risks based on crop type, region, credit score, and loan amount

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.template .env
```
Edit the `.env` file and add your OpenAI API key.

3. Run the agent:
```bash
python financial_agent.py
```

## Usage

The agent is designed to handle farmer queries like:
- "I have 5 acres of land in Kanpur district and want to grow wheat. What will be my approximate cost, and which government loan can I get for it? My CIBIL score is 720."
- "What are the risks of growing cotton in Maharashtra with a loan of 2 lakhs?"
- "Which government schemes are available for rice cultivation?"

## Agent Architecture

The agent uses LangGraph to create a stateful conversation flow:

1. **call_model**: Analyzes user query and decides which tools to use
2. **call_tool**: Executes the selected tools with appropriate parameters
3. **Conditional routing**: Continues tool execution or ends based on model decisions

## Tools

### calculate_crop_profitability
Provides cost and revenue analysis for crops in different regions.

### find_relevant_financial_products  
Searches for suitable loans, schemes, and insurance products.

### assess_farmer_risk_profile
Evaluates financial risks and provides recommendations.

## Important Notes

- The current implementation uses hardcoded data for demonstration purposes
- Always consult with banks and government officials before making financial decisions
- The agent provides guidance based on available data which may change with market conditions

## Next Steps

To make this production-ready:
1. Replace hardcoded data with real APIs/databases
2. Add more sophisticated risk assessment models
3. Include real-time market price data
4. Add support for more crops and regions

