#!/usr/bin/env python3
"""
Agricultural Advisory Multi-Agent Orchestrator using LangGraph

This orchestrator manages a sophisticated multi-agent system that provides
comprehensive farming advice by coordinating specialized agents in a dynamic,
stateful workflow.

Author: AI Assistant
Date: 2025
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, TypedDict, Literal
from datetime import datetime
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging with UTF-8 encoding for Windows compatibility
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('orchestrator.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class OrchestratorState(TypedDict):
    """State for the agricultural orchestrator workflow."""
    user_query: str
    todo_list: List[str]
    results_log: Dict[str, Any]
    current_task: Optional[str]
    iteration_count: int
    final_report: Optional[str]
    final_recommendation: Optional[str]


# ============================================================================
# AGENT PLACEHOLDER FUNCTIONS
# ============================================================================

def call_soil_agent(location: str) -> Dict[str, Any]:
    """
    Soil Agent: Provides soil health analysis and suitable crops.
    
    Args:
        location: Geographic location for soil analysis
        
    Returns:
        Dict containing soil health data and suitable crops
    """
    logger.info(f"[SOIL] Calling Soil Agent for location: {location}")
    
    # Mock data simulating ICAR/UN-FAO trained RAG system
    mock_response = {
        "agent": "soil_agent",
        "location": location,
        "soil_health": {
            "ph_level": 6.8,
            "organic_matter": "Medium (2.5%)",
            "nitrogen": "Low",
            "phosphorus": "Medium",
            "potassium": "High",
            "soil_type": "Alluvial",
            "drainage": "Good",
            "salinity": "Normal"
        },
        "suitable_crops": [
            "Rice", "Wheat", "Maize", "Sugarcane", "Cotton"
        ],
        "recommendations": [
            "Apply nitrogen-rich fertilizers before sowing",
            "Consider crop rotation with legumes",
            "Monitor soil moisture regularly"
        ],
        "fertilizer_requirements": {
            "NPK_ratio": "120:60:40 kg/ha",
            "organic_manure": "5-7 tonnes/ha"
        },
        "confidence_score": 0.85,
        "data_sources": ["ICAR database", "UN-FAO soil maps", "Local soil surveys"]
    }
    
    logger.info(f"[SOIL] Soil Agent completed: Found {len(mock_response['suitable_crops'])} suitable crops")
    return mock_response


def call_market_agent(crops: List[str], location: str) -> Dict[str, Any]:
    """
    Market Agent: Analyzes market prices and profitability.
    
    Args:
        crops: List of crops to analyze
        location: Geographic location for market analysis
        
    Returns:
        Dict containing market analysis and profitability data
    """
    logger.info(f"[MARKET] Calling Market Agent for crops: {crops} in {location}")
    
    # Mock market data
    crop_prices = {
        "Rice": {"current_price": 2100, "forecast_price": 2250, "demand": "High"},
        "Wheat": {"current_price": 2000, "forecast_price": 2100, "demand": "Medium"},
        "Maize": {"current_price": 1800, "forecast_price": 1950, "demand": "High"},
        "Sugarcane": {"current_price": 350, "forecast_price": 380, "demand": "Medium"},
        "Cotton": {"current_price": 5800, "forecast_price": 6200, "demand": "High"}
    }
    
    mock_response = {
        "agent": "market_agent",
        "location": location,
        "analyzed_crops": crops,
        "market_analysis": {},
        "profitability_ranking": [],
        "market_trends": {
            "overall_trend": "Bullish",
            "seasonal_factors": "Kharif season demand increasing",
            "export_opportunities": "Strong for Rice and Cotton"
        },
        "confidence_score": 0.78
    }
    
    # Generate analysis for each crop
    for crop in crops:
        if crop in crop_prices:
            price_data = crop_prices[crop]
            profitability = (price_data["forecast_price"] - price_data["current_price"]) / price_data["current_price"] * 100
            
            mock_response["market_analysis"][crop] = {
                "current_price_per_quintal": price_data["current_price"],
                "forecast_price_per_quintal": price_data["forecast_price"],
                "demand_level": price_data["demand"],
                "expected_profitability": f"{profitability:.1f}%",
                "market_volatility": "Medium",
                "storage_costs": "Low" if crop in ["Rice", "Wheat"] else "Medium"
            }
            
            mock_response["profitability_ranking"].append({
                "crop": crop,
                "profitability_score": profitability,
                "recommendation": "Highly Recommended" if profitability > 5 else "Recommended"
            })
    
    # Sort by profitability
    mock_response["profitability_ranking"].sort(key=lambda x: x["profitability_score"], reverse=True)
    
    logger.info(f"[MARKET] Market Agent completed: Analyzed {len(crops)} crops")
    return mock_response


def call_financier_agent(crops: List[str], farm_size: float = 5.0) -> Dict[str, Any]:
    """
    Financier Agent: Calculates costs, ROI, and financial planning.
    
    Args:
        crops: List of crops for financial analysis
        farm_size: Farm size in acres
        
    Returns:
        Dict containing financial analysis and ROI calculations
    """
    logger.info(f"[FINANCE] Calling Financier Agent for crops: {crops}, farm size: {farm_size} acres")
    
    # Mock cost data per acre
    crop_costs = {
        "Rice": {"seed": 2000, "fertilizer": 8000, "pesticide": 3000, "labor": 12000, "irrigation": 4000},
        "Wheat": {"seed": 1500, "fertilizer": 6000, "pesticide": 2500, "labor": 10000, "irrigation": 3000},
        "Maize": {"seed": 1800, "fertilizer": 7000, "pesticide": 2800, "labor": 11000, "irrigation": 3500},
        "Sugarcane": {"seed": 15000, "fertilizer": 12000, "pesticide": 5000, "labor": 20000, "irrigation": 8000},
        "Cotton": {"seed": 3000, "fertilizer": 10000, "pesticide": 8000, "labor": 15000, "irrigation": 6000}
    }
    
    crop_yields = {
        "Rice": {"yield_per_acre": 25, "price_per_quintal": 2100},
        "Wheat": {"yield_per_acre": 20, "price_per_quintal": 2000},
        "Maize": {"yield_per_acre": 30, "price_per_quintal": 1800},
        "Sugarcane": {"yield_per_acre": 400, "price_per_quintal": 350},
        "Cotton": {"yield_per_acre": 8, "price_per_quintal": 5800},
    }
    
    mock_response = {
        "agent": "financier_agent",
        "farm_size_acres": farm_size,
        "analyzed_crops": crops,
        "financial_analysis": {},
        "roi_ranking": [],
        "financing_options": [
            {
                "scheme": "Kisan Credit Card",
                "interest_rate": "7% per annum",
                "max_limit": "₹3,00,000",
                "eligibility": "All farmers"
            },
            {
                "scheme": "PM-KISAN",
                "benefit": "₹6,000 per year",
                "type": "Direct benefit transfer"
            }
        ],
        "confidence_score": 0.82
    }
    
    # Calculate financial metrics for each crop
    for crop in crops:
        if crop in crop_costs and crop in crop_yields:
            costs = crop_costs[crop]
            yields = crop_yields[crop]
            
            total_cost_per_acre = sum(costs.values())
            total_cost = total_cost_per_acre * farm_size
            
            revenue_per_acre = yields["yield_per_acre"] * yields["price_per_quintal"]
            total_revenue = revenue_per_acre * farm_size
            
            profit = total_revenue - total_cost
            roi = (profit / total_cost) * 100
            
            mock_response["financial_analysis"][crop] = {
                "cost_breakdown": costs,
                "total_cost_per_acre": total_cost_per_acre,
                "total_cost": total_cost,
                "expected_yield_per_acre": yields["yield_per_acre"],
                "expected_revenue_per_acre": revenue_per_acre,
                "total_revenue": total_revenue,
                "profit": profit,
                "roi_percentage": roi,
                "break_even_yield": total_cost_per_acre / yields["price_per_quintal"]
            }
            
            mock_response["roi_ranking"].append({
                "crop": crop,
                "roi": roi,
                "profit": profit,
                "recommendation": "Excellent" if roi > 50 else "Good" if roi > 25 else "Fair"
            })
    
    # Sort by ROI
    mock_response["roi_ranking"].sort(key=lambda x: x["roi"], reverse=True)
    
    logger.info(f"[FINANCE] Financier Agent completed: Analyzed {len(crops)} crops")
    return mock_response


def call_weather_agent(location: str) -> Dict[str, Any]:
    """
    Weather Agent: Provides weather forecasts and climate risk assessment.
    
    Args:
        location: Geographic location for weather analysis
        
    Returns:
        Dict containing weather patterns and risk assessment
    """
    logger.info(f"[WEATHER] Calling Weather Agent for location: {location}")
    
    mock_response = {
        "agent": "weather_agent",
        "location": location,
        "season": "Kharif 2025",
        "weather_forecast": {
            "monsoon_onset": "June 15, 2025",
            "expected_rainfall": "850-950 mm",
            "rainfall_distribution": "Well distributed",
            "temperature_range": "25-35°C",
            "humidity": "70-85%"
        },
        "climate_risks": {
            "drought_risk": "Low",
            "flood_risk": "Medium",
            "cyclone_risk": "Low",
            "hail_risk": "Low",
            "pest_disease_risk": "Medium"
        },
        "crop_suitability": {
            "Rice": {"suitability": "Excellent", "risk_level": "Low"},
            "Maize": {"suitability": "Good", "risk_level": "Low"},
            "Cotton": {"suitability": "Good", "risk_level": "Medium"},
            "Sugarcane": {"suitability": "Excellent", "risk_level": "Low"},
            "Wheat": {"suitability": "Not suitable", "risk_level": "High", "reason": "Kharif season not suitable for wheat"}
        },
        "recommendations": [
            "Start sowing by June 20 for optimal monsoon utilization",
            "Ensure proper drainage to prevent waterlogging",
            "Monitor weather alerts for extreme events"
        ],
        "confidence_score": 0.88
    }
    
    logger.info(f"[WEATHER] Weather Agent completed: Risk assessment for Kharif season")
    return mock_response


def call_policy_agent(location: str, crop_choice: str) -> Dict[str, Any]:
    """
    Policy Agent: Provides government policies, subsidies, and schemes.
    
    Args:
        location: Geographic location
        crop_choice: Selected crop for policy information
        
    Returns:
        Dict containing applicable government schemes and policies
    """
    logger.info(f"[POLICY] Calling Policy Agent for {crop_choice} in {location}")
    
    mock_response = {
        "agent": "policy_agent",
        "location": location,
        "crop": crop_choice,
        "applicable_schemes": [
            {
                "scheme_name": "PM-KISAN",
                "benefit": "₹6,000 per year in 3 installments",
                "eligibility": "All landholding farmers",
                "application_process": "Online through PM-KISAN portal"
            },
            {
                "scheme_name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
                "benefit": "Crop insurance coverage",
                "premium": "2% for Kharif crops",
                "coverage": "Up to sum insured"
            },
            {
                "scheme_name": "Soil Health Card Scheme",
                "benefit": "Free soil testing",
                "frequency": "Every 3 years",
                "recommendations": "Customized fertilizer recommendations"
            }
        ],
        "msp_info": {
            "crop": crop_choice,
            "msp_2024_25": {
                "Rice": "₹2,300 per quintal",
                "Wheat": "₹2,275 per quintal",
                "Maize": "₹2,090 per quintal",
                "Cotton": "₹7,121 per quintal"
            }.get(crop_choice, "MSP not declared for this crop")
        },
        "subsidies": [
            {
                "type": "Fertilizer Subsidy",
                "benefit": "50% subsidy on fertilizers",
                "application": "Through registered dealers"
            },
            {
                "type": "Seed Subsidy",
                "benefit": "25-50% subsidy on certified seeds",
                "application": "State agriculture department"
            }
        ],
        "compliance_requirements": [
            "Maintain proper land records",
            "Use certified seeds for insurance eligibility",
            "Follow recommended agricultural practices"
        ],
        "confidence_score": 0.90
    }
    
    logger.info(f"[POLICY] Policy Agent completed: Found {len(mock_response['applicable_schemes'])} schemes")
    return mock_response


# ============================================================================
# ORCHESTRATOR NODES
# ============================================================================

def planner_node(state: OrchestratorState) -> OrchestratorState:
    """
    Initial planning node that creates the todo list based on user query.
    """
    user_query = state["user_query"]
    logger.info(f"[PLANNER] Planning phase started for query: {user_query[:100]}...")
    
    # Extract location from query (simple heuristic)
    location = "Kanpur, Uttar Pradesh"  # Default, could be extracted using NLP
    if "kanpur" in user_query.lower():
        location = "Kanpur, Uttar Pradesh"
    elif "punjab" in user_query.lower():
        location = "Punjab"
    elif "delhi" in user_query.lower():
        location = "Delhi"
    elif "mumbai" in user_query.lower():
        location = "Mumbai, Maharashtra"
    
    # Create initial todo list
    initial_todo = [
        f"call_soil_agent:{location}",
        f"call_weather_agent:{location}"
    ]
    
    logger.info(f"[PLANNER] Initial plan created with {len(initial_todo)} tasks")
    
    return {
        **state,
        "todo_list": initial_todo,
        "results_log": {},
        "iteration_count": 0
    }


def executor_node(state: OrchestratorState) -> OrchestratorState:
    """
    Executes the current task from the todo list.
    """
    todo_list = state["todo_list"]
    results_log = state["results_log"]
    iteration_count = state["iteration_count"]
    
    if not todo_list:
        logger.warning("[EXECUTOR] No tasks in todo list")
        return state
    
    # Pop the first task
    current_task = todo_list.pop(0)
    logger.info(f"[EXECUTOR] Executing task {iteration_count + 1}: {current_task}")
    
    # Parse task and execute appropriate agent
    try:
        if current_task.startswith("call_soil_agent:"):
            location = current_task.split(":", 1)[1]
            result = call_soil_agent(location)
            results_log["soil_agent"] = result
            
        elif current_task.startswith("call_weather_agent:"):
            location = current_task.split(":", 1)[1]
            result = call_weather_agent(location)
            results_log["weather_agent"] = result
            
        elif current_task.startswith("call_market_agent:"):
            params = current_task.split(":", 1)[1]
            crops_str, location = params.split("|")
            crops = json.loads(crops_str)
            result = call_market_agent(crops, location)
            results_log["market_agent"] = result
            
        elif current_task.startswith("call_financier_agent:"):
            params = current_task.split(":", 1)[1]
            crops = json.loads(params)
            result = call_financier_agent(crops)
            results_log["financier_agent"] = result
            
        elif current_task.startswith("call_policy_agent:"):
            params = current_task.split(":", 1)[1]
            location, crop = params.split("|")
            result = call_policy_agent(location, crop)
            results_log[f"policy_agent_{crop.lower()}"] = result
            
        else:
            logger.error(f"[EXECUTOR] Unknown task: {current_task}")
            
    except Exception as e:
        logger.error(f"[EXECUTOR] Error executing task {current_task}: {e}")
    
    return {
        **state,
        "todo_list": todo_list,
        "results_log": results_log,
        "current_task": current_task,
        "iteration_count": iteration_count + 1
    }


def replanner_node(state: OrchestratorState) -> OrchestratorState:
    """
    Re-plans and adds new tasks based on current results.
    """
    todo_list = state["todo_list"]
    results_log = state["results_log"]
    current_task = state["current_task"]
    
    logger.info(f"[REPLANNER] Re-planning after task: {current_task}")
    
    # Add new tasks based on completed work
    new_tasks = []
    
    # If soil agent completed, add market analysis for suitable crops
    if ("soil_agent" in results_log and 
        "market_agent" not in results_log and 
        not any("call_market_agent" in task for task in todo_list)):
        suitable_crops = results_log["soil_agent"]["suitable_crops"]
        location = results_log["soil_agent"]["location"]
        crops_json = json.dumps(suitable_crops)
        new_tasks.append(f"call_market_agent:{crops_json}|{location}")
        logger.info(f"[REPLANNER] Added market analysis for {len(suitable_crops)} crops")
    
    # If market agent completed, add financial analysis
    if ("market_agent" in results_log and 
        "financier_agent" not in results_log and 
        not any("call_financier_agent" in task for task in todo_list)):
        analyzed_crops = results_log["market_agent"]["analyzed_crops"]
        crops_json = json.dumps(analyzed_crops)
        new_tasks.append(f"call_financier_agent:{crops_json}")
        logger.info(f"[REPLANNER] Added financial analysis for {len(analyzed_crops)} crops")
    
    # If we have top profitable crops, add policy analysis
    if "financier_agent" in results_log:
        roi_ranking = results_log["financier_agent"]["roi_ranking"]
        location = results_log.get("soil_agent", {}).get("location", "Kanpur, Uttar Pradesh")
        
        # Add policy analysis for top 2 crops (only if not already done)
        for i, crop_data in enumerate(roi_ranking[:2]):
            crop = crop_data["crop"]
            task_key = f"policy_agent_{crop.lower()}"
            policy_task = f"call_policy_agent:{location}|{crop}"
            
            if (task_key not in results_log and 
                not any(policy_task in task for task in todo_list) and
                not any(policy_task in task for task in new_tasks)):
                new_tasks.append(policy_task)
                logger.info(f"[REPLANNER] Added policy analysis for {crop}")
    
    # Add new tasks to the beginning of todo list (priority)
    todo_list = new_tasks + todo_list
    
    logger.info(f"[REPLANNER] Re-planning complete: {len(new_tasks)} new tasks added, {len(todo_list)} total tasks remaining")
    
    return {
        **state,
        "todo_list": todo_list
    }


def should_continue(state: OrchestratorState) -> Literal["continue", "consolidate"]:
    """
    Determines whether to continue processing or move to consolidation.
    """
    todo_list = state["todo_list"]
    iteration_count = state["iteration_count"]
    
    # Safety check to prevent infinite loops
    if iteration_count > 10:  # Reduced limit to prevent infinite loops
        logger.warning("[CONTROLLER] Maximum iterations reached, moving to consolidation")
        return "consolidate"
    
    if todo_list:
        logger.info(f"[CONTROLLER] Continuing: {len(todo_list)} tasks remaining")
        return "continue"
    else:
        logger.info("[CONTROLLER] All tasks completed, moving to consolidation")
        return "consolidate"


def consolidate_results_node(state: OrchestratorState) -> OrchestratorState:
    """
    Consolidates all results into a comprehensive report.
    """
    results_log = state["results_log"]
    user_query = state["user_query"]
    
    logger.info("[CONSOLIDATE] Consolidating results from all agents...")
    
    # Create comprehensive report
    report_sections = []
    report_sections.append("# AGRICULTURAL ADVISORY REPORT")
    report_sections.append("=" * 50)
    report_sections.append(f"Query: {user_query}")
    report_sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_sections.append("")
    
    # Soil Analysis Section
    if "soil_agent" in results_log:
        soil_data = results_log["soil_agent"]
        report_sections.append("## SOIL ANALYSIS")
        report_sections.append("-" * 20)
        report_sections.append(f"Location: {soil_data['location']}")
        report_sections.append(f"Soil Type: {soil_data['soil_health']['soil_type']}")
        report_sections.append(f"pH Level: {soil_data['soil_health']['ph_level']}")
        report_sections.append(f"Suitable Crops: {', '.join(soil_data['suitable_crops'])}")
        report_sections.append("")
    
    # Weather Analysis Section
    if "weather_agent" in results_log:
        weather_data = results_log["weather_agent"]
        report_sections.append("## WEATHER FORECAST")
        report_sections.append("-" * 20)
        report_sections.append(f"Season: {weather_data['season']}")
        report_sections.append(f"Expected Rainfall: {weather_data['weather_forecast']['expected_rainfall']}")
        report_sections.append(f"Temperature Range: {weather_data['weather_forecast']['temperature_range']}")
        report_sections.append("")
    
    # Market Analysis Section
    if "market_agent" in results_log:
        market_data = results_log["market_agent"]
        report_sections.append("## MARKET ANALYSIS")
        report_sections.append("-" * 20)
        for crop, analysis in market_data["market_analysis"].items():
            report_sections.append(f"{crop}:")
            report_sections.append(f"  Current Price: ₹{analysis['current_price_per_quintal']}/quintal")
            report_sections.append(f"  Forecast Price: ₹{analysis['forecast_price_per_quintal']}/quintal")
            report_sections.append(f"  Expected Profitability: {analysis['expected_profitability']}")
        report_sections.append("")
    
    # Financial Analysis Section
    if "financier_agent" in results_log:
        finance_data = results_log["financier_agent"]
        report_sections.append("## FINANCIAL ANALYSIS")
        report_sections.append("-" * 20)
        report_sections.append("ROI Ranking:")
        for i, crop_roi in enumerate(finance_data["roi_ranking"][:3], 1):
            report_sections.append(f"{i}. {crop_roi['crop']}: {crop_roi['roi']:.1f}% ROI (₹{crop_roi['profit']:,.0f} profit)")
        report_sections.append("")
    
    # Policy Information Section
    policy_crops = [key for key in results_log.keys() if key.startswith("policy_agent_")]
    if policy_crops:
        report_sections.append("## GOVERNMENT SCHEMES & POLICIES")
        report_sections.append("-" * 20)
        for policy_key in policy_crops:
            policy_data = results_log[policy_key]
            crop = policy_data["crop"]
            report_sections.append(f"For {crop}:")
            report_sections.append(f"  MSP: {policy_data['msp_info']['msp_2024_25']}")
            report_sections.append(f"  Available Schemes: {len(policy_data['applicable_schemes'])}")
        report_sections.append("")
    
    # Create final report
    final_report = "\n".join(report_sections)
    
    # Save to file
    report_file = Path("final_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(final_report)
    
    logger.info(f"[CONSOLIDATE] Report saved to {report_file}")
    
    return {
        **state,
        "final_report": final_report
    }


def reasoning_llm_node(state: OrchestratorState) -> OrchestratorState:
    """
    Final reasoning using LLM to generate comprehensive recommendation.
    """
    final_report = state["final_report"]
    user_query = state["user_query"]
    
    logger.info("[REASONING] Generating final recommendation using LLM...")
    
    # Initialize LLM
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        logger.warning("[REASONING] No valid OpenAI API key, using fallback reasoning")
        
        # Fallback reasoning without LLM
        results_log = state["results_log"]
        fallback_recommendation = generate_fallback_recommendation(results_log, user_query)
        
        return {
            **state,
            "final_recommendation": fallback_recommendation
        }
    
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
        reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert agricultural advisor with deep knowledge of farming, economics, and policy. 
            Your task is to analyze the comprehensive data provided and generate a single, actionable recommendation 
            that directly answers the user's query.
            
            Consider all aspects: soil suitability, weather conditions, market profitability, financial viability, 
            and government support. Provide a clear, practical recommendation with specific next steps."""),
            
            ("user", """Based on the following comprehensive agricultural analysis, provide a definitive recommendation 
            for the farmer's query.

            ORIGINAL QUERY: {query}

            ANALYSIS DATA:
            {report}

            Please provide:
            1. RECOMMENDED CROP: Which single crop should the farmer choose and why?
            2. JUSTIFICATION: Clear reasoning based on the data
            3. ACTION PLAN: Specific steps the farmer should take
            4. RISK MITIGATION: How to minimize risks
            5. EXPECTED OUTCOMES: What the farmer can expect

            Be specific, practical, and actionable.""")
        ])
        
        chain = reasoning_prompt | llm
        
        response = chain.invoke({
            "query": user_query,
            "report": final_report
        })
        
        final_recommendation = response.content
        logger.info("[REASONING] LLM reasoning completed successfully")
        
    except Exception as e:
        logger.error(f"[REASONING] LLM reasoning failed: {e}")
        # Fallback to rule-based reasoning
        results_log = state["results_log"]
        final_recommendation = generate_fallback_recommendation(results_log, user_query)
    
    # Ensure we return the complete final state
    final_state = {
        **state,
        "final_recommendation": final_recommendation
    }
    
    logger.info("[REASONING] Final reasoning node completed, ready to end")
    return final_state


def generate_fallback_recommendation(results_log: Dict[str, Any], user_query: str) -> str:
    """
    Generate a recommendation without LLM using rule-based logic.
    """
    logger.info("[FALLBACK] Generating fallback recommendation using rule-based logic")
    
    recommendation_parts = []
    recommendation_parts.append("# AGRICULTURAL RECOMMENDATION")
    recommendation_parts.append("=" * 40)
    
    # Find the most profitable crop
    if "financier_agent" in results_log:
        roi_ranking = results_log["financier_agent"]["roi_ranking"]
        if roi_ranking:
            top_crop = roi_ranking[0]
            recommendation_parts.append(f"RECOMMENDED CROP: {top_crop['crop']}")
            recommendation_parts.append(f"Expected ROI: {top_crop['roi']:.1f}%")
            recommendation_parts.append(f"Expected Profit: ₹{top_crop['profit']:,.0f}")
            recommendation_parts.append("")
    
    # Add weather considerations
    if "weather_agent" in results_log:
        weather_data = results_log["weather_agent"]
        recommendation_parts.append("WEATHER CONSIDERATIONS:")
        recommendation_parts.append(f"- {weather_data['weather_forecast']['expected_rainfall']} rainfall expected")
        recommendation_parts.append(f"- Start sowing by June 20 for optimal monsoon utilization")
        recommendation_parts.append("")
    
    # Add policy benefits
    policy_keys = [k for k in results_log.keys() if k.startswith("policy_agent_")]
    if policy_keys:
        policy_data = results_log[policy_keys[0]]
        recommendation_parts.append("GOVERNMENT SUPPORT:")
        recommendation_parts.append("- Apply for PM-KISAN (₹6,000/year)")
        recommendation_parts.append("- Get crop insurance under PMFBY")
        recommendation_parts.append("- Utilize fertilizer subsidies")
        recommendation_parts.append("")
    
    recommendation_parts.append("This recommendation is based on comprehensive analysis of soil, weather, market, and policy factors.")
    
    return "\n".join(recommendation_parts)


# ============================================================================
# ORCHESTRATOR GRAPH CONSTRUCTION
# ============================================================================

def create_orchestrator_graph() -> StateGraph:
    """
    Creates and returns the LangGraph orchestrator.
    """
    logger.info("[GRAPH] Building orchestrator graph...")
    
    # Create the graph
    workflow = StateGraph(OrchestratorState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("consolidate", consolidate_results_node)
    workflow.add_node("reasoning", reasoning_llm_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replanner")
    
    # Conditional edge from replanner
    workflow.add_conditional_edges(
        "replanner",
        should_continue,
        {
            "continue": "executor",
            "consolidate": "consolidate"
        }
    )
    
    workflow.add_edge("consolidate", "reasoning")
    workflow.add_edge("reasoning", END)
    
    logger.info("[GRAPH] Orchestrator graph built successfully")
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_agricultural_advisor(user_query: str) -> Dict[str, Any]:
    """
    Main function to run the agricultural advisory system.
    
    Args:
        user_query: The user's farming question
        
    Returns:
        Dict containing the final recommendation and all results
    """
    logger.info("[MAIN] Starting Agricultural Advisory System")
    logger.info(f"[MAIN] User Query: {user_query}")
    
    # Create orchestrator
    orchestrator = create_orchestrator_graph()
    
    # Initial state
    initial_state = {
        "user_query": user_query,
        "todo_list": [],
        "results_log": {},
        "current_task": None,
        "iteration_count": 0,
        "final_report": None,
        "final_recommendation": None
    }
    
    # Run the orchestrator with increased recursion limit
    try:
        final_state = orchestrator.invoke(
            initial_state, 
            config={"recursion_limit": 50}  # Much higher limit to handle complex workflows
        )
        
        logger.info("[MAIN] Agricultural advisory system completed successfully")
        
        # Print final recommendation
        print("\n" + "="*80)
        print("AGRICULTURAL ADVISORY RECOMMENDATION")
        print("="*80)
        print(final_state["final_recommendation"])
        print("="*80)
        
        return {
            "success": True,
            "recommendation": final_state["final_recommendation"],
            "detailed_report": final_state["final_report"],
            "agent_results": final_state["results_log"],
            "iterations": final_state["iteration_count"]
        }
        
    except Exception as e:
        logger.error(f"[MAIN] Agricultural advisory system failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "recommendation": None
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agricultural Advisory Multi-Agent System")
    parser.add_argument("query", nargs="?", default=None, help="Your farming question")
    parser.add_argument("--example", action="store_true", help="Run with example query")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Use example query if no query provided
    if args.example or not args.query:
        example_query = ("Considering my 5-acre farm in Kanpur, Uttar Pradesh, "
                        "what is the most profitable and suitable crop for me to "
                        "plant for the upcoming Kharif season?")
        user_query = example_query
        print(f"Using example query: {user_query}")
    else:
        user_query = args.query
    
    # Check API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        print("Warning: No valid OpenAI API key found.")
        print("The system will use fallback reasoning without LLM.")
        print("For best results, set OPENAI_API_KEY in your .env file.")
        print()
    
    # Run the system
    result = run_agricultural_advisor(user_query)
    
    if result["success"]:
        print(f"\n📊 System completed in {result['iterations']} iterations")
        print(f"📄 Detailed report saved to: final_report.txt")
    else:
        print(f"\n❌ System failed: {result['error']}")


if __name__ == "__main__":
    main()
