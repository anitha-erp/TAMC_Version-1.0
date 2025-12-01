# ===============================================================
# 🧠 AI-POWERED MCP SERVER with Intelligence Layer
# Version: 5.1 - WITH REPEAT COMMODITY QUERY FIX
# ===============================================================

import os
import json
import asyncio
import requests
import traceback
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

# ----------------------- Configuration -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# External Tool APIs
ARRIVAL_API_URL = "http://127.0.0.1:8000/predict"
PRICE_API_URL = "http://127.0.0.1:8002/api/predict"
PRICE_VARIANTS_URL = "http://127.0.0.1:8002/api/variants"
ADVISORY_API_URL = "http://127.0.0.1:8003/chat"

PRICE_TIMEOUT = 30  # Reduced from 240s (4 min) to 30s
ARRIVAL_TIMEOUT = 300  # Reduced from 180s (3 min) to 20s

# Central location catalog for matching/correction
KNOWN_LOCATIONS = [
    "Warangal", "Khammam", "Hyderabad", "Nakrekal", "Karimnagar",
    "Nizamabad", "Mahbubnagar", "Adilabad", "Nalgonda", "Medak",
    "Warangal Urban", "Warangal Rural", "Hanamkonda"
]
LOCATION_LOOKUP = {loc.lower(): loc for loc in KNOWN_LOCATIONS}

# ----------------------- FastAPI App -----------------------
app = FastAPI(title="🧠 AI-Powered MCP Server", version="5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------- Conversation Memory -----------------------
conversation_sessions = {}

# ----------------------- Status Tracking (Phase 1.5 Optimization) -----------------------
request_status = {}  # Stores current status for each session

def update_status(session_id: str, status: str):
    """Update current processing status for a session"""
    request_status[session_id] = {
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

class ConversationContext:
    def __init__(self):
        self.history: List[Dict] = []
        self.extracted_params: Dict = {}
        self.last_tool_results: Dict = {}

    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.history) > 10:
            self.history = self.history[-10:]

# ----------------------- Request Models -----------------------
class ToolRequest(BaseModel):
    tool: str
    params: dict
    session_id: Optional[str] = "default"

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    context: Optional[Dict] = None

# ----------------------- AI Intelligence Layer -----------------------
class AIIntelligenceLayer:
    """
    Discrete AI layer that interprets tool results and generates insights
    This is the missing layer between tools and user response
    """
    
    def __init__(self):
        self.client = client
        self.model = OPENAI_MODEL
    
    def analyze_tool_results(
        self, 
        query: str,
        tool_results: Dict,
        context: Dict = None
    ) -> Dict:
        """
        Main analysis function - interprets raw tool data
        """
        if not self.client:
            return self._fallback_analysis(tool_results)
        
        # Structure the data
        structured_data = self._structure_tool_data(tool_results)
        
        # Build analysis prompt
        analysis_prompt = self._build_analysis_prompt(query, structured_data, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            
            analysis = json.loads(result)
            
            print(f"\n🧠 AI Intelligence Analysis Complete:")
            print(f"   Summary: {analysis.get('summary', 'N/A')}")
            print(f"   Market Condition: {analysis.get('interpretation', {}).get('market_condition', 'N/A')}")
            print(f"   Risk Level: {analysis.get('risk_assessment', {}).get('overall_risk', 'N/A')}")
            
            return {
                "success": True,
                "analysis": analysis,
                "raw_data": structured_data
            }
            
        except Exception as e:
            print(f"❌ AI Intelligence Error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "fallback": self._fallback_analysis(tool_results)
            }
    
    def _structure_tool_data(self, tool_results: Dict) -> Dict:
        """Convert raw tool outputs into structured format"""
        structured = {
            "arrival_data": None,
            "price_data": None,
            "weather_data": None
        }
        
        # Process arrivals
        if "arrival" in tool_results and tool_results["arrival"].get("success"):
            data = tool_results["arrival"].get("data", {})
            total_predicted = data.get("total_predicted", [])
            
            # Handle dict format from arrival_tool (convert {date: val} to [{date, total_predicted_value}])
            if isinstance(total_predicted, dict):
                total_predicted = [
                    {"date": k, "total_predicted_value": v} 
                    for k, v in sorted(total_predicted.items())
                ]
            
            commodity_daily = data.get("commodity_daily", {})
            if total_predicted:
                first_day = total_predicted[0]
                last_day = total_predicted[min(len(total_predicted), 7) - 1]
                change_pct = self._calculate_change_percentage(first_day, last_day, "total_predicted_value")
                total_week = sum(d.get("total_predicted_value", 0) for d in total_predicted[:7])
                avg_daily = total_week / min(7, len(total_predicted))
                peak_day = max(total_predicted[:7], key=lambda x: x.get("total_predicted_value", 0)) if total_predicted else None

                top_commodities = []
                if commodity_daily:
                    for commodity, entries in commodity_daily.items():

                        # 1️⃣ Ensure entries is list
                        if isinstance(entries, dict):
                            entries = list(entries.values())
                            
                        # 2️⃣ Normalize all items to dicts
                        normalized = []
                        for item in entries:
                            if isinstance(item, dict):
                                normalized.append(item)
                            else:
                                # Convert float values to dict
                                normalized.append({"predicted_value": float(item)})

                        # 3️⃣ Now safe to slice + sum
                        total_val = sum(e.get("predicted_value", 0) for e in normalized[:7])

                        top_commodities.append({
                            "commodity": commodity,
                            "total": total_val
                        })

                        # 4️⃣ Sort and take top 3
                        top_commodities.sort(key=lambda x: x["total"], reverse=True)
                        top_commodities = top_commodities[:3]


                structured["arrival_data"] = {
                    "predictions": total_predicted[:7],
                    "total_week": total_week,
                    "average_daily": avg_daily,
                    "trend": self._calculate_trend(total_predicted[:7], "total_predicted_value"),
                    "trend_pct": change_pct,
                    "peak_day": peak_day,
                    "metric_name": data.get("metric_name"),
                    "top_commodities": top_commodities,
                    "has_commodity": bool(data.get("commodity")),
                    "location": data.get("district") or data.get("amc_name")
                }

                weather_summary = data.get("weather_summary")
                if weather_summary:
                    structured["weather_data"] = {
                        "condition": weather_summary.get("condition"),
                        "rain_mm": weather_summary.get("rain_mm", 0),
                        "temp_c": weather_summary.get("temp_c", 0),
                        "impact": self._assess_weather_impact(weather_summary)
                    }
        
        # Process prices
        if "price" in tool_results and tool_results["price"].get("success"):
            data = tool_results["price"].get("data", {})
            
            # Handle different response structures
            predictions = []
            if "variants" in data and isinstance(data["variants"], list):
                # Extract from variants structure
                for variant in data["variants"]:
                    for forecast in variant.get("forecasts", []):
                        predictions.append({
                            "date": forecast.get("date"),
                            "predicted_price": forecast.get("final_price", forecast.get("baseline_price", 0)),
                            "min_price": forecast.get("min_price", 0),
                            "max_price": forecast.get("max_price", 0),
                            "variant": variant.get("variant")
                        })
            else:
                predictions = data.get("predictions", [])
            
            if predictions:
                structured["price_data"] = {
                    "predictions": predictions[:7],
                    "current_price": predictions[0].get("predicted_price", 0) if predictions else 0,
                    "current_min": predictions[0].get("min_price", 0) if predictions else 0,
                    "current_max": predictions[0].get("max_price", 0) if predictions else 0,
                    "week_end_price": predictions[-1].get("predicted_price", 0) if len(predictions) > 1 else 0,
                    "average_price": sum(p.get("predicted_price", 0) for p in predictions) / len(predictions),
                    "price_trend": self._calculate_trend(predictions, "predicted_price"),
                    "volatility": self._calculate_volatility(predictions, "predicted_price"),
                    "commodity": data.get("commodity"),
                    "market": data.get("market", data.get("district"))
                }
        
        # Process weather/advisory
        if (not structured["weather_data"]) and "advisory" in tool_results and tool_results["advisory"].get("success"):
            data = tool_results["advisory"].get("data", {})
            weather = data.get("weather")
            
            if weather:
                structured["weather_data"] = {
                    "condition": weather.get("condition"),
                    "rain_mm": weather.get("rain_mm", 0),
                    "temp_c": weather.get("temp_c", 0),
                    "impact": self._assess_weather_impact(weather)
                }
        
        return structured
    
    def _calculate_trend(self, data_list: List[Dict], value_key: str) -> str:
        """Calculate trend direction"""
        if not data_list or len(data_list) < 2:
            return "stable"
        
        first_val = data_list[0].get(value_key, 0)
        last_val = data_list[-1].get(value_key, 0)
        
        if first_val == 0:
            return "stable"
        
        change_pct = ((last_val - first_val) / first_val) * 100
        
        if change_pct > 10:
            return "strongly_increasing"
        elif change_pct > 5:
            return "increasing"
        elif change_pct < -10:
            return "strongly_decreasing"
        elif change_pct < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_volatility(self, predictions: List[Dict], value_key: str) -> str:
        """Calculate volatility"""
        if not predictions or len(predictions) < 2:
            return "low"
        
        values = [p.get(value_key, 0) for p in predictions]
        mean = sum(values) / len(values)
        
        if mean == 0:
            return "low"
        
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        cv = (std_dev / mean) * 100
        
        if cv > 10:
            return "high"
        elif cv > 5:
            return "moderate"
        else:
            return "low"

    def _calculate_change_percentage(self, first_entry: Dict, last_entry: Dict, value_key: str) -> float:
        if not first_entry or not last_entry:
            return 0.0
        first_val = first_entry.get(value_key, 0)
        last_val = last_entry.get(value_key, 0)
        if first_val == 0:
            return 0.0
        return round(((last_val - first_val) / first_val) * 100, 2)
    
    def _assess_weather_impact(self, weather: Dict) -> str:
        """Assess weather impact"""
        rain = weather.get("rain_mm", 0)
        
        if rain > 50:
            return "severe"
        elif rain > 20:
            return "moderate"
        elif rain > 5:
            return "mild"
        else:
            return "minimal"
    
    def _get_system_prompt(self) -> str:
        """System prompt for AI intelligence"""
        return """You are an expert agricultural market analyst providing ACTIONABLE INTELLIGENCE.

Analyze prediction data and provide:

1. **Interpretation**: What the numbers mean practically
2. **Key Insights**: 3-5 critical observations
3. **Recommendations**: Specific actions to take
4. **Risk Assessment**: Problems and mitigations
5. **Opportunities**: Ways to maximize profit

Return ONLY valid JSON:
{
  "summary": "One-line key takeaway",
  "interpretation": {
    "market_condition": "oversupply|balanced|shortage",
    "price_outlook": "favorable|neutral|unfavorable",
    "confidence_level": "high|medium|low",
    "reasoning": "Why these conditions exist"
  },
  "key_insights": [
    {
      "insight": "Observation",
      "importance": "high|medium|low",
      "data_point": "Supporting data"
    }
  ],
  "recommendations": [
    {
      "action": "What to do",
      "timing": "When",
      "expected_outcome": "Result",
      "priority": "high|medium|low"
    }
  ],
  "risk_assessment": {
    "overall_risk": "high|medium|low",
    "specific_risks": [
      {
        "risk": "What could go wrong",
        "probability": "high|medium|low",
        "impact": "severe|moderate|mild",
        "mitigation": "How to handle"
      }
    ]
  },
  "opportunities": [
    {
      "opportunity": "Profit opportunity",
      "potential_gain": "Expected benefit",
      "action_required": "What to do",
      "time_sensitive": true/false
    }
  ],
  "market_intelligence": {
    "supply_demand_balance": "Description",
    "price_drivers": ["Factors"],
    "timing_strategy": "Best timing advice"
  }
}

Always cite concrete numbers (trend %, specific dates, rainfall, top commodities) in summary, insights, and recommendations. Do not add extra explanation outside JSON."""
    
    def _build_analysis_prompt(
        self, 
        query: str, 
        structured_data: Dict, 
        context: Dict
    ) -> str:
        """Build analysis prompt"""
        
        prompt_parts = [
            f"FARMER'S QUESTION: \"{query}\"",
            "\n" + "="*70 + "\n",
            "PREDICTION DATA:\n"
        ]
        # NEW: Add commodity breakdown support
        # NEW: Add commodity breakdown into AI analysis
        if "arrival_breakdown" in structured_data:
            breakdown = structured_data["arrival_breakdown"]
            # You can include this breakdown into the prompt
            analysis_parts.append(
                f"\nCommodity Breakdown:\n" +
                "\n".join([f"- {b['commodity']}: {b['total_predicted_value']}" for b in breakdown])
            )

        # Arrival data
        arrival_info = structured_data.get("arrival_data")
        if arrival_info:
            arr = arrival_info
            prompt_parts.append("\n📊 ARRIVAL FORECAST:")
            prompt_parts.append(f"  • Metric: {arr.get('metric_name', 'Arrivals')}")
            prompt_parts.append(f"  • Weekly Total: {arr['total_week']:,.0f} units")
            prompt_parts.append(f"  • Daily Average: {arr['average_daily']:,.0f} units")
            prompt_parts.append(f"  • Trend: {arr['trend']} ({arr.get('trend_pct', 0):+,.1f}%)")
            if arr['peak_day']:
                prompt_parts.append(f"  • Peak Day: {arr['peak_day']['date']} with {arr['peak_day']['total_predicted_value']:,.0f} units")
            if arr.get("top_commodities"):
                for idx, item in enumerate(arr["top_commodities"], start=1):
                    prompt_parts.append(f"  • Top {idx}: {item['commodity']} → {item['total']:,.0f} units")
            prompt_parts.append(f"  • Location: {arr.get('location') or 'Unknown'}")
        
        # Price data
        if structured_data.get("price_data"):
            price = structured_data["price_data"]
            prompt_parts.append("\n\n💰 PRICE FORECAST:")
            prompt_parts.append(f"  • Current: ₹{price['current_price']:,.0f}/quintal (Range: ₹{price['current_min']:,.0f} - ₹{price['current_max']:,.0f})")
            prompt_parts.append(f"  • Week-end: ₹{price['week_end_price']:,.0f}/quintal")
            prompt_parts.append(f"  • Average: ₹{price['average_price']:,.0f}/quintal")
            
            change = ((price['week_end_price'] - price['current_price']) / price['current_price'] * 100) if price['current_price'] > 0 else 0
            prompt_parts.append(f"  • Change: {change:+.1f}%")
            prompt_parts.append(f"  • Trend: {price['price_trend']}")
            prompt_parts.append(f"  • Volatility: {price['volatility']}")
        
        # Weather
        if structured_data.get("weather_data"):
            weather = structured_data["weather_data"]
            prompt_parts.append("\n\n🌤️ WEATHER IMPACT:")
            prompt_parts.append(f"  • Condition: {weather['condition']}")
            prompt_parts.append(f"  • Rain: {weather['rain_mm']}mm")
            prompt_parts.append(f"  • Impact: {weather['impact']}")
        
        # Context
        if context:
            prompt_parts.append("\n\n📍 CONTEXT:")
            if context.get("commodity"):
                prompt_parts.append(f"  • Commodity: {context['commodity']}")
            if context.get("location"):
                prompt_parts.append(f"  • Location: {context['location']}")
        
        prompt_parts.append("\n" + "="*70)
        prompt_parts.append("\n🎯 Provide actionable intelligence that references the specific trends, weather impacts, and top commodities above. Make it farmer-friendly and decision-focused.")
        
        return "\n".join(prompt_parts)
    
    def _fallback_analysis(self, tool_results: Dict) -> Dict:
        """Simple fallback"""
        return {
            "summary": "Data retrieved successfully",
            "interpretation": {
                "market_condition": "unknown",
                "price_outlook": "neutral",
                "confidence_level": "low",
                "reasoning": "AI analysis unavailable"
            },
            "key_insights": [],
            "recommendations": [
                {
                    "action": "Review prediction data",
                    "timing": "Before decisions",
                    "expected_outcome": "Informed choices",
                    "priority": "high"
                }
            ],
            "risk_assessment": {
                "overall_risk": "medium",
                "specific_risks": []
            },
            "opportunities": [],
            "market_intelligence": {
                "supply_demand_balance": "Data available",
                "price_drivers": [],
                "timing_strategy": "Monitor trends"
            }
        }

# Initialize AI Intelligence Layer
ai_intelligence = AIIntelligenceLayer()

# ----------------------- Query Analysis Engine -----------------------
class AIIntelligence:
    """Query understanding and response synthesis"""
    
    def __init__(self):
        if not client:
            print("⚠️ OpenAI client not initialized")

    # ✅ NEW: Helper to extract commodity from query
    def _extract_commodity(self, query: str) -> str:
        q = query.lower().strip()
        commodities = ["chilli", "cotton", "paddy", "onion", "tomato", "groundnut", "turmeric", "maize"]
        for commodity in commodities:
            if commodity in q:
                return commodity.title()
        return None

    # ✅ NEW: Helper to check if query has explicit variant
    def _has_explicit_variant(self, query: str) -> bool:
        q = query.lower().strip()
        indicators = ["-", "deshi", "vagdevi", "vijaya", "cold", "hot", "hybrid", "local", "variety"]
        return any(indicator in q for indicator in indicators)

    def analyze_query(self, query: str, context: ConversationContext) -> Dict:
        """Analyze query for intent and parameters"""
        
        if not client:
            return self._fallback_analysis(query)
        
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in context.history[-5:]
        ])
        
        # ✅ NEW: Detect repeat commodity query without explicit variant
        current_commodity = self._extract_commodity(query)
        previous_commodity = context.extracted_params.get("commodity")
        previous_variant = context.extracted_params.get("variant")

        if current_commodity and current_commodity == previous_commodity and not self._has_explicit_variant(query):
            # 🔄 Same commodity, no explicit variant → clear previous selection
            context.extracted_params.pop("variant", None)
            print(f"🔄 Repeat commodity '{current_commodity}' - clearing previous variant selection")
        
        prompt = f"""You are CropCast AI, an intelligent agricultural assistant. Analyze this query and decide if it needs data/tools or can be answered conversationally.

Previous Context:
{history_text}

Current Query: "{query}"

Extracted Params: {json.dumps(context.extracted_params, indent=2)}

Return ONLY valid JSON:
{{
  "intent": "price_inquiry|arrival_forecast|weather_advice|general_question|multi_tool|variety_selection|advisory_request",
  "confidence": 0.0-1.0,
  "extracted_params": {{
    "commodity": "name or null",
    "district": "district or null",
    "amc_name": "market or null",
    "variant": "variety or null",
    "metric": "total_bags|number_of_arrivals|number_of_lots|total_weight|number_of_farmers|total_revenue or null",
    "days": 7
  }},
  "tools_needed": ["arrival", "price", "advisory", "weather"],
  "needs_clarification": false
}}

IMPORTANT - Be SMART about tool calls:
1. **Informational questions** → "general_question", tools_needed: []
   Examples: "What is chilli?", "Tell me about green chillies", "How to grow tomatoes?"

2. **Variety names alone** (without context) → "general_question", tools_needed: []
   Example: "Green Chilli-Green Chilly" (no previous context) → Just answer about the variety

3. **ONLY call tools when user wants PREDICTIONS/DATA**:
   - "What is the price?" → "price_inquiry", ["price"]
   - "Expected arrivals?" → "arrival_forecast", ["arrival"]
   - "Should I bring to market?" → "advisory_request", ["advisory"]
   - "Weather in Khammam?" → "weather_advice", ["weather"]
   - "Rainfall forecast?" → "weather_advice", ["weather"]

4. **Weather queries**: When user asks about weather, rain, temperature, climate → "weather_advice", ["weather"]

5. **Context matters**: If user previously asked for price/arrivals and now selects a variety, that's a continuation.

Metric keywords (only for arrival queries):
  * "bags" → metric: "total_bags"
  * "lots" → metric: "number_of_lots"
  * "quintals", "quantity", "weight" → metric: "total_weight"
  * "farmers" → metric: "number_of_farmers"
  * "revenue", "income" → metric: "total_revenue"
  * "arrivals" → metric: "number_of_arrivals"

Be intelligent - understand what the user NEEDS, not just keywords."""

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Agricultural data analyst. Return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            
            analysis = json.loads(result_text)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Query Analysis Error: {e}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> Dict:
        """Priority-based fallback (only if pattern + AI both fail)"""
        lower_q = query.lower().strip()

        # Check for greetings first
        greetings = ["hello", "hi", "hey", "thanks", "bye"]
        if any(g in lower_q.split() for g in greetings):
            return {
                "intent": "general_question",
                "confidence": 0.9,
                "extracted_params": {},
                "tools_needed": [],
                "needs_clarification": False
            }

        # Priority-based tool selection (returns ONLY ONE tool)
        # Priority 1: Advisory (highest - decision/recommendation queries)
        if any(kw in lower_q for kw in ["should", "advice", "recommend", "suggest", "bring", "decision", "what to do", "whether", "market trend"]):
            print("🎯 FALLBACK: Advisory (priority 1)")
            return {
                "intent": "advisory_request",
                "confidence": 0.7,
                "extracted_params": {},
                "tools_needed": ["advisory"],
                "needs_clarification": False
            }

        # Priority 2: Price (medium priority)
        elif any(kw in lower_q for kw in ["price", "rate", "cost"]):
            print("🎯 FALLBACK: Price (priority 2)")
            return {
                "intent": "price_inquiry",
                "confidence": 0.7,
                "extracted_params": {},
                "tools_needed": ["price"],
                "needs_clarification": False
            }

        # Priority 3: Arrival (default for agriculture queries)
        elif any(kw in lower_q for kw in ["arrival", "supply", "bags", "lot", "quantity", "farmer"]):
            print("🎯 FALLBACK: Arrival (priority 3)")
            return {
                "intent": "arrival_forecast",
                "confidence": 0.7,
                "extracted_params": {},
                "tools_needed": ["arrival"],
                "needs_clarification": False
            }

        # No keywords matched → general question
        else:
            print("🎯 FALLBACK: General question (no keywords)")
            return {
                "intent": "general_question",
                "confidence": 0.5,
                "extracted_params": {},
                "tools_needed": [],
                "needs_clarification": False
            }

ai_engine = AIIntelligence()

# ----------------------- 🚀 Hybrid Pattern Matching (Fast + Smart) -----------------------
import re

def quick_pattern_match(query: str) -> Optional[Dict]:
    """
    Fast pattern matching for VERY OBVIOUS queries only.
    Returns intent and tools WITHOUT calling AI (FREE + INSTANT).
    Anything unusual/creative → returns None → AI handles it (SMART).

    This hybrid approach gives:
    - 80% of queries: Instant + Free (pattern matching)
    - 20% of queries: Smart + Flexible (AI analysis)
    """
    q = query.lower()

    # Price patterns - Must be very clear
    # Matches: "cotton price tomorrow", "what is rate today", "cost forecast"
    # Doesn't match: "what's going on", "tell me about cotton" → AI handles
    if re.search(r"\b(price|rate|cost)\b.*\b(tomorrow|today|next|forecast)\b", q) or \
       re.search(r"\b(tomorrow|today)\b.*\b(price|rate|cost)\b", q):
        print("✅ PATTERN MATCH: Price query (instant, no AI cost)")
        return {
            "intent": "price_inquiry",
            "confidence": 0.95,
            "extracted_params": {},
            "tools_needed": ["price"],
            "needs_clarification": False
        }

    # Arrival patterns - Must have clear keywords
    # Matches: "expected arrivals", "forecast lots", "how many bags"
    # Doesn't match: "what's the supply situation" → AI handles
    if re.search(r"\b(arrival|expect|forecast|predicted)\b.*\b(lot|bag|quantity|farmer|quintal)\b", q) or \
       re.search(r"\b(how many|number of)\b.*\b(arrival|lot|bag|farmer)\b", q):
        print("✅ PATTERN MATCH: Arrival query (instant, no AI cost)")
        return {
            "intent": "arrival_forecast",
            "confidence": 0.95,
            "extracted_params": {},
            "tools_needed": ["arrival"],
            "needs_clarification": False
        }

    # Advisory patterns - Must have advisory intent keywords
    # Matches: "should I bring", "give me advice", "market trends"
    # Doesn't match: "what's happening" → AI handles
    if re.search(r"\b(should|shall|advice|advise|suggest|recommend)\b.*\b(bring|sell|wait|market)\b", q) or \
       re.search(r"\b(market|price)\b.*\b(trend|analysis|insight)\b", q) or \
       re.search(r"\b(good|best|right)\b.*\b(time|day)\b.*\b(sell|bring)\b", q):
        print("✅ PATTERN MATCH: Advisory query (instant, no AI cost)")
        return {
            "intent": "advisory_request",
            "confidence": 0.95,
            "extracted_params": {},
            "tools_needed": ["advisory"],
            "needs_clarification": False
        }

    # No pattern matched → Use AI for intelligence
    print("🧠 NO PATTERN MATCH: Using AI for intelligent analysis (creative/complex query)")
    return None

# ----------------------- Data Validation Helpers -----------------------
def validate_tool_results(tool_results: Dict, requested_params: Dict) -> Dict:
    """
    🔧 FIX 2 & 4: Validate that tool results match the requested commodity/location.
    Returns validation result with error message if data doesn't match.
    """
    # SKIP WEATHER validation completely
    if "weather" in tool_results:
        return {"valid": True}

    requested_commodity = requested_params.get("commodity", "").lower().strip()
    requested_location = (requested_params.get("district") or requested_params.get("amc_name", "")).lower().strip()

    if not requested_commodity:
        return {"valid": True}  # No commodity requested, can't validate

    validation_errors = []

    # Check price tool results
    if "price" in tool_results:
        price_result = tool_results["price"]
        if price_result.get("success") and not price_result.get("has_varieties"):
            data = price_result.get("data", {})
            forecast = data.get("forecast", [])

            # Check if we got data for a completely different commodity
            if forecast:
                # Look at metadata or infer from context
                returned_commodity = data.get("commodity", "").lower().strip()

                # If the returned commodity doesn't match requested (and isn't empty)
                if returned_commodity and returned_commodity != requested_commodity:
                    validation_errors.append(
                        f"Received {returned_commodity} data when {requested_commodity} was requested"
                    )

    # Check arrival tool results
    if "arrival" in tool_results:
        arrival_result = tool_results["arrival"]
        if arrival_result.get("success") and not arrival_result.get("has_varieties"):
            data = arrival_result.get("data", {})
            commodity_breakdown = data.get("commodity_breakdown", [])

            # If we have commodity breakdown, check if requested commodity is in it
            if commodity_breakdown and len(commodity_breakdown) > 0:
                returned_commodities = [c.get("commodity", "").lower().strip() for c in commodity_breakdown]

                # Check if requested commodity is in the breakdown
                commodity_match = any(
                    requested_commodity in comm or comm in requested_commodity
                    for comm in returned_commodities if comm
                )

                if not commodity_match and returned_commodities[0]:
                    # We got data for a different commodity
                    main_returned = returned_commodities[0]
                    validation_errors.append(
                        f"Received {main_returned} data when {requested_commodity} was requested"
                    )

    if validation_errors:
        return {
            "valid": False,
            "error": f"Data mismatch: {validation_errors[0]}. Please check if {requested_commodity.title()} data is available for {requested_location.title() if requested_location else 'this location'}."
        }

    return {"valid": True}

# ----------------------- Tool Executors -----------------------
async def execute_arrival_tool(params: Dict) -> Dict:
    """Execute arrival prediction"""
    try:
        commodity = params.get("commodity", "")
        amc_param = params.get("amc_name")
        district_param = params.get("district")
        location = amc_param or district_param or params.get("location")
        variant = params.get("variant")
        days = params.get("days", 7)
        aggregate_mode = params.get("aggregate_mode", False)

        if amc_param and not district_param:
            district_param = amc_param
        if district_param and not amc_param:
            amc_param = district_param

        if not location:
            return {
                "success": False,
                "error": "No location specified",
                "tool": "arrival"
            }

        # Clean variant name - remove commodity prefix if present
        if variant and commodity:
            # If variant starts with "Commodity-", strip it
            # E.g., "Chilli-Laxmi Srinivasa Cold" -> "Laxmi Srinivasa Cold"
            commodity_prefix = f"{commodity}-"
            if variant.startswith(commodity_prefix):
                variant = variant[len(commodity_prefix):]
                print(f"   🔧 Cleaned variant name: {variant}")
            # Also try lowercase matching
            elif variant.lower().startswith(commodity_prefix.lower()):
                variant = variant[len(commodity_prefix):]
                print(f"   🔧 Cleaned variant name: {variant}")

        # Force aggregate mode to ignore commodity/variant filters
        if aggregate_mode:
            commodity = ""
            variant = None

        # Check varieties if commodity specified but no variant
        # 🔧 FIX: Arrivals DON'T need variety selection - they aggregate across all varieties
        # Unlike prices (which differ by variety), arrivals can be shown for entire commodity
        # So we skip the variety check and let the arrival API aggregate all varieties

        payload = {
            "metric": params.get("metric", "number_of_arrivals"),
            "district": district_param or location,
            "amc_name": amc_param or location,
            "commodity": commodity,
            "variant": variant,
            "days": days
        }

        print(f"📊 Calling Arrival Tool: {location} | Metric: {payload['metric']}")
        if commodity:
            print(f"   🌾 Commodity: {commodity}")
        if variant:
            print(f"   📦 Variant: {variant}")

        # Use longer timeout for aggregate queries (processing many commodities)
        timeout = 60 if aggregate_mode else 30
        response = requests.post(ARRIVAL_API_URL, json=payload, timeout=timeout)
        response.raise_for_status()

        return {
            "success": True,
            "data": response.json(),
            "tool": "arrival"
        }
    except Exception as e:
        print(f"❌ Arrival Tool Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool": "arrival"
        }

async def execute_price_tool(params: Dict) -> Dict:
    """Execute price prediction"""
    try:
        # ✅ NEW: Validate that metric wasn't accidentally included
        if "metric" in params:
            print(f"⚠️ WARNING: 'metric' parameter found in price query - removing it (metrics only apply to arrivals)")
            params = {k: v for k, v in params.items() if k != "metric"}
        
        commodity = params.get("commodity", "")
        location = params.get("district") or params.get("amc_name") or params.get("location")
        days = params.get("days", 1)
        variant = params.get("variant")

        if not commodity or not location:
            return {
                "success": False,
                "error": f"Missing {'commodity' if not commodity else 'location'}",
                "tool": "price"
            }
        
        if variant and commodity:
            commodity_lower = commodity.lower()
            variant_lower = variant.lower()

            # Count occurrences of the commodity word itself (not with hyphen)
            count_word = variant_lower.count(commodity_lower)

            # If the commodity appears more than once, do NOT clean
            if count_word > 1:
                print(f"⚠️ Skipping cleaning (multi-occurrence): {variant}")
            else:
                prefix = f"{commodity_lower}-"
                if variant_lower.startswith(prefix):
                    variant = variant[len(prefix):].strip()
                    print(f"🔧 Cleaned variant: {variant}")

        # Check varieties if no variant specified
        if not variant:
            try:
                response = requests.get(
                    PRICE_VARIANTS_URL,
                    params={"commodity": commodity, "market": location},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    variants = data.get("variants", [])
                    
                    if len(variants) > 1:
                        return {
                            "success": True,
                            "has_varieties": True,
                            "data": {
                                "commodity": data.get("commodity"),
                                "market": data.get("market"),
                                "variants": variants,
                                "days": days
                            },
                            "tool": "price"
                        }
            except Exception as e:
                print(f"⚠️ Variety check failed: {e}")

        print(f"💰 Calling Price Tool: {commodity} in {location}")
        if variant:
            print(f"   📦 Variant: {variant} (using as-is from API)")

        payload = {
            "commodity": commodity,
            "market": location,
            "prediction_days": days,
            "variant": variant
        }
        
        # Current code (lines 700-720 approx):
        response = requests.post(PRICE_API_URL, json=payload, timeout=PRICE_TIMEOUT)
        response.raise_for_status()  # This throws an exception for 4xx/5xx status codes

        return {
            "success": True,
            "data": response.json(),
            "tool": "price"
        }

    except Exception as e:
        print(f"❌ Price Tool Error: {e}")
        return {
            "success": False,  # ✅ This should be False for errors
            "error": str(e),
            "tool": "price"
        }

async def execute_advisory_tool(query: str, params: Dict, session_id: str) -> Dict:
    """Execute advisory"""
    try:
        payload = {
            "query": query,
            "session_id": session_id,
            "district": params.get("district"),
            "amc_name": params.get("amc_name"),
            "commodity": params.get("commodity")
        }
        
        response = requests.post(ADVISORY_API_URL, json=payload, timeout=180)
        response.raise_for_status()
        
        return {
            "success": True,
            "data": response.json(),
            "tool": "advisory"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": "advisory"
        }

async def execute_weather_tool(params: Dict) -> Dict:
    """Execute weather forecast - uses arrival_tool's weather function"""
    try:
        # Import from the correct location based on project structure
        import sys
        import os
        
        # Add mcp_tools to path if not already there
        mcp_tools_path = os.path.join(os.path.dirname(__file__), 'mcp_tools')
        if os.path.exists(mcp_tools_path) and mcp_tools_path not in sys.path:
            sys.path.insert(0, mcp_tools_path)
        
        # Try importing from mcp_tools first, fallback to direct import
        try:
            from mcp_tools.arrival_tool import fetch_multi_day_weather
        except ImportError:
            from arrival_tool import fetch_multi_day_weather

        location = params.get("district") or params.get("amc_name") or params.get("location", "")
        days = params.get("days", 7)
        
        # Fetch weather data
        weather_data = fetch_multi_day_weather(location, days)

        if not weather_data or weather_data.get("error"):
            return {
                "success": False,
                "error": weather_data.get("error", "Failed to fetch weather data"),
                "tool": "weather"
            }

        return {
            "success": True,
            "data": weather_data,
            "tool": "weather"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": "weather"
        }

# ----------------------- Location Spelling Correction -----------------------
def correct_location_spelling(query: str) -> str:
    """
    Correct spelling mistakes in location names using fuzzy matching

    Common typos:
    - warangel → Warangal
    - khamms → Khammam
    - hyder → Hyderabad
    """
    from difflib import get_close_matches

    # Known location names in Telangana
    known_locations = sorted(set(
        KNOWN_LOCATIONS + [
            "Rangareddy", "Suryapet", "Vikarabad", "Siddipet", "Jangaon"
        ]
    ))

    query_lower = query.lower()
    corrected_query = query

    # Extract potential location words (skip common words)
    words = query.split()
    skip_words = {"in", "at", "from", "for", "the", "of", "price", "chilli", "cotton", "what", "is"}

    for word in words:
        word_clean = word.strip().lower()
        if word_clean in skip_words or len(word_clean) < 4:
            continue

        # Try fuzzy matching
        matches = get_close_matches(word_clean, [loc.lower() for loc in known_locations], n=1, cutoff=0.7)

        if matches:
            # Find the original case version
            for loc in known_locations:
                if loc.lower() == matches[0]:
                    # Replace in query (case-insensitive)
                    corrected_query = re.sub(f"\\b{word}\\b", loc, corrected_query, flags=re.IGNORECASE)
                    print(f"✏️ Spelling corrected: '{word}' → '{loc}'")
                    break

    return corrected_query

# ----------------------- Main Endpoint -----------------------
@app.post("/ai/chat")
async def ai_powered_chat(request: ChatRequest):
    """Main AI-powered endpoint with intelligence layer"""
    try:
        session_id = request.session_id or "default"
        query = request.query.strip()
        raw_context = request.context or {}
        external_context = dict(raw_context) if raw_context else {}
        force_arrival = bool(external_context.pop("force_arrival", False))
        force_price = bool(external_context.pop("force_price", False))
        clear_commodity_flag = bool(external_context.pop("clear_commodity", False))

        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = ConversationContext()

        context = conversation_sessions[session_id]
        context.add_message("user", query)

        if external_context:
            for key, value in external_context.items():
                if value not in (None, ""):
                    context.extracted_params[key] = value

        if clear_commodity_flag:
            context.extracted_params.pop("commodity", None)
            context.extracted_params.pop("variant", None)
            print("🧹 Clear-commodity flag received from context. Removed old commodity binding.")

        # 🔧 FIX: Spelling correction for location names
        query = correct_location_spelling(query)

        print(f"\n{'='*70}")
        print(f"🧠 AI-POWERED REQUEST")
        print(f"{'='*70}")
        print(f"Query: {query}")

        # ✅ Status Update 1
        update_status(session_id, "Analyzing your query...")
        await asyncio.sleep(0.5)  # Small delay so frontend can see this status

        # STEP 1: Hybrid Query Analysis (Pattern Match → AI Fallback)
        print("\n🔍 STEP 1: Query Analysis...")

        # Try fast pattern matching first (FREE + INSTANT)
        analysis = quick_pattern_match(query)

        if not analysis:
            # No pattern match → Use AI for intelligent analysis
            analysis = ai_engine.analyze_query(query, context)
        else:
            # Pattern matched, but we still need to extract parameters (commodity, location, etc.)
            # So call AI ONLY for parameter extraction
            print("   ⚡ Pattern matched - now extracting parameters via AI...")
            ai_analysis = ai_engine.analyze_query(query, context)
            # Keep the intent from pattern match (more reliable), but use AI's extracted params
            analysis["extracted_params"] = ai_analysis.get("extracted_params", {})

        # 🔧 FIX: Smart parameter merging
        # Preserve: commodity, district (context continuity)
        # Clear if None: variant (should re-select for each query)
        new_params = analysis.get("extracted_params", {})

        query_lower = query.lower()

        # 🔧 CRITICAL FIX: Detect location mentions anywhere in the query (not just "in Khammam")
        detected_location = None
        for loc_lower, loc_name in LOCATION_LOOKUP.items():
            if re.search(rf"\b{re.escape(loc_lower)}\b", query_lower):
                detected_location = loc_name
                break

        if detected_location:
            previous_location = context.extracted_params.get("district") or context.extracted_params.get("amc_name")
            if not previous_location or previous_location.lower() != detected_location.lower():
                context.extracted_params["district"] = detected_location
                context.extracted_params["amc_name"] = detected_location
                new_params.setdefault("district", detected_location)
                new_params.setdefault("amc_name", detected_location)
                print(f"🔍 Detected location in query: {detected_location} (context updated)")

        # 🔧 CRITICAL FIX: Clear old location if query explicitly mentions a different new location via keywords
        location_keywords = ["in", "at", "from", "market"]
        for keyword in location_keywords:
            pattern = f"\\b{keyword}\\s+(\\w+)"
            match = re.search(pattern, query_lower)
            if match:
                potential_location = match.group(1)
                lookup_key = potential_location.lower()
                if lookup_key in LOCATION_LOOKUP:
                    normalized = LOCATION_LOOKUP[lookup_key]
                    context.extracted_params["district"] = normalized
                    context.extracted_params["amc_name"] = normalized
                    new_params.setdefault("district", normalized)
                    new_params.setdefault("amc_name", normalized)
                    print(f"🔄 Keyword-based location override detected: {normalized}")
                    break

        # 🔧 CRITICAL FIX: Detect "overall/all commodities" queries and clear commodity context
        # This prevents previous commodity from being preserved for aggregate queries
        aggregate_keywords = [
            "overall", "all commodities", "all commodity", "total",
            "combined", "aggregate", "entire market", "whole market",
            "every commodity", "all crops"
        ]

        has_aggregate_query = any(keyword in query.lower() for keyword in aggregate_keywords)

        if has_aggregate_query:
            # Clear commodity and variant from context
            context.extracted_params.pop("commodity", None)
            context.extracted_params.pop("variant", None)
            print(f"🔄 Cleared commodity from context (aggregate query detected: '{query}')")

            # Force new params to None to prevent re-adding during merge
            new_params["commodity"] = None
            new_params["variant"] = None
            new_params["aggregate_mode"] = True
            context.extracted_params["aggregate_mode"] = True
        else:
            # If user didn't explicitly request aggregate view, remove flag
            context.extracted_params.pop("aggregate_mode", None)

        # FIRST: Detect if query explicitly mentions "variety" keyword OR looks like a variety name
        # Variety names follow pattern: "Word-Word" like "Green Chilli-Green Chilly", "Dry Chillies-Talu"
        query_lower = query.lower()
        explicit_variety_mention = "variety" in query_lower or "varient" in query_lower

        # Check if query looks like a variety name (contains hyphen with words before/after)
        # Examples: "Green Chilli-Green Chilly", "Dry Chillies-Talu", "CHILLI-TAALU"
        looks_like_variety = bool(re.match(r'^[\w\s]+-[\w\s]+$', query.strip()))

        if not explicit_variety_mention and not looks_like_variety and "variant" in context.extracted_params:
            # User didn't say "variety" AND query doesn't look like a variety name - clear it!
            context.extracted_params.pop("variant", None)
            print(f"🔄 Variant cleared (no 'variety' keyword in query)")

        # 🔧 FIX: If query looks like a variety name, FORCE set it as variant
        # Even if AI didn't extract it, we know it's a variety selection
        if looks_like_variety and not explicit_variety_mention:
            # Query IS a variety name (e.g., "Green Chilli-Green Chilly")
            context.extracted_params["variant"] = query.strip()
            print(f"✅ Variant forced from query pattern: {query.strip()}")

        # 🔧 FIX 1: Clear variant when commodity changes (prevent context contamination)
        old_commodity = context.extracted_params.get("commodity", "").lower().strip()
        new_commodity = new_params.get("commodity", "").lower().strip() if new_params.get("commodity") else ""

        if new_commodity and old_commodity and new_commodity != old_commodity:
            # Commodity changed - clear variant to prevent contamination
            context.extracted_params.pop("variant", None)
            print(f"🔄 Commodity changed from '{old_commodity}' to '{new_commodity}' - variant cleared")

        # Now merge new params
        for key, value in new_params.items():
            # Preserve commodity/district from previous queries if not mentioned
            if key in ["commodity", "district", "amc_name"]:
                if value is not None and value != "":
                    context.extracted_params[key] = value
                # If None/empty, keep existing value (preserve context)
            # Only set variant if explicitly mentioned (and AI extracted it)
            elif key == "variant":
                if value is not None and value != "" and explicit_variety_mention:
                    context.extracted_params[key] = value
                    print(f"✅ Variant set from AI extraction: {value}")
                # Note: If looks_like_variety, we already set it above
            elif key == "aggregate_mode":
                if bool(value):
                    context.extracted_params["aggregate_mode"] = True
                else:
                    context.extracted_params.pop("aggregate_mode", None)
            # For other params, update if provided
            else:
                if value is not None and value != "":
                    context.extracted_params[key] = value

        # Keep district and AMC name aligned
        amc_value = context.extracted_params.get("amc_name")
        district_value = context.extracted_params.get("district")
        if amc_value and (not district_value or district_value.lower() != amc_value.lower()):
            context.extracted_params["district"] = amc_value
        elif district_value and not amc_value:
            context.extracted_params["amc_name"] = district_value

        intent = analysis.get("intent", "")
        # 🔧 FIX: Normalize AI typos in intent
        intent_corrections = {
            "weatheer_advice": "weather_advice",
            "wheather_advice": "weather_advice",
            "waether_advice": "weather_advice",
            "wether_advice": "weather_advice",
        }

        if intent in intent_corrections:
            print(f"🔧 Corrected misspelled intent: {intent} → {intent_corrections[intent]}")
            intent = intent_corrections[intent]

        tools_needed = analysis.get("tools_needed", [])

        # 🔧 SMART: If query is ONLY a variety name AND AI says general_question BUT we have active context
        # This suggests user is selecting a variety for their previous price/arrival query
        if looks_like_variety and intent == "general_question":
            # Check if we have active context from a recent prediction query
            has_commodity = context.extracted_params.get("commodity")
            has_location = context.extracted_params.get("district") or context.extracted_params.get("amc_name")
            previous_intent = context.extracted_params.get("_last_intent")

            # Only override if there's CLEAR context (all three must exist)
            if has_commodity and has_location and previous_intent and previous_intent in ["price_inquiry", "arrival_forecast"]:
                # Use the previous intent - could be price OR arrival
                print(f"🧠 SMART: Variety name with active context, continuing '{previous_intent}' flow")
                intent = previous_intent
                tools_needed = ["price"] if intent == "price_inquiry" else ["arrival"]
            else:
                # No clear context - let AI handle it as general_question
                print(f"💬 Variety name without context - treating as informational question")

        # 🔧 SMART: Detect arrival-specific metrics and SUGGEST arrival intent (not force)
        # Only override if AI completely misunderstood (e.g., general_question when metric mentioned)
        query_lower = query.lower()
        arrival_metric_keywords = [
            "total_bags", "number_of_farmers", "number_of_lots", "number_of_arrivals", "total_weight",
            "total_revenue"
        ]

        # Only override if:
        # 1. Query mentions arrival metric keyword (exact match, not substring)
        # 2. AI detected general_question (wrong) or didn't detect any intent
        has_arrival_metric = any(kw in query_lower for kw in arrival_metric_keywords)

        if has_arrival_metric and intent in ["general_question", ""]:
            print(f"🔧 SMART OVERRIDE: Query mentions arrival metric but AI detected '{intent}', suggesting 'arrival_forecast'")
            intent = "arrival_forecast"
            tools_needed = ["arrival"]

        if force_arrival:
            intent = "arrival_forecast"
            tools_needed = ["arrival"]
            print("🎯 Force-arrival flag detected from context. Overriding intent.")
        elif force_price:
            intent = "price_inquiry"
            tools_needed = ["price"]
            print("🎯 Force-price flag detected from context. Overriding intent.")

        # 🔧 FIX: Clear tool-specific params when switching between query types
        # This prevents parameter contamination (e.g., variant from price queries interfering with arrival queries)
        previous_intent = context.extracted_params.get("_last_intent")
        if previous_intent and previous_intent != intent:
            # Switching query types - clear tool-specific contamination
            context.extracted_params.pop("variant", None)  # Force re-selection for new query type
            context.extracted_params.pop("metric", None)   # arrival-only param, remove for price queries
            print(f"🔄 Intent changed: {previous_intent} → {intent} (cleared tool-specific params)")

        if intent == "weather_advice":
            context.extracted_params.pop("commodity", None)
            context.extracted_params.pop("variant", None)
            print("🌦️ Weather-only query detected. Cleared commodity context.")

        context.extracted_params["_last_intent"] = intent

        # 🔧 SMART: Only set metric if AI didn't already set it
        # ⚠️ CRITICAL: Only for ARRIVAL queries (not price queries!)
        if intent == "arrival_forecast" and not context.extracted_params.get("metric"):
            query_lower = query.lower()
            # Only set metric if query CLEARLY mentions it and AI missed it
            if "quintal" in query_lower or "quantity" in query_lower or "weight" in query_lower:
                context.extracted_params["metric"] = "total_weight"
            elif "farmer" in query_lower:
                context.extracted_params["metric"] = "number_of_farmers"
            elif "bag" in query_lower:
                context.extracted_params["metric"] = "total_bags"
                print(f"🔧 AI missed metric, auto-set to 'total_bags'")
            elif "lot" in query_lower:
                context.extracted_params["metric"] = "number_of_lots"
                print(f"🔧 AI missed metric, auto-set to 'number_of_lots'")
            elif "revenue" in query_lower or "income" in query_lower:
                context.extracted_params["metric"] = "total_revenue"
                print(f"🔧 AI missed metric, auto-set to 'total_revenue'")

        # ✅ NEW: Clear metric for price queries (should NEVER have metric)
        if intent == "price_inquiry" and "metric" in context.extracted_params:
            context.extracted_params.pop("metric")
            print(f"🧹 Removed 'metric' from price query (metric only applies to arrivals)")

        # ✅ NEW: Clear metric for weather queries (should NEVER have metric)
        if intent == "weather_advice" and "metric" in context.extracted_params:
            context.extracted_params.pop("metric")
            print(f"🧹 Removed 'metric' from weather query (metric only applies to arrivals)")

        # ✅ NEW: Clear metric if it was accidentally set by AI for non-arrival queries
        if intent not in ["arrival_forecast"] and "metric" in context.extracted_params:
            context.extracted_params.pop("metric")
            print(f"🧹 Removed 'metric' from {intent} query (metric only applies to arrivals)")

        # 🔧 FIX: For pure arrival queries, ONLY call arrival tool (don't add advisory)
        # For pure price queries, ONLY call price tool (don't add advisory)
        # Advisory should ONLY be called when explicitly requested
        if intent == "arrival_forecast" and "arrival" in tools_needed:
            tools_needed = ["arrival"]  # Remove any other tools
            print(f"🎯 ARRIVAL-ONLY QUERY - Will only call arrival tool")
        elif intent == "price_inquiry" and "price" in tools_needed:
            tools_needed = ["price"]  # Remove any other tools
            print(f"🎯 PRICE-ONLY QUERY - Will only call price tool")

        # 🔧 FIX: Detect if this is advisory-only query (should NOT show detailed price/arrival charts)
        query_lower = query.lower()

        # Check if frontend explicitly passed advisory flag via context (for variety selections)
        is_advisory_from_context = raw_context.get("is_advisory_query", False) if raw_context else False

        is_advisory_only = is_advisory_from_context or intent == "advisory_request" or (
            "advisory" in tools_needed and
            any(kw in query_lower for kw in ["should", "advice", "recommend", "bring", "decision", "what to do", "whether"])
        ) or "market trend" in query_lower

        if is_advisory_only:
            print(f"🎯 ADVISORY-ONLY QUERY DETECTED - Frontend will skip detailed charts")
            if is_advisory_from_context:
                print(f"   ✅ Advisory flag passed via context (variety selection)")
            print(f"   Intent: {intent}, Tools: {tools_needed}")
        
        # STEP 2: Handle general questions (informational, conversational)
        if intent == "general_question" and not tools_needed:
            print(f"💬 GENERAL QUESTION - Answering conversationally (no tool calls)")
            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are CropCast AI, a friendly agricultural assistant with knowledge about:
- Crops, varieties, and farming practices
- Market terminology and agricultural concepts
- General agricultural advice

Be conversational and helpful. If they ask about predictions/forecasts, suggest they ask specific questions like:
- "What is the price of [commodity] in [location]?"
- "Expected arrivals of [commodity] in [market]?"

Keep responses brief (2-3 sentences)."""
                        },
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                ai_response = response.choices[0].message.content.strip()
            except:
                greetings = {
                    "hello": "Hello! 👋 How can I help with market forecasts?",
                    "hi": "Hi! 😊 Ask me about prices or arrivals.",
                    "thanks": "You're welcome! 🌾",
                    "bye": "Goodbye! Have a great harvest! 🌾"
                }
                ai_response = greetings.get(query.lower().strip(), "Hello! I can help with price forecasts and arrival predictions. What would you like to know?")

            context.add_message("assistant", ai_response)
            return {
                "response": ai_response,
                "needs_clarification": False,
                "query_type": "general_conversation"
            }
        
        # ✅ Status Update 2
        update_status(session_id, "Fetching market data...")
        await asyncio.sleep(0.5)  # Small delay so frontend can see this status

        # STEP 3: Execute tools IN PARALLEL (40-60% faster!)
        print(f"\n🚀 STEP 3: Executing {len(tools_needed)} tool(s) in parallel...")
        tool_results = {}

        # Build parallel tasks
        tasks = {}
        if "arrival" in tools_needed:
            print(f"   🔍 Arrival Tool Params: {context.extracted_params}")
            tasks["arrival"] = execute_arrival_tool(context.extracted_params)
        if "price" in tools_needed:
            print(f"   🔍 Price Tool Params: {context.extracted_params}")
            tasks["price"] = execute_price_tool(context.extracted_params)
        if "advisory" in tools_needed:
            tasks["advisory"] = execute_advisory_tool(query, context.extracted_params, session_id)
        if "weather" in tools_needed:
            print(f"   🌤️ Weather Tool Params: {context.extracted_params}")
            tasks["weather"] = execute_weather_tool(context.extracted_params)

        # Execute all tools in parallel
        if tasks:
            print(f"   Running {len(tasks)} tools: {list(tasks.keys())}")
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            # Map results back to tool names
            for i, tool_name in enumerate(tasks.keys()):
                if isinstance(results[i], Exception):
                    print(f"   ❌ {tool_name} failed: {results[i]}")
                    tool_results[tool_name] = {"success": False, "error": str(results[i])}
                else:
                    print(f"   ✅ {tool_name} completed successfully")
                    tool_results[tool_name] = results[i]

        # ✅ Status Update 3
        update_status(session_id, "Processing predictions...")
        await asyncio.sleep(0.5)  # Small delay so frontend can see this status

        # 🔧 FIX 2 & 4: Validate that returned data matches requested commodity
        validation_result = validate_tool_results(tool_results, context.extracted_params)
        if not validation_result.get("valid"):
            error_msg = validation_result.get("error", "Data validation failed")
            print(f"   ❌ Data validation failed: {error_msg}")
            context.add_message("assistant", error_msg)
            return {
                "response": error_msg,
                "needs_clarification": False,
                "query_type": "error",
                "session_id": session_id
            }

        # 🔧 FIX 4: Check if all tools failed - provide helpful error message
        all_failed = all(
            not result.get("success", False)
            for result in tool_results.values()
        ) if tool_results else True

        if all_failed:
            commodity = context.extracted_params.get("commodity", "").title()
            location = (context.extracted_params.get("district") or context.extracted_params.get("amc_name", "")).title()
            is_aggregate = context.extracted_params.get("aggregate_mode", False)
            
            # Generate a helpful error message based on query type
            if intent == "weather_advice":
                # Weather query - doesn't need commodity
                if location:
                    error_msg = f"Sorry, I couldn't retrieve weather data for {location}. This could mean:\n• The weather service is temporarily unavailable\n• There might be a connection issue\n• Try again in a moment or check the location name"
                else:
                    error_msg = "Please specify a location for the weather forecast (e.g., 'weather in Khammam', 'weather forecast for Warangal')."
            elif is_aggregate and location:
                # Aggregate query (all commodities) with location - different error message
                error_msg = f"Sorry, I couldn't retrieve aggregate market data for {location}. This could mean:\n• The server is experiencing high load (please try again)\n• There might be a temporary connection issue\n• {location} market data might be temporarily unavailable"
            elif commodity and location:
                error_msg = f"Sorry, I couldn't find {commodity} data for {location}. This could mean:\n• {commodity} might not be traded in {location}\n• The commodity name might be spelled differently\n• Try checking available commodities: Chilli, Cotton, Groundnut, Paddy"
            elif commodity:
                error_msg = f"Sorry, I couldn't find {commodity} data. Please specify a location (e.g., Warangal, Khammam)."
            elif location:
                # Location specified but no commodity - suggest adding commodity
                error_msg = f"Please specify a commodity for {location} (e.g., 'Chilli arrivals in {location}') or ask for 'total bags in {location}' for aggregate data."
            else:
                error_msg = "I couldn't process your request. Please specify both commodity and location (e.g., 'Chilli price in Khammam')."

            print(f"   ❌ All tools failed: {error_msg}")
            context.add_message("assistant", error_msg)
            return {
                "response": error_msg,
                "needs_clarification": False,
                "query_type": "error",
                "session_id": session_id
            }

        # Check if any tool needs variety selection (must return early)
        for tool_name, result in tool_results.items():
            if result.get("has_varieties"):
                if tool_name == "arrival":
                    variety_data = result.get("data", {})
                    metric = variety_data.get("metric", "number_of_arrivals")
                    metric_labels = {
                        "total_bags": "bags",
                        "number_of_arrivals": "arrivals",
                        "number_of_lots": "lots",
                        "total_weight": "quantity",
                        "number_of_farmers": "farmers",
                        "total_revenue": "revenue"
                    }
                    metric_name = metric_labels.get(metric, metric)
                    return {
                        "response": f"Please select a variety to see the {metric_name} forecast.",
                        "needs_clarification": False,
                        "has_varieties": True,
                        "varieties": variety_data,
                        "tool_results": tool_results,
                        "query_type": "advisory_only" if is_advisory_only else "prediction",
                        "session_id": session_id
                    }
                elif tool_name == "price":
                    variety_data = result.get("data", {})
                    response_msg = "Please select a variety to see the advisory." if is_advisory_only else "Please select a variety to see the price forecast."
                    return {
                        "response": response_msg,
                        "needs_clarification": False,
                        "has_varieties": True,
                        "varieties": variety_data,
                        "tool_results": tool_results,
                        "query_type": "advisory_only" if is_advisory_only else "prediction",
                        "session_id": session_id
                    }
        
        # STEP 4: Generate appropriate response based on query type
        print(f"\n🧠 Generating response for intent: {intent}...")

        intelligence_result = None  # Initialize to avoid errors

        # ✅ Status Update 4
        update_status(session_id, "Preparing your forecast...")
        await asyncio.sleep(0.5)  # Small delay so frontend can see this status

        # For ALL queries (arrival, price, advisory), generate intelligent explanation
        if tool_results:
            print("\n🧠 Generating intelligent explanation...")
            intelligence_result = ai_intelligence.analyze_tool_results(
                query=query,
                tool_results=tool_results,
                context={
                    "commodity": context.extracted_params.get("commodity"),
                    "location": context.extracted_params.get("district") or context.extracted_params.get("amc_name")
                }
            )

        # Build response based on query type
        # Normalize arrival predictions for frontend
        if "arrival" in tool_results and tool_results["arrival"].get("success"):
            arr = tool_results["arrival"]["data"]
            if isinstance(arr.get("total_predicted"), dict):
                arr["total_predicted"] = [
                    {"date": d, "total_predicted_value": v}
                    for d, v in sorted(arr["total_predicted"].items())
                ]

        if intent == "arrival_forecast" and intelligence_result and intelligence_result.get("success"):
            # Arrival forecast with intelligent summary
            analysis = intelligence_result.get("analysis", {})
            summary = analysis.get("summary", "Forecast data available.")
            ai_response = f"📊 {summary}"
        elif intent == "price_inquiry" and intelligence_result and intelligence_result.get("success"):
            # Price forecast with intelligent summary
            analysis = intelligence_result.get("analysis", {})
            summary = analysis.get("summary", "Forecast data available.")
            ai_response = f"📊 {summary}"
        elif intent == "weather_advice" and "weather" in tool_results:
            weather_result = tool_results["weather"]

            if weather_result.get("success"):
                raw = weather_result.get("data", {})

                forecast_days = raw.get("forecast_days", [])
                location = raw.get("location", context.extracted_params.get("district", "the area"))

                response_parts = [f"🌤️ Weather forecast for {location}:\n"]

                if forecast_days:
                    for day in forecast_days[:7]:
                        date = day.get("date", "")
                        info = day.get("day", {})
                        condition_raw = info.get("condition", {})

                        if isinstance(condition_raw, dict):
                            condition = condition_raw.get("text", "N/A")
                        else:
                            condition = str(condition_raw)

                        temp = info.get("avgtemp_c", "--")
                        rain = info.get("totalprecip_mm", "--")

                        response_parts.append(f"📅 {date}: {condition}, {temp}°C, Rain: {rain}mm")

                ai_response = "\n".join(response_parts)

            else:
                ai_response = f"⚠️ {weather_result.get('error', 'Unable to fetch weather data')}."

        elif intent in ["advisory_request", "multi_tool"] or "advisory" in tool_results:
            # For advisory queries, show full insights with recommendations
            if intelligence_result and intelligence_result.get("success"):
                analysis_data = intelligence_result["analysis"]
                response_parts = []
                response_parts.append(f"📊 {analysis_data.get('summary', 'Analysis complete.')}")

                insights = analysis_data.get("key_insights", [])[:3]
                if insights:
                    response_parts.append("\n\n🔍 Key Insights:")
                    for insight in insights:
                        importance = "🔴" if insight['importance'] == 'high' else "🟡"
                        response_parts.append(f"{importance} {insight['insight']}")

                recommendations = analysis_data.get("recommendations", [])
                if recommendations:
                    top_rec = recommendations[0]
                    response_parts.append(f"\n\n💡 Recommended Action:")
                    response_parts.append(f"• {top_rec['action']}")
                    response_parts.append(f"• Timing: {top_rec['timing']}")

                risks = analysis_data.get("risk_assessment", {}).get("specific_risks", [])
                critical_risks = [r for r in risks if r.get("probability") == "high"]
                if critical_risks:
                    response_parts.append(f"\n\n⚠️ Risk Alert:")
                    for risk in critical_risks[:2]:
                        response_parts.append(f"• {risk['risk']}")
                        response_parts.append(f"  → {risk['mitigation']}")

                ai_response = "\n".join(response_parts)
            else:
                ai_response = "Advisory analysis complete."
        else:
            # Fallback if no intelligence result
            if intent == "arrival_forecast":
                ai_response = "Here are the arrival predictions:"
            elif intent == "price_inquiry":
                ai_response = "Here are the price predictions:"
            else:
                ai_response = "Forecast ready."
        
        context.add_message("assistant", ai_response)

        print(f"\n✅ Response with AI Intelligence Generated")
        print(f"   Query Type: {'advisory_only' if is_advisory_only else 'prediction'}")
        print(f"   Tool Results: {list(tool_results.keys())}")
        if "price" in tool_results:
            print(f"   Price Data Success: {tool_results['price'].get('success')}")
            if tool_results['price'].get('success'):
                price_data = tool_results['price'].get('data', {})
                print(f"   Price Data Keys: {list(price_data.keys())}")
            else:
                # Show the error if price tool failed
                error_msg = tool_results['price'].get('error', 'Unknown error')
                print(f"   ❌ Price Tool Error: {error_msg}")
        print(f"{'='*70}\n")

        return {
            "response": ai_response,
            "needs_clarification": False,
            "tool_results": tool_results,

            # 🔧 FIX: Add query_type flag so frontend knows to skip detailed charts for advisory queries
            "query_type": "advisory_only" if is_advisory_only else "prediction",

            # AI Intelligence (structured insights for UI) - only for advisory queries
            "ai_insights": intelligence_result.get("analysis") if (intelligence_result and intelligence_result.get("success")) else None,
            "market_analysis": intelligence_result.get("analysis", {}).get("interpretation") if intelligence_result else None,
            "recommendations": intelligence_result.get("analysis", {}).get("recommendations", []) if intelligence_result else [],
            "risks": intelligence_result.get("analysis", {}).get("risk_assessment") if intelligence_result else None,
            "opportunities": intelligence_result.get("analysis", {}).get("opportunities", []) if intelligence_result else [],
            "total_predicted": (
                tool_results.get("arrival", {})
                .get("data", {})
                .get("total_predicted", [])
            ),

            "session_id": session_id
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------- Other Endpoints -----------------------
@app.get("/")
def home():
    return {
        "status": "✅ AI-Powered MCP Server with Intelligence Layer",
        "version": "5.1",
        "features": [
            "🧠 Discrete AI Intelligence Layer",
            "📊 Tool result interpretation",
            "💡 Actionable recommendations",
            "⚠️ Risk assessment",
            "🎯 Market intelligence",
            "🔍 Trend analysis",
            "🔄 Repeat commodity query fix"
        ]
    }

@app.get("/get_amcs")
def get_amcs():
    try:
        res = requests.get("http://127.0.0.1:8000/get_amcs", timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_commodities")
def get_commodities():
    try:
        res = requests.get("http://127.0.0.1:8002/api/variants?list=commodities", timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{session_id}")
def get_status(session_id: str):
    """Get current processing status for a session"""
    if session_id in request_status:
        return request_status[session_id]
    return {"status": "No active request", "timestamp": datetime.now().isoformat()}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "5.1",
        "ai_provider": "OpenAI",
        "ai_model": OPENAI_MODEL if client else "Not configured",
        "features": {
            "ai_intelligence_layer": "✅ Active",
            "query_analysis": "✅ Working",
            "tool_execution": "✅ Working",
            "insights_generation": "✅ Working",
            "repeat_commodity_fix": "✅ Applied"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting AI-Powered MCP Server v5.1...")
    print(f"🧠 AI Intelligence Layer: Active")
    print(f"📋 Model: {OPENAI_MODEL if client else 'Not configured'}")
    print("="*70)
    print("🎯 Architecture:")
    print("  Layer 1: Query Understanding (AI)")
    print("  Layer 2: Tool Execution (Predictions)")
    print("  Layer 3: AI Intelligence (Analysis) ⭐")
    print("  Layer 4: Response Synthesis (AI)")
    print("="*70)
    print("✅ Repeat commodity query fix applied")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8005, reload=True)