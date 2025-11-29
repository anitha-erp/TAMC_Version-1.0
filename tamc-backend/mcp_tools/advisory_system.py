#!/usr/bin/env python3
"""
advisory_system.py

Pure FastAPI implementation of the Smart Conversational Agricultural Advisory Bot.
- No CLI mode (API only)
- Session-based conversation contexts
- Replaces test_code -> arrival_tool and price_pred -> price_tool
- Async-safe and defensive about sync/async model functions
"""

from typing import Optional, Dict, Any, List
import asyncio
import logging
import os
import json
import traceback
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalizer import clean_amc, clean_district, clean_commodity
from weather_helper import WeatherHelper

# OpenAI client (keeps same import style as original)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # we'll handle missing client gracefully

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("advisory_system")

# -------------------- Environment & Keys --------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_IN_API_KEY", "")

if OpenAI and OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)
    logger.info("✅ OpenAI client configured")
else:
    client = None
    logger.warning("⚠️ OpenAI client not configured or OpenAI package missing")

if WEATHER_API_KEY:
    masked = f"{WEATHER_API_KEY[:4]}...{WEATHER_API_KEY[-4:]}" if len(WEATHER_API_KEY) > 8 else "****"
    logger.info(f"✅ Weather API key found: {masked}")
else:
    logger.warning("⚠️ WEATHER_API_KEY missing")

# -------------------- External tool modules --------------------
# Replace previous modules with arrival_tool and price_tool
arrival_tool = None
price_tool = None

try:
    import arrival_tool as arrival_tool_module  # user-supplied module
    arrival_tool = arrival_tool_module
    logger.info("✅ arrival_tool imported")
except Exception as e:
    logger.warning(f"⚠️ arrival_tool not imported: {e}")

try:
    import price_tool as price_tool_module  # user-supplied module
    price_tool = price_tool_module
    logger.info("✅ price_tool imported")
except Exception as e:
    logger.warning(f"⚠️ price_tool not imported: {e}")


# ==================== Conversation Context ====================
class ConversationContext:
    def __init__(self):
        self.district: Optional[str] = None
        self.amc_name: Optional[str] = None
        self.commodity: Optional[str] = None
        self.state: str = "Telangana"
        self.conversation_history: List[Dict[str, str]] = []

    def update_from_query(self, extracted_params: Dict[str, Any]):
        if extracted_params.get("district"):
            self.district = extracted_params["district"]
        if extracted_params.get("amc_name"):
            self.amc_name = extracted_params["amc_name"]
        if extracted_params.get("commodity"):
            self.commodity = extracted_params["commodity"]
        if extracted_params.get("state"):
            self.state = extracted_params["state"]

    def get_context_summary(self) -> str:
        parts = []
        if self.district:
            parts.append(f"District: {self.district}")
        if self.amc_name:
            parts.append(f"Market: {self.amc_name}")
        if self.commodity:
            parts.append(f"Crop: {self.commodity}")
        return ", ".join(parts) if parts else "No specific location/crop set"

    def add_to_history(self, query: str, response: str):
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_summary": (response[:200] if response else "")
        })
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]


# ==================== LLM Helpers (parameter extraction + classification + advisory) ====================
# These functions are defensive: if OpenAI client is missing, fall back to heuristic behavior.

async def _call_openai_chat(model: str, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.0):
    """Wrapper to call OpenAI chat completion. Runs in threadpool if sync-only client."""
    if client is None:
        raise RuntimeError("OpenAI client not configured")
    # Some OpenAI clients are sync. Use asyncio.to_thread to avoid blocking.
    def _sync_call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature
        )
    return await asyncio.to_thread(_sync_call)


async def extract_parameters_from_query(query: str, context: ConversationContext) -> Dict[str, Any]:
    """Use LLM to extract parameters; fallback to simple regex heuristics if LLM not available."""
    if client is None:
        # Simple heuristic extraction
        district = None
        amc_name = None
        commodity = None
        days = 7
        # crude regex: 'in <word>' or 'at <word>' or 'for <commodity>'
        m = None
        import re
        m = re.search(r"\bin\s+([A-Z][a-zA-Z]+)", query)
        if m:
            district = m.group(1)
        m2 = re.search(r"\b(mandi|market|amc)\s+([A-Za-z0-9\s]+)", query.lower())
        if m2:
            amc_name = m2.group(2).strip().title()
        m3 = re.search(r"\b(chilli|tomato|paddy|rice|maize|cotton|onion|potato|chili|chilli)\b", query.lower())
        if m3:
            commodity = m3.group(1)
        m4 = re.search(r"(\d+)\s*days?", query.lower())
        if m4:
            days = int(m4.group(1))
        return {
            "district": district or context.district,
            "amc_name": amc_name or context.amc_name,
            "commodity": commodity or context.commodity,
            "state": context.state,
            "days": days,
            "needs_clarification": False,
            "missing_params": []
        }

    # Build extraction prompt
    extraction_prompt = f"""
Extract agricultural parameters from this farmer's question. Use context if needed.

Current Context:
- District: {context.district or "Not set"}
- Market: {context.amc_name or "Not set"}
- Commodity: {context.commodity or "Not set"}
- State: {context.state or "Telangana"}

Farmer's Question: "{query}"

Return ONLY a JSON object with these fields (use null if not mentioned):
{{
  "district": "district name or null",
  "amc_name": "market/AMC name or null",
  "commodity": "crop/commodity name or null",
  "state": "state name or null",
  "days": 7,
  "needs_clarification": false,
  "missing_params": []
}}
"""
    try:
        resp = await _call_openai_chat(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract agricultural parameters from queries. Return only valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ],
            max_tokens=250,
            temperature=0.0
        )
        result_text = resp.choices[0].message.content.strip()
        # strip code fences if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        extracted = json.loads(result_text)
        final_params = {
            "district": clean_district(extracted.get("district") or context.district),
            "amc_name": clean_amc(extracted.get("amc_name") or context.amc_name),
            "commodity": clean_commodity(extracted.get("commodity") or context.commodity),
            "state": extracted.get("state") or context.state,
            "days": extracted.get("days", 7),
            "needs_clarification": extracted.get("needs_clarification", False),
            "missing_params": extracted.get("missing_params", [])
        }

        return final_params
    except Exception as e:
        logger.error(f"Parameter extraction error: {e}")
        logger.debug(traceback.format_exc())
        # fallback heuristic
        return await extract_parameters_from_query(query, context)  # calls heuristic branch


async def classify_query_with_llm(query: str, context: ConversationContext) -> Dict[str, bool]:
    """Return which modules to include. Falls back to simple keyword heuristics."""
    if client is None:
        return {
            "include_weather": "weather" in query.lower() or "rain" in query.lower() or "forecast" in query.lower(),
            "include_arrivals": "arrival" in query.lower() or "supply" in query.lower() or "arrival prediction" in query.lower(),
            "include_price": "price" in query.lower() or "sell" in query.lower() or "market price" in query.lower(),
            "include_disease": "disease" in query.lower() or "pest" in query.lower(),
            "include_crop_rotation": False,
            "include_profitability": "profit" in query.lower() or "profitable" in query.lower()
        }

    classification_prompt = f"""
Analyze this farmer's question and determine which data modules are needed.

Context: {context.get_context_summary()}
Farmer's Question: "{query}"

Available Modules:
1. weather - Weather forecasts, rain, temperature
2. arrivals - Market supply, arrival predictions
3. price - Price predictions and trends
4. disease - Crop diseases, pests
5. crop_rotation - What to plant next
6. profitability - Which crop is most profitable

Return ONLY a JSON object:
{{
  "weather": true/false,
  "arrivals": true/false,
  "price": true/false,
  "disease": true/false,
  "crop_rotation": true/false,
  "profitability": true/false
}}
"""
    try:
        resp = await _call_openai_chat(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Query classifier. Return only valid JSON."},
                {"role": "user", "content": classification_prompt}
            ],
            max_tokens=150,
            temperature=0.0
        )
        result_text = resp.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        classification = json.loads(result_text)
        return {
            "include_weather": classification.get("weather", False),
            "include_arrivals": classification.get("arrivals", False),
            "include_price": classification.get("price", False),
            "include_disease": classification.get("disease", False),
            "include_crop_rotation": classification.get("crop_rotation", False),
            "include_profitability": classification.get("profitability", False)
        }
    except Exception as e:
        logger.error(f"Classification error: {e}")
        logger.debug(traceback.format_exc())
        # heuristic fallback
        return await classify_query_with_llm(query, context)  # will call heuristic branch when client is None


async def generate_comprehensive_advisory(user_query: str, data: Dict[str, Any], context: ConversationContext) -> str:
    """Generate friendly advisory using LLM; fallback to basic templated advice."""
    if client is None:
        # Simple fallback advisory
        lines = []
        if data.get("weather"):
            w = data["weather"]
            lines.append(f"Weather: {w.get('summary', 'unavailable')}.")
        if data.get("arrivals"):
            lines.append("Arrivals data suggests check local mandi for updates before selling.")
        if data.get("price"):
            lines.append("Price predictions available — consider timing sales for better prices.")
        lines.append("For more detailed advice, configure the OpenAI API key.")
        return " ".join(lines)

    # Build prompt
    weather = data.get("weather") or {}
    price = data.get("price") or {}
    arrivals = data.get("arrivals") or {}
    conversation_history = "\n".join([f"Q: {h['query']}" for h in context.conversation_history[-3:]])

    context_parts = [f"Context: {context.get_context_summary()}"]
    insights = weather.get("insights", []) if weather else []

    if insights:
        try:
            avg_temp = sum(i["features"]["max_temp"] for i in insights) / len(insights)
            total_rain = sum(i["features"]["rainfall"] for i in insights)

            context_parts.append(
                f"Weather: {total_rain:.0f}mm rain expected, avg max temp {avg_temp:.1f}°C"
            )
            context_parts.append(f"Weather Impact: {weather.get('overall_recommendation')}")
        except:
            pass


    if price and "result" in price:
        # best-effort reading of predictions
        preds = price["result"].get("total_predictions", []) if isinstance(price["result"], dict) else []
        if preds:
            first_price = preds[0].get("average_predicted_value", 0)
            last_price = preds[-1].get("average_predicted_value", 0)
            if first_price:
                change = ((last_price / first_price) - 1) * 100
                trend = "rising" if change > 5 else "falling" if change < -5 else "stable"
                context_parts.append(f"Price Trend: {trend} ({change:+.1f}%)")
                context_parts.append(f"Current avg price: ₹{first_price:.2f}/quintal")

    context_str = "\n".join(context_parts)

    prompt = f"""
You are an expert agricultural advisor in India having a conversation with a farmer.

Previous conversation (last 3):
{conversation_history}

Current Question: "{user_query}"

Data Available:
{context_str}

Provide practical, conversational advice (2-4 sentences). Be:
- Direct and actionable
- Friendly and conversational
- Specific about timing and steps
- Consider farmer's previous questions

IMPORTANT: Keep it brief and natural, like talking to a friend.
"""
    try:
        resp = await _call_openai_chat(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Expert agricultural advisor. Natural, brief, actionable advice."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        result = resp.choices[0].message.content
        return (result or "Unable to generate advice.").strip()
    except Exception as e:
        logger.error(f"Advisory generation error: {e}")
        logger.debug(traceback.format_exc())
        return "Could not generate advice at this time."


# ==================== Weather Fetcher ====================
# ==================== Enhanced Weather Advisory (WeatherHelper) ====================

def enhanced_weather_advisory(location: str, commodity: str = None, days: int = 3) -> Dict:
    """
    Get enhanced weather advisory using WeatherHelper.
    Includes summary, features, adjustment factor and impact.
    """
    try:
        helper = WeatherHelper()

        weather_insights = []
        for day_offset in range(days):
            forecast_date = (
                datetime.now().date() + timedelta(days=day_offset + 1)
            ).strftime("%Y-%m-%d")

            weather_data = helper.get_weather_input(
                district=location,
                date=forecast_date,
                commodity=commodity,
                include_adjustment=bool(commodity)
            )

            weather_insights.append({
                "date": forecast_date,
                "summary": weather_data["summary"],
                "features": weather_data["features"],
                "impact": weather_data.get("adjustment_message", "No specific impact"),
                "adjustment_factor": weather_data.get("adjustment_factor", 1.0)
            })

        return {
            "location": location,
            "commodity": commodity,
            "forecast_days": days,
            "insights": weather_insights,
            "overall_recommendation": _generate_recommendation(weather_insights, commodity)
        }

    except Exception as e:
        return {
            "location": location,
            "error": str(e),
            "insights": [],
            "overall_recommendation": "Weather data unavailable"
        }


def _generate_recommendation(insights, commodity):
    """
    Generate human-friendly weather recommendation based on adjustment factors.
    """
    if not insights:
        return "No weather data available for recommendations"

    avg_adj = sum(i.get("adjustment_factor", 1.0) for i in insights) / len(insights)

    if avg_adj < 0.95:
        return f"⚠️ Unfavorable weather for {commodity}. Consider protective measures and stay alert for diseases."
    elif avg_adj > 1.02:
        return f"✅ Favorable conditions for {commodity}. Good time for field activities."
    else:
        return f"➡️ Neutral weather for {commodity}. No major risks expected."

def simple_weather_fetch(location: str, days: int = 3) -> Dict[str, Any]:
    default_weather = {
        "summary": "Weather data unavailable",
        "forecast": [],
        "condition": "Unknown",
        "rain_mm": 0,
        "temp_c": 0,
        "num_days": 0,
        "error": None
    }
    if not WEATHER_API_KEY or len(WEATHER_API_KEY) < 10:
        default_weather["error"] = "API key not configured"
        return default_weather

    try:
        url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            "key": WEATHER_API_KEY,
            "q": f"{location},India",
            "days": max(1, min(days, 7)),
            "aqi": "no",
            "alerts": "no"
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            default_weather["error"] = f"HTTP {resp.status_code}"
            return default_weather
        data = resp.json()
        forecast_days = data.get("forecast", {}).get("forecastday", [])
        if not forecast_days:
            return default_weather

        simplified = []
        total_rain = 0
        total_temp = 0
        for day in forecast_days:
            dd = day.get("day", {})
            rain = dd.get("totalprecip_mm", 0)
            temp = dd.get("avgtemp_c", 0)
            total_rain += rain
            total_temp += temp
            simplified.append({
                "date": day.get("date"),
                "rain_mm": round(rain, 1),
                "temp_c": round(temp, 1),
                "condition": dd.get("condition", {}).get("text", "Unknown")
            })
        loc = data.get("location", {}).get("name", location)
        return {
            "summary": simplified[0]["condition"],
            "condition": simplified[0]["condition"],
            "rain_mm": round(total_rain, 2),
            "temp_c": round(total_temp / len(forecast_days), 1),
            "forecast": simplified,
            "num_days": len(simplified),
            "source": "WeatherAPI.com",
            "location": loc
        }
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        logger.debug(traceback.format_exc())
        default_weather["error"] = str(e)
        return default_weather


# ==================== Arrival & Price Tool Wrappers ====================
async def call_arrival_predictor_async(district: Optional[str], amc_name: Optional[str], commodity: Optional[str], days: int):
    """Call arrival prediction from arrival_tool. Works with both sync and async functions."""
    if arrival_tool is None:
        return None
    try:
        # Prefer an async function name if present
        if hasattr(arrival_tool, "enhanced_prediction_with_telangana"):
            enhanced = getattr(arrival_tool, "enhanced_prediction_with_telangana")
            # Build a fallback request object/dict if module doesn't provide PredictionRequest
            req_obj = None
            if hasattr(arrival_tool, "PredictionRequest"):
                PR = getattr(arrival_tool, "PredictionRequest")
                try:
                    req_obj = PR(metric="total_bags", district=district, amc_name=amc_name, commodity=commodity, days=days)
                except Exception:
                    # attempt dict fallback
                    req_obj = {"metric": "total_bags", "district": district, "amc_name": amc_name, "commodity": commodity, "days": days}
            else:
                req_obj = {"metric": "total_bags", "district": district, "amc_name": amc_name, "commodity": commodity, "days": days}

            if asyncio.iscoroutinefunction(enhanced):
                return {"source": "Arrival Prediction (arrival_tool)", "result": await enhanced(req_obj)}
            else:
                # run in threadpool if sync
                return {"source": "Arrival Prediction (arrival_tool)", "result": await asyncio.to_thread(enhanced, req_obj)}

        # fallback: maybe module exposes simple predict function
        if hasattr(arrival_tool, "predict_arrival"):
            fn = getattr(arrival_tool, "predict_arrival")
            if asyncio.iscoroutinefunction(fn):
                return {"source": "Arrival Prediction (arrival_tool)", "result": await fn(district=district, amc_name=amc_name, commodity=commodity, days=days)}
            else:
                return {"source": "Arrival Prediction (arrival_tool)", "result": await asyncio.to_thread(fn, district, amc_name, commodity, days)}
    except Exception as e:
        logger.warning(f"Arrival predictor error: {e}")
        logger.debug(traceback.format_exc())
    return None


async def call_price_predictor_async(district: Optional[str], amc_name: Optional[str], commodity: Optional[str], days: int):
    """Call price prediction from price_tool. Supports sync/async."""
    if price_tool is None:
        return None
    try:
        # If price_tool exposes a class SimpleTFTModel or similar:
        if hasattr(price_tool, "SimpleTFTModel"):
            cls = getattr(price_tool, "SimpleTFTModel")
            # instantiate in thread if sync constructor is heavy
            def _construct_and_predict():
                try:
                    model = cls()
                    if hasattr(model, "predict"):
                        return model.predict(metric="avg_price", district=district, amc_name=amc_name, commodity=commodity, days=days)
                    # fallback if top-level function exists
                    if hasattr(price_tool, "predict_price"):
                        return price_tool.predict_price(district=district, amc_name=amc_name, commodity=commodity, days=days)
                    return None
                except Exception as e:
                    logger.warning(f"price_tool model predict failed: {e}")
                    return None

            result = await asyncio.to_thread(_construct_and_predict)
            if result is not None:
                return {"source": "Price Prediction (price_tool)", "result": result}

        # fallback: top-level async/sync function
        if hasattr(price_tool, "predict_price"):
            fn = getattr(price_tool, "predict_price")
            if asyncio.iscoroutinefunction(fn):
                return {"source": "Price Prediction (price_tool)", "result": await fn(district=district, amc_name=amc_name, commodity=commodity, days=days)}
            else:
                return {"source": "Price Prediction (price_tool)", "result": await asyncio.to_thread(fn, district, amc_name, commodity, days)}

    except Exception as e:
        logger.warning(f"Price predictor error: {e}")
        logger.debug(traceback.format_exc())
    return None


# ==================== Main Processing Pipeline ====================
async def process_query_async(query: str, context: ConversationContext) -> Dict[str, Any]:
    logger.info(f"Processing query: {query}")
    extracted = await extract_parameters_from_query(query, context)
    context.update_from_query(extracted)

    if extracted.get("needs_clarification") and extracted.get("missing_params"):
        missing = ", ".join(extracted["missing_params"])
        return {
            "needs_clarification": True,
            "message": f"I need more information: {missing}.",
            "context": context.get_context_summary()
        }

    classification = await classify_query_with_llm(query, context)
    logger.info(f"Classification: {classification}")

    # Prepare coroutines for parallel execution
    tasks = []
    results = {"weather": None, "arrivals": None, "price": None}
    modules_used: List[str] = []

    if classification.get("include_weather"):
        location = extracted.get("district") or extracted.get("amc_name") or "Hyderabad"
        weather = enhanced_weather_advisory(
            location=location,
            commodity=extracted.get("commodity"),
            days=extracted.get("days", 3)
        )
        
        # ✅ Wrap weather in consistent structure so MCP understands it
        results["weather"] = {
            "summary": (
                weather.get("insights", [{}])[0].get("summary")
                if weather.get("insights")
                else weather.get("summary", "Weather data unavailable")
            ),
            "insights": weather.get("insights", []),
            "overall_recommendation": weather.get("overall_recommendation", "No weather insights available"),
            "raw": weather
        }

        if weather and weather.get("insights"):
            modules_used.append("Weather")


    # Kick off arrivals and price in parallel
    arrival_task = None
    price_task = None
    if classification.get("include_arrivals"):
        arrival_task = asyncio.create_task(call_arrival_predictor_async(
            extracted.get("district"),
            extracted.get("amc_name"),
            extracted.get("commodity"),
            extracted.get("days", 7)
        ))
    if classification.get("include_price"):
        price_task = asyncio.create_task(call_price_predictor_async(
            extracted.get("district"),
            extracted.get("amc_name"),
            extracted.get("commodity"),
            extracted.get("days", 7)
        ))

    # await tasks if created
    if arrival_task:
        try:
            arrival_res = await arrival_task
            results["arrivals"] = arrival_res
            if arrival_res:
                modules_used.append("Arrivals")
        except Exception as e:
            logger.warning(f"Arrival task error: {e}")

    if price_task:
        try:
            price_res = await price_task
            results["price"] = price_res
            if price_res:
                modules_used.append("Price")
        except Exception as e:
            logger.warning(f"Price task error: {e}")

    summary = {
        "district": extracted.get("district"),
        "commodity": extracted.get("commodity"),
        "weather": results["weather"],
        "price": results["price"],
        "arrivals": results["arrivals"]
    }

    advice = await generate_comprehensive_advisory(query, summary, context)
    context.add_to_history(query, advice)

    # Build price summary structure for response (best-effort)
    price_summary = None
    if classification.get("include_price") and results["price"]:
        try:
            preds = results["price"]["result"].get("total_predictions", []) if isinstance(results["price"].get("result"), dict) else []
            price_summary = {"predictions": preds[:3]}
        except Exception:
            price_summary = {"predictions": []}

    return {
        "needs_clarification": False,
        "advice": advice,
        "context": context.get_context_summary(),
        "modules_used": modules_used,
        "weather": results["weather"] if classification.get("include_weather") else None,
        "price_summary": price_summary
    }


# ==================== FastAPI App & Endpoints ====================
app = FastAPI(title="Smart Agricultural Advisory Bot API", version="4.0")

# in-memory session contexts (for MCP, replace with persistent store if required)
session_contexts: Dict[str, ConversationContext] = {}


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    district: Optional[str] = None
    amc_name: Optional[str] = None
    commodity: Optional[str] = None


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        sid = req.session_id or "default"
        if sid not in session_contexts:
            session_contexts[sid] = ConversationContext()
        context = session_contexts[sid]

        # allow callers to seed context
        if req.district:
            context.district = req.district
        if req.amc_name:
            context.amc_name = req.amc_name
        if req.commodity:
            context.commodity = req.commodity

        result = await process_query_async(req.query, context)
        return {"success": True, "session_id": sid, **result}
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset_context")
def reset_context(session_id: str = "default"):
    try:
        if session_id in session_contexts:
            del session_contexts[session_id]
        return {"success": True, "message": "Context reset"}
    except Exception as e:
        logger.error(f"Reset context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "4.0",
        "features": ["conversational", "context-aware", "arrival_tool", "price_tool"]
    }


@app.get("/")
def root():
    return {
        "name": "Smart Agricultural Advisory Bot",
        "version": "4.0",
        "endpoints": {
            "POST /chat": "Send queries (body: ChatRequest)",
            "POST /reset_context?session_id=:id": "Reset session",
            "GET /health": "Health check"
        }
    }


# ==================== Run server when executed directly ====================
if __name__ == "__main__":
    import sys

    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    elif len(sys.argv) > 2 and sys.argv[1] in ("serve", "api", "server"):
        try:
            port = int(sys.argv[2])
        except Exception:
            port = 8000

    logger.info("🚀 Starting advisory_system FastAPI server")
    uvicorn.run("advisory_system:app", host="0.0.0.0", port=port, reload=False)
