# client.py
import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()

MCP_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_SERVER_PORT", 8005))
SERVER_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp/run"
TIMEOUT = 30  # seconds

# -------------------------------------------------------
# 🔍 Detect Tool & Extract Parameters
# -------------------------------------------------------
def detect_tool_and_extract(query: str):
    """Detect which tool to use and extract relevant parameters dynamically."""
    q = query.lower().strip()

    # Location extraction
    location = None
    loc_match = re.search(r"\b(?:in|at|for)\s+([a-zA-Z ]{2,40})", q)
    if loc_match:
        location = loc_match.group(1).strip().title()

    # Commodity extraction
    commodity = None
    com_match = re.search(r"(?:price|rate|cost|of)\s+([a-zA-Z ]{2,30})", q)
    if com_match:
        commodity = com_match.group(1).strip().title()

    # Tool detection
    if any(k in q for k in ["price", "rate", "cost", "₹", "quintal"]):
        return "price", {"commodity": commodity or "", "district": location or ""}
    elif any(k in q for k in ["arrival", "arrivals", "expected no", "expected number", "how many arrivals"]):
        return "arrival", {"amc_name": location or "", "days": 7}
    elif any(k in q for k in ["weather", "rain", "temperature", "forecast"]):
        return "weather", {"location": location or "", "days": 1}
    elif any(k in q for k in ["bring", "bags", "should i bring", "suggest", "advice"]):
        return "advice", {"location": location or "", "commodity": commodity or ""}
    else:
        return "chat", {"query": query}


# -------------------------------------------------------
# 🌐 Call MCP Server
# -------------------------------------------------------
def call_server(tool: str, params: dict):
    """Send structured payload to MCP server and get response."""
    payload = {"tool": tool, "params": params}
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "⏳ Request timed out. Server may be busy."}
    except requests.exceptions.ConnectionError as e:
        return {"status": "error", "message": f"🔌 Connection error: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"❌ Unexpected error: {e}"}


# -------------------------------------------------------
# 🖨 Pretty Print Outputs
# -------------------------------------------------------
def pretty_print(tool: str, data: dict):
    """Nicely display responses from MCP tools."""
    divider = "----------------------------------------------------------"

    if not isinstance(data, dict):
        print(divider)
        print(data)
        print(divider)
        return

    if data.get("status") == "error":
        print(divider)
        print(f"❌ {tool.capitalize()} Tool Error:")
        print(f"🔹 {data.get('message', 'Unknown error.').strip()}")
        print(divider)
        return

    # 💰 PRICE TOOL
    if tool == "price":
        print(divider)
        print(f"💰 Price Forecast for {data.get('commodity', 'Commodity')} in {data.get('district', 'District')}:")
        print(f"📅 Date: {data.get('date', 'Today')}")
        print(f"🔹 Min Price: ₹{int(data.get('min_price', 0)):,}/quintal")
        print(f"🔹 Modal Price: ₹{int(data.get('modal_price', 0)):,}/quintal")
        print(f"🔹 Max Price: ₹{int(data.get('max_price', 0)):,}/quintal")
        print(divider)
        return

    # 📈 ARRIVAL TOOL
    if tool == "arrival":
        print(divider)
        print(f"📈 Arrival Forecast for {data.get('amc_name', 'Market')}:")
        total = data.get("total_predicted", [])
        if total:
            for entry in total:
                print(f"  📅 {entry.get('date', 'N/A')}: {int(entry.get('total_predicted_value', 0)):,} arrivals")
        else:
            print("  No arrival forecast data available.")
        print(divider)
        return

    # 🌤 WEATHER TOOL
    if tool == "weather":
        print(divider)
        loc = data.get("location", "Location")
        forecast = data.get("forecast", [{}])[0]
        print(f"🌦 Weather Forecast for {loc}:")
        print(f"🔹 Condition: {forecast.get('condition', 'N/A')}")
        print(f"🔹 Rain Chance: {forecast.get('rain_chance', 'N/A')}%")
        print(f"🔹 Temperature: {forecast.get('temperature', 'N/A')}°C")
        print(divider)
        return

    # 🌾 ADVICE TOOL
    if tool == "advice":
        print(divider)
        print(f"🌾 Farming Advice for {data.get('location', 'your area')}:")
        print(f"🔹 Commodity: {data.get('commodity', 'N/A')}")
        if "advice" in data:
            print(f"💡 Advice: {data['advice']}")
        else:
            print("💡 No specific advice available.")
        print(divider)
        return

    # 💬 CHAT TOOL
    if tool == "chat":
        print(divider)
        print(f"🤖 {data.get('reply', 'No response from chatbot.')}")
        print(divider)
        return

    # 🧩 Default Fallback
    print(divider)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print(divider)


# -------------------------------------------------------
# 🚀 Main CLI Entry
# -------------------------------------------------------
if __name__ == "__main__":
    print("🌾 Welcome to TAMC Agri Assistant CLI (type 'exit' to quit)")
    print("----------------------------------------------------------")

    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye, Farmer!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("👋 Goodbye, Farmer!")
            break

        tool, params = detect_tool_and_extract(query)
        print(f"🔧 Tool selected: {tool}")

        response = call_server(tool, params)
        pretty_print(tool, response)
