# client_api.py
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import re
import os

MCP_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8005/mcp/run")

app = FastAPI(title="TAMC Client API Bridge")

# Allow React frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Detect tool -----------------
def detect_tool_and_extract(query: str):
    q = query.lower().strip()
    location = None
    match = re.search(r"\bin\s+([a-zA-Z ]{2,30})", q)
    if match:
        location = match.group(1).strip()

    commodity = None
    m = re.search(r"(?:price|rate|cost)\s+(?:for|of)\s+([a-zA-Z ]{2,30})", q)
    if m:
        commodity = m.group(1).strip()

    if any(k in q for k in ["price", "rate", "cost", "₹", "quintal"]):
        return "price", {"commodity": commodity or "", "district": location or ""}
    if any(k in q for k in ["arrival", "arrivals", "expected", "bags", "forecast"]):
        return "arrival", {"amc_name": location or "", "days": 7}
    if any(k in q for k in ["weather", "rain", "temperature", "humidity", "sunny"]):
        return "weather", {"location": location or "", "days": 1}
    return None, None


# ----------------- API Endpoint -----------------
@app.post("/ask")
def ask_question(payload: dict):
    query = payload.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    tool, params = detect_tool_and_extract(query)
    if not tool:
        return {"status": "unknown", "message": "Unable to detect tool"}

    try:
        res = requests.post(MCP_URL, json={"tool": tool, "params": params}, timeout=60)
        res.raise_for_status()
        data = res.json()
        return {"tool": tool, "query": query, "result": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contacting MCP server: {e}")


if __name__ == "__main__":
    uvicorn.run("client_api:app", host="0.0.0.0", port=8010, reload=True)
