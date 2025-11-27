import os
import sys
import time
import subprocess
import requests
import threading
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

TOOLS = {
    "arrival": {"path": "mcp_tools/arrival_tool.py", "port": 8000},
    "price": {"path": "mcp_tools/price_tool.py", "port": 8002},
    "advisory": {"path": "mcp_tools/advisory_system.py", "port": 8003},
    "mcp": {"path": "mcp_server.py", "port": 8005},
}


# --------------------------------------------------------------
# 🧩 Start any tool inline (runs inside same terminal)
# --------------------------------------------------------------
def start_tool_inline(name, tool):
    print(f"\n🚀 Launching {name}_tool (port {tool['port']})...")
    process = subprocess.Popen(
        [sys.executable, tool["path"]],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=os.environ.copy(),
    )

    # Wait up to 90s for port to come alive
    for _ in range(150):
        try:
            requests.get(f"http://127.0.0.1:{tool['port']}/", timeout=2)
            print(f"✅ {name}_tool started successfully on port {tool['port']}")
            return process
        except Exception:
            time.sleep(2)

    print(f"⚠️ {name}_tool did not respond on port {tool['port']} (continuing...)")
    return process


# --------------------------------------------------------------
# 🧠 AI-Enhanced Request Interceptor (port 8010)
# --------------------------------------------------------------
def start_ai_interceptor():
    """
    Smart interceptor that adds AI intelligence to requests
    """
    app = FastAPI(title="AI-Enhanced Smart Interceptor", version="2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import AI components
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            ai_model = genai.GenerativeModel("gemini-2.0-flash-exp")
            print("🧠 AI Enhancement: Gemini API configured")
        else:
            ai_model = None
            print("⚠️ AI Enhancement: Gemini API not configured (will use fallback)")
    except Exception as e:
        ai_model = None
        print(f"⚠️ AI Enhancement: Could not load AI ({e})")

    def enhance_params_with_ai(query, params):
        """Use AI to enhance parameter extraction"""
        if not ai_model or not query:
            return params
        
        try:
            prompt = f"""Extract agricultural parameters from this query: "{query}"

Current params: {params}

Return ONLY valid JSON with these fields (use null if not found):
{{
  "commodity": "commodity name or null",
  "district": "district/location or null",
  "amc_name": "market name or null",
  "days": number or 7,
  "metric": "metric name or null"
}}"""
            
            response = ai_model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean markdown
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            import json
            ai_params = json.loads(result_text)
            
            # Merge AI params with existing params (AI doesn't overwrite existing)
            for key, value in ai_params.items():
                if value and not params.get(key):
                    params[key] = value
            
            print(f"🧠 AI Enhanced params: {params}")
            return params
            
        except Exception as e:
            print(f"⚠️ AI enhancement failed: {e}")
            return params

    @app.middleware("http")
    async def intercept_requests(request: Request, call_next):
        if request.url.path == "/mcp/run":
            body = await request.json()
            tool = body.get("tool", "")
            params = body.get("params", {})
            
            # AI Enhancement: Use AI to improve parameter extraction
            query = params.get("query", "")
            if query and ai_model:
                params = enhance_params_with_ai(query, params)
                body["params"] = params
            
            # Legacy auto-correction
            if tool == "arrival":
                amc_name = params.get("amc_name", "")
                if amc_name.lower() in ["are", "ar", "arr"]:
                    print("🧠 Auto-corrected amc_name → Khammam")
                    body["params"]["amc_name"] = "Khammam"

            import httpx
            try:
                from fastapi.responses import JSONResponse
            except Exception:
                from starlette.responses import JSONResponse

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"http://127.0.0.1:{TOOLS['mcp']['port']}/mcp/run",
                        json=body,
                        timeout=180,
                    )
                    return JSONResponse(status_code=response.status_code, content=response.json())
                except Exception as e:
                    return JSONResponse(status_code=500, content={"error": str(e)})

        return await call_next(request)

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "ai_enabled": ai_model is not None,
            "version": "2.0"
        }

    print("\n🧠 Starting AI-Enhanced Smart Interceptor on port 8010...")
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")


# --------------------------------------------------------------
# 🧩 Advisory Tool (safe start)
# --------------------------------------------------------------
def start_advisory_safely():
    """Start advisory_tool by running its serve command safely."""
    print("\n🕒 Starting advisory_tool...")
    env = os.environ.copy()
    env["DISABLE_SUBTOOLS"] = "1"  # prevents advisory from spawning other tools

    # Run the advisory tool with its serve command
    process = subprocess.Popen(
        [sys.executable, "mcp_tools/advisory_system.py", "serve", "8003"],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Wait until the /health endpoint or base URL is reachable
    for _ in range(60):  # 2 minutes max
        try:
            res = requests.get("http://127.0.0.1:8003/health", timeout=3)
            if res.status_code < 500:
                print("✅ advisory_tool started successfully on port 8003")
                return process
        except Exception:
            time.sleep(2)
    print("⚠️ advisory_tool did not respond on port 8003 (continuing...)")
    return process


# --------------------------------------------------------------
# 🚀 Main orchestrator
# --------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧠 LAUNCHING AI-ENHANCED TAMC SYSTEM")
    print("="*70)
    print("\n🎯 Architecture:")
    print("  Layer 1: Prediction Tools (arrival, price, advisory)")
    print("  Layer 2: MCP Server (tool orchestration)")
    print("  Layer 3: AI Interceptor (intelligent enhancement)")
    print("\n✨ AI Features:")
    print("  🧠 Intelligent parameter extraction")
    print("  🎯 Query understanding & enhancement")
    print("  🔀 Smart tool coordination")
    print("  💬 Context-aware processing")
    print("="*70)

    processes = []

    # Step 1: Start core prediction tools
    print("\n📊 STEP 1: Starting prediction tools...")
    for name in ["arrival", "price"]:
        p = start_tool_inline(name, TOOLS[name])
        if p:
            processes.append(p)
        time.sleep(3)

    # Step 2: Start advisory system
    print("\n🧠 STEP 2: Starting advisory system...")
    time.sleep(5)
    p = start_advisory_safely()
    if p:
        processes.append(p)

    # Step 3: Start AI-Enhanced Interceptor (background)
    print("\n🧠 STEP 3: Starting AI-Enhanced Interceptor...")
    time.sleep(3)
    threading.Thread(target=start_ai_interceptor, daemon=True).start()
    
    # Wait for interceptor to start
    for attempt in range(30):
        try:
            res = requests.get("http://127.0.0.1:8010/health", timeout=2)
            if res.status_code == 200:
                print("✅ AI-Enhanced Interceptor started on port 8010")
                break
        except Exception:
            time.sleep(1)

    # Step 4: Start MCP Server
    print("\n" + "="*70)
    print("🚀 STEP 4: Starting MCP Server...")
    print("="*70)
    print("\n📡 All Services Running:")
    print("  • Port 8000: Arrival Prediction Tool")
    print("  • Port 8002: Price Prediction Tool")
    print("  • Port 8003: Advisory System")
    print("  • Port 8005: MCP Server (Main)")
    print("  • Port 8010: AI Interceptor (Enhancement Layer)")
    print("\n💡 Frontend should connect to:")
    print("  • Primary: http://localhost:8005 (MCP Server)")
    print("  • Alternative: http://localhost:8010 (with AI enhancement)")
    print("\n🧠 AI Capabilities:")
    print("  • Natural language query understanding")
    print("  • Automatic parameter extraction")
    print("  • Context-aware responses")
    print("  • Multi-tool coordination")
    print("="*70 + "\n")

    # MCP Server runs in the same process (blocking)
    try:
        subprocess.run([sys.executable, TOOLS["mcp"]["path"]])
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown signal received...")
    finally:
        print("\n🧹 Shutting down all tools...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("✅ Shutdown complete!")