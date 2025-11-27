# mcp_tools/weather_tool.py

import requests
from datetime import datetime

def get_weather(location: str, days: int = 1):
    try:
        api_url = f"https://wttr.in/{location}?format=j1"
        data = requests.get(api_url, timeout=10).json()

        today = data["weather"][0]
        condition = today["hourly"][4]["weatherDesc"][0]["value"]
        rain_chance = int(today["hourly"][4]["chanceofrain"])
        temp = float(today["hourly"][4]["tempC"])

        forecast = [{
            "condition": condition,
            "rain_chance": rain_chance,
            "temperature": temp
        }]

        return {"status": "success", "forecast": forecast}

    except Exception as e:
        return {"status": "error", "message": str(e)}
