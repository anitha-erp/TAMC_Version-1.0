# ai_intelligence_layer.py
# Discrete AI Layer Between Tools and User Response

import json
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class AIIntelligenceLayer:
    """
    Discrete layer that interprets tool results and generates intelligent insights
    This sits between tool execution and response formatting
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    def analyze_tool_results(
        self, 
        query: str,
        tool_results: Dict,
        context: Dict = None
    ) -> Dict:
        """
        Main analysis function that interprets raw tool results
        
        Args:
            query: Original user question
            tool_results: Raw outputs from arrival_tool, price_tool, advisory_tool
            context: Additional context (commodity, location, etc.)
        
        Returns:
            Dict containing:
                - interpretation: Natural language explanation
                - insights: Key takeaways
                - recommendations: Actionable advice
                - risk_assessment: Identified risks
                - opportunities: Potential gains
                - market_intelligence: Market dynamics analysis
        """
        
        # Extract and structure data from tool results
        structured_data = self._structure_tool_data(tool_results)
        
        # Build comprehensive analysis prompt
        analysis_prompt = self._build_analysis_prompt(
            query, 
            structured_data, 
            context
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse AI response
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            
            analysis = json.loads(result)
            
            return {
                "success": True,
                "analysis": analysis,
                "raw_data": structured_data  # Include raw data for transparency
            }
            
        except Exception as e:
            print(f"AI Analysis Error: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback": self._generate_fallback_analysis(structured_data)
            }
    
    def _structure_tool_data(self, tool_results: Dict) -> Dict:
        """
        Convert raw tool outputs into structured format for AI analysis
        """
        structured = {
            "arrival_data": None,
            "price_data": None,
            "advisory_data": None,
            "telangana_data": None,
            "weather_data": None
        }
        
        # Process arrival tool results
        if "arrival" in tool_results and tool_results["arrival"].get("success"):
            data = tool_results["arrival"].get("data", {})
            total_predicted = data.get("total_predicted", [])
            
            if total_predicted:
                structured["arrival_data"] = {
                    "predictions": total_predicted[:7],  # Next 7 days
                    "total_week": sum(d.get("total_predicted_value", 0) for d in total_predicted[:7]),
                    "average_daily": sum(d.get("total_predicted_value", 0) for d in total_predicted[:7]) / min(7, len(total_predicted)),
                    "trend": self._calculate_trend(total_predicted[:7], "total_predicted_value"),
                    "peak_day": max(total_predicted[:7], key=lambda x: x.get("total_predicted_value", 0)) if total_predicted else None,
                    "telangana_insight": data.get("telangana_insight")
                }
        
        # Process price tool results
        if "price" in tool_results and tool_results["price"].get("success"):
            data = tool_results["price"].get("data", {})
            predictions = data.get("predictions", [])
            
            if predictions:
                structured["price_data"] = {
                    "predictions": predictions[:7],
                    "current_price": predictions[0].get("predicted_price", 0) if predictions else 0,
                    "week_end_price": predictions[-1].get("predicted_price", 0) if len(predictions) > 1 else 0,
                    "average_price": sum(p.get("predicted_price", 0) for p in predictions) / len(predictions),
                    "price_trend": self._calculate_trend(predictions, "predicted_price"),
                    "volatility": self._calculate_volatility(predictions, "predicted_price"),
                    "commodity": data.get("commodity"),
                    "district": data.get("district")
                }
        
        # Process advisory/weather data
        if "advisory" in tool_results and tool_results["advisory"].get("success"):
            data = tool_results["advisory"].get("data", {})
            weather = data.get("weather")
            
            if weather:
                structured["weather_data"] = {
                    "condition": weather.get("condition"),
                    "rain_mm": weather.get("rain_mm", 0),
                    "temp_c": weather.get("temp_c", 0),
                    "impact": self._assess_weather_impact(weather)
                }
            
            structured["advisory_data"] = {
                "advice": data.get("advice"),
                "modules_used": data.get("modules_used", [])
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
        """Calculate price volatility"""
        if not predictions or len(predictions) < 2:
            return "low"
        
        values = [p.get(value_key, 0) for p in predictions]
        mean = sum(values) / len(values)
        
        if mean == 0:
            return "low"
        
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        cv = (std_dev / mean) * 100  # Coefficient of variation
        
        if cv > 10:
            return "high"
        elif cv > 5:
            return "moderate"
        else:
            return "low"
    
    def _assess_weather_impact(self, weather: Dict) -> str:
        """Assess weather impact level"""
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
        """System prompt for AI analysis"""
        return """You are CropCast AI, an expert agricultural market analyst with deep knowledge of:
- Supply and demand dynamics
- Price forecasting and market trends
- Weather impact on agriculture
- Risk management strategies
- Business optimization

Your role is to provide ACTIONABLE INTELLIGENCE by analyzing raw prediction data.

Analyze the data and provide:

1. **Interpretation**: Explain what the numbers mean in practical terms
2. **Key Insights**: 3-5 critical observations about market conditions
3. **Recommendations**: Specific actions the farmer should take
4. **Risk Assessment**: Potential problems and how to mitigate them
5. **Opportunities**: Ways to maximize profit
6. **Market Intelligence**: Supply-demand dynamics and pricing trends

Return ONLY valid JSON with this structure:
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
      "insight": "Specific observation",
      "importance": "high|medium|low",
      "data_point": "Supporting data"
    }
  ],
  "recommendations": [
    {
      "action": "What to do",
      "timing": "When to do it",
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
        "mitigation": "How to handle it"
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
    "price_drivers": ["Key factors affecting price"],
    "competitive_position": "Farmer's position in market",
    "timing_strategy": "Best timing advice"
  }
}"""
    
    def _build_analysis_prompt(
        self, 
        query: str, 
        structured_data: Dict, 
        context: Dict
    ) -> str:
        """Build analysis prompt from structured data"""
        
        prompt_parts = [
            f"FARMER'S QUESTION: \"{query}\"",
            "\n" + "="*70 + "\n",
            "PREDICTION DATA TO ANALYZE:\n"
        ]
        
        # Add arrival analysis
        if structured_data.get("arrival_data"):
            arr = structured_data["arrival_data"]
            prompt_parts.append("\n📊 ARRIVAL FORECAST:")
            prompt_parts.append(f"  • Total Week Supply: {arr['total_week']:,.0f} units")
            prompt_parts.append(f"  • Daily Average: {arr['average_daily']:,.0f} units")
            prompt_parts.append(f"  • Trend: {arr['trend']}")
            
            if arr['peak_day']:
                prompt_parts.append(f"  • Peak Day: {arr['peak_day']['date']} with {arr['peak_day']['total_predicted_value']:,.0f} units")
            
            if arr.get('telangana_insight'):
                prompt_parts.append(f"  • Telangana Market: {arr['telangana_insight']}")
        
        # Add price analysis
        if structured_data.get("price_data"):
            price = structured_data["price_data"]
            prompt_parts.append("\n\n💰 PRICE FORECAST:")
            prompt_parts.append(f"  • Current Price: ₹{price['current_price']:,.0f}/quintal")
            prompt_parts.append(f"  • Week-end Price: ₹{price['week_end_price']:,.0f}/quintal")
            prompt_parts.append(f"  • Average Price: ₹{price['average_price']:,.0f}/quintal")
            
            change = ((price['week_end_price'] - price['current_price']) / price['current_price'] * 100) if price['current_price'] > 0 else 0
            prompt_parts.append(f"  • Price Change: {change:+.1f}%")
            prompt_parts.append(f"  • Trend: {price['price_trend']}")
            prompt_parts.append(f"  • Volatility: {price['volatility']}")
        
        # Add weather analysis
        if structured_data.get("weather_data"):
            weather = structured_data["weather_data"]
            prompt_parts.append("\n\n🌤️ WEATHER CONDITIONS:")
            prompt_parts.append(f"  • Condition: {weather['condition']}")
            prompt_parts.append(f"  • Temperature: {weather['temp_c']}°C")
            prompt_parts.append(f"  • Expected Rain: {weather['rain_mm']}mm")
            prompt_parts.append(f"  • Impact Level: {weather['impact']}")
        
        # Add context
        if context:
            prompt_parts.append("\n\n📍 CONTEXT:")
            if context.get("commodity"):
                prompt_parts.append(f"  • Commodity: {context['commodity']}")
            if context.get("location"):
                prompt_parts.append(f"  • Location: {context['location']}")
        
        prompt_parts.append("\n" + "="*70)
        prompt_parts.append("\n🎯 REQUIRED ANALYSIS:")
        prompt_parts.append("Provide intelligent, actionable insights - not just data summary.")
        prompt_parts.append("Focus on:")
        prompt_parts.append("1. What's really happening (market dynamics)")
        prompt_parts.append("2. Why it's happening (cause and effect)")
        prompt_parts.append("3. What the farmer should DO (specific actions)")
        prompt_parts.append("4. Risks and how to handle them")
        prompt_parts.append("5. Opportunities to maximize profit")
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_analysis(self, structured_data: Dict) -> Dict:
        """Simple fallback when AI fails"""
        return {
            "summary": "Data analysis available",
            "interpretation": {
                "market_condition": "unknown",
                "price_outlook": "neutral",
                "confidence_level": "low",
                "reasoning": "Unable to perform AI analysis"
            },
            "key_insights": [],
            "recommendations": [
                {
                    "action": "Review the prediction data carefully",
                    "timing": "Before making decisions",
                    "expected_outcome": "Informed decision making",
                    "priority": "high"
                }
            ],
            "risk_assessment": {
                "overall_risk": "medium",
                "specific_risks": []
            },
            "opportunities": [],
            "market_intelligence": {
                "supply_demand_balance": "Data available in raw results",
                "price_drivers": [],
                "competitive_position": "Review predictions",
                "timing_strategy": "Monitor daily trends"
            }
        }


# Integration function for MCP Server
async def enhance_response_with_ai(
    query: str,
    tool_results: Dict,
    context: Dict = None
) -> Dict:
    """
    Main function to integrate AI intelligence into MCP server
    
    Usage in mcp_server.py:
    
    # After tool execution
    enhanced = await enhance_response_with_ai(
        query=user_query,
        tool_results=tool_results,
        context={"commodity": "Chilli", "location": "Warangal"}
    )
    
    return {
        "response": enhanced["analysis"]["summary"],
        "ai_insights": enhanced["analysis"],
        "tool_data": enhanced["raw_data"]
    }
    """
    ai_layer = AIIntelligenceLayer()
    return ai_layer.analyze_tool_results(query, tool_results, context)


if __name__ == "__main__":
    # Test example
    import asyncio
    
    async def test():
        # Simulate tool results
        tool_results = {
            "arrival": {
                "success": True,
                "data": {
                    "total_predicted": [
                        {"date": "2025-11-08", "total_predicted_value": 1872},
                        {"date": "2025-11-09", "total_predicted_value": 1945},
                        {"date": "2025-11-10", "total_predicted_value": 2012},
                        {"date": "2025-11-11", "total_predicted_value": 2089},
                        {"date": "2025-11-12", "total_predicted_value": 2134},
                        {"date": "2025-11-13", "total_predicted_value": 2187},
                        {"date": "2025-11-14", "total_predicted_value": 2241}
                    ],
                    "telangana_insight": "Increasing trend: +12% over 7 days"
                }
            },
            "price": {
                "success": True,
                "data": {
                    "predictions": [
                        {"date": "2025-11-08", "predicted_price": 5200},
                        {"date": "2025-11-09", "predicted_price": 5150},
                        {"date": "2025-11-10", "predicted_price": 5100},
                        {"date": "2025-11-11", "predicted_price": 5050},
                        {"date": "2025-11-12", "predicted_price": 5000},
                        {"date": "2025-11-13", "predicted_price": 4950},
                        {"date": "2025-11-14", "predicted_price": 4900}
                    ],
                    "commodity": "Chilli",
                    "district": "Warangal"
                }
            },
            "advisory": {
                "success": True,
                "data": {
                    "weather": {
                        "condition": "Light Rain",
                        "rain_mm": 15,
                        "temp_c": 28
                    },
                    "advice": "Monitor market conditions"
                }
            }
        }
        
        result = await enhance_response_with_ai(
            query="What are the expected arrivals next week in Warangal?",
            tool_results=tool_results,
            context={"commodity": "Chilli", "location": "Warangal"}
        )
        
        print("\n" + "="*70)
        print("AI INTELLIGENCE ANALYSIS")
        print("="*70)
        print(json.dumps(result["analysis"], indent=2))
    
    asyncio.run(test())