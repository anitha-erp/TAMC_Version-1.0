# ================================================================
# 🧠 ENHANCED AI INTELLIGENCE LAYER
# True AI-Powered Insights & Recommendations
# ================================================================

import os
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class EnhancedAIIntelligence:
    """
    True AI Intelligence Layer - Adds business insights, not just formatting
    """
    
    def __init__(self):
        self.client = client
        self.model = OPENAI_MODEL
    
    # ================================================================
    # 🎯 Core AI Intelligence Functions
    # ================================================================
    
    async def generate_smart_insights(
        self, 
        query: str,
        tool_results: Dict,
        context: Dict = None
    ) -> Dict:
        """
        Generate intelligent insights from tool results
        Returns: {
            "response": "Natural language response",
            "insights": {...},
            "recommendations": [...],
            "alerts": [...]
        }
        """
        
        # Extract data from tool results
        arrival_data = tool_results.get('arrival', {}).get('data', {})
        price_data = tool_results.get('price', {}).get('data', {})
        weather_data = tool_results.get('weather', {}).get('data', {})
        
        # Build comprehensive analysis prompt
        analysis_prompt = self._build_analysis_prompt(
            query, arrival_data, price_data, weather_data, context
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
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse structured response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            
            structured_response = json.loads(result_text)
            
            return structured_response
            
        except Exception as e:
            print(f"❌ AI Intelligence Error: {e}")
            return self._fallback_response(tool_results)
    
    def _get_system_prompt(self) -> str:
        """System prompt for AI intelligence"""
        return """You are CropCast AI, an expert agricultural market analyst with deep knowledge of:
- Market dynamics and supply-demand patterns
- Price trends and forecasting
- Weather impact on agriculture
- Business optimization strategies
- Risk management in agriculture

Your role is to provide ACTIONABLE INTELLIGENCE, not just data formatting.

Analyze the data and provide:
1. **Market Context**: Is this normal? High? Low? Why?
2. **Trend Analysis**: What's the pattern? Where is it heading?
3. **Strategic Recommendations**: What should the farmer DO?
4. **Risk Alerts**: What could go wrong?
5. **Opportunities**: How to maximize profit?

Return ONLY valid JSON with this structure:
{
  "summary": "One-sentence key takeaway",
  "response": "Natural conversational response with insights (3-4 sentences)",
  "market_context": {
    "situation": "high_supply|normal|low_supply|shortage",
    "reason": "Why is this happening?",
    "comparison": "Compared to historical average/trends"
  },
  "trend_analysis": {
    "direction": "increasing|stable|decreasing",
    "strength": "strong|moderate|weak",
    "forecast": "What to expect in near future"
  },
  "recommendations": [
    {
      "action": "Specific action to take",
      "reasoning": "Why this action",
      "priority": "high|medium|low",
      "timing": "When to do it"
    }
  ],
  "risk_alerts": [
    {
      "type": "price_drop|oversupply|weather|demand_shift",
      "description": "What's the risk",
      "probability": "high|medium|low",
      "mitigation": "How to handle it"
    }
  ],
  "opportunities": [
    {
      "opportunity": "What opportunity exists",
      "potential_benefit": "Expected gain",
      "action_required": "What to do"
    }
  ]
}"""
    
    def _build_analysis_prompt(
        self, 
        query: str,
        arrival_data: Dict,
        price_data: Dict,
        weather_data: Dict,
        context: Dict
    ) -> str:
        """Build comprehensive analysis prompt"""
        
        prompt_parts = [
            f"FARMER'S QUESTION: \"{query}\"\n",
            "="*70,
            "\n📊 DATA AVAILABLE:\n"
        ]
        
        # Add arrival data analysis
        if arrival_data and 'total_predicted' in arrival_data:
            predictions = arrival_data['total_predicted'][:7]  # Next 7 days
            
            prompt_parts.append("\n🚚 ARRIVAL FORECAST:")
            
            if predictions:
                total_week = sum(p.get('total_predicted_value', 0) for p in predictions)
                avg_daily = total_week / len(predictions)
                first_day = predictions[0].get('total_predicted_value', 0)
                last_day = predictions[-1].get('total_predicted_value', 0)
                
                trend = "increasing" if last_day > first_day * 1.1 else \
                        "decreasing" if last_day < first_day * 0.9 else "stable"
                
                prompt_parts.append(f"  • Week Total: {total_week:,.0f} units")
                prompt_parts.append(f"  • Daily Average: {avg_daily:,.0f} units")
                prompt_parts.append(f"  • Trend: {trend}")
                prompt_parts.append(f"  • First day: {first_day:,.0f} → Last day: {last_day:,.0f}")
                
                # Add daily breakdown
                prompt_parts.append("  • Daily breakdown:")
                for p in predictions[:3]:
                    date = p.get('date', 'N/A')
                    value = p.get('total_predicted_value', 0)
                    prompt_parts.append(f"    - {date}: {value:,.0f}")
        
        # Add price data analysis
        if price_data and 'predictions' in price_data:
            predictions = price_data['predictions'][:7]
            
            prompt_parts.append("\n\n💰 PRICE FORECAST:")
            
            if predictions:
                first_price = predictions[0].get('predicted_price', 0)
                last_price = predictions[-1].get('predicted_price', 0)
                avg_price = sum(p.get('predicted_price', 0) for p in predictions) / len(predictions)
                
                price_change = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
                
                prompt_parts.append(f"  • Current Price: ₹{first_price:,.0f}/quintal")
                prompt_parts.append(f"  • Week-end Price: ₹{last_price:,.0f}/quintal")
                prompt_parts.append(f"  • Average: ₹{avg_price:,.0f}/quintal")
                prompt_parts.append(f"  • Price Change: {price_change:+.1f}%")
                
                # Price trend
                trend = "rising" if price_change > 5 else \
                        "falling" if price_change < -5 else "stable"
                prompt_parts.append(f"  • Trend: {trend}")
        
        # Add weather impact
        if weather_data:
            prompt_parts.append("\n\n🌤️ WEATHER CONDITIONS:")
            rain = weather_data.get('rain_mm', 0)
            temp = weather_data.get('temp_c', 0)
            condition = weather_data.get('condition', 'Unknown')
            
            prompt_parts.append(f"  • Condition: {condition}")
            prompt_parts.append(f"  • Temperature: {temp}°C")
            prompt_parts.append(f"  • Expected Rain: {rain}mm")
            
            if rain > 50:
                prompt_parts.append("  • ⚠️ HEAVY RAIN ALERT - Transport disruption likely")
            elif rain > 20:
                prompt_parts.append("  • ⚠️ Moderate rain - Some impact expected")
        
        # Add context if available
        if context:
            prompt_parts.append("\n\n📝 CONTEXT:")
            if context.get('commodity'):
                prompt_parts.append(f"  • Commodity: {context['commodity']}")
            if context.get('location'):
                prompt_parts.append(f"  • Location: {context['location']}")
        
        prompt_parts.append("\n" + "="*70)
        prompt_parts.append("\n🎯 REQUIRED ANALYSIS:")
        prompt_parts.append("Provide intelligent insights, not just data summary:")
        prompt_parts.append("1. What's REALLY happening in this market?")
        prompt_parts.append("2. Why is this happening? (supply/demand dynamics)")
        prompt_parts.append("3. What should the farmer DO? (specific actions)")
        prompt_parts.append("4. What risks exist? (and how to handle them)")
        prompt_parts.append("5. What opportunities exist? (how to profit)")
        
        return "\n".join(prompt_parts)
    
    def _fallback_response(self, tool_results: Dict) -> Dict:
        """Simple fallback when AI fails"""
        return {
            "summary": "Data retrieved successfully",
            "response": "Forecast data available. Check detailed results below.",
            "market_context": {
                "situation": "normal",
                "reason": "Data available",
                "comparison": "Unable to provide comparison"
            },
            "trend_analysis": {
                "direction": "stable",
                "strength": "moderate",
                "forecast": "Continuing current pattern"
            },
            "recommendations": [
                {
                    "action": "Review the forecast data",
                    "reasoning": "Make informed decisions",
                    "priority": "medium",
                    "timing": "Before planning sales"
                }
            ],
            "risk_alerts": [],
            "opportunities": []
        }
    
    # ================================================================
    # 🔍 Comparative Intelligence
    # ================================================================
    
    async def compare_markets(
        self,
        commodity: str,
        markets: List[str],
        metric: str = "price"
    ) -> Dict:
        """
        Compare across multiple markets to find best opportunities
        """
        prompt = f"""Analyze which market offers the best opportunity for {commodity}.

Markets to compare: {', '.join(markets)}
Metric: {metric}

Consider:
- Price differences
- Transportation costs
- Market size/demand
- Competition levels

Return JSON:
{{
  "best_market": "market name",
  "reasoning": "why this market",
  "price_advantage": "how much more profit",
  "risk_factors": ["list of risks"],
  "action_plan": "specific steps to take"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are CropCast AI, a market comparison expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            result = response.choices[0].message.content.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            
            return json.loads(result)
        except Exception as e:
            print(f"❌ Market comparison error: {e}")
            return {"error": str(e)}
    
    # ================================================================
    # 📈 Predictive Intelligence
    # ================================================================
    
    async def predict_optimal_timing(
        self,
        commodity: str,
        arrival_forecast: List[Dict],
        price_forecast: List[Dict]
    ) -> Dict:
        """
        Determine optimal time to sell based on price and supply
        """
        
        # Calculate supply-demand score for each day
        timing_data = []
        for i in range(min(len(arrival_forecast), len(price_forecast))):
            arrival = arrival_forecast[i].get('total_predicted_value', 0)
            price = price_forecast[i].get('predicted_price', 0)
            date = arrival_forecast[i].get('date', '')

            timing_data.append({
                "date": date,
                "price": price,
                "supply": arrival,
                "score": 0  # Will calculate below
            })

        # 🔧 FIX 3: Improved scoring - prioritize highest price
        # Farmers primarily want maximum revenue, supply is secondary consideration
        if timing_data:
            max_arrival = max([d['supply'] for d in timing_data]) if any(d['supply'] > 0 for d in timing_data) else 1

            for day in timing_data:
                price = day['price']
                arrival = day['supply']

                # 80% weight on price, 20% penalty for high supply (if prices are similar)
                # This ensures we recommend highest price day, but prefer lower supply if prices are close
                price_weight = 0.85
                supply_penalty_weight = 0.15
                supply_penalty = (arrival / max_arrival) if max_arrival > 0 else 0

                day['score'] = (price * price_weight) - (price * supply_penalty * supply_penalty_weight)

        # Find best day (highest score = highest price with reasonable supply)
        best_day = max(timing_data, key=lambda x: x['score']) if timing_data else None

        if not best_day:
            return {"error": "No timing data available"}
        
        prompt = f"""Based on this week's forecast data for {commodity}:

{json.dumps(timing_data, indent=2)}

The calculated optimal day is {best_day['date']} with:
- Price: ₹{best_day['price']:.0f}
- Supply: {best_day['supply']:.0f} units
- Score: {best_day['score']:.2f}

Provide strategic timing recommendation:
{{
  "recommended_day": "date",
  "reasoning": "why this day (market dynamics)",
  "expected_benefit": "quantified benefit",
  "alternative_days": ["backup options"],
  "risk_warning": "what could change the plan"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are CropCast AI, a timing optimization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            result = response.choices[0].message.content.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            
            return json.loads(result)
        except Exception as e:
            print(f"❌ Timing prediction error: {e}")
            return {"error": str(e)}
    
    # ================================================================
    # ⚠️ Risk Intelligence
    # ================================================================
    
    async def analyze_risks(
        self,
        commodity: str,
        forecasts: Dict,
        weather: Dict
    ) -> Dict:
        """
        Comprehensive risk analysis
        """
        prompt = f"""Analyze risks for {commodity} farmer based on:

FORECASTS:
{json.dumps(forecasts, indent=2)}

WEATHER:
{json.dumps(weather, indent=2)}

Identify all risks and mitigation strategies:
{{
  "critical_risks": [
    {{
      "risk": "description",
      "probability": "high|medium|low",
      "impact": "financial impact estimate",
      "mitigation": "specific action to take",
      "urgency": "immediate|this_week|monitor"
    }}
  ],
  "overall_risk_level": "high|medium|low",
  "insurance_recommendation": "should consider crop insurance? why?"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are CropCast AI, an agricultural risk analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            
            return json.loads(result)
        except Exception as e:
            print(f"❌ Risk analysis error: {e}")
            return {"error": str(e)}


# ================================================================
# 🎯 Integration Functions for MCP Server
# ================================================================

async def enhance_mcp_response(
    query: str,
    tool_results: Dict,
    context: Dict = None
) -> Dict:
    """
    Main function to enhance MCP server responses with AI intelligence
    
    Usage in mcp_server.py:
    
    # After executing tools
    enhanced = await enhance_mcp_response(
        query=query,
        tool_results=tool_results,
        context=context.extracted_params
    )
    
    return {
        "response": enhanced["response"],
        "insights": enhanced,
        "tool_results": tool_results
    }
    """
    ai = EnhancedAIIntelligence()
    return await ai.generate_smart_insights(query, tool_results, context)


async def get_market_comparison(
    commodity: str,
    markets: List[str]
) -> Dict:
    """Compare markets for best opportunity"""
    ai = EnhancedAIIntelligence()
    return await ai.compare_markets(commodity, markets)


async def get_optimal_timing(
    commodity: str,
    arrival_forecast: List[Dict],
    price_forecast: List[Dict]
) -> Dict:
    """Get optimal selling timing"""
    ai = EnhancedAIIntelligence()
    return await ai.predict_optimal_timing(commodity, arrival_forecast, price_forecast)


async def get_risk_analysis(
    commodity: str,
    forecasts: Dict,
    weather: Dict
) -> Dict:
    """Comprehensive risk analysis"""
    ai = EnhancedAIIntelligence()
    return await ai.analyze_risks(commodity, forecasts, weather)


# ================================================================
# 🧪 Test Example
# ================================================================

if __name__ == "__main__":
    import asyncio
    
    # Example usage
    async def test():
        ai = EnhancedAIIntelligence()
        
        # Simulate tool results
        tool_results = {
            "arrival": {
                "data": {
                    "total_predicted": [
                        {"date": "2025-11-06", "total_predicted_value": 1872},
                        {"date": "2025-11-07", "total_predicted_value": 1899},
                        {"date": "2025-11-08", "total_predicted_value": 1947},
                        {"date": "2025-11-09", "total_predicted_value": 1995},
                        {"date": "2025-11-10", "total_predicted_value": 2032},
                        {"date": "2025-11-11", "total_predicted_value": 2042},
                        {"date": "2025-11-12", "total_predicted_value": 2077}
                    ]
                }
            },
            "price": {
                "data": {
                    "predictions": [
                        {"date": "2025-11-06", "predicted_price": 5200},
                        {"date": "2025-11-07", "predicted_price": 5150},
                        {"date": "2025-11-08", "predicted_price": 5100},
                        {"date": "2025-11-09", "predicted_price": 5050},
                        {"date": "2025-11-10", "predicted_price": 5000},
                        {"date": "2025-11-11", "predicted_price": 4950},
                        {"date": "2025-11-12", "predicted_price": 4900}
                    ]
                }
            }
        }
        
        context = {
            "commodity": "Chilli",
            "location": "Warangal"
        }
        
        result = await ai.generate_smart_insights(
            query="what is the expected no of arrivals next week in Warangal",
            tool_results=tool_results,
            context=context
        )
        
        print("\n" + "="*70)
        print("🧠 AI-ENHANCED RESPONSE")
        print("="*70)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())