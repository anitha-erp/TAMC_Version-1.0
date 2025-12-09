"""
AI Validation Layer for Forecast Predictions
Provides confidence scoring and anomaly detection without changing prediction values
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import logging

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

logging.basicConfig(level=logging.INFO)


def calculate_stats(predictions: List[Dict], historical_data: Optional[List[float]] = None) -> Dict:
    """Calculate statistical metrics for predictions"""
    values = [p.get('predicted_value', 0) for p in predictions]
    
    stats = {
        'mean': np.mean(values) if values else 0,
        'std': np.std(values) if values else 0,
        'min': np.min(values) if values else 0,
        'max': np.max(values) if values else 0,
        'trend': 'stable'
    }
    
    # Calculate trend
    if len(values) >= 2:
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        change_pct = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
        
        if change_pct > 10:
            stats['trend'] = 'increasing'
        elif change_pct < -10:
            stats['trend'] = 'decreasing'
    
    # Add historical context if available
    if historical_data and len(historical_data) > 0:
        stats['historical_mean'] = np.mean(historical_data)
        stats['historical_std'] = np.std(historical_data)
        stats['historical_max'] = np.max(historical_data)
        stats['historical_min'] = np.min(historical_data)
    
    return stats


def detect_anomalies(
    predictions: List[Dict],
    historical_stats: Dict,
    threshold_multiplier: float = 2.0
) -> Dict:
    """
    Detect anomalies in predictions compared to historical data
    
    Returns:
        {
            'has_anomaly': bool,
            'anomalies': List[Dict],
            'severity': str,
            'summary': str
        }
    """
    anomalies = []
    
    if not predictions:
        return {'has_anomaly': False, 'anomalies': [], 'severity': 'none', 'summary': 'No predictions to analyze'}
    
    pred_values = [p.get('predicted_value', 0) for p in predictions]
    pred_mean = np.mean(pred_values)
    
    hist_mean = historical_stats.get('historical_mean', pred_mean)
    hist_std = historical_stats.get('historical_std', 0)
    hist_max = historical_stats.get('historical_max', pred_mean)
    
    # Check 1: Spike Detection (prediction >> historical average)
    if hist_mean > 0:
        ratio = pred_mean / hist_mean
        if ratio > threshold_multiplier:
            anomalies.append({
                'type': 'spike',
                'severity': 'high' if ratio > 3 else 'medium',
                'message': f'Prediction is {ratio:.1f}x higher than historical average',
                'predicted': round(pred_mean, 2),
                'historical': round(hist_mean, 2),
                'ratio': round(ratio, 2)
            })
    
    # Check 2: Drop Detection (prediction << historical average)
    if hist_mean > 0:
        ratio = pred_mean / hist_mean
        if ratio < (1 / threshold_multiplier):
            anomalies.append({
                'type': 'drop',
                'severity': 'high' if ratio < 0.33 else 'medium',
                'message': f'Prediction is {(1-ratio)*100:.0f}% lower than historical average',
                'predicted': round(pred_mean, 2),
                'historical': round(hist_mean, 2),
                'ratio': round(ratio, 2)
            })
    
    # Check 3: Outlier Detection (beyond 3 standard deviations)
    if hist_std > 0:
        z_score = abs(pred_mean - hist_mean) / hist_std
        if z_score > 3:
            anomalies.append({
                'type': 'outlier',
                'severity': 'medium',
                'message': f'Prediction is {z_score:.1f} standard deviations from historical mean',
                'z_score': round(z_score, 2)
            })
    
    # Check 4: Volatility Detection (large day-to-day changes)
    if len(pred_values) >= 2:
        daily_changes = [abs(pred_values[i] - pred_values[i-1]) for i in range(1, len(pred_values))]
        max_change = max(daily_changes) if daily_changes else 0
        avg_value = np.mean(pred_values)
        
        if avg_value > 0:
            volatility_pct = (max_change / avg_value) * 100
            if volatility_pct > 50:
                anomalies.append({
                    'type': 'volatility',
                    'severity': 'low',
                    'message': f'High day-to-day volatility detected ({volatility_pct:.0f}% change)',
                    'max_change': round(max_change, 2),
                    'volatility_pct': round(volatility_pct, 2)
                })
    
    # Determine overall severity
    severity = 'none'
    if anomalies:
        severities = [a['severity'] for a in anomalies]
        if 'high' in severities:
            severity = 'high'
        elif 'medium' in severities:
            severity = 'medium'
        else:
            severity = 'low'
    
    # Create summary
    summary = 'No anomalies detected'
    if anomalies:
        summary = f"{len(anomalies)} anomal{'y' if len(anomalies) == 1 else 'ies'} detected: " + \
                  ", ".join([a['type'] for a in anomalies])
    
    return {
        'has_anomaly': len(anomalies) > 0,
        'anomalies': anomalies,
        'severity': severity,
        'summary': summary
    }


def calculate_confidence_score(
    predictions: List[Dict],
    historical_stats: Dict,
    weather: Optional[Dict] = None,
    commodity: Optional[str] = None
) -> Dict:
    """
    Use AI to calculate confidence score for predictions
    """
    # ... implementation ...

    """
    Use AI to calculate confidence score for predictions
    
    Returns:
        {
            'confidence_score': int (0-100),
            'confidence_level': str (low/medium/high),
            'reasoning': str,
            'factors': Dict
        }
    """
    
    # Calculate basic statistics
    pred_stats = calculate_stats(predictions)
    
    # Build prompt for AI
    prompt = f"""Analyze this agricultural forecast prediction and provide a confidence score (0-100).

PREDICTION DATA:
- Commodity: {commodity or 'Unknown'}
- Forecast Period: {len(predictions)} days
- Average Predicted Value: {pred_stats['mean']:.0f}
- Trend: {pred_stats['trend']}
- Range: {pred_stats['min']:.0f} to {pred_stats['max']:.0f}

HISTORICAL CONTEXT:
- Historical Average: {historical_stats.get('historical_mean', 0):.0f}
- Historical Std Dev: {historical_stats.get('historical_std', 0):.0f}
- Historical Range: {historical_stats.get('historical_min', 0):.0f} to {historical_stats.get('historical_max', 0):.0f}

WEATHER CONDITIONS:
"""
    
    if weather:
        prompt += f"""- Condition: {weather.get('condition', 'Unknown')}
- Rain: {weather.get('rain_mm', 0)}mm
- Temperature: {weather.get('temp_c', 0)}°C
"""
    else:
        prompt += "- No weather data available\n"
    
    prompt += """
ANALYSIS REQUIRED:
Provide a JSON response with:
1. confidence_score (0-100): How reliable is this prediction?
2. confidence_level (low/medium/high)
3. reasoning (1-2 sentences): Why this confidence level?
4. factors: {
     "data_quality": 0-100,
     "weather_impact": 0-100,
     "volatility": 0-100
   }

Consider:
- Is prediction within reasonable range of historical data?
- How stable is the trend?
- Weather impact on reliability
- Data quality and completeness

Return ONLY valid JSON, no markdown.
"""
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an agricultural data analyst. Provide confidence assessments for forecast predictions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        
        confidence_data = json.loads(result_text)
        
        return confidence_data
        
    except Exception as e:
        logging.error(f"AI confidence scoring error: {e}")
        # Fallback to rule-based confidence
        return _fallback_confidence(pred_stats, historical_stats)


def _fallback_confidence(pred_stats: Dict, historical_stats: Dict) -> Dict:
    """Fallback rule-based confidence when AI fails"""
    
    # Simple rule-based confidence
    confidence_score = 70  # Default medium
    
    # Adjust based on historical comparison
    if historical_stats.get('historical_mean', 0) > 0:
        ratio = pred_stats['mean'] / historical_stats['historical_mean']
        if 0.8 <= ratio <= 1.2:
            confidence_score += 15  # Close to historical average
        elif ratio > 2 or ratio < 0.5:
            confidence_score -= 20  # Far from historical average
    
    # Adjust based on trend stability
    if pred_stats['trend'] == 'stable':
        confidence_score += 10
    
    confidence_score = max(0, min(100, confidence_score))
    
    if confidence_score >= 75:
        level = 'high'
    elif confidence_score >= 50:
        level = 'medium'
    else:
        level = 'low'
    
    return {
        'confidence_score': confidence_score,
        'confidence_level': level,
        'reasoning': 'Rule-based confidence assessment (AI unavailable)',
        'factors': {
            'data_quality': 70,
            'weather_impact': 70,
            'volatility': 70
        }
    }


def validate_forecast(
    predictions: List[Dict],
    historical_data: Optional[List[float]] = None,
    weather: Optional[Dict] = None,
    commodity: Optional[str] = None,
    forecast_type: str = "arrival"
) -> Dict:
    """
    Main validation function - combines confidence scoring and anomaly detection
    
    Args:
        predictions: List of prediction dictionaries
        historical_data: List of historical values for comparison
        weather: Weather data dictionary
        commodity: Commodity name
        forecast_type: 'arrival' or 'price'
    
    Returns:
        {
            'confidence': {...},
            'anomalies': {...},
            'validation_summary': str
        }
    """
    
    # Calculate historical statistics
    historical_stats = {}
    if historical_data:
        historical_stats = {
            'historical_mean': np.mean(historical_data),
            'historical_std': np.std(historical_data),
            'historical_max': np.max(historical_data),
            'historical_min': np.min(historical_data)
        }
    
    # Run anomaly detection
    anomaly_result = detect_anomalies(predictions, historical_stats)
    
    # Run AI confidence scoring
    confidence_result = calculate_confidence_score(
        predictions,
        historical_stats,
        weather,
        commodity
    )
    
    # Create validation summary
    summary_parts = []
    summary_parts.append(f"Confidence: {confidence_result['confidence_level'].upper()} ({confidence_result['confidence_score']}%)")
    
    if anomaly_result['has_anomaly']:
        summary_parts.append(f"⚠️ {anomaly_result['summary']}")
    else:
        summary_parts.append("✓ No anomalies detected")
    
    validation_summary = " | ".join(summary_parts)
    
    return {
        'confidence': confidence_result,
        'anomalies': anomaly_result,
        'validation_summary': validation_summary,
        'forecast_type': forecast_type
    }
