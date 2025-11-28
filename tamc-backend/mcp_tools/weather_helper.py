"""
Weather Helper Module for Price and Arrival Predictions

This module provides a simple interface to get weather data as input features
for price and arrival prediction models using WeatherAPI.com.

Usage:
    from weather_helper import get_weather_input
    
    weather_data = get_weather_input(
        commodity="Chilli",
        district="Warangal",
        date="2025-11-21"
    )
    
    # Use weather_data in your prediction model
    prediction = model.predict(weather_data['features'])
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import requests
import os
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


class WeatherHelper:
    """Helper class to provide weather data for prediction models"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the weather helper with API key"""
        self.api_key = api_key or WEATHER_API_KEY
        self.base_url = "http://api.weatherapi.com/v1"
    
    def _fetch_weather_data(self, location: str, date: str) -> Dict[str, Any]:
        """
        Fetch weather data from WeatherAPI.com
        
        Args:
            location: Location name (district)
            date: Date in YYYY-MM-DD format
        
        Returns:
            Weather data dictionary or None if error
        """
        if not self.api_key:
            return None
        
        try:
            # Check if date is in the past or future
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            today = datetime.now().date()
            days_diff = (target_date - today).days
            
            # Use forecast API for future dates (up to 14 days)
            # Use history API for past dates (up to 7 days back on free plan)
            if days_diff > 0 and days_diff <= 14:
                # Future forecast
                url = f"{self.base_url}/forecast.json"
                params = {
                    "key": self.api_key,
                    "q": location,
                    "days": min(days_diff + 1, 14),
                    "aqi": "no"
                }
            elif days_diff < 0 and days_diff >= -7:
                # Historical data
                url = f"{self.base_url}/history.json"
                params = {
                    "key": self.api_key,
                    "q": location,
                    "dt": date
                }
            elif days_diff == 0:
                # Today's forecast
                url = f"{self.base_url}/forecast.json"
                params = {
                    "key": self.api_key,
                    "q": location,
                    "days": 1,
                    "aqi": "no"
                }
            else:
                # Too far in future or past
                return None
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract the specific day's data
            if "forecast" in data and "forecastday" in data["forecast"]:
                for day in data["forecast"]["forecastday"]:
                    if day["date"] == date:
                        return day
            
            return None
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_weather_input(
        self,
        district: str,
        date: str,
        commodity: Optional[str] = None,
        include_adjustment: bool = True
    ) -> Dict[str, Any]:
        """
        Get weather data formatted as input for prediction models.
        
        Args:
            district: District name (e.g., "Warangal", "Khammam")
            date: Date in YYYY-MM-DD format
            commodity: Commodity name (required if include_adjustment=True)
            include_adjustment: Whether to include commodity-specific adjustment
        
        Returns:
            Dictionary containing:
                - features: Dict of weather features for ML models
                - adjustment_factor: Price adjustment factor (if commodity provided)
                - summary: Human-readable weather summary
                - metadata: Additional context information
        
        Example:
            >>> helper = WeatherHelper()
            >>> data = helper.get_weather_input("Warangal", "2025-11-21", "Chilli")
            >>> print(data['features']['max_temp'])
            31.1
            >>> print(data['adjustment_factor'])
            1.0
        """
        try:
            # Fetch weather data
            weather_data = self._fetch_weather_data(district, date)
            
            if weather_data:
                day_data = weather_data.get("day", {})
                
                # Prepare feature dictionary for ML models
                features = {
                    'max_temp': day_data.get('maxtemp_c', 0.0),
                    'min_temp': day_data.get('mintemp_c', 0.0),
                    'rainfall': day_data.get('totalprecip_mm', 0.0),
                    'temp_anomaly': 0.0,  # Would need historical data to calculate
                    'rain_anomaly': 0.0,   # Would need historical data to calculate
                    'heat_stress_risk': 1 if day_data.get('maxtemp_c', 0) > 38 else 0,
                    'enso_index': 0.0,     # Would need ENSO data integration
                }
                
                # Create summary
                condition = day_data.get('condition', {}).get('text', 'Unknown')
                summary = f"{condition}, {features['max_temp']}°C max, {features['rainfall']}mm rain"
            else:
                # Default values if weather data unavailable
                features = {
                    'max_temp': 30.0,
                    'min_temp': 20.0,
                    'rainfall': 0.0,
                    'temp_anomaly': 0.0,
                    'rain_anomaly': 0.0,
                    'heat_stress_risk': 0,
                    'enso_index': 0.0,
                }
                summary = 'Weather data unavailable'
            
            # Initialize result
            result_data = {
                'features': features,
                'summary': summary,
                'metadata': {
                    'district': district,
                    'date': date,
                    'data_source': 'weatherapi.com',
                    'forecast_type': 'daily'
                }
            }
            
            # Calculate commodity-specific adjustment if requested
            if include_adjustment and commodity:
                adjustment_data = self._calculate_weather_adjustment(
                    features, commodity
                )
                result_data['adjustment_factor'] = adjustment_data['adjustment_factor']
                result_data['adjustment_message'] = adjustment_data['message']
                result_data['metadata']['commodity'] = commodity
            else:
                result_data['adjustment_factor'] = 1.0
                result_data['adjustment_message'] = 'No commodity-specific adjustment'
            
            return result_data
            
        except Exception as e:
            # Return default values on error
            return {
                'features': {
                    'max_temp': 30.0,
                    'min_temp': 20.0,
                    'rainfall': 0.0,
                    'temp_anomaly': 0.0,
                    'rain_anomaly': 0.0,
                    'heat_stress_risk': 0,
                    'enso_index': 0.0,
                },
                'adjustment_factor': 1.0,
                'adjustment_message': f'Error fetching weather data: {str(e)}',
                'summary': 'Weather data unavailable',
                'metadata': {
                    'district': district,
                    'date': date,
                    'error': str(e)
                }
            }
    
    def _calculate_weather_adjustment(
        self,
        features: Dict[str, Any],
        commodity: str
    ) -> Dict[str, Any]:
        """
        Calculate commodity-specific weather adjustment factor.
        
        Args:
            features: Weather features dictionary
            commodity: Commodity name
        
        Returns:
            Dictionary with adjustment_factor and message
        """
        adjustment = 1.0
        messages = []
        
        commodity_lower = commodity.lower()
        
        # Commodity-specific weather adjustments
        if 'chilli' in commodity_lower:
            # Heavy rain is bad for chilli
            if features['rainfall'] > 10:
                adjustment *= 0.92  # -8% due to heavy rain
                messages.append(f"Heavy rainfall ({features['rainfall']}mm) may damage chilli crops")
            
            # Extreme heat is also bad
            if features['max_temp'] > 40:
                adjustment *= 0.95  # -5% due to heat stress
                messages.append(f"Extreme heat ({features['max_temp']}°C) may stress chilli plants")
        
        elif 'cotton' in commodity_lower:
            # Cotton needs moderate rain
            if features['rainfall'] < 2:
                adjustment *= 0.94  # -6% due to drought
                messages.append(f"Low rainfall ({features['rainfall']}mm) may reduce cotton yield")
            elif features['rainfall'] > 20:
                adjustment *= 0.96  # -4% due to excess rain
                messages.append(f"Excessive rainfall ({features['rainfall']}mm) may affect cotton quality")
        
        elif 'groundnut' in commodity_lower:
            # Groundnut sensitive to waterlogging
            if features['rainfall'] > 15:
                adjustment *= 0.90  # -10% due to waterlogging risk
                messages.append(f"Heavy rain ({features['rainfall']}mm) increases waterlogging risk for groundnut")
        
        elif 'turmeric' in commodity_lower:
            # Turmeric needs consistent moisture
            if features['rainfall'] < 3:
                adjustment *= 0.96  # -4% due to low moisture
                messages.append(f"Low rainfall ({features['rainfall']}mm) may affect turmeric growth")
        
        # General ENSO effects
        if features['enso_index'] == 1:  # El Niño
            adjustment *= 0.97
            messages.append("El Niño conditions may reduce rainfall")
        elif features['enso_index'] == -1:  # La Niña
            adjustment *= 1.02
            messages.append("La Niña conditions may increase rainfall")
        
        message = " | ".join(messages) if messages else "Weather conditions neutral"
        
        return {
            'adjustment_factor': round(adjustment, 4),
            'message': message
        }
    
    def get_weather_features_array(
        self,
        district: str,
        date: str
    ) -> list:
        """
        Get weather features as a simple array for ML models.
        
        Args:
            district: District name
            date: Date in YYYY-MM-DD format
        
        Returns:
            List of weather features in order:
            [max_temp, min_temp, rainfall, temp_anomaly, rain_anomaly, 
             heat_stress_risk, enso_index]
        
        Example:
            >>> helper = WeatherHelper()
            >>> features = helper.get_weather_features_array("Warangal", "2025-11-21")
            >>> print(features)
            [31.1, 19.5, 0.0, -0.9, -5.0, 1, 0.0]
        """
        data = self.get_weather_input(district, date, include_adjustment=False)
        features = data['features']
        
        return [
            features['max_temp'],
            features['min_temp'],
            features['rainfall'],
            features['temp_anomaly'],
            features['rain_anomaly'],
            features['heat_stress_risk'],
            features['enso_index']
        ]
    
    def get_batch_weather_input(
        self,
        requests: list[Dict[str, str]]
    ) -> list[Dict[str, Any]]:
        """
        Get weather data for multiple district-date combinations.
        
        Args:
            requests: List of dicts with 'district', 'date', and optional 'commodity'
        
        Returns:
            List of weather data dictionaries
        
        Example:
            >>> helper = WeatherHelper()
            >>> requests = [
            ...     {'district': 'Warangal', 'date': '2025-11-21', 'commodity': 'Chilli'},
            ...     {'district': 'Khammam', 'date': '2025-11-22', 'commodity': 'Turmeric'}
            ... ]
            >>> results = helper.get_batch_weather_input(requests)
        """
        results = []
        for req in requests:
            district = req.get('district')
            date = req.get('date')
            commodity = req.get('commodity')
            
            if district and date:
                weather_data = self.get_weather_input(
                    district=district,
                    date=date,
                    commodity=commodity,
                    include_adjustment=bool(commodity)
                )
                results.append(weather_data)
        
        return results


# Convenience function for quick access
def get_weather_input(
    district: str,
    date: str,
    commodity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to get weather data without instantiating WeatherHelper.
    
    Args:
        district: District name
        date: Date in YYYY-MM-DD format
        commodity: Optional commodity name for adjustment factor
    
    Returns:
        Dictionary with weather features and adjustment data
    
    Example:
        >>> from weather_helper import get_weather_input
        >>> weather = get_weather_input("Warangal", "2025-11-21", "Chilli")
        >>> print(weather['features']['max_temp'])
        31.1
    """
    helper = WeatherHelper()
    return helper.get_weather_input(district, date, commodity)


def get_weather_features_array(district: str, date: str) -> list:
    """
    Convenience function to get weather features as an array.
    
    Args:
        district: District name
        date: Date in YYYY-MM-DD format
    
    Returns:
        List of weather feature values
    
    Example:
        >>> from weather_helper import get_weather_features_array
        >>> features = get_weather_features_array("Warangal", "2025-11-21")
        >>> print(features)
        [31.1, 19.5, 0.0, -0.9, -5.0, 1, 0.0]
    """
    helper = WeatherHelper()
    return helper.get_weather_features_array(district, date)