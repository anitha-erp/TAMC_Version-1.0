"""
Enhanced Agricultural Forecasting System with Telangana Data Integration
Version 4.0 - Full Updated Code
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict, Counter
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pymysql
import pandas as pd
import numpy as np
import logging
import warnings
from datetime import datetime, timedelta
import os
import requests
from dotenv import load_dotenv
import re
import dateutil.parser
import google.generativeai as genai
import asyncio
import threading
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Deep Learning imports
import keras
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Import cultivation module


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from normalizer import clean_amc, clean_district, clean_commodity, clean_mandal

# Import AI validation module
try:
    from ai_validation import validate_forecast
    AI_VALIDATION_ENABLED = True
except ImportError:
    logging.warning("AI validation module not found. Validation features will be disabled.")
    AI_VALIDATION_ENABLED = False

# Ensure project root (tamc-backend) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ✅ Absolute imports (work when running directly)
from scrapers.cultivation_scrape import CultivationDataService
from scrapers.telangana_integration_service import (
    get_telangana_service,
    get_telangana_arrivals,
    get_telangana_stats,
    get_telangana_trend,
    get_telangana_data_info
)

from weather_helper import get_weather_input

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

app = FastAPI(title="Enhanced LSTM+GRU Agricultural Forecasting System with Telangana Integration", version="4.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
genai.configure(api_key=GEMINI_API_KEY)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

chat_sessions = defaultdict(list)
prediction_cache = {}
prediction_cache_lock = threading.Lock()
conversation_state: Dict[str, Dict] = {}

METRIC_OPTIONS = [
    {"metric": "number_of_arrivals", "label": "Arrivals (count)", "keywords": ["arrival", "arrivals"]},
    {"metric": "total_bags", "label": "Bags", "keywords": ["bag", "bags"]},
    {"metric": "number_of_lots", "label": "Lots", "keywords": ["lot", "lots"]},
    {"metric": "total_weight", "label": "Quantity (quintals)", "keywords": ["quintal", "quantity", "weight", "qty"]},
    {"metric": "number_of_farmers", "label": "Farmers", "keywords": ["farmer", "farmers"]},
    {"metric": "total_revenue", "label": "Revenue", "keywords": ["revenue", "income", "earning", "amount"]},
]

PREDICTION_KEYWORDS = (
    "predict",
    "forecast",
    "arrival",
    "arrivals",
    "bags",
    "lots",
    "quantity",
    "quintal",
    "weight",
    "farmer",
    "revenue",
    "tomorrow",
    "next",
    "expect"
)

COMMODITY_SUGGESTION_LIMIT = 5


def _ensure_session_state(session_id: str) -> Dict:
    if session_id not in conversation_state:
        conversation_state[session_id] = {
            "pending_request": None,
            "awaiting_metric": False,
            "awaiting_commodity": False,
            "commodity_options": None,
            "commodity_term": None
        }
    return conversation_state[session_id]


def _clear_session_state(state: Dict):
    state["pending_request"] = None
    state["awaiting_metric"] = False
    state["awaiting_commodity"] = False
    state["commodity_options"] = None
    state["commodity_term"] = None


def _looks_like_new_prediction_query(message_lower: str) -> bool:
    tokens = [t for t in re.split(r"\W+", message_lower) if t]
    return len(tokens) > 3 and any(keyword in message_lower for keyword in PREDICTION_KEYWORDS)


def _format_metric_prompt(location: Optional[str]) -> str:
    header = "Which value should I forecast"
    if location:
        header += f" for {location}"
    header += "? Reply with the number or name.\n"
    lines = [
        f"{idx}. {option['label']}"
        for idx, option in enumerate(METRIC_OPTIONS, start=1)
    ]
    return header + "\n".join(lines)


def _format_commodity_prompt(term: str, options: List[str]) -> str:
    header = f"I found multiple commodities matching '{term}'. Please pick one:\n"
    lines = [
        f"{idx}. {name}"
        for idx, name in enumerate(options, start=1)
    ]
    return header + "\n".join(lines)


def _resolve_metric_choice(message: str) -> Optional[str]:
    text = message.strip().lower()
    if not text:
        return None
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(METRIC_OPTIONS):
            return METRIC_OPTIONS[idx - 1]["metric"]
    for option in METRIC_OPTIONS:
        if option["label"].lower().startswith(text):
            return option["metric"]
        if any(keyword in text for keyword in option["keywords"]):
            return option["metric"]
    return None


def _resolve_commodity_choice(message: str, options: List[str]) -> Optional[str]:
    text = message.strip().lower()
    if not text:
        return None
    if text.isdigit():
        idx = int(text)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    simple_text = re.sub(r"[^a-z0-9]", "", text)
    for option in options:
        option_lower = option.lower()
        if text in option_lower or option_lower in text:
            return option
        option_simple = re.sub(r"[^a-z0-9]", "", option_lower)
        if simple_text and simple_text in option_simple:
            return option
    return None


async def _fulfill_prediction_request(pred_payload: Dict, session_id: str) -> ChatResponse:
    """Execute prediction and format ChatResponse"""
    pred_request = PredictionRequest(**pred_payload)
    prediction_result = await enhanced_prediction_with_telangana(pred_request)

    if "error" in prediction_result:
        error_response = f"❌ {prediction_result['error']}"
        chat_sessions[session_id].append({"role": "assistant", "content": error_response})
        return ChatResponse(response=error_response, query_type="PREDICTION_ERROR")

    total_predicted = prediction_result.get('total_predicted', [])
    if not total_predicted:
        no_data_response = "❌ No historical data available for this location."
        chat_sessions[session_id].append({"role": "assistant", "content": no_data_response})
        return ChatResponse(response=no_data_response, query_type="NO_DATA")

    formatted_response = response_formatter.format_prediction_response(prediction_result)
    if prediction_result.get('telangana_insight'):
        formatted_response += f"\n\n{prediction_result['telangana_insight']}"

    chat_sessions[session_id].append({"role": "assistant", "content": formatted_response})
    return ChatResponse(
        response=formatted_response,
        prediction_data=prediction_result,
        telangana_insight=prediction_result.get('telangana_insight'),
        query_type="PREDICTION_SUCCESS"
    )


async def _handle_pending_clarifications(message: str, session_id: str, state: Dict) -> Optional[ChatResponse]:
    pending = state.get("pending_request")
    if not pending:
        return None

    # Commodity clarification
    if state.get("awaiting_commodity"):
        options = state.get("commodity_options") or []
        selection = _resolve_commodity_choice(message, options)
        if selection:
            pending["commodity"] = selection
            state["awaiting_commodity"] = False
            state["commodity_options"] = None
            state["commodity_term"] = None
        else:
            prompt = _format_commodity_prompt(state.get("commodity_term") or "your commodity", options or ["commodity"])
            chat_sessions[session_id].append({"role": "assistant", "content": prompt})
            return ChatResponse(response=prompt, query_type="CLARIFICATION")

    # Metric clarification
    if state.get("awaiting_metric"):
        metric_choice = _resolve_metric_choice(message)
        if metric_choice:
            pending["metric"] = metric_choice
            state["awaiting_metric"] = False
        else:
            prompt = _format_metric_prompt(pending.get("district") or pending.get("amc_name"))
            chat_sessions[session_id].append({"role": "assistant", "content": prompt})
            return ChatResponse(response=prompt, query_type="CLARIFICATION")

    if state.get("awaiting_metric") or state.get("awaiting_commodity"):
        return None

    # All clarifications resolved
    state["pending_request"] = None
    response = await _fulfill_prediction_request(pending, session_id)
    _clear_session_state(state)
    return response

# ========== Helper Functions ==========
def _apply_weather_factor_to_arrivals(predictions, commodity, district):
    """
    Fetch and store weather factors ONLY.
    Do NOT apply multipliers here.
    """
    for pred in predictions:
        try:
            weather_data = get_weather_input(
                district=district,
                date=pred["date"],
                commodity=commodity
            )

            raw_weather_factor = weather_data["adjustment_factor"]

            # Clamp to ±10%
            weather_factor = max(0.90, min(1.10, raw_weather_factor))

            pred["weather_factor"] = round(weather_factor, 4)
            pred["weather_impact"] = weather_data.get("adjustment_message", "")

            pred["baseline_value"] = pred.get("predicted_value", 0)

        except Exception as e:
            pred["weather_factor"] = 1.0
            pred["weather_impact"] = "Weather data unavailable"

    return predictions

def get_telangana_cultivation_factor(commodity: str, district: str = None) -> float:
    """
    Get cultivation factor from Telangana arrival trends
    
    Returns:
        float: Impact factor between 0.8 and 1.2
    """
    try:
        # Get recent trend
        trend_data = get_telangana_trend(commodity, market_name=district, days=14)
        
        if not trend_data:
            return 1.0
        
        # Convert trend to factor
        trend_type = trend_data.get('trend', 'stable')
        change_percent = trend_data.get('change_percent', 0)
        
        if trend_type == 'increasing':
            # Strong increase = more supply expected
            factor = 1.0 + min(change_percent / 100 * 0.5, 0.2)
        elif trend_type == 'decreasing':
            # Strong decrease = less supply expected
            factor = 1.0 - min(abs(change_percent) / 100 * 0.5, 0.2)
        else:
            factor = 1.0
        
        # Clamp between 0.8 and 1.2
        return max(0.8, min(1.2, factor))
        
    except Exception as e:
        logging.error(f"Error getting Telangana factor: {e}")
        return 1.0


# Cache for cultivation factors to avoid repeated HTTP calls
_cultivation_factor_cache = {}
_cultivation_cache_ttl = 3600  # 1 hour

# Cache for seasonal factors
_seasonal_factor_cache = {}
_seasonal_cache_ttl = 86400  # 24 hours

def _analyze_seasonal_patterns(commodity: str, district: str = None) -> Dict[int, float]:
    """
    Dynamically analyze historical arrival patterns by month.
    Returns a dictionary mapping month (1-12) to seasonal multiplier.
    
    Args:
        commodity: Commodity name
        district: Optional district filter
    
    Returns:
        Dict[int, float]: Month -> seasonal multiplier (0.85 to 1.30)
    """
    cache_key = f"seasonal_{commodity}_{district}"
    current_time = datetime.now().timestamp()
    
    # Check cache first
    if cache_key in _seasonal_factor_cache:
        cached_data, cached_time = _seasonal_factor_cache[cache_key]
        age = current_time - cached_time
        if age < _seasonal_cache_ttl:
            logging.info(f"✅ Seasonal cache HIT for {commodity} (age: {age/3600:.1f}h)")
            return cached_data
        else:
            del _seasonal_factor_cache[cache_key]
    
    try:
        conn = get_connection()
        
        # Build query to get monthly averages over past 2 years
        where_conditions = []
        params = []
        
        if commodity:
            where_conditions.append("commodity_name LIKE %s")
            params.append(f"%{commodity}%")
        
        if district:
            where_conditions.append("(district LIKE %s OR amc_name LIKE %s)")
            params.extend([f"%{district}%", f"%{district}%"])
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        query = f"""
            SELECT 
                MONTH(created_at) as month,
                COUNT(*) as arrival_count,
                SUM(no_of_bags) as total_bags
            FROM lots_new
            WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 730 DAY)
              AND {where_clause}
            GROUP BY MONTH(created_at)
            ORDER BY month
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
        
        conn.close()
        
        if not results or len(results) < 6:  # Need at least 6 months of data
            logging.warning(f"⚠️ Insufficient seasonal data for {commodity}, using neutral factors")
            return {month: 1.0 for month in range(1, 13)}
        
        # Calculate monthly averages
        monthly_data = {}
        total_arrivals = 0
        
        for row in results:
            month = row['month']
            count = row['arrival_count'] or 0
            monthly_data[month] = count
            total_arrivals += count
        
        # Calculate overall average
        num_months = len(monthly_data)
        overall_avg = total_arrivals / num_months if num_months > 0 else 1
        
        if overall_avg == 0:
            return {month: 1.0 for month in range(1, 13)}
        
        # Calculate seasonal multipliers
        seasonal_factors = {}
        for month in range(1, 13):
            if month in monthly_data:
                # Raw multiplier = month_avg / overall_avg
                raw_multiplier = monthly_data[month] / overall_avg
                
                # Clamp to 0.85 - 1.30 range
                clamped = max(0.85, min(1.30, raw_multiplier))
                seasonal_factors[month] = round(clamped, 3)
            else:
                # No data for this month, use neutral
                seasonal_factors[month] = 1.0
        
        # Cache the results
        _seasonal_factor_cache[cache_key] = (seasonal_factors, current_time)
        
        logging.info(
            f"📊 Seasonal analysis for {commodity}: "
            f"Peak months: {[m for m, f in seasonal_factors.items() if f > 1.1]}, "
            f"Low months: {[m for m, f in seasonal_factors.items() if f < 0.95]}"
        )
        
        return seasonal_factors
        
    except Exception as e:
        logging.error(f"Error analyzing seasonal patterns: {e}")
        # Return neutral factors on error
        return {month: 1.0 for month in range(1, 13)}


def _get_seasonal_factor(date_str: str, commodity: str, district: str = None) -> float:
    """
    Get seasonal factor for a specific date and commodity.
    
    Args:
        date_str: Date in 'YYYY-MM-DD' format
        commodity: Commodity name
        district: Optional district filter
    
    Returns:
        float: Seasonal multiplier (0.85 to 1.30)
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        month = date_obj.month
        
        seasonal_patterns = _analyze_seasonal_patterns(commodity, district)
        return seasonal_patterns.get(month, 1.0)
        
    except Exception as e:
        logging.error(f"Error getting seasonal factor: {e}")
        return 1.0


def _apply_cultivation_factor_enhanced(
    predictions: list,
    commodity: str,
    district: str | None
) -> list:
    """
    Enhanced version with BALANCED K-factors:
    - Cultivation softened (50% influence)
    - Telangana trend softened (40% influence)
    - Final factor = multiplicative (not weighted average)
    """
    if not predictions:
        return predictions
    
    try:
        cache_key = f"{commodity}_{district}"
        current_time = datetime.now().timestamp()
        
        # -----------------------
        # 1. CHECK CACHE
        # -----------------------
        if cache_key in _cultivation_factor_cache:
            cached_data, cached_time = _cultivation_factor_cache[cache_key]
            
            if current_time - cached_time < _cultivation_cache_ttl:
                # RAW values from cache
                raw_cultivation = cached_data['cultivation']
                raw_telangana = cached_data['telangana']
                
                # Apply BALANCED smoothing
                cultivation_factor = (0.5 * raw_cultivation) + 0.5
                telangana_factor = 1.0 + (raw_telangana - 1.0) * 0.4
                
                combined_factor = cultivation_factor * telangana_factor

            else:
                # Cache expired
                del _cultivation_factor_cache[cache_key]
                cache_key = None
        else:
            cache_key = None
        
        # -----------------------
        # 2. FETCH RAW FACTORS IF NOT CACHED
        # -----------------------
        if cache_key is None:
            # RAW cultivation factor (UPAG)
            try:
                raw_cultivation = (
                    CultivationDataService()
                    .get_cultivation_factor(commodity, district)
                    .get("impact_factor", 1.0)
                )
            except Exception:
                raw_cultivation = 1.0
            
            # RAW Telangana trend factor
            raw_telangana = get_telangana_cultivation_factor(commodity, district)
            
            # Apply BALANCED smoothing
            cultivation_factor = (0.5 * raw_cultivation) + 0.5      # 50% influence
            telangana_factor = (0.6 * 1.0) + (0.4 * raw_telangana)  # 40% influence
            
            # Multiplicative combination (balanced)
            combined_factor = cultivation_factor * telangana_factor
            
            # Cache RAW values (not softened ones)
            _cultivation_factor_cache[f"{commodity}_{district}"] = (
                {
                    'cultivation': raw_cultivation,
                    'telangana': raw_telangana,
                    'combined': combined_factor
                },
                current_time
            )
        
        # -----------------------
        # 3. APPLY TO PREDICTIONS
        # -----------------------
        for p in predictions:
            p["predicted_value"] *= cultivation_factor
            p["predicted_value"] *= telangana_factor
            
            p["cultivation_factor"] = round(cultivation_factor, 3)
            p["telangana_factor"] = round(telangana_factor, 3)
            p["combined_factor"] = round(combined_factor, 3)
        
        logging.info(
            f"🌾 Applied Balanced Factors → "
            f"Cultivation={cultivation_factor:.3f} | "
            f"Telangana={telangana_factor:.3f} | "
            f"Combined={combined_factor:.3f}"
        )
        
        return predictions
    
    except Exception as e:
        logging.error(f"Error applying enhanced cultivation factors: {e}")
        return predictions


# ========== Pydantic Models ==========
class PredictionRequest(BaseModel):
    metric: str = Field(default="number_of_arrivals", description="Metric to predict")
    district: Optional[str] = None
    amc_name: Optional[str] = None
    commodity: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    days: Optional[int] = Field(default=7, description="Number of days to forecast")

class ChatRequest(BaseModel):
    message: str
    session_id: str
    district: Optional[str] = None
    amc_name: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    prediction_data: Optional[Dict] = None
    weather_summary: Optional[Dict] = None
    telangana_insight: Optional[str] = None
    query_type: Optional[str] = None

# ========== Database Configuration ==========
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "cursorclass": pymysql.cursors.DictCursor,
}

def get_connection():
    return pymysql.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        db=DB_CONFIG["database"],
        port=DB_CONFIG["port"],
        cursorclass=DB_CONFIG["cursorclass"]
    )

def make_cache_key(data):
    """Generate unique cache key for predictions including location"""
    district = data.get('district') or 'no_district'
    amc_name = data.get('amc_name') or 'no_amc'
    return (
        f"lstm_{data.get('metric')}_{district}_{amc_name}_"
        f"{data.get('commodity')}_{data.get('start_date')}_{data.get('end_date')}_{data.get('days')}"
    )

# ========== Enhanced Model Cache ==========
class ModelCache:
    """Persistent model cache with automatic invalidation"""
    def __init__(self, cache_dir="./model_cache", max_age_hours=24):
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        self.memory_cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        self.lock = threading.Lock()
    
    def _get_cache_key(self, amc, commodity, metric, district=None):
        """Generate unique cache key including location"""
        district_part = district if district else "no_district"
        key_str = f"{district_part}_{amc}_{commodity}_{metric}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get file path for cached model"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def get(self, amc, commodity, metric, district=None):
        """Retrieve cached model if available and fresh"""
        with self.lock:
            cache_key = self._get_cache_key(amc, commodity, metric, district)
            
            # Check memory cache first
            if cache_key in self.memory_cache:
                model_data, timestamp = self.memory_cache[cache_key]
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                if age_hours < self.max_age_hours:
                    return model_data
                else:
                    del self.memory_cache[cache_key]
            
            # Check disk cache
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                try:
                    file_age = (datetime.now() - datetime.fromtimestamp(
                        os.path.getmtime(cache_path)
                    )).total_seconds() / 3600
                    
                    if file_age < self.max_age_hours:
                        with open(cache_path, 'rb') as f:
                            model_data = pickle.load(f)
                        self.memory_cache[cache_key] = (model_data, datetime.now())
                        return model_data
                    else:
                        os.remove(cache_path)
                except Exception as e:
                    logging.warning(f"Failed to load cached model: {e}")
            
            return None
    
    def set(self, amc, commodity, metric, model_data, district=None):
        """Save model to cache"""
        with self.lock:
            cache_key = self._get_cache_key(amc, commodity, metric, district)
            
            # Save to memory
            self.memory_cache[cache_key] = (model_data, datetime.now())
            
            # Save to disk
            cache_path = self._get_cache_path(cache_key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(model_data, f)
            except Exception as e:
                logging.warning(f"Failed to save model to cache: {e}")
    
    def clear_old_cache(self):
        """Clear old cache files"""
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                file_age = (datetime.now() - datetime.fromtimestamp(
                    os.path.getmtime(filepath)
                )).total_seconds() / 3600
                
                if file_age > self.max_age_hours:
                    os.remove(filepath)
                    logging.info(f"Removed old cache file: {filename}")
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")

# Initialize global model cache
model_cache = ModelCache()

# ========== Optimized LSTM+GRU Forecaster ==========
class OptimizedLSTMGRUForecaster:
    """Optimized forecaster with reduced complexity and batch prediction"""
    
    def __init__(self, sequence_length=10, n_features=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
    
    def build_model(self):
        """Build optimized model with reduced complexity"""
        model = keras.Sequential([
            # Reduced LSTM units
            layers.LSTM(
                32,
                return_sequences=True, 
                input_shape=(self.sequence_length, self.n_features),
                dropout=0.1
            ),
            
            # Single GRU layer
            layers.GRU(
                16,
                return_sequences=False,
                dropout=0.1
            ),
            
            # Simplified dense layers
            layers.Dense(8, activation='relu'),
            layers.Dense(1)
        ])
        
        # Higher learning rate for faster convergence
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.005),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def create_sequences(self, data):
        """Optimized sequence creation"""
        if len(data) <= self.sequence_length:
            return np.array([]), np.array([])
        
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train_fast(self, df, target_col='y', epochs=15, batch_size=32):
        """Fast training with early stopping"""
        values = df[target_col].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        X, y = self.create_sequences(scaled_values)
        
        if len(X) < 10:
            return False
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        
        # Use smaller validation split
        split_idx = max(1, int(len(X) * 0.85))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if self.model is None:
            self.build_model()
        
        # Aggressive early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        try:
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test) if len(X_test) > 0 else None,
                callbacks=[early_stopping] if len(X_test) > 0 else [],
                verbose=0
            )
            self.is_trained = True
            return True
        except Exception as e:
            logging.error(f"Training error: {e}")
            return False
    
    def predict_batch(self, df, target_col='y', periods=7):
        """Batch prediction with drift prevention constraints"""
        if not self.is_trained:
            return []
        
        values = df[target_col].values.reshape(-1, 1)
        scaled_values = self.scaler.transform(values)
        
        if len(scaled_values) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(scaled_values), 1))
            scaled_values = np.vstack([padding, scaled_values])
        
        # Calculate historical statistics for constraints
        historical_mean = values.mean()
        historical_std = values.std()
        recent_mean = values[-30:].mean() if len(values) >= 30 else historical_mean
        last_value = values[-1][0]
        
        # Get last sequence
        last_sequence = scaled_values[-self.sequence_length:].reshape(
            1, self.sequence_length, self.n_features
        )
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for i in range(periods):
            next_pred_scaled = self.model.predict(current_sequence, verbose=0)[0, 0]
            
            # Inverse transform to get actual value
            next_pred = self.scaler.inverse_transform([[next_pred_scaled]])[0, 0]
            
            # CONSTRAINT 1: Limit daily change to ±15%
            if predictions:
                max_change = predictions[-1] * 0.15
                next_pred = np.clip(next_pred, 
                                  predictions[-1] - max_change,
                                  predictions[-1] + max_change)
            else:
                # First prediction: limit change from last historical value
                max_change = last_value * 0.15
                next_pred = np.clip(next_pred,
                                  last_value - max_change,
                                  last_value + max_change)
            
            # CONSTRAINT 2: Keep within reasonable bounds (mean ± 2σ)
            lower_bound = max(0, recent_mean - 2 * historical_std)
            upper_bound = recent_mean + 2 * historical_std
            next_pred = np.clip(next_pred, lower_bound, upper_bound)
            
            # CONSTRAINT 3: Apply mean reversion for longer horizons
            decay_factor = 1.0 - (i * 0.015)  # 1.5% decay per day
            next_pred = next_pred * decay_factor + recent_mean * (1 - decay_factor)
            
            predictions.append(next_pred)
            
            # Update sequence with CONSTRAINED prediction
            next_pred_scaled = self.scaler.transform([[next_pred]])[0, 0]
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred_scaled
        
        return np.array(predictions)

# ========== Parallel Processing Engine ==========
class ParallelForecastEngine:
    """Process multiple AMC-commodity combinations in parallel"""
    
    def __init__(self, max_workers=8):
        self.max_workers = max_workers
    
    def process_single_forecast(self, args):
        """Process single AMC-commodity forecast"""
        amc, commodity, df_subset, metric, forecast_days, weather_info, district = args
        
        try:
            # Check cache first
            cached_model = model_cache.get(amc, commodity, metric, district)
            
            if cached_model:
                forecaster = cached_model['forecaster']
                logging.info(f"✅ Using cached model for {district or 'Unknown'} - {amc} - {commodity}")
                is_cached = True
            else:
                # Train new model
                df_train = df_subset[df_subset['y'] > 0][['date', 'y']].copy()
                
                if len(df_train) < 20:
                    return None
                
                logging.info(f"🔄 Training new model for {district or 'Unknown'} - {amc} - {commodity}")
                forecaster = OptimizedLSTMGRUForecaster(
                    sequence_length=10,
                    n_features=1
                )
                
                # Fast training
                success = forecaster.train_fast(
                    df_train, 
                    target_col='y', 
                    epochs=15,
                    batch_size=32
                )
                
                if not success:
                    return None
                
                # Cache the trained model
                model_cache.set(amc, commodity, metric, {
                    'forecaster': forecaster,
                    'trained_at': datetime.now()
                }, district)
                is_cached = False
            
            # Generate predictions
            df_train = df_subset[df_subset['y'] > 0][['date', 'y']].copy()
            future_values = forecaster.predict_batch(
                df_train, 
                target_col='y', 
                periods=forecast_days
            )
            
            # Create forecast list
            start_pred = datetime.now().date() + timedelta(days=1)
            pred_dates = pd.date_range(start=start_pred, periods=forecast_days)
            
            forecast = []
            for date, value in zip(pred_dates, future_values):
                forecast.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_value': max(float(value), 0)
                })
            
            # Adjust for weather
            adjusted = adjust_predictions_for_weather(forecast, weather_info)
            
            # Apply enhanced cultivation factor
            adjusted = _apply_cultivation_factor_enhanced(
                adjusted,
                commodity,
                district
            )
            
            # 🔧 NEW: AI Validation Layer
            validation_result = None
            if AI_VALIDATION_ENABLED:
                try:
                    # Extract historical values for comparison
                    historical_values = df_train['y'].tolist() if not df_train.empty else None
                    
                    # Run synchronous validation
                    validation_result = validate_forecast(
                        predictions=adjusted,
                        historical_data=historical_values,
                        weather=weather_info,
                        commodity=commodity,
                        forecast_type="arrival"
                    )
                    logging.info(f"✅ AI Validation: {validation_result['validation_summary']}")
                except Exception as e:
                    logging.error(f"AI validation error: {e}")
                    validation_result = None
            
            return {
                'key': f"{amc} - {commodity}",
                'forecast': adjusted,
                'reasoning': f"LSTM+GRU (cached)" if is_cached else "LSTM+GRU (trained)",
                'is_cached': is_cached,
                'ai_validation': validation_result  # NEW: AI validation results
            }
            
        except Exception as e:
            logging.error(f"Forecast error for {amc}-{commodity}: {e}")
            return None
    
    async def process_all_forecasts(self, forecast_args_list):
        """Process all forecasts in parallel"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, self.process_single_forecast, args)
                for args in forecast_args_list
            ]
            
            results = await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]

# ========== Dynamic Parameter Extraction ==========
class ParameterExtractor:
    """Extract location/time/metric/commodity from free text"""
    
    def extract_location(self, message: str) -> tuple:
        """Query DB for best matching district or amc_name"""
        message_lower = message.lower()
        tokens = [t for t in re.split(r"\W+", message_lower) if len(t) > 2]
        if not tokens:
            return None, None

        conn = get_connection()
        try:
            # Try exact match first
            for token in tokens:
                with conn.cursor() as cursor:
                    exact_query = """
                        SELECT DISTINCT district, amc_name
                        FROM lots_new
                        WHERE (district IS NOT NULL AND LOWER(district) = LOWER(%s))
                           OR (amc_name IS NOT NULL AND LOWER(amc_name) = LOWER(%s))
                        LIMIT 1
                    """
                    cursor.execute(exact_query, [token, token])
                    results = cursor.fetchall()
                    
                    if results:
                        row = results[0]
                        district = row.get('district') if row.get('district') else None
                        amc_name = row.get('amc_name') if row.get('amc_name') else None
                        logging.info(f"🎯 Exact location match: district={district}, amc={amc_name}")
                        return district, amc_name
            
            # Fallback to fuzzy match
            for token in tokens:
                with conn.cursor() as cursor:
                    query = """
                        SELECT DISTINCT district, amc_name
                        FROM lots_new
                        WHERE (district IS NOT NULL AND LOWER(district) LIKE %s)
                           OR (amc_name IS NOT NULL AND LOWER(amc_name) LIKE %s)
                        LIMIT 5
                    """
                    pattern = f"%{token}%"
                    cursor.execute(query, [pattern, pattern])
                    results = cursor.fetchall()
                    
                    if results:
                        row = results[0]
                        district = row.get('district') if row.get('district') else None
                        amc_name = row.get('amc_name') if row.get('amc_name') else None
                        logging.info(f"🔍 Fuzzy location match: district={district}, amc={amc_name}")
                        return district, amc_name
            
            logging.warning(f"❌ No location found in message: {message}")
            return None, None
        finally:
            conn.close()
    
    def extract_metric(self, message: str) -> str:
        """Detect metric mentions - return specific metric or default"""
        message_lower = message.lower()

        # Check for SPECIFIC metrics first (skip generic "arrival/arrivals")
        for option in METRIC_OPTIONS:
            if option["metric"] == "number_of_arrivals":
                continue  # Skip generic arrival keywords
            if any(keyword in message_lower for keyword in option["keywords"]):
                return option["metric"]

        # If only generic "arrival/arrivals" mentioned, return number_of_arrivals as fallback
        # But the clarification logic will catch this and show interactive buttons
        if "arrival" in message_lower:
            return "number_of_arrivals"

        # Final fallback
        return 'number_of_arrivals'

    def detect_metric_mentions(self, message: str) -> Dict[str, List[str]]:
        """Return which metrics are explicitly referenced"""
        message_lower = message.lower()
        mentions: Dict[str, List[str]] = {}

        # IMPORTANT: Don't count generic "arrival/arrivals" as a specific metric
        # Only count it if there are specific keywords like "bag", "lot", "quintal", etc.
        for option in METRIC_OPTIONS:
            # Skip checking "number_of_arrivals" keywords (arrival/arrivals)
            # These are too generic and should trigger interactive selection
            if option["metric"] == "number_of_arrivals":
                continue

            hits = [kw for kw in option["keywords"] if kw in message_lower]
            if hits:
                mentions[option["metric"]] = hits
        return mentions
    
    def extract_timeframe(self, message: str) -> tuple:
        """Identify start_date and number of days"""
        message_lower = message.lower()

        match = re.search(r"on ([\d/\-\.]+)", message_lower)
        if match:
            try:
                parsed = dateutil.parser.parse(match.group(1), dayfirst=True).date()
                return parsed.strftime('%Y-%m-%d'), 1
            except:
                pass

        if "tomorrow" in message_lower:
            start_date = (datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
            return start_date, 1

        m = re.search(r"next (\d+) days?", message_lower)
        if m:
            days = int(m.group(1))
            start_date = (datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
            return start_date, days

        if "next week" in message_lower:
            start_date = (datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
            return start_date, 7

        start_date = (datetime.now().date() + timedelta(days=1)).strftime('%Y-%m-%d')
        return start_date, 1
    
    def extract_commodity(self, message: str) -> Optional[str]:
        """Look up distinct commodity names from DB"""
        message_lower = message.lower()
        normalized = re.sub(r"[^a-z0-9 ]", " ", message_lower)
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT commodity_name FROM lots_new WHERE commodity_name IS NOT NULL LIMIT 200;")
                results = cursor.fetchall()
                
                if not results:
                    return None
                
                for row in results:
                    comm = row.get('commodity_name')
                    if not comm:
                        continue
                    comm_lower = comm.lower()
                    if comm_lower in message_lower:
                        return comm
                    clean_comm = re.sub(r"[^a-z0-9 ]", " ", comm_lower)
                    if clean_comm.strip() and clean_comm in normalized:
                        return comm
                return None
        except Exception:
            return None
        finally:
            conn.close()

    def get_commodity_candidates(self, message: str, limit: int = COMMODITY_SUGGESTION_LIMIT) -> List[str]:
        tokens = [t for t in re.split(r"\\W+", message.lower()) if len(t) >= 4]
        if not tokens:
            return []
        conn = get_connection()
        suggestions: List[str] = []
        seen = set()
        try:
            with conn.cursor() as cursor:
                for token in tokens:
                    pattern = f"%{token}%"
                    cursor.execute(
                        """
                        SELECT DISTINCT commodity_name
                        FROM lots_new
                        WHERE commodity_name IS NOT NULL
                          AND LOWER(commodity_name) LIKE %s
                        ORDER BY commodity_name
                        LIMIT 10
                        """,
                        [pattern]
                    )
                    for row in cursor.fetchall():
                        name = row.get("commodity_name")
                        if name and name not in seen:
                            suggestions.append(name)
                            seen.add(name)
                        if len(suggestions) >= limit:
                            return suggestions
        except Exception:
            return suggestions
        finally:
            conn.close()
        return suggestions

param_extractor = ParameterExtractor()

# ========== Response Formatter ==========
class ResponseFormatter:
    def format_prediction_response(self, prediction_data: Dict) -> str:
        """Convert prediction to conversational format"""
        try:
            total_predicted = prediction_data.get('total_predicted', [])
            metric_name = prediction_data.get('metric_name', 'Value')
            weather_summary = prediction_data.get('weather_summary', {})
            cached_count = prediction_data.get('cached_models', 0)
            total_models = len(prediction_data.get('prediction_keys', []))

            if not total_predicted:
                return "❌ Insufficient historical data for this location."

            first_pred = total_predicted[0]
            date_str = datetime.strptime(first_pred['date'], '%Y-%m-%d').strftime('%d %b')
            predicted_value = round(first_pred['total_predicted_value'])

            trend_emoji = "📊"
            trend_text = ""
            if len(total_predicted) > 1:
                total_sum = sum(day['total_predicted_value'] for day in total_predicted)
                avg_daily = total_sum / len(total_predicted)

                if predicted_value > avg_daily * 1.1:
                    trend_emoji = "📈"
                    trend_text = "Strong start expected"
                elif predicted_value < avg_daily * 0.9:
                    trend_emoji = "📉"
                    trend_text = "Starting lower"
                else:
                    trend_text = "Steady trend"

            response_parts = [
                f"{trend_emoji} {metric_name} Forecast",
                f"📅 {date_str}: {predicted_value:,} {metric_name.lower()}"
            ]

            if trend_text:
                response_parts.append(f"📊 Trend: {trend_text}")

            if len(total_predicted) > 1:
                week_total = sum(day['total_predicted_value'] for day in total_predicted[:7])
                response_parts.append(f"📋 Week Total: ~{week_total:,}")

                daily_breakdown = []
                for day in total_predicted[:3]:
                    day_date = datetime.strptime(day['date'], '%Y-%m-%d').strftime('%d/%m')
                    day_value = round(day['total_predicted_value'])
                    daily_breakdown.append(f"{day_date}: {day_value:,}")

                if daily_breakdown:
                    response_parts.append(f"📈 Next 3 days: {' | '.join(daily_breakdown)}")

            if weather_summary and weather_summary.get('rain_mm', 0) > 10:
                rain_mm = weather_summary.get('rain_mm', 0)
                response_parts.append(f"🌧 Weather Alert: {rain_mm:.0f}mm rain expected")
            elif weather_summary and weather_summary.get('rain_mm', 0) > 0:
                response_parts.append("🌤 Weather: Light rain possible")
            else:
                response_parts.append("☀ Weather: Clear conditions")

            weather_factor_summary = prediction_data.get('weather_factor_summary', {})
            heavy = weather_factor_summary.get("heavy", 0)
            moderate = weather_factor_summary.get("moderate", 0)
            if heavy or moderate:
                impact_bits = []
                if heavy:
                    impact_bits.append(f"{heavy} heavy-rain day(s)")
                if moderate:
                    impact_bits.append(f"{moderate} light-rain day(s)")
                response_parts.append(f"☔ Weather impacts: {', '.join(impact_bits)}")

            coverage_days = prediction_data.get('weather_coverage_days') or 0
            if coverage_days and coverage_days < len(total_predicted):
                response_parts.append(
                    f"⚠️ Weather data applied to first {coverage_days} day(s); later days reuse the latest outlook."
                )
            
            # Add cache info
            if cached_count > 0:
                response_parts.append(f"⚡ Fast response: {cached_count}/{total_models} cached models")

            return "\n".join(response_parts)

        except Exception as e:
            logging.error(f"Format error: {e}")
            return "✅ Prediction generated successfully!"

response_formatter = ResponseFormatter()

# ========== Weather Cache (Phase 1 Optimization) ==========
weather_cache = {}
WEATHER_CACHE_TTL = 6 * 3600  # 6 hours in seconds

def _get_weather_cache_key(city: str, num_days: int) -> str:
    """Generate cache key for weather data"""
    return f"weather_{city.lower().strip()}_{num_days}"

def _get_cached_weather(city: str, num_days: int):
    """Get weather from cache if available and not expired"""
    cache_key = _get_weather_cache_key(city, num_days)
    if cache_key in weather_cache:
        cached_data, cached_time = weather_cache[cache_key]
        age = datetime.now().timestamp() - cached_time
        if age < WEATHER_CACHE_TTL:
            logging.info(f"✅ Weather cache HIT for {city} (age: {age/60:.1f} min)")
            return cached_data
        else:
            logging.info(f"⏰ Weather cache EXPIRED for {city} (age: {age/3600:.1f} hours)")
            del weather_cache[cache_key]
    return None

def _set_weather_cache(city: str, num_days: int, data):
    """Store weather data in cache"""
    cache_key = _get_weather_cache_key(city, num_days)
    weather_cache[cache_key] = (data, datetime.now().timestamp())
    logging.info(f"💾 Weather cached for {city} (TTL: 6 hours)")

# ========== Weather Functions ==========
def _fetch_weatherapi(city: str, num_days: int):
    try:
        url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            "key": WEATHER_API_KEY,
            "q": f"{city},India",
            "days": max(1, min(num_days, 14)),
            "aqi": "no",
            "alerts": "no"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        forecast_days = data.get("forecast", {}).get("forecastday", [])
        if not forecast_days:
            return None

        condition_counts = Counter()
        total_rain = 0.0
        total_temp = 0.0
        for day in forecast_days:
            condition = day["day"]["condition"]["text"]
            condition_counts[condition] += 1
            total_rain += day["day"].get("totalprecip_mm", 0.0)
            total_temp += day["day"].get("avgtemp_c", 0.0)
        most_common_condition = condition_counts.most_common(1)[0][0] if condition_counts else "Unknown"
        return {
            "provider": "weatherapi",
            "condition": most_common_condition,
            "rain_mm": round(total_rain, 2),
            "temp_c": round(total_temp / len(forecast_days), 1) if forecast_days else 0,
            "forecast_days": forecast_days,
            "num_days": len(forecast_days)
        }
    except Exception as e:
        logging.error(f"Weather API fetch error: {e}")
        return None


def _fetch_weather_from_wttr(city: str, num_days: int):
    try:
        desired = max(1, min(num_days, 7))
        response = requests.get(
            f"https://wttr.in/{city}",
            params={"format": "j1", "num_of_days": desired},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        weather_days = data.get("weather", [])[:desired]
        if not weather_days:
            return None

        converted = []
        condition_counts = Counter()
        total_rain = 0.0
        total_temp = 0.0
        for day in weather_days:
            hourly = day.get("hourly", [])
            condition = "Unknown"
            if hourly:
                mid = min(4, len(hourly) - 1)
                desc = hourly[mid].get("weatherDesc", [])
                if desc:
                    condition = desc[0].get("value", "Unknown")
            avg_temp = float(day.get("avgtempC") or day.get("maxtempC") or 0)
            precip = 0.0
            for hour in hourly:
                try:
                    precip += float(hour.get("precipMM") or 0)
                except (TypeError, ValueError):
                    continue
            total_rain += precip
            total_temp += avg_temp
            condition_counts[condition] += 1
            converted.append({
                "date": day.get("date"),
                "day": {
                    "condition": {"text": condition},
                    "totalprecip_mm": precip,
                    "avgtemp_c": avg_temp
                }
            })

        most_common_condition = condition_counts.most_common(1)[0][0] if condition_counts else "Unknown"
        return {
            "provider": "wttr",
            "condition": most_common_condition,
            "rain_mm": round(total_rain, 2),
            "temp_c": round(total_temp / len(weather_days), 1) if weather_days else 0,
            "forecast_days": converted,
            "num_days": len(weather_days)
        }
    except Exception as e:
        logging.error(f"wttr fallback weather error: {e}")
        return None


def fetch_multi_day_weather(city, num_days=7):
    """Fetch multi-day weather forecast with fallback blending"""
    if not city:
        return None
    desired_days = max(1, min(num_days, 14))

    # ✅ Phase 1 Optimization: Check cache first
    cached_weather = _get_cached_weather(city, desired_days)
    if cached_weather:
        return cached_weather

    weather_data = None
    if WEATHER_API_KEY:
        weather_data = _fetch_weatherapi(city, desired_days)

    fallback_data = None
    if not weather_data or weather_data.get("num_days", 0) < desired_days:
        fallback_data = _fetch_weather_from_wttr(city, desired_days)

    if weather_data and fallback_data:
        combined = {pd.to_datetime(day["date"]).date(): day for day in weather_data["forecast_days"]}
        for fallback_day in fallback_data["forecast_days"]:
            date_key = pd.to_datetime(fallback_day["date"]).date()
            if date_key not in combined and len(combined) < desired_days:
                combined[date_key] = fallback_day
        sorted_days = [combined[k] for k in sorted(combined.keys())][:desired_days]
        avg_temp = sum(day["day"].get("avgtemp_c", 0.0) for day in sorted_days) / len(sorted_days)
        total_rain = sum(day["day"].get("totalprecip_mm", 0.0) for day in sorted_days)
        condition_counts = Counter(day["day"]["condition"]["text"] for day in sorted_days)
        weather_data.update({
            "forecast_days": sorted_days,
            "num_days": len(sorted_days),
            "temp_c": round(avg_temp, 1),
            "rain_mm": round(total_rain, 2),
            "condition": condition_counts.most_common(1)[0][0] if condition_counts else weather_data.get("condition", "Unknown"),
            "provider": f"{weather_data.get('provider', 'weatherapi')}+wttr"
        })
        # ✅ Phase 1 Optimization: Cache the result
        _set_weather_cache(city, desired_days, weather_data)
        return weather_data

    # ✅ Phase 1 Optimization: Cache the result
    result = weather_data or fallback_data
    if result:
        _set_weather_cache(city, desired_days, result)
    return result

def adjust_predictions_for_weather(predictions_list, weather_info):
    """Adjust predictions based on weather"""
    if not weather_info or "forecast_days" not in weather_info:
        return predictions_list

    try:
        rain_map = {}
        for d in weather_info.get("forecast_days", []):
            try:
                day_date = pd.to_datetime(d.get("date")).date()
                rain_mm = d.get("day", {}).get("totalprecip_mm", 0.0)
                rain_map[day_date] = rain_mm
            except Exception:
                continue

        sorted_dates = sorted(rain_map.keys())

        def lookup_rain(date_obj):
            if not sorted_dates:
                return 0.0
            if date_obj in rain_map:
                return rain_map[date_obj]
            pos = bisect_left(sorted_dates, date_obj)
            if pos == 0:
                return rain_map[sorted_dates[0]]
            if pos >= len(sorted_dates):
                return rain_map[sorted_dates[-1]]
            prev_date = sorted_dates[pos - 1]
            next_date = sorted_dates[pos]
            if abs((date_obj - prev_date).days) <= abs((next_date - date_obj).days):
                return rain_map[prev_date]
            return rain_map[next_date]

        adjusted = []
        for pred in predictions_list:
            pred_copy = pred.copy()
            date_obj = pd.to_datetime(pred['date']).date()
            rain = lookup_rain(date_obj)

            if rain >= 10:
                factor = 0.7
                drop_pct = 30
                weather_factor = "heavy"
            elif rain >= 5:
                factor = 0.85
                drop_pct = 15
                weather_factor = "moderate"
            else:
                factor = 1.0
                drop_pct = 0
                weather_factor = "none"

            pred_copy['predicted_value'] *= factor
            pred_copy['weather_drop_pct'] = drop_pct
            pred_copy['weather_factor'] = factor
            pred_copy['weather_intensity'] = weather_factor
            pred_copy['predicted_value'] = float(pred_copy['predicted_value'])
            adjusted.append(pred_copy)

        return adjusted
    except Exception as e:
        logging.error(f"Weather adjustment error: {e}")
        return predictions_list


def static_reply(msg):
    """Quick static replies"""
    msg = msg.lower().strip()
    responses = {
        "hi": "Hello 👋! How can I help you today?",
        "hello": "Hi there! 😊 What would you like to know?",
        "bye": "Goodbye! Stay safe and have a good harvest!",
        "thanks": "You're welcome!",
        "who are you": "I am your Agri Assistant 🤖 using optimized LSTM+GRU AI models with Telangana market data!",
        "help": "Ask me: 'forecast for tomorrow', 'should I bring bags', etc."
    }
    for keyword, reply in responses.items():
        if keyword in msg:
            return reply
    return None

# ========== Enhanced Prediction Engine ==========
async def enhanced_prediction_with_telangana(params: PredictionRequest) -> Dict:
    """
    Enhanced prediction that incorporates Telangana arrival data
    """
    try:
        start_time = datetime.now()
        data = params.dict()
        
        # 🔥 Cleaned inputs
        data["district"] = clean_district(data.get("district"))
        data["amc_name"] = clean_amc(data.get("amc_name"))
        data["commodity"] = clean_commodity(data.get("commodity"))

        cache_key = make_cache_key(data)
        
        # Check prediction cache
        with prediction_cache_lock:
            if cache_key in prediction_cache:
                cached_result = prediction_cache[cache_key]
                logging.info(f"✅ Returning cached prediction result")
                return cached_result
        
        valid_metrics = {
            "total_revenue", "total_bags", "total_weight",
            "number_of_arrivals", "number_of_lots", "number_of_farmers"
        }
        metric_display_map = {
            "total_revenue": "Total Revenue",
            "total_bags": "Total Bags",
            "total_weight": "Total Weight",
            "number_of_arrivals": "Number of Arrivals",
            "number_of_lots": "Number of Lots",
            "number_of_farmers": "Number of Farmers",
        }

        # Extract parameters
        metric = data.get("metric", "total_bags")
        district = data.get("district")
        amc_name = data.get("amc_name")
        commodity = data.get("commodity")
        days = data.get("days", 7)

        if metric not in valid_metrics:
            return {"error": f"Invalid metric '{metric}'."}

        forecast_days = int(days)

        # Build optimized query
        location_conditions = []
        params_list = []
        
        if district:
            location_conditions.append("district = %s")
            params_list.append(district)
        if amc_name:
            location_conditions.append("amc_name = %s")
            params_list.append(amc_name)
        
        where_clause = ""
        if location_conditions:
            where_clause = " AND (" + " OR ".join(location_conditions) + ")"
        
        if commodity:
            where_clause += " AND commodity_name LIKE %s"
            params_list.append(f"%{commodity}%")

        # Optimized query
        # NOTE: number_of_farmers approximated as count of lots (1 lot ≈ 1 farmer/group)
        # NOTE: number_of_arrivals counts individual lot entries
        query = f"""
            SELECT
                DATE(created_at) AS date,
                amc_name,
                commodity_name,
                SUM(CASE WHEN '{metric}' = 'total_bags' THEN no_of_bags
                         WHEN '{metric}' = 'number_of_arrivals' THEN 1
                         WHEN '{metric}' = 'number_of_lots' THEN 1
                         WHEN '{metric}' = 'total_weight' THEN aprox_quantity
                         WHEN '{metric}' = 'number_of_farmers' THEN 1
                         WHEN '{metric}' = 'total_revenue' THEN aprox_quantity * rate_for_qui
                         ELSE 0 END) AS metric_value
            FROM lots_new
            WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 180 DAY)
            {where_clause}
            GROUP BY date, amc_name, commodity_name
            HAVING metric_value > 0
            ORDER BY date DESC
        """

        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                if params_list:
                    cursor.execute(query, params_list)
                else:
                    cursor.execute(query)
                results = cursor.fetchall()
        finally:
            conn.close()

        logging.info(f"📊 Query returned {len(results)} records")

        if not results:
            # Fallback fuzzy search
            conn = get_connection()
            search_location = district or amc_name or ""
            fallback_query = f"""
                SELECT
                    DATE(created_at) AS date,
                    amc_name,
                    commodity_name,
                    SUM(CASE WHEN '{metric}' = 'total_bags' THEN no_of_bags
                             WHEN '{metric}' = 'number_of_arrivals' THEN 1
                             WHEN '{metric}' = 'number_of_lots' THEN 1
                             WHEN '{metric}' = 'total_weight' THEN aprox_quantity
                             WHEN '{metric}' = 'number_of_farmers' THEN 1
                             WHEN '{metric}' = 'total_revenue' THEN aprox_quantity * rate_for_qui
                             ELSE 0 END) AS metric_value
                FROM lots_new
                WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 180 DAY)
                  AND (LOWER(district) LIKE LOWER(%s) OR LOWER(amc_name) LIKE LOWER(%s))
                GROUP BY date, amc_name, commodity_name
                HAVING metric_value > 0
                ORDER BY date DESC
            """
            pattern = f"%{search_location}%"

            try:
                with conn.cursor() as cursor:
                    cursor.execute(fallback_query, [pattern, pattern])
                    results = cursor.fetchall()
            finally:
                conn.close()

            logging.info(f"🔍 Fallback search returned {len(results)} records")

        if not results:
            return {"error": f"No historical data found for '{district or amc_name}'. Please check the spelling."}
        # Convert to DataFrame
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'metric_value': 'y'})

        # Get weather info
        location = district or amc_name or "Hyderabad"
        weather_info = fetch_multi_day_weather(location, forecast_days)
        
        # Optimization: If too many commodities for an aggregate query, group them first
        unique_commodities = df['commodity_name'].nunique()
        if not commodity and unique_commodities > 10:
            logging.info(f"⚡ High commodity count ({unique_commodities}) detected. Switching to Aggregate Mode (BUT keeping commodity-wise data).")
    
            # (A) Build aggregated TOTAL data
            df_agg = df.groupby(['date', 'amc_name'])['y'].sum().reset_index()
            df_agg['commodity_name'] = 'All Commodities'

            # (B) Keep original df for commodity-wise forecasts
            df_original = df.copy()

            # Store aggregate separately (DO NOT replace df)
            df_total = df_agg

            # Build commodity breakdown
            logging.info("⚡ Aggregate Mode Enabled → Building Commodity Breakdown")

            original_df = pd.DataFrame(results)
            original_df['date'] = pd.to_datetime(original_df['date'])
            original_df = original_df.rename(columns={'metric_value': 'y'})

            breakdown = []
            for comm, grp in original_df.groupby("commodity_name"):
                total_val = grp['y'].sum()
                breakdown.append({
                    "commodity": comm,
                    "total_predicted_value": float(round(total_val, 2))
                })

            breakdown.sort(key=lambda x: x['total_predicted_value'], reverse=True)
            commodity_breakdown = breakdown

        else:
            # Normal case → no aggregation
            df_original = df.copy()
            df_total = None


        # Prepare parallel processing
        forecast_engine = ParallelForecastEngine(max_workers=8)

        forecast_args = []

        # (1) Commodity-wise predictions (always needed)
        for (amc, comm), df_subset in df_original.groupby(['amc_name', 'commodity_name']):
            forecast_args.append((amc, comm, df_subset, metric, forecast_days, weather_info, district))

        # Define aggregate_mode correctly
        aggregate_mode = (not commodity) and (unique_commodities > 10)

        # (2) Add TOTAL prediction only when aggregate mode is used
        if aggregate_mode and df_total is not None:
            for amc, df_amc in df_total.groupby('amc_name'):
                forecast_args.append((amc, "All Commodities", df_amc, metric, forecast_days, weather_info, district))

        # Process all forecasts in parallel
        logging.info(f"🚀 Processing {len(forecast_args)} forecasts in parallel...")
        results = await forecast_engine.process_all_forecasts(forecast_args)
        
        # Aggregate results
        predictions = {}
        reasoning = {}
        cached_count = 0
        weather_factor_counts = {"heavy": 0, "moderate": 0, "none": 0}
        
        for result in results:
            predictions[result['key']] = result['forecast']
            reasoning[result['key']] = result['reasoning']
            if result.get('is_cached'):
                cached_count += 1
        
        if not predictions:
            return {"error": "No predictions could be generated. Insufficient data."}
        
        # Calculate totals
        date_totals = defaultdict(float)
        date_weather_drop_totals = defaultdict(float)
        commodity_totals = defaultdict(lambda: defaultdict(float))
        
        for key, forecast_list in predictions.items():
            amc_name_val, commodity_val = key.split(" - ", 1) if " - " in key else ("Unknown", key)
            
            # 🔧 CRITICAL FIX: Skip "All Commodities" to prevent double-counting
            # "All Commodities" is already the sum of all individual commodities
            # Adding it to date_totals would double the total!
            is_aggregate = commodity_val == "All Commodities"
            
            for record in forecast_list:
                date = record['date']
                pred_val = record['predicted_value']
                drop_pct = record.get('weather_drop_pct', 0)
                intensity = record.get('weather_intensity', 'none')
                
                # Update totals - SKIP aggregate to avoid double-counting
                if not is_aggregate:
                    date_totals[date] += pred_val
                
                # Always update commodity_totals (for breakdown display)
                commodity_totals[commodity_val][date] += pred_val
                
                if intensity not in weather_factor_counts:
                    intensity = 'none'
                weather_factor_counts[intensity] += 1
        
        telangana_data = None
        telangana_stats = None
        telangana_trend = None
        telangana_insight = None
        
        if commodity:
            telangana_data = get_telangana_arrivals(commodity, district, days=60)
            
            if telangana_data:
                logging.info(f"✅ Found Telangana data for {commodity}")
                telangana_stats = get_telangana_stats(commodity, district)
                telangana_trend = get_telangana_trend(commodity, district, days=7)
                
                if telangana_trend:
                    trend_text = telangana_trend.get('trend', 'stable')
                    change_pct = telangana_trend.get('change_percent', 0)
                    telangana_insight = (
                        f"📊 Telangana Market Trend: {trend_text.capitalize()} "
                        f"({change_pct:+.1f}% over 7 days)"
                    )
            else:
                logging.info(f"ℹ️ No Telangana data available for {commodity}")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"⏱️ Prediction completed in {elapsed_time:.2f} seconds")
        
        # Prepare response variables
        total_daily = dict(date_totals)
        
        # 🔧 FIX: Exclude "All Commodities" from commodity_daily breakdown
        # It should only appear in the total, not as a separate commodity line
        commodity_daily = {
            k: dict(v) 
            for k, v in commodity_totals.items() 
            if k != "All Commodities"  # Filter out the aggregate
        }
        metric_name = metric_display_map.get(metric, metric)

        # 🔧 Normalize commodity_daily for frontend (always array)
        for comm, val in commodity_daily.items():
            # Case 1: dict → convert to list of {date,predicted_value}
            if isinstance(val, dict):
                commodity_daily[comm] = [
                    {"date": d, "predicted_value": v}
                    for d, v in sorted(val.items())
                ]

            # Case 2: value is single number (aggregate mode)
            elif isinstance(val, (int, float)):
                commodity_daily[comm] = [
                    {
                        "date": rec["date"],
                        "predicted_value": rec["total_predicted_value"]
                    }
                    for rec in total_predicted
                ]

        # Extract AI validation from forecasts (use first available)
        ai_validation_result = None
        for forecast_list in predictions.values():
            if forecast_list and isinstance(forecast_list, list):
                for forecast_item in forecast_list:
                    if isinstance(forecast_item, dict) and 'ai_validation' in forecast_item:
                        ai_validation_result = forecast_item['ai_validation']
                        break
                if ai_validation_result:
                    break
        
        response = {
            "commodity_predictions": predictions,
            "prediction_keys": list(predictions.keys()),
            "total_predicted": total_daily,
            "metric_name": metric_name,
            "reasoning": reasoning,
            "overall_reasoning": f"Enhanced LSTM+GRU forecast with Telangana data for {metric_name}",
            "weather_summary": weather_info,
            "model_type": "LSTM+GRU (Enhanced)",
            "cached_models": cached_count,
            "total_models": len(predictions),
            "processing_time": round(elapsed_time, 2),
            "commodity_daily": commodity_daily,
            "commodity_breakdown": commodity_breakdown if not commodity and unique_commodities > 10 else None,
            "telangana_data": {
                "available": telangana_data is not None,
                "data_points": len(telangana_data['dates']) if telangana_data else 0,
                "date_range": {
                    "start": telangana_data['dates'][0] if telangana_data else None,
                    "end": telangana_data['dates'][-1] if telangana_data else None
                } if telangana_data else None,
                "statistics": telangana_stats,
                "trend": telangana_trend
            },
            "telangana_insight": telangana_insight,
            "weather_factor_summary": weather_factor_counts,
            "weather_coverage_days": weather_info.get("num_days") if weather_info else 0,
            "ai_validation": ai_validation_result  # ✅ NEW: AI validation results
        }

        # Cache the result
        with prediction_cache_lock:
            prediction_cache[cache_key] = response

        return response

    except Exception as e:
        logging.error(f"Enhanced prediction error: {e}", exc_info=True)
        return {"error": str(e)}

# ========== FastAPI Routes ==========
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced LSTM+GRU Agricultural Forecasting System with Telangana Integration",
        "version": "4.0",
        "status": "operational"
    }

@app.get("/get_amcs")
async def get_amcs():
    """Get list of available AMCs"""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT amc_name FROM lots_new WHERE amc_name IS NOT NULL ORDER BY amc_name;")
                results = cursor.fetchall()
                amc_names = [row['amc_name'] for row in results if row.get('amc_name')]
        finally:
            conn.close()
        return {"amc_names": amc_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Enhanced prediction endpoint with Telangana integration"""
    try:
        result = await enhanced_prediction_with_telangana(request)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def enhanced_chat(request: ChatRequest):
    """Enhanced chat endpoint with intelligent routing"""
    try:
        message = request.message.strip()
        session_id = request.session_id
        district = request.district
        amc_name = request.amc_name

        if not message or not session_id:
            raise HTTPException(status_code=400, detail="Message and session_id required")

        # Initialize session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        state = _ensure_session_state(session_id)
        lower_msg = message.lower()

        # Handle pending clarifications
        if state.get("pending_request"):
            if _looks_like_new_prediction_query(lower_msg):
                _clear_session_state(state)
            else:
                clarification = await _handle_pending_clarifications(message, session_id, state)
                if clarification:
                    return clarification

        chat_sessions[session_id].append({"role": "user", "content": message})

        # Check for static responses
        static_response = static_reply(message)
        if static_response:
            chat_sessions[session_id].append({"role": "assistant", "content": static_response})
            return ChatResponse(response=static_response, query_type="STATIC")

        # Extract parameters dynamically
        extracted_district, extracted_amc = param_extractor.extract_location(message)
        metric = param_extractor.extract_metric(message)
        start_date, days = param_extractor.extract_timeframe(message)
        commodity = param_extractor.extract_commodity(message)

        final_district = district or extracted_district
        final_amc = amc_name or extracted_amc
        logging.info(f"🔍 Extracted params: district={final_district}, amc={final_amc}, commodity={commodity}, metric={metric}")

        # Determine if this is a prediction request
        is_prediction_query = any(word in lower_msg for word in PREDICTION_KEYWORDS)

        # Handle prediction requests
        if is_prediction_query:
            if not final_district and not final_amc:
                no_location_msg = "❌ Please specify a location (district or AMC name) for the forecast.\n\nExamples:\n- 'predict arrivals in Warangal'\n- 'forecast for Khammam next week'\n- 'bags expected in Nakrekal tomorrow'"
                chat_sessions[session_id].append({"role": "assistant", "content": no_location_msg})
                _clear_session_state(state)
                return ChatResponse(response=no_location_msg, query_type="LOCATION_REQUIRED")
            
            logging.info(f"🎯 Prediction request for district={final_district}, amc={final_amc}, commodity={commodity}")
            metric_mentions = param_extractor.detect_metric_mentions(message)
            # Show metric choice if NO specific metrics mentioned (only generic "arrivals")
            need_metric_choice = not metric_mentions

            commodity_candidates = []
            commodity_terms = [t for t in re.split(r"\W+", lower_msg) if len(t) >= 4]
            commodity_term = commodity_terms[0] if commodity_terms else "your commodity"
            if not commodity:
                commodity_candidates = param_extractor.get_commodity_candidates(message)
                if len(commodity_candidates) == 1:
                    commodity = commodity_candidates[0]
            need_commodity_choice = not commodity and len(commodity_candidates) > 1

            pred_payload = {
                "metric": metric,
                "district": final_district,
                "amc_name": final_amc,
                "commodity": commodity,
                "start_date": start_date,
                "days": days
            }

            clarification_needed = False
            if need_commodity_choice:
                clarification_needed = True
                state["pending_request"] = pred_payload
                state["awaiting_commodity"] = True
                state["commodity_options"] = commodity_candidates[:COMMODITY_SUGGESTION_LIMIT]
                state["commodity_term"] = commodity_term
                prompt = _format_commodity_prompt(commodity_term, state["commodity_options"])
                chat_sessions[session_id].append({"role": "assistant", "content": prompt})
                return ChatResponse(response=prompt, query_type="CLARIFICATION")

            if need_metric_choice:
                clarification_needed = True
                state["pending_request"] = pred_payload
                state["awaiting_metric"] = True
                prompt = _format_metric_prompt(final_district or final_amc)
                chat_sessions[session_id].append({"role": "assistant", "content": prompt})
                return ChatResponse(response=prompt, query_type="CLARIFICATION")

            if not clarification_needed:
                _clear_session_state(state)
                return await _fulfill_prediction_request(pred_payload, session_id)

        # Handle weather queries
        elif any(word in lower_msg for word in ['weather', 'rain', 'bring bags', 'should i come']):
            location = final_district or final_amc
            if not location:
                response_text = "Please provide your location (district or AMC) so I can check the weather."
                chat_sessions[session_id].append({"role": "assistant", "content": response_text})
                return ChatResponse(response=response_text, query_type="LOCATION_NEEDED")

            weather_info = fetch_multi_day_weather(location, num_days=min(days, 7))
            
            if not weather_info:
                response_text = f"❌ Could not fetch weather data for {location}."
                chat_sessions[session_id].append({"role": "assistant", "content": response_text})
                return ChatResponse(response=response_text, query_type="WEATHER_ERROR")

            # Generate weather summary
            condition = weather_info.get('condition', 'Unknown')
            rain_mm = weather_info.get('rain_mm', 0)
            temp_c = weather_info.get('temp_c', 0)

            if rain_mm >= 10:
                advice = f"🌧 Heavy rain expected in {location} ({rain_mm}mm). Not recommended to bring bags."
            elif rain_mm > 0:
                advice = f"🌦 Light rain expected in {location} ({rain_mm}mm). Proceed with caution."
            else:
                advice = f"☀ Clear weather in {location}. Good conditions for bringing bags."

            weather_response = f"{advice}\n🌡 Temperature: {temp_c}°C | Condition: {condition}"
            
            chat_sessions[session_id].append({"role": "assistant", "content": weather_response})
            return ChatResponse(
                response=weather_response,
                weather_summary=weather_info,
                query_type="WEATHER_ADVICE"
            )

        # Fallback to Gemini AI
        else:
            history = chat_sessions[session_id][-10:]
            prompt = "You are an Agricultural AI Assistant using optimized LSTM+GRU models with Telangana market data integration.\n"
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "Assistant:"

            try:
                model = genai.GenerativeModel(GEMINI_MODEL)
                gemini_response = model.generate_content(prompt)
                if gemini_response and gemini_response.text:
                    assistant_reply = gemini_response.text.strip()
                    chat_sessions[session_id].append({"role": "assistant", "content": assistant_reply})
                    return ChatResponse(response=assistant_reply, query_type="AI_GENERAL")
            except Exception as e:
                logging.error(f"Gemini error: {e}")

            fallback = "I'm not sure how to answer that. Ask me about market forecasts or weather advice!"
            chat_sessions[session_id].append({"role": "assistant", "content": fallback})
            return ChatResponse(response=fallback, query_type="FALLBACK")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Chat error: {e}", exc_info=True)
        error_msg = f"Something went wrong: {str(e)}"
        return ChatResponse(response=error_msg, query_type="ERROR")

# ========== Telangana Data Endpoints ==========
@app.get("/telangana/info")
async def telangana_data_info():
    """Get information about Telangana scraped data"""
    try:
        info = get_telangana_data_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/telangana/commodities")
async def get_telangana_commodities():
    """Get available commodities from Telangana data"""
    try:
        service = get_telangana_service()
        commodities = service.get_available_commodities()
        return {"commodities": commodities, "count": len(commodities)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/telangana/markets/{commodity}")
async def get_telangana_markets(commodity: str):
    """Get available markets for a commodity"""
    try:
        service = get_telangana_service()
        markets = service.get_available_markets(commodity)
        return {"commodity": commodity, "markets": markets, "count": len(markets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/telangana/arrivals/{commodity}")
async def get_telangana_commodity_data(
    commodity: str,
    market: str = None,
    days: int = 60
):
    """Get arrival data for a commodity"""
    try:
        data = get_telangana_arrivals(commodity, market, days)
        
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for commodity: {commodity}"
            )
        
        # Get statistics
        stats = get_telangana_stats(commodity, market)
        trend = get_telangana_trend(commodity, market, days=7)
        
        return {
            "commodity": commodity,
            "market": market,
            "data": data,
            "statistics": stats,
            "trend": trend
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/telangana/force-update")
async def force_telangana_update():
    """Force an immediate update of Telangana data"""
    try:
        service = get_telangana_service()
        
        # Run update in background
        loop = asyncio.get_event_loop()
        
        def update_task():
            return service.force_update()
        
        success = await loop.run_in_executor(None, update_task)
        
        if success:
            return {
                "status": "success",
                "message": "Data update completed",
                "info": service.get_data_info()
            }
        else:
            return {
                "status": "failed",
                "message": "Data update failed"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== Debug Endpoints ==========
@app.get("/debug/locations")
async def debug_locations():
    """Debug endpoint to see available locations"""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT district, COUNT(*) as record_count 
                    FROM lots_new WHERE district IS NOT NULL 
                    GROUP BY district ORDER BY record_count DESC LIMIT 20
                """)
                district_results = cursor.fetchall()
                
                cursor.execute("""
                    SELECT DISTINCT amc_name, COUNT(*) as record_count 
                    FROM lots_new WHERE amc_name IS NOT NULL 
                    GROUP BY amc_name ORDER BY record_count DESC LIMIT 20
                """)
                amc_results = cursor.fetchall()
        finally:
            conn.close()
        
        return {
            "top_districts": district_results,
            "top_amcs": amc_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get information about the enhanced model"""
    return {
        "model_type": "Enhanced Hybrid LSTM+GRU with Telangana Integration",
        "version": "4.0",
        "architecture": {
            "lstm_units": 32,
            "gru_units": 16,
            "dense_layers": [8, 1],
            "dropout_rate": 0.1,
            "sequence_length": 10
        },
        "data_sources": [
            "Historical TAMC database",
            "Telangana market arrivals (auto-updated)",
            "Cultivation area data",
            "Weather forecasts"
        ],
        "optimizations": [
            "Model caching (24-hour TTL)",
            "Parallel processing (4 workers)",
            "Reduced model complexity",
            "Early stopping (patience=3)",
            "Optimized database queries",
            "Batch predictions",
            "Smart cache invalidation",
            "Telangana data integration"
        ],
        "features": [
            "Long Short-Term Memory (LSTM) for long-term patterns",
            "Gated Recurrent Unit (GRU) for recent trends",
            "Weather impact adjustment",
            "Cultivation area factors",
            "Telangana market trends",
            "Dynamic parameter extraction",
            "Fast training (15 epochs)"
        ]
    }

@app.get("/cache/status")
async def cache_status():
    """Get cache statistics"""
    try:
        memory_cache_size = len(model_cache.memory_cache)
        
        disk_cache_files = 0
        if os.path.exists(model_cache.cache_dir):
            disk_cache_files = len([f for f in os.listdir(model_cache.cache_dir) if f.endswith('.pkl')])
        
        prediction_cache_size = len(prediction_cache)
        
        # Get Telangana data status
        telangana_info = get_telangana_data_info()
        
        return {
            "model_cache": {
                "memory": memory_cache_size,
                "disk": disk_cache_files,
                "max_age_hours": model_cache.max_age_hours
            },
            "prediction_cache": {
                "size": prediction_cache_size
            },
            "telangana_data": telangana_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    try:
        # Clear model cache
        with model_cache.lock:
            model_cache.memory_cache.clear()
            if os.path.exists(model_cache.cache_dir):
                for filename in os.listdir(model_cache.cache_dir):
                    filepath = os.path.join(model_cache.cache_dir, filename)
                    os.remove(filepath)
        
        # Clear prediction cache
        with prediction_cache_lock:
            prediction_cache.clear()
        
        return {
            "status": "success",
            "message": "All caches cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    telangana_info = get_telangana_data_info()
    
    return {
        "status": "healthy",
        "version": "4.0",
        "model": "Enhanced LSTM+GRU with Telangana Integration",
        "timestamp": datetime.now().isoformat(),
        "cache_enabled": True,
        "telangana_data_status": telangana_info.get('status', 'unknown')
    }

# ========== Background Tasks ==========
async def cleanup_old_cache():
    """Periodic cache cleanup"""
    while True:
        try:
            await asyncio.sleep(86400)  # Run once every 24 hours
            model_cache.clear_old_cache()
            logging.info("🧹 Cache cleanup completed (24h interval)")
        except Exception as e:
            logging.error(f"Cache cleanup error: {e}")

@app.on_event("startup")
async def startup_event_enhanced():
    """Enhanced startup with Telangana service initialization"""
    # Start cleanup tasks
    asyncio.create_task(cleanup_old_cache())
    
    # Initialize Telangana service
    logging.info("🌾 Initializing Telangana data service...")
    service = get_telangana_service()
    
    # Check data status
    info = service.get_data_info()
    if info['status'] == 'no_data':
        logging.info("⚠️ No Telangana data found. Triggering initial scrape...")
        # Run initial scrape in background
        threading.Thread(target=service.scrape_and_update, args=(60,), daemon=True).start()
    else:
        logging.info(f"✅ Telangana data loaded: {info['total_records']} records")
        logging.info(f"📅 Last update: {info['last_update']}")
        logging.info(f"📊 Commodities: {info['unique_commodities']}, Markets: {info['unique_markets']}")
    
    logging.info("🚀 Enhanced LSTM+GRU Forecasting System with Telangana Integration started")
    logging.info("📡 API available at: http://localhost:8000")
    logging.info("📚 Docs available at: http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logging.info("👋 Shutting down Enhanced LSTM+GRU Forecasting System")

# ========== CLI Mode ==========
def cli_loop():
    """Interactive CLI for testing predictions"""
    print("\n🌾 Enhanced LSTM+GRU Agricultural Forecasting System CLI")
    print("=" * 70)
    print("✨ Now with Telangana Market Data Integration!")
    print("=" * 70)
    print("Examples:")
    print("  - 'forecast for Hyderabad tomorrow'")
    print("  - 'should I bring bags to Warangal next week'")
    print("  - 'predict tomato arrivals for Guntur next 3 days'")
    print("  - 'telangana info' - Check Telangana data status")
    print("=" * 70)
    
    # Show Telangana data status
    telangana_info = get_telangana_data_info()
    print(f"\n📊 Telangana Data Status: {telangana_info.get('status', 'unknown')}")
    if telangana_info.get('status') == 'available':
        print(f"   Records: {telangana_info.get('total_records', 0):,}")
        print(f"   Commodities: {telangana_info.get('unique_commodities', 0)}")
        print(f"   Last Update: {telangana_info.get('last_update', 'N/A')}")
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye! 👋")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye! 🌾")
            break
        
        # Special commands
        if user_input.lower() == "telangana info":
            info = get_telangana_data_info()
            print("\n" + "=" * 70)
            print("📊 Telangana Data Information")
            print("=" * 70)
            for key, value in info.items():
                print(f"  {key}: {value}")
            print("=" * 70)
            continue
        
        if user_input.lower() == "telangana update":
            print("\n🔄 Forcing Telangana data update...")
            service = get_telangana_service()
            success = service.force_update()
            if success:
                print("✅ Update completed successfully!")
                info = service.get_data_info()
                print(f"   Records: {info['total_records']:,}")
            else:
                print("❌ Update failed. Check logs for details.")
            continue
        
        # Check static replies
        static_resp = static_reply(user_input)
        if static_resp:
            print(f"\n{static_resp}\n")
            continue
        
        # Extract parameters
        district, amc = param_extractor.extract_location(user_input)
        metric = param_extractor.extract_metric(user_input)
        start_date, days = param_extractor.extract_timeframe(user_input)
        commodity = param_extractor.extract_commodity(user_input)
        
        if not district and not amc:
            print("\n❌ Could not detect location. Please mention a district or AMC name.\n")
            continue
        
        # Create prediction request
        print(f"\n⏳ Running enhanced LSTM+GRU prediction with Telangana data...")
        print(f"   Location: {district or amc}")
        if commodity:
            print(f"   Commodity: {commodity}")
        print(f"   Metric: {metric}")
        print(f"   Forecast: {days} days")
        
        pred_request = PredictionRequest(
            metric=metric,
            district=district,
            amc_name=amc,
            commodity=commodity,
            start_date=start_date,
            days=days
        )
        
        try:
            result = asyncio.run(enhanced_prediction_with_telangana(pred_request))
            
            if "error" in result:
                print(f"\n❌ Error: {result['error']}\n")
                continue
            
            # Display results
            print("\n" + "=" * 70)
            formatted = response_formatter.format_prediction_response(result)
            print(formatted)
            
            # Add Telangana insight
            if result.get('telangana_insight'):
                print(f"\n{result['telangana_insight']}")
            
            print("=" * 70)
            
            # Show model info
            print(f"\n🤖 Model: {result.get('model_type', 'LSTM+GRU')}")
            print(f"📋 Metric: {result.get('metric_name', 'Unknown')}")
            print(f"⚡ Cached Models: {result.get('cached_models', 0)}/{result.get('total_models', 0)}")
            print(f"⏱️ Processing Time: {result.get('processing_time', 0)}s")
            
            # Telangana data info
            telangana_data = result.get('telangana_data', {})
            if telangana_data.get('available'):
                print(f"\n🌾 Telangana Data:")
                print(f"   Data Points: {telangana_data.get('data_points', 0)}")
                if telangana_data.get('statistics'):
                    stats = telangana_data['statistics']
                    print(f"   Average Arrivals: {stats.get('mean', 0):.2f} Qtls")
                    print(f"   Range: {stats.get('min', 0):.2f} - {stats.get('max', 0):.2f} Qtls")
            else:
                print(f"\nℹ️ No Telangana data available for this commodity")
            
            weather = result.get('weather_summary')
            if weather:
                print(f"\n🌤 Weather: {weather.get('condition')} | {weather.get('temp_c')}°C | {weather.get('rain_mm')}mm rain")
            
            print()
            
        except Exception as e:
            print(f"\n❌ Prediction failed: {e}\n")
            logging.error(f"CLI prediction error: {e}", exc_info=True)

# ========== Main Execution ==========
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run in CLI mode
        cli_loop()
    else:
        # Run FastAPI server
        import uvicorn
        print("\n" + "=" * 70)
        print("🚀 Starting Enhanced LSTM+GRU Agricultural Forecasting API")
        print("=" * 70)
        print("🌾 Features:")
        print("   ✅ LSTM+GRU Deep Learning Models")
        print("   ✅ Telangana Market Data Integration (Auto-updated)")
        print("   ✅ Cultivation Area Factors")
        print("   ✅ Weather Impact Analysis")
        print("   ✅ Model Caching (24-hour TTL)")
        print("   ✅ Parallel Processing (4 workers)")
        print("   ✅ Smart Parameter Extraction")
        print("=" * 70)
        print("📡 API will be available at: http://localhost:8000")
        print("📚 Docs available at: http://localhost:8000/docs")
        print("🔧 CLI mode: python script.py cli")
        print("=" * 70)
        print("\n⚡ Optimizations enabled:")
        print("   - Model caching (24-hour TTL)")
        print("   - Parallel processing (4 workers)")
        print("   - Optimized queries (180-day window)")
        print("   - Reduced model complexity")
        print("   - Fast training (15 epochs)")
        print("   - Telangana data auto-updates (daily at 2:00 AM)")
        print("\n🔍 New Telangana Endpoints:")
        print("   GET  /telangana/info - Data status")
        print("   GET  /telangana/commodities - Available commodities")
        print("   GET  /telangana/markets/{commodity} - Markets for commodity")
        print("   GET  /telangana/arrivals/{commodity} - Arrival data")
        print("   POST /telangana/force-update - Force data update")
        print("\n🎯 Enhanced Prediction:")
        print("   - Combines TAMC + Telangana market data")
        print("   - Weighted factors: 60% cultivation + 40% Telangana trends")
        print("   - Real-time market insights")
        print("=" * 70 + "\n")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)