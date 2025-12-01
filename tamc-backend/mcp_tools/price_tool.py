# main.py - Enhanced Agricultural Price Prediction with Sentiment Analysis
# Integrates: GRU Model + Weather + Disease Risk + News Sentiment + NCDEX + Napanta

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pymysql
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import requests
import torch
import torch.nn as nn
import os
import random
import pickle
from sqlalchemy import create_engine
import time
import contextlib
import io
import uvicorn
import concurrent.futures
import re
from textblob import TextBlob
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalizer import clean_amc, clean_district, clean_commodity

# Ensure project root is in Python path
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import helper scripts
try:
    from scrapers.ncdex_scrape import get_or_update_ncdex_data
    from scrapers.naapanta_scrape import get_historical_prices_for_prediction
    from scrapers.news_sentiment import get_news_sentiment_score, get_sentiment_summary
    SCRAPERS_AVAILABLE = True
except ModuleNotFoundError as e:
    print("⚠️ Warning: Could not import scrapers. Some features may be limited.")
    SCRAPERS_AVAILABLE = False

# Import weather helper for enhanced weather integration
try:
    from weather_helper import WeatherHelper
    WEATHER_HELPER_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Could not import weather_helper. Using basic weather fetcher.")
    WEATHER_HELPER_AVAILABLE = False

# ========== REPRODUCIBILITY ==========
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# ========== CONFIGURATION ==========
warnings.filterwarnings("ignore")

# Get database credentials from environment variables
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_NAME = os.getenv("DB_NAME", "")

password = quote_plus(DB_PASSWORD)
engine = create_engine(f"mysql+pymysql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Get API keys from environment variables
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

MODEL_CACHE_DIR = "trained_models"
MODEL_CACHE_DURATION = timedelta(hours=24)
CACHE_DURATION_SECONDS = 24 * 60 * 60
DB_CACHE_FILE = "merged_lots_data.csv"
NEWS_CACHE_FILE = "news_sentiment_cache.json"

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

MARKETS_TO_SCRAPE = ["Warangal", "Khammam", "Nakrekal"]
COMMODITIES_TO_SCRAPE = ["Chilli", "Cotton", "Lemon", "Groundnut"]
HISTORICAL_SCRAPE_DAYS = 60

# ========== PYDANTIC MODELS ==========

class PredictionRequest(BaseModel):
    commodity: str = Field(..., description="Commodity name (e.g., 'Chilli', 'Cotton')")
    market: str = Field(..., description="Market name (e.g., 'Warangal', 'Khammam')")
    prediction_days: int = Field(1, ge=1, le=30, description="Number of days to predict (1-30)")
    variant: Optional[str] = Field(None, description="Specific variant (optional)")

class SentimentInfo(BaseModel):
    avg_sentiment: float
    sentiment_label: str
    news_count: int
    top_keywords: List[str]
    sample_headlines: List[str]

class PriceForecast(BaseModel):
    date: str
    baseline_price: float
    weather_adjustment: float
    disease_risk: float
    disease_adjustment: float
    sentiment_adjustment: float
    final_price: float
    min_price: float
    max_price: float
    weather_reason: str
    disease_reason: str
    sentiment_reason: str

class VariantForecast(BaseModel):
    variant: str
    forecasts: List[PriceForecast]
    sentiment_info: Optional[SentimentInfo]

class PredictionResponse(BaseModel):
    commodity: str
    market: str
    prediction_days: int
    variants: List[VariantForecast]
    message: str

class NCDEXRequest(BaseModel):
    commodity: str = Field(..., description="NCDEX commodity name")

class NCDEXResponse(BaseModel):
    commodity: str
    spot_price: Optional[float]
    message: str

class VariantsResponse(BaseModel):
    commodity: str
    market: str
    variants: List[str]
    count: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    cache_status: Dict[str, Any]
    sentiment_status: Dict[str, Any]

# ========== NEWS SENTIMENT ANALYZER ==========

class NewsSentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.cache_file = NEWS_CACHE_FILE
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """Load cached news sentiment data"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    import json
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save news sentiment cache"""
        try:
            import json
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except:
            pass
    
    def _is_cache_fresh(self, commodity, hours=6):
        """Check if cached data is still fresh (default: 6 hours)"""
        if commodity not in self.cache:
            return False
        cache_time = self.cache[commodity].get('timestamp')
        if not cache_time:
            return False
        cache_dt = datetime.fromisoformat(cache_time)
        return (datetime.now() - cache_dt).total_seconds() < (hours * 3600)
    
    def fetch_news(self, query, days_back=7, page_size=20):
        """Fetch recent news from NewsAPI"""
        if not self.api_key:
            return None
        
        url = "https://newsapi.org/v2/everything"
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        params = {
            "q": query,
            "from": from_date,
            "language": "en",
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "apiKey": self.api_key
        }
        
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            if data.get("status") != "ok":
                return None
            
            articles = data.get("articles", [])
            df = pd.DataFrame([{
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "publishedAt": a.get("publishedAt", ""),
                "source": a.get("source", {}).get("name", "")
            } for a in articles])
            
            return df
        except Exception as e:
            print(f"⚠️ News API error: {e}")
            return None
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        if not text or pd.isna(text):
            return 0.0
        
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def extract_keywords(self, text, top_n=3):
        """Simple keyword extraction (frequency-based)"""
        if not text or pd.isna(text):
            return []
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
                      'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might'}
        
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        words = [w for w in words if w not in stop_words]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(words)
        
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def get_sentiment_analysis(self, commodity, force_refresh=False):
        """Get comprehensive sentiment analysis for a commodity"""
        commodity_key = commodity.lower()
        
        # Check cache first
        if not force_refresh and self._is_cache_fresh(commodity_key):
            print(f"   📰 Using cached sentiment data for {commodity}")
            return self.cache[commodity_key]['data']
        
        print(f"   📰 Fetching fresh news for {commodity}...")
        
        # Fetch news
        news_df = self.fetch_news(commodity)
        
        if news_df is None or news_df.empty:
            print(f"   ⚠️ No news found for {commodity}, using neutral sentiment")
            result = {
                'avg_sentiment': 0.0,
                'sentiment_label': 'Neutral',
                'news_count': 0,
                'top_keywords': [],
                'sample_headlines': [],
                'articles': []
            }
        else:
            # Analyze sentiment
            news_df['sentiment'] = news_df.apply(
                lambda row: self.analyze_sentiment(
                    f"{row['title']} {row['description']}"
                ), axis=1
            )
            
            # Extract keywords
            all_text = ' '.join(
                news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
            )
            top_keywords = self.extract_keywords(all_text, top_n=5)
            
            # Label sentiment
            news_df['sentiment_label'] = news_df['sentiment'].apply(
                lambda s: "Positive" if s > 0.1 else ("Negative" if s < -0.1 else "Neutral")
            )
            
            avg_sentiment = float(news_df['sentiment'].mean())
            
            result = {
                'avg_sentiment': round(avg_sentiment, 3),
                'sentiment_label': "Positive" if avg_sentiment > 0.1 else (
                    "Negative" if avg_sentiment < -0.1 else "Neutral"
                ),
                'news_count': len(news_df),
                'top_keywords': top_keywords,
                'sample_headlines': news_df['title'].head(3).tolist(),
                'articles': news_df[['title', 'sentiment', 'sentiment_label', 'source']].head(10).to_dict('records')
            }
        
        # Cache the result
        self.cache[commodity_key] = {
            'timestamp': datetime.now().isoformat(),
            'data': result
        }
        self._save_cache()
        
        return result

# ========== DATA LOADING FUNCTIONS ==========

def standardize_naapanta_data(df, market_name):
    """Converts raw Naapanta DataFrame to match DB schema"""
    if df.empty:
        return pd.DataFrame()

    date_col = 'Fetch_Date' if 'Fetch_Date' in df.columns else 'Date'
    comm_col = 'Commodity'
    var_col = 'Variety'
    min_col = 'Min_Price' if 'Min_Price' in df.columns else 'MinPrice'
    max_col = 'Max_Price' if 'Max_Price' in df.columns else 'MaxPrice'
    avg_col = 'Avg_Price' if 'Avg_Price' in df.columns else 'ModalPrice'

    if not all(c in df.columns for c in [date_col, comm_col, var_col, min_col, max_col, avg_col]):
        return pd.DataFrame()

    df_clean = df.copy()

    try:
        df_clean['date'] = pd.to_datetime(df_clean[date_col], dayfirst=True, errors='coerce')
    except Exception:
        df_clean['date'] = pd.to_datetime(df_clean[date_col], errors='coerce')

    df_clean[comm_col] = df_clean[comm_col].str.strip().str.title()
    df_clean[var_col] = df_clean[var_col].str.strip().str.title()

    df_clean['commodity_name'] = df_clean.apply(
        lambda row: f"{row[comm_col]}-{row[var_col]}" if str(row[var_col]).lower() not in ['nan', 'other'] else row[comm_col],
        axis=1
    )

    standardized_df = pd.DataFrame({
        'date': df_clean['date'],
        'amc_name': market_name.lower(),
        'commodity_name': df_clean['commodity_name'],
        'arrivals': 0,
        'min_price': pd.to_numeric(df_clean[min_col], errors='coerce').fillna(0),
        'max_price': pd.to_numeric(df_clean[max_col], errors='coerce').fillna(0),
        'avg_price': pd.to_numeric(df_clean[avg_col], errors='coerce').fillna(0),
        'district': market_name.lower()
    })

    standardized_df = standardized_df.dropna(subset=['date'])
    return standardized_df

def preprocess_old(df):
    df = df.copy()

    df["posted_on"] = pd.to_datetime(df["posted_on"], errors="coerce")

    # STEP 1 — Extract correct mandi price
    # Use rate_for_qui (price per quintal)
    price = pd.to_numeric(df.get("rate_for_qui", 0), errors="coerce").fillna(0)

    # If rate_for_qui missing -> fallback to bid_amount
    fallback = pd.to_numeric(df.get("bid_amount", 0), errors="coerce").fillna(0)

    price = price.where(price > 0, fallback)

    # STEP 2 — Build clean dataframe
    df_clean = pd.DataFrame({
        "date": df["posted_on"],
        "amc_name": df["amc_name"].astype(str).str.lower(),
        "commodity_name": df["commodity_name"].astype(str).str.lower(),
        "arrivals": df["aprox_quantity"].fillna(0),
        "min_price": price,
        "max_price": price,
        "avg_price": price,
        "district": df["district"].astype(str).str.lower() if "district" in df else None
    })

    # STEP 3 — DO NOT DROP ZERO PRICES
    df_clean = df_clean.dropna(subset=["date", "commodity_name"])

    return df_clean

def get_or_update_db_data(cache_file=DB_CACHE_FILE):
    """Build or refresh the data cache (24-hour validity)"""
    if os.path.exists(cache_file):
        file_mod_time = os.path.getmtime(cache_file)
        file_age = time.time() - file_mod_time

        if file_age < CACHE_DURATION_SECONDS:
            print(f"✅ Loading static data from '{cache_file}' cache (less than 24h old)...")
            return True
        else:
            print(f"🕒 Static data cache is older than 24 hours. Rebuilding...")
    else:
        print(f"'{cache_file}' not found. Building new historical data cache...")

    print("This may take several minutes as it includes live scraping.")

    all_data_frames = []

    # Load from database
    try:
        print("📄 Connecting to the database...")
        df_old = pd.read_sql("SELECT * FROM lots_new", engine)
        print("✅ Database connection successful. 'lots_new' table loaded.")

        df_old_clean = preprocess_old(df_old)
        all_data_frames.append(df_old_clean)
        print(f"   -> Loaded {len(df_old_clean)} records from database.")

    except Exception as e:
        print(f"\n❌ WARNING: Could not connect to the database.")
        print(f"   DETAILS: {e}")
        print("   -> Continuing by building cache ONLY from Naapanta data.")

    # Load from Naapanta (SKIP if naapanta_scrape not available)
    try:
        # Check if scraping function is available
        from scrapers.naapanta_scrape import get_historical_prices_for_prediction

        print(f"\n🌎 Fetching {HISTORICAL_SCRAPE_DAYS} days of historical data from Naapanta...")
        print(f"   Markets: {MARKETS_TO_SCRAPE}")
        print(f"   Commodities: {COMMODITIES_TO_SCRAPE}")

        all_naapanta_data = []

        for market in MARKETS_TO_SCRAPE:
            for commodity in COMMODITIES_TO_SCRAPE:
                print(f"   ...Scraping {commodity} in {market}...")

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                get_historical_prices_for_prediction,
                                "Telangana", market, market, commodity,
                                HISTORICAL_SCRAPE_DAYS
                            )
                            naapanta_hist_df = future.result(timeout=120)
                    except Exception as e:
                        naapanta_hist_df = None

                if naapanta_hist_df is not None:
                    standardized_df = standardize_naapanta_data(naapanta_hist_df, market)

                    if not standardized_df.empty:
                        print(f"     -> Found {len(standardized_df)} historical records.")
                        all_naapanta_data.append(standardized_df)
                    else:
                        print(f"     -> No data found.")
                else:
                    print(f"     -> Scrape failed.")

        if all_naapanta_data:
            df_naapanta_combined = pd.concat(all_naapanta_data, ignore_index=True)
            all_data_frames.append(df_naapanta_combined)
            print(f"✅ Total Naapanta records fetched: {len(df_naapanta_combined)}")
        else:
            print("⚠️ No data was fetched from Naapanta.")
    except ImportError:
        print("⚠️ Naapanta scraping not available - using database data only")

    if not all_data_frames:
        print("❌ CRITICAL: No data loaded from DB or Naapanta. Cache not created.")
        return False

    print("\n💾 Combining and cleaning all data sources...")
    merged_df = pd.concat(all_data_frames, ignore_index=True)

    merged_df = merged_df.dropna(subset=["date", "commodity_name", "amc_name"])
    merged_df['max_price'] = pd.to_numeric(merged_df['max_price'], errors='coerce')

    merged_df = merged_df.sort_values("date")
    merged_df = merged_df.drop_duplicates(
        subset=["date", "amc_name", "commodity_name"],
        keep="last"
    )

    merged_df = merged_df.reset_index(drop=True)

    try:
        merged_df.to_csv(cache_file, index=False)
        print(f"✅ Fresh '{cache_file}' created with {len(merged_df)} combined records.")
        return True
    except Exception as e:
        print(f"❌ CRITICAL: Failed to save cache file '{cache_file}': {e}")
        return False

# ========== WEATHER FETCHER ==========

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.weatherapi.com/v1"

    def get_weather_forecast(self, city, days=1):
        """Fetch weather forecast for next N days (max 3 for free plan)"""
        if not self.api_key:
            return []

        days_to_fetch = min(days, 3)

        try:
            url = f"{self.base_url}/forecast.json?key={self.api_key}&q={city}&days={days_to_fetch}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            res = response.json()

            forecasts = []
            for day_data in res["forecast"]["forecastday"]:
                forecast_day = day_data["day"]
                forecasts.append({
                    "date": day_data["date"],
                    "temp": forecast_day["avgtemp_c"],
                    "condition": forecast_day["condition"]["text"].lower(),
                    "total_rain_mm": forecast_day.get("totalprecip_mm", 0),
                    "will_it_rain": forecast_day.get("daily_will_it_rain", 0) > 0
                })
            return forecasts

        except Exception as e:
            return []

# ========== GRU MODEL ==========

class PriceGRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ========== MODEL CACHE MANAGER ==========

class ModelCacheManager:
    def __init__(self, cache_dir="trained_models", cache_duration_hours=24):
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=cache_duration_hours)
        os.makedirs(cache_dir, exist_ok=True)

    def get_model_path(self, market, commodity, variant):
        safe_name = f"{market}_{commodity}_{variant}".replace(" ", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe_name}.pt")

    def load_model(self, market, commodity, variant):
        model_path = self.get_model_path(market, commodity, variant)

        if not os.path.exists(model_path):
            return None

        file_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        if datetime.now() - file_time > self.cache_duration:
            print(f"     ⏰ Model cache for {variant} expired (>24h)")
            return None

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = PriceGRUModel(input_size=1, hidden_size=64, num_layers=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            age_hours = (datetime.now() - file_time).total_seconds() / 3600
            print(f"   ✅ Using cached model for {variant} (age: {age_hours:.1f}h)")
            return model
        except Exception as e:
            print(f"     ⚠️ Failed to load cached model for {variant}: {e}")
            return None

    def save_model(self, model, market, commodity, variant):
        model_path = self.get_model_path(market, commodity, variant)
        try:
            torch.save(model.to('cpu').state_dict(), model_path)
            print(f"     💾 Saved model for {variant}")
        except Exception as e:
            print(f"     ⚠️ Failed to save model for {variant}: {e}")

# ========== GRU PREDICTOR CLASS ==========

class GRUPricePredictor:
    def __init__(self):
        # Use WeatherHelper if available, otherwise fall back to WeatherFetcher
        if WEATHER_HELPER_AVAILABLE:
            self.weather_helper = WeatherHelper(WEATHER_API_KEY)
            self.weather_fetcher = None
            print("✅ Using WeatherHelper for enhanced weather integration (up to 14-day forecasts)")
        else:
            self.weather_fetcher = WeatherFetcher(WEATHER_API_KEY)
            self.weather_helper = None
            print("⚠️ Using basic WeatherFetcher (limited to 3-day forecasts)")

        self.sentiment_analyzer = NewsSentimentAnalyzer(NEWS_API_KEY)

        print(f"📄 Loading '{DB_CACHE_FILE}' into memory...")
        self.full_historical_data = pd.read_csv(DB_CACHE_FILE, parse_dates=["date"])
        print(f"✅ Loaded {len(self.full_historical_data)} records from local cache.")

        self.model_cache = ModelCacheManager(cache_dir=MODEL_CACHE_DIR, cache_duration_hours=24)

        print("🌐 Loading NCDEX market data (cached/scheduled)...")
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_or_update_ncdex_data)
                self.ncdex_data = future.result(timeout=120)

            if self.ncdex_data is not None:
                print(f"✅ Loaded {len(self.ncdex_data)} NCDEX records.")
            else:
                print("⚠️ NCDEX scrape returned no data.")
        except Exception as e:
            print(f"⚠️ Error loading NCDEX data: {e}")
            self.ncdex_data = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"✅ GPU (CUDA) detected! Training will be much faster.")
        else:
            print(f"⚠️ No GPU (CUDA) detected. Training will be on CPU (slower).")

    def get_historical_data(self, market=None, commodity=None, days=1825):
        df = self.full_historical_data.copy()
        df['amc_name'] = df['amc_name'].astype(str).str.lower()
        df['commodity_name'] = df['commodity_name'].astype(str).str.lower()

        market_clean = market.lower().strip()
        commodity_clean = commodity.lower().strip()

        df_filtered = df[
            df['amc_name'].str.contains(market_clean, na=False) &
            df['commodity_name'].str.contains(commodity_clean, na=False)
        ]

        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        df_filtered = df_filtered[df_filtered['date'] >= cutoff_date]

        return df_filtered.sort_values("date").reset_index(drop=True)

    def prepare_sequences(self, series, seq_len=14):
        X, y = [], []
        for i in range(len(series) - seq_len):
            X.append(series[i:i + seq_len])
            y.append(series[i + seq_len])
        return np.array(X), np.array(y)

    def train_gru_model(self, series, seq_len=14, lr=0.01, variant_name=""):
        if len(series) < seq_len + 1:
            return None, None, None

        mean_val = float(series.mean())
        std_val = float(series.std()) if series.std() > 0 else 1.0
        scaled = (series - mean_val) / std_val

        X_np, y_np = self.prepare_sequences(scaled, seq_len=seq_len)
        if X_np.size == 0:
            return None, None, None

        split_idx = int(len(X_np) * 0.8)
        if split_idx < 1 or len(X_np) - split_idx < 1:
            X_train_np, y_train_np = X_np, y_np
            X_val_np, y_val_np = X_np, y_np
        else:
            X_train_np, y_train_np = X_np[:split_idx], y_np[:split_idx]
            X_val_np, y_val_np = X_np[split_idx:], y_np[split_idx:]

        X_train = torch.tensor(X_train_np, dtype=torch.float32).reshape(-1, seq_len, 1).to(self.device)
        y_train = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1).to(self.device)
        X_val = torch.tensor(X_val_np, dtype=torch.float32).reshape(-1, seq_len, 1).to(self.device)
        y_val = torch.tensor(y_val_np, dtype=torch.float32).reshape(-1, 1).to(self.device)

        model = PriceGRUModel(input_size=1, hidden_size=64, num_layers=2).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        max_epochs = 120
        patience = 5
        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            out_train = model(X_train)
            loss_train = loss_fn(out_train, y_train)
            loss_train.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_val = model(X_val)
                loss_val = loss_fn(out_val, y_val)

            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        if best_model_state:
            model.load_state_dict(best_model_state)

        model.eval()
        return model, mean_val, std_val

    def get_available_variants(self, market, commodity):
        PRICE_COLUMN = "max_price"
        historical_data = self.get_historical_data(market=market, commodity=commodity)

        if historical_data.empty:
            return None, None

        # DO NOT FILTER OUT DB ROWS
        historical_data_original = historical_data.copy()

        variants = historical_data_original['commodity_name'].unique()

        if len(variants) == 0:
            return None, None

        return historical_data_original, variants

    def predict_for_single_variant(self, variant_df, variant, market, commodity, prediction_days):
        PRICE_COLUMN = "max_price"

        # Updated moving average window
        MA_WINDOW = 14

        # Updated sequence length
        seq_len = 21

        raw_series_pd = variant_df[PRICE_COLUMN]
        ma_series_pd = raw_series_pd.rolling(window=MA_WINDOW).mean()
        ma_series_clean_pd = ma_series_pd.dropna()
        series = np.array(ma_series_clean_pd.values, dtype=np.float32)

        if len(series) < seq_len + 1:
            raw_series_np = np.array(raw_series_pd.values, dtype=np.float32)
            last_price = raw_series_np[-1] if len(raw_series_np) > 0 else 5000

            preds = []
            for i in range(prediction_days):
                pred_date = (datetime.now().date() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                daily_price = last_price * (1 + np.random.uniform(-0.03, 0.03))
                
                # Calculate min/max range (+/- 5%)
                min_pred = daily_price * 0.95
                max_pred = daily_price * 1.05
                
                preds.append({
                    "date": pred_date, 
                    "predicted_value": round(max(daily_price, 0), 2),
                    "min_price": round(max(min_pred, 0), 2),
                    "max_price": round(max(max_pred, 0), 2)
                })
            return preds

        cached_model = self.model_cache.load_model(market, commodity, variant)

        if cached_model is not None:
            model = cached_model
            mean_val = float(series.mean())
            std_val = float(series.std()) if series.std() > 0 else 1.0
        else:
            model, mean_val, std_val = self.train_gru_model(series, seq_len=seq_len, lr=0.01, variant_name=variant)

            if model is None:
                raw_series_np = np.array(raw_series_pd.values, dtype=np.float32)
                last_price = raw_series_np[-1] if len(raw_series_np) > 0 else 5000

                preds = []
                for i in range(prediction_days):
                    pred_date = (datetime.now().date() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                    daily_price = last_price * (1 + np.random.uniform(-0.03, 0.03))
                    
                    # Calculate min/max range (+/- 5%)
                    min_pred = daily_price * 0.95
                    max_pred = daily_price * 1.05
                    
                    preds.append({
                        "date": pred_date, 
                        "predicted_value": round(max(daily_price, 0), 2),
                        "min_price": round(max(min_pred, 0), 2),
                        "max_price": round(max(max_pred, 0), 2)
                    })
                return preds

            self.model_cache.save_model(model, market, commodity, variant)

        scaled = (series - mean_val) / std_val
        last_seq = torch.tensor(scaled[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)

        preds = []
        last_price = float(series[-1])

        model.to(self.device)
        model.eval()

        for day_i in range(prediction_days):
            with torch.no_grad():
                pred_scaled_tensor = model(last_seq)
            pred_scaled = float(pred_scaled_tensor.cpu().numpy().reshape(-1)[0])
            pred = pred_scaled * std_val + mean_val

            pred = 0.6 * pred + 0.4 * last_price

            # Updated clipping
            min_allowed = last_price * 0.85
            max_allowed = last_price * 1.15
            pred = float(np.clip(pred, min_allowed, max_allowed))

            pred = float(np.clip(pred, 500, 60000))

            pred_date = (datetime.now().date() + timedelta(days=day_i + 1)).strftime("%Y-%m-%d")
            
            # Calculate min/max range (+/- 5%)
            min_pred = pred * 0.95
            max_pred = pred * 1.05
            
            preds.append({
                "date": pred_date, 
                "predicted_value": round(pred, 2),
                "min_price": round(min_pred, 2),
                "max_price": round(max_pred, 2)
            })

            new_point_scaled = (pred - mean_val) / std_val
            new_point_tensor = torch.tensor([[[new_point_scaled]]], dtype=torch.float32).to(self.device)
            last_seq = torch.cat([last_seq[:, 1:, :], new_point_tensor], dim=1)
            last_price = pred

        return preds


    def estimate_disease_risk(self, commodity, weather_forecast):
        """Estimate disease risk based on weather conditions"""
        condition = (weather_forecast.get("condition", "") or "").lower()
        temp = weather_forecast.get("temp", 0) or 0
        rain = weather_forecast.get("total_rain_mm", 0) or 0

        risk = 0.0
        reason = "No elevated disease risk detected."

        if "chilli" in commodity.lower():
            if rain > 10 and 25 <= temp <= 35:
                risk = 0.8
                reason = "Heavy rain + warm temps — high risk for fungal diseases (e.g., anthracnose, leaf spot)."
            elif rain > 3:
                risk = 0.5
                reason = "Moderate rain — moderate fungal disease risk."
            elif "humid" in condition or "overcast" in condition:
                risk = 0.4
                reason = "Humid/overcast conditions — moderate fungal pressure."

        elif "cotton" in commodity.lower():
            if rain > 5 and temp > 28:
                risk = 0.7
                reason = "Rain + heat may cause boll rot or quality issues."
            elif "humid" in condition:
                risk = 0.5
                reason = "Humid weather — moderate disease risk."

        elif "groundnut" in commodity.lower() or "peanut" in commodity.lower():
            if rain > 10:
                risk = 0.9
                reason = "Excess rain — very high risk of root/pod rot."
            elif rain > 3:
                risk = 0.5
                reason = "Light-moderate rain — some soil-borne disease risk."

        elif "turmeric" in commodity.lower() or "paddy" in commodity.lower() or "maize" in commodity.lower():
            if rain > 8 and temp >= 20:
                risk = 0.6
                reason = "Wet warm conditions — fungal/bacterial disease risk."
            elif "rain" in condition:
                risk = 0.4
                reason = "Rainy conditions — increased disease likelihood."

        else:
            if rain > 10 or "rain" in condition:
                risk = 0.4
                reason = "Rainy/humid weather — mild fungal disease risk."

        risk = float(max(0.0, min(1.0, risk)))
        return risk, reason

    def get_environmental_adjustment(self, commodity, weather_forecast, sentiment_score):
        """Calculate combined adjustments from weather, disease, and sentiment"""
        condition = (weather_forecast.get("condition", "") or "").lower()
        total_rain = weather_forecast.get("total_rain_mm", 0) or 0
        is_rain = bool(weather_forecast.get("will_it_rain", False))

        weather_adj = 0.0
        weather_msgs = []

        # Weather-based adjustments
        if "chilli" in commodity.lower():
            if (is_rain and total_rain > 5) or "rain" in condition:
                weather_adj = -0.15
                weather_msgs.append("Heavy/consistent rain slows drying and reduces quality.")
            elif (is_rain and total_rain > 0.5) or "drizzle" in condition:
                weather_adj = -0.05
                weather_msgs.append("Light rain/drizzle — minor impact on drying.")
            elif "clear" in condition or "sunny" in condition:
                weather_adj = 0.05
                weather_msgs.append("Clear skies — good for drying, likely quality uplift.")
            elif "cloudy" in condition or "overcast" in condition:
                weather_adj = -0.02
                weather_msgs.append("Cloudy — mild negative impact on drying/colour.")

        elif "cotton" in commodity.lower():
            if (is_rain and total_rain > 2) or "rain" in condition:
                weather_adj = -0.20
                weather_msgs.append("Rain on open bolls may lower grade and price significantly.")
            elif "drought" in condition:
                weather_adj = 0.10
                weather_msgs.append("Dry conditions can reduce supply — price may increase.")

        elif "groundnut" in commodity.lower() or "peanut" in commodity.lower():
            if total_rain > 10:
                weather_adj = -0.15
                weather_msgs.append("Heavy rain during harvest can cause rot/sprouting — negative for prices.")
            elif total_rain > 0.5:
                weather_adj = -0.05
                weather_msgs.append("Light rain — can increase supply (slight negative effect).")

        else:
            if "rain" in condition or total_rain > 5:
                weather_adj = -0.05
                weather_msgs.append("Rainy conditions may reduce quality or delay market arrivals.")
            elif "sunny" in condition:
                weather_adj = 0.03
                weather_msgs.append("Sunny conditions are generally favorable for quality.")

        # Disease risk assessment
        disease_risk, disease_msg = self.estimate_disease_risk(commodity, weather_forecast)
        disease_adj = -0.20 * disease_risk

        # Sentiment-based adjustment
        # Scale sentiment (-1 to +1) to price adjustment (-10% to +10%)
        sentiment_adj = float(np.clip(sentiment_score * 0.10, -0.10, 0.10))
        
        if abs(sentiment_score) < 0.05:
            sentiment_msg = "News sentiment neutral — minimal market impact."
        elif sentiment_score > 0.3:
            sentiment_msg = f"Strong positive news sentiment ({sentiment_score:.2f}) — bullish market outlook, upward price pressure."
        elif sentiment_score > 0.1:
            sentiment_msg = f"Positive news sentiment ({sentiment_score:.2f}) — favorable market conditions."
        elif sentiment_score < -0.3:
            sentiment_msg = f"Strong negative news sentiment ({sentiment_score:.2f}) — bearish outlook, downward price pressure."
        elif sentiment_score < -0.1:
            sentiment_msg = f"Negative news sentiment ({sentiment_score:.2f}) — unfavorable market conditions."
        else:
            sentiment_msg = f"Mild news sentiment ({sentiment_score:.2f}) — limited market impact."

        weather_msg = " ".join(weather_msgs) if weather_msgs else "Weather conditions neutral."

        info = {
            'weather_adj': float(weather_adj),
            'weather_msg': weather_msg,
            'disease_risk': float(disease_risk),
            'disease_adj': float(disease_adj),
            'disease_msg': disease_msg,
            'sentiment_adj': float(sentiment_adj),
            'sentiment_msg': sentiment_msg
        }

        return info

# ========== FASTAPI APPLICATION ==========

app = FastAPI(
    title="Agricultural Price Prediction API with Sentiment Analysis",
    description="GRU-based price prediction with weather, disease risk, and news sentiment analysis",
    version="2.0.0"
)

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize data and predictor on startup"""
    global predictor
    print("\n" + "=" * 50)
    print("🚀 Starting Agricultural Price Prediction API v2.0")
    print("   Features: GRU Model + Weather + Disease + Sentiment")
    print("=" * 50)
    
    # Load or refresh data cache
    if not get_or_update_db_data():
        print("❌ CRITICAL: Failed to load data. Exiting.")
        exit()
    
    # Initialize predictor
    print("\n📊 Initializing Enhanced Price Predictor...")
    predictor = GRUPricePredictor()
    print("✅ API Ready!\n")

@app.get("/")
async def root():
    return {
        "message": "Agricultural Price Prediction API v2.0",
        "features": ["GRU Model", "Weather Analysis", "Disease Risk", "News Sentiment"],
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "predict": "/api/predict",
            "variants": "/api/variants",
            "ncdex": "/api/ncdex",
            "sentiment": "/api/sentiment/{commodity}"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with sentiment analysis status"""
    cache_exists = os.path.exists(DB_CACHE_FILE)
    cache_age = None
    
    if cache_exists:
        file_mod_time = os.path.getmtime(DB_CACHE_FILE)
        cache_age = (time.time() - file_mod_time) / 3600
    
    sentiment_cache_exists = os.path.exists(NEWS_CACHE_FILE)
    sentiment_cache_age = None
    
    if sentiment_cache_exists:
        file_mod_time = os.path.getmtime(NEWS_CACHE_FILE)
        sentiment_cache_age = (time.time() - file_mod_time) / 3600
    
    return HealthResponse(
        status="healthy" if predictor is not None else "initializing",
        timestamp=datetime.now().isoformat(),
        cache_status={
            "exists": cache_exists,
            "age_hours": round(cache_age, 2) if cache_age else None,
            "is_fresh": cache_age < 24 if cache_age else False
        },
        sentiment_status={
            "enabled": NEWS_API_KEY is not None,
            "cache_exists": sentiment_cache_exists,
            "cache_age_hours": round(sentiment_cache_age, 2) if sentiment_cache_age else None
        }
    )

@app.get("/api/variants", response_model=VariantsResponse)
async def get_variants(
    commodity: str = Query(..., description="Commodity name"),
    market: str = Query(..., description="Market name")
):
    """Get available variants for a commodity in a market"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    hist_data, variants = predictor.get_available_variants(market, commodity)
    
    if hist_data is None or variants is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for commodity '{commodity}' in market '{market}'"
        )
    
    return VariantsResponse(
        commodity=commodity.title(),
        market=market.title(),
        variants=list(variants),  # ✅ Use raw variants exactly as they exist
        count=len(variants)
    )

@app.get("/api/sentiment/{commodity}")
async def get_sentiment(commodity: str, force_refresh: bool = False):
    """Get news sentiment analysis for a commodity"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        sentiment_data = predictor.sentiment_analyzer.get_sentiment_analysis(
            commodity, force_refresh=force_refresh
        )
        return {
            "commodity": commodity.title(),
            "sentiment": sentiment_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sentiment: {str(e)}")

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_prices(request: PredictionRequest):
    """Predict prices with weather, disease, and sentiment analysis"""
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    market = clean_amc(request.market)
    commodity = clean_commodity(request.commodity)
    prediction_days = request.prediction_days
    specific_variant = request.variant
    
    # Get variants
    hist_data, variants = predictor.get_available_variants(market, commodity)
    
    if hist_data is None or variants is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for commodity '{commodity}' in market '{market}'"
        )
    
    # If user selected a specific variant
    if specific_variant:
        v_lower = specific_variant.lower()
        matching_variants = [v for v in variants if v_lower in v.lower()]
        
        if not matching_variants:
            raise HTTPException(
                status_code=404,
                detail=f"Variant '{specific_variant}' not found. Available: {[v.title() for v in variants]}"
            )
        
        variants = matching_variants

    # --------------------------------------------------------------------
    # 🔥 WEATHER FETCHING (Fixed + WeatherHelper)
    # --------------------------------------------------------------------
    if predictor.weather_helper:
        days_to_fetch = min(prediction_days, 14)
        weather_forecast_list = []

        for day_offset in range(days_to_fetch):
            forecast_date = (
                datetime.now().date() + timedelta(days=day_offset + 1)
            ).strftime("%Y-%m-%d")

            weather_data = predictor.weather_helper.get_weather_input(
                district=market,
                date=forecast_date,
                commodity=commodity,
                include_adjustment=True
            )

            weather_forecast_list.append({
                "date": forecast_date,
                "temp": weather_data["features"]["max_temp"],
                "condition": weather_data["summary"],
                "total_rain_mm": weather_data["features"]["rainfall"],
                "will_it_rain": weather_data["features"]["rainfall"] > 0,
                "weather_adjustment": weather_data["adjustment_factor"],
                "weather_message": weather_data["adjustment_message"],
            })

    else:
        # Fallback to old 3-day WeatherAPI
        days_to_fetch = min(prediction_days, 3)
        weather_forecast_list = predictor.weather_fetcher.get_weather_forecast(
            market, days=days_to_fetch
        ) if predictor.weather_fetcher else []

    # --------------------------------------------------------------------
    # 🔥 SENTIMENT ANALYSIS
    # --------------------------------------------------------------------
    print(f"📰 Analyzing news sentiment for {commodity}...")
    sentiment_data = predictor.sentiment_analyzer.get_sentiment_analysis(commodity)
    sentiment_score = sentiment_data.get("avg_sentiment", 0.0)

    variant_forecasts = []

    # --------------------------------------------------------------------
    # 🔥 PROCESS EACH VARIANT
    # --------------------------------------------------------------------
    for variant in variants:

        variant_df = hist_data[hist_data["commodity_name"] == variant]

        baseline_preds = predictor.predict_for_single_variant(
            variant_df, variant, market, commodity, prediction_days
        )

        forecasts = []

        for i, p in enumerate(baseline_preds):
            baseline_price = p["predicted_value"]

            # -----------------------------
            # WEATHER + DISEASE + SENTIMENT
            # -----------------------------
            if i < len(weather_forecast_list):

                weather_for_this_day = weather_forecast_list[i]

                # ---------- WeatherHelper Active ----------
                if predictor.weather_helper and "weather_adjustment" in weather_for_this_day:
                    weather_multiplier = weather_for_this_day["weather_adjustment"]
                    weather_adj = weather_multiplier - 1.0
                    weather_msg = weather_for_this_day["weather_message"]

                    # Disease risk based on rainfall/temp
                    disease_risk, disease_msg = predictor.estimate_disease_risk(
                        commodity, weather_for_this_day
                    )
                    disease_adj = -disease_risk * 0.15

                else:
                    # ---------- OLD Weather System Fallback ----------
                    adjustments = predictor.get_environmental_adjustment(
                        commodity, weather_for_this_day, sentiment_score
                    )
                    weather_adj = adjustments["weather_adj"]
                    disease_risk = adjustments["disease_risk"]
                    disease_adj = adjustments["disease_adj"]
                    weather_msg = adjustments["weather_msg"]
                    disease_msg = adjustments["disease_msg"]

                # Sentiment adj
                sentiment_adj = float(np.clip(sentiment_score * 0.10, -0.10, 0.10))
                sentiment_msg = f"Sentiment score: {sentiment_score:.2f}"

                total_multiplier = 1 + weather_adj + disease_adj + sentiment_adj

                final_price = baseline_price * total_multiplier
                min_price = p["min_price"] * total_multiplier
                max_price = p["max_price"] * total_multiplier

            else:
                # -----------------------------
                # No weather data → sentiment only
                # -----------------------------
                weather_adj = 0
                disease_risk = 0
                disease_adj = 0
                weather_msg = "No weather data (API limit)"
                disease_msg = "No disease assessment available"

                sentiment_adj = float(np.clip(sentiment_score * 0.10, -0.10, 0.10))
                sentiment_msg = f"Sentiment score: {sentiment_score:.2f}"

                total_multiplier = 1 + sentiment_adj

                final_price = baseline_price * total_multiplier
                min_price = p["min_price"] * total_multiplier
                max_price = p["max_price"] * total_multiplier

            # -----------------------------
            # Add formatted forecast result
            # -----------------------------
            forecasts.append(PriceForecast(
                date=p["date"],
                baseline_price=round(baseline_price, 2),
                weather_adjustment=round(weather_adj * 100, 2),
                disease_risk=round(disease_risk, 2),
                disease_adjustment=round(disease_adj * 100, 2),
                sentiment_adjustment=round(sentiment_adj * 100, 2),
                final_price=round(final_price, 2),
                min_price=round(min_price, 2),
                max_price=round(max_price, 2),
                weather_reason=weather_msg,
                disease_reason=disease_msg,
                sentiment_reason=sentiment_msg
            ))

        # Attach sentiment block
        sentiment_info = SentimentInfo(
            avg_sentiment=round(sentiment_score, 3),
            sentiment_label=sentiment_data.get("sentiment_label", "Neutral"),
            news_count=sentiment_data.get("news_count", 0),
            top_keywords=sentiment_data.get("top_keywords", []),
            sample_headlines=sentiment_data.get("sample_headlines", [])
        )

        variant_forecasts.append(VariantForecast(
            variant=variant.title(),
            forecasts=forecasts,
            sentiment_info=sentiment_info
        ))

    # --------------------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------------------
    return PredictionResponse(
        commodity=commodity.title(),
        market=market.title(),
        prediction_days=prediction_days,
        variants=variant_forecasts,
        message=f"Successfully generated forecasts for {len(variants)} variant(s) with weather, disease & sentiment analysis."
    )


@app.post("/api/ncdex", response_model=NCDEXResponse)
async def get_ncdex_price(request: NCDEXRequest):
    """Get NCDEX spot price for a commodity"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    if predictor.ncdex_data is None or predictor.ncdex_data.empty:
        raise HTTPException(status_code=503, detail="NCDEX data not available")
    
    commodity = request.commodity.lower()
    
    try:
        ncdex_df = predictor.ncdex_data.copy()
        mask = ncdex_df.apply(lambda row: row.astype(str).str.lower().str.contains(commodity).any(), axis=1)
        found = ncdex_df[mask]
        
        if found.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Commodity '{request.commodity}' not found in NCDEX data"
            )
        
        possible_price_cols = [c for c in found.columns if any(x in c.lower() for x in ["ltp", "last", "close", "price", "spot"])]
        
        if possible_price_cols:
            spot_prices = pd.to_numeric(
                found[possible_price_cols[0]].str.replace(',', '').str.extract(r'(\d+\.?\d*)')[0],
                errors='coerce'
            ).dropna().values
        else:
            def find_first_num(row):
                for v in row:
                    try:
                        return float(str(v).replace(',', ''))
                    except:
                        continue
                return np.nan
            spot_prices = found.apply(find_first_num, axis=1).dropna().values
        
        if len(spot_prices) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Could not extract price for '{request.commodity}'"
            )
        
        return NCDEXResponse(
            commodity=request.commodity.title(),
            spot_price=round(float(spot_prices[0]), 2),
            message="NCDEX spot price retrieved"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/markets")
async def get_markets():
    """Get list of available markets"""
    return {"markets": MARKETS_TO_SCRAPE, "count": len(MARKETS_TO_SCRAPE)}

@app.get("/api/commodities")
async def get_commodities():
    """Get list of available commodities"""
    return {"commodities": COMMODITIES_TO_SCRAPE, "count": len(COMMODITIES_TO_SCRAPE)}

# ========== RUN SERVER ==========

if __name__ == "__main__":
    import sys
    
    if "--train" in sys.argv:
        print("--- 🌙 BATCH TRAINING MODE ---")
        
        if not get_or_update_db_data():
            exit()
        
        model = GRUPricePredictor()
        
        total_start = datetime.now()
        
        for market in MARKETS_TO_SCRAPE:
            for commodity in COMMODITIES_TO_SCRAPE:
                print(f"\n--- Training: {commodity} in {market} ---")
                try:
                    hist_data, variants = model.get_available_variants(market.lower(), commodity.lower())
                    if hist_data is not None:
                        for variant in variants:
                            variant_df = hist_data[hist_data['commodity_name'] == variant]
                            model.predict_for_single_variant(
                                variant_df, variant, market.lower(), commodity.lower(), 1
                            )
                except Exception as e:
                    print(f"⚠️ Error: {e}")
        
        print(f"\n✅ Training complete in {datetime.now() - total_start}")
    
    else:
        uvicorn.run("price_tool:app", host="0.0.0.0", port=8002, reload=False)