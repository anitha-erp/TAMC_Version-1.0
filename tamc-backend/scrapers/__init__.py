# scrapers/__init__.py
"""
Agricultural data scrapers package

Contains scrapers for:
- NCDEX commodity prices
- Napanta mandi prices
- News sentiment analysis
"""

__version__ = "2.0.0"

# Import main functions for easy access
try:
    from .ncdex_scrape import get_or_update_ncdex_data
except ImportError:
    print("⚠️ ncdex_scrape not available")
    get_or_update_ncdex_data = None

try:
    from .naapanta_scrape import get_historical_prices_for_prediction
except ImportError:
    print("⚠️ naapanta_scrape not available")
    get_historical_prices_for_prediction = None

try:
    from .news_sentiment import (
        get_news_sentiment_score,
        get_sentiment_summary,
        fetch_recent_news,
        analyze_news_sentiment
    )
except ImportError:
    print("⚠️ news_sentiment not available")
    get_news_sentiment_score = None
    get_sentiment_summary = None
    fetch_recent_news = None
    analyze_news_sentiment = None

__all__ = [
    'get_or_update_ncdex_data',
    'get_historical_prices_for_prediction',
    'get_news_sentiment_score',
    'get_sentiment_summary',
    'fetch_recent_news',
    'analyze_news_sentiment'
]