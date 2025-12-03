# scrapers/news_sentiment.py
# News sentiment scraper for agricultural commodities
# Uses NewsAPI + TextBlob for sentiment analysis + KeyBERT for keyword extraction

import requests
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
import json
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional: KeyBERT for better keyword extraction
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("WARNING: KeyBERT not installed. Using simple keyword extraction.")

# Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
CACHE_DIR = "news_cache"
CACHE_DURATION_HOURS = 6  # Cache news for 6 hours

os.makedirs(CACHE_DIR, exist_ok=True)

# ========== CACHING FUNCTIONS ==========

def get_cache_path(commodity):
    """Get cache file path for a commodity"""
    safe_name = commodity.lower().replace(" ", "_").replace("-", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}_news.json")

def is_cache_fresh(commodity, max_age_hours=CACHE_DURATION_HOURS):
    """Check if cached news data is still fresh"""
    cache_path = get_cache_path(commodity)
    
    if not os.path.exists(cache_path):
        return False
    
    file_age = time.time() - os.path.getmtime(cache_path)
    file_age_hours = file_age / 3600
    
    return file_age_hours < max_age_hours

def load_from_cache(commodity):
    """Load news data from cache"""
    cache_path = get_cache_path(commodity)
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert back to DataFrame
        if 'articles' in data and data['articles']:
            df = pd.DataFrame(data['articles'])
            print(f"✅ Loaded {len(df)} cached articles for '{commodity}'")
            return df, data.get('metadata', {})
        return None, None
    except Exception as e:
        print(f"WARNING: Error loading cache: {e}")
        return None, None

def save_to_cache(commodity, df, metadata):
    """Save news data to cache"""
    cache_path = get_cache_path(commodity)
    
    try:
        # Convert DataFrame to dict
        articles = df.to_dict('records') if df is not None and not df.empty else []
        
        cache_data = {
            'commodity': commodity,
            'timestamp': datetime.now().isoformat(),
            'articles': articles,
            'metadata': metadata
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Cached {len(articles)} articles for '{commodity}'")
    except Exception as e:
        print(f"WARNING: Error saving cache: {e}")

# ========== NEWS FETCHING ==========

def fetch_recent_news(query="agriculture", days_back=7, page_size=50):
    """
    Fetch recent news related to a keyword using NewsAPI.
    
    Args:
        query: Search term (commodity name)
        days_back: Number of days to look back
        page_size: Number of articles to fetch (max 100)
    
    Returns:
        DataFrame with news articles or None if error
    """
    if not NEWS_API_KEY:
        print("❌ NEWS_API_KEY not set")
        return None
    
    url = "https://newsapi.org/v2/everything"
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "pageSize": min(page_size, 100),  # API limit
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        if data.get("status") != "ok":
            print(f"WARNING: News API error: {data.get('message', 'Unknown error')}")
            return None

        articles = data.get("articles", [])
        
        if not articles:
            print(f"📰 No articles found for '{query}'")
            return None
        
        print(f"📰 Fetched {len(articles)} articles for '{query}'")
        
        df = pd.DataFrame([{
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "url": a.get("url", ""),
            "publishedAt": a.get("publishedAt", ""),
            "source": a.get("source", {}).get("name", "Unknown"),
            "author": a.get("author", "Unknown")
        } for a in articles])
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching news: {e}")
        return None
    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return None

# ========== SENTIMENT ANALYSIS ==========

def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob.
    
    Returns:
        float: Sentiment polarity (-1 to +1)
    """
    if not text or pd.isna(text):
        return 0.0
    
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity
    except Exception:
        return 0.0

def extract_keywords_keybert(text, top_n=3):
    """Extract keywords using KeyBERT (advanced)"""
    if not KEYBERT_AVAILABLE or not text or pd.isna(text):
        return []
    
    try:
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(str(text), top_n=top_n)
        return [k for k, _ in keywords]
    except Exception:
        return []

def extract_keywords_simple(text, top_n=5):
    """Extract keywords using simple frequency analysis"""
    if not text or pd.isna(text):
        return []
    
    import re
    from collections import Counter
    
    # Common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
        'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them'
    }
    
    # Extract words (4+ characters)
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    words = [w for w in words if w not in stop_words]
    
    # Count frequency
    word_freq = Counter(words)
    
    return [word for word, _ in word_freq.most_common(top_n)]

def analyze_news_sentiment(news_df, use_keybert=True):
    """
    Perform sentiment analysis and keyword extraction on news articles.
    
    Args:
        news_df: DataFrame with news articles
        use_keybert: Whether to use KeyBERT (if available)
    
    Returns:
        DataFrame with added sentiment columns
    """
    if news_df is None or news_df.empty:
        print("❌ No news articles to analyze")
        return None

    print(f"🔍 Analyzing sentiment for {len(news_df)} articles...")
    
    sentiments = []
    keywords_list = []

    for _, row in news_df.iterrows():
        # Combine title and description for analysis
        text = f"{row.get('title', '')} {row.get('description', '')}"
        
        # Sentiment analysis
        sentiment = analyze_sentiment(text)
        sentiments.append(sentiment)
        
        # Keyword extraction
        if use_keybert and KEYBERT_AVAILABLE:
            keywords = extract_keywords_keybert(text, top_n=3)
        else:
            keywords = extract_keywords_simple(text, top_n=3)
        
        keywords_list.append(", ".join(keywords) if keywords else "")

    # Add columns to DataFrame
    news_df["sentiment"] = sentiments
    news_df["keywords"] = keywords_list
    news_df["sentiment_label"] = news_df["sentiment"].apply(
        lambda s: "Positive" if s > 0.1 else ("Negative" if s < -0.1 else "Neutral")
    )

    print("✅ Sentiment analysis complete")
    return news_df

# ========== MAIN FUNCTION ==========

def get_news_sentiment_score(commodity_name, days_back=7, page_size=50, force_refresh=False):
    """
    Fetch and analyze news for a commodity, return sentiment score and DataFrame.
    Uses caching to avoid excessive API calls.
    
    Args:
        commodity_name: Name of the commodity
        days_back: Number of days to look back
        page_size: Number of articles to fetch
        force_refresh: Force refresh even if cache is fresh
    
    Returns:
        tuple: (average_sentiment_score, news_dataframe, metadata_dict)
    """
    print(f"\n{'='*60}")
    print(f"📊 News Sentiment Analysis: {commodity_name.upper()}")
    print(f"{'='*60}")
    
    # Check cache first
    if not force_refresh and is_cache_fresh(commodity_name):
        df, metadata = load_from_cache(commodity_name)
        if df is not None:
            avg_sentiment = df["sentiment"].mean()
            print(f"📈 Average sentiment (cached): {avg_sentiment:.3f}")
            return avg_sentiment, df, metadata
    
    # Fetch fresh news
    print(f"🌐 Fetching fresh news for '{commodity_name}'...")
    df = fetch_recent_news(query=commodity_name, days_back=days_back, page_size=page_size)
    
    if df is None or df.empty:
        print(f"WARNING: No news available for '{commodity_name}'")
        return 0.0, None, {"error": "No news found"}
    
    # Analyze sentiment
    df = analyze_news_sentiment(df, use_keybert=KEYBERT_AVAILABLE)
    
    if df is None or df.empty:
        return 0.0, None, {"error": "Sentiment analysis failed"}
    
    # Calculate statistics
    avg_sentiment = float(df["sentiment"].mean())
    
    metadata = {
        "commodity": commodity_name,
        "avg_sentiment": round(avg_sentiment, 3),
        "sentiment_label": "Positive" if avg_sentiment > 0.1 else (
            "Negative" if avg_sentiment < -0.1 else "Neutral"
        ),
        "total_articles": len(df),
        "positive_count": len(df[df["sentiment"] > 0.1]),
        "negative_count": len(df[df["sentiment"] < -0.1]),
        "neutral_count": len(df[(df["sentiment"] >= -0.1) & (df["sentiment"] <= 0.1)]),
        "date_range": f"{days_back} days",
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    # Save to cache
    save_to_cache(commodity_name, df, metadata)
    
    # Print summary
    print(f"\n📈 Sentiment Summary:")
    print(f"   Average: {avg_sentiment:.3f} ({metadata['sentiment_label']})")
    print(f"   Positive: {metadata['positive_count']} articles")
    print(f"   Negative: {metadata['negative_count']} articles")
    print(f"   Neutral: {metadata['neutral_count']} articles")
    
    # Show top keywords
    all_keywords = []
    for kw_str in df["keywords"].dropna():
        all_keywords.extend(kw_str.split(", "))
    
    if all_keywords:
        from collections import Counter
        top_keywords = Counter(all_keywords).most_common(5)
        print(f"\n🔑 Top Keywords: {', '.join([k for k, _ in top_keywords])}")
    
    print(f"{'='*60}\n")
    
    return avg_sentiment, df, metadata

# ========== UTILITY FUNCTIONS ==========

def get_sentiment_summary(commodity_name, force_refresh=False):
    """
    Get a quick sentiment summary without full DataFrame.
    
    Returns:
        dict with sentiment statistics
    """
    avg, df, metadata = get_news_sentiment_score(
        commodity_name, 
        force_refresh=force_refresh
    )
    
    if df is None:
        return {
            "commodity": commodity_name,
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "article_count": 0,
            "error": "No data available"
        }
    
    return metadata

def clear_cache(commodity_name=None):
    """Clear news cache for a specific commodity or all commodities"""
    if commodity_name:
        cache_path = get_cache_path(commodity_name)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"🗑️ Cleared cache for {commodity_name}")
    else:
        # Clear all cache files
        for file in os.listdir(CACHE_DIR):
            if file.endswith("_news.json"):
                os.remove(os.path.join(CACHE_DIR, file))
        print("🗑️ Cleared all news cache")

# ========== COMMAND LINE INTERFACE ==========

if __name__ == "__main__":
    import sys
    
    # Default test commodity
    test_commodity = "tomato" if len(sys.argv) < 2 else sys.argv[1]
    
    # Check for force refresh flag
    force_refresh = "--refresh" in sys.argv or "-r" in sys.argv
    
    # Run analysis
    avg_sentiment, df, metadata = get_news_sentiment_score(
        test_commodity,
        force_refresh=force_refresh
    )
    
    if df is not None and not df.empty:
        print("\n📋 Sample Articles:")
        print(df[["title", "sentiment", "sentiment_label", "source"]].head(10).to_string(index=False))
        
        print(f"\n💾 Full data saved to: {get_cache_path(test_commodity)}")
    else:
        print(f"\n❌ No data available for '{test_commodity}'")
    
    print(f"\n📊 Final Sentiment Score: {avg_sentiment:.3f}")
