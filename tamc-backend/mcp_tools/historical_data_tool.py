# historical_data_tool.py - Query actual historical data from database/CSV
# Enables queries like "what was the price yesterday" or "show arrivals from last week"

import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import os

# Path to the merged data cache
DB_CACHE_FILE = "merged_lots_data.csv"

class HistoricalDataTool:
    """Tool for querying actual historical price and arrival data"""
    
    def __init__(self):
        """Initialize with cached historical data"""
        if os.path.exists(DB_CACHE_FILE):
            print(f"📊 Loading historical data from {DB_CACHE_FILE}...")
            self.data = pd.read_csv(DB_CACHE_FILE, parse_dates=["date"])
            print(f"✅ Loaded {len(self.data)} historical records")
        else:
            print(f"⚠️ Warning: {DB_CACHE_FILE} not found")
            self.data = pd.DataFrame()
    
    def parse_historical_date(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Parse natural language date references into start_date and end_date.
        
        Supports:
        - "yesterday" -> yesterday's date
        - "last week" -> last 7 days
        - "last month" -> last 30 days
        - "N days ago" -> N days ago
        - "on [date]" -> specific date
        
        Returns:
            Tuple of (start_date, end_date) or None if no date found
        """
        query_lower = query.lower()
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Yesterday
        if "yesterday" in query_lower:
            yesterday = today - timedelta(days=1)
            return (yesterday, yesterday)
        
        # Last week (7 days)
        if "last week" in query_lower or "past week" in query_lower:
            start_date = today - timedelta(days=7)
            end_date = today - timedelta(days=1)  # Up to yesterday
            return (start_date, end_date)
        
        # Last month (30 days)
        if "last month" in query_lower or "past month" in query_lower:
            start_date = today - timedelta(days=30)
            end_date = today - timedelta(days=1)
            return (start_date, end_date)
        
        # N days ago (e.g., "3 days ago", "5 days ago")
        days_ago_match = re.search(r'(\d+)\s+days?\s+ago', query_lower)
        if days_ago_match:
            n_days = int(days_ago_match.group(1))
            target_date = today - timedelta(days=n_days)
            return (target_date, target_date)
        
        # Last N days (e.g., "last 5 days", "past 3 days")
        last_n_days_match = re.search(r'(?:last|past)\s+(\d+)\s+days?', query_lower)
        if last_n_days_match:
            n_days = int(last_n_days_match.group(1))
            start_date = today - timedelta(days=n_days)
            end_date = today - timedelta(days=1)
            return (start_date, end_date)
        
        # Specific weekday (e.g., "last monday", "last friday")
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        for day_name, day_num in weekdays.items():
            if f"last {day_name}" in query_lower:
                # Find the most recent occurrence of that weekday
                days_back = (today.weekday() - day_num) % 7
                if days_back == 0:  # If today is that day, go back 7 days
                    days_back = 7
                target_date = today - timedelta(days=days_back)
                return (target_date, target_date)
        
        # No date pattern found
        return None
    
    def get_historical_prices(
        self,
        commodity: str,
        market: str,
        start_date: datetime,
        end_date: datetime,
        variant: Optional[str] = None
    ) -> Dict:
        """
        Retrieve historical price data for a commodity in a market.
        
        Args:
            commodity: Commodity name (e.g., "cotton", "chilli")
            market: Market/AMC name (e.g., "warangal", "khammam")
            start_date: Start date for query
            end_date: End date for query
            variant: Optional specific variant
        
        Returns:
            Dict with success status and historical price data
        """
        if self.data.empty:
            return {
                "success": False,
                "error": "Historical data not available"
            }
        
        # Normalize inputs
        commodity_clean = commodity.lower().strip()
        market_clean = market.lower().strip()
        
        # Filter data
        df = self.data.copy()
        df['amc_name'] = df['amc_name'].astype(str).str.lower()
        df['commodity_name'] = df['commodity_name'].astype(str).str.lower()
        
        # Filter by market and commodity
        df_filtered = df[
            df['amc_name'].str.contains(market_clean, na=False) &
            df['commodity_name'].str.contains(commodity_clean, na=False)
        ]
        
        # Filter by variant if specified
        if variant:
            variant_clean = variant.lower().strip()
            df_filtered = df_filtered[
                df_filtered['commodity_name'].str.contains(variant_clean, na=False)
            ]
        
        # Filter by date range
        df_filtered = df_filtered[
            (df_filtered['date'] >= start_date) &
            (df_filtered['date'] <= end_date)
        ]
        
        if df_filtered.empty:
            return {
                "success": False,
                "error": f"No historical data found for {commodity} in {market} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
            }
        
        # Format response similar to prediction tool
        historical_data = []
        for _, row in df_filtered.iterrows():
            historical_data.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "actual_price": float(row['avg_price']) if pd.notna(row['avg_price']) else 0,
                "min_price": float(row['min_price']) if pd.notna(row['min_price']) else 0,
                "max_price": float(row['max_price']) if pd.notna(row['max_price']) else 0,
                "commodity_name": row['commodity_name']
            })
        
        # Sort by date
        historical_data.sort(key=lambda x: x['date'])
        
        return {
            "success": True,
            "data": {
                "commodity": commodity,
                "market": market,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "historical_prices": historical_data,
                "count": len(historical_data)
            }
        }
    
    def get_historical_arrivals(
        self,
        commodity: str,
        market: str,
        start_date: datetime,
        end_date: datetime,
        metric: str = "arrivals",
        variant: Optional[str] = None
    ) -> Dict:
        """
        Retrieve historical arrival data for a commodity in a market.
        
        Args:
            commodity: Commodity name (e.g., "cotton", "chilli")
            market: Market/AMC name (e.g., "warangal", "khammam")
            start_date: Start date for query
            end_date: End date for query
            metric: Metric to retrieve (arrivals, bags, etc.)
            variant: Optional specific variant
        
        Returns:
            Dict with success status and historical arrival data
        """
        if self.data.empty:
            return {
                "success": False,
                "error": "Historical data not available"
            }
        
        # Normalize inputs
        commodity_clean = commodity.lower().strip()
        market_clean = market.lower().strip()
        
        # Filter data
        df = self.data.copy()
        df['amc_name'] = df['amc_name'].astype(str).str.lower()
        df['commodity_name'] = df['commodity_name'].astype(str).str.lower()
        
        # Filter by market and commodity
        df_filtered = df[
            df['amc_name'].str.contains(market_clean, na=False) &
            df['commodity_name'].str.contains(commodity_clean, na=False)
        ]
        
        # Filter by variant if specified
        if variant:
            variant_clean = variant.lower().strip()
            df_filtered = df_filtered[
                df_filtered['commodity_name'].str.contains(variant_clean, na=False)
            ]
        
        # Filter by date range
        df_filtered = df_filtered[
            (df_filtered['date'] >= start_date) &
            (df_filtered['date'] <= end_date)
        ]
        
        if df_filtered.empty:
            return {
                "success": False,
                "error": f"No historical data found for {commodity} in {market} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
            }
        
        # Format response similar to arrival tool
        historical_data = []
        for _, row in df_filtered.iterrows():
            historical_data.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "actual_arrivals": float(row['arrivals']) if pd.notna(row['arrivals']) else 0,
                "commodity_name": row['commodity_name']
            })
        
        # Sort by date
        historical_data.sort(key=lambda x: x['date'])
        
        return {
            "success": True,
            "data": {
                "commodity": commodity,
                "market": market,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "metric": metric,
                "historical_arrivals": historical_data,
                "count": len(historical_data)
            }
        }
    
    def query_historical_data(self, params: Dict) -> Dict:
        """
        Main entry point for historical data queries.
        
        Args:
            params: Dictionary with query parameters including:
                - query: Original user query
                - commodity: Commodity name
                - market/district/amc_name: Market location
                - variant: Optional variant
                - query_type: "price" or "arrival"
        
        Returns:
            Dict with success status and historical data
        """
        query = params.get("query", "")
        commodity = params.get("commodity", "")
        market = params.get("market") or params.get("district") or params.get("amc_name", "")
        variant = params.get("variant")
        query_type = params.get("query_type", "price")  # Default to price
        
        # Parse date from query
        date_range = self.parse_historical_date(query)
        
        if not date_range:
            return {
                "success": False,
                "error": "Could not parse date from query. Try: 'yesterday', 'last week', '3 days ago'"
            }
        
        start_date, end_date = date_range
        
        if not commodity or not market:
            return {
                "success": False,
                "error": "Please specify both commodity and market for historical query"
            }
        
        # Route to appropriate handler
        if query_type == "arrival":
            metric = params.get("metric", "arrivals")
            return self.get_historical_arrivals(
                commodity, market, start_date, end_date, metric, variant
            )
        else:  # Default to price
            return self.get_historical_prices(
                commodity, market, start_date, end_date, variant
            )


# Singleton instance
_historical_tool = None

def get_historical_tool() -> HistoricalDataTool:
    """Get or create singleton instance of historical data tool"""
    global _historical_tool
    if _historical_tool is None:
        _historical_tool = HistoricalDataTool()
    return _historical_tool


# Convenience function for MCP server
def query_historical_data(params: Dict) -> Dict:
    """Query historical data - main entry point for MCP server"""
    tool = get_historical_tool()
    return tool.query_historical_data(params)
