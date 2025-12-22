# historical_data_tool.py - Query actual historical data from database/CSV
# Enables queries like "what was the price yesterday" or "show arrivals from last week"

import pandas as pd
import re
from datetime import datetime, timedelta
import calendar
from typing import Dict, Optional, Tuple, List
import os
import pymysql
import pymysql.cursors

# Path to the merged data cache
DB_CACHE_FILE = "merged_lots_data.csv"

# Database configuration (same as arrival_tool.py)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
    "cursorclass": pymysql.cursors.DictCursor,
}

def get_db_connection():
    """Get database connection"""
    return pymysql.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        db=DB_CONFIG["database"],
        port=DB_CONFIG["port"],
        cursorclass=DB_CONFIG["cursorclass"]
    )

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
        
        # Today
        if "today" in query_lower:
            return (today, today)

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
        
        # Specific date parsing (e.g., "1st dec 2025", "on 1 december", "december 1")
        # Month names mapping
        months = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }
        
        # Pattern 1: "1st dec 2025", "2nd december 2024", "3rd jan"
        # Matches: 1st, 2nd, 3rd, 4th, 21st, 22nd, 23rd, 31st, etc.
        ordinal_pattern = r'(\d{1,2})(?:st|nd|rd|th)\s+(' + '|'.join(months.keys()) + r')(?:\s+(\d{4}))?'
        ordinal_match = re.search(ordinal_pattern, query_lower)
        if ordinal_match:
            day = int(ordinal_match.group(1))
            month_name = ordinal_match.group(2)
            year = int(ordinal_match.group(3)) if ordinal_match.group(3) else today.year
            month = months[month_name]
            
            try:
                target_date = datetime(year, month, day)
                return (target_date, target_date)
            except ValueError:
                pass  # Invalid date, continue to next pattern
        
        # Pattern 2: "december 1", "dec 15 2025", "january 20"
        # Pattern 2a: Month + Year (e.g., "oct 2024", "december 2023") -> entire month
        month_year_pattern = r'(' + '|'.join(months.keys()) + r')\s+(\d{4})'
        month_year_match = re.search(month_year_pattern, query_lower)
        if month_year_match:
            month_name = month_year_match.group(1)
            year = int(month_year_match.group(2))
            month = months[month_name]
            try:
                start_date = datetime(year, month, 1)
                last_day = calendar.monthrange(year, month)[1]
                end_date = datetime(year, month, last_day)
                return (start_date, end_date)
            except ValueError:
                pass

        # Pattern 2b: "december 1", "dec 15 2025", "january 20"
        month_day_pattern = r'(' + '|'.join(months.keys()) + r')\s+(\d{1,2})(?:\s+(\d{4}))?'
        month_day_match = re.search(month_day_pattern, query_lower)
        if month_day_match:
            month_name = month_day_match.group(1)
            day = int(month_day_match.group(2))
            year = int(month_day_match.group(3)) if month_day_match.group(3) else today.year
            month = months[month_name]
            
            try:
                target_date = datetime(year, month, day)
                return (target_date, target_date)
            except ValueError:
                pass
        
        # Pattern 3: "1 december", "15 dec 2025"
        day_month_pattern = r'(\d{1,2})\s+(' + '|'.join(months.keys()) + r')(?:\s+(\d{4}))?'
        day_month_match = re.search(day_month_pattern, query_lower)
        if day_month_match:
            day = int(day_month_match.group(1))
            month_name = day_month_match.group(2)
            year = int(day_month_match.group(3)) if day_month_match.group(3) else today.year
            month = months[month_name]
            
            try:
                target_date = datetime(year, month, day)
                return (target_date, target_date)
            except ValueError:
                pass
        
        # Pattern 4: Numeric dates "12/1/2025", "1-12-2025", "2025-12-01"
        # Try YYYY-MM-DD format first
        iso_pattern = r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})'
        iso_match = re.search(iso_pattern, query_lower)
        if iso_match:
            year = int(iso_match.group(1))
            month = int(iso_match.group(2))
            day = int(iso_match.group(3))
            
            try:
                target_date = datetime(year, month, day)
                return (target_date, target_date)
            except ValueError:
                pass
        
        # Try DD/MM/YYYY or MM/DD/YYYY format
        numeric_pattern = r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'
        numeric_match = re.search(numeric_pattern, query_lower)
        if numeric_match:
            # Assume DD/MM/YYYY format (common in India)
            day = int(numeric_match.group(1))
            month = int(numeric_match.group(2))
            year = int(numeric_match.group(3))
            
            try:
                target_date = datetime(year, month, day)
                return (target_date, target_date)
            except ValueError:
                # Try MM/DD/YYYY if DD/MM/YYYY fails
                try:
                    month = int(numeric_match.group(1))
                    day = int(numeric_match.group(2))
                    target_date = datetime(year, month, day)
                    return (target_date, target_date)
                except ValueError:
                    pass
        
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
        
        # Filter by date range
        df_filtered = df[
            (df['date'] >= start_date) &
            (df['date'] <= end_date)
        ]

        # Filter by market and commodity
        df_filtered = df_filtered[
            df_filtered['amc_name'].str.contains(market_clean, na=False) &
            df_filtered['commodity_name'].str.contains(commodity_clean, na=False)
        ]
        
        # Filter by variant if specified
        if variant:
            variant_clean = variant.lower().strip()
            df_filtered = df_filtered[
                df_filtered['commodity_name'].str.contains(variant_clean, na=False)
            ]
            
            # 🔧 FIX: Validate that variant exists
            if df_filtered.empty:
                return {
                    "success": False,
                    "error": f"Variant '{variant}' not found for {commodity} in {market} during the specified period. Please check the variant name or ask for available variants."
                }
        else:
            # Check if there are multiple variants for this commodity
            # Get unique commodity names (variants)
            unique_variants = df_filtered['commodity_name'].unique().tolist()
            
            # If multiple variants exist, ask user to select one
            if len(unique_variants) > 1:
                return {
                    "success": True,
                    "has_varieties": True,
                    "data": {
                        "commodity": commodity,
                        "market": market,
                        "variants": sorted(unique_variants),
                        "start_date": start_date.strftime('%Y-%m-%d'),
                        "end_date": end_date.strftime('%Y-%m-%d')
                    }
                }
        

        
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
        Queries the live database (lots_new table) for actual arrival metrics.
        
        Args:
            commodity: Commodity name (e.g., "cotton", "chilli")
            market: Market/AMC name (e.g., "warangal", "khammam") or None for all markets
            start_date: Start date for query
            end_date: End date for query
            metric: Metric to retrieve (bags, lots, arrivals, etc.)
            variant: Optional specific variant
        
        Returns:
            Dict with success status and historical arrival data
        """
        try:
            conn = get_db_connection()
            
            # Normalize inputs
            commodity_clean = commodity.lower().strip() if commodity else None
            market_clean = market.lower().strip() if market else None
            
            # Build WHERE conditions
            where_conditions = []
            params = []
            
            # Date range filter
            where_conditions.append("DATE(created_at) >= %s AND DATE(created_at) <= %s")
            params.extend([start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
            
            # Market filter (only if market is specified)
            if market_clean:
                where_conditions.append("(LOWER(amc_name) LIKE %s OR LOWER(district) LIKE %s)")
                params.extend([f"%{market_clean}%", f"%{market_clean}%"])
            
            # Commodity filter (only if specified)
            if commodity_clean:
                where_conditions.append("LOWER(commodity_name) LIKE %s")
                params.append(f"%{commodity_clean}%")
                
                # Variant filter (only if commodity is specified)
                if variant:
                    variant_clean = variant.lower().strip()
                    where_conditions.append("LOWER(commodity_name) LIKE %s")
                    params.append(f"%{variant_clean}%")
            
            where_clause = " AND ".join(where_conditions)
            
            # Determine what to aggregate based on metric
            if metric == "bags":
                aggregate_column = "SUM(no_of_bags)"
                metric_label = "bags"
            elif metric == "lots":
                aggregate_column = "COUNT(*)"
                metric_label = "lots"
            elif metric == "farmers":
                aggregate_column = "COUNT(DISTINCT farmer_name)"
                metric_label = "farmers"
            elif metric == "weight":
                aggregate_column = "SUM(aprox_quantity)"
                metric_label = "weight"
            else:
                # Default to count of records (arrivals)
                aggregate_column = "COUNT(*)"
                metric_label = "arrivals"
            
            # Check for multiple variants if commodity specified but no variant
            if commodity_clean and not variant:
                variant_check_query = f"""
                    SELECT DISTINCT commodity_name
                    FROM lots_new
                    WHERE {where_clause}
                    LIMIT 10
                """
                
                with conn.cursor() as cursor:
                    cursor.execute(variant_check_query, params)
                    variants = [row['commodity_name'] for row in cursor.fetchall()]
                
                # If multiple variants exist, ask user to select one
                if len(variants) > 1:
                    conn.close()
                    return {
                        "success": True,
                        "has_varieties": True,
                        "data": {
                            "commodity": commodity,
                            "market": market if market else "All Markets",
                            "variants": sorted(variants),
                            "metric": metric,
                            "start_date": start_date.strftime('%Y-%m-%d'),
                            "end_date": end_date.strftime('%Y-%m-%d')
                        }
                    }
            
            # Query for aggregated data by date
            query = f"""
                SELECT 
                    DATE(created_at) as date,
                    {aggregate_column} as value
                FROM lots_new
                WHERE {where_clause}
                GROUP BY DATE(created_at)
                ORDER BY date
            """
            
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
            
            conn.close()
            
            if not results:
                market_display = market if market else "all markets"
                commodity_display = commodity if commodity else "all commodities"
                return {
                    "success": False,
                    "error": f"No historical data found for {commodity_display} in {market_display} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
                }
            
            # Format response
            historical_data = []
            for row in results:
                historical_data.append({
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "actual_arrivals": float(row['value']) if row['value'] else 0,
                    "commodity_name": commodity if commodity else "All Commodities"
                })
            
            market_display = market if market else "All Markets"
            commodity_display = commodity if commodity else "All Commodities"
            
            return {
                "success": True,
                "data": {
                    "commodity": commodity_display,
                    "market": market_display,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "metric": metric,
                    "historical_arrivals": historical_data,
                    "count": len(historical_data)
                }
            }
            
        except Exception as e:
            print(f"❌ Database error in get_historical_arrivals: {e}")
            return {
                "success": False,
                "error": f"Database error: {str(e)}"
            }
    
    def query_historical_data(self, params: Dict) -> Dict:
        """
        Main entry point for historical data queries.
        
        Args:
            params: Dictionary with query parameters including:
                - query: Original user query
                - commodity: Commodity name
                - market/district/amc_name: Market location (None for all markets)
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
        
        # If commodity not specified for PRICE queries, return list of available commodities
        # For ARRIVAL queries, we'll aggregate across all commodities
        if not commodity and query_type == "price":
            # For price queries without commodity, we need a market to show available commodities
            if not market:
                return {
                    "success": False,
                    "error": "Please specify a market for historical price query (or specify a commodity)"
                }
            
            # Get unique commodities for this market
            df = self.data.copy()
            df['amc_name'] = df['amc_name'].astype(str).str.lower()
            market_clean = market.lower().strip()
            
            df_filtered = df[df['amc_name'].str.contains(market_clean, na=False)]
            
            if df_filtered.empty:
                return {
                    "success": False,
                    "error": f"No data found for market {market}"
                }
            
            commodities = sorted(df_filtered['commodity_name'].unique().tolist())
            
            return {
                "success": True,
                "needs_commodity_selection": True,
                "data": {
                    "market": market,
                    "commodities": commodities,
                    "query_type": query_type,
                    "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                }
            }
        
        # Route to appropriate handler
        # Note: market can be None for aggregate queries
        if query_type == "arrival":
            metric = params.get("metric", "arrivals")
            return self.get_historical_arrivals(
                commodity, market, start_date, end_date, metric, variant
            )
        else:  # Default to price
            # Price queries require a market (can't aggregate prices across markets meaningfully)
            if not market:
                return {
                    "success": False,
                    "error": "Please specify a market for historical price query"
                }
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
