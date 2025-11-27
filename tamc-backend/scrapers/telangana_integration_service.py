from __future__ import annotations

"""
Telangana Agricultural Data Scraper Integration Service
Handles automatic daily scraping and provides data access functions
"""

import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import os

# Import your existing scraper
from scrapers.telangana_scrape import (
    TelanganaAgriScraper, 
    ArrivalAnalyzer,
    get_arrivals_for_prediction
)

logging.basicConfig(level=logging.INFO)

DEFAULT_HISTORY_DAYS = int(os.getenv("TELANGANA_HISTORY_DAYS", "60"))
BOOTSTRAP_DAYS = int(os.getenv("TELANGANA_BOOTSTRAP_DAYS", "14"))
SCRAPE_DELAY = float(os.getenv("TELANGANA_SCRAPE_DELAY", "0.2"))  # Faster: 0.2s
SCRAPE_WORKERS = int(os.getenv("TELANGANA_SCRAPE_WORKERS", "3"))  # Parallel: 3 workers

class TelanganaDataService:
    """
    Service for managing Telangana commodity data with auto-updates
    """
    
    def __init__(self, data_file='telangana_commodity_data.csv'):
        self.data_file = data_file
        self.scraper = TelanganaAgriScraper(default_delay=SCRAPE_DELAY)
        self.analyzer = None
        self.last_update = None
        self.is_scraping = False
        self.scraping_lock = threading.Lock()
        self.history_days = DEFAULT_HISTORY_DAYS
        
        # Check if data exists and load it
        if Path(self.data_file).exists():
            self.analyzer = ArrivalAnalyzer(self.data_file)
            self.last_update = datetime.fromtimestamp(
                Path(self.data_file).stat().st_mtime
            )
            logging.info(f"✅ Loaded existing data from {self.data_file}")
            logging.info(f"📅 Last update: {self.last_update}")
        else:
            logging.info(f"⚠️ No existing data found. Initial scraping required.")
    
    def _load_existing_dataframe(self):
        if self.analyzer and self.analyzer.df is not None:
            return self.analyzer.df
        if Path(self.data_file).exists():
            df = pd.read_csv(self.data_file)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            return df
        return None

    def scrape_and_update(self, days=None, force_full=False):
        """
        Scrape new data and update the CSV file
        Thread-safe with lock to prevent concurrent scraping
        """
        days = days or self.history_days
        with self.scraping_lock:
            if self.is_scraping:
                logging.warning("⏳ Scraping already in progress, skipping...")
                return False
            
            self.is_scraping = True
        
        try:
            logging.info(f"🔄 Starting data scraping (force_full={force_full})...")
            start_time = datetime.now()
            
            existing_df = None if force_full else self._load_existing_dataframe()
            end_date = datetime.now()

            if existing_df is None or existing_df.empty:
                logging.info("📂 No cached data found. Performing bootstrap scrape.")
                start_date = end_date - timedelta(days=max(days, BOOTSTRAP_DAYS))
                append = False
            else:
                last_date = existing_df['Date'].max().date()
                start_date = datetime.combine(last_date + timedelta(days=1), datetime.min.time())
                append = True
                if start_date.date() > end_date.date():
                    logging.info("✅ Dataset already up to date. No scraping needed.")
                    self.last_update = datetime.now()
                    return True
                logging.info(f"📈 Incremental scrape from {start_date.date()} to {end_date.date()}")

            data = self.scraper.scrape_date_range(
                start_date,
                end_date,
                delay=SCRAPE_DELAY,
                max_workers=SCRAPE_WORKERS
            )
            
            if not data:
                logging.error("❌ No data scraped")
                return False
            
            # Save to CSV
            df = self.scraper.save_to_csv(data, self.data_file, append=append)
            
            # Reload analyzer with new data
            self.analyzer = ArrivalAnalyzer(self.data_file)
            self.last_update = datetime.now()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logging.info(f"✅ Scraping completed in {elapsed:.1f} seconds")
            logging.info(f"📊 Total records: {len(df)}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Scraping error: {e}")
            return False
        finally:
            with self.scraping_lock:
                self.is_scraping = False
    
    def get_commodity_arrivals(self, commodity_name, market_name=None, days=60):
        """
        Get arrival data for prediction models
        
        Returns:
            dict: {'dates': list, 'arrivals': list} or None
        """
        if self.analyzer is None:
            logging.warning("⚠️ No data available. Running initial scrape...")
            self.scrape_and_update(force_full=True)
            
            if self.analyzer is None:
                logging.error("❌ Failed to initialize data")
                return None
        
        # Check if data is stale (older than 24 hours)
        if self.last_update:
            age_hours = (datetime.now() - self.last_update).total_seconds() / 3600
            if age_hours > 24:
                logging.info(f"⏰ Data is {age_hours:.1f} hours old. Triggering update...")
                # Trigger update in background (non-blocking)
                threading.Thread(
                    target=self.scrape_and_update,
                    kwargs={"force_full": False},
                    daemon=True
                ).start()
        
        # Get arrival data
        return self.analyzer.get_historical_arrivals_array(
            commodity_name, 
            market_name, 
            days=days
        )
    
    def get_arrival_statistics(self, commodity_name, market_name=None):
        """Get statistical summary of arrivals"""
        if self.analyzer is None:
            return None
        return self.analyzer.get_arrival_statistics(commodity_name, market_name)
    
    def get_recent_trend(self, commodity_name, market_name=None, days=7):
        """Get recent trend information"""
        if self.analyzer is None:
            return None
        return self.analyzer.get_recent_trend(commodity_name, market_name, days)
    
    def get_available_commodities(self):
        """Get list of available commodities"""
        if self.analyzer is None:
            return []
        return self.analyzer.get_commodity_list()
    
    def get_available_markets(self, commodity_name=None):
        """Get list of available markets"""
        if self.analyzer is None:
            return []
        return self.analyzer.get_market_list(commodity_name)
    
    def get_data_info(self):
        """Get information about the current dataset"""
        if self.analyzer is None or self.analyzer.df is None:
            return {
                'status': 'no_data',
                'message': 'No data available'
            }
        
        df = self.analyzer.df
        return {
            'status': 'available',
            'total_records': len(df),
            'date_range': {
                'start': df['Date'].min().strftime('%Y-%m-%d'),
                'end': df['Date'].max().strftime('%Y-%m-%d')
            },
            'unique_commodities': df['Commodity Name'].nunique(),
            'unique_markets': df['Market Name'].nunique(),
            'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else None,
            'age_hours': (datetime.now() - self.last_update).total_seconds() / 3600 if self.last_update else None
        }
    
    def schedule_daily_updates(self, update_time="02:00"):
        """
        Schedule daily automatic updates
        
        Args:
            update_time: Time to run updates (24-hour format, e.g., "02:00")
        """
        def job():
            logging.info("⏰ Scheduled update triggered")
            self.scrape_and_update()
        
        # Schedule daily update
        schedule.every().day.at(update_time).do(job)
        
        # Run scheduler in background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logging.info(f"📅 Scheduled daily updates at {update_time}")
    
    def force_update(self):
        """Force an immediate update"""
        logging.info("🔄 Forcing immediate data update...")
        return self.scrape_and_update(force_full=True)


# Global service instance
_telangana_service = None

def get_telangana_service():
    """Get or create the global Telangana data service instance"""
    global _telangana_service
    if _telangana_service is None:
        _telangana_service = TelanganaDataService()
        # Schedule daily updates at 2 AM
        _telangana_service.schedule_daily_updates("02:00")
    return _telangana_service


# Convenience functions for easy integration
def get_telangana_arrivals(commodity_name, market_name=None, days=60):
    """
    Get Telangana commodity arrivals for prediction
    
    Args:
        commodity_name: Name of commodity (e.g., 'Tomato', 'Onions')
        market_name: Optional market name for filtering
        days: Number of historical days
    
    Returns:
        dict: {'dates': list, 'arrivals': list} or None
    
    Example:
        >>> data = get_telangana_arrivals('Tomato', days=60)
        >>> if data:
        >>>     print(f"Got {len(data['dates'])} days of data")
    """
    service = get_telangana_service()
    return service.get_commodity_arrivals(commodity_name, market_name, days)


def get_telangana_stats(commodity_name, market_name=None):
    """
    Get statistical summary for a commodity
    
    Returns:
        dict: Statistics (mean, median, std, min, max, etc.) or None
    """
    service = get_telangana_service()
    return service.get_arrival_statistics(commodity_name, market_name)


def get_telangana_trend(commodity_name, market_name=None, days=7):
    """
    Get recent trend information
    
    Returns:
        dict: Trend info (trend, change_percent, etc.) or None
    """
    service = get_telangana_service()
    return service.get_recent_trend(commodity_name, market_name, days)


def get_telangana_data_info():
    """
    Get information about the Telangana dataset
    
    Returns:
        dict: Dataset information
    """
    service = get_telangana_service()
    return service.get_data_info()


def force_telangana_update():
    """Force an immediate update of Telangana data"""
    service = get_telangana_service()
    return service.force_update()


# Example usage for testing
if __name__ == "__main__":
    print("="*70)
    print("Telangana Data Service - Integration Test")
    print("="*70)
    
    # Get service instance
    service = get_telangana_service()
    
    # Get dataset info
    info = service.get_data_info()
    print("\n📊 Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example: Get Tomato arrivals
    print("\n🍅 Testing Tomato arrivals...")
    tomato_data = get_telangana_arrivals('Tomato', days=30)
    
    if tomato_data:
        print(f"✅ Retrieved {len(tomato_data['dates'])} days of Tomato data")
        print(f"   Date range: {tomato_data['dates'][0]} to {tomato_data['dates'][-1]}")
        print(f"   Average arrivals: {sum(tomato_data['arrivals'])/len(tomato_data['arrivals']):.2f} Qtls")
        
        # Get statistics
        stats = get_telangana_stats('Tomato')
        if stats:
            print(f"\n📈 Tomato Statistics:")
            print(f"   Mean: {stats['mean']:.2f} Qtls")
            print(f"   Median: {stats['median']:.2f} Qtls")
            print(f"   Range: {stats['min']:.2f} - {stats['max']:.2f} Qtls")
        
        # Get trend
        trend = get_telangana_trend('Tomato', days=7)
        if trend:
            print(f"\n📊 Recent Trend (7 days):")
            print(f"   Trend: {trend['trend']}")
            print(f"   Change: {trend['change_percent']:.2f}%")
    else:
        print("❌ No data available for Tomato")
    
    # List available commodities
    commodities = service.get_available_commodities()
    print(f"\n📋 Available Commodities ({len(commodities)}):")
    print(f"   {', '.join(commodities[:10])}...")
    
    print("\n" + "="*70)
    print("✅ Integration test completed")
    print("💡 The service will auto-update daily at 2:00 AM")
    print("="*70)
