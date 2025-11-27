"""
Telangana Agricultural Marketing Data Scraper & Analyzer
Single file solution for scraping and analyzing commodity arrival data
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

class TelanganaAgriScraper:
    def __init__(self, default_delay: float = 0.3):
        self.base_url = "http://183.82.5.184/tgmarketing/DailyArrivalsnPrices.aspx"
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.data_file = 'telangana_commodity_data.csv'
        self.default_delay = default_delay

    def _create_session(self):
        session = requests.Session()
        session.headers.update(self.headers)
        return session
    
    def get_viewstate(self, soup):
        """Extract ASP.NET ViewState and EventValidation from the page"""
        viewstate = soup.find('input', {'name': '__VIEWSTATE'})
        eventvalidation = soup.find('input', {'name': '__EVENTVALIDATION'})
        viewstategenerator = soup.find('input', {'name': '__VIEWSTATEGENERATOR'})
        
        return {
            '__VIEWSTATE': viewstate['value'] if viewstate else '',
            '__EVENTVALIDATION': eventvalidation['value'] if eventvalidation else '',
            '__VIEWSTATEGENERATOR': viewstategenerator['value'] if viewstategenerator else ''
        }
    
    def parse_data(self, html_content):
        """Parse the commodity data from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        data = []
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) >= 7:
                    cols_text = [col.get_text(strip=True) for col in cols]
                    if cols_text[0] not in ['Market Name', '']:
                        data.append(cols_text)
        
        return data
    
    def scrape_date(self, target_date, session: Optional[requests.Session] = None):
        """Scrape data for a specific date"""
        sess = session or self.session
        try:
            response = sess.get(self.base_url, headers=self.headers, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            form_data = self.get_viewstate(soup)
            date_str = target_date.strftime('%d/%m/%Y')
            
            form_data.update({
                'ctl00$ContentPlaceHolder1$txtDate': date_str,
                'ctl00$ContentPlaceHolder1$btnSubmit': 'Submit',
                '__EVENTTARGET': '',
                '__EVENTARGUMENT': ''
            })
            
            response = sess.post(
                self.base_url, 
                data=form_data, 
                headers=self.headers,
                timeout=30
            )
            
            data = self.parse_data(response.text)
            
            for row in data:
                row.insert(0, date_str)
            
            return data
            
        except Exception as e:
            print(f"Error scraping date {target_date}: {str(e)}")
            return []
    
    def scrape_date_range(self, start_date, end_date, delay=None, max_workers: int = 1):
        """Scrape data for a date range"""
        all_data = []
        
        print(f"Scraping data from {start_date.date()} to {end_date.date()}")
        
        current_date = start_date
        dates: List[datetime] = []
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

        delay = self.default_delay if delay is None else delay
        max_workers = max(1, max_workers)

        if max_workers == 1:
            for day in dates:
                print(f"Scraping {day.date()}...")
                data = self.scrape_date(day)
                all_data.extend(data)
                if delay:
                    time.sleep(delay)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            print(f"Using parallel scraping with {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.scrape_date, day, self._create_session()): day for day in dates}
                for future in as_completed(futures):
                    day = futures[future]
                    try:
                        data = future.result()
                        all_data.extend(data)
                        print(f"Scraped {day.date()} ✔")
                    except Exception as exc:
                        print(f"Error scraping {day.date()}: {exc}")
                    finally:
                        if delay:
                            time.sleep(delay)
        
        return all_data
    
    def scrape_60_days(self, delay=2):
        """Scrape data for the last 60 days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        return self.scrape_date_range(start_date, end_date, delay)
    
    def save_to_csv(self, data, filename=None, append=False):
        """Save scraped data to CSV"""
        if filename is None:
            filename = self.data_file
            
        columns = [
            'Date', 'Market Name', 'Commodity Name', 'Variety Name', 
            'Arrivals(Qtls)', 'Maximum', 'Minimum', 'Modal', 'Purchase By'
        ]
        
        if not data:
            print("No data to save.")
            return None

        df = pd.DataFrame(data, columns=columns)
        
        # Convert date to proper datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        
        # Clean and convert numeric columns
        numeric_cols = ['Arrivals(Qtls)', 'Maximum', 'Minimum', 'Modal']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].str.replace(',', '').astype(float, errors='ignore')
        
        # Sort by date and remove invalid dates
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.dropna(subset=['Date'])
        
        if append and os.path.exists(filename):
            existing = pd.read_csv(filename)
            existing['Date'] = pd.to_datetime(existing['Date'], errors='coerce')
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(
                subset=['Date', 'Market Name', 'Commodity Name', 'Variety Name'],
                keep='last'
            )
            df = df.sort_values('Date').reset_index(drop=True)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\nData saved to {filename}")
        print(f"Total records: {len(df)}")
        return df
    
    def load_data(self, filename=None):
        """Load data from CSV file"""
        if filename is None:
            filename = self.data_file
            
        if not os.path.exists(filename):
            print(f"File {filename} not found. Please scrape data first.")
            return None
        
        df = pd.read_csv(filename)
        df['Date'] = pd.to_datetime(df['Date'])
        return df


class ArrivalAnalyzer:
    """Analyzer for commodity arrival predictions"""
    
    def __init__(self, data_file='telangana_commodity_data.csv'):
        """Initialize with data file"""
        self.data_file = data_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the scraped data"""
        if os.path.exists(self.data_file):
            self.df = pd.read_csv(self.data_file)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            print(f"Loaded {len(self.df)} records from {self.data_file}")
        else:
            print(f"Data file {self.data_file} not found. Please scrape data first.")
    
    def get_commodity_arrivals(self, commodity_name, market_name=None, 
                              start_date=None, end_date=None):
        """
        Get arrival data for a specific commodity
        
        Parameters:
        -----------
        commodity_name : str
            Name of the commodity (e.g., 'Tomato', 'Onions')
        market_name : str, optional
            Specific market name. If None, aggregates all markets
        start_date : str or datetime, optional
            Start date for filtering (format: 'YYYY-MM-DD')
        end_date : str or datetime, optional
            End date for filtering (format: 'YYYY-MM-DD')
        
        Returns:
        --------
        pd.DataFrame : Arrival data with columns [Date, Arrivals, Market Name (optional)]
        """
        if self.df is None:
            print("No data loaded. Please load data first.")
            return None
        
        # Filter by commodity
        data = self.df[self.df['Commodity Name'] == commodity_name].copy()
        
        if len(data) == 0:
            print(f"No data found for commodity: {commodity_name}")
            return None
        
        # Filter by market if specified
        if market_name:
            data = data[data['Market Name'] == market_name]
            if len(data) == 0:
                print(f"No data found for market: {market_name}")
                return None
        
        # Filter by date range
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data['Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data['Date'] <= end_date]
        
        # Group by date and sum arrivals
        if market_name:
            result = data[['Date', 'Arrivals(Qtls)', 'Market Name']].copy()
        else:
            result = data.groupby('Date')['Arrivals(Qtls)'].sum().reset_index()
        
        result = result.sort_values('Date').reset_index(drop=True)
        return result
    
    def get_historical_arrivals_array(self, commodity_name, market_name=None, 
                                     days=60):
        """
        Get historical arrivals as a simple array for ML models
        
        Parameters:
        -----------
        commodity_name : str
            Name of the commodity
        market_name : str, optional
            Specific market name
        days : int
            Number of days of historical data (default: 60)
        
        Returns:
        --------
        dict : {'dates': list, 'arrivals': list}
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = self.get_commodity_arrivals(
            commodity_name, 
            market_name, 
            start_date, 
            end_date
        )
        
        if data is None or len(data) == 0:
            return None
        
        return {
            'dates': data['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'arrivals': data['Arrivals(Qtls)'].tolist()
        }
    
    def get_market_list(self, commodity_name=None):
        """Get list of markets, optionally filtered by commodity"""
        if self.df is None:
            return []
        
        if commodity_name:
            markets = self.df[self.df['Commodity Name'] == commodity_name]['Market Name'].unique()
        else:
            markets = self.df['Market Name'].unique()
        
        return sorted(markets.tolist())
    
    def get_commodity_list(self):
        """Get list of all commodities"""
        if self.df is None:
            return []
        return sorted(self.df['Commodity Name'].unique().tolist())
    
    def get_arrival_statistics(self, commodity_name, market_name=None):
        """
        Get statistical summary of arrivals
        
        Returns:
        --------
        dict : Statistics including mean, median, std, min, max
        """
        data = self.get_commodity_arrivals(commodity_name, market_name)
        
        if data is None or len(data) == 0:
            return None
        
        arrivals = data['Arrivals(Qtls)']
        
        stats = {
            'mean': arrivals.mean(),
            'median': arrivals.median(),
            'std': arrivals.std(),
            'min': arrivals.min(),
            'max': arrivals.max(),
            'total': arrivals.sum(),
            'count': len(arrivals)
        }
        
        return stats
    
    def get_recent_trend(self, commodity_name, market_name=None, days=7):
        """
        Get recent trend (increasing/decreasing/stable)
        
        Returns:
        --------
        dict : {'trend': str, 'change_percent': float, 'recent_avg': float, 'previous_avg': float}
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days*2)
        
        data = self.get_commodity_arrivals(commodity_name, market_name, start_date, end_date)
        
        if data is None or len(data) < days:
            return None
        
        # Split into recent and previous periods
        mid_point = len(data) - days
        previous_data = data.iloc[:mid_point]['Arrivals(Qtls)']
        recent_data = data.iloc[mid_point:]['Arrivals(Qtls)']
        
        previous_avg = previous_data.mean()
        recent_avg = recent_data.mean()
        
        if previous_avg == 0:
            change_percent = 0
        else:
            change_percent = ((recent_avg - previous_avg) / previous_avg) * 100
        
        if change_percent > 10:
            trend = 'increasing'
        elif change_percent < -10:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_percent': round(change_percent, 2),
            'recent_avg': round(recent_avg, 2),
            'previous_avg': round(previous_avg, 2)
        }


# Convenience functions for easy use
def scrape_data(days=60, filename='telangana_commodity_data.csv'):
    """
    Scrape data for specified number of days
    
    Parameters:
    -----------
    days : int
        Number of days to scrape (default: 60)
    filename : str
        Output CSV filename
    
    Returns:
    --------
    pd.DataFrame : Scraped data
    """
    scraper = TelanganaAgriScraper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = scraper.scrape_date_range(start_date, end_date)
    return scraper.save_to_csv(data, filename)


def get_arrivals_for_prediction(commodity_name, market_name=None, days=60):
    """
    Get arrival data ready for ML prediction models
    
    Parameters:
    -----------
    commodity_name : str
        Commodity name (e.g., 'Tomato')
    market_name : str, optional
        Market name (if None, aggregates all markets)
    days : int
        Number of historical days (default: 60)
    
    Returns:
    --------
    dict : {'dates': list, 'arrivals': list} or None
    
    Example:
    --------
    >>> data = get_arrivals_for_prediction('Tomato', days=60)
    >>> print(data['dates'])  # ['2025-08-30', '2025-08-31', ...]
    >>> print(data['arrivals'])  # [1234.5, 1456.7, ...]
    """
    analyzer = ArrivalAnalyzer()
    return analyzer.get_historical_arrivals_array(commodity_name, market_name, days)


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("Telangana Agricultural Marketing Data Tool")
    print("="*60)
    
    # Example 1: Scrape 60 days of data
    print("\n[1] Scraping 60 days of data...")
    scraper = TelanganaAgriScraper()
    data = scraper.scrape_60_days(delay=2)
    df = scraper.save_to_csv(data)
    
    print("\n" + "="*60)
    print("Data Summary")
    print("="*60)
    print(f"Date range: {df['Date'].min().strftime('%d/%m/%Y')} to {df['Date'].max().strftime('%d/%m/%Y')}")
    print(f"Total records: {len(df)}")
    print(f"Unique markets: {df['Market Name'].nunique()}")
    print(f"Unique commodities: {df['Commodity Name'].nunique()}")
    
    # Example 2: Analyze arrivals
    print("\n[2] Analyzing arrival data...")
    analyzer = ArrivalAnalyzer()
    
    # Get available commodities
    commodities = analyzer.get_commodity_list()
    print(f"\nAvailable commodities: {', '.join(commodities[:10])}...")
    
    # Example 3: Get arrivals for prediction (Tomato)
    print("\n[3] Getting arrival data for prediction model...")
    tomato_data = get_arrivals_for_prediction('Tomato', days=60)
    
    if tomato_data:
        print(f"\nTomato arrival data:")
        print(f"  - Date range: {tomato_data['dates'][0]} to {tomato_data['dates'][-1]}")
        print(f"  - Number of days: {len(tomato_data['dates'])}")
        print(f"  - Average arrivals: {sum(tomato_data['arrivals'])/len(tomato_data['arrivals']):.2f} Qtls")
        print(f"  - Recent 5 days arrivals: {tomato_data['arrivals'][-5:]}")
        
        # Get statistics
        stats = analyzer.get_arrival_statistics('Tomato')
        print(f"\nTomato Statistics:")
        print(f"  - Mean: {stats['mean']:.2f} Qtls")
        print(f"  - Median: {stats['median']:.2f} Qtls")
        print(f"  - Std Dev: {stats['std']:.2f} Qtls")
        print(f"  - Range: {stats['min']:.2f} - {stats['max']:.2f} Qtls")
        
        # Get trend
        trend = analyzer.get_recent_trend('Tomato', days=7)
        print(f"\nRecent Trend (Last 7 days):")
        print(f"  - Trend: {trend['trend']}")
        print(f"  - Change: {trend['change_percent']:.2f}%")
        print(f"  - Recent avg: {trend['recent_avg']:.2f} Qtls")
    
    print("\n" + "="*60)
    print("Usage for Prediction Model:")
    print("="*60)
