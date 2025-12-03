from __future__ import annotations

import os
import pickle
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

NAPANTA_DEFAULT_DAYS = int(os.getenv("NAPANTA_DEFAULT_DAYS", "30"))
NAPANTA_MAX_DAYS = int(os.getenv("NAPANTA_MAX_DAYS", "90"))
NAPANTA_DELAY_SECONDS = float(os.getenv("NAPANTA_DELAY_SECONDS", "0.2"))  # Faster: 0.2s
NAPANTA_WAIT_TIMEOUT = int(os.getenv("NAPANTA_WAIT_TIMEOUT", "10000"))  # Faster: 10s
NAPANTA_PAGE_TIMEOUT = int(os.getenv("NAPANTA_PAGE_TIMEOUT", "20000"))  # Faster: 20s
NAPANTA_HEADLESS = os.getenv("NAPANTA_HEADLESS", "true").lower() != "false"
NAPANTA_OUTPUT_DIR = os.getenv("NAPANTA_OUTPUT_DIR", ".")
NAPANTA_CACHE_FILE = os.getenv("NAPANTA_CACHE_FILE", "napanta_cache.pkl")
NAPANTA_CACHE_DURATION = timedelta(hours=24)
NAPANTA_MAX_CONSECUTIVE_FAILURES = int(os.getenv("NAPANTA_MAX_CONSECUTIVE_FAILURES", "3"))  # Exit faster


class NapantaPriceScraper:
    def __init__(self, *, headless: bool = NAPANTA_HEADLESS):
        self.base_url = "https://www.napanta.com/market-price"
        self.headless = headless
        self.request_delay = max(0.0, NAPANTA_DELAY_SECONDS)
        self.wait_timeout = NAPANTA_WAIT_TIMEOUT
        self.page_timeout = NAPANTA_PAGE_TIMEOUT

    @contextmanager
    def _browser_page(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self.headless)
            page = browser.new_page()
            try:
                yield page
            finally:
                browser.close()

    @staticmethod
    def _normalize_for_url(text: str | None) -> str:
        if not text:
            return ""
        return text.lower().strip().replace(" ", "-").replace("_", "-")

    def _build_url(self, state: str, district: str, market: str, date_str: str) -> str:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%d-%b-%Y").lower()
        state_url = self._normalize_for_url(state)
        district_url = self._normalize_for_url(district)
        market_url = self._normalize_for_url(market) or district_url
        return f"{self.base_url}/{state_url}/{district_url}/{market_url}/{formatted_date}"

    def _load_existing_output(self, path: Path) -> Tuple[Optional[pd.DataFrame], set[str]]:
        if not path.exists():
            return None, set()
        df = pd.read_csv(path)
        if "Fetch_Date" in df.columns:
            df["Fetch_Date"] = pd.to_datetime(df["Fetch_Date"], errors="coerce")
            df = df.dropna(subset=["Fetch_Date"])
            existing_dates = set(df["Fetch_Date"].dt.strftime("%Y-%m-%d"))
        else:
            existing_dates = set()
        return df, existing_dates

    def _scrape_single_date(self, page, state: str, district: str, market: str, date_str: str, retry_count: int = 2) -> Optional[pd.DataFrame]:
        url = self._build_url(state, district, market, date_str)

        for attempt in range(retry_count + 1):
            try:
                page.goto(url, timeout=self.page_timeout, wait_until="domcontentloaded")
                try:
                    page.wait_for_selector("table.table, table", timeout=self.wait_timeout)
                except PlaywrightTimeoutError:
                    page.wait_for_timeout(1000)

                table_payload = page.evaluate(
                    """
                    () => {
                        const table = document.querySelector('table.table') || document.querySelector('table');
                        if (!table) return null;

                        const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());
                        const rows = Array.from(table.querySelectorAll('tbody tr'))
                            .map(row => Array.from(row.querySelectorAll('td')).map(td => td.textContent.trim()))
                            .filter(row => row.length >= 4 && row[0] && row[0] !== 'Commodity');

                        if (rows.length === 0) {
                            const fallbackRows = Array.from(table.querySelectorAll('tr'))
                                .map(row => Array.from(row.querySelectorAll('td')).map(td => td.textContent.trim()))
                                .filter(row => row.length >= 4 && row[0] && row[0] !== 'Commodity');
                            rows.push(...fallbackRows);
                        }

                        return { headers, rows };
                    }
                    """
                )

                if not table_payload or not table_payload["rows"]:
                    if attempt < retry_count:
                        continue  # Retry
                    return None

                rows = table_payload["rows"]
                headers = table_payload.get("headers") or []
                columns = ['Commodity', 'Market', 'Variety', 'Max_Price', 'Avg_Price', 'Min_Price', 'Last_Updated', 'Trend', 'Extra']
                frame = pd.DataFrame(rows, columns=columns[: len(rows[0])])
                frame['Fetch_Date'] = date_str

                for col in ['Max_Price', 'Avg_Price', 'Min_Price']:
                    if col in frame.columns:
                        frame[col] = (
                            frame[col]
                            .str.replace('₹', '', regex=False)
                            .str.replace(',', '', regex=False)
                            .str.strip()
                        )
                        frame[col] = pd.to_numeric(frame[col], errors='coerce')

                return frame

            except PlaywrightTimeoutError:
                if attempt < retry_count:
                    print(f"WARNING: Timeout on attempt {attempt + 1}/{retry_count + 1}, retrying...")
                    page.wait_for_timeout(1000)
                    continue
                return None
            except Exception as exc:
                if attempt < retry_count:
                    print(f"WARNING: Error on attempt {attempt + 1}: {exc}, retrying...")
                    page.wait_for_timeout(1000)
                    continue
                print(f"WARNING: Error fetching {date_str} after {retry_count + 1} attempts: {exc}")
                return None

        return None

    def get_available_filters(self):
        print("🔍 Fetching available filter options...")
        with self._browser_page() as page:
            page.goto(self.base_url, timeout=self.page_timeout, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            try:
                filters = page.evaluate(
                    """
                    () => {
                        const getOptions = (selector) => {
                            const select = document.querySelector(selector);
                            if (!select) return [];
                            return Array.from(select.options)
                                .filter(opt => opt.value && opt.value !== '0')
                                .map(opt => ({ value: opt.value, text: opt.textContent.trim() }));
                        };
                        return {
                            states: getOptions('#ddlState'),
                            districts: getOptions('#ddlDistrict'),
                            markets: getOptions('#ddlMarket')
                        };
                    }
                    """
                )
                return filters
            except Exception as exc:
                print(f"ERROR: Error fetching filters: {exc}")
                return None

    def fetch_historical_prices(
        self,
        state: str,
        district: str,
        market: Optional[str] = None,
        commodity: Optional[str] = None,
        days_back: int = NAPANTA_DEFAULT_DAYS,
    ) -> Optional[pd.DataFrame]:
        if not state or not district:
            raise ValueError("State and district are required for Naapanta scraping.")

        days_back = max(1, min(days_back, NAPANTA_MAX_DAYS))
        market = market or district

        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days_back)]

        output_name = f"napanta_historical_{state.replace(' ', '_')}_{district.replace(' ', '_')}"
        if market and market != district:
            output_name += f"_{market.replace(' ', '_')}"
        if commodity:
            output_name += f"_{commodity.replace(' ', '_')}"
        output_name += f"_{days_back}days.csv"
        output_dir = Path(NAPANTA_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name

        existing_df, existing_dates = self._load_existing_output(output_path)
        new_frames: List[pd.DataFrame] = []
        consecutive_failures = 0

        print(f"\n Naapanta scraper | {state} > {district} > {market} | days: {days_back}")
        if existing_dates:
            print(f" Skipping {len(existing_dates)} date(s) already in {output_name}")

        with self._browser_page() as page:
            for idx, date_str in enumerate(dates, 1):
                if date_str in existing_dates:
                    continue

                print(f" Fetching {date_str} ({idx}/{days_back})...")
                df = self._scrape_single_date(page, state, district, market, date_str)
                if df is None or df.empty:
                    consecutive_failures += 1
                    print(f"   WARNING: No data for {date_str}")
                    if consecutive_failures >= NAPANTA_MAX_CONSECUTIVE_FAILURES:
                        print(f"   ERROR: {NAPANTA_MAX_CONSECUTIVE_FAILURES} consecutive failures, stopping scrape.")
                        break
                else:
                    consecutive_failures = 0
                    if commodity and 'Commodity' in df.columns:
                        df = df[df['Commodity'].str.lower().str.contains(commodity.lower(), na=False)]
                    if not df.empty:
                        new_frames.append(df)
                if self.request_delay:
                    page.wait_for_timeout(int(self.request_delay * 1000))

        if not new_frames and existing_df is None:
            print("ERROR: No historical data fetched")
            return None

        combined_df = pd.concat([df for df in [existing_df, *new_frames] if df is not None], ignore_index=True)
        combined_df['Fetch_Date'] = pd.to_datetime(combined_df['Fetch_Date'])
        combined_df = (
            combined_df
            .drop_duplicates(subset=['Fetch_Date', 'Commodity', 'Market', 'Variety'], keep='last')
            .sort_values('Fetch_Date', ascending=False)
            .reset_index(drop=True)
        )

        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n Saved {len(combined_df)} rows to {output_path}")

        if commodity and 'Commodity' in combined_df.columns:
            sample = combined_df[combined_df['Commodity'].str.lower().str.contains(commodity.lower(), na=False)]
        else:
            sample = combined_df

        if not sample.empty:
            first = sample.iloc[0]['Commodity'] if 'Commodity' in sample.columns else commodity or 'Commodity'
            trend = sample[['Fetch_Date', 'Avg_Price', 'Min_Price', 'Max_Price']].head(5)
            trend['Fetch_Date'] = trend['Fetch_Date'].dt.strftime('%Y-%m-%d')
            print(f" Recent price trend for {first}:")
            print(trend.to_string(index=False))

        return combined_df


def get_historical_prices_for_prediction(state, district, market=None, commodity=None, days_back=NAPANTA_DEFAULT_DAYS):
    scraper = NapantaPriceScraper()
    return scraper.fetch_historical_prices(state, district, market=market, commodity=commodity, days_back=days_back)


def interactive_historical_scraper():
    scraper = NapantaPriceScraper()
    print("=" * 70)
    print("⚡ NAPANTA FAST HISTORICAL PRICE SCRAPER")
    print("=" * 70)

    filters = scraper.get_available_filters()
    if filters:
        print("\n📍 Sample Available States:")
        for opt in filters['states'][:10]:
            print(f"   - {opt['text']}")
        print("\n📍 Sample Available Districts:")
        for opt in filters['districts'][:10]:
            print(f"   - {opt['text']}")

    state = input("State (e.g., Telangana): ").strip()
    district = input("District (e.g., Warangal): ").strip()
    market = input("Market (optional, press Enter to use district name): ").strip()
    commodity = input("Commodity (e.g., Potato, or press Enter for all): ").strip()

    try:
        days_back = int(input("Days of history to fetch (default 30, max 90): ").strip() or NAPANTA_DEFAULT_DAYS)
    except ValueError:
        days_back = NAPANTA_DEFAULT_DAYS

    days_back = max(1, min(days_back, NAPANTA_MAX_DAYS))

    df = scraper.fetch_historical_prices(
        state=state or 'Telangana',
        district=district or 'Warangal',
        market=market or None,
        commodity=commodity or None,
        days_back=days_back,
    )

    if df is not None:
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Historical price data ready for prediction model")
        print("=" * 70)
    else:
        print("\nERROR: Failed to extract historical data")
    return df


if __name__ == "__main__":
    interactive_historical_scraper()


def get_or_update_naapanta_data(state, district, market=None, commodity=None, days_back=NAPANTA_DEFAULT_DAYS):
    try:
        cache_path = Path(NAPANTA_CACHE_FILE)
        if cache_path.exists():
            with cache_path.open("rb") as fh:
                cache = pickle.load(fh)
            last_update = cache.get("last_update")
            cached_df = cache.get("data")
            if cached_df is not None and last_update and datetime.now() - last_update < NAPANTA_CACHE_DURATION:
                print("✅ Using cached Naapanta data (updated within 24 hours).")
                return cached_df

        print("🌐 Fetching fresh Naapanta market data (older than 24 hrs)...")
        df = get_historical_prices_for_prediction(
            state=state,
            district=district,
            market=market,
            commodity=commodity,
            days_back=days_back,
        )
        if df is not None and not df.empty:
            with cache_path.open("wb") as fh:
                pickle.dump({"data": df, "last_update": datetime.now()}, fh)
            print(f"✅ Cached {len(df)} Naapanta records for future use.")
        return df
    except Exception as exc:
        print(f"WARNING: Error fetching Naapanta data: {exc}")
        return None

