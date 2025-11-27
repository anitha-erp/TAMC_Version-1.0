from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

NCDEX_URL = "https://www.ncdex.com/market-watch/live_quotes"
NCDEX_HEADLESS = os.getenv("NCDEX_HEADLESS", "true").lower() != "false"
NCDEX_WAIT_TIMEOUT = int(os.getenv("NCDEX_WAIT_TIMEOUT", "15000"))  # Faster: 15s
NCDEX_PAGE_TIMEOUT = int(os.getenv("NCDEX_PAGE_TIMEOUT", "30000"))  # Faster: 30s
NCDEX_MIN_ROWS = int(os.getenv("NCDEX_MIN_ROWS", "5"))
NCDEX_CACHE_CSV = Path(os.getenv("NCDEX_CACHE_CSV", "ncdex_prices.csv"))
NCDEX_CACHE_TTL = timedelta(hours=int(os.getenv("NCDEX_CACHE_TTL_HOURS", "24")))
NCDEX_RETRY_COUNT = int(os.getenv("NCDEX_RETRY_COUNT", "2"))  # Retry on failure


def _extract_table(page) -> Optional[pd.DataFrame]:
    payload = page.evaluate(
        """
        () => {
            const table = document.querySelector('table.quote-table');
            if (!table) return null;

            const headers = Array.from(table.querySelectorAll('thead th'))
                .map(th => th.textContent.trim())
                .filter(Boolean);

            const rows = Array.from(table.querySelectorAll('tbody tr'))
                .map(row => Array.from(row.querySelectorAll('td')).map(td => td.textContent.trim()))
                .filter(row => row.filter(Boolean).length >= 4);

            return { headers, rows };
        }
        """
    )
    if not payload or not payload["rows"]:
        return None

    headers = payload.get("headers") or []
    rows = payload["rows"]
    df = pd.DataFrame(rows, columns=headers[: len(rows[0])] or None)
    df = df.replace('', pd.NA)
    if 'Graph' in df.columns:
        df = df.drop(columns=['Graph'])
    df.insert(0, 'Scraped_At', datetime.now().isoformat(timespec='seconds'))
    return df


def fetch_ncdex_table() -> Optional[pd.DataFrame]:
    print(f"🌐 Fetching NCDEX quotes from {NCDEX_URL}")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=NCDEX_HEADLESS)
        page = browser.new_page()
        try:
            page.goto(NCDEX_URL, timeout=NCDEX_PAGE_TIMEOUT, wait_until="domcontentloaded")
            page.wait_for_selector('table.quote-table tbody tr', timeout=NCDEX_WAIT_TIMEOUT)
            df = _extract_table(page)
            if df is None or len(df) < NCDEX_MIN_ROWS:
                print("❌ Quote table not ready or returned insufficient rows.")
                return None
            print(f"✅ Retrieved {len(df)} NCDEX rows")
            return df
        except PlaywrightTimeoutError:
            print("❌ Timed out waiting for NCDEX table to load.")
            return None
        finally:
            browser.close()


def fetch_live_ncdex_data(headless: bool = NCDEX_HEADLESS):
    return fetch_ncdex_table()


def get_or_update_ncdex_data() -> Optional[pd.DataFrame]:
    if NCDEX_CACHE_CSV.exists():
        age = datetime.now() - datetime.fromtimestamp(NCDEX_CACHE_CSV.stat().st_mtime)
        if age < NCDEX_CACHE_TTL:
            print("✅ Using cached NCDEX data (updated within 24 hours).")
            return pd.read_csv(NCDEX_CACHE_CSV)

    print("🔄 Cache stale or missing. Scraping NCDEX...")
    df = fetch_ncdex_table()
    if df is not None:
        df.to_csv(NCDEX_CACHE_CSV, index=False, encoding='utf-8-sig')
        print(f"💾 Saved NCDEX data to {NCDEX_CACHE_CSV}")
    return df


if __name__ == "__main__":
    result = fetch_ncdex_table()
    if result is not None:
        print(result.head())
    else:
        print("Failed to fetch NCDEX quotes.")
