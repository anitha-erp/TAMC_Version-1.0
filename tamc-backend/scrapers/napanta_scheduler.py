#!/usr/bin/env python3
# Napanta Page Debugger - Inspect what's actually on the page

from playwright.sync_api import sync_playwright
import time

NAPANTA_BASE = "https://www.napanta.com/market-price"

def debug_page():
    with sync_playwright() as pw:
        # Launch in headful mode so you can see what's happening
        browser = pw.chromium.launch(
            headless=False,  # VISIBLE BROWSER
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        
        print(f"\n{'='*60}")
        print(f"NAVIGATING TO: {NAPANTA_BASE}")
        print(f"{'='*60}\n")
        
        try:
            page.goto(NAPANTA_BASE, timeout=60000)
            print("✓ Page loaded")
            
            # Wait a bit for JavaScript to execute
            print("\nWaiting 5 seconds for page to fully load...")
            time.sleep(5)
            
            # Take screenshot
            page.screenshot(path="napanta_debug.png")
            print("✓ Screenshot saved: napanta_debug.png")
            
            # Get page title
            title = page.title()
            print(f"\nPage Title: {title}")
            
            # Get page URL (in case of redirects)
            current_url = page.url
            print(f"Current URL: {current_url}")
            
            # Check for common selectors
            print(f"\n{'='*60}")
            print("CHECKING FOR FORM ELEMENTS")
            print(f"{'='*60}\n")
            
            selectors_to_check = [
                "select#state",
                "select[id='state']",
                "select[name='state']",
                "#state",
                "select#district",
                "select#market",
                "input[type='date']",
                "input[type=date]",
                "button:has-text('GO')",
                "form",
                "select",  # Any select element
            ]
            
            for selector in selectors_to_check:
                try:
                    count = page.locator(selector).count()
                    if count > 0:
                        print(f"✓ Found {count}x: {selector}")
                        # Get attributes of first match
                        if count > 0:
                            try:
                                elem = page.locator(selector).first
                                visible = elem.is_visible()
                                print(f"  - Visible: {visible}")
                                if selector.startswith("select"):
                                    options = elem.locator("option").count()
                                    print(f"  - Options: {options}")
                            except:
                                pass
                    else:
                        print(f"✗ Not found: {selector}")
                except Exception as e:
                    print(f"✗ Error checking {selector}: {e}")
            
            # Get all select elements and their IDs/names
            print(f"\n{'='*60}")
            print("ALL SELECT ELEMENTS ON PAGE")
            print(f"{'='*60}\n")
            
            selects = page.locator("select").all()
            if selects:
                for i, select in enumerate(selects):
                    try:
                        elem_id = select.get_attribute("id")
                        elem_name = select.get_attribute("name")
                        elem_class = select.get_attribute("class")
                        visible = select.is_visible()
                        print(f"Select #{i+1}:")
                        print(f"  ID: {elem_id}")
                        print(f"  Name: {elem_name}")
                        print(f"  Class: {elem_class}")
                        print(f"  Visible: {visible}")
                        print()
                    except Exception as e:
                        print(f"  Error reading select: {e}")
            else:
                print("No select elements found on page!")
            
            # Check for error messages or blocks
            print(f"\n{'='*60}")
            print("CHECKING FOR ERRORS/BLOCKS")
            print(f"{'='*60}\n")
            
            error_indicators = [
                "Access Denied",
                "403",
                "blocked",
                "captcha",
                "CAPTCHA",
                "robot",
                "bot",
                "security",
                "verify you are human"
            ]
            
            page_content = page.content()
            for indicator in error_indicators:
                if indicator.lower() in page_content.lower():
                    print(f"⚠ Found: '{indicator}' in page content")
            
            # Save HTML for manual inspection
            with open("napanta_debug.html", "w", encoding="utf-8") as f:
                f.write(page_content)
            print(f"\n✓ Full HTML saved: napanta_debug.html")
            
            # Get console messages
            print(f"\n{'='*60}")
            print("CONSOLE MESSAGES")
            print(f"{'='*60}\n")
            
            def log_console(msg):
                print(f"Console [{msg.type}]: {msg.text}")
            
            page.on("console", log_console)
            
            # Wait to observe
            print("\n\nBrowser will stay open for 30 seconds so you can inspect...")
            print("Check the browser window to see what's actually displayed!")
            time.sleep(30)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            # Still try to save what we can
            try:
                page.screenshot(path="napanta_error.png")
                print("\n✓ Error screenshot saved: napanta_error.png")
            except:
                pass
        
        finally:
            browser.close()
            print("\n✓ Browser closed")

if __name__ == "__main__":
    print("="*60)
    print("NAPANTA PAGE DEBUGGER")
    print("="*60)
    print("\nThis script will:")
    print("1. Open the page in a VISIBLE browser")
    print("2. Check what elements are present")
    print("3. Save screenshots and HTML")
    print("4. Keep browser open for 30 seconds for inspection")
    print("\nPress Ctrl+C to stop early if needed\n")
    
    input("Press ENTER to start debugging...")
    
    debug_page()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
    print("\nCheck these files:")
    print("- napanta_debug.png (screenshot)")
    print("- napanta_debug.html (full page HTML)")
    print("\nLook for:")
    print("- Are the dropdowns visible?")
    print("- Are there any error/block messages?")
    print("- What's the actual structure of the page?")