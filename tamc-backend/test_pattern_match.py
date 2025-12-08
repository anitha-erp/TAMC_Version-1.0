import re
from typing import Optional, Dict

def quick_pattern_match(query: str) -> Optional[Dict]:
    """Test the pattern matching logic"""
    q = query.lower()
    
    # Month names for specific date detection
    month_pattern = r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"
    
    # Check for specific date patterns
    has_specific_date = (
        re.search(r"\d{1,2}(?:st|nd|rd|th)\s+" + month_pattern, q) or  # "1st dec", "2nd january"
        re.search(month_pattern + r"\s+\d{1,2}", q) or  # "december 1", "jan 15"
        re.search(r"\d{1,2}\s+" + month_pattern, q) or  # "1 december", "15 jan"
        re.search(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", q) or  # "2025-12-01", "2025/12/01"
        re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", q)  # "01/12/2025", "01-12-2025"
    )
    
    # Check for relative date patterns
    has_relative_date = (
        re.search(r"(yesterday|last\s+(week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday))", q) or
        re.search(r"\b\d+\s+days?\s+ago\b", q) or
        re.search(r"\b(was|were)\b.*\b(price|rate|cost|arrival|bag|lot)\b", q) or
        re.search(r"\b(price|rate|cost|arrival|bag|lot)\b.*(yesterday|last\s+week|ago|was|were)", q) or
        re.search(r"(total|number\s+of)\s+(bags?|lots?|arrivals?|farmers?).*(yesterday|last\s+(week|month|day))", q)
    )
    
    # Check for "on" keyword with date
    has_on_date = re.search(r"\bon\s+", q) and has_specific_date
    
    print(f"Testing Query: '{query}'")
    print(f"   has_specific_date: {has_specific_date}")
    print(f"   has_relative_date: {has_relative_date}")
    print(f"   has_on_date: {has_on_date}")
    
    if has_relative_date or has_specific_date or has_on_date:
        print("MATCH: Historical data query")
        return {
            "intent": "historical_query",
            "confidence": 0.95,
            "tools_needed": ["historical"]
        }
    else:
        print("NO MATCH: Not a historical query")
        return None

# Test the query
if __name__ == "__main__":
    test_queries = [
        "arrivals on 1/12/2025 in warangal",
        "total bags yesterday in khammam",
        "price 3 days ago",
        "arrivals tomorrow",  # Should NOT match (future)
    ]
    
    for query in test_queries:
        result = quick_pattern_match(query)
        print(f"Result: {result}")
        print("-" * 70)
