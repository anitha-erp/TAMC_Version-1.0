import pytest
import pandas as pd
from datetime import datetime, timedelta
from mcp_tools.historical_data_tool import HistoricalDataTool

@pytest.fixture
def tool():
    tool = HistoricalDataTool()
    # Create sample data for testing
    data = {
        'date': [datetime.now() - timedelta(days=i) for i in range(10)],
        'amc_name': ['Warangal'] * 10,
        'commodity_name': ['Cotton'] * 5 + ['Chilli'] * 5,
        'avg_price': [5000 + i*100 for i in range(10)],
        'min_price': [4800 + i*100 for i in range(10)],
        'max_price': [5200 + i*100 for i in range(10)],
        'arrivals': [100 + i*10 for i in range(10)]
    }
    tool.data = pd.DataFrame(data)
    return tool

def test_date_parsing_yesterday(tool):
    query = "price yesterday"
    start, end = tool.parse_historical_date(query)
    expected = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    assert start == expected
    assert end == expected

def test_date_parsing_last_week(tool):
    query = "arrivals last week"
    start, end = tool.parse_historical_date(query)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    expected_start = today - timedelta(days=7)
    expected_end = today - timedelta(days=1)
    assert start == expected_start
    assert end == expected_end

def test_date_parsing_n_days_ago(tool):
    query = "price 3 days ago"
    start, end = tool.parse_historical_date(query)
    expected = (datetime.now() - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    assert start == expected
    assert end == expected

def test_get_historical_prices(tool):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(days=5)
    end_date = today
    
    result = tool.get_historical_prices("Cotton", "Warangal", start_date, end_date)
    assert result["success"] == True
    assert len(result["data"]["historical_prices"]) > 0
    assert result["data"]["commodity"] == "Cotton"
    assert result["data"]["market"] == "Warangal"

def test_get_historical_arrivals(tool):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = today - timedelta(days=5)
    end_date = today
    
    result = tool.get_historical_arrivals("Chilli", "Warangal", start_date, end_date)
    assert result["success"] == True
    assert len(result["data"]["historical_arrivals"]) > 0
    assert result["data"]["commodity"] == "Chilli"

def test_no_data_found(tool):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    result = tool.get_historical_prices("Mango", "Warangal", today, today)
    assert result["success"] == False
    assert "No historical data found" in result["error"]
