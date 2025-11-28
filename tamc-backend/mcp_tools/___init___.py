# mcp_tools/__init__.py
"""
mcp_tools package initializer.

This file imports the tool modules so that external code can do:
    from mcp_tools import price_tool, arrival_tool, advice_tool, weather_tool, chat_tool

and also allows `from mcp_tools import *` to work as expected.
"""

# Import submodules so they exist at package-level (helps pylint and IDEs)
from mcp_tools import price_tool
from mcp_tools import arrival_tool
from mcp_tools import advice_tool
from mcp_tools import chat_tool
from mcp_tools import weather_helper

# Public API of the package
__all__ = [
    "price_tool",
    "arrival_tool",
    "advice_tool",
    "chat_tool",
    "weather_helper",
]
