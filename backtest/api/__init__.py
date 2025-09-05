"""
API endpoints and WebSocket handlers for real-time backtest communication.

This package contains the web interface components:
- REST API endpoints for backtest management
- WebSocket handlers for real-time progress updates
- Flask application for serving the dashboard
"""

from .backtest_api import BacktestAPI
from .websocket_handler import WebSocketHandler

__all__ = ['BacktestAPI', 'WebSocketHandler']