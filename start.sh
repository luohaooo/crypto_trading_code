#!/bin/bash
# Crypto Data Visualization Web App Startup Script

echo "Starting Crypto Data Visualization Web Application..."
echo "Available at: http://localhost:5000"
echo ""
echo "Features:"
echo "- 548+ cryptocurrency symbols"
echo "- Minute-level price data from 2019-2025"
echo "- Interactive charts with time window selection"
echo "- Line chart and OHLC visualization modes"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Install dependencies if needed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing Flask..."
    pip install flask pandas
fi

# Start the Flask application
cd /home/craz/crypto/crypto-trading
python app.py