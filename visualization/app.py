from flask import Flask, request, jsonify, render_template
import sys
import os
from datetime import datetime

# Add parent directory to Python path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import (
    get_available_symbols,
    get_symbol_data,
    get_symbol_info
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/symbols')
def api_symbols():
    """API endpoint to get all available symbols"""
    symbols = get_available_symbols()
    return jsonify(symbols)

@app.route('/api/data/<symbol>')
def api_data(symbol):
    """API endpoint to get price data for a symbol"""
    try:
        # Get query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        # Parse dates if provided
        start_date = None
        end_date = None
        if start_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        if end_date_str:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        
        df = get_symbol_data(symbol, start_date, end_date)
        if df is None:
            return jsonify({'error': 'Symbol not found or no data available'}), 404
        
        # Convert to format suitable for Chart.js - ALL data points
        data = []
        for _, row in df.iterrows():
            data.append({
                'time': row['open_time'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        return jsonify({
            'symbol': symbol,
            'data': data,
            'count': len(data),
            'start_date': df['open_time'].min().strftime('%Y-%m-%d') if len(df) > 0 else None,
            'end_date': df['open_time'].max().strftime('%Y-%m-%d') if len(df) > 0 else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbol-info/<symbol>')
def api_symbol_info(symbol):
    """API endpoint to get basic info about a symbol"""
    try:
        info = get_symbol_info(symbol)
        if info is None:
            return jsonify({'error': 'Symbol not found or no data available'}), 404
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)