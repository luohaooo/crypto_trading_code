from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import glob
from datetime import datetime

app = Flask(__name__)

DATA_DIR = "/home/craz/crypto/crypto-data/future_data_2"

def get_available_symbols():
    """Get list of all available crypto symbols"""
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def get_symbol_data(symbol, start_date=None, end_date=None):
    """Get price data for a symbol within date range - loads ALL data points"""
    symbol_dir = os.path.join(DATA_DIR, symbol)
    if not os.path.exists(symbol_dir):
        return None
    
    # Get all CSV files for the symbol
    csv_files = glob.glob(os.path.join(symbol_dir, "*.csv"))
    csv_files.sort()
    
    # Filter files by date range if specified
    if start_date or end_date:
        filtered_files = []
        for file in csv_files:
            file_date_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
            try:
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d').date()
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                filtered_files.append(file)
            except ValueError:
                continue
        csv_files = filtered_files
    else:
        # If no date range specified, limit to recent files to avoid memory issues
        if len(csv_files) > 30:  # Limit to last 30 days
            csv_files = csv_files[-30:]
    
    if not csv_files:
        return None
    
    # Read and combine data - NO SAMPLING, load all data points
    dfs = []
    for i, file in enumerate(csv_files):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            
            # Progress indicator for long operations
            if len(csv_files) > 10 and i % 5 == 0:
                print(f"Processing file {i+1}/{len(csv_files)}: {file}")
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['open_time'] = pd.to_datetime(combined_df['open_time'])
    combined_df = combined_df.sort_values('open_time')
    
    print(f"Loaded {len(combined_df)} data points for {symbol}")
    
    return combined_df

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
        symbol_dir = os.path.join(DATA_DIR, symbol)
        if not os.path.exists(symbol_dir):
            return jsonify({'error': 'Symbol not found'}), 404
        
        csv_files = glob.glob(os.path.join(symbol_dir, "*.csv"))
        if not csv_files:
            return jsonify({'error': 'No data files found'}), 404
        
        csv_files.sort()
        
        # Get date range
        first_file = os.path.basename(csv_files[0]).split('_')[-1].replace('.csv', '')
        last_file = os.path.basename(csv_files[-1]).split('_')[-1].replace('.csv', '')
        
        return jsonify({
            'symbol': symbol,
            'first_date': first_file,
            'last_date': last_file,
            'total_files': len(csv_files)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)