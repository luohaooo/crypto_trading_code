from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import glob
import pickle
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

DATA_DIR = "/home/craz/crypto/crypto-data/future_data_2"
PICKLE_CACHE_DIR = "/home/craz/crypto/crypto-data/pickle_cache"

# Create pickle cache directory if it doesn't exist
Path(PICKLE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

def get_available_symbols():
    """Get list of all available crypto symbols"""
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def get_pickle_path(symbol):
    """Get the pickle file path for a symbol"""
    return os.path.join(PICKLE_CACHE_DIR, f"{symbol}_full_data.pkl")

def get_pickle_metadata_path(symbol):
    """Get the pickle metadata file path for a symbol"""
    return os.path.join(PICKLE_CACHE_DIR, f"{symbol}_metadata.pkl")

def is_pickle_cache_valid(symbol):
    """Check if pickle cache exists and is up to date"""
    pickle_path = get_pickle_path(symbol)
    metadata_path = get_pickle_metadata_path(symbol)
    
    if not os.path.exists(pickle_path) or not os.path.exists(metadata_path):
        return False
    
    try:
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Check if CSV files have changed since cache creation
        symbol_dir = os.path.join(DATA_DIR, symbol)
        csv_files = glob.glob(os.path.join(symbol_dir, "*.csv"))
        
        # Compare file count and last modification times
        if len(csv_files) != metadata.get('file_count', 0):
            return False
        
        # Check if any CSV file is newer than the cache
        cache_time = metadata.get('cache_time', 0)
        for csv_file in csv_files:
            if os.path.getmtime(csv_file) > cache_time:
                return False
        
        return True
        
    except Exception as e:
        print(f"Error checking pickle cache validity: {e}")
        return False

def create_pickle_cache(symbol):
    """Create pickle cache for a symbol by loading all CSV data"""
    print(f"Creating pickle cache for {symbol}...")
    
    symbol_dir = os.path.join(DATA_DIR, symbol)
    if not os.path.exists(symbol_dir):
        return None
    
    # Get all CSV files for the symbol
    csv_files = glob.glob(os.path.join(symbol_dir, "*.csv"))
    csv_files.sort()
    
    if not csv_files:
        return None
    
    # Read and combine all data
    dfs = []
    for i, file in enumerate(csv_files):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            
            # Progress indicator
            if len(csv_files) > 10 and i % 10 == 0:
                print(f"Processing file {i+1}/{len(csv_files)} for pickle cache")
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
    
    if not dfs:
        return None
    
    # Combine and process data
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['open_time'] = pd.to_datetime(combined_df['open_time'])
    combined_df = combined_df.sort_values('open_time')
    
    try:
        # Save data to pickle
        pickle_path = get_pickle_path(symbol)
        with open(pickle_path, 'wb') as f:
            pickle.dump(combined_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'file_count': len(csv_files),
            'record_count': len(combined_df),
            'cache_time': datetime.now().timestamp(),
            'date_range': {
                'start': combined_df['open_time'].min().isoformat(),
                'end': combined_df['open_time'].max().isoformat()
            }
        }
        
        metadata_path = get_pickle_metadata_path(symbol)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Created pickle cache for {symbol}: {len(combined_df):,} records")
        return combined_df
        
    except Exception as e:
        print(f"Error creating pickle cache: {e}")
        # If pickle creation fails, return the DataFrame anyway
        return combined_df

def load_from_pickle_cache(symbol):
    """Load data from pickle cache"""
    pickle_path = get_pickle_path(symbol)
    
    try:
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        print(f"📦 Loaded {len(df):,} records from pickle cache for {symbol}")
        return df
        
    except Exception as e:
        print(f"Error loading pickle cache: {e}")
        return None

def get_symbol_data(symbol, start_date=None, end_date=None):
    """Get price data for a symbol within date range - uses pickle cache for performance"""
    
    # Try to load from pickle cache first
    if is_pickle_cache_valid(symbol):
        df = load_from_pickle_cache(symbol)
        if df is not None:
            # Apply date filtering if needed
            if start_date or end_date:
                if start_date:
                    df = df[df['open_time'].dt.date >= start_date]
                if end_date:
                    df = df[df['open_time'].dt.date <= end_date]
            return df
    
    # If no valid cache, load from CSV and create cache
    df = create_pickle_cache(symbol)
    if df is None:
        return None
    
    # Apply date filtering if needed
    if start_date or end_date:
        if start_date:
            df = df[df['open_time'].dt.date >= start_date]
        if end_date:
            df = df[df['open_time'].dt.date <= end_date]
    
    print(f"Returning {len(df):,} data points for {symbol}")
    return df

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