from flask import Flask, request, jsonify, render_template
# from flask_socketio import SocketIO, emit
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import threading
import json

# Add parent directories to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import get_available_symbols
from backtest.core.engine import BacktestEngine
from backtest.strategies.long_short_factor import LongShortFactorStrategy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'backtest-secret-key'
# socketio = SocketIO(app, cors_allowed_origins="*")

# Global state for backtest management
active_backtests: Dict[str, Dict[str, Any]] = {}
backtest_counter = 0


@app.route('/')
def index():
    """Serve the main backtesting interface."""
    return render_template('backtest.html')


@app.route('/api/symbols')
def api_symbols():
    """Get available cryptocurrency symbols."""
    try:
        symbols = get_available_symbols()
        return jsonify({
            'symbols': symbols,
            'count': len(symbols)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest/start', methods=['POST'])
def start_backtest():
    """Start a new backtest."""
    global backtest_counter
    
    try:
        config = request.get_json()
        
        # Validate configuration
        required_fields = ['start_date', 'end_date', 'factor_config', 'strategy_config']
        for field in required_fields:
            if field not in config:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Parse dates
        start_date = datetime.fromisoformat(config['start_date'].replace('Z', '+00:00')).replace(tzinfo=None)
        end_date = datetime.fromisoformat(config['end_date'].replace('Z', '+00:00')).replace(tzinfo=None)
        
        # Create backtest ID
        backtest_counter += 1
        backtest_id = f"backtest_{backtest_counter}"
        
        # Initialize backtest engine
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=config.get('initial_capital', 1000000),
            rebalance_frequency=config.get('rebalance_frequency', '1D'),
            transaction_cost=config.get('transaction_cost', 0.001),
            max_symbols=config.get('max_symbols', 100)
        )
        
        # Initialize symbol universe
        symbols = config.get('symbols')
        universe = engine.initialize_universe(symbols)
        
        # Store backtest state
        active_backtests[backtest_id] = {
            'engine': engine,
            'config': config,
            'status': 'starting',
            'start_time': datetime.now(),
            'progress': 0.0,
            'universe': universe
        }
        
        # Start backtest in background thread
        thread = threading.Thread(
            target=run_backtest_async,
            args=(backtest_id, config['factor_config'], config['strategy_config'])
        )
        thread.start()
        
        return jsonify({
            'backtest_id': backtest_id,
            'status': 'started',
            'universe_size': len(universe),
            'estimated_time_minutes': _estimate_backtest_time(start_date, end_date, len(universe))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest/<backtest_id>/status')
def get_backtest_status(backtest_id: str):
    """Get current backtest status."""
    if backtest_id not in active_backtests:
        return jsonify({'error': 'Backtest not found'}), 404
    
    backtest_info = active_backtests[backtest_id]
    
    return jsonify({
        'backtest_id': backtest_id,
        'status': backtest_info['status'],
        'progress': backtest_info.get('progress', 0.0),
        'current_date': backtest_info.get('current_date'),
        'start_time': backtest_info['start_time'].isoformat(),
        'universe_size': len(backtest_info['universe']),
        'eta_minutes': backtest_info.get('eta_minutes')
    })


@app.route('/api/backtest/<backtest_id>/results')
def get_backtest_results(backtest_id: str):
    """Get backtest results."""
    if backtest_id not in active_backtests:
        return jsonify({'error': 'Backtest not found'}), 404
    
    backtest_info = active_backtests[backtest_id]
    
    if backtest_info['status'] not in ['completed', 'failed']:
        return jsonify({
            'backtest_id': backtest_id,
            'status': backtest_info['status'],
            'message': 'Backtest not yet completed'
        })
    
    results = backtest_info.get('results', {})
    return jsonify({
        'backtest_id': backtest_id,
        'status': backtest_info['status'],
        'results': results,
        'config': backtest_info['config']
    })


@app.route('/api/backtest/<backtest_id>/stop', methods=['POST'])
def stop_backtest(backtest_id: str):
    """Stop a running backtest."""
    if backtest_id not in active_backtests:
        return jsonify({'error': 'Backtest not found'}), 404
    
    backtest_info = active_backtests[backtest_id]
    
    if backtest_info['status'] == 'running':
        backtest_info['status'] = 'stopping'
        # The background thread will check this status and stop
    
    return jsonify({
        'backtest_id': backtest_id,
        'status': backtest_info['status']
    })


# @socketio.on('connect')
# def handle_connect():
#     """Handle WebSocket connection."""
#     print('Client connected')
#     emit('connected', {'message': 'Connected to backtest server'})


# @socketio.on('subscribe_backtest')
# def handle_subscribe(data):
#     """Handle subscription to backtest updates."""
#     backtest_id = data.get('backtest_id')
#     if backtest_id and backtest_id in active_backtests:
#         # Send current status
#         backtest_info = active_backtests[backtest_id]
#         emit('backtest_status', {
#             'backtest_id': backtest_id,
#             'status': backtest_info['status'],
#             'progress': backtest_info.get('progress', 0.0)
#         })


def run_backtest_async(backtest_id: str, factor_config: Dict, strategy_config: Dict):
    """Run backtest in background thread with real-time updates."""
    try:
        backtest_info = active_backtests[backtest_id]
        engine = backtest_info['engine']
        backtest_info['status'] = 'running'
        
        def progress_callback(progress_data: Dict):
            """Callback for progress updates."""
            if backtest_info['status'] == 'stopping':
                return False  # Signal to stop
            
            backtest_info['progress'] = progress_data['progress']
            backtest_info['current_date'] = progress_data['current_date']
            
            # Calculate ETA
            elapsed_minutes = (datetime.now() - backtest_info['start_time']).total_seconds() / 60
            if progress_data['progress'] > 0:
                eta_minutes = (elapsed_minutes / (progress_data['progress'] / 100)) - elapsed_minutes
                backtest_info['eta_minutes'] = max(0, eta_minutes)
            
            # Send real-time update via WebSocket
            # socketio.emit('backtest_progress', {
            #     'backtest_id': backtest_id,
            #     'progress': progress_data['progress'],
            #     'current_date': progress_data['current_date'],
            #     'eta_minutes': backtest_info.get('eta_minutes', 0)
            # })
            
            return True  # Continue
        
        # Run the backtest
        results = engine.run_backtest(
            factor_config=factor_config,
            strategy_config=strategy_config,
            progress_callback=progress_callback
        )
        
        # Store results
        backtest_info['results'] = results
        backtest_info['status'] = results['status']
        backtest_info['end_time'] = datetime.now()
        
        # Send completion notification
        # socketio.emit('backtest_complete', {
        #     'backtest_id': backtest_id,
        #     'status': results['status'],
        #     'results': results
        # })
        
    except Exception as e:
        backtest_info['status'] = 'failed'
        backtest_info['error'] = str(e)
        backtest_info['end_time'] = datetime.now()
        
        # socketio.emit('backtest_error', {
        #     'backtest_id': backtest_id,
        #     'error': str(e)
        # })


def _estimate_backtest_time(start_date: datetime, end_date: datetime, symbol_count: int) -> float:
    """Estimate backtest completion time in minutes."""
    days = (end_date - start_date).days
    # Rough estimate: 1 second per day per 10 symbols
    estimated_seconds = (days * symbol_count) / 10
    return max(1.0, estimated_seconds / 60)  # At least 1 minute


if __name__ == '__main__':
    print("Starting Market-Neutral Backtest Server...")
    print("Dashboard available at: http://localhost:5002")
    print("Also available at: http://127.0.0.1:5002")
    app.run(debug=True, host='127.0.0.1', port=5002)