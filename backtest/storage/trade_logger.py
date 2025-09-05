import sqlite3
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging


class TradeLogger:
    """
    SQLite-based trade logging and result storage system.
    
    Provides persistent storage for backtest results, trade logs,
    and performance metrics with efficient querying capabilities.
    """
    
    def __init__(self, db_path: str = "backtest_results.db"):
        """
        Initialize trade logger with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """
        Initialize database tables.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Backtests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    id TEXT PRIMARY KEY,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    rebalance_frequency TEXT NOT NULL,
                    transaction_cost REAL NOT NULL,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    universe_size INTEGER,
                    total_periods INTEGER
                )
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    value REAL NOT NULL,
                    commission REAL NOT NULL,
                    reason TEXT,
                    FOREIGN KEY (backtest_id) REFERENCES backtests (id)
                )
            """)
            
            # Portfolio snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    long_exposure REAL NOT NULL,
                    short_exposure REAL NOT NULL,
                    net_exposure REAL NOT NULL,
                    position_count INTEGER NOT NULL,
                    positions TEXT,
                    FOREIGN KEY (backtest_id) REFERENCES backtests (id)
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT NOT NULL,
                    total_return REAL,
                    annualized_return REAL,
                    volatility REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    final_portfolio_value REAL,
                    FOREIGN KEY (backtest_id) REFERENCES backtests (id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_backtest_timestamp ON trades (backtest_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_backtest_timestamp ON portfolio_snapshots (backtest_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, timestamp)")
            
            conn.commit()
    
    def create_backtest_record(self, 
                              backtest_id: str,
                              config: Dict[str, Any],
                              universe: List[str]) -> bool:
        """
        Create a new backtest record.
        
        Args:
            backtest_id: Unique backtest identifier
            config: Backtest configuration
            universe: List of symbols in universe
            
        Returns:
            True if record created successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO backtests (
                        id, start_date, end_date, initial_capital, rebalance_frequency,
                        transaction_cost, config, status, created_at, universe_size
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_id,
                    config.get('start_date'),
                    config.get('end_date'),
                    config.get('initial_capital', 1000000),
                    config.get('rebalance_frequency', '1D'),
                    config.get('transaction_cost', 0.001),
                    json.dumps(config),
                    'running',
                    datetime.now().isoformat(),
                    len(universe)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating backtest record: {e}")
            return False
    
    def log_trade(self, backtest_id: str, trade: Dict[str, Any]) -> bool:
        """
        Log a single trade.
        
        Args:
            backtest_id: Backtest identifier
            trade: Trade dictionary with required fields
            
        Returns:
            True if trade logged successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        backtest_id, timestamp, symbol, side, action,
                        quantity, price, value, commission, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_id,
                    trade['timestamp'],
                    trade['symbol'],
                    trade['side'],
                    trade['action'],
                    trade['quantity'],
                    trade['price'],
                    trade['value'],
                    trade.get('commission', 0),
                    trade.get('reason', '')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
            return False
    
    def log_portfolio_snapshot(self, backtest_id: str, snapshot: Dict[str, Any]) -> bool:
        """
        Log a portfolio snapshot.
        
        Args:
            backtest_id: Backtest identifier
            snapshot: Portfolio snapshot data
            
        Returns:
            True if snapshot logged successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO portfolio_snapshots (
                        backtest_id, timestamp, total_value, cash, long_exposure,
                        short_exposure, net_exposure, position_count, positions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    backtest_id,
                    snapshot['timestamp'],
                    snapshot['total_value'],
                    snapshot['cash'],
                    snapshot.get('long_exposure', 0),
                    snapshot.get('short_exposure', 0),
                    snapshot.get('net_exposure', 0),
                    snapshot.get('position_count', 0),
                    json.dumps(snapshot.get('positions', {}))
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging portfolio snapshot: {e}")
            return False
    
    def update_backtest_completion(self, 
                                  backtest_id: str, 
                                  status: str,
                                  performance_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update backtest completion status and metrics.
        
        Args:
            backtest_id: Backtest identifier
            status: Final status ('completed', 'failed', 'stopped')
            performance_metrics: Optional performance metrics
            
        Returns:
            True if updated successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update backtest status
                cursor.execute("""
                    UPDATE backtests 
                    SET status = ?, completed_at = ?
                    WHERE id = ?
                """, (status, datetime.now().isoformat(), backtest_id))
                
                # Insert performance metrics if provided
                if performance_metrics:
                    cursor.execute("""
                        INSERT INTO performance_metrics (
                            backtest_id, total_return, annualized_return, volatility,
                            sharpe_ratio, max_drawdown, total_trades, winning_trades,
                            final_portfolio_value
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        backtest_id,
                        performance_metrics.get('total_return'),
                        performance_metrics.get('annualized_return'),
                        performance_metrics.get('volatility'),
                        performance_metrics.get('sharpe_ratio'),
                        performance_metrics.get('max_drawdown'),
                        performance_metrics.get('total_trades'),
                        performance_metrics.get('winning_trades'),
                        performance_metrics.get('final_portfolio_value')
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating backtest completion: {e}")
            return False
    
    def get_backtest_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get history of recent backtests.
        
        Args:
            limit: Maximum number of backtests to return
            
        Returns:
            List of backtest records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT b.*, pm.total_return, pm.sharpe_ratio, pm.max_drawdown
                    FROM backtests b
                    LEFT JOIN performance_metrics pm ON b.id = pm.backtest_id
                    ORDER BY b.created_at DESC
                    LIMIT ?
                """, (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Error getting backtest history: {e}")
            return []
    
    def get_trade_log(self, backtest_id: str) -> pd.DataFrame:
        """
        Get trade log for a backtest as DataFrame.
        
        Args:
            backtest_id: Backtest identifier
            
        Returns:
            DataFrame with trade log
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, symbol, side, action, quantity, price, value, commission, reason
                    FROM trades
                    WHERE backtest_id = ?
                    ORDER BY timestamp
                """
                
                df = pd.read_sql_query(query, conn, params=(backtest_id,))
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting trade log: {e}")
            return pd.DataFrame()
    
    def get_portfolio_history(self, backtest_id: str) -> pd.DataFrame:
        """
        Get portfolio history for a backtest as DataFrame.
        
        Args:
            backtest_id: Backtest identifier
            
        Returns:
            DataFrame with portfolio history
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, total_value, cash, long_exposure, short_exposure,
                           net_exposure, position_count
                    FROM portfolio_snapshots
                    WHERE backtest_id = ?
                    ORDER BY timestamp
                """
                
                df = pd.read_sql_query(query, conn, params=(backtest_id,))
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {e}")
            return pd.DataFrame()
    
    def export_backtest_results(self, backtest_id: str, export_path: str) -> bool:
        """
        Export complete backtest results to files.
        
        Args:
            backtest_id: Backtest identifier
            export_path: Directory path for export
            
        Returns:
            True if export successful
        """
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Export trade log
            trade_log = self.get_trade_log(backtest_id)
            if not trade_log.empty:
                trade_log.to_csv(export_dir / f"{backtest_id}_trades.csv", index=False)
            
            # Export portfolio history
            portfolio_history = self.get_portfolio_history(backtest_id)
            if not portfolio_history.empty:
                portfolio_history.to_csv(export_dir / f"{backtest_id}_portfolio.csv", index=False)
            
            # Export backtest metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT b.*, pm.*
                    FROM backtests b
                    LEFT JOIN performance_metrics pm ON b.id = pm.backtest_id
                    WHERE b.id = ?
                """, (backtest_id,))
                
                result = cursor.fetchone()
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    metadata = dict(zip(columns, result))
                    
                    with open(export_dir / f"{backtest_id}_metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting backtest results: {e}")
            return False
    
    def cleanup_old_backtests(self, days_old: int = 30) -> int:
        """
        Clean up backtest records older than specified days.
        
        Args:
            days_old: Number of days to keep
            
        Returns:
            Number of records cleaned up
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get backtest IDs to delete
                cursor.execute("""
                    SELECT id FROM backtests 
                    WHERE created_at < ? AND status IN ('completed', 'failed')
                """, (cutoff_date.isoformat(),))
                
                backtest_ids = [row[0] for row in cursor.fetchall()]
                
                if backtest_ids:
                    # Delete related records
                    for backtest_id in backtest_ids:
                        cursor.execute("DELETE FROM trades WHERE backtest_id = ?", (backtest_id,))
                        cursor.execute("DELETE FROM portfolio_snapshots WHERE backtest_id = ?", (backtest_id,))
                        cursor.execute("DELETE FROM performance_metrics WHERE backtest_id = ?", (backtest_id,))
                        cursor.execute("DELETE FROM backtests WHERE id = ?", (backtest_id,))
                    
                    conn.commit()
                
                return len(backtest_ids)
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old backtests: {e}")
            return 0