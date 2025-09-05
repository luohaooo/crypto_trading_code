import sys
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.data_loader import get_symbol_data, get_available_symbols
from .memory_manager import MemoryManager


class DataManager:
    """
    Coordinates multi-symbol data loading with memory optimization.
    
    Handles efficient loading of cryptocurrency data across multiple symbols
    with chunked processing and intelligent caching.
    """
    
    def __init__(self, memory_manager: MemoryManager, max_workers: int = 10):
        """
        Initialize data manager.
        
        Args:
            memory_manager: Memory management instance
            max_workers: Maximum number of parallel workers for data loading
        """
        self.memory_manager = memory_manager
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def load_multi_symbol_data(self, 
                              symbols: List[str],
                              start_date: datetime,
                              end_date: datetime,
                              chunk_size_days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols with chunked processing.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date for data loading
            end_date: End date for data loading  
            chunk_size_days: Size of chunks in days for memory management
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        self.logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Create date key for caching
        date_key = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        symbol_data = {}
        
        # Check if we need to use chunked loading based on date range
        date_range_days = (end_date - start_date).days
        
        if date_range_days > chunk_size_days:
            symbol_data = self._load_chunked_data(
                symbols, start_date, end_date, chunk_size_days, date_key
            )
        else:
            symbol_data = self._load_batch_data(
                symbols, start_date.date(), end_date.date(), date_key
            )
        
        self.logger.info(f"Successfully loaded data for {len(symbol_data)} symbols")
        return symbol_data
    
    def _load_batch_data(self, 
                        symbols: List[str],
                        start_date: datetime,
                        end_date: datetime, 
                        date_key: str) -> Dict[str, pd.DataFrame]:
        """
        Load data for all symbols in a single batch.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date
            end_date: End date
            date_key: Cache key for date range
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        symbol_data = {}
        
        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Check cache first
            cache_futures = {}
            load_futures = {}
            
            for symbol in symbols:
                cached_data = self.memory_manager.get_cached_data(symbol, date_key)
                if cached_data is not None:
                    symbol_data[symbol] = cached_data
                else:
                    future = executor.submit(
                        self._load_single_symbol, 
                        symbol, start_date, end_date
                    )
                    load_futures[future] = symbol
            
            # Collect results from parallel loading
            for future in as_completed(load_futures):
                symbol = load_futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        # Optimize DataFrame for memory usage
                        df_optimized = self.memory_manager.optimize_dataframe(df)
                        symbol_data[symbol] = df_optimized
                        
                        # Cache the data
                        self.memory_manager.cache_data(symbol, date_key, df_optimized)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load data for {symbol}: {e}")
        
        return symbol_data
    
    def _load_chunked_data(self, 
                          symbols: List[str],
                          start_date: datetime,
                          end_date: datetime,
                          chunk_size_days: int,
                          base_date_key: str) -> Dict[str, pd.DataFrame]:
        """
        Load data in chunks to manage memory usage.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date
            end_date: End date
            chunk_size_days: Size of chunks in days
            base_date_key: Base cache key
            
        Returns:
            Dictionary mapping symbol to concatenated DataFrame
        """
        symbol_data = {symbol: [] for symbol in symbols}
        
        # Generate chunk date ranges
        current_date = start_date
        chunk_num = 0
        
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_size_days), end_date)
            chunk_date_key = f"{base_date_key}_chunk_{chunk_num}"
            
            self.logger.info(f"Loading chunk {chunk_num}: {current_date} to {chunk_end}")
            
            # Load chunk data
            chunk_data = self._load_batch_data(
                symbols, current_date.date(), chunk_end.date(), chunk_date_key
            )
            
            # Accumulate data
            for symbol, df in chunk_data.items():
                if df is not None:
                    symbol_data[symbol].append(df)
            
            # Memory cleanup after each chunk
            if chunk_num % 3 == 0:  # Every 3 chunks
                self.memory_manager.cleanup_expired_data(max_age_hours=1)
            
            current_date = chunk_end + timedelta(days=1)
            chunk_num += 1
        
        # Concatenate chunks for each symbol
        final_symbol_data = {}
        for symbol, df_list in symbol_data.items():
            if df_list:
                try:
                    combined_df = pd.concat(df_list, ignore_index=True)
                    combined_df = combined_df.sort_values('open_time')
                    combined_df = combined_df.drop_duplicates(subset=['open_time'])
                    final_symbol_data[symbol] = combined_df
                except Exception as e:
                    self.logger.error(f"Failed to concatenate data for {symbol}: {e}")
        
        return final_symbol_data
    
    def _load_single_symbol(self, 
                           symbol: str, 
                           start_date: datetime, 
                           end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Load data for a single symbol.
        
        Args:
            symbol: Symbol to load
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with price data or None if failed
        """
        try:
            # Convert datetime to date if needed
            start_date_obj = start_date.date() if hasattr(start_date, 'date') else start_date
            end_date_obj = end_date.date() if hasattr(end_date, 'date') else end_date
            
            df = get_symbol_data(symbol, start_date_obj, end_date_obj)
            if df is not None and len(df) > 0:
                # Ensure proper datetime formatting
                if 'open_time' in df.columns:
                    df['open_time'] = pd.to_datetime(df['open_time'])
                return df
            return None
        except Exception as e:
            self.logger.error(f"Error loading {symbol}: {e}")
            return None
    
    def get_symbol_universe(self, 
                           min_data_points: int = 1000,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[str]:
        """
        Get filtered symbol universe based on data availability.
        
        Args:
            min_data_points: Minimum number of data points required
            start_date: Optional start date for data checking
            end_date: Optional end date for data checking
            
        Returns:
            List of symbols with sufficient data
        """
        all_symbols = get_available_symbols()
        valid_symbols = []
        
        self.logger.info(f"Filtering {len(all_symbols)} symbols for data quality")
        
        # Sample check with a smaller date range if dates provided
        check_start = start_date or datetime(2023, 1, 1)
        check_end = end_date or datetime(2023, 2, 1)  # 1 month sample
        
        # Use parallel processing for symbol validation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._validate_symbol_data,
                    symbol, check_start, check_end, min_data_points
                ): symbol
                for symbol in all_symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    is_valid = future.result()
                    if is_valid:
                        valid_symbols.append(symbol)
                except Exception as e:
                    self.logger.warning(f"Failed to validate {symbol}: {e}")
        
        self.logger.info(f"Found {len(valid_symbols)} valid symbols")
        return sorted(valid_symbols)
    
    def _validate_symbol_data(self, 
                             symbol: str, 
                             start_date: datetime, 
                             end_date: datetime,
                             min_data_points: int) -> bool:
        """
        Validate if a symbol has sufficient data quality.
        
        Args:
            symbol: Symbol to validate
            start_date: Start date for validation
            end_date: End date for validation
            min_data_points: Minimum required data points
            
        Returns:
            True if symbol has sufficient data quality
        """
        try:
            df = get_symbol_data(symbol, start_date.date(), end_date.date())
            if df is None or len(df) < min_data_points:
                return False
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                return False
            
            # Check for excessive missing data
            missing_pct = df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns))
            if missing_pct > 0.1:  # More than 10% missing data
                return False
            
            return True
            
        except Exception:
            return False