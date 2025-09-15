"""
OHLC Image Dataset Generator

This module generates deep learning datasets from OHLC cryptocurrency data.
It converts OHLC data to images and calculates future return labels for model training.

Features:
- Load data using monthly cache for efficiency
- Generate OHLC images at multiple timeframes (3min, 15min, 1h)
- Sample every 4 hours for dataset collection
- Calculate future returns at multiple horizons (1h to 168h)
- Save as PyTorch tensors organized by month
- Generate both image data and corresponding labels
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# Import required modules
from utils.data_loader import load_usdt_symbols_from_month_cache, get_usdt_symbols
from figure_model.ohlc_preprocessor import aggregate_ohlc_data
from figure_model.ohlc2fig import ohlc_to_image_without_volume, ohlc_to_image_with_volume


class OHLCImageDatasetGenerator:
    """
    Generator for OHLC image datasets with future return labels.
    
    This class handles:
    1. Data loading from monthly cache
    2. Data aggregation to different timeframes
    3. Image generation from OHLC data
    4. Future return calculation at multiple horizons
    5. Dataset organization and saving
    """
    
    def __init__(self, 
                 output_dir: str = "/home/craz/crypto/model_training/ohlc_img_dataset",
                 window_size: int = 20,
                 image_height: int = 64,
                 sampling_interval: str = '4h'):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir: Directory to save generated datasets
            window_size: Number of OHLC bars per image (default: 20)
            image_height: Height of generated images (default: 64)
            sampling_interval: Interval between samples (default: '4h')
        """
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.image_height = image_height
        self.sampling_interval = sampling_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Aggregation timeframes
        self.timeframes = ['3min', '15min', '1h']
        
        # Future return horizons (in hours)
        self.return_horizons = [1, 2, 4, 8, 12, 16, 18, 24, 30, 36, 48, 72, 96, 120, 144, 168]
        
        print(f"Initialized OHLC Image Dataset Generator:")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Window size: {self.window_size}")
        print(f"  Image height: {self.image_height}")
        print(f"  Sampling interval: {self.sampling_interval}")
        print(f"  Timeframes: {self.timeframes}")
        print(f"  Return horizons: {self.return_horizons}")
    
    def load_monthly_data(self, year: int, month: int, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load data for a specific month.
        
        Args:
            year: Year to load
            month: Month to load
            symbols: List of symbols to load (if None, loads all USDT symbols)
            
        Returns:
            DataFrame with 1-minute OHLC data
        """
        start_date = f"{year:04d}-{month:02d}-01"
        
        # Calculate end date (last day of month)
        if month == 12:
            end_date = f"{year+1:04d}-01-01"
        else:
            end_date = f"{year:04d}-{month+1:02d}-01"
        
        # Load data using monthly cache
        data = load_usdt_symbols_from_month_cache(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            use_fallback=True
        )
        
        print(f"Loaded data for {year:04d}-{month:02d}: {len(data):,} records, {len(data.index.get_level_values('symbol').unique())} symbols")
        
        return data
    
    def aggregate_data(self, data_1min: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Aggregate 1-minute data to multiple timeframes.
        
        Args:
            data_1min: 1-minute OHLC data
            
        Returns:
            Dictionary with timeframes as keys and aggregated data as values
        """
        aggregated_data = {}
        
        for timeframe in self.timeframes:
            print(f"Aggregating to {timeframe}...")
            try:
                aggregated = aggregate_ohlc_data(data_1min, timeframe)
                aggregated_data[timeframe] = aggregated
                print(f"  {timeframe}: {len(aggregated):,} records")
            except Exception as e:
                print(f"  Error aggregating to {timeframe}: {e}")
                continue
        
        return aggregated_data
    
    def generate_sampling_timestamps(self, data: pd.DataFrame) -> List[pd.Timestamp]:
        """
        Generate timestamps for sampling (every 4 hours).
        
        Args:
            data: OHLC data with datetime index
            
        Returns:
            List of timestamps for sampling
        """
        # Get all timestamps
        all_timestamps = data.index.get_level_values('open_time').unique().sort_values()
        
        # Start from the first timestamp that allows a full window
        start_time = all_timestamps[self.window_size - 1]
        
        # Generate sampling timestamps
        sampling_freq = pd.Timedelta(self.sampling_interval)
        sampling_timestamps = []
        
        current_time = start_time
        while current_time <= all_timestamps[-1]:
            if current_time in all_timestamps:
                sampling_timestamps.append(current_time)
            current_time += sampling_freq
        
        print(f"Generated {len(sampling_timestamps)} sampling timestamps")
        return sampling_timestamps
    
    def extract_window_data(self, data: pd.DataFrame, timestamp: pd.Timestamp, symbol: str) -> Optional[pd.DataFrame]:
        """
        Extract a window of data ending at the specified timestamp for a symbol.
        
        Args:
            data: OHLC data
            timestamp: End timestamp of the window
            symbol: Symbol to extract data for
            
        Returns:
            DataFrame with window_size rows of OHLC data, or None if insufficient data
        """
        try:
            # Get symbol data
            symbol_data = data.xs(symbol, level='symbol')
            
            # Find the position of the timestamp
            timestamps = symbol_data.index
            if timestamp not in timestamps:
                return None
            
            timestamp_idx = timestamps.get_loc(timestamp)
            
            # Check if we have enough data
            if timestamp_idx < self.window_size - 1:
                return None
            
            # Extract window
            start_idx = timestamp_idx - self.window_size + 1
            end_idx = timestamp_idx + 1
            
            window_data = symbol_data.iloc[start_idx:end_idx].copy()
            
            if len(window_data) != self.window_size:
                return None
                
            return window_data
            
        except Exception as e:
            return None
    
    def calculate_future_returns(self, data: pd.DataFrame, timestamp: pd.Timestamp, symbol: str) -> Optional[np.ndarray]:
        """
        Calculate future returns for specified horizons.
        
        Args:
            data: OHLC data (should be 1-hour data for consistent return calculation)
            timestamp: Current timestamp
            symbol: Symbol to calculate returns for
            
        Returns:
            Array of future returns for each horizon, or None if insufficient future data
        """
        try:
            # Get symbol data
            symbol_data = data.xs(symbol, level='symbol')
            
            # Get current price (close price at timestamp)
            if timestamp not in symbol_data.index:
                return None
            
            current_price = symbol_data.loc[timestamp, 'close']
            
            # Calculate future returns for each horizon
            future_returns = []
            
            for horizon_hours in self.return_horizons:
                future_timestamp = timestamp + pd.Timedelta(hours=horizon_hours)
                
                # Find the closest available timestamp
                available_timestamps = symbol_data.index
                future_candidates = available_timestamps[available_timestamps >= future_timestamp]
                
                if len(future_candidates) == 0:
                    return None  # Insufficient future data
                
                closest_future_timestamp = future_candidates[0]
                future_price = symbol_data.loc[closest_future_timestamp, 'close']
                
                # Calculate return
                return_pct = (future_price - current_price) / current_price
                future_returns.append(return_pct)
            
            return np.array(future_returns, dtype=np.float32)
            
        except Exception as e:
            return None
    
    def ohlc_to_image(self, ohlc_data: pd.DataFrame, include_volume: bool = False) -> np.ndarray:
        """
        Convert OHLC data to image array.
        
        Args:
            ohlc_data: OHLC DataFrame with window_size rows
            include_volume: Whether to include volume in the image
            
        Returns:
            Image array of shape (height, width)
        """
        if include_volume:
            image = ohlc_to_image_with_volume(
                ohlc_data,
                window=self.window_size,
                height=self.image_height
            )
        else:
            image = ohlc_to_image_without_volume(
                ohlc_data,
                window=self.window_size,
                height=self.image_height
            )
        
        return image
    
    def generate_monthly_dataset(self, year: int, month: int, symbols: Optional[List[str]] = None, max_samples_per_symbol: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate dataset for a specific month.
        
        Args:
            year: Year to process
            month: Month to process  
            symbols: List of symbols to process (if None, uses all USDT symbols)
            max_samples_per_symbol: Maximum samples per symbol (for testing)
            
        Returns:
            Dictionary with image tensors and label tensors for each timeframe
        """
        print(f"\n{'='*60}")
        print(f"Generating dataset for {year:04d}-{month:02d}")
        print(f"{'='*60}")
        
        # Load monthly data (1-minute)
        data_1min = self.load_monthly_data(year, month, symbols)
        
        if data_1min is None or len(data_1min) == 0:
            print("No data available for this month")
            return {}
        
        # Aggregate to different timeframes
        aggregated_data = self.aggregate_data(data_1min)
        
        # Also need 1-hour data for return calculation
        if '1h' not in aggregated_data:
            print("Error: 1-hour aggregation failed, cannot calculate returns")
            return {}
        
        data_1h = aggregated_data['1h']
        
        # Get available symbols
        available_symbols = data_1min.index.get_level_values('symbol').unique().tolist()
        if symbols:
            available_symbols = [s for s in symbols if s in available_symbols]
        
        print(f"Processing {len(available_symbols)} symbols")
        
        # Initialize results dictionary
        results = {}
        
        # Process each timeframe
        for timeframe in self.timeframes:
            if timeframe not in aggregated_data:
                print(f"Skipping {timeframe} (aggregation failed)")
                continue
                
            print(f"\nProcessing timeframe: {timeframe}")
            
            data_tf = aggregated_data[timeframe]
            
            # Generate sampling timestamps
            sampling_timestamps = self.generate_sampling_timestamps(data_tf)
            
            # Collect images and labels
            images = []
            labels = []
            metadata = []  # For debugging and analysis
            
            total_samples = 0
            successful_samples = 0
            
            for symbol in available_symbols:
                symbol_samples = 0
                
                for timestamp in sampling_timestamps:
                    # Check max samples limit
                    if max_samples_per_symbol and symbol_samples >= max_samples_per_symbol:
                        break
                    
                    total_samples += 1
                    
                    # Extract window data for image generation
                    window_data = self.extract_window_data(data_tf, timestamp, symbol)
                    if window_data is None:
                        continue
                    
                    # Calculate future returns
                    future_returns = self.calculate_future_returns(data_1h, timestamp, symbol)
                    if future_returns is None:
                        continue
                    
                    # Generate image
                    try:
                        image = self.ohlc_to_image(window_data, include_volume=False)
                        
                        # Add to collections
                        images.append(image)
                        labels.append(future_returns)
                        metadata.append({
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'timeframe': timeframe
                        })
                        
                        successful_samples += 1
                        symbol_samples += 1
                        
                    except Exception as e:
                        print(f"Error generating image for {symbol} at {timestamp}: {e}")
                        continue
                
                if symbol_samples > 0:
                    print(f"  {symbol}: {symbol_samples} samples")
            
            print(f"Total samples: {successful_samples}/{total_samples} successful")
            
            if len(images) > 0:
                # Convert to tensors
                # Images: (N, 1, Height, Width) 
                images_tensor = torch.from_numpy(np.stack(images)).unsqueeze(1).float()
                
                # Labels: (N, num_horizons)
                labels_tensor = torch.from_numpy(np.stack(labels)).float()
                
                print(f"Final tensor shapes:")
                print(f"  Images: {images_tensor.shape}")
                print(f"  Labels: {labels_tensor.shape}")
                
                results[timeframe] = {
                    'images': images_tensor,
                    'labels': labels_tensor,
                    'metadata': metadata
                }
            else:
                print(f"No successful samples for {timeframe}")
        
        return results
    
    def save_monthly_dataset(self, year: int, month: int, dataset: Dict[str, torch.Tensor]):
        """
        Save monthly dataset to disk.
        
        Args:
            year: Year of the dataset
            month: Month of the dataset
            dataset: Dataset dictionary with tensors
        """
        month_str = f"{year:04d}-{month:02d}"
        
        for timeframe, data in dataset.items():
            # Create filename
            filename = f"ohlc_dataset_{timeframe}_{month_str}.pt"
            filepath = self.output_dir / filename
            
            # Save tensors and metadata
            torch.save({
                'images': data['images'],
                'labels': data['labels'],
                'metadata': data['metadata'],
                'config': {
                    'year': year,
                    'month': month,
                    'timeframe': timeframe,
                    'window_size': self.window_size,
                    'image_height': self.image_height,
                    'sampling_interval': self.sampling_interval,
                    'return_horizons': self.return_horizons,
                    'num_samples': len(data['images'])
                }
            }, filepath)
            
            print(f"Saved {filepath}: {len(data['images'])} samples")
    
    def process_multiple_months(self, start_year: int, start_month: int, end_year: int, end_month: int, 
                               symbols: Optional[List[str]] = None, max_samples_per_symbol: Optional[int] = None):
        """
        Process multiple months in batch.
        
        Args:
            start_year: Starting year
            start_month: Starting month
            end_year: Ending year
            end_month: Ending month
            symbols: List of symbols to process
            max_samples_per_symbol: Maximum samples per symbol (for testing)
        """
        print(f"Processing months from {start_year:04d}-{start_month:02d} to {end_year:04d}-{end_month:02d}")
        
        current_year = start_year
        current_month = start_month
        
        while (current_year, current_month) <= (end_year, end_month):
            try:
                # Generate dataset for current month
                dataset = self.generate_monthly_dataset(
                    current_year, current_month, symbols, max_samples_per_symbol
                )
                
                # Save dataset
                if dataset:
                    self.save_monthly_dataset(current_year, current_month, dataset)
                else:
                    print(f"No dataset generated for {current_year:04d}-{current_month:02d}")
                
            except Exception as e:
                print(f"Error processing {current_year:04d}-{current_month:02d}: {e}")
                import traceback
                traceback.print_exc()
            
            # Move to next month
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1
        
        print(f"\nBatch processing completed!")
        print(f"Output directory: {self.output_dir}")
        
        # List generated files
        generated_files = list(self.output_dir.glob("*.pt"))
        print(f"Generated {len(generated_files)} dataset files:")
        for file in sorted(generated_files):
            size_mb = file.stat().st_size / (1024*1024)
            print(f"  {file.name}: {size_mb:.1f} MB")


def get_dataset_info(dataset_path: str) -> Dict:
    """
    Get information about a saved dataset.
    
    Args:
        dataset_path: Path to the .pt dataset file
        
    Returns:
        Dictionary with dataset information
    """
    try:
        data = torch.load(dataset_path)
        
        info = {
            'file': dataset_path,
            'config': data.get('config', {}),
            'images_shape': data['images'].shape,
            'labels_shape': data['labels'].shape,
            'num_samples': len(data['images']),
            'file_size_mb': Path(dataset_path).stat().st_size / (1024*1024)
        }
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


def list_all_datasets(dataset_dir: str = "/home/craz/crypto/model_training/ohlc_img_dataset"):
    """
    List all generated datasets with their information.
    
    Args:
        dataset_dir: Directory containing dataset files
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        print(f"Dataset directory does not exist: {dataset_dir}")
        return
    
    dataset_files = list(dataset_dir.glob("*.pt"))
    
    if not dataset_files:
        print(f"No dataset files found in {dataset_dir}")
        return
    
    print(f"Found {len(dataset_files)} dataset files:")
    print("-" * 80)
    
    total_size = 0
    total_samples = 0
    
    for file in sorted(dataset_files):
        info = get_dataset_info(str(file))
        
        if 'error' in info:
            print(f"{file.name}: Error - {info['error']}")
        else:
            config = info['config']
            print(f"{file.name}:")
            print(f"  Timeframe: {config.get('timeframe', 'unknown')}")
            print(f"  Month: {config.get('year', '?')}-{config.get('month', '?'):02d}")
            print(f"  Samples: {info['num_samples']:,}")
            print(f"  Image shape: {info['images_shape']}")
            print(f"  Labels shape: {info['labels_shape']}")
            print(f"  File size: {info['file_size_mb']:.1f} MB")
            print()
            
            total_size += info['file_size_mb']
            total_samples += info['num_samples']
    
    print("-" * 80)
    print(f"Total: {len(dataset_files)} files, {total_samples:,} samples, {total_size:.1f} MB")


if __name__ == "__main__":
    # Example usage
    generator = OHLCImageDatasetGenerator()
    
    # Test with a small subset
    print("Testing with limited data...")
    
    # Process one month with limited symbols and samples
    test_symbols = ['BTCUSDT', 'ETHUSDT']
    dataset = generator.generate_monthly_dataset(
        year=2020, 
        month=1, 
        symbols=test_symbols,
        max_samples_per_symbol=10  # Limit for testing
    )
    
    if dataset:
        generator.save_monthly_dataset(2020, 1, dataset)
        print("\nTest completed successfully!")
        
        # Show dataset info
        print("\nGenerated datasets:")
        list_all_datasets()
    else:
        print("Test failed - no dataset generated")