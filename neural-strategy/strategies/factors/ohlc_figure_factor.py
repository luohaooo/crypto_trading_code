"""
OHLC Figure Factor Implementation

Advanced factor that uses a trained CNN model to predict returns from OHLC candlestick images
across multiple timeframes (3min, 15min, 1h). The model processes visual patterns in price data
to generate trading signals for symbol ranking.
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# Add parent directories to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

neural_strategy_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, neural_strategy_root)

figure_model_root = os.path.join(project_root, 'figure_model')
sys.path.insert(0, figure_model_root)

from .base_factor import BaseFactor

try:
    from ohlc2fig import ohlc_to_image_with_volume
except ImportError:
    # Fallback import path
    sys.path.append(os.path.join(project_root, 'figure_model'))
    from ohlc2fig import ohlc_to_image_with_volume


class OHLCImageCNN(nn.Module):
    def __init__(self):
        super(OHLCImageCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(46080, 1),
        )
        self.softmax = nn.Softmax(dim=1)
       
    def forward(self, x):
        # x = x.reshape(-1,1,64,60)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1,46080)
        x = self.fc1(x)
        # x = self.softmax(x)
        return x



class OHLCFigureFactor(BaseFactor):
    """
    OHLC Figure-based Factor using trained CNN for return prediction.
    
    This factor converts multi-timeframe OHLC data into images and uses a trained
    CNN model to predict future returns. Higher predicted returns indicate long
    candidates, lower predicted returns indicate short candidates.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 name: Optional[str] = None,
                 lookback_periods: int = 20,
                 timeframes: List[str] = ['3min', '15min', '1h'],
                 confidence_threshold: Optional[float] = None):
        """
        Initialize OHLC Figure Factor.
        
        Args:
            model_path: Path to trained model file. If None, uses default path.
            device: Device for model inference ('cpu', 'cuda', or 'auto')
            name: Factor name (auto-generated if None)
            lookback_periods: Number of OHLC periods for image generation (default 20)
            timeframes: List of timeframes for multi-timeframe analysis
            confidence_threshold: Minimum prediction confidence for filtering (optional)
        """
        if name is None:
            name = f"OHLC_Figure_{'+'.join(timeframes)}"
            
        super().__init__(name, lookback_periods)
        
        # Configuration
        self.timeframes = timeframes
        self.confidence_threshold = confidence_threshold
        self.image_height = 64
        self.image_width = 60  # 20 periods * 3 pixels per period
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Set model path
        if model_path is None:
            raise ValueError(
                "Model path must be provided. Please set model_path parameter or configure NEURAL_MODEL_PATH environment variable."
            )
        else:
            self.model_path = model_path
            
        # Initialize model
        self.model = None
        self._load_model()

        # Cache for repeated calculations
        self._image_cache = {}

        # Preloaded image cache for batch processing
        self._preloaded_images = {}  # {(symbol, timestamp, timeframe): tensor}
        self._batch_data = None
        self._batch_timestamps = []
        self._is_preloaded = False
        
    def _load_model(self):
        """Load the trained CNN model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            # Create model architecture
            self.model = OHLCImageCNN()
            
            # Load saved state dict
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume checkpoint is the state dict directly
                self.model.load_state_dict(checkpoint)
            
            # Move to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Loaded OHLC CNN model from {self.model_path}")
            print(f"📱 Using device: {self.device}")
            
        except Exception as e:
            warnings.warn(f"Failed to load OHLC model: {e}")
            self.model = None

    def preload_images(self, data: pd.DataFrame, timestamps: List[pd.Timestamp]):
        """
        Preload images for all symbols and timestamps to improve performance.

        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamps: List of timestamps to preload
        """
        print(f"🚀 开始预加载图像数据...")
        print(f"   时间框架: {self.timeframes}")
        print(f"   时间戳数量: {len(timestamps)}")

        # Store data and timestamps for future use
        self._batch_data = data
        self._batch_timestamps = timestamps

        # Get all available symbols
        all_symbols = data.index.get_level_values('symbol').unique().tolist()
        print(f"   交易对数量: {len(all_symbols)}")

        # Clear existing cache
        self._preloaded_images = {}

        # Aggregate data for all timeframes first using optimized batch method
        print("📊 聚合数据到不同时间框架...")
        aggregated_data = {}

        # Use the optimized aggregation method from the training notebook
        for timeframe in tqdm(self.timeframes, desc="聚合时间框架"):
            try:
                # Map timeframe to pandas frequency
                freq_map = {'3min': '3min', '15min': '15min', '1h': '60min', '2h': '120min', '4h': '240min'}
                if timeframe not in freq_map:
                    print(f"   ❌ 不支持的时间框架: {timeframe}")
                    continue

                freq = freq_map[timeframe]

                # Reset index for efficient resampling
                temp_data = data.reset_index()
                temp_data = temp_data.set_index('open_time')

                # Optimized batch aggregation: group by symbol and resample all at once
                print(f"   处理 {timeframe}...")

                # Group by symbol for parallel processing
                symbol_groups = temp_data.groupby('symbol')
                aggregated_list = []

                # Process symbols in batches for memory efficiency
                symbols = list(symbol_groups.groups.keys())
                batch_size = 50  # Process 50 symbols at once

                for i in range(0, len(symbols), batch_size):
                    batch_symbols = symbols[i:i + batch_size]

                    for symbol in batch_symbols:
                        try:
                            group = symbol_groups.get_group(symbol)

                            # Fast resampling with explicit aggregation
                            resampled = group.resample(freq, label='left', closed='left').agg({
                                'open': 'first',
                                'high': 'max',
                                'low': 'min',
                                'close': 'last',
                                'volume': 'sum'
                            }).dropna()

                            if len(resampled) > 0:
                                resampled['symbol'] = symbol
                                resampled = resampled.reset_index().set_index(['open_time', 'symbol'])
                                aggregated_list.append(resampled)

                        except Exception as e:
                            # Skip problematic symbols
                            continue

                if aggregated_list:
                    # Efficient concatenation and sorting
                    aggregated_data[timeframe] = pd.concat(aggregated_list, copy=False).sort_index()
                    print(f"   ✅ {timeframe}: {len(aggregated_data[timeframe]):,} 条记录")
                else:
                    print(f"   ❌ {timeframe}: 聚合失败")

            except Exception as e:
                print(f"   ❌ {timeframe} 聚合出错: {e}")
                continue

        # Batch generate images for all combinations - optimized order
        print("🎨 批量生成图像...")

        # Pre-calculate valid combinations to avoid wasted iterations
        valid_timeframes = list(aggregated_data.keys())
        print(f"有效时间框架: {valid_timeframes}")

        if not valid_timeframes:
            print("❌ 没有有效的时间框架数据")
            return

        total_combinations = len(all_symbols) * len(timestamps) * len(valid_timeframes)
        successful_images = 0

        with tqdm(total=total_combinations, desc="生成图像") as pbar:
            for timeframe in valid_timeframes:
                tf_data = aggregated_data[timeframe]
                print(f"   处理时间框架: {timeframe}")

                # Pre-build symbol index for faster lookup
                tf_symbols = set(tf_data.index.get_level_values('symbol').unique())
                valid_symbols_for_tf = [s for s in all_symbols if s in tf_symbols]

                print(f"   有效交易对: {len(valid_symbols_for_tf)}/{len(all_symbols)}")

                # Batch process symbols for this timeframe
                for symbol in valid_symbols_for_tf:
                    # Get all data for this symbol at once
                    try:
                        symbol_data = tf_data.loc[tf_data.index.get_level_values('symbol') == symbol]
                        symbol_timestamps = symbol_data.index.get_level_values('open_time')

                        # Process all timestamps for this symbol in batch
                        for timestamp in timestamps:
                            try:
                                # Fast window extraction using vectorized operations
                                window_data = self._extract_ohlc_window_vectorized(
                                    symbol_data, symbol_timestamps, timestamp, self.lookback_periods
                                )

                                if window_data is not None and len(window_data) == self.lookback_periods:
                                    # Generate image
                                    image = ohlc_to_image_with_volume(
                                        window_data,
                                        window=self.lookback_periods,
                                        height=self.image_height,
                                        open_col='open',
                                        high_col='high',
                                        low_col='low',
                                        close_col='close',
                                        volume_col='volume'
                                    )

                                    if image is not None and image.shape == (self.image_height, self.image_width):
                                        # Store in cache as tensor
                                        cache_key = (symbol, timestamp, timeframe)
                                        self._preloaded_images[cache_key] = torch.from_numpy(image).float()
                                        successful_images += 1

                            except Exception as e:
                                pass  # Skip failed combinations silently

                            pbar.update(1)

                    except Exception as e:
                        # Skip this symbol entirely if data access fails
                        pbar.update(len(timestamps))
                        continue

                # Skip any remaining combinations for symbols not in this timeframe
                remaining_invalid_symbols = len(all_symbols) - len(valid_symbols_for_tf)
                if remaining_invalid_symbols > 0:
                    pbar.update(remaining_invalid_symbols * len(timestamps))

        self._is_preloaded = True

        # Statistics
        success_rate = successful_images / total_combinations * 100 if total_combinations > 0 else 0
        memory_mb = successful_images * self.image_height * self.image_width * 4 / (1024 * 1024)  # float32

        print(f"✅ 预加载完成!")
        print(f"   成功生成: {successful_images:,} 张图像")
        print(f"   总组合数: {total_combinations:,}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   内存占用: ~{memory_mb:.1f} MB")

    def _extract_ohlc_window_vectorized(self, symbol_data: pd.DataFrame,
                                       symbol_timestamps: pd.Index,
                                       end_time: pd.Timestamp,
                                       window_size: int) -> Optional[pd.DataFrame]:
        """Vectorized version of OHLC window extraction for better performance."""
        try:
            # Use searchsorted for fast timestamp lookup
            valid_timestamps = symbol_timestamps[symbol_timestamps <= end_time]
            if len(valid_timestamps) < window_size:
                return None

            # Get the actual end time (closest available timestamp)
            actual_end_time = valid_timestamps[-1]

            # Find position efficiently
            end_pos = symbol_timestamps.get_loc(actual_end_time)

            if end_pos < window_size - 1:
                return None

            # Extract window using iloc for fastest access
            start_pos = end_pos - window_size + 1
            window_data = symbol_data.iloc[start_pos:end_pos + 1]

            if len(window_data) != window_size:
                return None

            # Remove symbol level from index for image generation
            return window_data.droplevel('symbol')

        except Exception:
            return None

    def _extract_ohlc_window_fast(self, data: pd.DataFrame, symbol: str,
                                 end_time: pd.Timestamp, window_size: int) -> Optional[pd.DataFrame]:
        """Fast version of OHLC window extraction for preloading."""
        try:
            # Get symbol data
            symbol_data = data.loc[data.index.get_level_values('symbol') == symbol]

            if len(symbol_data) < window_size:
                return None

            # Get timestamps and find end position
            timestamps = symbol_data.index.get_level_values('open_time')

            # Find the position closest to end_time
            valid_timestamps = timestamps[timestamps <= end_time]
            if len(valid_timestamps) == 0:
                return None

            actual_end_time = valid_timestamps.max()
            end_idx = timestamps.get_loc(actual_end_time)

            if end_idx < window_size - 1:
                return None

            # Extract window
            start_idx = end_idx - window_size + 1
            window_data = symbol_data.iloc[start_idx:end_idx + 1]

            if len(window_data) != window_size:
                return None

            return window_data.droplevel('symbol')

        except Exception:
            return None
    
    def calculate(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        """
        Calculate factor values using CNN predictions on OHLC images.

        Args:
            data: Multi-symbol OHLCV data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp for factor calculation

        Returns:
            pd.Series: Factor values indexed by symbol (higher = long candidates)
        """
        if self.model is None:
            warnings.warn("OHLC model not loaded, returning empty factor scores")
            return pd.Series(dtype=float, name=f'{self.name}_factor')

        # Validate data availability
        if not self.validate_data(data, timestamp):
            warnings.warn(f"Insufficient data for OHLC factor at {timestamp}")
            return pd.Series(dtype=float, name=f'{self.name}_factor')

        try:
            # Get current symbols
            current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
            symbols = current_data.index.get_level_values('symbol').unique().tolist()

            if len(symbols) == 0:
                return pd.Series(dtype=float, name=f'{self.name}_factor')

            # Use preloaded images if available
            if self._is_preloaded:
                images_tensor = self._get_preloaded_images(timestamp, symbols)
            else:
                # Fall back to the original method
                images_tensor = self._generate_multi_timeframe_images(data, timestamp, symbols)

            if images_tensor is None or images_tensor.shape[0] == 0:
                return pd.Series(dtype=float, name=f'{self.name}_factor')

            # Get model predictions
            predictions = self._predict_returns(images_tensor)

            # Create factor series (only for symbols that have valid predictions)
            valid_symbols = symbols[:len(predictions)]
            factor_scores = pd.Series(predictions, index=valid_symbols, name=f'{self.name}_factor')

            # Apply confidence filtering if specified
            if self.confidence_threshold is not None:
                confidence_mask = np.abs(predictions) >= self.confidence_threshold
                factor_scores = factor_scores[confidence_mask]

            return factor_scores.fillna(0.0)

        except Exception as e:
            warnings.warn(f"Error calculating OHLC factor: {e}")
            return pd.Series(dtype=float, name=f'{self.name}_factor')

    def _get_preloaded_images(self, timestamp: pd.Timestamp, symbols: List[str]) -> Optional[torch.Tensor]:
        """
        Get preloaded images for the given timestamp and symbols.

        Args:
            timestamp: Current timestamp
            symbols: List of symbols to get images for

        Returns:
            torch.Tensor: Batch of multi-timeframe images (N, 3, 64, 60) or None if error
        """
        try:
            batch_images = []
            valid_symbols = []

            for symbol in symbols:
                try:
                    # Collect images for all timeframes
                    symbol_images = []

                    for timeframe in self.timeframes:
                        cache_key = (symbol, timestamp, timeframe)

                        if cache_key in self._preloaded_images:
                            image_tensor = self._preloaded_images[cache_key]
                            symbol_images.append(image_tensor)
                        else:
                            # Missing image for this timeframe
                            break

                    # Only add if all timeframes are available
                    if len(symbol_images) == len(self.timeframes):
                        # Stack timeframes as channels (H, W) -> (3, H, W)
                        stacked_image = torch.stack(symbol_images, dim=0)
                        batch_images.append(stacked_image)
                        valid_symbols.append(symbol)

                except Exception as e:
                    warnings.warn(f"Failed to get preloaded images for {symbol}: {e}")
                    continue

            if len(batch_images) == 0:
                return None

            # Convert to tensor (N, 3, H, W)
            images_tensor = torch.stack(batch_images, dim=0)

            # Ensure correct shape (N, 3, 64, 60)
            expected_shape = (len(valid_symbols), len(self.timeframes), self.image_height, self.image_width)
            if images_tensor.shape != expected_shape:
                warnings.warn(f"Preloaded tensor shape mismatch: expected {expected_shape}, got {images_tensor.shape}")
                return None

            # Normalize to [0, 1] range
            images_tensor = images_tensor.clamp(0, 1)

            return images_tensor.to(self.device)

        except Exception as e:
            warnings.warn(f"Error getting preloaded images: {e}")
            return None
    
    def _generate_multi_timeframe_images(self, data: pd.DataFrame, timestamp: pd.Timestamp, 
                                       symbols: List[str]) -> Optional[torch.Tensor]:
        """
        Generate multi-timeframe OHLC images for all symbols.
        
        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            symbols: List of symbols to process
            
        Returns:
            torch.Tensor: Batch of multi-timeframe images (N, 3, 64, 60) or None if error
        """
        try:
            batch_images = []
            valid_symbols = []
            
            for symbol in tqdm(symbols, desc="Processing symbols", disable=len(symbols) < 10):
                try:
                    # Generate images for each timeframe
                    symbol_images = []
                    
                    for timeframe in self.timeframes:
                        image = self._generate_single_timeframe_image(
                            data, timestamp, symbol, timeframe
                        )
                        if image is not None:
                            # Ensure image is exactly (64, 60)
                            if image.shape != (self.image_height, self.image_width):
                                warnings.warn(f"Image shape mismatch for {symbol}-{timeframe}: "
                                             f"expected ({self.image_height}, {self.image_width}), "
                                             f"got {image.shape}")
                                continue
                            symbol_images.append(image)
                        else:
                            break  # Skip symbol if any timeframe fails
                    
                    # Only add if all timeframes successful
                    if len(symbol_images) == len(self.timeframes):
                        # Stack timeframes as channels (H, W) -> (3, H, W)
                        stacked_image = np.stack(symbol_images, axis=0)
                        batch_images.append(stacked_image)
                        valid_symbols.append(symbol)
                        
                except Exception as e:
                    warnings.warn(f"Failed to generate images for {symbol}: {e}")
                    continue
            
            if len(batch_images) == 0:
                return None
            
            # Convert to tensor (N, 3, H, W)
            batch_array = np.stack(batch_images, axis=0)
            images_tensor = torch.from_numpy(batch_array).float()
            
            # Ensure correct shape (N, 3, 64, 60)
            expected_shape = (len(valid_symbols), len(self.timeframes), self.image_height, self.image_width)
            if images_tensor.shape != expected_shape:
                warnings.warn(f"Tensor shape mismatch: expected {expected_shape}, got {images_tensor.shape}")
                return None
            
            # Normalize to [0, 1] range (images are in {0, 1} already)
            images_tensor = images_tensor.clamp(0, 1)
            
            return images_tensor.to(self.device)
            
        except Exception as e:
            warnings.warn(f"Error generating multi-timeframe images: {e}")
            return None
    
    def _generate_single_timeframe_image(self, data: pd.DataFrame, timestamp: pd.Timestamp,
                                       symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """
        Generate OHLC image for a single symbol and timeframe.
        
        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            symbol: Symbol to process
            timeframe: Timeframe for aggregation ('3min', '15min', '1h')
            
        Returns:
            np.ndarray: OHLC image (64, 60) or None if error
        """
        try:
            # Create cache key for efficient repeated calls
            cache_key = (symbol, timeframe, timestamp.isoformat())
            if cache_key in self._image_cache:
                return self._image_cache[cache_key]
            
            # Get symbol-specific data
            symbol_data = data.loc[data.index.get_level_values('symbol') == symbol].copy()
            
            if len(symbol_data) < self.lookback_periods:
                return None
            
            # Aggregate to target timeframe
            aggregated_data = self._aggregate_to_timeframe(symbol_data, timeframe, timestamp)
            # print(f"已完成 Aggregate to {timeframe}")
            
            if aggregated_data is None or len(aggregated_data) < self.lookback_periods:
                return None
            
            # Take the most recent lookback_periods
            recent_data = aggregated_data.tail(self.lookback_periods).copy()
            
            # Generate OHLC image
            image = ohlc_to_image_with_volume(
                recent_data,
                window=self.lookback_periods,
                height=self.image_height,
                open_col='open',
                high_col='high',
                low_col='low',
                close_col='close',
                volume_col='volume'
            )
            
            # Cache the result
            self._image_cache[cache_key] = image
            
            # Limit cache size to prevent memory issues
            if len(self._image_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self._image_cache.keys())[:200]
                for key in oldest_keys:
                    del self._image_cache[key]
            
            return image
            
        except Exception as e:
            warnings.warn(f"Error generating image for {symbol}-{timeframe}: {e}")
            return None
    
    def _aggregate_to_timeframe(self, symbol_data: pd.DataFrame, timeframe: str, 
                              timestamp: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Aggregate minute-level data to target timeframe.
        
        Args:
            symbol_data: Single symbol OHLCV data
            timeframe: Target timeframe ('3min', '15min', '1h')
            timestamp: Current timestamp
            
        Returns:
            pd.DataFrame: Aggregated OHLCV data or None if error
        """
        try:
            # Parse timeframe
            if timeframe == '3min':
                freq = '3min'
                periods_needed = self.lookback_periods * 3  # 3 minutes of 1-min data
            elif timeframe == '15min':
                freq = '15min'
                periods_needed = self.lookback_periods * 15  # 15 minutes of 1-min data
            elif timeframe == '1h':
                freq = '60min'
                periods_needed = self.lookback_periods * 60  # 60 minutes of 1-min data
            elif timeframe == '2h':
                freq = '120min'
                periods_needed = self.lookback_periods * 120  # 120 minutes of 1-min data
            elif timeframe == '4h':
                freq = '240min'
                periods_needed = self.lookback_periods * 240  # 240 minutes of 1-min data
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Get sufficient historical data
            end_time = timestamp
            available_times = symbol_data.index.get_level_values('open_time').unique()
            available_times = available_times[available_times <= end_time].sort_values()
            
            if len(available_times) < periods_needed:
                return None
            
            # Get data for aggregation
            start_idx = max(0, len(available_times) - periods_needed - 60)  # Extra buffer
            start_time = available_times[start_idx]
            
            window_data = symbol_data.loc[
                (symbol_data.index.get_level_values('open_time') >= start_time) &
                (symbol_data.index.get_level_values('open_time') <= end_time)
            ].copy()
            
            if len(window_data) == 0:
                return None
            
            # Reset index to work with time-based resampling
            window_data = window_data.reset_index()
            window_data = window_data.set_index('open_time')
            
            # Resample and aggregate
            aggregated = window_data.groupby(pd.Grouper(freq=freq)).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return aggregated
            
        except Exception as e:
            warnings.warn(f"Error aggregating to {timeframe}: {e}")
            return None
    
    def _predict_returns(self, images_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate return predictions from OHLC images using the trained model.
        
        Args:
            images_tensor: Batch of images (N, 3, 64, 60)
            
        Returns:
            np.ndarray: Predicted returns (N,)
        """
        try:
            with torch.no_grad():
                # Forward pass
                predictions = self.model(images_tensor)
                
                # Convert to numpy and flatten
                predictions_np = predictions.cpu().numpy().flatten()
                
                return predictions_np
                
        except Exception as e:
            warnings.warn(f"Error making predictions: {e}")
            return np.array([])
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        self._image_cache.clear()
        self._preloaded_images.clear()
        self._batch_data = None
        self._batch_timestamps = []
        self._is_preloaded = False

    def is_preloaded(self) -> bool:
        """Check if images are preloaded."""
        return self._is_preloaded

    def get_preload_stats(self) -> Dict:
        """Get statistics about preloaded images."""
        if not self._is_preloaded:
            return {"status": "not_preloaded"}

        total_images = len(self._preloaded_images)
        memory_mb = total_images * self.image_height * self.image_width * 4 / (1024 * 1024)

        return {
            "status": "preloaded",
            "total_images": total_images,
            "memory_mb": round(memory_mb, 1),
            "timestamps_count": len(self._batch_timestamps),
            "timeframes": self.timeframes
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            'model_path': self.model_path,
            'device': str(self.device),
            'timeframes': self.timeframes,
            'lookback_periods': self.lookback_periods,
            'image_size': (self.image_height, self.image_width),
            'model_loaded': self.model is not None,
            'cache_size': len(self._image_cache),
            'preload_status': self.is_preloaded(),
            'preloaded_images': len(self._preloaded_images) if self._is_preloaded else 0
        }

        if self._is_preloaded:
            info.update(self.get_preload_stats())

        return info