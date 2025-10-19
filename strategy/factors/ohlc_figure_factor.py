"""
OHLC Figure Factor Implementation

Advanced factor that uses a trained CNN model to predict returns from OHLC candlestick images
across multiple timeframes (1h, 2h, 4h). The model processes visual patterns in price data
to generate trading signals for symbol ranking.

Note: This implementation expects HOURLY input data and prevents look-ahead bias.
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

from .factor_utils.ohlc2fig import ohlc_to_image_with_volume
from .base_factor import BaseFactor


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
                 timeframes: List[str] = ['1h', '2h', '4h']):
        """
        Initialize OHLC Figure Factor.

        Args:
            model_path: Path to trained model file. If None, uses default path.
            device: Device for model inference ('cpu', 'cuda', or 'auto')
            name: Factor name (auto-generated if None)
            lookback_periods: Number of OHLC periods for image generation (default 20)
            timeframes: List of timeframes for multi-timeframe analysis
        """
        if name is None:
            name = f"OHLC_Figure_{'+'.join(timeframes)}"

        super().__init__(name)

        # Configuration
        self.lookback_periods = lookback_periods
        self.timeframes = timeframes
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

    def calculate(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        """
        Calculate factor values using CNN predictions on OHLC images.

        Args:
            data: Multi-symbol OHLCV hourly data with MultiIndex (open_time, symbol)
            timestamp: Current timestamp for factor calculation (整点)

        Returns:
            pd.Series: Factor values indexed by symbol (higher = long candidates)
                      Returns NaN for symbols without sufficient historical data (80 hours)

        Note:
            - Requires 80 hours of historical data (20 periods × 4h max timeframe)
            - Prevents look-ahead bias: at timestamp T, only uses data before T
        """
        if self.model is None:
            warnings.warn("OHLC model not loaded, returning empty factor scores")
            return pd.Series(dtype=float, name=f'{self.name}_factor')

        try:
            # Get current symbols
            current_data = data.loc[data.index.get_level_values('open_time') == timestamp]
            symbols = current_data.index.get_level_values('symbol').unique().tolist()

            if len(symbols) == 0:
                return pd.Series(dtype=float, name=f'{self.name}_factor')

            # Generate images on-the-fly (data sufficiency check is done inside)
            images_tensor, valid_symbols = self._generate_multi_timeframe_images(data, timestamp, symbols)

            if images_tensor is None or images_tensor.shape[0] == 0:
                # No valid images generated
                return pd.Series(dtype=float, name=f'{self.name}_factor')

            # Get model predictions
            predictions = self._predict_returns(images_tensor)

            # Create factor series for valid symbols
            factor_scores = pd.Series(predictions, index=valid_symbols, name=f'{self.name}_factor')

            return factor_scores

        except Exception as e:
            warnings.warn(f"Error calculating OHLC factor: {e}")
            return pd.Series(dtype=float, name=f'{self.name}_factor')

    def _generate_multi_timeframe_images(self, data: pd.DataFrame, timestamp: pd.Timestamp,
                                       symbols: List[str]) -> tuple[Optional[torch.Tensor], List[str]]:
        """
        Generate multi-timeframe OHLC images for all symbols.

        Args:
            data: Multi-symbol OHLCV data
            timestamp: Current timestamp
            symbols: List of symbols to process

        Returns:
            tuple: (images_tensor, valid_symbols)
                - images_tensor: Batch of multi-timeframe images (N, 3, 64, 60) or None if error
                - valid_symbols: List of symbols that successfully generated images
        """
        MIN_HISTORY_HOURS = 80  # Required for 4h timeframe (20 periods × 4h)

        try:
            batch_images = []
            valid_symbols = []
            skipped_count = 0

            for symbol in tqdm(symbols, desc="Processing symbols", disable=len(symbols) < 10):
                try:
                    # Fast check: Does this symbol have enough historical data?
                    symbol_data = data.loc[data.index.get_level_values('symbol') == symbol]
                    available_times = symbol_data.index.get_level_values('open_time').unique()
                    historical_times = available_times[available_times < timestamp]

                    if len(historical_times) < MIN_HISTORY_HOURS:
                        # Skip symbol with insufficient data
                        skipped_count += 1
                        continue

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

            if skipped_count > 0:
                print(f"⚠️  Skipped {skipped_count} symbols with insufficient data (< {MIN_HISTORY_HOURS}h)")

            if len(batch_images) == 0:
                return None, []

            # Convert to tensor (N, 3, H, W)
            batch_array = np.stack(batch_images, axis=0)
            images_tensor = torch.from_numpy(batch_array).float()

            # Ensure correct shape (N, 3, 64, 60)
            expected_shape = (len(valid_symbols), len(self.timeframes), self.image_height, self.image_width)
            if images_tensor.shape != expected_shape:
                warnings.warn(f"Tensor shape mismatch: expected {expected_shape}, got {images_tensor.shape}")
                return None, []

            # Normalize to [0, 1] range (images are in {0, 1} already)
            images_tensor = images_tensor.clamp(0, 1)

            return images_tensor.to(self.device), valid_symbols

        except Exception as e:
            warnings.warn(f"Error generating multi-timeframe images: {e}")
            return None, []
    
    def _generate_single_timeframe_image(self, data: pd.DataFrame, timestamp: pd.Timestamp,
                                       symbol: str, timeframe: str) -> Optional[np.ndarray]:
        """
        Generate OHLC image for a single symbol and timeframe.

        Args:
            data: Multi-symbol OHLCV hourly data
            timestamp: Current timestamp (整点)
            symbol: Symbol to process
            timeframe: Timeframe for aggregation ('1h', '2h', '4h')

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
            # print(symbol, timeframe, aggregated_data)

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
        Aggregate hourly data to target timeframe using rolling window.

        Args:
            symbol_data: Single symbol OHLCV hourly data
            timeframe: Target timeframe ('1h', '2h', '4h')
            timestamp: Current timestamp (整点)

        Returns:
            pd.DataFrame: Aggregated OHLCV data or None if error

        Note:
            - Input data is assumed to be hourly
            - Uses rolling window aggregation (not fixed boundaries)
            - At timestamp 13:00 with 2h timeframe: aggregates [11:00, 12:00]
            - At timestamp 13:00 with 4h timeframe: aggregates [9:00, 10:00, 11:00, 12:00]
            - Prevents look-ahead bias by using only data before timestamp
        """
        try:
            # Parse timeframe and determine window size
            if timeframe == '1h':
                window_hours = 1
                periods_needed = self.lookback_periods  # 20 hours
            elif timeframe == '2h':
                window_hours = 2
                periods_needed = self.lookback_periods * 2  # 40 hours
            elif timeframe == '4h':
                window_hours = 4
                periods_needed = self.lookback_periods * 4  # 80 hours
            else:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            # Get available timestamps (only BEFORE current timestamp)
            available_times = symbol_data.index.get_level_values('open_time').unique()
            available_times = available_times[available_times < timestamp].sort_values()

            # Check if we have sufficient historical data
            if len(available_times) < periods_needed:
                return None

            # Reset index for easier processing
            symbol_data_reset = symbol_data.reset_index()
            symbol_data_reset = symbol_data_reset.set_index('open_time').sort_index()

            # For 1h timeframe, no aggregation needed
            if timeframe == '1h':
                selected_times = available_times[-periods_needed:]
                window_data = symbol_data_reset.loc[selected_times]

                if len(window_data) != periods_needed:
                    return None

                return window_data

            # For 2h and 4h, use rolling window aggregation
            # Take the most recent periods_needed hours
            recent_times = available_times[-periods_needed:]
            recent_data = symbol_data_reset.loc[recent_times].copy()

            # Create aggregated bars using rolling window
            # Generate lookback_periods bars in chronological order
            aggregated_bars = []

            # Generate bars from front to back (chronological order)
            for i in range(self.lookback_periods):
                # Calculate indices for this bar
                start_idx = i * window_hours
                end_idx = start_idx + window_hours

                if end_idx > len(recent_data):
                    # Not enough data
                    break

                window = recent_data.iloc[start_idx:end_idx]

                if len(window) != window_hours:
                    # Not enough data for this window
                    continue

                # Aggregate this window into one bar
                # open_time is the FIRST hour of the window (window start time)
                bar = {
                    'open_time': window.index[0],  # Use the first hour's timestamp as bar time
                    'open': window['open'].iloc[0],
                    'high': window['high'].max(),
                    'low': window['low'].min(),
                    'close': window['close'].iloc[-1],
                    'volume': window['volume'].sum()
                }
                aggregated_bars.append(bar)

            # Check if we got exactly lookback_periods bars
            if len(aggregated_bars) != self.lookback_periods:
                return None

            # Convert to DataFrame
            aggregated = pd.DataFrame(aggregated_bars)
            aggregated = aggregated.set_index('open_time')

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
            np.ndarray: Predicted returns (N,) - returns second value from each sample
        """
        try:
            with torch.no_grad():
                # Forward pass
                predictions = self.model(images_tensor)

                # Extract the second value from each sample output (class 1 probability)
                if predictions.shape[1] >= 2:
                    predictions_np = predictions[:, 1].cpu().numpy()
                else:
                    # Fallback to flatten if output has only one dimension
                    predictions_np = predictions.cpu().numpy().flatten()

                return predictions_np

        except Exception as e:
            warnings.warn(f"Error making predictions: {e}")
            return np.array([])
    
