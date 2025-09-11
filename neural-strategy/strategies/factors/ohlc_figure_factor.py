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
    from ohlc2fig import ohlc_to_image_without_volume
except ImportError:
    # Fallback import path
    sys.path.append(os.path.join(project_root, 'figure_model'))
    from ohlc2fig import ohlc_to_image_without_volume


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
            self.model_path = os.path.join(
                project_root, 'figure_model', 'model_saved', '2025-06-without-volume.pt'
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
            
            # Generate multi-timeframe images for all symbols
            images_tensor = self._generate_multi_timeframe_images(data, timestamp, symbols)
            
            if images_tensor is None or images_tensor.shape[0] == 0:
                return pd.Series(dtype=float, name=f'{self.name}_factor')
            
            # Get model predictions
            predictions = self._predict_returns(images_tensor)
            
            # Create factor series
            factor_scores = pd.Series(predictions, index=symbols, name=f'{self.name}_factor')
            
            # Apply confidence filtering if specified
            if self.confidence_threshold is not None:
                # Filter out low-confidence predictions (simple threshold on absolute value)
                confidence_mask = np.abs(predictions) >= self.confidence_threshold
                factor_scores = factor_scores[confidence_mask]
            
            return factor_scores.fillna(0.0)
            
        except Exception as e:
            warnings.warn(f"Error calculating OHLC factor: {e}")
            return pd.Series(dtype=float, name=f'{self.name}_factor')
    
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
            
            for symbol in symbols:
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
            
            if aggregated_data is None or len(aggregated_data) < self.lookback_periods:
                return None
            
            # Take the most recent lookback_periods
            recent_data = aggregated_data.tail(self.lookback_periods).copy()
            
            # Generate OHLC image
            image = ohlc_to_image_without_volume(
                recent_data,
                window=self.lookback_periods,
                height=self.image_height,
                open_col='open',
                high_col='high',
                low_col='low',
                close_col='close'
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
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'timeframes': self.timeframes,
            'lookback_periods': self.lookback_periods,
            'image_size': (self.image_height, self.image_width),
            'model_loaded': self.model is not None,
            'cache_size': len(self._image_cache)
        }