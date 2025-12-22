import pandas as pd
import numpy as np

def ohlc_to_image_with_volume(
    ohlc_df: pd.DataFrame,
    window: int = 20,
    height: int = 64,
    price_height: int = 51,
    volume_height: int = 12,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume"
) -> np.ndarray:
    """
    Convert an OHLC DataFrame (including volume) into a black-and-white pixel image matrix.

    Parameters
    ----------
    ohlc_df : pd.DataFrame
        Must contain columns ['time', open_col, high_col, low_col, close_col, volume_col].
    window : int
        Number of data points represented by one image (default 20).
    height : int
        Total image height (pixels), default 64.
    price_height : int
        Number of pixels allocated for the price area (top part), default 51.
    volume_height : int
        Number of pixels allocated for the volume area (bottom part), default 12.
    open_col, high_col, low_col, close_col : str
        Column names for OHLC, defaults are 'open','high','low','close'.
    volume_col : str
        Column name for volume, default 'volume'.

    Returns
    ----------
    image : np.ndarray, shape (height, width), dtype=uint8
        Black-and-white image matrix where 1 is white (drawn OHLC/volume) and 0 is black background.
    """
    # 1. Validate dimensions
    N = len(ohlc_df)
    width = 3 * N
    assert N == window, f"Number of rows ({N}) != window ({window})"

    # 2. Extract price and volume arrays
    opens  = ohlc_df[open_col].to_numpy()
    highs  = ohlc_df[high_col].to_numpy()
    lows   = ohlc_df[low_col].to_numpy()
    closes = ohlc_df[close_col].to_numpy()
    volume = ohlc_df[volume_col].to_numpy()

    # 3. Global high/low for vertical scaling (price)
    max_h = highs.max()
    min_l = lows.min()
    price_range = max_h - min_l
    # assert price_range > 0, "Max and min prices are equal; cannot scale"

    # 3b. Global max/min for volume scaling
    max_v = volume.max()
    min_v = volume.min()
    volume_range = max_v - min_v
    # assert volume_range > 0, "Max and min volumes are equal; cannot scale"

    # 4. Initialize black background image
    img = np.zeros((height, width), dtype=np.uint8)

    # 5. Define price -> pixel-row mapping for the price area (row 0 is top)
    def p2y(p):
        # map: max_h -> 0; min_l -> price_height-1
        return np.round((max_h - p) / price_range * (price_height - 1)).astype(int)

    if price_range > 0:
        y_o = p2y(opens)
        y_h = p2y(highs)
        y_l = p2y(lows)
        y_c = p2y(closes)
    else:
        y_o = (np.ones(N) * (price_height - 1)).astype(int)
        y_h = (np.zeros(N)).astype(int)
        y_l = (np.ones(N) * (price_height - 1)).astype(int)
        y_c = (np.zeros(N)).astype(int)


    # 6. Define volume -> pixel-row mapping for the volume area (below the price area)
    def v2y(v):
        # map: max_v -> price_height + 1; min_v -> height-1
        return np.round((max_v - v) / volume_range * (volume_height - 1) + price_height + 1).astype(int)
    
    if volume_range > 0:
        v_h = v2y(volume)
    else:
        v_h = (np.ones(N) * (price_height + 1)).astype(int)

    # 7. Draw each day: central column is the vertical line for price; bottom part draws volume bars;
    #    left/right columns mark open/close ticks
    for i in range(N):
        x0 = i * 3
        x1 = x0 + 1
        x2 = x0 + 2

        # 7a. Vertical line for price: from high to low (inclusive)
        y_start, y_end = y_h[i], y_l[i]
        img[y_start : y_end + 1, x1] = 1

        # Draw volume bar from its mapped row down to the bottom (height-1)
        y_volume_start = v_h[i]
        img[y_volume_start : height - 1, x1] = 1

        # 7b. Left tick = open; right tick = close
        img[y_o[i], x0] = 1
        img[y_c[i], x2] = 1
    
    range_magnitude = max_h/min_l - 1

    return img, range_magnitude