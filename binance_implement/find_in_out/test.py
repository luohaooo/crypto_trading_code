import pandas as pd
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from binance_implement.account_stat.asset_stat import get_usdt_futures_asset_total

API_KEY = os.environ["BINANCE_API_KEY"]
API_SECRET = os.environ["BINANCE_API_SECRET"]

x = get_usdt_futures_asset_total(API_KEY, API_SECRET)   
print(x['wallet_balance'])