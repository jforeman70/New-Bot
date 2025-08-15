import numpy as np
import requests
from typing import Dict, Tuple
from datetime import datetime, timedelta
import logging
import sys  # Add this for sys.exit() in usage example

# Configure logging
logging.basicConfig(
   level=logging.ERROR,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CriticalBotError(Exception):
   """Custom exception for critical trading bot failures that require immediate halt."""
   pass

def calculate_market_state(fmp_key: str) -> Tuple[float, float, Dict]:
   """
   Calculate market state coordinates [risk, momentum] using free/cheap data.
   
   Risk (0-1): Market fear level. 0=complacent, 1=panic
   Momentum (-1 to 1): Directional conviction. -1=strong down, 1=strong up
   
   Returns:
       (risk, momentum, metadata)
   
   Raises:
       CriticalBotError: On any data retrieval or parsing failure
   """
   
   try:
       # VIX for risk baseline (free from Yahoo or FMP)
       vix_url = f"https://financialmodelingprep.com/api/v3/quote/^VIX?apikey={fmp_key}"
       vix_response = requests.get(vix_url, timeout=10)
       vix_response.raise_for_status()
       vix_data = vix_response.json()
       
       if not vix_data or not isinstance(vix_data, list) or len(vix_data) == 0:
           raise ValueError(f"Invalid VIX data structure: {vix_data}")
       
       vix = vix_data[0].get('price')
       if vix is None or not isinstance(vix, (int, float)) or vix <= 0:
           raise ValueError(f"Invalid VIX price: {vix}")
       
       # SPY 5-day return for momentum
       spy_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/SPY?timeseries=5&apikey={fmp_key}"
       spy_response = requests.get(spy_url, timeout=10)
       spy_response.raise_for_status()
       spy_data = spy_response.json()
       
       if not spy_data or 'historical' not in spy_data:
           raise ValueError(f"Invalid SPY data structure: {spy_data}")
       
       spy_prices = spy_data['historical']
       if not spy_prices or len(spy_prices) < 2:
           raise ValueError(f"Insufficient SPY price history: {len(spy_prices)} data points")
       
       if not all('close' in p and isinstance(p['close'], (int, float)) for p in spy_prices):
           raise ValueError("Missing or invalid close prices in SPY data")
       
       spy_return = (spy_prices[0]['close'] / spy_prices[-1]['close'] - 1) * 100
       
       # Put/Call ratio for sentiment confirmation (if available)
       # High P/C = fear, Low P/C = greed
       pc_ratio = 1.0  # Default, replace with actual when available
       
       # Calculate risk coordinate (0-1 scale)
       # VIX: 10=low fear, 30=high fear, 50+=extreme
       risk = np.clip((vix - 10) / 30, 0, 1)  # Normalize VIX to 0-1
       
       # Adjust risk based on P/C ratio if available
       if pc_ratio > 1.2:  # High put buying
           risk = min(risk * 1.2, 1.0)
       elif pc_ratio < 0.8:  # High call buying  
           risk *= 0.8
       
       # Calculate momentum coordinate (-1 to 1)
       # 5-day SPY: -5%=strong down, +5%=strong up
       momentum = np.clip(spy_return / 5, -1, 1)
       
       # Add term structure twist (VIX9D/VIX if available)
       # This detects short-term stress building
       
       metadata = {
           'timestamp': datetime.now().isoformat(),
           'vix': vix,
           'spy_5d_return': spy_return,
           'pc_ratio': pc_ratio,
           'state_quality': 'high' if vix > 0 else 'degraded'
       }
       
       return risk, momentum, metadata
       
   except requests.exceptions.RequestException as e:
       logger.error(f"API connection error in calculate_market_state: {e}")
       logger.error(f"Failed URL: {vix_url if 'vix_url' in locals() else 'Unknown'}")
       raise CriticalBotError(f"Failed to retrieve market data: {e}")
       
   except (KeyError, ValueError, TypeError, IndexError) as e:
       logger.error(f"Data parsing error in calculate_market_state: {e}")
       logger.error(f"VIX data: {vix_data if 'vix_data' in locals() else 'Not retrieved'}")
       logger.error(f"SPY data: {spy_data if 'spy_data' in locals() else 'Not retrieved'}")
       raise CriticalBotError(f"Failed to parse market data: {e}")
       
   except Exception as e:
       logger.error(f"Unexpected error in calculate_market_state: {e}")
       logger.error(f"Error type: {type(e).__name__}")
       raise CriticalBotError(f"Critical failure in market state calculation: {e}")


# Usage example:
# try:
#     risk, momentum, meta = calculate_market_state('your_fmp_key')
#     print(f"Market State: Risk={risk:.2f}, Momentum={momentum:.2f}")
#     if risk > 0.7 and momentum < -0.3:
#         print("HIGH FEAR + DOWNWARD = Potential bounce setup")
# except CriticalBotError:
#     # System will halt - do not continue trading with bad data
#     sys.exit(1)