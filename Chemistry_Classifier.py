import numpy as np
import requests
from typing import Dict, List, Literal, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

ChemistryType = Literal['noble_gas', 'volatile_compound', 'phase_change', 'catalyst_accelerant']

@dataclass
class AssetChemistry:
   """Represents an asset's chemical properties and market behavior profile."""
   ticker: str
   chemistry_type: ChemistryType
   beta: float
   confidence: float
   metadata: Dict

class CriticalBotError(Exception):
   """Custom exception for critical trading bot failures."""
   pass

def classify_asset_chemistry(
   tickers: List[str], 
   fmp_key: str,
   market_state: Tuple[float, float]
) -> Dict[str, AssetChemistry]:
   """
   Classify assets based on their 'chemical' reaction to market states.
   
   Args:
       tickers: List of stock symbols to classify
       fmp_key: FMP API key for data retrieval
       market_state: Current (risk, momentum) coordinates from calculate_market_state
   
   Returns:
       Dict mapping ticker -> AssetChemistry profile
       
   Chemistry Types:
       - noble_gas: Defensive, stable regardless of market state
       - volatile_compound: Explosive growth in low-risk environments  
       - phase_change: Value stocks that transform risk into future momentum
       - catalyst_accelerant: Amplifies current market energy
   """
   
   try:
       # Input validation
       if not tickers:
           raise ValueError("Empty ticker list provided")
       if not fmp_key:
           raise ValueError("Missing FMP API key")
       if not isinstance(market_state, tuple) or len(market_state) != 2:
           raise ValueError(f"Invalid market_state format: {market_state}")
       
       # Batch request for efficiency (FMP allows 1000 per call)
       batch_size = min(len(tickers), 1000)
       symbols = ','.join(tickers[:batch_size])
       
       # Get fundamental metrics
       profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbols}?apikey={fmp_key}"
       quote_url = f"https://financialmodelingprep.com/api/v3/quote/{symbols}?apikey={fmp_key}"
       
       try:
           profile_response = requests.get(profile_url, timeout=30)
           profile_response.raise_for_status()
           profiles = profile_response.json()
       except requests.exceptions.RequestException as e:
           logger.error(f"Failed to fetch profiles from {profile_url}: {e}")
           raise CriticalBotError(f"Profile API request failed: {e}")
       except ValueError as e:
           logger.error(f"Invalid JSON in profile response: {e}")
           raise CriticalBotError(f"Profile data parsing failed: {e}")
       
       try:
           quote_response = requests.get(quote_url, timeout=30)
           quote_response.raise_for_status()
           quotes = quote_response.json()
       except requests.exceptions.RequestException as e:
           logger.error(f"Failed to fetch quotes from {quote_url}: {e}")
           raise CriticalBotError(f"Quote API request failed: {e}")
       except ValueError as e:
           logger.error(f"Invalid JSON in quote response: {e}")
           raise CriticalBotError(f"Quote data parsing failed: {e}")
       
       if not profiles or not isinstance(profiles, list):
           logger.error(f"Invalid profile data structure: {type(profiles)}")
           raise CriticalBotError(f"Empty or invalid profile data for tickers: {tickers}")
       
       if not quotes or not isinstance(quotes, list):
           logger.error(f"Invalid quote data structure: {type(quotes)}")
           raise CriticalBotError(f"Empty or invalid quote data for tickers: {tickers}")
       
       # Create lookup dictionaries
       profile_map = {}
       quote_map = {}
       
       for p in profiles:
           if not isinstance(p, dict) or 'symbol' not in p:
               logger.warning(f"Invalid profile entry: {p}")
               continue
           profile_map[p['symbol']] = p
           
       for q in quotes:
           if not isinstance(q, dict) or 'symbol' not in q:
               logger.warning(f"Invalid quote entry: {q}")
               continue
           quote_map[q['symbol']] = q
       
       classifications = {}
       risk, momentum = market_state
       
       for ticker in tickers[:batch_size]:
           try:
               if ticker not in profile_map or ticker not in quote_map:
                   logger.warning(f"Missing data for {ticker}, skipping")
                   continue
               
               profile = profile_map[ticker]
               quote = quote_map[ticker]
               
               # Extract metrics with validation
               beta = profile.get('beta')
               if beta is None or not isinstance(beta, (int, float)):
                   beta = 1.0
               
               pe_ratio = quote.get('pe')
               if pe_ratio is None or not isinstance(pe_ratio, (int, float)):
                   pe_ratio = 20.0
               
               price = quote.get('price')
               if price is None or not isinstance(price, (int, float)) or price <= 0:
                   logger.warning(f"Invalid price for {ticker}: {price}")
                   continue
               
               last_div = profile.get('lastDiv', 0)
               if last_div is None or not isinstance(last_div, (int, float)):
                   last_div = 0
               
               div_yield = last_div / price
               
               market_cap = quote.get('marketCap', 0)
               if market_cap is None or not isinstance(market_cap, (int, float)):
                   market_cap = 0
               
               avg_volume = quote.get('avgVolume', 0)
               if avg_volume is None or not isinstance(avg_volume, (int, float)):
                   avg_volume = 0
               
               change_pct = quote.get('changesPercentage', 0)
               if change_pct is None or not isinstance(change_pct, (int, float)):
                   change_pct = 0
               
               # Classification logic based on empirical thresholds
               chemistry_type: ChemistryType
               confidence = 0.0
               
               if beta < 0.7 and div_yield > 0.025:
                   # Noble gas: Low beta + dividend = defensive
                   chemistry_type = 'noble_gas'
                   confidence = min(1.0, (0.7 - beta) * 2 + div_yield * 10)
                   
               elif beta > 1.3 and pe_ratio > 30:
                   # Volatile compound: High beta + high P/E = growth
                   chemistry_type = 'volatile_compound'
                   confidence = min(1.0, (beta - 1.3) * 2 + (pe_ratio - 30) / 50)
                   
               elif pe_ratio < 15 and pe_ratio > 0:
                   # Phase change: Low P/E value stocks
                   chemistry_type = 'phase_change'
                   confidence = min(1.0, (15 - pe_ratio) / 10)
                   
               else:
                   # Catalyst accelerant: Everything else amplifies market moves
                   chemistry_type = 'catalyst_accelerant'
                   confidence = 0.5 + abs(change_pct) / 20  # Recent volatility
               
               # Adjust confidence based on current market state
               if chemistry_type == 'noble_gas' and risk > 0.7:
                   confidence *= 1.3  # More valuable in high risk
               elif chemistry_type == 'volatile_compound' and risk < 0.3:
                   confidence *= 1.3  # Thrives in low risk
               elif chemistry_type == 'phase_change' and risk > 0.6 and momentum < -0.2:
                   confidence *= 1.5  # Perfect setup for value
               
               confidence = min(1.0, confidence)
               
               classifications[ticker] = AssetChemistry(
                   ticker=ticker,
                   chemistry_type=chemistry_type,
                   beta=beta,
                   confidence=confidence,
                   metadata={
                       'pe_ratio': pe_ratio,
                       'div_yield': div_yield,
                       'market_cap': market_cap,
                       'avg_volume': avg_volume,
                       'risk': risk,
                       'momentum': momentum
                   }
               )
               
           except Exception as e:
               logger.error(f"Failed to classify {ticker}: {e}")
               logger.error(f"Profile data: {profile if 'profile' in locals() else 'Not available'}")
               logger.error(f"Quote data: {quote if 'quote' in locals() else 'Not available'}")
               continue
       
       if not classifications:
           logger.error(f"Failed to classify any assets from {len(tickers)} tickers")
           logger.error(f"Profile map size: {len(profile_map)}, Quote map size: {len(quote_map)}")
           raise CriticalBotError("Failed to classify any assets - no valid data")
           
       return classifications
       
   except CriticalBotError:
       raise
       
   except Exception as e:
       logger.error(f"Unexpected critical error in classify_asset_chemistry: {e}")
       logger.error(f"Error type: {type(e).__name__}")
       logger.error(f"Tickers: {tickers[:10] if tickers else 'None'}")
       raise CriticalBotError(f"Asset classification system failure: {e}")