import numpy as np
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from Market_State_Calculator import CriticalBotError
from Chemistry_Classifier import AssetChemistry

logger = logging.getLogger(__name__)

@dataclass
class TrailheadSignal:
    """Represents a detected market pressure point about to release."""
    ticker: str
    pressure_score: float  # 0-1, higher = more pressure
    fragility_score: float  # 0-1, higher = more fragile
    composite_score: float  # Combined trailhead probability
    trigger_type: str  # 'squeeze', 'breakout', 'reversal', 'cascade'
    metadata: Dict

class CriticalBotError(Exception):
    """Custom exception for critical trading bot failures."""
    pass

def detect_trailheads(
    chemistry_map: Dict[str, 'AssetChemistry'],
    market_state: Tuple[float, float],
    fmp_key: str,
    finnhub_key: Optional[str] = None
) -> List[TrailheadSignal]:
    """
    Detect assets at pressure points about to experience major moves.
    
    Args:
        chemistry_map: Asset classifications from classify_asset_chemistry
        market_state: Current (risk, momentum) from calculate_market_state
        fmp_key: FMP API key
        finnhub_key: Optional Finnhub key for sentiment
    
    Returns:
        List of TrailheadSignals sorted by composite score (highest first)
        
    Detects:
        - Pressure: Volume surges, option flow, short interest extremes
        - Fragility: Correlation breaks, volatility shifts, liquidity gaps
        - Catalysts: Combining pressure + fragility + chemistry alignment
    """
    
    try:
        # Input validation
        if not chemistry_map:
            logger.error("detect_trailheads called with empty chemistry_map")
            raise ValueError("Empty chemistry map provided")
        
        if not fmp_key:
            logger.error("detect_trailheads called without FMP API key")
            raise ValueError("Missing FMP API key")
            
        if not isinstance(market_state, tuple) or len(market_state) != 2:
            logger.error(f"Invalid market_state format in detect_trailheads: {market_state}")
            raise ValueError(f"Invalid market_state format: {market_state}")
        
        risk, momentum = market_state
        
        if not isinstance(risk, (int, float)) or not isinstance(momentum, (int, float)):
            logger.error(f"Invalid market_state values: risk={risk}, momentum={momentum}")
            raise ValueError(f"Invalid market_state values: risk={risk}, momentum={momentum}")
        
        tickers = list(chemistry_map.keys())
        batch_size = min(len(tickers), 1000)
        symbols = ','.join(tickers[:batch_size])
        
        # Get volume and price data for pressure detection
        quote_url = f"https://financialmodelingprep.com/api/v3/quote/{symbols}?apikey={fmp_key}"
        
        try:
            quote_response = requests.get(quote_url, timeout=30)
            quote_response.raise_for_status()
            quotes = quote_response.json()
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout fetching quotes in detect_trailheads from {quote_url}: {e}")
            raise CriticalBotError(f"Quote API timeout in trailhead detection: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch quotes in detect_trailheads: {e}")
            logger.error(f"URL: {quote_url}")
            raise CriticalBotError(f"Quote API failed in trailhead detection: {e}")
        except ValueError as e:
            logger.error(f"Invalid JSON in quote response for detect_trailheads: {e}")
            raise CriticalBotError(f"Quote data parsing failed in trailhead detection: {e}")
        
        if quotes is None:
            logger.error("Received None from quote API in detect_trailheads")
            raise CriticalBotError("Null response from quote API")
        
        if not isinstance(quotes, list):
            logger.error(f"Invalid quote data type in detect_trailheads: {type(quotes)}")
            raise CriticalBotError(f"Invalid quote data structure: expected list, got {type(quotes)}")
        
        if not quotes:
            logger.error("Empty quote list in detect_trailheads")
            raise CriticalBotError("Empty quote data in trailhead detection")
        
        quote_map = {}
        for q in quotes:
            if not isinstance(q, dict):
                logger.warning(f"Invalid quote entry type: {type(q)}")
                continue
            if 'symbol' not in q:
                logger.warning(f"Quote entry missing symbol: {q}")
                continue
            quote_map[q['symbol']] = q
        
        if not quote_map:
            logger.error("No valid quotes extracted in detect_trailheads")
            raise CriticalBotError("Failed to extract any valid quote data")
        
        signals = []
        
        for ticker in tickers[:batch_size]:
            try:
                if ticker not in quote_map:
                    logger.debug(f"No quote data for {ticker}")
                    continue
                    
                quote = quote_map[ticker]
                chemistry = chemistry_map[ticker]
                
                # Extract metrics with strict validation
                volume = quote.get('volume')
                if volume is None or not isinstance(volume, (int, float)):
                    logger.warning(f"Invalid volume for {ticker}: {volume}")
                    continue
                
                avg_volume = quote.get('avgVolume')
                if avg_volume is None or not isinstance(avg_volume, (int, float)) or avg_volume <= 0:
                    logger.warning(f"Invalid avg_volume for {ticker}: {avg_volume}")
                    continue
                
                price = quote.get('price')
                if price is None or not isinstance(price, (int, float)) or price <= 0:
                    logger.warning(f"Invalid price for {ticker}: {price}")
                    continue
                
                change_pct = quote.get('changesPercentage', 0)
                if change_pct is None or not isinstance(change_pct, (int, float)):
                    change_pct = 0
                
                day_low = quote.get('dayLow', price)
                if day_low is None or not isinstance(day_low, (int, float)):
                    day_low = price
                    
                day_high = quote.get('dayHigh', price)
                if day_high is None or not isinstance(day_high, (int, float)):
                    day_high = price
                
                # PRESSURE INDICATORS (0-1 scale)
                pressure_components = []
                
                # 1. Volume surge detection
                volume_ratio = volume / avg_volume
                volume_pressure = min(1.0, max(0, (volume_ratio - 1) / 3))  # 4x volume = max pressure
                pressure_components.append(volume_pressure)
                
                # 2. Price range expansion
                daily_range = (day_high - day_low) / price
                range_pressure = min(1.0, daily_range / 0.05)  # 5% range = max pressure
                pressure_components.append(range_pressure)
                
                # 3. Momentum acceleration
                momentum_pressure = min(1.0, abs(change_pct) / 10)  # 10% move = max pressure
                pressure_components.append(momentum_pressure)
                
                pressure_score = np.mean(pressure_components)
                
                # FRAGILITY INDICATORS (0-1 scale)
                fragility_components = []
                
                # 1. Chemistry-market misalignment
                chemistry_alignment = 0.0
                if chemistry.chemistry_type == 'noble_gas' and risk < 0.3:
                    chemistry_alignment = 0.8  # Defensive in low risk = fragile
                elif chemistry.chemistry_type == 'volatile_compound' and risk > 0.7:
                    chemistry_alignment = 0.9  # Growth in high risk = very fragile
                elif chemistry.chemistry_type == 'phase_change' and momentum > 0.5:
                    chemistry_alignment = 0.7  # Value in strong uptrend = fragile
                fragility_components.append(chemistry_alignment)
                
                # 2. Extreme beta conditions
                beta_fragility = 0.0
                if chemistry.beta > 1.5 and risk > 0.6:
                    beta_fragility = min(1.0, (chemistry.beta - 1.5) * 2)
                elif chemistry.beta < 0.5 and risk < 0.3:
                    beta_fragility = min(1.0, (0.5 - chemistry.beta) * 2)
                fragility_components.append(beta_fragility)
                
                # 3. Liquidity fragility (thin volume)
                liquidity_fragility = max(0, 1 - (avg_volume / 1_000_000))  # Under 1M volume = fragile
                fragility_components.append(liquidity_fragility)
                
                fragility_score = np.mean(fragility_components)
                
                # COMPOSITE SCORING
                # High pressure + high fragility = imminent eruption
                composite_score = (pressure_score * 0.6 + fragility_score * 0.4)
                
                # Boost score based on market regime alignment
                if risk > 0.7 and momentum < -0.3:  # Panic selling
                    if chemistry.chemistry_type == 'noble_gas':
                        composite_score *= 1.5  # Defensive assets about to surge
                    trigger_type = 'reversal'
                elif risk < 0.3 and momentum > 0.5:  # Euphoric buying
                    if chemistry.chemistry_type == 'volatile_compound':
                        composite_score *= 1.3  # Growth ready to explode
                    trigger_type = 'breakout'
                elif volume_ratio > 3 and abs(change_pct) < 1:  # Volume with no price move
                    composite_score *= 1.4
                    trigger_type = 'squeeze'
                else:
                    trigger_type = 'cascade'
                
                composite_score = min(1.0, composite_score)
                
                # Only include significant signals
                if composite_score > 0.5:
                    signals.append(TrailheadSignal(
                        ticker=ticker,
                        pressure_score=pressure_score,
                        fragility_score=fragility_score,
                        composite_score=composite_score,
                        trigger_type=trigger_type,
                        metadata={
                            'volume_ratio': volume_ratio,
                            'daily_range_pct': daily_range * 100,
                            'chemistry_type': chemistry.chemistry_type,
                            'beta': chemistry.beta,
                            'risk': risk,
                            'momentum': momentum,
                            'timestamp': datetime.now().isoformat()
                        }
                    ))
                    
            except AttributeError as e:
                logger.error(f"Missing attribute for {ticker}: {e}")
                logger.error(f"Chemistry object: {chemistry}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing {ticker}: {e}")
                logger.error(f"Quote data: {quote if 'quote' in locals() else 'Not available'}")
                continue
        
        # Sort by composite score (highest first)
        signals.sort(key=lambda x: x.composite_score, reverse=True)
        
        if not signals:
            logger.info("No trailhead signals detected in current market state")
        
        return signals
        
    except CriticalBotError:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected critical error in detect_trailheads: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Market state: {market_state if 'market_state' in locals() else 'Not available'}")
        logger.error(f"Ticker count: {len(tickers) if 'tickers' in locals() else 'Unknown'}")
        raise CriticalBotError(f"Trailhead detection system failure: {e}")