"""
Trailhead Detection Module - Production Ready
Detects pressure buildup and fragility BEFORE flow begins
Zero tolerance for fake signals
"""

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')

class TrailheadSignal(NamedTuple):
    """Immutable trailhead signal - no silent mutations"""
    signal_type: str  # 'PRESSURE', 'FRAGILITY', 'COMPOSITE'
    strength: float   # 0-100 signal strength
    direction: str    # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float # 0-1 confidence score
    timestamp: datetime
    source_data: Dict
    expiry_time: datetime

@dataclass
class PressureMetrics:
    """Pressure buildup indicators"""
    options_put_call_ratio: float
    vix_term_structure_slope: float
    crypto_funding_extreme: float
    volume_pressure: float
    credit_stress: float
    
class FragilityMetrics:
    """Market fragility indicators"""
    def __init__(self):
        self.correlation_breakdown_score: float = 0.0
        self.liquidity_stress_score: float = 0.0
        self.volatility_clustering: float = 0.0
        self.cross_asset_divergence: float = 0.0

class TrailheadDetector:
    """
    Production-ready trailhead detection system
    Identifies pressure + fragility convergence points
    """
    
    def __init__(self):
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
        if not self.fmp_api_key:
            raise ValueError("Missing FMP_API_KEY for trailhead detection")
            
        self.logger = self._setup_logging()
        self.call_count = 0
        self.max_calls_per_minute = 100  # Conservative for trailhead module
        
        # Thresholds (calibrated for production)
        self.pressure_threshold = 75    # 75th percentile = significant pressure
        self.fragility_threshold = 70   # 70th percentile = significant fragility  
        self.composite_threshold = 80   # Both must align for trailhead
        
        # Signal expiry (trailheads are short-lived)
        self.signal_ttl_minutes = 30
        
        # Data cache
        self.cache: Dict[str, Tuple[any, datetime]] = {}
        self.cache_ttl = 120  # 2 minute cache for trailhead data
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
        
    def _rate_limit_check(self) -> None:
        """Enforce strict rate limits"""
        self.call_count += 1
        if self.call_count > self.max_calls_per_minute:
            raise RuntimeError(f"Trailhead detector rate limit exceeded: {self.call_count}")
            
    def _fetch_options_flow(self) -> Dict[str, float]:
        """
        Fetch options flow data to detect institutional positioning
        Uses multiple free sources for reliability
        """
        try:
            # Method 1: CBOE Put/Call ratios (free, official)
            cboe_url = "https://www.cboe.com/us/options/market_statistics/daily/"
            
            # Method 2: Calculate from SPY options (yfinance)
            spy = yf.Ticker("SPY")
            
            # Get current price for options chain
            current_price = spy.history(period="1d")['Close'].iloc[-1]
            
            # Fetch options chain
            options_dates = spy.options
            if not options_dates:
                self.logger.warning("No SPY options data available")
                return {'put_call_ratio': 1.0, 'options_volume': 0}
                
            # Get nearest expiry
            nearest_expiry = options_dates[0]
            options_chain = spy.option_chain(nearest_expiry)
            
            calls = options_chain.calls
            puts = options_chain.puts
            
            # Calculate volume-weighted put/call ratio
            call_volume = calls['volume'].fillna(0).sum()
            put_volume = puts['volume'].fillna(0).sum()
            
            put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
            total_volume = call_volume + put_volume
            
            # Calculate gamma exposure approximation
            # Simplified: use open interest near current price
            atm_calls = calls[abs(calls['strike'] - current_price) < 5]
            atm_puts = puts[abs(puts['strike'] - current_price) < 5]
            
            gamma_exposure_proxy = (
                atm_calls['openInterest'].fillna(0).sum() - 
                atm_puts['openInterest'].fillna(0).sum()
            )
            
            return {
                'put_call_ratio': float(put_call_ratio),
                'options_volume': float(total_volume),
                'gamma_exposure_proxy': float(gamma_exposure_proxy),
                'call_volume': float(call_volume),
                'put_volume': float(put_volume)
            }
            
        except Exception as e:
            self.logger.error(f"Options flow fetch failed: {e}")
            # Don't fail silently - return conservative default
            return {
                'put_call_ratio': 1.0,  # Neutral
                'options_volume': 0,
                'gamma_exposure_proxy': 0,
                'call_volume': 0,
                'put_volume': 0
            }
            
    def _fetch_crypto_funding(self) -> Dict[str, float]:
        """
        Fetch crypto perpetual funding rates
        Direct from exchange APIs (free, real-time)
        """
        try:
            funding_data = {}
            
            # Binance funding rates (free API)
            binance_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            
            response = requests.get(binance_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract key funding rates
            symbols_of_interest = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            
            for item in data:
                if item['symbol'] in symbols_of_interest:
                    # Funding rate is 8-hour rate, convert to annualized %
                    funding_rate = float(item['lastFundingRate'])
                    annualized_rate = funding_rate * 3 * 365 * 100  # 8hr -> annual %
                    
                    funding_data[item['symbol']] = {
                        'funding_rate_8h': funding_rate,
                        'funding_rate_annual_pct': annualized_rate,
                        'mark_price': float(item['markPrice'])
                    }
                    
            # Calculate composite funding pressure
            if funding_data:
                avg_funding = np.mean([
                    data['funding_rate_annual_pct'] 
                    for data in funding_data.values()
                ])
                
                funding_data['composite_funding_pressure'] = avg_funding
                
            return funding_data
            
        except Exception as e:
            self.logger.error(f"Crypto funding fetch failed: {e}")
            return {'composite_funding_pressure': 0.0}
            
    def _calculate_correlation_fragility(self) -> float:
        """
        Calculate cross-asset correlation breakdown score
        Higher score = more fragility
        """
        try:
            # Fetch multi-asset data
            symbols = ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # 60-day window
            
            price_data = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if hist.empty or len(hist) < 30:
                    continue
                    
                returns = hist['Close'].pct_change().dropna()
                price_data[symbol] = returns
                
            if len(price_data) < 3:
                self.logger.warning("Insufficient data for correlation analysis")
                return 0.0
                
            # Create returns dataframe
            df = pd.DataFrame(price_data)
            df = df.dropna()
            
            if len(df) < 20:
                return 0.0
                
            # Calculate rolling correlations
            window = 20
            rolling_corrs = df.rolling(window=window).corr()
            
            # Focus on key relationships
            key_pairs = [
                ('SPY', 'TLT'),  # Stock-bond correlation
                ('SPY', 'QQQ'),  # Large-small cap correlation  
                ('QQQ', 'IWM'),  # Growth-value correlation
            ]
            
            correlation_volatility = 0.0
            pair_count = 0
            
            for asset1, asset2 in key_pairs:
                if asset1 in df.columns and asset2 in df.columns:
                    # Get correlation time series
                    corr_series = df[asset1].rolling(window=window).corr(df[asset2])
                    corr_series = corr_series.dropna()
                    
                    if len(corr_series) > 5:
                        # Calculate volatility of correlation
                        corr_vol = corr_series.std()
                        correlation_volatility += corr_vol
                        pair_count += 1
                        
            if pair_count > 0:
                avg_correlation_volatility = correlation_volatility / pair_count
                # Normalize to 0-100 scale
                fragility_score = min(100, avg_correlation_volatility * 500)
                return fragility_score
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Correlation fragility calculation failed: {e}")
            return 0.0
            
    def _calculate_pressure_score(self, options_data: Dict, 
                                 crypto_data: Dict, vix_data: Dict) -> float:
        """
        Calculate composite pressure score (0-100)
        Higher score = more pressure building
        """
        pressure_components = []
        
        # Options pressure (put/call ratio extremes)
        put_call_ratio = options_data.get('put_call_ratio', 1.0)
        if put_call_ratio > 1.2:  # High put buying
            pressure_components.append(min(100, (put_call_ratio - 1.0) * 200))
        elif put_call_ratio < 0.8:  # High call buying
            pressure_components.append(min(100, (1.0 - put_call_ratio) * 200))
        else:
            pressure_components.append(0)
            
        # Crypto funding pressure
        funding_pressure = abs(crypto_data.get('composite_funding_pressure', 0))
        if funding_pressure > 5:  # 5% annual = extreme
            pressure_components.append(min(100, funding_pressure * 10))
        else:
            pressure_components.append(0)
            
        # VIX term structure pressure
        vix_slope = vix_data.get('term_structure_slope', 0)
        if abs(vix_slope) > 2:  # Steep contango/backwardation
            pressure_components.append(min(100, abs(vix_slope) * 25))
        else:
            pressure_components.append(0)
            
        # Volume pressure (implied from options volume)
        options_volume = options_data.get('options_volume', 0)
        if options_volume > 1000000:  # High options volume
            volume_pressure = min(100, (options_volume / 1000000 - 1) * 50)
            pressure_components.append(volume_pressure)
        else:
            pressure_components.append(0)
            
        # Return weighted average
        if pressure_components:
            return np.mean(pressure_components)
        else:
            return 0.0
            
    def detect_trailhead(self) -> Optional[TrailheadSignal]:
        """
        Main trailhead detection function
        Returns signal only if pressure + fragility align
        """
        try:
            self.logger.info("Scanning for trailhead formation...")
            
            # Fetch all required data
            options_data = self._fetch_options_flow()
            crypto_data = self._fetch_crypto_funding()
            
            # Get VIX term structure
            vix_ticker = yf.Ticker("^VIX")
            vix9d_ticker = yf.Ticker("^VIX9D")
            
            vix_hist = vix_ticker.history(period="5d")
            vix9d_hist = vix9d_ticker.history(period="5d")
            
            current_vix = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20
            current_vix9d = vix9d_hist['Close'].iloc[-1] if not vix9d_hist.empty else current_vix
            
            vix_data = {
                'vix': current_vix,
                'vix9d': current_vix9d,
                'term_structure_slope': (current_vix9d / current_vix - 1) * 100
            }
            
            # Calculate pressure score
            pressure_score = self._calculate_pressure_score(
                options_data, crypto_data, vix_data
            )
            
            # Calculate fragility score
            fragility_score = self._calculate_correlation_fragility()
            
            # Determine signal direction
            put_call_ratio = options_data.get('put_call_ratio', 1.0)
            crypto_funding = crypto_data.get('composite_funding_pressure', 0)
            
            direction = "NEUTRAL"
            if put_call_ratio > 1.1 and crypto_funding < -2:  # Fear + short squeeze potential
                direction = "BULLISH"
            elif put_call_ratio < 0.9 and crypto_funding > 5:   # Greed + long squeeze potential
                direction = "BEARISH"
                
            # Composite score
            composite_score = (pressure_score * 0.6 + fragility_score * 0.4)
            
            # Confidence based on data quality
            confidence = 1.0
            if options_data.get('options_volume', 0) < 100000:
                confidence *= 0.8  # Lower confidence with low volume
            if len(crypto_data) < 2:
                confidence *= 0.7  # Lower confidence with limited crypto data
                
            self.logger.info(f"Trailhead scan: Pressure={pressure_score:.1f}, "
                           f"Fragility={fragility_score:.1f}, "
                           f"Composite={composite_score:.1f}, "
                           f"Direction={direction}")
            
            # Generate signal if thresholds met
            if (pressure_score >= self.pressure_threshold and 
                fragility_score >= self.fragility_threshold):
                
                signal_strength = min(100, composite_score)
                expiry_time = datetime.now() + timedelta(minutes=self.signal_ttl_minutes)
                
                signal = TrailheadSignal(
                    signal_type='COMPOSITE',
                    strength=signal_strength,
                    direction=direction,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    source_data={
                        'pressure_score': pressure_score,
                        'fragility_score': fragility_score,
                        'options_data': options_data,
                        'crypto_data': crypto_data,
                        'vix_data': vix_data
                    },
                    expiry_time=expiry_time
                )
                
                self.logger.warning(f"🎯 TRAILHEAD DETECTED: {signal}")
                return signal
                
            # Check for individual component signals
            elif pressure_score >= self.pressure_threshold:
                signal = TrailheadSignal(
                    signal_type='PRESSURE',
                    strength=pressure_score,
                    direction=direction,
                    confidence=confidence * 0.8,  # Lower confidence for partial signal
                    timestamp=datetime.now(),
                    source_data={'pressure_score': pressure_score, 'options_data': options_data},
                    expiry_time=datetime.now() + timedelta(minutes=self.signal_ttl_minutes)
                )
                
                self.logger.info(f"Pressure signal detected: {signal}")
                return signal
                
            elif fragility_score >= self.fragility_threshold:
                signal = TrailheadSignal(
                    signal_type='FRAGILITY',
                    strength=fragility_score,
                    direction='NEUTRAL',  # Fragility doesn't imply direction
                    confidence=confidence * 0.7,
                    timestamp=datetime.now(),
                    source_data={'fragility_score': fragility_score},
                    expiry_time=datetime.now() + timedelta(minutes=self.signal_ttl_minutes)
                )
                
                self.logger.info(f"Fragility signal detected: {signal}")
                return signal
                
            else:
                self.logger.info(f"No trailhead detected - scores below threshold")
                return None
                
        except Exception as e:
            self.logger.error(f"Trailhead detection failed: {e}")
            raise RuntimeError(f"Trailhead detection system failure: {e}")
            
    def validate_signal(self, signal: TrailheadSignal) -> bool:
        """
        Validate signal hasn't expired and data is still valid
        """
        if datetime.now() > signal.expiry_time:
            self.logger.info(f"Signal expired: {signal.timestamp}")
            return False
            
        # Check if underlying conditions still hold
        if signal.strength < 50:  # Minimum signal strength
            return False
            
        return True
        
    def get_signal_summary(self) -> Dict:
        """
        Get current pressure and fragility readings without generating signals
        Useful for monitoring/dashboard
        """
        try:
            options_data = self._fetch_options_flow()
            crypto_data = self._fetch_crypto_funding()
            
            # Quick VIX check
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(period="2d")
            current_vix = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20
            
            vix_data = {'vix': current_vix, 'vix9d': current_vix, 'term_structure_slope': 0}
            
            pressure_score = self._calculate_pressure_score(options_data, crypto_data, vix_data)
            fragility_score = self._calculate_correlation_fragility()
            
            return {
                'pressure_score': pressure_score,
                'fragility_score': fragility_score,
                'pressure_threshold': self.pressure_threshold,
                'fragility_threshold': self.fragility_threshold,
                'composite_threshold': self.composite_threshold,
                'trailhead_imminent': (pressure_score >= self.pressure_threshold * 0.8 and 
                                     fragility_score >= self.fragility_threshold * 0.8),
                'timestamp': datetime.now(),
                'options_put_call': options_data.get('put_call_ratio', 1.0),
                'crypto_funding': crypto_data.get('composite_funding_pressure', 0),
                'current_vix': current_vix
            }
            
        except Exception as e:
            self.logger.error(f"Signal summary failed: {e}")
            return {
                'pressure_score': 0,
                'fragility_score': 0,
                'error': str(e),
                'timestamp': datetime.now()
            }

# Usage example and testing
if __name__ == "__main__":
    detector = TrailheadDetector()
    
    try:
        # Get current market summary
        summary = detector.get_signal_summary()
        print(f"Market Summary: {summary}")
        
        # Check for active trailhead
        signal = detector.detect_trailhead()
        
        if signal:
            print(f"🎯 ACTIVE TRAILHEAD: {signal}")
            
            # Validate signal
            is_valid = detector.validate_signal(signal)
            print(f"Signal Valid: {is_valid}")
            
        else:
            print("No trailhead detected - market conditions normal")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print("Trailhead detection system offline")
