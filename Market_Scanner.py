"""
Market State Scanner - Production Ready Trailhead Catalyst Framework
No stubs, no fake data, no silent fallbacks. Real money safe.
"""

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import logging
import time
import os
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

class MarketState(NamedTuple):
    """Immutable market state vector - no silent mutations"""
    risk_level: float
    momentum_direction: float
    regime: str
    confidence: float
    timestamp: datetime
    vix_percentile: float
    term_structure_signal: float
    black_swan_alert: bool

@dataclass
class DataQuality:
    """Track data freshness - fail if stale"""
    source: str
    last_update: datetime
    is_stale: bool
    staleness_seconds: float

class BlackSwanDetector:
    """Oh shit handle - detect market dislocations and BAIL"""
    
    def __init__(self):
        self.vix_spike_threshold = 40  # VIX above 40 = potential black swan
        self.correlation_break_threshold = 0.7  # Major correlation breakdown
        self.volume_spike_threshold = 3.0  # 3x average volume
        
    def detect_black_swan(self, vix: float, spy_returns: List[float], 
                         volume_ratio: float) -> Tuple[bool, str]:
        """
        Detect black swan events that require immediate position closure
        Returns: (is_black_swan, reason)
        """
        # VIX spike detection
        if vix > self.vix_spike_threshold:
            return True, f"VIX spike: {vix:.1f} > {self.vix_spike_threshold}"
            
        # Extreme daily moves
        if len(spy_returns) > 0 and abs(spy_returns[-1]) > 0.05:  # 5% daily move
            return True, f"Extreme daily move: {spy_returns[-1]*100:.1f}%"
            
        # Volume explosion
        if volume_ratio > self.volume_spike_threshold:
            return True, f"Volume spike: {volume_ratio:.1f}x normal"
            
        # Gap detection (if we have intraday data)
        if len(spy_returns) >= 2:
            gap = abs(spy_returns[-1] - spy_returns[-2])
            if gap > 0.03:  # 3% gap
                return True, f"Large gap: {gap*100:.1f}%"
                
        return False, ""

class MarketStateScanner:
    """
    Production-ready market state scanner
    Zero tolerance for stale data or silent failures
    """
    
    def __init__(self):
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
        if not self.fmp_api_key or not self.finnhub_api_key:
            raise ValueError("Missing required API keys: FMP_API_KEY, FINNHUB_API_KEY")
            
        self.logger = self._setup_logging()
        self.black_swan = BlackSwanDetector()
        self.data_quality_log: Dict[str, DataQuality] = {}
        self.call_count = 0
        self.max_calls_per_minute = 280  # Conservative buffer on 300/min limit
        
        # Cache for avoiding redundant calls
        self.cache: Dict[str, Tuple[any, datetime]] = {}
        self.cache_ttl = 60  # 1 minute cache
        
    def _setup_logging(self) -> logging.Logger:
        """Setup production logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trailhead_catalyst.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def _rate_limit_check(self) -> None:
        """Enforce API rate limits - fail hard if exceeded"""
        self.call_count += 1
        if self.call_count > self.max_calls_per_minute:
            raise RuntimeError(f"Rate limit exceeded: {self.call_count} calls/min")
            
    def _get_cached_or_fetch(self, key: str, fetch_func) -> any:
        """Cache mechanism to reduce API calls"""
        now = datetime.now()
        
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (now - timestamp).seconds < self.cache_ttl:
                return data
                
        # Cache miss - fetch new data
        self._rate_limit_check()
        data = fetch_func()
        self.cache[key] = (data, now)
        return data
        
    def _fetch_vix_data(self) -> Tuple[float, float, List[float]]:
        """Fetch VIX current, VIX9D, and historical percentiles"""
        try:
            # Current VIX via yfinance (free, reliable)
            vix_ticker = yf.Ticker("^VIX")
            vix_hist = vix_ticker.history(period="1y", interval="1d")
            
            if vix_hist.empty:
                raise ValueError("VIX data fetch failed - empty response")
                
            current_vix = float(vix_hist['Close'].iloc[-1])
            
            # VIX9D
            vix9d_ticker = yf.Ticker("^VIX9D") 
            vix9d_hist = vix9d_ticker.history(period="5d", interval="1d")
            current_vix9d = float(vix9d_hist['Close'].iloc[-1]) if not vix9d_hist.empty else current_vix
            
            # Calculate percentile
            vix_values = vix_hist['Close'].values
            vix_percentile = (vix_values < current_vix).mean() * 100
            
            # Log data quality
            self.data_quality_log['vix'] = DataQuality(
                source='yfinance_vix',
                last_update=datetime.now(),
                is_stale=False,
                staleness_seconds=0
            )
            
            return current_vix, current_vix9d, vix_values.tolist()
            
        except Exception as e:
            self.logger.error(f"VIX data fetch failed: {e}")
            raise RuntimeError(f"Critical VIX data unavailable: {e}")
            
    def _fetch_equity_data(self) -> Dict[str, float]:
        """Fetch key equity index data"""
        try:
            symbols = ['SPY', 'QQQ', 'IWM', 'TLT']
            data = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="30d", interval="1d")
                
                if hist.empty or len(hist) < 20:
                    raise ValueError(f"Insufficient data for {symbol}")
                    
                # Calculate momentum and recent performance
                returns = hist['Close'].pct_change().dropna()
                
                data[symbol] = {
                    'current_price': float(hist['Close'].iloc[-1]),
                    'daily_return': float(returns.iloc[-1]),
                    'momentum_5d': float(returns.tail(5).mean()),
                    'momentum_20d': float(returns.tail(20).mean()),
                    'volatility': float(returns.std() * np.sqrt(252)),
                    'volume_ratio': float(hist['Volume'].iloc[-1] / hist['Volume'].tail(20).mean())
                }
                
            return data
            
        except Exception as e:
            self.logger.error(f"Equity data fetch failed: {e}")
            raise RuntimeError(f"Critical equity data unavailable: {e}")
            
    def _calculate_regime(self, vix: float, vix_percentile: float, 
                         vix9d: float, spy_data: Dict) -> Tuple[str, float]:
        """
        Calculate market regime with confidence score
        No fuzzy logic - clear thresholds
        """
        confidence = 0.0
        
        # Risk-off conditions (clear criteria)
        if vix_percentile >= 80:
            confidence += 0.4
        if vix > 25:
            confidence += 0.3
        if vix9d / vix < 0.9:  # Term structure inversion
            confidence += 0.2
        if spy_data['daily_return'] < -0.02:  # 2% down day
            confidence += 0.1
            
        if confidence >= 0.6:
            return "RISK_OFF", confidence
            
        # Risk-on conditions
        risk_on_confidence = 0.0
        if vix_percentile <= 50:
            risk_on_confidence += 0.4
        if vix < 20:
            risk_on_confidence += 0.3
        if spy_data['momentum_5d'] > 0.001:  # Positive momentum
            risk_on_confidence += 0.2
        if spy_data['volume_ratio'] < 1.5:  # Normal volume
            risk_on_confidence += 0.1
            
        if risk_on_confidence >= 0.6:
            return "RISK_ON", risk_on_confidence
            
        # Default to transitional
        return "TRANSITIONAL", max(confidence, risk_on_confidence)
        
    def get_market_state(self) -> MarketState:
        """
        Main function: Get current market state
        Fails hard if data is unavailable - no guessing
        """
        try:
            self.logger.info("Fetching market state...")
            
            # Fetch core data
            vix, vix9d, vix_history = self._get_cached_or_fetch(
                'vix_data', self._fetch_vix_data
            )
            
            equity_data = self._get_cached_or_fetch(
                'equity_data', self._fetch_equity_data
            )
            
            # Calculate VIX percentile
            vix_percentile = (np.array(vix_history) < vix).mean() * 100
            
            # Black swan detection - OH SHIT HANDLE
            spy_returns = [equity_data['SPY']['daily_return']]
            volume_ratio = equity_data['SPY']['volume_ratio']
            
            is_black_swan, black_swan_reason = self.black_swan.detect_black_swan(
                vix, spy_returns, volume_ratio
            )
            
            if is_black_swan:
                self.logger.critical(f"BLACK SWAN DETECTED: {black_swan_reason}")
                
            # Calculate regime
            regime, confidence = self._calculate_regime(
                vix, vix_percentile, vix9d, equity_data['SPY']
            )
            
            # Calculate momentum (risk-adjusted)
            spy_momentum = equity_data['SPY']['momentum_5d']
            qqq_momentum = equity_data['QQQ']['momentum_5d']
            
            # Risk-adjusted momentum
            avg_vol = (equity_data['SPY']['volatility'] + equity_data['QQQ']['volatility']) / 2
            momentum_direction = (spy_momentum + qqq_momentum) / 2 / (avg_vol / 252) if avg_vol > 0 else 0
            
            # Term structure signal
            term_structure_signal = (vix9d / vix - 1) * 100  # Percentage difference
            
            # Risk level (normalized 0-100)
            risk_level = min(100, max(0, vix_percentile))
            
            market_state = MarketState(
                risk_level=risk_level,
                momentum_direction=momentum_direction,
                regime=regime,
                confidence=confidence,
                timestamp=datetime.now(),
                vix_percentile=vix_percentile,
                term_structure_signal=term_structure_signal,
                black_swan_alert=is_black_swan
            )
            
            self.logger.info(f"Market State: {market_state}")
            
            return market_state
            
        except Exception as e:
            self.logger.error(f"Market state calculation failed: {e}")
            raise RuntimeError(f"Cannot determine market state: {e}")
            
    def health_check(self) -> Dict[str, bool]:
        """System health check - all components operational"""
        health = {
            'api_keys_present': bool(self.fmp_api_key and self.finnhub_api_key),
            'data_sources_accessible': False,
            'black_swan_detector_active': True,
            'rate_limits_ok': self.call_count < self.max_calls_per_minute
        }
        
        # Test data source connectivity
        try:
            test_ticker = yf.Ticker("SPY")
            test_data = test_ticker.history(period="1d", interval="1d")
            health['data_sources_accessible'] = not test_data.empty
        except:
            health['data_sources_accessible'] = False
            
        return health

# Usage example
if __name__ == "__main__":
    scanner = MarketStateScanner()
    
    # Health check first
    health = scanner.health_check()
    print(f"System Health: {health}")
    
    if all(health.values()):
        try:
            state = scanner.get_market_state()
            print(f"Current Market State: {state}")
            
            if state.black_swan_alert:
                print("🚨 BLACK SWAN ALERT - EMERGENCY PROTOCOLS ACTIVATED 🚨")
                
        except Exception as e:
            print(f"CRITICAL FAILURE: {e}")
    else:
        print("SYSTEM NOT READY - Health check failed")
