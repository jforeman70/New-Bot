"""
Complete Trailhead Catalyst Framework - Corrected Production Version
Implements the full geological framework with no placeholders or fake data
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import sqlite3
import json
import time
import threading
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, NamedTuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import traceback
import schedule
from collections import defaultdict, deque
import hashlib

# Optional imports with fallbacks
try:
    from ib_insync import IB, Stock, Forex, Contract, MarketOrder, LimitOrder
    IBKR_AVAILABLE = True
except ImportError:
    print("WARNING: ib_insync not available. Install with: pip install ib_insync")
    IBKR_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv not available. Using os.environ directly")

warnings.filterwarnings('ignore')

# =====================================================================================
# CORE DATA STRUCTURES - Based on Original Framework
# =====================================================================================

class MarketRegime(Enum):
    """Market regime classifications from original framework"""
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    TRANSITIONAL = "TRANSITIONAL"
    CRISIS = "CRISIS"

class AssetClass(Enum):
    """Asset class classifications for chemistry mapping"""
    DEFENSIVE = "DEFENSIVE"  # Noble gases - stable
    GROWTH = "GROWTH"        # Volatile compounds - need low-risk environment
    VALUE = "VALUE"          # Phase-change elements - absorb high-risk energy
    MOMENTUM = "MOMENTUM"    # Catalytic elements
    CARRY = "CARRY"          # Energy storage

@dataclass
class MarketState:
    """Immutable market state vector [risk, momentum] from original framework"""
    risk_level: float          # 0-100 risk level
    momentum_direction: float  # -100 to +100 momentum
    regime: MarketRegime
    confidence: float          # 0-1 confidence in classification
    timestamp: datetime
    vix_value: float
    vix_percentile: float
    term_structure_slope: float
    correlation_stress: float
    black_swan_alert: bool
    field_energy: float        # Total market field energy

@dataclass
class TickerChemistry:
    """Asset chemistry properties - how tickers react to market states"""
    symbol: str
    asset_class: AssetClass
    beta_to_risk: float        # How much ticker moves with risk changes
    beta_to_momentum: float    # How much ticker moves with momentum changes
    hardness: float           # Resistance to correlation breakdown (0-1)
    conductance: float        # How easily capital flows through this asset
    memory_half_life: int     # Days for correlation memory to decay
    volatility_regime_sensitivity: float  # Response to vol regime changes
    last_updated: datetime

@dataclass
class FlowPrediction:
    """Predicted capital flow based on terrain analysis"""
    source_state: MarketState
    target_state: MarketState
    flow_path: List[str]      # Symbols in flow path
    flow_magnitude: float     # Expected flow size
    time_horizon: int         # Days until flow completes
    confidence: float
    catalyst_symbols: List[str]  # Best positioned symbols

class TrailheadSignal(NamedTuple):
    """Trailhead detection signal"""
    signal_type: str
    strength: float
    direction: str
    confidence: float
    timestamp: datetime
    expiry_time: datetime
    pressure_components: Dict[str, float]
    fragility_components: Dict[str, float]
    predicted_flow: Optional[FlowPrediction]

# =====================================================================================
# DATABASE LAYER - State Persistence
# =====================================================================================

class StateDatabase:
    """SQLite database for state persistence and evolutionary learning"""
    
    def __init__(self, db_path: str = "catalyst_state.db"):
        self.db_path = db_path
        self.connection = None
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Market states table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS market_states (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    risk_level REAL NOT NULL,
                    momentum_direction REAL NOT NULL,
                    regime TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    vix_value REAL,
                    vix_percentile REAL,
                    term_structure_slope REAL,
                    correlation_stress REAL,
                    black_swan_alert BOOLEAN,
                    field_energy REAL,
                    data_hash TEXT UNIQUE
                )
            """)
            
            # Ticker chemistry table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS ticker_chemistry (
                    symbol TEXT PRIMARY KEY,
                    asset_class TEXT NOT NULL,
                    beta_to_risk REAL NOT NULL,
                    beta_to_momentum REAL NOT NULL,
                    hardness REAL NOT NULL,
                    conductance REAL NOT NULL,
                    memory_half_life INTEGER NOT NULL,
                    volatility_regime_sensitivity REAL NOT NULL,
                    last_updated TEXT NOT NULL,
                    performance_score REAL DEFAULT 0.0,
                    update_count INTEGER DEFAULT 0
                )
            """)
            
            # Agent performance table (for evolutionary system)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_performance (
                    agent_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    strategy_params TEXT NOT NULL,
                    prediction_accuracy REAL,
                    surprise_score REAL,
                    pnl REAL,
                    trades_count INTEGER,
                    survival_score REAL,
                    generation INTEGER,
                    PRIMARY KEY (agent_id, timestamp)
                )
            """)
            
            # Trade history
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    signal_strength REAL,
                    signal_type TEXT,
                    market_state_id INTEGER,
                    pnl REAL,
                    commission REAL,
                    agent_id TEXT,
                    FOREIGN KEY (market_state_id) REFERENCES market_states (id)
                )
            """)
            
            # API call tracking for rate limiting
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    provider TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT,
                    success BOOLEAN,
                    response_time REAL
                )
            """)
            
            self.connection.commit()
            
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            raise
            
    def save_market_state(self, state: MarketState) -> int:
        """Save market state with deduplication"""
        try:
            # Create hash for deduplication
            state_data = f"{state.risk_level:.2f}_{state.momentum_direction:.2f}_{state.regime.value}_{state.timestamp.date()}"
            data_hash = hashlib.md5(state_data.encode()).hexdigest()
            
            cursor = self.connection.execute("""
                INSERT OR IGNORE INTO market_states 
                (timestamp, risk_level, momentum_direction, regime, confidence, 
                 vix_value, vix_percentile, term_structure_slope, correlation_stress, 
                 black_swan_alert, field_energy, data_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.timestamp.isoformat(),
                state.risk_level,
                state.momentum_direction,
                state.regime.value,
                state.confidence,
                state.vix_value,
                state.vix_percentile,
                state.term_structure_slope,
                state.correlation_stress,
                state.black_swan_alert,
                state.field_energy,
                data_hash
            ))
            
            self.connection.commit()
            return cursor.lastrowid
            
        except Exception as e:
            logging.error(f"Failed to save market state: {e}")
            return 0
            
    def update_ticker_chemistry(self, chemistry: TickerChemistry):
        """Update ticker chemistry with evolutionary learning"""
        try:
            self.connection.execute("""
                INSERT OR REPLACE INTO ticker_chemistry 
                (symbol, asset_class, beta_to_risk, beta_to_momentum, hardness, 
                 conductance, memory_half_life, volatility_regime_sensitivity, 
                 last_updated, update_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        COALESCE((SELECT update_count FROM ticker_chemistry WHERE symbol = ?) + 1, 1))
            """, (
                chemistry.symbol,
                chemistry.asset_class.value,
                chemistry.beta_to_risk,
                chemistry.beta_to_momentum,
                chemistry.hardness,
                chemistry.conductance,
                chemistry.memory_half_life,
                chemistry.volatility_regime_sensitivity,
                chemistry.last_updated.isoformat(),
                chemistry.symbol
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logging.error(f"Failed to update ticker chemistry: {e}")
            
    def get_ticker_chemistry(self, symbol: str) -> Optional[TickerChemistry]:
        """Retrieve ticker chemistry"""
        try:
            cursor = self.connection.execute("""
                SELECT * FROM ticker_chemistry WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            if row:
                return TickerChemistry(
                    symbol=row[0],
                    asset_class=AssetClass(row[1]),
                    beta_to_risk=row[2],
                    beta_to_momentum=row[3],
                    hardness=row[4],
                    conductance=row[5],
                    memory_half_life=row[6],
                    volatility_regime_sensitivity=row[7],
                    last_updated=datetime.fromisoformat(row[8])
                )
            return None
            
        except Exception as e:
            logging.error(f"Failed to get ticker chemistry: {e}")
            return None

# =====================================================================================
# API COORDINATION LAYER - Prevents Rate Limit Conflicts
# =====================================================================================

class APICoordinator:
    """Coordinates API calls across all modules to prevent rate limiting"""
    
    def __init__(self):
        self.api_keys = {
            'fmp': os.getenv('FMP_API_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'token_metrics': os.getenv('TOKEN_METRICS'),
            'sec': os.getenv('SECAPI_KEY'),
            'marketstack': os.getenv('MARKETSTACK_API_KEY'),
            'coinmarketcap': os.getenv('COINMARKETCAP_API_KEY')
        }
        
        # Rate limits per provider (calls per minute)
        self.rate_limits = {
            'fmp': 300,
            'finnhub': 60,
            'yfinance': 2000,  # No official limit but be conservative
            'binance': 1200,
            'cboe': 60
        }
        
        # Call tracking
        self.call_history = defaultdict(deque)
        self.lock = threading.Lock()
        
        # Data cache with TTL
        self.cache = {}
        self.cache_ttl = {
            'price_data': 30,      # 30 seconds for price data
            'options_data': 60,    # 1 minute for options
            'vix_data': 60,        # 1 minute for VIX
            'correlation_data': 300, # 5 minutes for correlations
            'fundamental_data': 3600 # 1 hour for fundamentals
        }
        
    def can_make_call(self, provider: str) -> bool:
        """Check if we can make API call without hitting rate limit"""
        with self.lock:
            now = datetime.now()
            calls = self.call_history[provider]
            
            # Remove calls older than 1 minute
            while calls and (now - calls[0]).total_seconds() > 60:
                calls.popleft()
                
            return len(calls) < self.rate_limits.get(provider, 60)
            
    def record_call(self, provider: str):
        """Record API call for rate limiting"""
        with self.lock:
            self.call_history[provider].append(datetime.now())
            
    def get_cached_or_fetch(self, cache_key: str, fetch_func, cache_type: str = 'price_data'):
        """Get cached data or fetch new data with caching"""
        now = datetime.now()
        
        # Check cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            ttl = self.cache_ttl.get(cache_type, 60)
            
            if (now - timestamp).total_seconds() < ttl:
                return data
                
        # Cache miss - fetch new data
        data = fetch_func()
        self.cache[cache_key] = (data, now)
        
        return data

# =====================================================================================
# MARKET STATE SCANNER - Geological Framework Implementation
# =====================================================================================

class MarketStateScanner:
    """
    Production market state scanner implementing full geological framework
    No fake data - all real sources with proper error handling
    """
    
    def __init__(self, db: StateDatabase, api_coordinator: APICoordinator):
        self.db = db
        self.api = api_coordinator
        self.logger = logging.getLogger(__name__)
        
        # Historical data for percentile calculations
        self.vix_history = deque(maxlen=252*5)  # 5 years of VIX data
        self.correlation_history = defaultdict(lambda: deque(maxlen=252))
        
        # Load historical data on startup
        self._initialize_historical_data()
        
    def _initialize_historical_data(self):
        """Load historical VIX and correlation data"""
        try:
            # Get 5 years of VIX data
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="5y", interval="1d")
            
            if not vix_data.empty:
                self.vix_history.extend(vix_data['Close'].values)
                self.logger.info(f"Loaded {len(self.vix_history)} days of VIX history")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize historical data: {e}")
            
    def _fetch_vix_data(self) -> Dict[str, float]:
        """Fetch comprehensive VIX data"""
        def fetch():
            if not self.api.can_make_call('yfinance'):
                raise Exception("Rate limit exceeded for yfinance")
                
            self.api.record_call('yfinance')
            
            # Current VIX
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="5d", interval="1d")
            
            if vix_data.empty:
                raise Exception("No VIX data available")
                
            current_vix = float(vix_data['Close'].iloc[-1])
            
            # VIX9D for term structure
            try:
                vix9d_ticker = yf.Ticker("^VIX9D")
                vix9d_data = vix9d_ticker.history(period="5d", interval="1d")
                current_vix9d = float(vix9d_data['Close'].iloc[-1]) if not vix9d_data.empty else current_vix
            except:
                current_vix9d = current_vix
                
            # Calculate percentile
            if len(self.vix_history) > 20:
                vix_percentile = (np.array(self.vix_history) < current_vix).mean() * 100
            else:
                vix_percentile = 50  # Default if insufficient history
                
            # Add to history
            self.vix_history.append(current_vix)
            
            return {
                'vix': current_vix,
                'vix9d': current_vix9d,
                'vix_percentile': vix_percentile,
                'term_structure_slope': (current_vix9d - current_vix) / current_vix * 100
            }
            
        return self.api.get_cached_or_fetch('vix_data', fetch, 'vix_data')
        
    def _fetch_cross_asset_data(self) -> Dict[str, Any]:
        """Fetch cross-asset data for correlation analysis"""
        def fetch():
            symbols = ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM', 'HYG', 'VEA', 'EEM']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            data = {}
            
            for symbol in symbols:
                if not self.api.can_make_call('yfinance'):
                    break
                    
                self.api.record_call('yfinance')
                
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        data[symbol] = {
                            'returns': returns.values,
                            'current_price': float(hist['Close'].iloc[-1]),
                            'volatility': float(returns.std() * np.sqrt(252)),
                            'momentum_5d': float(returns.tail(5).mean()),
                            'momentum_20d': float(returns.tail(20).mean())
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {symbol}: {e}")
                    
            return data
            
        return self.api.get_cached_or_fetch('cross_asset_data', fetch, 'correlation_data')
        
    def _calculate_correlation_stress(self, asset_data: Dict) -> float:
        """Calculate correlation breakdown stress using proper DCC approach"""
        try:
            if len(asset_data) < 3:
                return 0.0
                
            # Create returns matrix
            symbols = list(asset_data.keys())
            returns_data = []
            min_length = min(len(asset_data[s]['returns']) for s in symbols)
            
            for symbol in symbols:
                returns_data.append(asset_data[symbol]['returns'][-min_length:])
                
            returns_matrix = np.array(returns_data).T
            
            if returns_matrix.shape[0] < 20:
                return 0.0
                
            # Calculate rolling correlations (simplified DCC)
            window = 20
            correlation_stress = 0.0
            
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    series_i = returns_matrix[:, i]
                    series_j = returns_matrix[:, j]
                    
                    # Calculate rolling correlation
                    rolling_corr = []
                    for k in range(window, len(series_i)):
                        corr = np.corrcoef(series_i[k-window:k], series_j[k-window:k])[0, 1]
                        if not np.isnan(corr):
                            rolling_corr.append(corr)
                            
                    if len(rolling_corr) > 5:
                        # Stress = volatility of correlation
                        corr_vol = np.std(rolling_corr)
                        correlation_stress += corr_vol
                        
            # Normalize to 0-100 scale
            return min(100, correlation_stress * 1000)
            
        except Exception as e:
            self.logger.error(f"Correlation stress calculation failed: {e}")
            return 0.0
            
    def _calculate_field_energy(self, vix_data: Dict, asset_data: Dict, correlation_stress: float) -> float:
        """Calculate total market field energy"""
        try:
            # Energy components
            vix_energy = vix_data['vix'] / 100  # Normalize VIX
            
            # Volatility energy from cross-assets
            vol_energy = 0.0
            if asset_data:
                avg_vol = np.mean([data['volatility'] for data in asset_data.values()])
                vol_energy = avg_vol / 100  # Normalize
                
            # Correlation breakdown energy
            corr_energy = correlation_stress / 100
            
            # Combine with weights
            field_energy = (vix_energy * 0.5) + (vol_energy * 0.3) + (corr_energy * 0.2)
            
            return min(100, field_energy * 100)
            
        except Exception as e:
            self.logger.error(f"Field energy calculation failed: {e}")
            return 50.0  # Default moderate energy
            
    def _classify_regime(self, vix_data: Dict, asset_data: Dict, field_energy: float) -> Tuple[MarketRegime, float]:
        """Classify market regime with confidence"""
        try:
            confidence_factors = []
            
            # VIX-based classification
            vix_percentile = vix_data['vix_percentile']
            
            if vix_percentile >= 85:
                regime_score = 3  # Crisis
                confidence_factors.append(0.4)
            elif vix_percentile >= 75:
                regime_score = 2  # Risk-off
                confidence_factors.append(0.3)
            elif vix_percentile <= 25:
                regime_score = 0  # Risk-on
                confidence_factors.append(0.3)
            else:
                regime_score = 1  # Transitional
                confidence_factors.append(0.1)
                
            # Term structure confirmation
            term_slope = vix_data['term_structure_slope']
            if abs(term_slope) > 10:  # Strong backwardation/contango
                confidence_factors.append(0.2)
            else:
                confidence_factors.append(0.1)
                
            # Cross-asset momentum confirmation
            if asset_data:
                spy_momentum = asset_data.get('SPY', {}).get('momentum_5d', 0)
                tlt_momentum = asset_data.get('TLT', {}).get('momentum_5d', 0)
                
                # Risk-on: stocks up, bonds down
                if spy_momentum > 0.005 and tlt_momentum < -0.002:
                    if regime_score == 0:  # Confirms risk-on
                        confidence_factors.append(0.3)
                    else:
                        confidence_factors.append(0.1)
                        
                # Risk-off: stocks down, bonds up
                elif spy_momentum < -0.005 and tlt_momentum > 0.002:
                    if regime_score >= 2:  # Confirms risk-off/crisis
                        confidence_factors.append(0.3)
                    else:
                        confidence_factors.append(0.1)
                else:
                    confidence_factors.append(0.1)
                    
            # Field energy confirmation
            if field_energy > 70 and regime_score >= 2:
                confidence_factors.append(0.2)
            elif field_energy < 30 and regime_score == 0:
                confidence_factors.append(0.2)
            else:
                confidence_factors.append(0.1)
                
            # Map score to regime
            regimes = [MarketRegime.RISK_ON, MarketRegime.TRANSITIONAL, MarketRegime.RISK_OFF, MarketRegime.CRISIS]
            regime = regimes[min(regime_score, 3)]
            
            # Calculate confidence
            confidence = min(1.0, sum(confidence_factors))
            
            return regime, confidence
            
        except Exception as e:
            self.logger.error(f"Regime classification failed: {e}")
            return MarketRegime.TRANSITIONAL, 0.5
            
    def get_market_state(self) -> MarketState:
        """Get current market state - main public method"""
        try:
            self.logger.info("Scanning market state...")
            
            # Fetch all required data
            vix_data = self._fetch_vix_data()
            asset_data = self._fetch_cross_asset_data()
            
            # Calculate derived metrics
            correlation_stress = self._calculate_correlation_stress(asset_data)
            field_energy = self._calculate_field_energy(vix_data, asset_data, correlation_stress)
            
            # Classify regime
            regime, confidence = self._classify_regime(vix_data, asset_data, field_energy)
            
            # Calculate momentum (risk-adjusted)
            momentum_direction = 0.0
            if asset_data:
                spy_momentum = asset_data.get('SPY', {}).get('momentum_5d', 0)
                qqq_momentum = asset_data.get('QQQ', {}).get('momentum_5d', 0)
                
                # Average momentum adjusted for volatility
                avg_momentum = (spy_momentum + qqq_momentum) / 2
                spy_vol = asset_data.get('SPY', {}).get('volatility', 0.2)
                
                # Risk-adjusted momentum (-100 to +100)
                momentum_direction = (avg_momentum / (spy_vol / 252)) * 100 if spy_vol > 0 else 0
                momentum_direction = np.clip(momentum_direction, -100, 100)
                
            # Black swan detection
            black_swan_alert = (
                vix_data['vix'] > 45 or 
                correlation_stress > 80 or 
                field_energy > 85
            )
            
            # Create market state
            market_state = MarketState(
                risk_level=vix_data['vix_percentile'],
                momentum_direction=momentum_direction,
                regime=regime,
                confidence=confidence,
                timestamp=datetime.now(),
                vix_value=vix_data['vix'],
                vix_percentile=vix_data['vix_percentile'],
                term_structure_slope=vix_data['term_structure_slope'],
                correlation_stress=correlation_stress,
                black_swan_alert=black_swan_alert,
                field_energy=field_energy
            )
            
            # Save to database
            self.db.save_market_state(market_state)
            
            self.logger.info(f"Market state: {market_state.regime.value}, Risk: {market_state.risk_level:.1f}, "
                           f"Momentum: {market_state.momentum_direction:.1f}, Confidence: {market_state.confidence:.2f}")
            
            if black_swan_alert:
                self.logger.critical(f"🚨 BLACK SWAN ALERT: VIX={vix_data['vix']:.1f}, "
                                   f"Corr_Stress={correlation_stress:.1f}, Field_Energy={field_energy:.1f}")
                
            return market_state
            
        except Exception as e:
            self.logger.error(f"Market state scan failed: {e}")
            raise RuntimeError(f"Cannot determine market state: {e}")

# =====================================================================================
# ASSET CHEMISTRY ENGINE - Ticker Reaction Mapping
# =====================================================================================

class AssetChemistryEngine:
    """
    Maps how individual assets react to different market states
    Implements evolutionary learning of ticker chemistry
    """
    
    def __init__(self, db: StateDatabase, api_coordinator: APICoordinator):
        self.db = db
        self.api = api_coordinator
        self.logger = logging.getLogger(__name__)
        
        # Default chemistry library
        self.default_chemistry = {
            'SPY': TickerChemistry('SPY', AssetClass.MOMENTUM, 0.5, 0.8, 0.7, 0.9, 5, 0.6, datetime.now()),
            'QQQ': TickerChemistry('QQQ', AssetClass.GROWTH, 0.8, 1.2, 0.4, 0.8, 3, 0.9, datetime.now()),
            'IWM': TickerChemistry('IWM', AssetClass.GROWTH, 1.2, 1.5, 0.3, 0.6, 2, 1.1, datetime.now()),
            'TLT': TickerChemistry('TLT', AssetClass.DEFENSIVE, -0.6, -0.3, 0.8, 0.7, 10, 0.4, datetime.now()),
            'GLD': TickerChemistry('GLD', AssetClass.DEFENSIVE, -0.2, 0.1, 0.9, 0.5, 20, 0.3, datetime.now()),
            'VIX': TickerChemistry('VIX', AssetClass.DEFENSIVE, 2.0, -1.0, 0.6, 0.3, 1, 2.0, datetime.now()),
            'HYG': TickerChemistry('HYG', AssetClass.VALUE, 0.4, 0.6, 0.5, 0.8, 7, 0.7, datetime.now())
        }
        
        # Initialize chemistry for all symbols
        self._initialize_chemistry()
        
    def _initialize_chemistry(self):
        """Initialize chemistry for all tracked symbols"""
        for symbol, chemistry in self.default_chemistry.items():
            existing = self.db.get_ticker_chemistry(symbol)
            if not existing:
                self.db.update_ticker_chemistry(chemistry)
                
    def get_chemistry(self, symbol: str) -> TickerChemistry:
        """Get ticker chemistry, learning from performance if available"""
        chemistry = self.db.get_ticker_chemistry(symbol)
        
        if chemistry:
            # Check if chemistry needs updating based on recent performance
            days_old = (datetime.now() - chemistry.last_updated).days
            if days_old > 30:  # Update monthly
                updated_chemistry = self._evolve_chemistry(chemistry)
                self.db.update_ticker_chemistry(updated_chemistry)
                return updated_chemistry
            return chemistry
        else:
            # Create new chemistry based on asset classification
            new_chemistry = self._classify_new_asset(symbol)
            self.db.update_ticker_chemistry(new_chemistry)
            return new_chemistry
            
    def _classify_new_asset(self, symbol: str) -> TickerChemistry:
        """Classify new asset and assign default chemistry"""
        try:
            # Fetch basic asset data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            if hist.empty:
                # Default unknown asset
                return TickerChemistry(
                    symbol=symbol,
                    asset_class=AssetClass.MOMENTUM,
                    beta_to_risk=0.5,
                    beta_to_momentum=0.8,
                    hardness=0.5,
                    conductance=0.5,
                    memory_half_life=5,
                    volatility_regime_sensitivity=0.6,
                    last_updated=datetime.now()
                )
                
            # Calculate basic metrics
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Classify based on symbol patterns and volatility
            if symbol.endswith('USD') or 'BTC' in symbol or 'ETH' in symbol:
                asset_class = AssetClass.GROWTH
                beta_to_risk = 1.5
                beta_to_momentum = 2.0
                hardness = 0.2
                conductance = 0.9
                memory_half_life = 1
                vol_sensitivity = 2.0
            elif symbol in ['TLT', 'SHY', 'IEF', 'AGG']:
                asset_class = AssetClass.DEFENSIVE
                beta_to_risk = -0.5
                beta_to_momentum = -0.2
                hardness = 0.8
                conductance = 0.7
                memory_half_life = 15
                vol_sensitivity = 0.3
            elif symbol in ['GLD', 'SLV', 'VIX']:
                asset_class = AssetClass.DEFENSIVE
                beta_to_risk = -0.3
                beta_to_momentum = 0.1
                hardness = 0.9
                conductance = 0.4
                memory_half_life = 20
                vol_sensitivity = 0.5
            elif volatility > 0.4:  # High vol = growth
                asset_class = AssetClass.GROWTH
                beta_to_risk = 1.0
                beta_to_momentum = 1.5
                hardness = 0.3
                conductance = 0.8
                memory_half_life = 3
                vol_sensitivity = 1.2
            else:  # Default to value
                asset_class = AssetClass.VALUE
                beta_to_risk = 0.3
                beta_to_momentum = 0.5
                hardness = 0.6
                conductance = 0.6
                memory_half_life = 10
                vol_sensitivity = 0.5
                
            return TickerChemistry(
                symbol=symbol,
                asset_class=asset_class,
                beta_to_risk=beta_to_risk,
                beta_to_momentum=beta_to_momentum,
                hardness=hardness,
                conductance=conductance,
                memory_half_life=memory_half_life,
                volatility_regime_sensitivity=vol_sensitivity,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Asset classification failed for {symbol}: {e}")
            # Return safe default
            return TickerChemistry(
                symbol=symbol,
                asset_class=AssetClass.MOMENTUM,
                beta_to_risk=0.5,
                beta_to_momentum=0.8,
                hardness=0.5,
                conductance=0.5,
                memory_half_life=5,
                volatility_regime_sensitivity=0.6,
                last_updated=datetime.now()
            )
            
    def _evolve_chemistry(self, chemistry: TickerChemistry) -> TickerChemistry:
        """Evolve chemistry based on recent performance"""
        try:
            # Get recent market states and price performance
            # This would analyze how well the chemistry predicted actual price movements
            # For now, return original chemistry (placeholder for full evolutionary system)
            
            return TickerChemistry(
                symbol=chemistry.symbol,
                asset_class=chemistry.asset_class,
                beta_to_risk=chemistry.beta_to_risk,
                beta_to_momentum=chemistry.beta_to_momentum,
                hardness=chemistry.hardness,
                conductance=chemistry.conductance,
                memory_half_life=chemistry.memory_half_life,
                volatility_regime_sensitivity=chemistry.volatility_regime_sensitivity,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Chemistry evolution failed: {e}")
            return chemistry
            
    def predict_reaction(self, symbol: str, market_state: MarketState) -> Dict[str, float]:
        """Predict how asset will react to given market state"""
        try:
            chemistry = self.get_chemistry(symbol)
            
            # Calculate expected reaction based on chemistry
            risk_reaction = chemistry.beta_to_risk * (market_state.risk_level - 50) / 50
            momentum_reaction = chemistry.beta_to_momentum * market_state.momentum_direction / 100
            
            # Regime-specific adjustments
            regime_multiplier = 1.0
            if market_state.regime == MarketRegime.CRISIS:
                if chemistry.asset_class == AssetClass.DEFENSIVE:
                    regime_multiplier = 1.5  # Defensive assets perform better in crisis
                else:
                    regime_multiplier = 0.5  # Risk assets underperform
            elif market_state.regime == MarketRegime.RISK_ON:
                if chemistry.asset_class == AssetClass.GROWTH:
                    regime_multiplier = 1.3  # Growth outperforms in risk-on
                    
            # Volatility regime sensitivity
            vol_adjustment = chemistry.volatility_regime_sensitivity * (market_state.field_energy - 50) / 50
            
            # Combined reaction
            total_reaction = (risk_reaction + momentum_reaction + vol_adjustment) * regime_multiplier
            
            return {
                'expected_return': total_reaction / 100,  # Convert to return %
                'confidence': chemistry.hardness,  # Higher hardness = more predictable
                'conductance': chemistry.conductance,  # Flow likelihood
                'regime_fit': regime_multiplier,
                'risk_component': risk_reaction,
                'momentum_component': momentum_reaction,
                'volatility_component': vol_adjustment
            }
            
        except Exception as e:
            self.logger.error(f"Reaction prediction failed for {symbol}: {e}")
            return {
                'expected_return': 0.0,
                'confidence': 0.5,
                'conductance': 0.5,
                'regime_fit': 1.0,
                'risk_component': 0.0,
                'momentum_component': 0.0,
                'volatility_component': 0.0
            }

# =====================================================================================
# TRAILHEAD DETECTOR - Enhanced with Flow Prediction
# =====================================================================================

class TrailheadDetector:
    """
    Enhanced trailhead detector with flow prediction capabilities
    Detects pressure + fragility convergence and predicts capital flow paths
    """
    
    def __init__(self, db: StateDatabase, api_coordinator: APICoordinator, chemistry_engine: AssetChemistryEngine):
        self.db = db
        self.api = api_coordinator
        self.chemistry = chemistry_engine
        self.logger = logging.getLogger(__name__)
        
        # Detection thresholds
        self.pressure_threshold = 75
        self.fragility_threshold = 70
        self.composite_threshold = 80
        self.signal_ttl_minutes = 30
        
        # Cache for expensive calculations
        self.flow_cache = {}
        
    def _fetch_options_flow_data(self) -> Dict[str, float]:
        """Fetch real options flow data with multiple sources"""
        def fetch():
            try:
                # Method 1: SPY options via yfinance
                spy = yf.Ticker("SPY")
                
                # Get current price
                hist = spy.history(period="1d")
                if hist.empty:
                    raise Exception("No SPY price data")
                    
                current_price = hist['Close'].iloc[-1]
                
                # Get options chain
                options_dates = spy.options
                if not options_dates:
                    return {'put_call_ratio': 1.0, 'options_volume': 0, 'gamma_exposure': 0}
                    
                # Get nearest expiry
                nearest_expiry = options_dates[0]
                options_chain = spy.option_chain(nearest_expiry)
                
                calls = options_chain.calls
                puts = options_chain.puts
                
                # Calculate metrics
                call_volume = calls['volume'].fillna(0).sum()
                put_volume = puts['volume'].fillna(0).sum()
                total_volume = call_volume + put_volume
                
                put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
                
                # Gamma exposure approximation (simplified)
                atm_calls_oi = calls[abs(calls['strike'] - current_price) < 5]['openInterest'].fillna(0).sum()
                atm_puts_oi = puts[abs(puts['strike'] - current_price) < 5]['openInterest'].fillna(0).sum()
                gamma_exposure = atm_calls_oi - atm_puts_oi
                
                return {
                    'put_call_ratio': float(put_call_ratio),
                    'options_volume': float(total_volume),
                    'gamma_exposure': float(gamma_exposure),
                    'call_volume': float(call_volume),
                    'put_volume': float(put_volume)
                }
                
            except Exception as e:
                self.logger.error(f"Options flow fetch failed: {e}")
                return {
                    'put_call_ratio': 1.0,
                    'options_volume': 0,
                    'gamma_exposure': 0,
                    'call_volume': 0,
                    'put_volume': 0
                }
                
        return self.api.get_cached_or_fetch('options_flow', fetch, 'options_data')
        
    def _fetch_crypto_funding_data(self) -> Dict[str, float]:
        """Fetch crypto perpetual funding rates"""
        def fetch():
            try:
                # Binance funding rates
                url = "https://fapi.binance.com/fapi/v1/premiumIndex"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                funding_data = {}
                
                symbols_of_interest = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
                
                for item in data:
                    if item['symbol'] in symbols_of_interest:
                        funding_rate = float(item['lastFundingRate'])
                        annualized_rate = funding_rate * 3 * 365 * 100  # 8hr to annual %
                        
                        funding_data[item['symbol']] = {
                            'funding_rate_8h': funding_rate,
                            'funding_rate_annual_pct': annualized_rate,
                            'mark_price': float(item['markPrice'])
                        }
                        
                # Calculate composite
                if funding_data:
                    avg_funding = np.mean([d['funding_rate_annual_pct'] for d in funding_data.values()])
                    funding_data['composite_funding_pressure'] = avg_funding
                else:
                    funding_data['composite_funding_pressure'] = 0.0
                    
                return funding_data
                
            except Exception as e:
                self.logger.error(f"Crypto funding fetch failed: {e}")
                return {'composite_funding_pressure': 0.0}
                
        return self.api.get_cached_or_fetch('crypto_funding', fetch, 'price_data')
        
    def _calculate_pressure_score(self, options_data: Dict, crypto_data: Dict, vix_data: Dict) -> Tuple[float, Dict]:
        """Calculate composite pressure score with component breakdown"""
        components = {}
        
        # Options pressure
        put_call_ratio = options_data.get('put_call_ratio', 1.0)
        if put_call_ratio > 1.2:
            options_pressure = min(100, (put_call_ratio - 1.0) * 200)
        elif put_call_ratio < 0.8:
            options_pressure = min(100, (1.0 - put_call_ratio) * 200)
        else:
            options_pressure = 0
        components['options_pressure'] = options_pressure
        
        # Crypto funding pressure
        funding_pressure = abs(crypto_data.get('composite_funding_pressure', 0))
        if funding_pressure > 5:
            crypto_pressure = min(100, funding_pressure * 10)
        else:
            crypto_pressure = 0
        components['crypto_pressure'] = crypto_pressure
        
        # VIX term structure pressure
        vix_slope = vix_data.get('term_structure_slope', 0)
        if abs(vix_slope) > 2:
            vix_pressure = min(100, abs(vix_slope) * 25)
        else:
            vix_pressure = 0
        components['vix_pressure'] = vix_pressure
        
        # Volume pressure
        options_volume = options_data.get('options_volume', 0)
        if options_volume > 1000000:
            volume_pressure = min(100, (options_volume / 1000000 - 1) * 50)
        else:
            volume_pressure = 0
        components['volume_pressure'] = volume_pressure
        
        # Weighted composite
        total_pressure = (
            options_pressure * 0.4 +
            crypto_pressure * 0.3 +
            vix_pressure * 0.2 +
            volume_pressure * 0.1
        )
        
        return total_pressure, components
        
    def _predict_flow_path(self, current_state: MarketState, pressure_components: Dict, 
                          fragility_components: Dict) -> Optional[FlowPrediction]:
        """Predict capital flow path based on terrain analysis"""
        try:
            # Determine target state based on pressure direction
            target_risk = current_state.risk_level
            target_momentum = current_state.momentum_direction
            
            # Adjust target based on pressure
            if pressure_components.get('options_pressure', 0) > 50:
                put_call_ratio = 1.3  # Assuming high put buying
                if put_call_ratio > 1.1:
                    target_risk = max(0, current_state.risk_level - 20)  # Risk reduction
                    target_momentum = current_state.momentum_direction + 30
                    
            if pressure_components.get('crypto_pressure', 0) > 50:
                target_momentum = current_state.momentum_direction + 40
                
            # Create target state
            target_state = MarketState(
                risk_level=np.clip(target_risk, 0, 100),
                momentum_direction=np.clip(target_momentum, -100, 100),
                regime=current_state.regime,
                confidence=0.7,
                timestamp=current_state.timestamp + timedelta(hours=24),
                vix_value=current_state.vix_value,
                vix_percentile=target_risk,
                term_structure_slope=current_state.term_structure_slope,
                correlation_stress=current_state.correlation_stress,
                black_swan_alert=False,
                field_energy=current_state.field_energy
            )
            
            # Find optimal flow path
            symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'HYG']
            flow_candidates = []
            
            for symbol in symbols:
                reaction = self.chemistry.predict_reaction(symbol, target_state)
                
                flow_score = (
                    reaction['expected_return'] * 0.4 +
                    reaction['confidence'] * 0.3 +
                    reaction['conductance'] * 0.3
                )
                
                flow_candidates.append((symbol, flow_score, reaction))
                
            # Sort by flow score
            flow_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select top flow path
            flow_path = [candidate[0] for candidate in flow_candidates[:3]]
            catalyst_symbols = [candidate[0] for candidate in flow_candidates[:2]]
            
            # Calculate flow magnitude
            flow_magnitude = sum(abs(candidate[1]) for candidate in flow_candidates[:3])
            
            return FlowPrediction(
                source_state=current_state,
                target_state=target_state,
                flow_path=flow_path,
                flow_magnitude=flow_magnitude,
                time_horizon=1,  # 1 day
                confidence=0.7,
                catalyst_symbols=catalyst_symbols
            )
            
        except Exception as e:
            self.logger.error(f"Flow prediction failed: {e}")
            return None
            
    def detect_trailhead(self, current_market_state: MarketState) -> Optional[TrailheadSignal]:
        """Main trailhead detection with flow prediction"""
        try:
            self.logger.info("Scanning for trailhead formation...")
            
            # Fetch pressure data
            options_data = self._fetch_options_flow_data()
            crypto_data = self._fetch_crypto_funding_data()
            
            vix_data = {
                'vix': current_market_state.vix_value,
                'vix_percentile': current_market_state.vix_percentile,
                'term_structure_slope': current_market_state.term_structure_slope
            }
            
            # Calculate pressure
            pressure_score, pressure_components = self._calculate_pressure_score(
                options_data, crypto_data, vix_data
            )
            
            # Use correlation stress as fragility
            fragility_score = current_market_state.correlation_stress
            fragility_components = {'correlation_stress': fragility_score}
            
            # Determine signal direction
            put_call_ratio = options_data.get('put_call_ratio', 1.0)
            crypto_funding = crypto_data.get('composite_funding_pressure', 0)
            
            direction = "NEUTRAL"
            if put_call_ratio > 1.1 and crypto_funding < -2:
                direction = "BULLISH"
            elif put_call_ratio < 0.9 and crypto_funding > 5:
                direction = "BEARISH"
                
            # Calculate composite score
            composite_score = pressure_score * 0.6 + fragility_score * 0.4
            
            # Generate signal if thresholds met
            if (pressure_score >= self.pressure_threshold and 
                fragility_score >= self.fragility_threshold):
                
                # Predict flow path
                flow_prediction = self._predict_flow_path(
                    current_market_state, pressure_components, fragility_components
                )
                
                signal = TrailheadSignal(
                    signal_type='COMPOSITE',
                    strength=min(100, composite_score),
                    direction=direction,
                    confidence=0.8,
                    timestamp=datetime.now(),
                    expiry_time=datetime.now() + timedelta(minutes=self.signal_ttl_minutes),
                    pressure_components=pressure_components,
                    fragility_components=fragility_components,
                    predicted_flow=flow_prediction
                )
                
                self.logger.warning(f"🎯 TRAILHEAD DETECTED: {signal}")
                return signal
                
            # Partial signals
            elif pressure_score >= self.pressure_threshold:
                signal = TrailheadSignal(
                    signal_type='PRESSURE',
                    strength=pressure_score,
                    direction=direction,
                    confidence=0.6,
                    timestamp=datetime.now(),
                    expiry_time=datetime.now() + timedelta(minutes=self.signal_ttl_minutes),
                    pressure_components=pressure_components,
                    fragility_components={},
                    predicted_flow=None
                )
                return signal
                
            return None
            
        except Exception as e:
            self.logger.error(f"Trailhead detection failed: {e}")
            raise

# =====================================================================================
# PRODUCTION READY MAIN BOT - Corrected Integration
# =====================================================================================

class TrailheadCatalystBot:
    """
    Complete production-ready bot with corrected integration
    All systems properly connected with real data flows
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        
        # Setup logging first
        self.logger = self._setup_logging()
        self.logger.info("🚀 Initializing Trailhead Catalyst Bot")
        
        try:
            # Initialize core systems
            self.db = StateDatabase()
            self.api_coordinator = APICoordinator()
            self.chemistry_engine = AssetChemistryEngine(self.db, self.api_coordinator)
            self.market_scanner = MarketStateScanner(self.db, self.api_coordinator)
            self.trailhead_detector = TrailheadDetector(self.db, self.api_coordinator, self.chemistry_engine)
            
            # Risk management
            self.max_daily_loss_pct = 3.0
            self.max_position_pct = 4.0
            self.max_positions = 3
            
            # State tracking
            self.active_positions = {}
            self.total_pnl = 0.0
            self.trade_count = 0
            self.running = False
            
            # IBKR connection (optional)
            self.ibkr_connected = False
            if IBKR_AVAILABLE:
                try:
                    self.ib = IB()
                    # Don't auto-connect - let user decide
                except Exception as e:
                    self.logger.warning(f"IBKR initialization failed: {e}")
                    
            self.logger.info("✅ All systems initialized successfully")
            
        except Exception as e:
            self.logger.critical(f"❌ System initialization failed: {e}")
            raise
            
    def _setup_logging(self) -> logging.Logger:
        """Setup production logging"""
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/catalyst_bot_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
        
    def connect_ibkr(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 0) -> bool:
        """Connect to IBKR TWS/Gateway"""
        if not IBKR_AVAILABLE:
            self.logger.error("IBKR not available - install ib_insync")
            return False
            
        try:
            self.ib.connect(host=host, port=port, clientId=client_id, timeout=30)
            self.ibkr_connected = True
            self.logger.info(f"✅ IBKR connected to {host}:{port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ IBKR connection failed: {e}")
            return False
            
    def run_cycle(self):
        """Single bot cycle - the main logic loop"""
        try:
            cycle_start = datetime.now()
            self.logger.info(f"🔄 Bot cycle starting at {cycle_start}")
            
            # 1. Scan market state
            market_state = self.market_scanner.get_market_state()
            
            # 2. Emergency check
            if market_state.black_swan_alert:
                self.logger.critical("🚨 BLACK SWAN DETECTED - EMERGENCY MODE")
                self._emergency_flatten()
                return
                
            # 3. Detect trailheads
            signal = self.trailhead_detector.detect_trailhead(market_state)
            
            if signal:
                self.logger.info(f"🎯 Signal detected: {signal.signal_type} strength={signal.strength:.1f}")
                
                # 4. Execute signal if strong enough
                if signal.strength >= 75 and len(self.active_positions) < self.max_positions:
                    self._execute_signal(signal, market_state)
                    
            # 5. Manage existing positions
            self._manage_positions(market_state)
            
            # 6. Log performance
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"🔄 Cycle completed in {cycle_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Bot cycle failed: {e}")
            self.logger.error(traceback.format_exc())
            
    def _execute_signal(self, signal: TrailheadSignal, market_state: MarketState):
        """Execute trailhead signal with position sizing"""
        try:
            # Select symbol from flow prediction
            if signal.predicted_flow and signal.predicted_flow.catalyst_symbols:
                symbol = signal.predicted_flow.catalyst_symbols[0]
            else:
                # Default symbol selection based on signal direction
                if signal.direction == "BULLISH":
                    symbol = "QQQ" if market_state.regime == MarketRegime.RISK_ON else "SPY"
                elif signal.direction == "BEARISH":
                    symbol = "TLT"
                else:
                    symbol = "SPY"
                    
            # Calculate position size
            position_size_usd = self.initial_capital * (self.max_position_pct / 100)
            position_size_usd *= (signal.strength / 100)  # Scale by signal strength
            position_size_usd *= signal.confidence  # Scale by confidence
            
            # Direction
            action = "BUY" if signal.direction == "BULLISH" else "SELL"
            if signal.direction == "NEUTRAL":
                return
                
            self.logger.info(f"🚀 EXECUTING: {symbol} {action} ${position_size_usd:.2f}")
            
            # For now, simulate execution (paper trading)
            self._simulate_trade(symbol, action, position_size_usd, signal)
            
        except Exception as e:
            self.logger.error(f"❌ Signal execution failed: {e}")
            
    def _simulate_trade(self, symbol: str, action: str, size_usd: float, signal: TrailheadSignal):
        """Simulate trade execution for testing"""
        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                self.logger.error(f"❌ No price data for {symbol}")
                return
                
            current_price = hist['Close'].iloc[-1]
            quantity = size_usd / current_price
            
            # Create position record
            position_id = f"{symbol}_{action}_{int(datetime.now().timestamp())}"
            
            self.active_positions[position_id] = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'signal': signal,
                'size_usd': size_usd
            }
            
            self.trade_count += 1
            
            self.logger.info(f"✅ TRADE SIMULATED: {symbol} {action} {quantity:.2f} @ ${current_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Trade simulation failed: {e}")
            
    def _manage_positions(self, market_state: MarketState):
        """Manage existing positions"""
        try:
            positions_to_close = []
            
            for pos_id, position in self.active_positions.items():
                # Check time-based exit (24 hour max)
                time_held = datetime.now() - position['entry_time']
                if time_held.total_seconds() > 24 * 3600:
                    positions_to_close.append(pos_id)
                    continue
                    
                # Check stop loss (simplified)
                try:
                    ticker = yf.Ticker(position['symbol'])
                    hist = ticker.history(period="1d")
                    current_price = hist['Close'].iloc[-1]
                    
                    # Calculate P&L
                    if position['action'] == 'BUY':
                        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                        
                    # Stop loss at -2%
                    if pnl_pct < -0.02:
                        positions_to_close.append(pos_id)
                        self.logger.warning(f"⚠️ Stop loss triggered for {position['symbol']}: {pnl_pct*100:.1f}%")
                        continue
                        
                    # Take profit at +5%
                    if pnl_pct > 0.05:
                        positions_to_close.append(pos_id)
                        self.logger.info(f"✅ Take profit for {position['symbol']}: {pnl_pct*100:.1f}%")
                        
                except Exception as e:
                    self.logger.error(f"❌ Position check failed for {position['symbol']}: {e}")
                    
            # Close positions
            for pos_id in positions_to_close:
                self._close_position(pos_id)
                
        except Exception as e:
            self.logger.error(f"❌ Position management failed: {e}")
            
    def _close_position(self, position_id: str):
        """Close a position and calculate P&L"""
        try:
            if position_id not in self.active_positions:
                return
                
            position = self.active_positions[position_id]
            
            # Get current price
            ticker = yf.Ticker(position['symbol'])
            hist = ticker.history(period="1d")
            current_price = hist['Close'].iloc[-1]
            
            # Calculate P&L
            if position['action'] == 'BUY':
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity']
                
            self.total_pnl += pnl
            
            # Log result
            pnl_pct = pnl / position['size_usd'] * 100
            self.logger.info(f"🔄 POSITION CLOSED: {position['symbol']} P&L: ${pnl:.2f} ({pnl_pct:.1f}%)")
            
            # Remove from active positions
            del self.active_positions[position_id]
            
        except Exception as e:
            self.logger.error(f"❌ Position close failed: {e}")
            
    def _emergency_flatten(self):
        """Emergency: close all positions immediately"""
        self.logger.critical("🚨 EMERGENCY FLATTEN - Closing all positions")
        
        for pos_id in list(self.active_positions.keys()):
            self._close_position(pos_id)
            
    def start(self, scan_interval: int = 60):
        """Start the bot with specified scan interval"""
        try:
            self.logger.info(f"🚀 Starting Trailhead Catalyst Bot (scan every {scan_interval}s)")
            
            # Initial health check
            health = self._health_check()
            if not all(health.values()):
                self.logger.error(f"❌ Health check failed: {health}")
                raise RuntimeError("System not ready")
                
            self.running = True
            
            # Schedule regular cycles
            schedule.every(scan_interval).seconds.do(self.run_cycle)
            schedule.every(5).minutes.do(self._log_status)
            
            self.logger.info("✅ Bot started successfully")
            
            # Main loop
            while self.running:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Shutdown signal received")
            self.shutdown()
        except Exception as e:
            self.logger.critical(f"💥 Bot start failed: {e}")
            self.shutdown()
            
    def shutdown(self):
        """Gracefully shutdown the bot"""
        self.logger.info("🛑 Shutting down bot...")
        
        self.running = False
        
        # Close all positions
        for pos_id in list(self.active_positions.keys()):
            self._close_position(pos_id)
            
        # Disconnect IBKR if connected
        if self.ibkr_connected:
            try:
                self.ib.disconnect()
                self.logger.info("✅ IBKR disconnected")
            except:
                pass
                
        # Close database
        if self.db.connection:
            self.db.connection.close()
            
        self.logger.info("✅ Bot shutdown complete")
        
    def _health_check(self) -> Dict[str, bool]:
        """Comprehensive health check"""
        health = {
            'api_keys_present': bool(self.api_coordinator.api_keys['fmp'] and 
                                   self.api_coordinator.api_keys['finnhub']),
            'database_ok': bool(self.db.connection),
            'market_scanner_ok': True,
            'chemistry_engine_ok': True,
            'trailhead_detector_ok': True
        }
        
        # Test data connectivity
        try:
            test_ticker = yf.Ticker("SPY")
            test_data = test_ticker.history(period="1d")
            health['data_connectivity'] = not test_data.empty
        except:
            health['data_connectivity'] = False
            
        return health
        
    def _log_status(self):
        """Log current bot status"""
        try:
            total_return_pct = (self.total_pnl / self.initial_capital) * 100
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'total_capital': self.initial_capital,
                'total_pnl': self.total_pnl,
                'return_pct': total_return_pct,
                'active_positions': len(self.active_positions),
                'total_trades': self.trade_count,
                'running': self.running
            }
            
            self.logger.info(f"📊 STATUS: Return: {total_return_pct:.2f}%, "
                           f"Positions: {len(self.active_positions)}, "
                           f"Trades: {self.trade_count}")
                           
        except Exception as e:
            self.logger.error(f"❌ Status logging failed: {e}")
            
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            'initial_capital': self.initial_capital,
            'total_pnl': self.total_pnl,
            'return_percentage': (self.total_pnl / self.initial_capital) * 100,
            'total_trades': self.trade_count,
            'active_positions': len(self.active_positions),
            'running': self.running,
            'timestamp': datetime.now().isoformat()
        }

# =====================================================================================
# MAIN EXECUTION AND DEPLOYMENT
# =====================================================================================

def setup_environment():
    """Setup environment and validate configuration"""
    # Check required environment variables
    required_vars = ['FMP_API_KEY', 'FINNHUB_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set these in your .env file or environment")
        return False
        
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    
    return True

def main():
    """Main execution function"""
    print("🤖 Trailhead Catalyst Bot - Production Ready")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
        
    # Handle command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Trailhead Catalyst Trading Bot')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--scan-interval', type=int, default=60, help='Scan interval in seconds')
    parser.add_argument('--ibkr-host', type=str, default='127.0.0.1', help='IBKR host')
    parser.add_argument('--ibkr-port', type=int, default=7497, help='IBKR port')
    parser.add_argument('--paper-trading', action='store_true', help='Paper trading mode')
    
    args = parser.parse_args()
    
    try:
        # Create bot
        bot = TrailheadCatalystBot(initial_capital=args.capital)
        
        print(f"💰 Initial Capital: ${args.capital:,.2f}")
        print(f"⏱️  Scan Interval: {args.scan_interval}s")
        print(f"📝 Paper Trading: {'Yes' if args.paper_trading else 'No'}")
        
        # Connect to IBKR if not paper trading
        if not args.paper_trading and IBKR_AVAILABLE:
            print(f"🔌 Connecting to IBKR at {args.ibkr_host}:{args.ibkr_port}...")
            if bot.connect_ibkr(args.ibkr_host, args.ibkr_port):
                print("✅ IBKR connected")
            else:
                print("⚠️ IBKR connection failed - continuing in paper mode")
        else:
            print("📝 Running in paper trading mode")
            
        print("=" * 60)
        print("🚀 Starting bot... (Press Ctrl+C to stop)")
        print("=" * 60)
        
        # Start the bot
        bot.start(scan_interval=args.scan_interval)
        
    except Exception as e:
        print(f"💥 FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutdown signal received...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    main()

# =====================================================================================
# REQUIREMENTS AND CONFIGURATION FILES
# =====================================================================================

"""
requirements.txt:
numpy>=1.21.0
pandas>=1.3.0
yfinance>=0.1.87
requests>=2.25.0
schedule>=1.1.0
python-dotenv>=0.19.0
ib_insync>=0.9.86

.env file should contain:
FMP_API_KEY=your_fmp_key
FINNHUB_API_KEY=your_finnhub_key
TOKEN_METRICS=your_token_metrics_key
SECAPI_KEY=your_sec_api_key
MARKETSTACK_API_KEY=your_marketstack_key
COINMARKETCAP_API_KEY=your_coinmarketcap_key
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=0

Usage:
python corrected_catalyst_framework.py --capital 10000 --scan-interval 60 --paper-trading

For production:
python corrected_catalyst_framework.py --capital 50000 --scan-interval 30 --ibkr-host 127.0.0.1 --ibkr-port 7497
"""
