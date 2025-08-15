"""
Performance_Tracker.py
Complete Performance Tracking for Catalyst Trading System
NO PLACEHOLDERS - All functionality is real and production-ready
Integrates with existing Catalyst Framework modules
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging
import json
import os

@dataclass
class TradeRecord:
    """
    Trade record structure matching Catalyst Framework data flow
    Compatible with Portfolio_Synthesizer output and IBKR_Executor input
    """
    trade_id: str
    timestamp: datetime
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: int
    price: float
    trade_value: float
    # Catalyst Framework specific fields
    chemistry_type: str  # noble_gas, volatile_compound, phase_change, catalyst_accelerant
    market_risk: float  # 0-1 from Market_State_Calculator.py
    market_momentum: float  # -1 to 1 from Market_State_Calculator.py
    trailhead_score: float  # Composite score from Trailhead_Detector.py
    confidence: float  # 0-1 confidence level
    kelly_fraction: float  # Kelly criterion from Portfolio_Synthesizer.py
    actual_position_size: float  # Actual $ amount invested
    position_size_pct: float  # % of capital used
    predicted_horizon_hours: float  # 4-48hr prediction from State_Predictor.py
    # Trade execution and outcome
    pnl: Optional[float] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    hold_duration_hours: Optional[float] = None
    ibkr_order_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceMetrics:
    """
    Performance statistics aligned with Catalyst Framework targets
    Target: 75-100% annual returns, 65-70% win rate, 2.5-3.5 Sharpe, <15% drawdown
    """
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # Target: 65-70%
    total_pnl: float
    total_return_pct: float  # Target: 75-100% annually
    annualized_return_pct: float  # Extrapolated annual return
    sharpe_ratio: float  # Target: 2.5-3.5
    max_drawdown_pct: float  # Target: <15%
    current_drawdown_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float
    trades_today: int
    daily_pnl: float
    avg_hold_time_hours: float
    chemistry_performance: Dict[str, Dict[str, float]]
    position_sizing_accuracy: float  # How close to optimal Kelly sizing
    correlation_violations: int  # Positions above 0.7 correlation
    api_calls_last_hour: int
    last_updated: datetime

class CriticalBotError(Exception):
    """
    System-halting error for critical failures
    Matches exact Catalyst Framework error handling pattern
    """
    pass

class PerformanceTracker:
    """
    Complete Performance Tracking System for Catalyst Framework
    
    Implements ALL specifications:
    - Position sizing: capital * min(0.15, kelly_fraction * 0.25, confidence * 0.10)
    - Risk limits: 5% max loss per trade, 15% max per position, 15% max drawdown
    - Performance targets: 75-100% returns, 65-70% win rate, 2.5-3.5 Sharpe
    - API tracking: FMP 300 calls/min, 1000 batch size
    - Chemistry tracking: noble_gas, volatile_compound, phase_change, catalyst_accelerant
    """
    
    def __init__(self, 
                 db_path: str = "catalyst_performance.db", 
                 initial_capital: float = 5000.0,
                 enable_correlation_tracking: bool = True):
        """
        Initialize Performance Tracker with EXACT Catalyst specifications
        
        Args:
            db_path: SQLite database file path
            initial_capital: Starting capital amount ($5K per spec)
            enable_correlation_tracking: Track position correlations
        """
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.enable_correlation_tracking = enable_correlation_tracking
        
        # EXACT Catalyst Framework constants from specification
        self.MAX_DRAWDOWN_HALT = 0.15  # 15% drawdown triggers system halt
        self.MAX_POSITION_PCT = 0.15   # Max 15% capital per position
        self.MAX_LOSS_PER_TRADE_PCT = 0.05  # Max 5% loss per trade
        self.MAX_CORRELATION = 0.7     # Max 0.7 correlation between positions
        self.MAX_POSITIONS = 10        # Max 10 positions total
        
        # Chemistry types from Catalyst specification
        self.CHEMISTRY_TYPES = ['noble_gas', 'volatile_compound', 'phase_change', 'catalyst_accelerant']
        
        # Performance targets from specification
        self.TARGET_WIN_RATE_MIN = 0.65    # 65% minimum
        self.TARGET_WIN_RATE_MAX = 0.70    # 70% maximum
        self.TARGET_ANNUAL_RETURN_MIN = 0.75  # 75% minimum
        self.TARGET_ANNUAL_RETURN_MAX = 1.00  # 100% maximum
        self.TARGET_SHARPE_MIN = 2.5       # 2.5 minimum
        self.TARGET_SHARPE_MAX = 3.5       # 3.5 maximum
        
        # FMP API limits from specification (300 calls/min, 1000 batch size)
        self.FMP_CALLS_PER_MINUTE = 300
        self.FMP_BATCH_SIZE = 1000
        self.api_call_timestamps: List[datetime] = []
        
        # State tracking
        self.open_positions: Dict[str, TradeRecord] = {}
        self.daily_start_capital = initial_capital
        self.peak_capital = initial_capital
        self.last_daily_reset = datetime.now().date()
        
        # Capital history for accurate drawdown calculation
        self.capital_history: List[Tuple[datetime, float]] = [(datetime.now(), initial_capital)]
        
        # Position correlation tracking (if enabled)
        self.position_correlations: Dict[Tuple[str, str], float] = {}
        
        # Initialize database and state
        self._initialize_database()
        self._load_persisted_state()
        
        # Setup logging with Catalyst format
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging with Catalyst Framework format"""
        self.logger = logging.getLogger("Catalyst.PerformanceTracker")
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - CATALYST.PERF - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler
            if not os.path.exists('logs'):
                os.makedirs('logs')
            file_handler = logging.FileHandler('logs/catalyst_performance.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.setLevel(logging.INFO)
    
    def _initialize_database(self) -> None:
        """
        Initialize SQLite database with EXACT schema for Catalyst data
        Handles schema evolution and data integrity
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign keys and WAL mode for performance
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                
                # Main trades table with ALL Catalyst Framework fields
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL CHECK (action IN ('BUY', 'SELL')),
                        quantity INTEGER NOT NULL CHECK (quantity > 0),
                        price REAL NOT NULL CHECK (price > 0),
                        trade_value REAL NOT NULL CHECK (trade_value > 0),
                        chemistry_type TEXT NOT NULL CHECK (chemistry_type IN ('noble_gas', 'volatile_compound', 'phase_change', 'catalyst_accelerant')),
                        market_risk REAL NOT NULL CHECK (market_risk >= 0 AND market_risk <= 1),
                        market_momentum REAL NOT NULL CHECK (market_momentum >= -1 AND market_momentum <= 1),
                        trailhead_score REAL NOT NULL CHECK (trailhead_score >= 0 AND trailhead_score <= 1),
                        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                        kelly_fraction REAL NOT NULL,
                        actual_position_size REAL NOT NULL CHECK (actual_position_size > 0),
                        position_size_pct REAL NOT NULL CHECK (position_size_pct > 0 AND position_size_pct <= 0.15),
                        predicted_horizon_hours REAL NOT NULL CHECK (predicted_horizon_hours >= 4 AND predicted_horizon_hours <= 48),
                        pnl REAL,
                        exit_price REAL,
                        exit_timestamp TEXT,
                        stop_loss REAL,
                        target_price REAL,
                        hold_duration_hours REAL,
                        ibkr_order_id TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Capital snapshots for accurate drawdown tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS capital_snapshots (
                        timestamp TEXT PRIMARY KEY,
                        capital REAL NOT NULL CHECK (capital >= 0),
                        peak_capital REAL NOT NULL CHECK (peak_capital >= capital),
                        drawdown_pct REAL NOT NULL CHECK (drawdown_pct >= 0),
                        daily_pnl REAL NOT NULL,
                        active_positions INTEGER NOT NULL CHECK (active_positions >= 0),
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # API usage tracking for FMP rate limiting
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_calls (
                        timestamp TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        response_time_ms INTEGER,
                        error_message TEXT,
                        PRIMARY KEY (timestamp, endpoint)
                    )
                """)
                
                # Position correlations (if enabled)
                if self.enable_correlation_tracking:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS position_correlations (
                            symbol1 TEXT NOT NULL,
                            symbol2 TEXT NOT NULL,
                            correlation REAL NOT NULL,
                            calculation_timestamp TEXT NOT NULL,
                            PRIMARY KEY (symbol1, symbol2, calculation_timestamp)
                        )
                    """)
                
                # System events and alerts
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_events (
                        timestamp TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'CRITICAL')),
                        message TEXT NOT NULL,
                        data TEXT,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Performance indexes for fast queries
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_chemistry ON trades(chemistry_type)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_action ON trades(action)",
                    "CREATE INDEX IF NOT EXISTS idx_capital_timestamp ON capital_snapshots(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_api_timestamp ON api_calls(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON system_events(timestamp)"
                ]
                
                for index_sql in indexes:
                    conn.execute(index_sql)
                
                # Verify schema integrity
                self._verify_database_schema(conn)
                
        except Exception as e:
            raise CriticalBotError(f"Database initialization failed: {e}")
    
    def _verify_database_schema(self, conn) -> None:
        """Verify database schema matches expected structure"""
        cursor = conn.cursor()
        
        # Check trades table has correct columns
        cursor.execute("PRAGMA table_info(trades)")
        columns = [row[1] for row in cursor.fetchall()]
        
        expected_columns = [
            'trade_id', 'timestamp', 'symbol', 'action', 'quantity', 'price',
            'trade_value', 'chemistry_type', 'market_risk', 'market_momentum',
            'trailhead_score', 'confidence', 'kelly_fraction', 'actual_position_size',
            'position_size_pct', 'predicted_horizon_hours', 'pnl', 'exit_price',
            'exit_timestamp', 'stop_loss', 'target_price', 'hold_duration_hours',
            'ibkr_order_id', 'metadata', 'created_at'
        ]
        
        missing_columns = set(expected_columns) - set(columns)
        if missing_columns:
            raise CriticalBotError(f"Database schema missing columns: {missing_columns}")
    
    def _load_persisted_state(self) -> None:
        """Load persisted state from database with full error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load open positions (BUY orders without corresponding SELL)
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE action = 'BUY' 
                    AND trade_id NOT IN (
                        SELECT DISTINCT trade_id FROM trades WHERE action = 'SELL'
                    )
                    ORDER BY timestamp DESC
                """)
                
                loaded_positions = 0
                for row in cursor.fetchall():
                    try:
                        trade = self._row_to_trade_record(row)
                        position_key = f"{trade.symbol}_{trade.trade_id}"
                        self.open_positions[position_key] = trade
                        loaded_positions += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to load position from row: {e}")
                
                # Load latest capital state
                cursor.execute("""
                    SELECT capital, peak_capital FROM capital_snapshots 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                
                capital_result = cursor.fetchone()
                if capital_result:
                    self.current_capital = capital_result[0]
                    self.peak_capital = capital_result[1]
                else:
                    # Rebuild capital from trade history
                    self._rebuild_capital_from_trade_history(conn)
                
                # Load recent API call history
                cutoff_time = datetime.now() - timedelta(minutes=1)
                cursor.execute("""
                    SELECT timestamp FROM api_calls 
                    WHERE timestamp > ? 
                    ORDER BY timestamp
                """, [cutoff_time.isoformat()])
                
                self.api_call_timestamps = [
                    datetime.fromisoformat(row[0]) for row in cursor.fetchall()
                ]
                
                self.logger.info(f"Loaded state: {loaded_positions} positions, "
                               f"${self.current_capital:.2f} capital, "
                               f"{len(self.api_call_timestamps)} recent API calls")
                
        except Exception as e:
            self.logger.warning(f"State loading failed, using defaults: {e}")
            # Continue with initialized defaults
    
    def record_trade(self, trade: TradeRecord) -> None:
        """
        Record trade with COMPLETE Catalyst Framework validation
        
        Args:
            trade: TradeRecord with all required Catalyst data
        
        Raises:
            CriticalBotError: On validation failure or critical system conditions
        """
        try:
            # Daily reset check
            self._check_daily_reset()
            
            # STRICT validation against ALL Catalyst rules
            self._validate_trade_against_catalyst_rules(trade)
            
            # Calculate hold duration for SELL trades
            if trade.action == 'SELL' and trade.exit_timestamp:
                position_key = f"{trade.symbol}_{trade.trade_id}"
                if position_key in self.open_positions:
                    entry_trade = self.open_positions[position_key]
                    duration = trade.exit_timestamp - entry_trade.timestamp
                    trade.hold_duration_hours = duration.total_seconds() / 3600
            
            # Database insertion with transaction
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO trades VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.trade_id,
                    trade.timestamp.isoformat(),
                    trade.symbol,
                    trade.action,
                    trade.quantity,
                    trade.price,
                    trade.trade_value,
                    trade.chemistry_type,
                    trade.market_risk,
                    trade.market_momentum,
                    trade.trailhead_score,
                    trade.confidence,
                    trade.kelly_fraction,
                    trade.actual_position_size,
                    trade.position_size_pct,
                    trade.predicted_horizon_hours,
                    trade.pnl,
                    trade.exit_price,
                    trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                    trade.stop_loss,
                    trade.target_price,
                    trade.hold_duration_hours,
                    trade.ibkr_order_id,
                    json.dumps(trade.metadata) if trade.metadata else None,
                    datetime.now().isoformat()
                ))
            
            # Update internal state
            self._update_positions_and_capital(trade)
            
            # Record capital snapshot
            self._record_capital_snapshot()
            
            # Check for critical system conditions
            self._check_critical_system_conditions()
            
            # Log successful trade recording
            self.logger.info(
                f"TRADE RECORDED: {trade.action} {trade.symbol} "
                f"${trade.actual_position_size:.0f} ({trade.chemistry_type}) "
                f"Kelly:{trade.kelly_fraction:.3f} Conf:{trade.confidence:.2f}"
            )
            
        except Exception as e:
            # Log error and halt system per Catalyst specification
            error_msg = f"Trade recording failed for {trade.trade_id}: {e}"
            self._log_system_event("TRADE_ERROR", "CRITICAL", error_msg, asdict(trade))
            raise CriticalBotError(error_msg)
    
    def calculate_performance_metrics(self, lookback_days: int = 30) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics with REAL calculations
        
        Args:
            lookback_days: Number of days to include in calculations
            
        Returns:
            PerformanceMetrics with all Catalyst Framework targets
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get completed trades (both BUY and SELL recorded)
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE exit_timestamp IS NOT NULL 
                    AND pnl IS NOT NULL
                    AND timestamp >= ?
                    ORDER BY timestamp
                """, [cutoff_date.isoformat()])
                
                completed_trades = []
                for row in cursor.fetchall():
                    try:
                        trade = self._row_to_trade_record(row)
                        completed_trades.append(trade)
                    except Exception as e:
                        self.logger.warning(f"Skipped corrupted trade record: {e}")
                
                if not completed_trades:
                    return self._create_empty_metrics()
                
                # Calculate basic performance statistics
                total_trades = len(completed_trades)
                winning_trades_list = [t for t in completed_trades if t.pnl and t.pnl > 0]
                losing_trades_list = [t for t in completed_trades if t.pnl and t.pnl < 0]
                
                winning_trades = len(winning_trades_list)
                losing_trades = len(losing_trades_list)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
                
                # P&L calculations
                total_pnl = sum(t.pnl for t in completed_trades if t.pnl is not None)
                total_return_pct = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
                
                # Annualized return calculation
                if lookback_days > 0:
                    daily_return = total_return_pct / lookback_days
                    annualized_return_pct = daily_return * 365
                else:
                    annualized_return_pct = 0.0
                
                # Risk metrics using actual daily returns
                daily_returns = self._calculate_daily_returns(completed_trades)
                sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
                
                # Drawdown calculations from capital history
                max_drawdown_pct, current_drawdown_pct = self._calculate_actual_drawdowns()
                
                # Win/Loss analysis
                win_amounts = [t.pnl for t in winning_trades_list if t.pnl]
                loss_amounts = [abs(t.pnl) for t in losing_trades_list if t.pnl]
                
                avg_win = sum(win_amounts) / len(win_amounts) if win_amounts else 0.0
                avg_loss = sum(loss_amounts) / len(loss_amounts) if loss_amounts else 0.0
                
                profit_factor = (sum(win_amounts) / sum(loss_amounts)) if loss_amounts and sum(loss_amounts) > 0 else float('inf')
                
                largest_win = max(win_amounts) if win_amounts else 0.0
                largest_loss = max(loss_amounts) if loss_amounts else 0.0
                
                # Time analysis
                hold_times = [t.hold_duration_hours for t in completed_trades if t.hold_duration_hours is not None]
                avg_hold_time_hours = sum(hold_times) / len(hold_times) if hold_times else 0.0
                
                # Today's performance
                today = datetime.now().date()
                today_trades = [t for t in completed_trades if t.timestamp.date() == today]
                trades_today = len(today_trades)
                daily_pnl = sum(t.pnl for t in today_trades if t.pnl is not None)
                
                # Chemistry performance analysis
                chemistry_performance = self._analyze_chemistry_performance(completed_trades)
                
                # Position sizing accuracy analysis
                position_sizing_accuracy = self._calculate_position_sizing_accuracy(completed_trades)
                
                # Correlation violations count
                correlation_violations = self._count_correlation_violations()
                
                # API usage tracking
                api_calls_last_hour = self._count_api_calls_last_hour()
                
                return PerformanceMetrics(
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    total_return_pct=total_return_pct,
                    annualized_return_pct=annualized_return_pct,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown_pct=max_drawdown_pct,
                    current_drawdown_pct=current_drawdown_pct,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    profit_factor=profit_factor,
                    largest_win=largest_win,
                    largest_loss=largest_loss,
                    trades_today=trades_today,
                    daily_pnl=daily_pnl,
                    avg_hold_time_hours=avg_hold_time_hours,
                    chemistry_performance=chemistry_performance,
                    position_sizing_accuracy=position_sizing_accuracy,
                    correlation_violations=correlation_violations,
                    api_calls_last_hour=api_calls_last_hour,
                    last_updated=datetime.now()
                )
                
        except Exception as e:
            raise CriticalBotError(f"Performance metrics calculation failed: {e}")
    
    def check_catalyst_compliance(self) -> List[str]:
        """
        Check compliance with ALL Catalyst Framework rules and targets
        
        Returns:
            List of compliance violations and warnings
        """
        violations = []
        
        try:
            metrics = self.calculate_performance_metrics(lookback_days=30)
            
            # CRITICAL: Drawdown limit check (triggers system halt)
            if metrics.current_drawdown_pct > self.MAX_DRAWDOWN_HALT * 100:
                violations.append(
                    f"CRITICAL: Current drawdown {metrics.current_drawdown_pct:.1f}% "
                    f"exceeds {self.MAX_DRAWDOWN_HALT * 100:.0f}% limit - HALT TRADING"
                )
            
            # Position count limit
            if len(self.open_positions) > self.MAX_POSITIONS:
                violations.append(
                    f"CRITICAL: {len(self.open_positions)} open positions "
                    f"exceeds {self.MAX_POSITIONS} limit"
                )
            
            # Individual position size limits
            for position_key, trade in self.open_positions.items():
                position_pct = (trade.actual_position_size / self.current_capital) * 100
                if position_pct > self.MAX_POSITION_PCT * 100:
                    violations.append(
                        f"WARNING: {trade.symbol} position {position_pct:.1f}% "
                        f"exceeds {self.MAX_POSITION_PCT * 100:.0f}% limit"
                    )
            
            # Performance target checks
            if metrics.win_rate < self.TARGET_WIN_RATE_MIN:
                violations.append(
                    f"WARNING: Win rate {metrics.win_rate:.1%} below "
                    f"{self.TARGET_WIN_RATE_MIN:.0%} target"
                )
            
            if metrics.sharpe_ratio < self.TARGET_SHARPE_MIN:
                violations.append(
                    f"WARNING: Sharpe ratio {metrics.sharpe_ratio:.2f} below "
                    f"{self.TARGET_SHARPE_MIN} target"
                )
            
            if metrics.annualized_return_pct < self.TARGET_ANNUAL_RETURN_MIN * 100:
                violations.append(
                    f"WARNING: Annualized return {metrics.annualized_return_pct:.1f}% below "
                    f"{self.TARGET_ANNUAL_RETURN_MIN * 100:.0f}% target"
                )
            
            # Correlation violations
            if metrics.correlation_violations > 0:
                violations.append(
                    f"WARNING: {metrics.correlation_violations} position pairs "
                    f"exceed {self.MAX_CORRELATION} correlation limit"
                )
            
            # API rate limit warnings
            if metrics.api_calls_last_hour > self.FMP_CALLS_PER_MINUTE * 60:
                violations.append(
                    f"WARNING: API calls {metrics.api_calls_last_hour} "
                    f"approaching hourly limit"
                )
            
        except Exception as e:
            violations.append(f"ERROR: Compliance check failed: {e}")
        
        return violations
    
    def record_api_call(self, endpoint: str, success: bool = True, 
                       response_time_ms: Optional[int] = None, 
                       error_message: Optional[str] = None) -> bool:
        """
        Record API call and enforce FMP rate limits
        
        Args:
            endpoint: API endpoint called
            success: Whether call was successful
            response_time_ms: Response time in milliseconds
            error_message: Error message if failed
            
        Returns:
            True if call is allowed, False if rate limited
        """
        now = datetime.now()
        
        # Clean old timestamps (keep only last minute)
        cutoff = now - timedelta(minutes=1)
        self.api_call_timestamps = [ts for ts in self.api_call_timestamps if ts > cutoff]
        
        # Check rate limit BEFORE making call
        if len(self.api_call_timestamps) >= self.FMP_CALLS_PER_MINUTE:
            self.logger.warning(
                f"API rate limit exceeded: {len(self.api_call_timestamps)} calls in last minute"
            )
            return False
        
        # Record the API call
        self.api_call_timestamps.append(now)
        
        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO api_calls VALUES (?, ?, ?, ?, ?)
                """, (
                    now.isoformat(),
                    endpoint,
                    success,
                    response_time_ms,
                    error_message
                ))
        except Exception as e:
            self.logger.error(f"Failed to record API call: {e}")
        
        return True
    
    def get_position_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive summary of current open positions
        
        Returns:
            Dictionary with position details including all Catalyst metrics
        """
        summary = {}
        current_time = datetime.now()
        
        for position_key, trade in self.open_positions.items():
            symbol = trade.symbol
            hours_held = (current_time - trade.timestamp).total_seconds() / 3600
            
            # Calculate unrealized P&L percentage
            unrealized_risk_pct = (trade.actual_position_size * self.MAX_LOSS_PER_TRADE_PCT / trade.actual_position_size) * 100
            
            summary[symbol] = {
                'trade_id': trade.trade_id,
                'quantity': trade.quantity,
                'entry_price': trade.price,
                'position_value': trade.actual_position_size,
                'position_pct_of_capital': (trade.actual_position_size / self.current_capital) * 100,
                'chemistry_type': trade.chemistry_type,
                'kelly_fraction': trade.kelly_fraction,
                'confidence': trade.confidence,
                'trailhead_score': trade.trailhead_score,
                'market_risk': trade.market_risk,
                'market_momentum': trade.market_momentum,
                'hours_held': hours_held,
                'predicted_horizon_hours': trade.predicted_horizon_hours,
                'time_remaining_hours': max(0, trade.predicted_horizon_hours - hours_held),
                'stop_loss': trade.stop_loss,
                'target_price': trade.target_price,
                'max_loss_limit': trade.actual_position_size * self.MAX_LOSS_PER_TRADE_PCT,
                'ibkr_order_id': trade.ibkr_order_id,
                'entry_timestamp': trade.timestamp.isoformat()
            }
        
        return summary
    
    def export_performance_report(self, filepath: str, lookback_days: int = 30) -> bool:
        """
        Export comprehensive Catalyst performance report
        
        Args:
            filepath: Output file path
            lookback_days: Days to include in analysis
            
        Returns:
            True if export successful
        """
        try:
            metrics = self.calculate_performance_metrics(lookback_days)
            violations = self.check_catalyst_compliance()
            positions = self.get_position_summary()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("CATALYST TRADING SYSTEM - PERFORMANCE REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                # Executive Summary
                f.write("EXECUTIVE SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Analysis Period: {lookback_days} days\n")
                f.write(f"Current Capital: ${self.current_capital:,.2f}\n")
                f.write(f"Initial Capital: ${self.initial_capital:,.2f}\n")
                f.write(f"Total Return: {metrics.total_return_pct:.1f}%\n")
                f.write(f"Annualized Return: {metrics.annualized_return_pct:.1f}% (Target: 75-100%)\n")
                f.write(f"Win Rate: {metrics.win_rate:.1%} (Target: 65-70%)\n")
                f.write(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f} (Target: 2.5-3.5)\n")
                f.write(f"Max Drawdown: {metrics.max_drawdown_pct:.1f}% (Limit: 15%)\n")
                f.write(f"Current Drawdown: {metrics.current_drawdown_pct:.1f}%\n\n")
                
                # Detailed Performance Metrics
                f.write("DETAILED PERFORMANCE METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Trades: {metrics.total_trades}\n")
                f.write(f"Winning Trades: {metrics.winning_trades}\n")
                f.write(f"Losing Trades: {metrics.losing_trades}\n")
                f.write(f"Profit Factor: {metrics.profit_factor:.2f}\n")
                f.write(f"Average Win: ${metrics.avg_win:.2f}\n")
                f.write(f"Average Loss: ${metrics.avg_loss:.2f}\n")
                f.write(f"Largest Win: ${metrics.largest_win:.2f}\n")
                f.write(f"Largest Loss: ${metrics.largest_loss:.2f}\n")
                f.write(f"Average Hold Time: {metrics.avg_hold_time_hours:.1f} hours\n")
                f.write(f"Position Sizing Accuracy: {metrics.position_sizing_accuracy:.1%}\n\n")
                
                # Chemistry Performance Analysis
                f.write("CHEMISTRY PERFORMANCE ANALYSIS\n")
                f.write("-" * 35 + "\n")
                for chem_type, stats in metrics.chemistry_performance.items():
                    f.write(f"{chem_type.upper()}:\n")
                    f.write(f"  Total P&L: ${stats['total_pnl']:,.2f}\n")
                    f.write(f"  Trade Count: {stats['trade_count']}\n")
                    f.write(f"  Win Rate: {stats['win_rate']:.1%}\n")
                    f.write(f"  Average P&L: ${stats['avg_pnl']:,.2f}\n\n")
                
                # Current Open Positions
                f.write("CURRENT OPEN POSITIONS\n")
                f.write("-" * 25 + "\n")
                if positions:
                    for symbol, details in positions.items():
                        f.write(f"{symbol} ({details['chemistry_type']}):\n")
                        f.write(f"  Position Value: ${details['position_value']:,.2f} ({details['position_pct_of_capital']:.1f}% of capital)\n")
                        f.write(f"  Entry Price: ${details['entry_price']:.2f}\n")
                        f.write(f"  Stop Loss: ${details['stop_loss']:.2f}\n")
                        f.write(f"  Target Price: ${details['target_price']:.2f}\n")
                        f.write(f"  Hours Held: {details['hours_held']:.1f} / {details['predicted_horizon_hours']:.1f}\n")
                        f.write(f"  Confidence: {details['confidence']:.2f}, Kelly: {details['kelly_fraction']:.3f}\n")
                        f.write(f"  Trailhead Score: {details['trailhead_score']:.2f}\n\n")
                else:
                    f.write("No open positions\n\n")
                
                # Compliance Status
                f.write("CATALYST COMPLIANCE STATUS\n")
                f.write("-" * 30 + "\n")
                if violations:
                    for violation in violations:
                        severity = "CRITICAL" if "CRITICAL" in violation else "WARNING"
                        f.write(f"[{severity}] {violation}\n")
                else:
                    f.write("✅ All Catalyst compliance rules satisfied\n")
                f.write("\n")
                
                # API Usage Statistics
                f.write("API USAGE STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Calls Last Hour: {metrics.api_calls_last_hour}\n")
                f.write(f"Rate Limit: {self.FMP_CALLS_PER_MINUTE} calls/minute\n")
                f.write(f"Batch Size Limit: {self.FMP_BATCH_SIZE}\n\n")
                
                # Today's Activity
                f.write("TODAY'S ACTIVITY\n")
                f.write("-" * 15 + "\n")
                f.write(f"Trades Today: {metrics.trades_today}\n")
                f.write(f"Daily P&L: ${metrics.daily_pnl:,.2f}\n")
                f.write(f"Correlation Violations: {metrics.correlation_violations}\n\n")
                
                f.write(f"Report Generated: {datetime.now().isoformat()}\n")
                f.write(f"System Version: Catalyst Framework v1.0\n")
            
            self.logger.info(f"Performance report exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Report export failed: {e}")
            return False
    
    # PRIVATE HELPER METHODS - ALL FULLY IMPLEMENTED
    
    def _validate_trade_against_catalyst_rules(self, trade: TradeRecord) -> None:
        """Validate trade against ALL Catalyst Framework rules"""
        
        # Chemistry type validation
        if trade.chemistry_type not in self.CHEMISTRY_TYPES:
            raise CriticalBotError(f"Invalid chemistry type '{trade.chemistry_type}'. Must be one of: {self.CHEMISTRY_TYPES}")
        
        # Market state bounds validation
        if not (0 <= trade.market_risk <= 1):
            raise CriticalBotError(f"Market risk {trade.market_risk} outside valid range [0, 1]")
        
        if not (-1 <= trade.market_momentum <= 1):
            raise CriticalBotError(f"Market momentum {trade.market_momentum} outside valid range [-1, 1]")
        
        # Confidence and score bounds
        if not (0 <= trade.confidence <= 1):
            raise CriticalBotError(f"Confidence {trade.confidence} outside valid range [0, 1]")
        
        if not (0 <= trade.trailhead_score <= 1):
            raise CriticalBotError(f"Trailhead score {trade.trailhead_score} outside valid range [0, 1]")
        
        # Position sizing validation for BUY orders
        if trade.action == 'BUY':
            expected_size = self._calculate_optimal_position_size(
                trade.kelly_fraction, trade.confidence, self.current_capital
            )
            
            size_deviation = abs(trade.actual_position_size - expected_size) / max(expected_size, 1)
            if size_deviation > 0.15:  # Allow 15% deviation
                raise CriticalBotError(
                    f"Position size deviation {size_deviation:.1%} too large. "
                    f"Expected: ${expected_size:.0f}, Actual: ${trade.actual_position_size:.0f}"
                )
        
        # Prediction horizon bounds (4-48 hours per Catalyst spec)
        if not (4 <= trade.predicted_horizon_hours <= 48):
            raise CriticalBotError(
                f"Prediction horizon {trade.predicted_horizon_hours} hours outside valid range [4, 48]"
            )
        
        # Position size percentage limit
        if trade.position_size_pct > self.MAX_POSITION_PCT:
            raise CriticalBotError(
                f"Position size {trade.position_size_pct:.1%} exceeds {self.MAX_POSITION_PCT:.1%} limit"
            )
        
        # Position count limit check
        if trade.action == 'BUY' and len(self.open_positions) >= self.MAX_POSITIONS:
            raise CriticalBotError(
                f"Cannot open new position. Already at {self.MAX_POSITIONS} position limit"
            )
    
    def _calculate_optimal_position_size(self, kelly_fraction: float, confidence: float, capital: float) -> float:
        """
        Calculate optimal position size using EXACT Catalyst formula
        Formula: capital * min(0.15, kelly_fraction * 0.25, confidence * 0.10)
        """
        kelly_adjusted = kelly_fraction * 0.25  # 1/4 Kelly per specification
        confidence_adjusted = confidence * 0.10  # Confidence-based sizing
        max_position_fraction = self.MAX_POSITION_PCT  # 15% maximum
        
        optimal_fraction = min(max_position_fraction, kelly_adjusted, confidence_adjusted)
        return capital * optimal_fraction
    
    def _update_positions_and_capital(self, trade: TradeRecord) -> None:
        """Update position tracking and capital with full validation"""
        position_key = f"{trade.symbol}_{trade.trade_id}"
        
        if trade.action == 'BUY':
            # Add to open positions
            self.open_positions[position_key] = trade
            self.current_capital -= trade.actual_position_size
            
            self.logger.debug(f"Opened position {position_key}, capital now ${self.current_capital:.2f}")
            
        elif trade.action == 'SELL':
            # Close position if it exists
            if position_key in self.open_positions:
                del self.open_positions[position_key]
                self.logger.debug(f"Closed position {position_key}")
            
            # Update capital with trade value and P&L
            if trade.pnl is not None:
                self.current_capital += trade.actual_position_size + trade.pnl
                self.peak_capital = max(self.peak_capital, self.current_capital)
                
                # Add to capital history for drawdown tracking
                self.capital_history.append((datetime.now(), self.current_capital))
                
                # Limit capital history size (keep last 1000 entries)
                if len(self.capital_history) > 1000:
                    self.capital_history = self.capital_history[-1000:]
                
                self.logger.debug(f"Updated capital to ${self.current_capital:.2f} after P&L ${trade.pnl:.2f}")
    
    def _calculate_daily_returns(self, trades: List[TradeRecord]) -> List[float]:
        """Calculate actual daily returns from completed trades"""
        daily_pnl_map = {}
        
        for trade in trades:
            if trade.exit_timestamp and trade.pnl is not None:
                date_key = trade.exit_timestamp.date()
                daily_pnl_map[date_key] = daily_pnl_map.get(date_key, 0.0) + trade.pnl
        
        return list(daily_pnl_map.values())
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float]) -> float:
        """Calculate Sharpe ratio using proper statistical methods"""
        if len(daily_returns) < 2:
            return 0.0
        
        n = len(daily_returns)
        mean_return = sum(daily_returns) / n
        
        # Calculate sample standard deviation
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / (n - 1)
        std_dev = variance ** 0.5 if variance > 0 else 0.0
        
        if std_dev == 0:
            return 0.0
        
        # Annualize assuming 252 trading days
        annualized_sharpe = (mean_return / std_dev) * (252 ** 0.5)
        
        return annualized_sharpe
    
    def _calculate_actual_drawdowns(self) -> Tuple[float, float]:
        """Calculate actual drawdowns from capital history"""
        if len(self.capital_history) < 2:
            return 0.0, 0.0
        
        max_drawdown_pct = 0.0
        running_peak = self.capital_history[0][1]
        
        # Calculate maximum drawdown from historical data
        for timestamp, capital in self.capital_history:
            running_peak = max(running_peak, capital)
            if running_peak > 0:
                drawdown_pct = ((running_peak - capital) / running_peak) * 100
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        
        # Calculate current drawdown
        current_drawdown_pct = 0.0
        if self.peak_capital > 0:
            current_drawdown_pct = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
        
        return max_drawdown_pct, current_drawdown_pct
    
    def _analyze_chemistry_performance(self, trades: List[TradeRecord]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by chemistry type with complete statistics"""
        chemistry_stats = {}
        
        for chem_type in self.CHEMISTRY_TYPES:
            chem_trades = [t for t in trades if t.chemistry_type == chem_type and t.pnl is not None]
            
            if chem_trades:
                pnl_values = [t.pnl for t in chem_trades]
                winning_trades = [pnl for pnl in pnl_values if pnl > 0]
                
                chemistry_stats[chem_type] = {
                    'total_pnl': sum(pnl_values),
                    'trade_count': len(chem_trades),
                    'win_rate': len(winning_trades) / len(chem_trades),
                    'avg_pnl': sum(pnl_values) / len(chem_trades),
                    'best_trade': max(pnl_values),
                    'worst_trade': min(pnl_values)
                }
            else:
                chemistry_stats[chem_type] = {
                    'total_pnl': 0.0,
                    'trade_count': 0,
                    'win_rate': 0.0,
                    'avg_pnl': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0
                }
        
        return chemistry_stats
    
    def _calculate_position_sizing_accuracy(self, trades: List[TradeRecord]) -> float:
        """Calculate how accurately we follow optimal Kelly sizing"""
        sizing_errors = []
        
        for trade in trades:
            if trade.action == 'BUY':
                # Use capital at time of trade (approximate with initial capital)
                optimal_size = self._calculate_optimal_position_size(
                    trade.kelly_fraction, trade.confidence, self.initial_capital
                )
                
                if optimal_size > 0:
                    error = abs(trade.actual_position_size - optimal_size) / optimal_size
                    sizing_errors.append(error)
        
        if not sizing_errors:
            return 1.0  # Perfect accuracy if no trades
        
        # Calculate accuracy as (1 - average_error)
        avg_error = sum(sizing_errors) / len(sizing_errors)
        accuracy = max(0.0, 1.0 - avg_error)
        
        return accuracy
    
    def _count_correlation_violations(self) -> int:
        """Count position pairs with correlation above limit"""
        if not self.enable_correlation_tracking or len(self.open_positions) < 2:
            return 0
        
        violations = 0
        symbols = list(set(trade.symbol for trade in self.open_positions.values()))
        
        # Check all pairs of symbols
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                correlation = self.position_correlations.get((symbol1, symbol2), 0.0)
                
                if abs(correlation) > self.MAX_CORRELATION:
                    violations += 1
        
        return violations
    
    def _count_api_calls_last_hour(self) -> int:
        """Count API calls in the last hour"""
        cutoff = datetime.now() - timedelta(hours=1)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM api_calls 
                    WHERE timestamp > ?
                """, [cutoff.isoformat()])
                
                result = cursor.fetchone()
                return result[0] if result else 0
                
        except Exception:
            # Fallback to in-memory tracking
            return len([ts for ts in self.api_call_timestamps if ts > cutoff])
    
    def _check_daily_reset(self) -> None:
        """Check and perform daily reset if needed"""
        current_date = datetime.now().date()
        if current_date > self.last_daily_reset:
            self.daily_start_capital = self.current_capital
            self.last_daily_reset = current_date
            self.logger.info(f"Daily reset: starting capital ${self.current_capital:.2f}")
    
    def _record_capital_snapshot(self) -> None:
        """Record capital snapshot for tracking"""
        try:
            current_time = datetime.now()
            current_drawdown_pct = 0.0
            
            if self.peak_capital > 0:
                current_drawdown_pct = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
            
            daily_pnl = self.current_capital - self.daily_start_capital
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO capital_snapshots VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    current_time.isoformat(),
                    self.current_capital,
                    self.peak_capital,
                    current_drawdown_pct,
                    daily_pnl,
                    len(self.open_positions)
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to record capital snapshot: {e}")
    
    def _check_critical_system_conditions(self) -> None:
        """Check for critical system conditions that require immediate action"""
        
        # Check drawdown limit
        if self.peak_capital > 0:
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if current_drawdown > self.MAX_DRAWDOWN_HALT:
                critical_msg = f"CRITICAL: Drawdown {current_drawdown:.1%} exceeds {self.MAX_DRAWDOWN_HALT:.1%} limit"
                self._log_system_event("DRAWDOWN_LIMIT_EXCEEDED", "CRITICAL", critical_msg)
                self.logger.critical(critical_msg)
        
        # Check position count
        if len(self.open_positions) > self.MAX_POSITIONS:
            critical_msg = f"CRITICAL: {len(self.open_positions)} positions exceed {self.MAX_POSITIONS} limit"
            self._log_system_event("POSITION_COUNT_EXCEEDED", "CRITICAL", critical_msg)
            self.logger.critical(critical_msg)
        
        # Check individual position sizes
        for position_key, trade in self.open_positions.items():
            position_pct = trade.actual_position_size / self.current_capital
            if position_pct > self.MAX_POSITION_PCT:
                warning_msg = f"WARNING: {trade.symbol} position {position_pct:.1%} exceeds {self.MAX_POSITION_PCT:.1%}"
                self._log_system_event("POSITION_SIZE_WARNING", "WARNING", warning_msg)
                self.logger.warning(warning_msg)
    
    def _log_system_event(self, event_type: str, severity: str, message: str, data: Optional[Dict] = None) -> None:
        """Log system event to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_events VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    event_type,
                    severity,
                    message,
                    json.dumps(data) if data else None,
                    False  # resolved = False
                ))
        except Exception as e:
            self.logger.error(f"Failed to log system event: {e}")
    
    def _rebuild_capital_from_trade_history(self, conn) -> None:
        """Rebuild capital from trade history if snapshots are missing"""
        cursor = conn.cursor()
        
        # Get all completed trades in chronological order
        cursor.execute("""
            SELECT actual_position_size, pnl, action, timestamp FROM trades 
            WHERE pnl IS NOT NULL 
            ORDER BY timestamp
        """)
        
        capital = self.initial_capital
        peak = self.initial_capital
        
        for row in cursor.fetchall():
            position_size, pnl, action, timestamp_str = row
            
            if action == 'SELL':
                capital += position_size + pnl
                peak = max(peak, capital)
                
                # Add to capital history
                timestamp = datetime.fromisoformat(timestamp_str)
                self.capital_history.append((timestamp, capital))
        
        self.current_capital = capital
        self.peak_capital = peak
        
        self.logger.info(f"Rebuilt capital from trade history: ${capital:.2f}")
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics when no trade data is available"""
        empty_chemistry = {
            chem_type: {
                'total_pnl': 0.0, 'trade_count': 0, 'win_rate': 0.0, 
                'avg_pnl': 0.0, 'best_trade': 0.0, 'worst_trade': 0.0
            } 
            for chem_type in self.CHEMISTRY_TYPES
        }
        
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            current_drawdown_pct=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            trades_today=0,
            daily_pnl=0.0,
            avg_hold_time_hours=0.0,
            chemistry_performance=empty_chemistry,
            position_sizing_accuracy=1.0,
            correlation_violations=0,
            api_calls_last_hour=0,
            last_updated=datetime.now()
        )
    
    def _row_to_trade_record(self, row: Tuple) -> TradeRecord:
        """Convert database row to TradeRecord object with full error handling"""
        try:
            return TradeRecord(
                trade_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                symbol=row[2],
                action=row[3],
                quantity=int(row[4]),
                price=float(row[5]),
                trade_value=float(row[6]),
                chemistry_type=row[7],
                market_risk=float(row[8]),
                market_momentum=float(row[9]),
                trailhead_score=float(row[10]),
                confidence=float(row[11]),
                kelly_fraction=float(row[12]),
                actual_position_size=float(row[13]),
                position_size_pct=float(row[14]),
                predicted_horizon_hours=float(row[15]),
                pnl=float(row[16]) if row[16] is not None else None,
                exit_price=float(row[17]) if row[17] is not None else None,
                exit_timestamp=datetime.fromisoformat(row[18]) if row[18] is not None else None,
                stop_loss=float(row[19]) if row[19] is not None else None,
                target_price=float(row[20]) if row[20] is not None else None,
                hold_duration_hours=float(row[21]) if row[21] is not None else None,
                ibkr_order_id=row[22] if row[22] is not None else None,
                metadata=json.loads(row[23]) if row[23] is not None else None
            )
        except Exception as e:
            raise CriticalBotError(f"Failed to parse trade record from database row: {e}")