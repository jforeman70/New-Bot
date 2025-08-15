"""
Execution Engine - Production Ready IBKR Integration
Real money execution with comprehensive error handling
"""

import time
import threading
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import queue
import warnings
warnings.filterwarnings('ignore')

# IBKR API imports
try:
    from ib_insync import IB, Stock, Forex, Future, Order, Trade, MarketOrder, LimitOrder
    from ib_insync import Contract, OrderStatus, Position, AccountValue
    IBKR_AVAILABLE = True
except ImportError:
    print("WARNING: ib_insync not installed. Run: pip install ib_insync")
    IBKR_AVAILABLE = False

class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"

@dataclass
class TradeOrder:
    """Immutable trade order specification"""
    order_id: str
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    account: str = ""
    
class ExecutionResult(NamedTuple):
    """Trade execution result"""
    order_id: str
    symbol: str
    status: OrderStatus
    filled_quantity: int
    avg_fill_price: float
    commission: float
    timestamp: datetime
    error_message: str = ""

class IBKRConnection:
    """IBKR TWS/Gateway connection manager"""
    
    def __init__(self):
        if not IBKR_AVAILABLE:
            raise ImportError("ib_insync required for IBKR connection")
            
        self.ib = IB()
        self.logger = logging.getLogger(__name__)
        
        # Connection parameters from environment
        self.host = os.getenv('IBKR_HOST', '127.0.0.1')
        self.port = int(os.getenv('IBKR_PORT', '7497'))  # 7497 = TWS, 4002 = Gateway
        self.client_id = int(os.getenv('IBKR_CLIENT_ID', '0'))
        self.account = os.getenv('IBKR_ACCOUNT', '')
        
        self.connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
    def connect(self) -> bool:
        """Establish connection to IBKR TWS/Gateway"""
        try:
            if self.connected:
                return True
                
            self.logger.info(f"Connecting to IBKR at {self.host}:{self.port}")
            
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=30
            )
            
            self.connected = True
            self.connection_attempts = 0
            
            # Verify account access
            if self.account:
                accounts = self.ib.managedAccounts()
                if self.account not in accounts:
                    raise ValueError(f"Account {self.account} not accessible. Available: {accounts}")
                    
            self.logger.info(f"IBKR connection established. Account: {self.account}")
            return True
            
        except Exception as e:
            self.connection_attempts += 1
            self.logger.error(f"IBKR connection failed (attempt {self.connection_attempts}): {e}")
            
            if self.connection_attempts >= self.max_connection_attempts:
                raise RuntimeError(f"IBKR connection failed after {self.max_connection_attempts} attempts")
                
            return False
            
    def disconnect(self):
        """Safely disconnect from IBKR"""
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                self.logger.info("IBKR disconnected")
        except Exception as e:
            self.logger.error(f"IBKR disconnect error: {e}")
            
    def is_connected(self) -> bool:
        """Check if still connected"""
        try:
            return self.ib.isConnected()
        except:
            return False

class ExecutionEngine:
    """
    Production-ready execution engine
    Handles real money trades with comprehensive safety checks
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.ibkr = IBKRConnection()
        
        # Execution parameters
        self.max_slippage_bps = 20        # 20 basis points max slippage
        self.order_timeout_seconds = 30   # 30 second order timeout
        self.max_daily_trades = 50        # Daily trade limit
        
        # State tracking
        self.active_orders: Dict[str, TradeOrder] = {}
        self.executed_trades: List[ExecutionResult] = []
        self.daily_trade_count = 0
        self.last_trade_date = datetime.now().date()
        
        # Order queue for async processing
        self.order_queue = queue.Queue()
        self.execution_thread = None
        self.stop_execution = False
        
        # Emergency circuit breaker
        self.circuit_breaker_active = False
        self.emergency_stop_reason = ""
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('execution_engine.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def _reset_daily_counter(self):
        """Reset daily trade counter"""
        today = datetime.now().date()
        if today != self.last_trade_date:
            self.daily_trade_count = 0
            self.last_trade_date = today
            
    def _create_contract(self, symbol: str) -> Contract:
        """Create IBKR contract object"""
        try:
            # Detect contract type
            if len(symbol) == 6 and symbol.isalpha():  # EURUSD format
                # Forex pair
                base_currency = symbol[:3]
                quote_currency = symbol[3:]
                contract = Forex(pair=symbol)
                
            elif symbol.endswith('USD') and len(symbol) == 6:  # BTCUSD format
                # Crypto
                contract = Contract(
                    symbol=symbol,
                    secType='CRYPTO',
                    exchange='PAXOS',
                    currency='USD'
                )
                
            else:
                # Stock/ETF
                contract = Stock(symbol=symbol, exchange='SMART', currency='USD')
                
            return contract
            
        except Exception as e:
            self.logger.error(f"Contract creation failed for {symbol}: {e}")
            raise ValueError(f"Invalid symbol: {symbol}")
            
    def _calculate_quantity(self, symbol: str, size_usd: float, 
                          current_price: float, leverage: float = 1.0) -> int:
        """Calculate share/unit quantity for USD size"""
        try:
            if current_price <= 0:
                raise ValueError(f"Invalid price for {symbol}: {current_price}")
                
            # Calculate base quantity
            base_quantity = size_usd / current_price
            
            # Apply leverage
            leveraged_quantity = base_quantity * leverage
            
            # Round to appropriate unit size
            if symbol.startswith(('BTC', 'ETH')):  # Crypto - smaller units
                quantity = round(leveraged_quantity, 4)
            elif len(symbol) == 6 and symbol.isalpha():  # Forex - standard lots
                quantity = round(leveraged_quantity / 100000) * 100000  # Round to nearest 100k
            else:  # Stocks - whole shares
                quantity = int(leveraged_quantity)
                
            if quantity <= 0:
                raise ValueError(f"Calculated quantity <= 0 for {symbol}")
                
            return quantity
            
        except Exception as e:
            self.logger.error(f"Quantity calculation failed: {e}")
            raise
            
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            if not self.ibkr.is_connected():
                self.ibkr.connect()
                
            contract = self._create_contract(symbol)
            
            # Request market data
            self.ibkr.ib.reqMktData(contract)
            time.sleep(1)  # Wait for data
            
            ticker = self.ibkr.ib.ticker(contract)
            
            if ticker.marketPrice() and ticker.marketPrice() > 0:
                price = ticker.marketPrice()
            elif ticker.close and ticker.close > 0:
                price = ticker.close
            else:
                raise ValueError(f"No valid price data for {symbol}")
                
            self.ibkr.ib.cancelMktData(contract)
            
            return float(price)
            
        except Exception as e:
            self.logger.error(f"Price fetch failed for {symbol}: {e}")
            raise RuntimeError(f"Cannot get price for {symbol}: {e}")
            
    def _validate_order(self, order: TradeOrder) -> bool:
        """Comprehensive order validation"""
        try:
            # Reset daily counter if needed
            self._reset_daily_counter()
            
            # Check daily trade limit
            if self.daily_trade_count >= self.max_daily_trades:
                self.logger.error(f"Daily trade limit exceeded: {self.daily_trade_count}")
                return False
                
            # Check circuit breaker
            if self.circuit_breaker_active:
                self.logger.error(f"Circuit breaker active: {self.emergency_stop_reason}")
                return False
                
            # Validate order parameters
            if order.quantity <= 0:
                self.logger.error(f"Invalid quantity: {order.quantity}")
                return False
                
            if order.action not in ['BUY', 'SELL']:
                self.logger.error(f"Invalid action: {order.action}")
                return False
                
            # Check connection
            if not self.ibkr.is_connected():
                if not self.ibkr.connect():
                    self.logger.error("IBKR connection failed")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False
            
    def execute_trade(self, symbol: str, action: str, size_usd: float,
                     leverage: float = 1.0, order_type: OrderType = OrderType.MARKET,
                     limit_price: Optional[float] = None) -> ExecutionResult:
        """
        Execute a trade with comprehensive safety checks
        """
        start_time = datetime.now()
        order_id = f"{symbol}_{action}_{int(start_time.timestamp())}"
        
        try:
            self.logger.info(f"Executing trade: {symbol} {action} ${size_usd:.2f}")
            
            # Get current price
            current_price = self._get_current_price(symbol)
            
            # Calculate quantity
            quantity = self._calculate_quantity(symbol, size_usd, current_price, leverage)
            
            # Create trade order
            trade_order = TradeOrder(
                order_id=order_id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                account=self.ibkr.account
            )
            
            # Validate order
            if not self._validate_order(trade_order):
                return ExecutionResult(
                    order_id=order_id,
                    symbol=symbol,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    avg_fill_price=0.0,
                    commission=0.0,
                    timestamp=datetime.now(),
                    error_message="Order validation failed"
                )
                
            # Create IBKR contract and order
            contract = self._create_contract(symbol)
            
            if order_type == OrderType.MARKET:
                ib_order = MarketOrder(action=action, totalQuantity=quantity)
            elif order_type == OrderType.LIMIT:
                if not limit_price:
                    limit_price = current_price
                ib_order = LimitOrder(action=action, totalQuantity=quantity, lmtPrice=limit_price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
                
            # Set account if specified
            if self.ibkr.account:
                ib_order.account = self.ibkr.account
                
            # Place order
            self.active_orders[order_id] = trade_order
            
            trade = self.ibkr.ib.placeOrder(contract, ib_order)
            
            # Wait for fill with timeout
            timeout_time = start_time + timedelta(seconds=self.order_timeout_seconds)
            
            while datetime.now() < timeout_time:
                self.ibkr.ib.sleep(0.1)  # Update order status
                
                if trade.orderStatus.status in ['Filled', 'Cancelled']:
                    break
                    
            # Process result
            if trade.orderStatus.status == 'Filled':
                filled_qty = trade.orderStatus.filled
                avg_price = trade.orderStatus.avgFillPrice
                commission = sum([fill.commission for fill in trade.fills])
                
                result = ExecutionResult(
                    order_id=order_id,
                    symbol=symbol,
                    status=OrderStatus.FILLED,
                    filled_quantity=int(filled_qty),
                    avg_fill_price=float(avg_price),
                    commission=float(commission),
                    timestamp=datetime.now()
                )
                
                # Check slippage
                if order_type == OrderType.MARKET:
                    slippage_bps = abs(avg_price - current_price) / current_price * 10000
                    if slippage_bps > self.max_slippage_bps:
                        self.logger.warning(f"High slippage: {slippage_bps:.1f} bps")
                        
                self.daily_trade_count += 1
                self.logger.info(f"Trade executed: {result}")
                
            elif trade.orderStatus.status == 'Cancelled':
                # Cancel order
                self.ibkr.ib.cancelOrder(ib_order)
                
                result = ExecutionResult(
                    order_id=order_id,
                    symbol=symbol,
                    status=OrderStatus.CANCELLED,
                    filled_quantity=0,
                    avg_fill_price=0.0,
                    commission=0.0,
                    timestamp=datetime.now(),
                    error_message="Order timeout"
                )
                
            else:
                # Order still pending or rejected
                if trade.orderStatus.status == 'Cancelled':
                    status = OrderStatus.CANCELLED
                else:
                    status = OrderStatus.REJECTED
                    
                result = ExecutionResult(
                    order_id=order_id,
                    symbol=symbol,
                    status=status,
                    filled_quantity=0,
                    avg_fill_price=0.0,
                    commission=0.0,
                    timestamp=datetime.now(),
                    error_message=f"Order status: {trade.orderStatus.status}"
                )
                
            # Cleanup
            if order_id in self.active_orders:
                del self.active_orders[order_id]
                
            self.executed_trades.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            
            # Cleanup on error
            if order_id in self.active_orders:
                del self.active_orders[order_id]
                
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                status=OrderStatus.REJECTED,
                filled_quantity=0,
                avg_fill_price=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                error_message=str(e)
            )
            
    def emergency_flatten_all(self) -> List[ExecutionResult]:
        """
        EMERGENCY: Close all positions immediately
        """
        self.logger.critical("🚨 EMERGENCY FLATTEN ALL POSITIONS 🚨")
        
        results = []
        
        try:
            if not self.ibkr.is_connected():
                self.ibkr.connect()
                
            # Get all positions
            positions = self.ibkr.ib.positions()
            
            for position in positions:
                if position.position != 0:  # Has open position
                    symbol = position.contract.symbol
                    current_position = position.position
                    
                    # Determine action to close position
                    action = 'SELL' if current_position > 0 else 'BUY'
                    quantity = abs(current_position)
                    
                    self.logger.critical(f"Emergency closing {symbol}: {action} {quantity}")
                    
                    # Create emergency market order
                    contract = position.contract
                    order = MarketOrder(action=action, totalQuantity=quantity)
                    
                    if self.ibkr.account:
                        order.account = self.ibkr.account
                        
                    # Place order immediately
                    trade = self.ibkr.ib.placeOrder(contract, order)
                    
                    # Wait briefly for fill
                    time.sleep(2)
                    self.ibkr.ib.sleep(0.1)
                    
                    result = ExecutionResult(
                        order_id=f"EMERGENCY_{symbol}_{int(datetime.now().timestamp())}",
                        symbol=symbol,
                        status=OrderStatus.FILLED if trade.orderStatus.status == 'Filled' else OrderStatus.PENDING,
                        filled_quantity=int(trade.orderStatus.filled) if trade.orderStatus.filled else 0,
                        avg_fill_price=float(trade.orderStatus.avgFillPrice) if trade.orderStatus.avgFillPrice else 0.0,
                        commission=0.0,
                        timestamp=datetime.now(),
                        error_message="Emergency flatten"
                    )
                    
                    results.append(result)
                    
        except Exception as e:
            self.logger.critical(f"Emergency flatten failed: {e}")
            
        return results
        
    def activate_circuit_breaker(self, reason: str):
        """Activate emergency circuit breaker"""
        self.circuit_breaker_active = True
        self.emergency_stop_reason = reason
        self.logger.critical(f"🔴 CIRCUIT BREAKER ACTIVATED: {reason}")
        
        # Emergency flatten all positions
        self.emergency_flatten_all()
        
    def deactivate_circuit_breaker(self, override_code: str = "MANUAL_RESET"):
        """Deactivate circuit breaker - use with extreme caution"""
        if override_code == "MANUAL_RESET":
            self.circuit_breaker_active = False
            self.emergency_stop_reason = ""
            self.logger.warning("🟢 Circuit breaker manually deactivated")
        else:
            self.logger.error("Invalid circuit breaker override code")
            
    def get_account_summary(self) -> Dict:
        """Get current account information"""
        try:
            if not self.ibkr.is_connected():
                self.ibkr.connect()
                
            account_values = self.ibkr.ib.accountValues()
            positions = self.ibkr.ib.positions()
            
            summary = {
                'account_id': self.ibkr.account,
                'timestamp': datetime.now(),
                'connected': self.ibkr.is_connected(),
                'circuit_breaker_active': self.circuit_breaker_active,
                'daily_trades': self.daily_trade_count,
                'active_orders': len(self.active_orders)
            }
            
            # Extract key account values
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    summary['net_liquidation'] = float(av.value)
                elif av.tag == 'TotalCashValue':
                    summary['cash'] = float(av.value)
                elif av.tag == 'GrossPositionValue':
                    summary['gross_position_value'] = float(av.value)
                elif av.tag == 'UnrealizedPnL':
                    summary['unrealized_pnl'] = float(av.value)
                    
            # Current positions
            summary['positions'] = {}
            for pos in positions:
                if pos.position != 0:
                    summary['positions'][pos.contract.symbol] = {
                        'quantity': pos.position,
                        'avg_cost': pos.avgCost,
                        'market_value': pos.marketValue,
                        'unrealized_pnl': pos.unrealizedPNL
                    }
                    
            return summary
            
        except Exception as e:
            self.logger.error(f"Account summary failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(),
                'connected': False
            }
            
    def health_check(self) -> Dict[str, bool]:
        """System health check"""
        health = {
            'ibkr_connected': self.ibkr.is_connected(),
            'circuit_breaker_ok': not self.circuit_breaker_active,
            'daily_trade_limit_ok': self.daily_trade_count < self.max_daily_trades,
            'no_stuck_orders': len(self.active_orders) == 0
        }
        
        return health

# Usage example
if __name__ == "__main__":
    if not IBKR_AVAILABLE:
        print("IBKR integration not available - install ib_insync")
        exit(1)
        
    engine = ExecutionEngine()
    
    try:
        # Health check
        health = engine.health_check()
        print(f"System Health: {health}")
        
        if all(health.values()):
            # Get account summary
            account = engine.get_account_summary()
            print(f"Account Summary: {account}")
            
            # Example trade (paper/demo only)
            # result = engine.execute_trade(
            #     symbol='SPY',
            #     action='BUY',
            #     size_usd=1000,
            #     leverage=1.0,
            #     order_type=OrderType.MARKET
            # )
            # print(f"Trade Result: {result}")
            
        else:
            print("❌ System not ready for trading")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
