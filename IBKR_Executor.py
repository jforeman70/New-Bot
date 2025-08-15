import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from ib_insync import IB, Stock, Option, LimitOrder, MarketOrder, StopOrder, Contract, Trade
import time

from Portfolio_Synthesizer import PortfolioPosition
from Market_State_Calculator import CriticalBotError

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Result of trade execution attempt."""
    ticker: str
    success: bool
    order_id: Optional[int]
    filled_qty: int
    avg_fill_price: float
    commission: float
    timestamp: datetime
    error_msg: Optional[str]

@dataclass
class PositionData:
    """Current position data from IBKR."""
    ticker: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    contract: Contract

@dataclass
class AccountSummary:
    """IBKR account summary data."""
    total_cash_value: float
    net_liquidation: float
    buying_power: float
    gross_position_value: float
    unrealized_pnl: float
    realized_pnl: float
    account_type: str

class IBKRExecutor:
    """Executes trades through Interactive Brokers API."""
    
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,  # 7497 for paper, 7496 for live
        client_id: int = 1,
        account: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        self.ib = IB()
        self.connected = False
        self.active_orders = {}
        self.execution_results = []
        self._last_account_sync = None
        self._account_cache = None
        self._positions_cache = None
        
    def connect(self) -> bool:
        """Establish connection to IBKR TWS/Gateway."""
        try:
            if self.connected:
                logger.info("Already connected to IBKR")
                return True
                
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            
            # Verify account access
            accounts = self.ib.managedAccounts()
            if not accounts:
                logger.error(f"FAIL-FAST: connect() - No accounts accessible through IBKR connection on {self.host}:{self.port}")
                raise CriticalBotError("No accounts accessible through IBKR connection - trading cannot proceed")
            
            if self.account is None:
                self.account = accounts[0]
                logger.info(f"Using account: {self.account}")
            
            # Initial account sync
            self.sync_account_data()
            
            logger.info(f"Connected to IBKR on {self.host}:{self.port}")
            return True
            
        except ConnectionError as e:
            logger.error(f"FAIL-FAST: connect() - Connection error to IBKR {self.host}:{self.port}: {e}")
            self.connected = False
            raise CriticalBotError(f"IBKR connection failed - cannot establish network connection: {e}")
        except TimeoutError as e:
            logger.error(f"FAIL-FAST: connect() - Timeout connecting to IBKR {self.host}:{self.port}: {e}")
            self.connected = False
            raise CriticalBotError(f"IBKR connection timeout - TWS/Gateway may not be running: {e}")
        except Exception as e:
            logger.error(f"FAIL-FAST: connect() - Unexpected error connecting to IBKR {self.host}:{self.port}: {type(e).__name__}: {e}")
            self.connected = False
            raise CriticalBotError(f"IBKR connection failed with unexpected error: {e}")
    
    def disconnect(self):
        """Disconnect from IBKR."""
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from IBKR")
        except Exception as e:
            logger.error(f"FAIL-FAST: disconnect() - Error disconnecting from IBKR: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Critical error during IBKR disconnection: {e}")
    
    def sync_account_data(self, force_refresh: bool = False) -> None:
        """Sync account data from IBKR. Cache for 30 seconds unless forced."""
        try:
            if not self.connected:
                logger.error(f"FAIL-FAST: sync_account_data() - Not connected to IBKR")
                raise CriticalBotError("Cannot sync account data - IBKR connection not established")
            
            now = datetime.now()
            if (not force_refresh and 
                self._last_account_sync and 
                (now - self._last_account_sync).seconds < 30):
                return  # Use cached data
            
            # Request account summary
            self.ib.reqAccountSummary()
            
            # Wait for data
            self.ib.sleep(2)
            
            # Update cache
            self._last_account_sync = now
            logger.debug("Account data synced from IBKR")
            
        except AttributeError as e:
            logger.error(f"FAIL-FAST: sync_account_data() - IBKR API attribute error: {e}")
            raise CriticalBotError(f"IBKR API attribute error during account sync: {e}")
        except RuntimeError as e:
            logger.error(f"FAIL-FAST: sync_account_data() - IBKR runtime error: {e}")
            raise CriticalBotError(f"IBKR runtime error during account sync: {e}")
        except Exception as e:
            logger.error(f"FAIL-FAST: sync_account_data() - Unexpected error during account sync: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Account sync failed with unexpected error: {e}")
    
    def get_account_value(self) -> float:
        """Get total portfolio value from IBKR. FIXES hard-coded 5000.0 capital."""
        try:
            self.sync_account_data()
            
            account_summary = self.ib.accountSummary(self.account)
            if account_summary is None:
                logger.error(f"FAIL-FAST: get_account_value() - accountSummary() returned None for account {self.account}")
                raise CriticalBotError("Account summary data is None - cannot determine portfolio value")
            
            net_liq = None
            
            for item in account_summary:
                if item is None:
                    logger.error(f"FAIL-FAST: get_account_value() - None item in account summary")
                    raise CriticalBotError("None item found in account summary data")
                    
                if item.tag == 'NetLiquidation' and item.currency == 'USD':
                    try:
                        net_liq = float(item.value)
                        break
                    except (ValueError, TypeError) as e:
                        logger.error(f"FAIL-FAST: get_account_value() - Cannot convert NetLiquidation value '{item.value}' to float: {e}")
                        raise CriticalBotError(f"Invalid NetLiquidation value: {item.value}")
            
            if net_liq is None:
                logger.error(f"FAIL-FAST: get_account_value() - NetLiquidation not found in account summary for account {self.account}")
                raise CriticalBotError("NetLiquidation value not found in account data - cannot determine portfolio value")
            
            if net_liq <= 0:
                logger.error(f"FAIL-FAST: get_account_value() - Invalid portfolio value: ${net_liq}")
                raise CriticalBotError(f"Invalid portfolio value: ${net_liq} - trading cannot proceed")
            
            logger.info(f"Account value: ${net_liq:,.2f}")
            return net_liq
            
        except CriticalBotError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"FAIL-FAST: get_account_value() - Unexpected error retrieving account value: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Account value retrieval failed: {e}")
    
    def get_buying_power(self) -> float:
        """Get available buying power from IBKR."""
        try:
            self.sync_account_data()
            
            account_summary = self.ib.accountSummary(self.account)
            if account_summary is None:
                logger.error(f"FAIL-FAST: get_buying_power() - accountSummary() returned None for account {self.account}")
                raise CriticalBotError("Account summary data is None - cannot determine buying power")
            
            buying_power = None
            
            for item in account_summary:
                if item is None:
                    logger.error(f"FAIL-FAST: get_buying_power() - None item in account summary")
                    raise CriticalBotError("None item found in account summary data")
                    
                if item.tag == 'BuyingPower' and item.currency == 'USD':
                    try:
                        buying_power = float(item.value)
                        break
                    except (ValueError, TypeError) as e:
                        logger.error(f"FAIL-FAST: get_buying_power() - Cannot convert BuyingPower value '{item.value}' to float: {e}")
                        raise CriticalBotError(f"Invalid BuyingPower value: {item.value}")
            
            if buying_power is None:
                # Fallback to available funds
                for item in account_summary:
                    if item.tag == 'AvailableFunds' and item.currency == 'USD':
                        try:
                            buying_power = float(item.value)
                            break
                        except (ValueError, TypeError) as e:
                            logger.error(f"FAIL-FAST: get_buying_power() - Cannot convert AvailableFunds value '{item.value}' to float: {e}")
                            raise CriticalBotError(f"Invalid AvailableFunds value: {item.value}")
            
            if buying_power is None:
                logger.error(f"FAIL-FAST: get_buying_power() - BuyingPower and AvailableFunds not found for account {self.account}")
                raise CriticalBotError("Buying power data not found in account summary")
            
            if buying_power < 0:
                logger.error(f"FAIL-FAST: get_buying_power() - Negative buying power: ${buying_power}")
                raise CriticalBotError(f"Negative buying power: ${buying_power} - account may be restricted")
            
            logger.info(f"Buying power: ${buying_power:,.2f}")
            return buying_power
            
        except CriticalBotError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"FAIL-FAST: get_buying_power() - Unexpected error retrieving buying power: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Buying power retrieval failed: {e}")
    
    def get_current_positions(self) -> Dict[str, PositionData]:
        """Get all current positions from IBKR."""
        try:
            self.sync_account_data()
            
            positions = self.ib.positions(self.account)
            if positions is None:
                logger.error(f"FAIL-FAST: get_current_positions() - positions() returned None for account {self.account}")
                raise CriticalBotError("Positions data is None - cannot retrieve current holdings")
            
            position_data = {}
            
            for pos in positions:
                if pos is None:
                    logger.error(f"FAIL-FAST: get_current_positions() - None position object in positions list")
                    raise CriticalBotError("None position object found in positions data")
                
                if pos.position != 0:  # Only active positions
                    if pos.contract is None:
                        logger.error(f"FAIL-FAST: get_current_positions() - Position contract is None")
                        raise CriticalBotError("Position contract is None - cannot identify position")
                    
                    if pos.contract.symbol is None:
                        logger.error(f"FAIL-FAST: get_current_positions() - Position symbol is None")
                        raise CriticalBotError("Position symbol is None - cannot identify ticker")
                    
                    ticker = pos.contract.symbol
                    
                    try:
                        quantity = int(pos.position)
                        avg_cost = float(pos.avgCost) if pos.avgCost is not None else 0.0
                        market_value = float(pos.marketValue) if pos.marketValue is not None else 0.0
                        unrealized_pnl = float(pos.unrealizedPNL) if pos.unrealizedPNL is not None else 0.0
                    except (ValueError, TypeError) as e:
                        logger.error(f"FAIL-FAST: get_current_positions() - Cannot convert position data for {ticker}: {e}")
                        raise CriticalBotError(f"Invalid position data for {ticker}: {e}")
                    
                    position_data[ticker] = PositionData(
                        ticker=ticker,
                        quantity=quantity,
                        avg_cost=avg_cost,
                        market_value=market_value,
                        unrealized_pnl=unrealized_pnl,
                        contract=pos.contract
                    )
            
            logger.info(f"Found {len(position_data)} active positions")
            return position_data
            
        except CriticalBotError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"FAIL-FAST: get_current_positions() - Unexpected error retrieving positions: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Position retrieval failed: {e}")
    
    def get_account_summary(self) -> AccountSummary:
        """Get comprehensive account summary."""
        try:
            self.sync_account_data()
            
            account_summary = self.ib.accountSummary(self.account)
            if account_summary is None:
                logger.error(f"FAIL-FAST: get_account_summary() - accountSummary() returned None for account {self.account}")
                raise CriticalBotError("Account summary data is None")
            
            summary_data = {}
            
            for item in account_summary:
                if item is None:
                    logger.error(f"FAIL-FAST: get_account_summary() - None item in account summary")
                    raise CriticalBotError("None item found in account summary data")
                
                if item.currency == 'USD':
                    try:
                        summary_data[item.tag] = float(item.value)
                    except (ValueError, TypeError) as e:
                        logger.error(f"FAIL-FAST: get_account_summary() - Cannot convert {item.tag} value '{item.value}' to float: {e}")
                        raise CriticalBotError(f"Invalid account summary value for {item.tag}: {item.value}")
            
            # Determine account type
            account_type = "Paper" if self.port == 7497 else "Live"
            
            # Validate required fields exist
            required_fields = ['NetLiquidation']
            for field in required_fields:
                if field not in summary_data:
                    logger.error(f"FAIL-FAST: get_account_summary() - Required field '{field}' missing from account data")
                    raise CriticalBotError(f"Required account field '{field}' not found")
            
            return AccountSummary(
                total_cash_value=summary_data.get('TotalCashValue', 0.0),
                net_liquidation=summary_data.get('NetLiquidation', 0.0),
                buying_power=summary_data.get('BuyingPower', 0.0),
                gross_position_value=summary_data.get('GrossPositionValue', 0.0),
                unrealized_pnl=summary_data.get('UnrealizedPnL', 0.0),
                realized_pnl=summary_data.get('RealizedPnL', 0.0),
                account_type=account_type
            )
            
        except CriticalBotError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"FAIL-FAST: get_account_summary() - Unexpected error creating account summary: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Account summary creation failed: {e}")
    
    def validate_paper_account(self) -> bool:
        """Validate we're connected to paper trading account."""
        try:
            if self.port != 7497:
                logger.error(f"FAIL-FAST: validate_paper_account() - Not connected to paper trading port. Current port: {self.port}, expected: 7497")
                raise CriticalBotError(f"Not connected to paper trading port - current port {self.port} is not 7497")
            
            account_summary = self.get_account_summary()
            if account_summary.account_type != "Paper":
                logger.error(f"FAIL-FAST: validate_paper_account() - Expected paper account but detected {account_summary.account_type} account")
                raise CriticalBotError(f"Expected paper account but connected to {account_summary.account_type} account")
            
            logger.info("Confirmed paper trading account")
            return True
            
        except CriticalBotError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"FAIL-FAST: validate_paper_account() - Unexpected error during paper account validation: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Paper account validation failed: {e}")
    
    def execute_trade(self, position: PortfolioPosition) -> ExecutionResult:
        """Execute a single trade based on portfolio position."""
        try:
            if not self.connected:
                logger.error(f"FAIL-FAST: execute_trade() - Not connected to IBKR")
                raise CriticalBotError("Cannot execute trade - IBKR connection not established")
            
            if position is None:
                logger.error(f"FAIL-FAST: execute_trade() - Position is None")
                raise CriticalBotError("Cannot execute trade - position is None")
            
            if position.ticker is None or position.ticker == "":
                logger.error(f"FAIL-FAST: execute_trade() - Invalid ticker: {position.ticker}")
                raise CriticalBotError(f"Cannot execute trade - invalid ticker: {position.ticker}")
            
            if position.position_size == 0:
                logger.error(f"FAIL-FAST: execute_trade() - Zero position size for {position.ticker}")
                raise CriticalBotError(f"Cannot execute trade - zero position size for {position.ticker}")
            
            if position.entry_price is None or position.entry_price <= 0:
                logger.error(f"FAIL-FAST: execute_trade() - Invalid entry price for {position.ticker}: {position.entry_price}")
                raise CriticalBotError(f"Cannot execute trade - invalid entry price for {position.ticker}: {position.entry_price}")
            
            # Create contract
            contract = Stock(position.ticker, 'SMART', 'USD')
            qualified_contracts = self.ib.qualifyContracts(contract)
            
            if not qualified_contracts:
                logger.error(f"FAIL-FAST: execute_trade() - Contract qualification failed for {position.ticker}")
                raise CriticalBotError(f"Contract qualification failed for {position.ticker}")
            
            contract = qualified_contracts[0]
            
            # Determine order type and quantity
            action = 'BUY' if position.position_size > 0 else 'SELL'
            quantity = abs(position.position_size)
            
            # Create limit order
            order = LimitOrder(action, quantity, position.entry_price)
            
            # Submit order
            trade = self.ib.placeOrder(contract, order)
            if trade is None:
                logger.error(f"FAIL-FAST: execute_trade() - placeOrder() returned None for {position.ticker}")
                raise CriticalBotError(f"Order placement failed for {position.ticker}")
            
            # Track active order
            if trade.order is None or trade.order.orderId is None:
                logger.error(f"FAIL-FAST: execute_trade() - Trade object missing order or orderId for {position.ticker}")
                raise CriticalBotError(f"Invalid trade object for {position.ticker}")
            
            self.active_orders[trade.order.orderId] = trade
            
            # Wait for fill or timeout
            timeout = 30  # seconds
            start_time = time.time()
            
            while not trade.isDone() and (time.time() - start_time) < timeout:
                self.ib.sleep(1)
            
            # Check result
            if trade.isDone() and trade.orderStatus.status == 'Filled':
                fill = trade.fills[-1] if trade.fills else None
                avg_price = fill.execution.price if fill else position.entry_price
                commission = sum(f.commissionReport.commission for f in trade.fills if f.commissionReport)
                
                result = ExecutionResult(
                    ticker=position.ticker,
                    success=True,
                    order_id=trade.order.orderId,
                    filled_qty=quantity,
                    avg_fill_price=avg_price,
                    commission=commission,
                    timestamp=datetime.now(),
                    error_msg=None
                )
                
                logger.info(f"Trade executed: {action} {quantity} {position.ticker} @ ${avg_price:.2f}")
                
            else:
                # Cancel unfilled order
                self.ib.cancelOrder(order)
                
                result = ExecutionResult(
                    ticker=position.ticker,
                    success=False,
                    order_id=trade.order.orderId if trade.order else None,
                    filled_qty=0,
                    avg_fill_price=0.0,
                    commission=0.0,
                    timestamp=datetime.now(),
                    error_msg=f"Order not filled within {timeout}s"
                )
                
                logger.warning(f"Trade failed: {position.ticker} - {result.error_msg}")
            
            # Remove from active orders
            if trade.order.orderId in self.active_orders:
                del self.active_orders[trade.order.orderId]
            
            self.execution_results.append(result)
            return result
            
        except CriticalBotError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"FAIL-FAST: execute_trade() - Unexpected error executing trade for {position.ticker if position else 'unknown'}: {type(e).__name__}: {e}")
            
            error_result = ExecutionResult(
                ticker=position.ticker if position else "unknown",
                success=False,
                order_id=None,
                filled_qty=0,
                avg_fill_price=0.0,
                commission=0.0,
                timestamp=datetime.now(),
                error_msg=str(e)
            )
            
            self.execution_results.append(error_result)
            raise CriticalBotError(f"Trade execution failed for {position.ticker if position else 'unknown'}: {e}")
    
    def get_market_data(self, ticker: str) -> Optional[float]:
        """Get current market price for ticker."""
        try:
            if ticker is None or ticker == "":
                logger.error(f"FAIL-FAST: get_market_data() - Invalid ticker: {ticker}")
                raise CriticalBotError(f"Cannot get market data - invalid ticker: {ticker}")
            
            contract = Stock(ticker, 'SMART', 'USD')
            qualified_contracts = self.ib.qualifyContracts(contract)
            
            if not qualified_contracts:
                logger.error(f"FAIL-FAST: get_market_data() - Contract qualification failed for {ticker}")
                raise CriticalBotError(f"Contract qualification failed for {ticker}")
            
            contract = qualified_contracts[0]
            
            ticker_data = self.ib.reqMktData(contract, '', False, False)
            if ticker_data is None:
                logger.error(f"FAIL-FAST: get_market_data() - reqMktData() returned None for {ticker}")
                raise CriticalBotError(f"Market data request failed for {ticker}")
            
            self.ib.sleep(2)  # Wait for data
            
            price = None
            if ticker_data.last and ticker_data.last > 0:
                try:
                    price = float(ticker_data.last)
                except (ValueError, TypeError) as e:
                    logger.error(f"FAIL-FAST: get_market_data() - Cannot convert last price '{ticker_data.last}' to float for {ticker}: {e}")
                    raise CriticalBotError(f"Invalid last price for {ticker}: {ticker_data.last}")
            elif ticker_data.close and ticker_data.close > 0:
                try:
                    price = float(ticker_data.close)
                except (ValueError, TypeError) as e:
                    logger.error(f"FAIL-FAST: get_market_data() - Cannot convert close price '{ticker_data.close}' to float for {ticker}: {e}")
                    raise CriticalBotError(f"Invalid close price for {ticker}: {ticker_data.close}")
            
            if price is None or price <= 0:
                logger.error(f"FAIL-FAST: get_market_data() - No valid market data available for {ticker}")
                raise CriticalBotError(f"No valid market data available for {ticker}")
            
            return price
                
        except CriticalBotError:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            logger.error(f"FAIL-FAST: get_market_data() - Unexpected error getting market data for {ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Market data retrieval failed for {ticker}: {e}")
