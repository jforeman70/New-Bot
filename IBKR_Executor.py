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
        
    def connect(self) -> bool:
        """Establish connection to IBKR TWS/Gateway."""
        try:
            if self.connected:
                logger.info("Already connected to IBKR")
                return True
                
            logger.info(f"Connecting to IBKR at {self.host}:{self.port}")
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            
            # Verify connection
            if not self.ib.isConnected():
                logger.error(f"connect(): Failed to establish IBKR connection to {self.host}:{self.port}")
                raise CriticalBotError("Failed to establish IBKR connection")
            
            self.connected = True
            managed_accounts = self.ib.managedAccounts()
            if not managed_accounts:
                logger.error("connect(): No managed accounts found after connection")
                raise CriticalBotError("No managed accounts available")
                
            logger.info(f"Connected to IBKR. Account: {managed_accounts}")
            
            # Set account if not specified
            if not self.account and managed_accounts:
                self.account = managed_accounts[0]
                logger.info(f"Using account: {self.account}")
            
            return True
            
        except Exception as e:
            logger.error(f"connect(): IBKR connection failed with error: {e}")
            self.connected = False
            raise CriticalBotError(f"Cannot connect to IBKR: {e}")
    
    def execute_positions(
        self,
        positions: List[PortfolioPosition],
        use_options: bool = False,
        max_slippage: float = 0.01  # 1% max slippage
    ) -> List[ExecutionResult]:
        """
        Execute portfolio positions with smart order routing.
        
        Args:
            positions: List of positions to execute
            use_options: Whether to use options for leverage
            max_slippage: Maximum acceptable slippage
            
        Returns:
            List of execution results
        """
        try:
            if not positions:
                logger.error("execute_positions(): No positions provided")
                raise CriticalBotError("Cannot execute empty positions list")
                
            if not self.connected:
                self.connect()
            
            results = []
            
            for position in positions:
                try:
                    if not position or not hasattr(position, 'ticker') or not position.ticker:
                        logger.error(f"execute_positions(): Invalid position object: {position}")
                        raise CriticalBotError("Invalid position object - missing ticker")
                        
                    logger.info(f"Executing {position.ticker}: {position.position_size} shares")
                    
                    if use_options and position.confidence > 0.75:
                        # Use options for high confidence trades
                        result = self._execute_option_position(position)
                    else:
                        # Standard stock execution
                        result = self._execute_stock_position(position, max_slippage)
                    
                    results.append(result)
                    
                    # Brief pause to avoid order flooding
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"execute_positions(): Failed to execute {position.ticker}: {e}")
                    raise CriticalBotError(f"Position execution failed for {position.ticker}: {e}")
            
            self.execution_results.extend(results)
            return results
            
        except Exception as e:
            logger.error(f"execute_positions(): Critical failure in position execution: {e}")
            raise CriticalBotError(f"Execute positions failed: {e}")
    
    def _execute_stock_position(
        self,
        position: PortfolioPosition,
        max_slippage: float
    ) -> ExecutionResult:
        """Execute a stock position with bracket orders."""
        try:
            # Create contract
            if not position.ticker:
                logger.error("_execute_stock_position(): Position ticker is None or empty")
                raise CriticalBotError("Invalid ticker for stock position")
                
            contract = Stock(position.ticker, 'SMART', 'USD')
            
            # Qualify contract to ensure it's valid
            qualified_contracts = self.ib.qualifyContracts(contract)
            if not qualified_contracts:
                logger.error(f"_execute_stock_position(): Failed to qualify contract for {position.ticker}")
                raise CriticalBotError(f"Cannot qualify contract for {position.ticker}")
            
            # Get current market price for slippage check
            ticker_data = self.ib.reqMktData(contract)
            if not ticker_data:
                logger.error(f"_execute_stock_position(): Failed to get market data for {position.ticker}")
                raise CriticalBotError(f"Cannot get market data for {position.ticker}")
                
            time.sleep(1)  # Wait for data to populate
            
            current_price = ticker_data.marketPrice
            if not current_price or current_price <= 0:
                current_price = ticker_data.last or ticker_data.close or position.entry_price
                if not current_price or current_price <= 0:
                    logger.error(f"_execute_stock_position(): No valid price data for {position.ticker}")
                    raise CriticalBotError(f"Cannot get valid price for {position.ticker}")
            
            # Check slippage
            if not position.entry_price or position.entry_price <= 0:
                logger.error(f"_execute_stock_position(): Invalid entry price for {position.ticker}: {position.entry_price}")
                raise CriticalBotError(f"Invalid entry price for {position.ticker}")
                
            slippage = abs(current_price - position.entry_price) / position.entry_price
            if slippage > max_slippage:
                logger.warning(f"{position.ticker}: Slippage {slippage:.2%} exceeds max {max_slippage:.2%}")
                if position.confidence < 0.8:  # Skip if not high confidence
                    logger.error(f"_execute_stock_position(): Excessive slippage for low confidence trade: {slippage:.2%}")
                    raise CriticalBotError(f"Excessive slippage: {slippage:.2%}")
            
            # Create bracket order (entry + target + stop)
            entry_order = LimitOrder(
                action='BUY',
                totalQuantity=position.position_size,
                lmtPrice=round(min(current_price * 1.001, position.entry_price), 2),
                tif='GTC',  # Good til cancelled
                outsideRth=True,  # Allow outside regular trading hours
                account=self.account
            )
            
            # Place main order
            trade = self.ib.placeOrder(contract, entry_order)
            if not trade:
                logger.error(f"_execute_stock_position(): Failed to place order for {position.ticker}")
                raise CriticalBotError(f"Failed to place order for {position.ticker}")
            
            # Wait for fill or timeout
            timeout = 30  # seconds
            start_time = time.time()
            
            while not trade.isDone() and (time.time() - start_time) < timeout:
                time.sleep(1)
            
            if trade.orderStatus.status == 'Filled':
                # Place bracket orders
                self._place_bracket_orders(contract, position, trade)
                
                return ExecutionResult(
                    ticker=position.ticker,
                    success=True,
                    order_id=trade.order.orderId,
                    filled_qty=trade.orderStatus.filled,
                    avg_fill_price=trade.orderStatus.avgFillPrice,
                    commission=trade.commission if trade.commission else 0,
                    timestamp=datetime.now(),
                    error_msg=None
                )
            else:
                # Cancel if not filled
                self.ib.cancelOrder(trade.order)
                logger.error(f"_execute_stock_position(): Order not filled within {timeout}s for {position.ticker}")
                raise CriticalBotError(f"Order not filled within {timeout}s")
                
        except Exception as e:
            logger.error(f"_execute_stock_position(): Stock execution failed for {position.ticker}: {e}")
            raise CriticalBotError(f"Stock execution failed for {position.ticker}: {e}")
    
    def _execute_option_position(self, position: PortfolioPosition) -> ExecutionResult:
        """Execute option position for leverage."""
        try:
            if not hasattr(position, 'chemistry_type') or not position.chemistry_type:
                logger.error(f"_execute_option_position(): Missing chemistry_type for {position.ticker}")
                raise CriticalBotError(f"Missing chemistry_type for option execution: {position.ticker}")
                
            # Determine option strategy based on chemistry
            if position.chemistry_type == 'volatile_compound':
                # Buy calls for growth stocks
                return self._buy_call_option(position)
            elif position.chemistry_type == 'noble_gas':
                # Buy puts for hedging
                return self._buy_put_option(position)
            else:
                # Default to stock
                return self._execute_stock_position(position, 0.01)
                
        except Exception as e:
            logger.error(f"_execute_option_position(): Option execution failed for {position.ticker}: {e}")
            raise CriticalBotError(f"Option execution failed for {position.ticker}: {e}")
    
    def _buy_call_option(self, position: PortfolioPosition) -> ExecutionResult:
        """Buy call options for bullish positions."""
        try:
            if not position.entry_price or position.entry_price <= 0:
                logger.error(f"_buy_call_option(): Invalid entry price for {position.ticker}: {position.entry_price}")
                raise CriticalBotError(f"Invalid entry price for call option: {position.ticker}")
                
            # Calculate option parameters
            strike = round(position.entry_price * 1.02)  # 2% OTM
            expiry = (datetime.now() + timedelta(days=7)).strftime('%Y%m%d')  # Weekly
            
            # Create option contract
            contract = Option(
                symbol=position.ticker,
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right='C',
                exchange='SMART'
            )
            
            # Qualify contract
            contracts = self.ib.qualifyContracts(contract)
            if not contracts:
                logger.error(f"_buy_call_option(): No valid option contract found for {position.ticker}")
                raise CriticalBotError(f"No valid option contract found for {position.ticker}")
            
            contract = contracts[0]
            
            # Calculate quantity (each option = 100 shares)
            if not position.position_size or position.position_size <= 0:
                logger.error(f"_buy_call_option(): Invalid position size for {position.ticker}: {position.position_size}")
                raise CriticalBotError(f"Invalid position size for call option: {position.ticker}")
                
            option_qty = max(1, position.position_size // 100)
            
            # Get market price for limit order
            ticker_data = self.ib.reqMktData(contract)
            if not ticker_data:
                logger.error(f"_buy_call_option(): Failed to get option market data for {position.ticker}")
                raise CriticalBotError(f"Cannot get option market data for {position.ticker}")
                
            time.sleep(1)
            
            limit_price = ticker_data.marketPrice or ticker_data.last or ticker_data.close
            if not limit_price:
                # Use market order if no price available
                order = MarketOrder(
                    action='BUY',
                    totalQuantity=option_qty,
                    account=self.account
                )
            else:
                # Place limit order with small buffer
                order = LimitOrder(
                    action='BUY',
                    totalQuantity=option_qty,
                    lmtPrice=round(limit_price * 1.05, 2),  # 5% buffer
                    account=self.account
                )
            
            trade = self.ib.placeOrder(contract, order)
            if not trade:
                logger.error(f"_buy_call_option(): Failed to place call option order for {position.ticker}")
                raise CriticalBotError(f"Failed to place call option order for {position.ticker}")
                
            time.sleep(5)  # Wait for execution
            
            return ExecutionResult(
                ticker=f"{position.ticker}_CALL",
                success=trade.orderStatus.status == 'Filled',
                order_id=trade.order.orderId,
                filled_qty=trade.orderStatus.filled,
                avg_fill_price=trade.orderStatus.avgFillPrice,
                commission=trade.commission if trade.commission else 0,
                timestamp=datetime.now(),
                error_msg=None if trade.isDone() else "Partial fill"
            )
            
        except Exception as e:
            logger.error(f"_buy_call_option(): Call option execution failed for {position.ticker}: {e}")
            raise CriticalBotError(f"Call option execution failed for {position.ticker}: {e}")
    
    def _buy_put_option(self, position: PortfolioPosition) -> ExecutionResult:
        """Buy put options for hedging."""
        try:
            if not position.stop_loss or position.stop_loss <= 0:
                logger.error(f"_buy_put_option(): Invalid stop loss for {position.ticker}: {position.stop_loss}")
                raise CriticalBotError(f"Invalid stop loss for put option: {position.ticker}")
                
            strike = round(position.stop_loss)
            expiry = (datetime.now() + timedelta(days=30)).strftime('%Y%m%d')
            
            contract = Option(
                symbol=position.ticker,
                lastTradeDateOrContractMonth=expiry,
                strike=strike,
                right='P',
                exchange='SMART'
            )
            
            contracts = self.ib.qualifyContracts(contract)
            if not contracts:
                logger.error(f"_buy_put_option(): No valid put contract found for {position.ticker}")
                raise CriticalBotError(f"No valid put contract found for {position.ticker}")
            
            contract = contracts[0]
            
            if not position.position_size or position.position_size <= 0:
                logger.error(f"_buy_put_option(): Invalid position size for {position.ticker}: {position.position_size}")
                raise CriticalBotError(f"Invalid position size for put option: {position.ticker}")
                
            option_qty = max(1, position.position_size // 100)
            
            order = MarketOrder(
                action='BUY', 
                totalQuantity=option_qty,
                account=self.account
            )
            
            trade = self.ib.placeOrder(contract, order)
            if not trade:
                logger.error(f"_buy_put_option(): Failed to place put option order for {position.ticker}")
                raise CriticalBotError(f"Failed to place put option order for {position.ticker}")
                
            time.sleep(3)  # Wait for execution
            
            return ExecutionResult(
                ticker=f"{position.ticker}_PUT",
                success=trade.isDone(),
                order_id=trade.order.orderId,
                filled_qty=trade.orderStatus.filled,
                avg_fill_price=trade.orderStatus.avgFillPrice,
                commission=trade.commission if trade.commission else 0,
                timestamp=datetime.now(),
                error_msg=None
            )
            
        except Exception as e:
            logger.error(f"_buy_put_option(): Put option execution failed for {position.ticker}: {e}")
            raise CriticalBotError(f"Put option execution failed for {position.ticker}: {e}")
    
    def _place_bracket_orders(
        self,
        contract: Contract,
        position: PortfolioPosition,
        parent_trade: Trade
    ):
        """Place target and stop loss orders."""
        try:
            if not position.target_price or position.target_price <= 0:
                logger.error(f"_place_bracket_orders(): Invalid target price for {position.ticker}: {position.target_price}")
                raise CriticalBotError(f"Invalid target price for bracket orders: {position.ticker}")
                
            if not position.stop_loss or position.stop_loss <= 0:
                logger.error(f"_place_bracket_orders(): Invalid stop loss for {position.ticker}: {position.stop_loss}")
                raise CriticalBotError(f"Invalid stop loss for bracket orders: {position.ticker}")
                
            if not parent_trade or not parent_trade.order or not parent_trade.order.orderId:
                logger.error(f"_place_bracket_orders(): Invalid parent trade for {position.ticker}")
                raise CriticalBotError(f"Invalid parent trade for bracket orders: {position.ticker}")
            
            # Target order
            target_order = LimitOrder(
                action='SELL',
                totalQuantity=position.position_size,
                lmtPrice=round(position.target_price, 2),
                parentId=parent_trade.order.orderId,
                tif='GTC',
                account=self.account
            )
            
            # Stop loss order
            stop_order = StopOrder(
                action='SELL',
                totalQuantity=position.position_size,
                stopPrice=round(position.stop_loss, 2),
                parentId=parent_trade.order.orderId,
                tif='GTC',
                account=self.account
            )
            
            # Place orders
            target_trade = self.ib.placeOrder(contract, target_order)
            if not target_trade:
                logger.error(f"_place_bracket_orders(): Failed to place target order for {position.ticker}")
                raise CriticalBotError(f"Failed to place target order for {position.ticker}")
                
            stop_trade = self.ib.placeOrder(contract, stop_order)
            if not stop_trade:
                logger.error(f"_place_bracket_orders(): Failed to place stop order for {position.ticker}")
                raise CriticalBotError(f"Failed to place stop order for {position.ticker}")
            
            logger.info(f"Bracket orders placed for {position.ticker}: "
                       f"Target=${position.target_price:.2f}, Stop=${position.stop_loss:.2f}")
                       
        except Exception as e:
            logger.error(f"_place_bracket_orders(): Failed to place bracket orders for {position.ticker}: {e}")
            raise CriticalBotError(f"Failed to place bracket orders for {position.ticker}: {e}")
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        try:
            if not self.connected:
                self.connect()
                
            positions = self.ib.positions(account=self.account)
            if positions is None:
                logger.error("get_positions(): IBKR returned None for positions")
                raise CriticalBotError("Cannot retrieve positions from IBKR")
                
            return [{
                'symbol': pos.contract.symbol,
                'quantity': pos.position,
                'avg_cost': pos.avgCost,
                'market_value': pos.marketValue,
                'unrealized_pnl': pos.unrealizedPNL
            } for pos in positions]
            
        except Exception as e:
            logger.error(f"get_positions(): Failed to get positions: {e}")
            raise CriticalBotError(f"Failed to get positions: {e}")
    
    def disconnect(self):
        """Disconnect from IBKR."""
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from IBKR")
        except Exception as e:
            logger.error(f"disconnect(): Failed to disconnect from IBKR: {e}")
            raise CriticalBotError(f"Failed to disconnect from IBKR: {e}")