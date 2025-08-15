"""
Risk Manager for Catalyst Trading System

Geological Risk Philosophy:
- Positions are geological formations that must withstand market pressure
- Portfolio heat represents tectonic stress that can cause system failure
- Correlation limits prevent cascade failures like geological fault lines
- Dynamic stops adjust based on formation stability (confidence)

Author: Catalyst Trading System
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from Market_State_Calculator import CriticalBotError

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk assessment for current portfolio state."""
    portfolio_heat: float  # Current drawdown percentage
    position_count: int
    max_position_risk: float
    total_capital_at_risk: float
    correlation_violations: int
    risk_score: float  # 0-1, higher is riskier
    recommendations: List[str]

@dataclass
class PositionRisk:
    """Risk assessment for individual position."""
    ticker: str
    position_size: float
    entry_price: float
    current_price: Optional[float]
    stop_loss: float
    max_loss_pct: float
    correlation_risk: float
    confidence_adjusted_size: float
    approved: bool
    rejection_reason: Optional[str]

class RiskManager:
    """
    Geological Risk Management System
    
    Core Principles:
    1. Position limits: 5% max loss per trade
    2. Portfolio heat: 15% max drawdown
    3. Correlation limits: Max 0.7 between positions
    4. Stop losses: 2-5% based on confidence
    5. Size adjustment: Reduce after losses
    """
    
    def __init__(
        self,
        capital: float,
        max_position_risk: float = 0.05,  # 5% max loss per position
        max_portfolio_heat: float = 0.15,  # 15% max portfolio drawdown
        max_correlation: float = 0.7,      # Max correlation between positions
        heat_reduction_factor: float = 0.8,  # Reduce size after losses
        max_positions: int = 10            # Max positions from spec
    ):
        """Initialize risk management system."""
        try:
            if capital is None or not isinstance(capital, (int, float)) or capital <= 0:
                logger.error(f"RiskManager.__init__: Invalid capital value: {capital}")
                raise CriticalBotError(f"Capital must be positive number, got: {capital}")
                
            self.capital = float(capital)
            self.initial_capital = float(capital)
            self.max_position_risk = float(max_position_risk)
            self.max_portfolio_heat = float(max_portfolio_heat)
            self.max_correlation = float(max_correlation)
            self.heat_reduction_factor = float(heat_reduction_factor)
            self.max_positions = int(max_positions)
            
            # Risk state tracking
            self.current_heat = 0.0
            self.peak_capital = float(capital)
            self.consecutive_losses = 0
            self.last_loss_date = None
            
            # Performance tracking for risk adjustment
            self.recent_trades = []
            
            # Historical returns for correlation calculation
            self.return_history = defaultdict(list)
            
            logger.info(f"Risk Manager initialized - Capital: ${capital:,.0f}")
            
        except (ValueError, TypeError) as e:
            logger.error(f"RiskManager.__init__: Type conversion error: {e}")
            raise CriticalBotError(f"Failed to initialize RiskManager with invalid types: {e}")
        except Exception as e:
            logger.error(f"RiskManager.__init__: Unexpected error: {e}")
            raise CriticalBotError(f"Failed to initialize RiskManager: {e}")

    def validate_portfolio(
        self,
        proposed_positions: List,
        current_positions: List,
        market_state: Tuple[float, float],
        current_prices: Optional[Dict[str, float]] = None
    ) -> Tuple[List, RiskMetrics]:
        """Validate proposed portfolio against risk limits."""
        try:
            # Validate inputs - halt system on bad data
            if proposed_positions is None:
                logger.error("validate_portfolio: proposed_positions is None")
                raise CriticalBotError("proposed_positions cannot be None")
                
            if current_positions is None:
                logger.error("validate_portfolio: current_positions is None")
                raise CriticalBotError("current_positions cannot be None")
                
            if market_state is None:
                logger.error("validate_portfolio: market_state is None")
                raise CriticalBotError("market_state cannot be None")
                
            if not isinstance(market_state, (tuple, list)) or len(market_state) != 2:
                logger.error(f"validate_portfolio: Invalid market_state format: {market_state}")
                raise CriticalBotError("Market state must be tuple of (risk, momentum)")
        
            try:
                risk, momentum = float(market_state[0]), float(market_state[1])
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"validate_portfolio: Market state conversion error: {e}")
                raise CriticalBotError(f"Market state values must be numeric: {e}")
                
            if not (0 <= risk <= 1) or not (-1 <= momentum <= 1):
                logger.error(f"validate_portfolio: Market state out of range - risk: {risk}, momentum: {momentum}")
                raise CriticalBotError(f"Market state values out of range: risk={risk}, momentum={momentum}")
            
            # Update portfolio heat
            self._update_portfolio_heat(current_positions, current_prices)
            
            # Validate each position
            validated_positions = []
            approved_positions = []
            correlation_violations = 0
            
            for i, position in enumerate(proposed_positions):
                try:
                    if position is None:
                        logger.error(f"validate_portfolio: Position {i} is None")
                        raise CriticalBotError(f"Position {i} cannot be None")
                        
                    position_risk = self._validate_single_position(
                        position, current_positions, (risk, momentum), current_prices
                    )
                    validated_positions.append(position_risk)
                    
                    if position_risk.approved:
                        approved_position = self._create_approved_position(position, position_risk)
                        if approved_position:
                            approved_positions.append(approved_position)
                            
                            if position_risk.correlation_risk > self.max_correlation:
                                correlation_violations += 1
                                
                except Exception as e:
                    ticker = self._safe_get_attr(position, 'ticker', 'UNKNOWN')
                    logger.error(f"validate_portfolio: Error validating position {ticker}: {e}")
                    raise CriticalBotError(f"Failed to validate position {ticker}: {e}")
            
            # Calculate risk metrics
            try:
                risk_metrics = self._calculate_portfolio_risk(
                    validated_positions, correlation_violations, (risk, momentum)
                )
            except Exception as e:
                logger.error(f"validate_portfolio: Error calculating portfolio risk: {e}")
                raise CriticalBotError(f"Failed to calculate portfolio risk: {e}")
            
            return approved_positions, risk_metrics
            
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"validate_portfolio: Unexpected error: {e}")
            raise CriticalBotError(f"Unexpected error in portfolio validation: {e}")

    def _safe_get_attr(self, obj, attr: str, default=None):
        """Safely get attribute from object."""
        try:
            if obj is None:
                return default
            return getattr(obj, attr, default)
        except Exception as e:
            logger.error(f"_safe_get_attr: Error getting attribute {attr}: {e}")
            return default

    def _create_approved_position(self, position, position_risk: PositionRisk):
        """Create approved position with proper attribute handling."""
        try:
            if position is None:
                logger.error("_create_approved_position: position is None")
                raise CriticalBotError("Cannot create approved position from None")
                
            if position_risk is None:
                logger.error("_create_approved_position: position_risk is None")
                raise CriticalBotError("Cannot create approved position without risk data")
            
            # Import here to avoid circular imports
            try:
                from Portfolio_Synthesizer import PortfolioPosition
                
                return PortfolioPosition(
                    ticker=position_risk.ticker,
                    position_size=int(position_risk.confidence_adjusted_size),
                    entry_price=position_risk.entry_price,
                    target_price=self._safe_get_attr(position, 'target_price', position_risk.entry_price * 1.1),
                    stop_price=position_risk.stop_loss,
                    confidence=self._safe_get_attr(position, 'confidence', 0.5),
                    chemistry=self._safe_get_attr(position, 'chemistry', None)
                )
            except ImportError as e:
                logger.warning(f"_create_approved_position: Import error, using fallback: {e}")
                # Create dict-like object if import fails
                return {
                    'ticker': position_risk.ticker,
                    'position_size': int(position_risk.confidence_adjusted_size),
                    'entry_price': position_risk.entry_price,
                    'target_price': self._safe_get_attr(position, 'target_price', position_risk.entry_price * 1.1),
                    'stop_price': position_risk.stop_loss,
                    'confidence': self._safe_get_attr(position, 'confidence', 0.5),
                    'chemistry': self._safe_get_attr(position, 'chemistry', None)
                }
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"_create_approved_position: Unexpected error: {e}")
            raise CriticalBotError(f"Failed to create approved position: {e}")

    def _validate_single_position(
        self,
        position,
        current_positions: List,
        market_state: Tuple[float, float],
        current_prices: Optional[Dict[str, float]]
    ) -> PositionRisk:
        """Validate individual position."""
        try:
            if position is None:
                logger.error("_validate_single_position: position is None")
                raise CriticalBotError("Position cannot be None")
            
            # Extract and validate position attributes
            ticker = self._safe_get_attr(position, 'ticker', None)
            entry_price = self._safe_get_attr(position, 'entry_price', None)
            confidence = self._safe_get_attr(position, 'confidence', None)
            
            if not ticker or not isinstance(ticker, str):
                logger.error(f"_validate_single_position: Invalid ticker: {ticker}")
                return self._create_rejection("Invalid or missing ticker", ticker or "UNKNOWN")
            
            try:
                entry_price = float(entry_price) if entry_price is not None else None
                confidence = float(confidence) if confidence is not None else None
            except (ValueError, TypeError) as e:
                logger.error(f"_validate_single_position: Type conversion error for {ticker}: {e}")
                return self._create_rejection(f"Non-numeric price or confidence: {e}", ticker)
                
            if entry_price is None or entry_price <= 0:
                logger.error(f"_validate_single_position: Invalid entry_price for {ticker}: {entry_price}")
                return self._create_rejection("Invalid or missing entry price", ticker)
            if confidence is None or not (0 <= confidence <= 1):
                logger.error(f"_validate_single_position: Invalid confidence for {ticker}: {confidence}")
                return self._create_rejection("Invalid or missing confidence", ticker)
            
            if current_prices is not None and not isinstance(current_prices, dict):
                logger.error(f"_validate_single_position: current_prices not a dict: {type(current_prices)}")
                raise CriticalBotError("current_prices must be dict or None")
            
            current_price = current_prices.get(ticker, entry_price) if current_prices else entry_price
            
            # Stop loss: 2-5% based on confidence (from spec)
            stop_pct = 0.02 + (0.03 * (1 - confidence))
            stop_price = entry_price * (1 - stop_pct)
            
            # Calculate Kelly fraction
            target_price = self._safe_get_attr(position, 'target_price', entry_price * 1.1)
            kelly_fraction = self._calculate_kelly_fraction(confidence, entry_price, target_price)
            
            # Position sizing formula from spec
            spec_position_size_pct = min(
                0.15,  # Max 15%
                kelly_fraction * 0.25,  # 1/4 Kelly
                confidence * 0.10  # Confidence-based
            )
            
            # Calculate position size
            if self.capital <= 0:
                logger.error(f"_validate_single_position: Zero or negative capital: {self.capital}")
                return self._create_rejection("Zero or negative capital", ticker)
                
            spec_position_value = self.capital * spec_position_size_pct
            spec_position_size = spec_position_value / entry_price
            
            # Apply size reduction after losses
            size_multiplier = 1.0
            if self.consecutive_losses > 0:
                size_multiplier = self.heat_reduction_factor ** min(self.consecutive_losses, 5)
            
            adjusted_size = max(1, spec_position_size * size_multiplier)  # Minimum 1 share
            max_loss = adjusted_size * (entry_price - stop_price)
            max_loss_pct = max_loss / self.capital
            
            # Calculate correlation risk
            correlation_risk = self._calculate_correlation_risk(ticker, current_positions)
            
            # Validation checks
            approved = True
            rejection_reason = None
            
            if max_loss_pct > self.max_position_risk:
                approved = False
                rejection_reason = f"Exceeds {self.max_position_risk:.0%} max loss: {max_loss_pct:.1%}"
            elif self.current_heat + max_loss_pct > self.max_portfolio_heat:
                approved = False
                rejection_reason = f"Would exceed {self.max_portfolio_heat:.0%} portfolio heat"
            elif correlation_risk > self.max_correlation:
                approved = False
                rejection_reason = f"Correlation exceeds {self.max_correlation:.1f}: {correlation_risk:.2f}"
            elif len(current_positions) >= self.max_positions:
                approved = False
                rejection_reason = f"Maximum {self.max_positions} positions already held"
            elif adjusted_size < 1:
                approved = False
                rejection_reason = "Position size too small after risk adjustment"
            
            return PositionRisk(
                ticker=ticker,
                position_size=self._safe_get_attr(position, 'position_size', adjusted_size),
                entry_price=entry_price,
                current_price=current_price,
                stop_loss=stop_price,
                max_loss_pct=max_loss_pct,
                correlation_risk=correlation_risk,
                confidence_adjusted_size=adjusted_size,
                approved=approved,
                rejection_reason=rejection_reason
            )
            
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            ticker = self._safe_get_attr(position, 'ticker', 'UNKNOWN')
            logger.error(f"_validate_single_position: Unexpected error for {ticker}: {e}")
            raise CriticalBotError(f"Failed to validate position {ticker}: {e}")

    def _create_rejection(self, reason: str, ticker: str = "UNKNOWN") -> PositionRisk:
        """Helper to create rejection position risk."""
        try:
            if reason is None or not isinstance(reason, str):
                logger.error(f"_create_rejection: Invalid reason: {reason}")
                reason = "Invalid rejection reason"
                
            return PositionRisk(
                ticker=ticker,
                position_size=0,
                entry_price=0,
                current_price=None,
                stop_loss=0,
                max_loss_pct=0,
                correlation_risk=0,
                confidence_adjusted_size=0,
                approved=False,
                rejection_reason=reason
            )
        except Exception as e:
            logger.error(f"_create_rejection: Error creating rejection: {e}")
            raise CriticalBotError(f"Failed to create position rejection: {e}")

    def _calculate_kelly_fraction(self, confidence: float, entry_price: float, target_price: Union[float, None]) -> float:
        """Calculate Kelly criterion fraction based on actual trade parameters."""
        try:
            if confidence is None or entry_price is None:
                logger.error(f"_calculate_kelly_fraction: None values - confidence: {confidence}, entry_price: {entry_price}")
                raise CriticalBotError("Kelly calculation requires non-None confidence and entry_price")
                
            if target_price is None:
                target_price = entry_price * 1.1  # Default 10% target
                
            target_price = float(target_price)
            
            win_prob = confidence
            potential_gain = target_price - entry_price
            
            if potential_gain <= 0:
                return 0.01  # Minimum allocation for defensive positions
                
            # Calculate potential loss using stop loss
            stop_pct = 0.02 + (0.03 * (1 - confidence))
            potential_loss = entry_price * stop_pct
            
            if potential_loss <= 0:
                logger.warning("_calculate_kelly_fraction: Zero potential loss, using conservative fallback")
                return 0.05
                
            reward_to_risk_ratio = potential_gain / potential_loss
            
            # Kelly = (bp - q) / b where b = reward/risk ratio, p = win prob, q = loss prob
            kelly = (reward_to_risk_ratio * win_prob - (1 - win_prob)) / reward_to_risk_ratio
            
            # Cap Kelly at reasonable levels and ensure positive
            result = max(0.01, min(kelly, 0.25))
            
            if np.isnan(result) or np.isinf(result):
                logger.error(f"_calculate_kelly_fraction: Invalid result: {result}")
                raise CriticalBotError("Kelly calculation produced invalid result")
                
            return result
            
        except (ValueError, ZeroDivisionError, TypeError) as e:
            logger.error(f"_calculate_kelly_fraction: Calculation error: {e}")
            raise CriticalBotError(f"Kelly fraction calculation failed: {e}")
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"_calculate_kelly_fraction: Unexpected error: {e}")
            raise CriticalBotError(f"Unexpected error in Kelly calculation: {e}")

    def _calculate_correlation_risk(self, ticker: str, current_positions: List) -> float:
        """Calculate correlation risk with existing positions."""
        try:
            if ticker is None or not isinstance(ticker, str):
                logger.error(f"_calculate_correlation_risk: Invalid ticker: {ticker}")
                raise CriticalBotError("Ticker must be valid string")
                
            if current_positions is None:
                logger.error("_calculate_correlation_risk: current_positions is None")
                raise CriticalBotError("current_positions cannot be None")
                
            if not current_positions:
                return 0.0
            
            max_correlation = 0.0
            
            for i, pos in enumerate(current_positions):
                try:
                    if pos is None:
                        logger.warning(f"_calculate_correlation_risk: Position {i} is None, skipping")
                        continue
                        
                    existing_ticker = self._safe_get_attr(pos, 'ticker', None)
                    if not existing_ticker or existing_ticker == ticker:
                        continue
                        
                    correlation = self._estimate_correlation(ticker, existing_ticker)
                    if correlation is None or np.isnan(correlation) or np.isinf(correlation):
                        logger.error(f"_calculate_correlation_risk: Invalid correlation: {correlation}")
                        raise CriticalBotError("Correlation calculation produced invalid result")
                        
                    max_correlation = max(max_correlation, correlation)
                    
                except Exception as e:
                    logger.error(f"_calculate_correlation_risk: Error processing position {i}: {e}")
                    raise CriticalBotError(f"Failed to calculate correlation for position {i}: {e}")
            
            return max_correlation
            
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"_calculate_correlation_risk: Unexpected error: {e}")
            raise CriticalBotError(f"Unexpected error in correlation calculation: {e}")

    def _estimate_correlation(self, ticker1: str, ticker2: str) -> float:
        """Estimate correlation between two tickers."""
        try:
            if not ticker1 or not ticker2 or not isinstance(ticker1, str) or not isinstance(ticker2, str):
                logger.error(f"_estimate_correlation: Invalid tickers: {ticker1}, {ticker2}")
                raise CriticalBotError("Both tickers must be valid strings")
                
            if ticker1 == ticker2:
                return 1.0
            
            # Use return history if available
            if (ticker1 in self.return_history and ticker2 in self.return_history and
                len(self.return_history[ticker1]) > 20 and len(self.return_history[ticker2]) > 20):
                
                try:
                    returns1 = np.array(self.return_history[ticker1][-50:])
                    returns2 = np.array(self.return_history[ticker2][-50:])
                    
                    if len(returns1) == 0 or len(returns2) == 0:
                        logger.warning("_estimate_correlation: Empty return arrays, using sector correlation")
                        return self._sector_correlation(ticker1, ticker2)
                    
                    min_len = min(len(returns1), len(returns2))
                    if min_len < 20:
                        return self._sector_correlation(ticker1, ticker2)
                    
                    # Align arrays
                    returns1 = returns1[-min_len:]
                    returns2 = returns2[-min_len:]
                    
                    # Calculate correlation
                    correlation_matrix = np.corrcoef(returns1, returns2)
                    
                    if correlation_matrix.shape != (2, 2):
                        logger.error(f"_estimate_correlation: Invalid correlation matrix shape: {correlation_matrix.shape}")
                        raise CriticalBotError("Correlation matrix has unexpected shape")
                        
                    correlation = correlation_matrix[0, 1]
                    
                    if np.isnan(correlation) or np.isinf(correlation):
                        logger.warning("_estimate_correlation: Invalid correlation, using sector fallback")
                        return self._sector_correlation(ticker1, ticker2)
                        
                    return abs(correlation)  # Use absolute value for risk calculation
                    
                except (IndexError, ValueError) as e:
                    logger.error(f"_estimate_correlation: Array operation error: {e}")
                    raise CriticalBotError(f"Correlation calculation failed: {e}")
            
            return self._sector_correlation(ticker1, ticker2)
            
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"_estimate_correlation: Unexpected error: {e}")
            raise CriticalBotError(f"Unexpected error in correlation estimation: {e}")

    def _sector_correlation(self, ticker1: str, ticker2: str) -> float:
        """Estimate correlation based on sector/asset type."""
        try:
            if not ticker1 or not ticker2:
                logger.error(f"_sector_correlation: Invalid tickers: {ticker1}, {ticker2}")
                raise CriticalBotError("Tickers cannot be empty for sector correlation")
                
            # Market ETFs - high correlation
            broad_market = {'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'}
            if ticker1 in broad_market and ticker2 in broad_market:
                return 0.85
            
            # Sector ETFs - moderate to high correlation with market
            if ticker1 in broad_market or ticker2 in broad_market:
                return 0.6
            
            # Individual stocks - moderate market correlation
            return 0.45
            
        except Exception as e:
            logger.error(f"_sector_correlation: Error: {e}")
            raise CriticalBotError(f"Sector correlation calculation failed: {e}")

    def _update_portfolio_heat(self, current_positions: List, current_prices: Optional[Dict[str, float]]):
        """Update current portfolio drawdown."""
        try:
            if current_positions is None:
                logger.error("_update_portfolio_heat: current_positions is None")
                raise CriticalBotError("current_positions cannot be None")
                
            if not current_positions or not current_prices:
                self.current_heat = 0.0
                return
            
            if not isinstance(current_prices, dict):
                logger.error(f"_update_portfolio_heat: current_prices not a dict: {type(current_prices)}")
                raise CriticalBotError("current_prices must be a dictionary")
            
            total_position_value = 0.0
            
            for i, position in enumerate(current_positions):
                try:
                    if position is None:
                        logger.warning(f"_update_portfolio_heat: Position {i} is None, skipping")
                        continue
                        
                    ticker = self._safe_get_attr(position, 'ticker', None)
                    position_size = self._safe_get_attr(position, 'position_size', 0)
                    
                    if not ticker or position_size == 0:
                        continue
                        
                    position_size = float(position_size)
                    current_price = current_prices.get(ticker, 0)
                    
                    if current_price is None:
                        logger.warning(f"_update_portfolio_heat: No price for {ticker}")
                        continue
                        
                    current_price = float(current_price)
                    
                    if current_price > 0:
                        position_value = abs(position_size) * current_price
                        total_position_value += position_value
                        
                except (ValueError, TypeError) as e:
                    logger.error(f"_update_portfolio_heat: Error processing position {i}: {e}")
                    raise CriticalBotError(f"Failed to process position {i}: {e}")
            
            # Current portfolio value is positions plus remaining cash
            position_cost = 0.0
            for position in current_positions:
                if position is None:
                    continue
                try:
                    size = self._safe_get_attr(position, 'position_size', 0)
                    price = self._safe_get_attr(position, 'entry_price', 0)
                    if size and price:
                        position_cost += abs(float(size)) * float(price)
                except (ValueError, TypeError):
                    continue
            
            remaining_cash = max(0, self.capital - position_cost)
            total_portfolio_value = total_position_value + remaining_cash
            
            # Update peak and calculate heat
            if total_portfolio_value > 0:
                self.peak_capital = max(self.peak_capital, total_portfolio_value)
                self.current_heat = max(0.0, (self.peak_capital - total_portfolio_value) / self.peak_capital)
            else:
                logger.error("_update_portfolio_heat: Total portfolio value is zero or negative")
                raise CriticalBotError("Portfolio value calculation resulted in non-positive value")
                
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"_update_portfolio_heat: Unexpected error: {e}")
            raise CriticalBotError(f"Failed to update portfolio heat: {e}")

    def _calculate_portfolio_risk(
        self,
        validated_positions: List[PositionRisk],
        correlation_violations: int,
        market_state: Tuple[float, float]
    ) -> RiskMetrics:
        """Calculate overall portfolio risk metrics."""
        try:
            if validated_positions is None:
                logger.error("_calculate_portfolio_risk: validated_positions is None")
                raise CriticalBotError("validated_positions cannot be None")
                
            if market_state is None or len(market_state) != 2:
                logger.error(f"_calculate_portfolio_risk: Invalid market_state: {market_state}")
                raise CriticalBotError("market_state must be tuple of length 2")
            
            approved_positions = [pos for pos in validated_positions if pos and pos.approved]
            
            if not approved_positions:
                return RiskMetrics(
                    portfolio_heat=self.current_heat,
                    position_count=0,
                    max_position_risk=0.0,
                    total_capital_at_risk=0.0,
                    correlation_violations=correlation_violations,
                    risk_score=min(1.0, self.current_heat * 2),
                    recommendations=["No approved positions"]
                )
            
            try:
                total_risk = sum(pos.max_loss_pct for pos in approved_positions if pos.max_loss_pct is not None)
                max_risk = max(pos.max_loss_pct for pos in approved_positions if pos.max_loss_pct is not None)
                
                if np.isnan(total_risk) or np.isinf(total_risk) or np.isnan(max_risk) or np.isinf(max_risk):
                    logger.error(f"_calculate_portfolio_risk: Invalid risk values - total: {total_risk}, max: {max_risk}")
                    raise CriticalBotError("Risk calculation produced invalid values")
                    
            except (ValueError, TypeError) as e:
                logger.error(f"_calculate_portfolio_risk: Risk calculation error: {e}")
                raise CriticalBotError(f"Failed to calculate position risks: {e}")
            
            # Risk score calculation
            heat_weight = 0.4 if self.current_heat > 0 else 0.2
            position_weight = 0.4
            market_weight = 0.2 if self.current_heat == 0 else 0.4
            
            heat_risk = self.current_heat / self.max_portfolio_heat if self.max_portfolio_heat > 0 else 0
            position_risk = total_risk / self.max_portfolio_heat if self.max_portfolio_heat > 0 else 0
            market_risk = market_state[0]
            
            risk_score = min(1.0, heat_risk * heat_weight + position_risk * position_weight + market_risk * market_weight)
            
            if np.isnan(risk_score) or np.isinf(risk_score):
                logger.error(f"_calculate_portfolio_risk: Invalid risk_score: {risk_score}")
                raise CriticalBotError("Risk score calculation produced invalid result")
            
            # Generate recommendations
            recommendations = []
            if self.current_heat > 0.08:
                recommendations.append(f"Portfolio heat: {self.current_heat:.1%}")
            if correlation_violations > 2:
                recommendations.append(f"High correlation: {correlation_violations} violations")
            if market_state[0] > 0.8:
                recommendations.append(f"Extreme market risk: {market_state[0]:.0%}")
            if total_risk > self.max_portfolio_heat * 0.75:
                recommendations.append(f"Total risk high: {total_risk:.1%}")
            if self.consecutive_losses > 3:
                recommendations.append(f"Consecutive losses: {self.consecutive_losses}")
            if len(approved_positions) >= self.max_positions * 0.8:
                recommendations.append(f"Near position limit: {len(approved_positions)}")
            
            return RiskMetrics(
                portfolio_heat=self.current_heat,
                position_count=len(approved_positions),
                max_position_risk=max_risk,
                total_capital_at_risk=total_risk,
                correlation_violations=correlation_violations,
                risk_score=risk_score,
                recommendations=recommendations
            )
            
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"_calculate_portfolio_risk: Unexpected error: {e}")
            raise CriticalBotError(f"Failed to calculate portfolio risk: {e}")

    def record_trade_result(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        trade_date: datetime
    ):
        """Record trade results for risk adjustment."""
        try:
            # Validate inputs
            if ticker is None or not isinstance(ticker, str) or not ticker.strip():
                logger.error(f"record_trade_result: Invalid ticker: {ticker}")
                raise CriticalBotError("Invalid ticker for trade record")
        
            try:
                entry_price = float(entry_price) if entry_price is not None else None
                exit_price = float(exit_price) if exit_price is not None else None
                position_size = float(position_size) if position_size is not None else None
            except (ValueError, TypeError) as e:
                logger.error(f"record_trade_result: Type conversion error: {e}")
                raise CriticalBotError(f"Trade prices and size must be numeric: {e}")
                
            if entry_price is None or exit_price is None or position_size is None:
                logger.error(f"record_trade_result: None values - entry: {entry_price}, exit: {exit_price}, size: {position_size}")
                raise CriticalBotError("Trade prices and position size cannot be None")
                
            if entry_price <= 0 or exit_price <= 0:
                logger.error(f"record_trade_result: Non-positive prices - entry: {entry_price}, exit: {exit_price}")
                raise CriticalBotError("Trade prices must be positive")
            if position_size == 0:
                logger.error("record_trade_result: Position size is zero")
                raise CriticalBotError("Position size cannot be zero")
            if trade_date is None or not isinstance(trade_date, datetime):
                logger.error(f"record_trade_result: Invalid trade_date: {trade_date}")
                raise CriticalBotError("Trade date must be datetime object")
        
            pnl = (exit_price - entry_price) * position_size
            return_pct = (exit_price - entry_price) / entry_price
            
            if np.isnan(pnl) or np.isinf(pnl) or np.isnan(return_pct) or np.isinf(return_pct):
                logger.error(f"record_trade_result: Invalid calculated values - pnl: {pnl}, return_pct: {return_pct}")
                raise CriticalBotError("Trade calculation produced invalid results")
            
            # Store return for correlation calculation
            if ticker not in self.return_history:
                self.return_history[ticker] = []
                
            self.return_history[ticker].append(return_pct)
            if len(self.return_history[ticker]) > 250:  # Keep ~1 year of data
                self.return_history[ticker] = self.return_history[ticker][-250:]
            
            trade_record = {
                'ticker': ticker,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'pnl': pnl,
                'return_pct': return_pct,
                'date': trade_date
            }
            
            self.recent_trades.append(trade_record)
            
            # Keep last 100 trades for analysis
            if len(self.recent_trades) > 100:
                self.recent_trades = self.recent_trades[-100:]
            
            # Update consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
                self.last_loss_date = trade_date
            else:
                self.consecutive_losses = 0
            
            # Update capital
            old_capital = self.capital
            self.capital += pnl
            
            if self.capital < 0:
                logger.error(f"record_trade_result: Capital went negative: {self.capital}")
                raise CriticalBotError("Capital became negative after trade")
            
            logger.info(f"Trade recorded: {ticker} P&L: ${pnl:.2f}, Capital: ${self.capital:.2f}")
            
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"record_trade_result: Unexpected error: {e}")
            raise CriticalBotError(f"Failed to record trade result: {e}")

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from recent trades."""
        try:
            if risk_free_rate is None:
                logger.error("calculate_sharpe_ratio: risk_free_rate is None")
                raise CriticalBotError("risk_free_rate cannot be None")
                
            risk_free_rate = float(risk_free_rate)
            
            if len(self.recent_trades) < 20:
                return 0.0
            
            try:
                returns = [trade['return_pct'] for trade in self.recent_trades if 'return_pct' in trade]
                
                if not returns:
                    logger.warning("calculate_sharpe_ratio: No returns found in recent trades")
                    return 0.0
                
                # Check for invalid returns
                valid_returns = []
                for ret in returns:
                    if ret is not None and not (np.isnan(ret) or np.isinf(ret)):
                        valid_returns.append(float(ret))
                
                if len(valid_returns) < 10:
                    logger.warning("calculate_sharpe_ratio: Insufficient valid returns")
                    return 0.0
                
                mean_return = np.mean(valid_returns)
                std_return = np.std(valid_returns, ddof=1) if len(valid_returns) > 1 else 0
                
                if np.isnan(mean_return) or np.isinf(mean_return) or np.isnan(std_return) or np.isinf(std_return):
                    logger.error(f"calculate_sharpe_ratio: Invalid statistics - mean: {mean_return}, std: {std_return}")
                    raise CriticalBotError("Sharpe calculation produced invalid statistics")
                
                if std_return == 0:
                    return 0.0
                
                # Annualize assuming ~250 trading days
                annual_return = mean_return * 250
                annual_std = std_return * np.sqrt(250)
                
                sharpe = (annual_return - risk_free_rate) / annual_std
                
                if np.isnan(sharpe) or np.isinf(sharpe):
                    logger.error(f"calculate_sharpe_ratio: Invalid Sharpe ratio: {sharpe}")
                    raise CriticalBotError("Sharpe ratio calculation produced invalid result")
                
                return sharpe
                
            except (ValueError, TypeError, ZeroDivisionError) as e:
                logger.error(f"calculate_sharpe_ratio: Calculation error: {e}")
                raise CriticalBotError(f"Sharpe ratio calculation failed: {e}")
            
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"calculate_sharpe_ratio: Unexpected error: {e}")
            raise CriticalBotError(f"Unexpected error in Sharpe calculation: {e}")

    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary."""
        try:
            if not self.recent_trades:
                return {
                    'current_heat': self.current_heat,
                    'peak_capital': self.peak_capital,
                    'current_capital': self.capital,
                    'total_return': 0.0,
                    'consecutive_losses': self.consecutive_losses,
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            try:
                recent_pnl = sum(trade['pnl'] for trade in self.recent_trades[-10:] if 'pnl' in trade and trade['pnl'] is not None)
                total_pnl = sum(trade['pnl'] for trade in self.recent_trades if 'pnl' in trade and trade['pnl'] is not None)
                winning_trades = len([t for t in self.recent_trades if 'pnl' in t and t['pnl'] is not None and t['pnl'] > 0])
                win_rate = winning_trades / len(self.recent_trades) if len(self.recent_trades) > 0 else 0.0
                
                if np.isnan(recent_pnl) or np.isinf(recent_pnl):
                    logger.error(f"get_risk_summary: Invalid recent_pnl: {recent_pnl}")
                    raise CriticalBotError("Recent P&L calculation produced invalid result")
                
                total_return = (self.capital - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0.0
                
                if np.isnan(total_return) or np.isinf(total_return):
                    logger.error(f"get_risk_summary: Invalid total_return: {total_return}")
                    raise CriticalBotError("Total return calculation produced invalid result")
                
                avg_trade_size = 0.0
                if self.recent_trades:
                    valid_sizes = []
                    for t in self.recent_trades:
                        if 'position_size' in t and 'entry_price' in t and t['position_size'] is not None and t['entry_price'] is not None:
                            try:
                                size = abs(float(t['position_size']) * float(t['entry_price']))
                                if not (np.isnan(size) or np.isinf(size)):
                                    valid_sizes.append(size)
                            except (ValueError, TypeError):
                                continue
                    
                    if valid_sizes:
                        avg_trade_size = np.mean(valid_sizes)
                        if np.isnan(avg_trade_size) or np.isinf(avg_trade_size):
                            avg_trade_size = 0.0
                
                return {
                    'current_heat': self.current_heat,
                    'peak_capital': self.peak_capital,
                    'current_capital': self.capital,
                    'total_return': total_return,
                    'total_pnl': total_pnl,
                    'consecutive_losses': self.consecutive_losses,
                    'max_position_risk': self.max_position_risk,
                    'max_portfolio_heat': self.max_portfolio_heat,
                    'recent_pnl': recent_pnl,
                    'win_rate': win_rate,
                    'sharpe_ratio': self.calculate_sharpe_ratio(),
                    'total_trades': len(self.recent_trades),
                    'avg_trade_size': avg_trade_size
                }
                
            except (ValueError, TypeError, ZeroDivisionError) as e:
                logger.error(f"get_risk_summary: Calculation error: {e}")
                raise CriticalBotError(f"Risk summary calculation failed: {e}")
            
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"get_risk_summary: Unexpected error: {e}")
            raise CriticalBotError(f"Failed to generate risk summary: {e}")

    def should_halt_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be halted due to risk."""
        try:
            if self.current_heat > self.max_portfolio_heat:
                return True, f"Portfolio heat exceeded {self.max_portfolio_heat:.0%}: {self.current_heat:.1%}"
            
            if self.capital <= self.initial_capital * 0.4:  # 60% loss threshold
                return True, f"Capital critically low: ${self.capital:.0f}"
            
            if self.consecutive_losses > 8:
                return True, f"Excessive consecutive losses: {self.consecutive_losses}"
            
            # Check for rapid capital loss in recent trades
            if len(self.recent_trades) >= 10:
                try:
                    recent_pnl = sum(t['pnl'] for t in self.recent_trades[-10:] if 'pnl' in t and t['pnl'] is not None)
                    if np.isnan(recent_pnl) or np.isinf(recent_pnl):
                        logger.error(f"should_halt_trading: Invalid recent_pnl: {recent_pnl}")
                        raise CriticalBotError("Recent P&L calculation invalid")
                        
                    if recent_pnl < -self.initial_capital * 0.15:  # Lost >15% in last 10 trades
                        return True, f"Rapid capital loss: ${recent_pnl:.0f}"
                        
                except (ValueError, TypeError) as e:
                    logger.error(f"should_halt_trading: Error calculating recent P&L: {e}")
                    raise CriticalBotError(f"Failed to calculate recent P&L: {e}")
            
            return False, ""
            
        except CriticalBotError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"should_halt_trading: Unexpected error: {e}")
            raise CriticalBotError(f"Failed to determine trading halt status: {e}")

    def get_position_limits(self) -> Dict:
        """Get current position limits for external validation."""
        try:
            return {
                'max_positions': self.max_positions,
                'max_position_risk': self.max_position_risk,
                'max_portfolio_heat': self.max_portfolio_heat,
                'max_correlation': self.max_correlation,
                'current_positions_allowed': max(0, self.max_positions),
                'remaining_capital_risk': max(0, self.max_portfolio_heat - self.current_heat),
                'size_reduction_factor': self.heat_reduction_factor ** min(self.consecutive_losses, 5)
            }
        except Exception as e:
            logger.error(f"get_position_limits: Unexpected error: {e}")
            raise CriticalBotError(f"Failed to get position limits: {e}")