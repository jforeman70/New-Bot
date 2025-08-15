import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import requests

from Market_State_Calculator import CriticalBotError
from Chemistry_Classifier import AssetChemistry
from Trailhead_Detector import TrailheadSignal

logger = logging.getLogger(__name__)

@dataclass
class PortfolioPosition:
    """Represents a synthesized portfolio position."""
    ticker: str
    weight: float  # Portfolio weight 0-1
    position_size: int  # Number of shares
    entry_price: float
    target_price: float
    stop_loss: float
    chemistry_type: str
    confidence: float
    metadata: Dict

class PortfolioSynthesizer:
    """Synthesizes optimal portfolio based on market state and trailhead signals."""
    
    def __init__(
        self,
        max_positions: int = 10,
        max_position_size: float = 0.15,  # 15% max per position
        min_confidence: float = 0.6,
        capital: float = 5000.0,
        fmp_key: str = None
    ):
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.capital = capital
        self.fmp_key = fmp_key
        
    def synthesize_portfolio(
        self,
        trailhead_signals: List[TrailheadSignal],
        chemistry_map: Dict[str, AssetChemistry],
        current_state: Tuple[float, float],
        target_state: Tuple[float, float]
    ) -> List[PortfolioPosition]:
        """
        Create optimal portfolio for predicted market destination.
        
        Args:
            trailhead_signals: Detected pressure points from detect_trailheads
            chemistry_map: Asset classifications from classify_asset_chemistry
            current_state: Current (risk, momentum) from calculate_market_state
            target_state: Predicted future (risk, momentum) coordinates
            
        Returns:
            List of PortfolioPositions sized for optimal capture
        """
        
        try:
            # Input validation
            if trailhead_signals is None:
                logger.error("Null trailhead_signals in synthesize_portfolio")
                raise ValueError("Trailhead signals cannot be None")
            
            if not trailhead_signals:
                logger.info("No trailhead signals to synthesize")
                return []
                
            if chemistry_map is None:
                logger.error("Null chemistry_map in synthesize_portfolio")
                raise ValueError("Chemistry map cannot be None")
                
            if not chemistry_map:
                logger.error("Empty chemistry map in portfolio synthesis")
                raise CriticalBotError("Cannot synthesize without chemistry data")
            
            if not isinstance(current_state, tuple) or len(current_state) != 2:
                logger.error(f"Invalid current_state: {current_state}")
                raise ValueError(f"Invalid current_state format: {current_state}")
                
            if not isinstance(target_state, tuple) or len(target_state) != 2:
                logger.error(f"Invalid target_state: {target_state}")
                raise ValueError(f"Invalid target_state format: {target_state}")
            
            current_risk, current_momentum = current_state
            target_risk, target_momentum = target_state
            
            if not all(isinstance(x, (int, float)) for x in [current_risk, current_momentum, target_risk, target_momentum]):
                logger.error(f"Invalid state values: current={current_state}, target={target_state}")
                raise ValueError("State values must be numeric")
            
            # Calculate state transition vector
            risk_delta = target_risk - current_risk
            momentum_delta = target_momentum - current_momentum
            transition_magnitude = np.sqrt(risk_delta**2 + momentum_delta**2)
            
            # Get current prices for position sizing
            tickers = [s.ticker for s in trailhead_signals[:self.max_positions * 2]]
            
            try:
                prices = self._get_current_prices(tickers)
            except Exception as e:
                logger.error(f"Price fetching failed in synthesize_portfolio: {e}")
                raise CriticalBotError(f"Cannot synthesize without price data: {e}")
            
            if not prices:
                logger.error("No prices retrieved for any tickers")
                raise CriticalBotError("Failed to get any price data for portfolio synthesis")
            
            portfolio = []
            allocated_weight = 0.0
            
            for signal in trailhead_signals:
                try:
                    if signal is None:
                        logger.warning("Null signal in trailhead_signals")
                        continue
                        
                    if not hasattr(signal, 'ticker'):
                        logger.warning(f"Signal missing ticker attribute: {signal}")
                        continue
                    
                    if signal.ticker not in chemistry_map:
                        logger.warning(f"No chemistry for {signal.ticker}")
                        continue
                    
                    if signal.ticker not in prices:
                        logger.warning(f"No price for {signal.ticker}")
                        continue
                    
                    chemistry = chemistry_map[signal.ticker]
                    entry_price = prices[signal.ticker]
                    
                    if entry_price is None or entry_price <= 0:
                        logger.warning(f"Invalid price for {signal.ticker}: {entry_price}")
                        continue
                    
                    # Calculate chemistry-state alignment score
                    alignment_score = self._calculate_alignment(
                        chemistry, target_risk, target_momentum, risk_delta, momentum_delta
                    )
                    
                    # Combine trailhead signal with chemistry alignment
                    if not hasattr(signal, 'composite_score') or signal.composite_score is None:
                        logger.warning(f"Missing composite_score for {signal.ticker}")
                        continue
                        
                    if not hasattr(chemistry, 'confidence') or chemistry.confidence is None:
                        logger.warning(f"Missing chemistry confidence for {signal.ticker}")
                        continue
                    
                    combined_confidence = (
                        signal.composite_score * 0.4 +
                        alignment_score * 0.4 +
                        chemistry.confidence * 0.2
                    )
                    
                    if combined_confidence < self.min_confidence:
                        continue
                    
                    # Position sizing based on confidence and transition magnitude
                    base_weight = min(
                        self.max_position_size,
                        combined_confidence * 0.15 * max(0.1, transition_magnitude)
                    )
                    
                    # Kelly-inspired sizing with safety factor
                    kelly_fraction = (combined_confidence - 0.5) / 2  # Simplified Kelly
                    position_weight = base_weight * kelly_fraction * 0.25  # 1/4 Kelly
                    
                    # Ensure we don't over-allocate
                    if allocated_weight + position_weight > 0.95:
                        position_weight = max(0, 0.95 - allocated_weight)
                    
                    if position_weight < 0.01:  # Skip tiny positions
                        continue
                    
                    # Calculate position details
                    position_capital = self.capital * position_weight
                    shares = int(position_capital / entry_price)
                    
                    if shares == 0:
                        continue
                    
                    # Set targets based on chemistry and transition
                    target_multiplier = 1 + (0.1 * transition_magnitude * combined_confidence)
                    stop_multiplier = 1 - (0.02 + 0.03 * (1 - combined_confidence))
                    
                    portfolio.append(PortfolioPosition(
                        ticker=signal.ticker,
                        weight=position_weight,
                        position_size=shares,
                        entry_price=entry_price,
                        target_price=entry_price * target_multiplier,
                        stop_loss=entry_price * stop_multiplier,
                        chemistry_type=chemistry.chemistry_type,
                        confidence=combined_confidence,
                        metadata={
                            'trailhead_type': signal.trigger_type,
                            'alignment_score': alignment_score,
                            'transition_magnitude': transition_magnitude,
                            'synthesized_at': datetime.now().isoformat()
                        }
                    ))
                    
                    allocated_weight += position_weight
                    
                    if len(portfolio) >= self.max_positions:
                        break
                        
                except AttributeError as e:
                    logger.error(f"Missing attribute for {signal.ticker if hasattr(signal, 'ticker') else 'unknown'}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to synthesize position for {signal.ticker if hasattr(signal, 'ticker') else 'unknown'}: {e}")
                    logger.error(f"Signal data: {signal}")
                    continue
            
            # Sort by confidence
            portfolio.sort(key=lambda x: x.confidence, reverse=True)
            
            if not portfolio:
                logger.warning("No positions synthesized from available signals")
            else:
                logger.info(f"Synthesized {len(portfolio)} positions, {allocated_weight:.1%} allocated")
            
            return portfolio
            
        except CriticalBotError:
            raise
        except ValueError as e:
            logger.error(f"Validation error in synthesize_portfolio: {e}")
            raise CriticalBotError(f"Invalid input data: {e}")
        except Exception as e:
            logger.error(f"Unexpected critical error in synthesize_portfolio: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise CriticalBotError(f"Portfolio synthesis system failure: {e}")
    
    def _get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch current prices for position sizing."""
        try:
            if not self.fmp_key:
                logger.error("FMP key not configured in PortfolioSynthesizer")
                raise CriticalBotError("FMP key required for price fetching")
            
            if not tickers:
                logger.error("Empty ticker list in _get_current_prices")
                raise ValueError("Cannot fetch prices for empty ticker list")
            
            symbols = ','.join(tickers)
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbols}?apikey={self.fmp_key}"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                quotes = response.json()
            except requests.exceptions.Timeout as e:
                logger.error(f"Timeout fetching prices from {url}: {e}")
                raise CriticalBotError(f"Price API timeout: {e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed in _get_current_prices: {e}")
                raise CriticalBotError(f"Price API request failed: {e}")
            except ValueError as e:
                logger.error(f"Invalid JSON in price response: {e}")
                raise CriticalBotError(f"Price data parsing failed: {e}")
            
            if quotes is None:
                logger.error("Null response from price API")
                raise CriticalBotError("Null price data received")
            
            if not isinstance(quotes, list):
                logger.error(f"Invalid price data type: {type(quotes)}")
                raise CriticalBotError(f"Invalid price data structure: expected list, got {type(quotes)}")
            
            prices = {}
            for quote in quotes:
                if not isinstance(quote, dict):
                    logger.warning(f"Invalid quote entry: {quote}")
                    continue
                    
                symbol = quote.get('symbol')
                price = quote.get('price')
                
                if symbol and price is not None and isinstance(price, (int, float)) and price > 0:
                    prices[symbol] = price
                else:
                    logger.warning(f"Invalid price data for {symbol}: {price}")
            
            if not prices:
                logger.error(f"No valid prices extracted from {len(quotes)} quotes")
                raise CriticalBotError("Failed to extract any valid prices")
            
            return prices
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in _get_current_prices: {e}")
            logger.error(f"Tickers requested: {tickers[:10] if tickers else 'None'}")
            raise CriticalBotError(f"Price fetching system failure: {e}")
    
    def _calculate_alignment(
        self,
        chemistry: AssetChemistry,
        target_risk: float,
        target_momentum: float,
        risk_delta: float,
        momentum_delta: float
    ) -> float:
        """Calculate how well asset chemistry aligns with state transition."""
        
        try:
            if not hasattr(chemistry, 'chemistry_type'):
                logger.error(f"Chemistry missing chemistry_type: {chemistry}")
                raise ValueError("Invalid chemistry object")
            
            alignment = 0.0
            
            # Noble gases thrive in high risk
            if chemistry.chemistry_type == 'noble_gas':
                alignment = target_risk * 0.8 + max(0, -momentum_delta) * 0.2
                
            # Volatile compounds need low risk, high momentum
            elif chemistry.chemistry_type == 'volatile_compound':
                alignment = (1 - target_risk) * 0.6 + max(0, target_momentum) * 0.4
                
            # Phase change captures risk-to-value transitions
            elif chemistry.chemistry_type == 'phase_change':
                if risk_delta < 0 and momentum_delta > 0:  # Risk falling, momentum rising
                    alignment = 0.9
                else:
                    alignment = 0.3
                    
            # Catalyst accelerants amplify any strong move
            elif chemistry.chemistry_type == 'catalyst_accelerant':
                alignment = abs(momentum_delta) * 0.7 + abs(risk_delta) * 0.3
            else:
                logger.warning(f"Unknown chemistry type: {chemistry.chemistry_type}")
                alignment = 0.5
            
            return min(1.0, max(0.0, alignment))
            
        except Exception as e:
            logger.error(f"Error in _calculate_alignment: {e}")
            logger.error(f"Chemistry: {chemistry}")
            raise CriticalBotError(f"Alignment calculation failed: {e}")