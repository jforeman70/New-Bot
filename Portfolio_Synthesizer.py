import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

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
        ibkr_executor,  # CORRECTED: Inject IBKR executor instead of hard-coded capital
        max_positions: int = 10,
        max_position_size: float = 0.15,  # 15% max per position
        min_confidence: float = 0.6,
        fmp_key: str = None
    ):
        try:
            if ibkr_executor is None:
                logger.error("FAIL-FAST: PortfolioSynthesizer.__init__() - ibkr_executor is None")
                raise CriticalBotError("Cannot initialize PortfolioSynthesizer - IBKR executor is None")
            
            if not isinstance(max_positions, int) or max_positions <= 0:
                logger.error(f"FAIL-FAST: PortfolioSynthesizer.__init__() - Invalid max_positions: {max_positions}")
                raise CriticalBotError(f"Invalid max_positions parameter: {max_positions}")
            
            if not isinstance(max_position_size, (int, float)) or not (0 < max_position_size <= 1):
                logger.error(f"FAIL-FAST: PortfolioSynthesizer.__init__() - Invalid max_position_size: {max_position_size}")
                raise CriticalBotError(f"Invalid max_position_size parameter: {max_position_size}")
            
            if not isinstance(min_confidence, (int, float)) or not (0 <= min_confidence <= 1):
                logger.error(f"FAIL-FAST: PortfolioSynthesizer.__init__() - Invalid min_confidence: {min_confidence}")
                raise CriticalBotError(f"Invalid min_confidence parameter: {min_confidence}")
            
            self.ibkr_executor = ibkr_executor  # CORRECTED: Store IBKR executor reference
            self.max_positions = max_positions
            self.max_position_size = max_position_size
            self.min_confidence = min_confidence
            self.fmp_key = fmp_key
            
            # Cache for performance
            self._last_capital_check = None
            self._cached_capital = None
            self._last_positions_check = None
            self._cached_positions = None
            
        except CriticalBotError:
            raise  # Re-raise critical errors
        except Exception as e:
            logger.error(f"FAIL-FAST: PortfolioSynthesizer.__init__() - Initialization failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"PortfolioSynthesizer initialization failed: {e}")

    def synthesize_portfolio(
        self,
        trailhead_signals: List[TrailheadSignal],
        chemistry_map: Dict[str, AssetChemistry],
        current_state: Tuple[float, float],
        target_state: Tuple[float, float]
    ) -> List[PortfolioPosition]:
        """
        🏔️ CUTTING-EDGE: Synthesize geological portfolio formation based on pressure point analysis.
        
        Revolutionary Portfolio Synthesis Algorithm:
        - GEOLOGICAL POSITIONING: Position at critical pressure points before energy release
        - CHEMICAL REACTION SIZING: Size positions based on chemical reaction potential
        - TECTONIC DIVERSIFICATION: Spread risk across different geological formations
        - SEISMIC TIMING: Enter positions at optimal pressure buildup moments
        
        Geological Portfolio Physics:
        🏔️ FORMATION POSITIONING = Strategic placement at geological pressure points
        ⚗️ CHEMICAL SIZING = Position size based on reaction potential and catalysis
        🌋 PRESSURE WEIGHTING = Allocate more capital to higher pressure areas
        🎯 PRECISION ENTRY = Exact entry points at geological fault lines
        
        Portfolio Construction Philosophy:
        - Each position is a geological formation that will release energy
        - Position sizing based on pressure buildup and structural fragility
        - Chemical diversification across reaction types
        - Risk management through geological understanding
        
        Args:
            trailhead_signals: Detected pressure points from detect_trailheads
            chemistry_map: Asset chemical profiles from classify_asset_chemistry
            current_state: Current geological pressure (risk, momentum) coordinates
            target_state: Predicted future state coordinates from State_Predictor
            
        Returns:
            List of optimized portfolio positions ready for execution
            
        Raises:
            CriticalBotError: On portfolio synthesis failure
        """
        
        try:
            # Geological survey validation with fail-fast
            if trailhead_signals is None:
                logger.error("FAIL-FAST: synthesize_portfolio() - trailhead_signals parameter is None")
                raise CriticalBotError("Cannot synthesize portfolio - pressure point data is None")
            
            if not isinstance(trailhead_signals, list):
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Invalid trailhead_signals type: {type(trailhead_signals)}")
                raise CriticalBotError(f"Invalid pressure point data type: {type(trailhead_signals)}")
            
            if chemistry_map is None:
                logger.error("FAIL-FAST: synthesize_portfolio() - chemistry_map parameter is None")
                raise CriticalBotError("Cannot synthesize portfolio - chemical analysis data is None")
            
            if not isinstance(chemistry_map, dict):
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Invalid chemistry_map type: {type(chemistry_map)}")
                raise CriticalBotError(f"Invalid chemical analysis data type: {type(chemistry_map)}")
            
            if current_state is None:
                logger.error("FAIL-FAST: synthesize_portfolio() - current_state parameter is None")
                raise CriticalBotError("Cannot synthesize portfolio - current geological state is None")
            
            if target_state is None:
                logger.error("FAIL-FAST: synthesize_portfolio() - target_state parameter is None")
                raise CriticalBotError("Cannot synthesize portfolio - target geological state is None")
            
            # Validate state coordinates
            try:
                current_risk, current_momentum = current_state
                target_risk, target_momentum = target_state
                
                current_risk = float(current_risk)
                current_momentum = float(current_momentum)
                target_risk = float(target_risk)
                target_momentum = float(target_momentum)
                
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Invalid state coordinates: current={current_state}, target={target_state}, error: {e}")
                raise CriticalBotError(f"Invalid geological state coordinates: {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - State coordinate extraction failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"State coordinate extraction failed: {e}")
            
            # Validate state bounds
            for name, value in [
                ('current_risk', current_risk), ('target_risk', target_risk)
            ]:
                try:
                    if not (0 <= value <= 1):
                        logger.error(f"FAIL-FAST: synthesize_portfolio() - {name} out of bounds: {value}")
                        raise CriticalBotError(f"Geological {name} exceeds safe limits: {value}")
                except Exception as e:
                    logger.error(f"FAIL-FAST: synthesize_portfolio() - Risk validation failed for {name}: {type(e).__name__}: {e}")
                    raise CriticalBotError(f"Risk validation failed for {name}: {e}")
            
            for name, value in [
                ('current_momentum', current_momentum), ('target_momentum', target_momentum)
            ]:
                try:
                    if not (-1 <= value <= 1):
                        logger.error(f"FAIL-FAST: synthesize_portfolio() - {name} out of bounds: {value}")
                        raise CriticalBotError(f"Geological {name} exceeds safe limits: {value}")
                except Exception as e:
                    logger.error(f"FAIL-FAST: synthesize_portfolio() - Momentum validation failed for {name}: {type(e).__name__}: {e}")
                    raise CriticalBotError(f"Momentum validation failed for {name}: {e}")
            
            # Get real capital and positions from IBKR
            try:
                current_capital = self._get_current_capital()
                buying_power = self._get_buying_power()
                current_positions = self._get_current_positions()
            except CriticalBotError:
                raise  # Re-raise critical errors
            except Exception as e:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - IBKR account data retrieval failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Cannot access account data for portfolio synthesis: {e}")
            
            if current_capital is None or current_capital <= 0:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Invalid capital value: {current_capital}")
                raise CriticalBotError(f"Invalid account capital value: {current_capital}")
            
            if buying_power is None or buying_power < 0:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Invalid buying power: {buying_power}")
                raise CriticalBotError(f"Invalid buying power value: {buying_power}")
            
            if current_positions is None:
                logger.error("FAIL-FAST: synthesize_portfolio() - Current positions is None")
                raise CriticalBotError("Current positions data is None")
            
            logger.info(f"Geological portfolio synthesis: Capital=${current_capital:,.2f}, "
                       f"Buying Power=${buying_power:,.2f}, "
                       f"Current Positions={len(current_positions)}, "
                       f"Pressure Points={len(trailhead_signals)}")
            
            # Filter and validate signals
            try:
                validated_signals = self._validate_trailhead_signals(trailhead_signals)
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Signal validation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Trailhead signal validation failed: {e}")
            
            if not validated_signals:
                logger.warning("No validated geological pressure points found")
                return []
            
            # Calculate geological diversification strategy
            try:
                diversification_matrix = self._calculate_geological_diversification(
                    validated_signals, chemistry_map, current_state, target_state
                )
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Diversification calculation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Geological diversification calculation failed: {e}")
            
            if diversification_matrix is None:
                logger.error("FAIL-FAST: synthesize_portfolio() - Diversification matrix is None")
                raise CriticalBotError("Diversification matrix calculation returned None")
            
            # Synthesize portfolio positions
            portfolio_positions = []
            total_allocated = 0.0
            
            for signal in validated_signals[:self.max_positions]:
                try:
                    if signal is None:
                        logger.warning("Skipping None signal in validated signals")
                        continue
                    
                    # Get chemistry data
                    chemistry = chemistry_map.get(signal.ticker)
                    if chemistry is None:
                        logger.warning(f"No chemistry data for pressure point {signal.ticker}")
                        continue
                    
                    # Check for existing position
                    if signal.ticker in current_positions:
                        logger.info(f"Already holding geological formation {signal.ticker}")
                        continue
                    
                    # Get current market price - CORRECTED: Real price from IBKR
                    try:
                        current_price = self._get_market_price(signal.ticker)
                    except CriticalBotError:
                        raise
                    except Exception as e:
                        logger.error(f"FAIL-FAST: synthesize_portfolio() - Market price retrieval failed for {signal.ticker}: {type(e).__name__}: {e}")
                        continue  # Skip this position but continue with others
                    
                    if current_price is None or current_price <= 0:
                        logger.warning(f"No market data for geological formation {signal.ticker}")
                        continue
                    
                    # Calculate geological position sizing
                    remaining_capital = buying_power - total_allocated
                    if remaining_capital < 100:  # Minimum $100 position
                        logger.info("Insufficient geological survey funding for additional formations")
                        break
                    
                    try:
                        position_allocation = self._calculate_geological_position_sizing(
                            signal, chemistry, current_state, target_state,
                            remaining_capital, current_capital, diversification_matrix,
                            current_price  # CORRECTED: Pass real price
                        )
                    except CriticalBotError:
                        raise
                    except Exception as e:
                        logger.error(f"FAIL-FAST: synthesize_portfolio() - Position sizing failed for {signal.ticker}: {type(e).__name__}: {e}")
                        continue  # Skip this position but continue with others
                    
                    if position_allocation is None:
                        logger.error(f"FAIL-FAST: synthesize_portfolio() - Position allocation is None for {signal.ticker}")
                        continue
                    
                    if 'shares' not in position_allocation:
                        logger.error(f"FAIL-FAST: synthesize_portfolio() - Missing 'shares' in position allocation for {signal.ticker}")
                        continue
                    
                    if position_allocation['shares'] == 0:
                        logger.debug(f"Zero position size calculated for {signal.ticker}")
                        continue
                    
                    # Calculate precision entry and exit points
                    try:
                        entry_points = self._calculate_precision_entry_points(
                            signal, chemistry, current_price, current_state, target_state
                        )
                    except CriticalBotError:
                        raise
                    except Exception as e:
                        logger.error(f"FAIL-FAST: synthesize_portfolio() - Entry point calculation failed for {signal.ticker}: {type(e).__name__}: {e}")
                        continue  # Skip this position but continue with others
                    
                    if entry_points is None:
                        logger.error(f"FAIL-FAST: synthesize_portfolio() - Entry points is None for {signal.ticker}")
                        continue
                    
                    required_fields = ['entry_price', 'target_price', 'stop_loss']
                    for field in required_fields:
                        if field not in entry_points:
                            logger.error(f"FAIL-FAST: synthesize_portfolio() - Missing '{field}' in entry points for {signal.ticker}")
                            raise CriticalBotError(f"Missing '{field}' in entry points calculation for {signal.ticker}")
                    
                    # Create portfolio position
                    try:
                        position = PortfolioPosition(
                            ticker=signal.ticker,
                            weight=position_allocation['weight'],
                            position_size=position_allocation['shares'],
                            entry_price=entry_points['entry_price'],
                            target_price=entry_points['target_price'],
                            stop_loss=entry_points['stop_loss'],
                            chemistry_type=chemistry.chemistry_type,
                            confidence=signal.composite_score,
                            metadata={
                                'geological_analysis': {
                                    'pressure_score': signal.pressure_score,
                                    'fragility_score': signal.fragility_score,
                                    'trigger_type': signal.trigger_type,
                                    'formation_type': chemistry.chemistry_type
                                },
                                'positioning_strategy': position_allocation['strategy'],
                                'entry_analysis': entry_points,
                                'market_coordinates': {
                                    'current_state': current_state,
                                    'target_state': target_state,
                                    'state_transition': {
                                        'risk_delta': target_risk - current_risk,
                                        'momentum_delta': target_momentum - current_momentum
                                    }
                                },
                                'diversification_factor': diversification_matrix.get(signal.ticker, 1.0),
                                'synthesis_timestamp': datetime.now().isoformat()
                            }
                        )
                    except Exception as e:
                        logger.error(f"FAIL-FAST: synthesize_portfolio() - Position object creation failed for {signal.ticker}: {type(e).__name__}: {e}")
                        raise CriticalBotError(f"Position object creation failed for {signal.ticker}: {e}")
                    
                    portfolio_positions.append(position)
                    total_allocated += position_allocation['dollar_amount']
                    
                    logger.info(f"Geological formation synthesized: {signal.ticker} "
                               f"({position_allocation['shares']} shares @ ${entry_points['entry_price']:.2f})")
                    
                except CriticalBotError:
                    raise  # Re-raise critical errors
                except Exception as e:
                    logger.error(f"FAIL-FAST: synthesize_portfolio() - Position synthesis failed for {signal.ticker}: {type(e).__name__}: {e}")
                    continue  # Skip this position but continue with others
            
            # Validate final portfolio
            try:
                portfolio_validation = self._validate_geological_portfolio(portfolio_positions, current_capital)
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Portfolio validation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Portfolio validation system error: {e}")
            
            if portfolio_validation is None:
                logger.error("FAIL-FAST: synthesize_portfolio() - Portfolio validation returned None")
                raise CriticalBotError("Portfolio validation returned None")
            
            if 'valid' not in portfolio_validation:
                logger.error("FAIL-FAST: synthesize_portfolio() - Missing 'valid' field in portfolio validation")
                raise CriticalBotError("Portfolio validation missing 'valid' field")
            
            if not portfolio_validation['valid']:
                reasons = portfolio_validation.get('reasons', ['Unknown validation failure'])
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Portfolio validation failed: {reasons}")
                raise CriticalBotError(f"Geological portfolio formation invalid: {reasons}")
            
            # Portfolio synthesis summary
            try:
                total_weight = sum(p.weight for p in portfolio_positions)
                avg_confidence = np.mean([p.confidence for p in portfolio_positions]) if portfolio_positions else 0
            except Exception as e:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Portfolio summary calculation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Portfolio summary calculation failed: {e}")
            
            logger.info(f"Geological portfolio synthesized: {len(portfolio_positions)} formations, "
                       f"${total_allocated:,.2f} allocated ({total_weight:.1%} of capital), "
                       f"Avg confidence: {avg_confidence:.2f}")
            
            return portfolio_positions
            
        except CriticalBotError:
            raise  # Re-raise critical errors
        except Exception as e:
            logger.error(f"FAIL-FAST: synthesize_portfolio() - Catastrophic portfolio synthesis failure: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Portfolio synthesis system failure: {e}")

    def _get_current_capital(self) -> float:
        """Get current portfolio value from IBKR account."""
        try:
            now = datetime.now()
            if (self._last_capital_check and 
                (now - self._last_capital_check).seconds < 60 and
                self._cached_capital is not None):
                return self._cached_capital
            
            try:
                capital = self.ibkr_executor.get_account_value()
            except AttributeError as e:
                logger.error(f"FAIL-FAST: _get_current_capital() - IBKR executor missing get_account_value method: {e}")
                raise CriticalBotError(f"IBKR executor method missing: get_account_value - {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: _get_current_capital() - IBKR get_account_value call failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"IBKR account value retrieval failed: {e}")
            
            if capital is None:
                logger.error("FAIL-FAST: _get_current_capital() - IBKR returned None for account value")
                raise CriticalBotError("IBKR returned None for account value")
            
            try:
                capital = float(capital)
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _get_current_capital() - Invalid capital value type: {type(capital)}, value: {capital}, error: {e}")
                raise CriticalBotError(f"Invalid capital value from IBKR: {capital}")
            
            if capital <= 0:
                logger.error(f"FAIL-FAST: _get_current_capital() - Non-positive capital value: {capital}")
                raise CriticalBotError(f"Invalid account capital value: {capital}")
            
            if np.isnan(capital) or np.isinf(capital):
                logger.error(f"FAIL-FAST: _get_current_capital() - NaN/Inf capital value: {capital}")
                raise CriticalBotError(f"NaN/Inf capital value from IBKR: {capital}")
            
            self._cached_capital = capital
            self._last_capital_check = now
            
            return capital
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: _get_current_capital() - Capital retrieval failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Capital retrieval failed: {e}")

    def _get_buying_power(self) -> float:
        """Get available buying power from IBKR."""
        try:
            try:
                buying_power = self.ibkr_executor.get_buying_power()
            except AttributeError as e:
                logger.error(f"FAIL-FAST: _get_buying_power() - IBKR executor missing get_buying_power method: {e}")
                raise CriticalBotError(f"IBKR executor method missing: get_buying_power - {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: _get_buying_power() - IBKR get_buying_power call failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"IBKR buying power retrieval failed: {e}")
            
            if buying_power is None:
                logger.error("FAIL-FAST: _get_buying_power() - IBKR returned None for buying power")
                raise CriticalBotError("IBKR returned None for buying power")
            
            try:
                buying_power = float(buying_power)
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _get_buying_power() - Invalid buying power type: {type(buying_power)}, value: {buying_power}, error: {e}")
                raise CriticalBotError(f"Invalid buying power value from IBKR: {buying_power}")
            
            if buying_power < 0:
                logger.error(f"FAIL-FAST: _get_buying_power() - Negative buying power: {buying_power}")
                raise CriticalBotError(f"Invalid buying power value: {buying_power}")
            
            if np.isnan(buying_power) or np.isinf(buying_power):
                logger.error(f"FAIL-FAST: _get_buying_power() - NaN/Inf buying power: {buying_power}")
                raise CriticalBotError(f"NaN/Inf buying power from IBKR: {buying_power}")
            
            return buying_power
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: _get_buying_power() - Buying power retrieval failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Buying power retrieval failed: {e}")

    def _get_current_positions(self) -> Dict:
        """Get current positions from IBKR."""
        try:
            now = datetime.now()
            if (self._last_positions_check and 
                (now - self._last_positions_check).seconds < 30 and
                self._cached_positions is not None):
                return self._cached_positions
            
            try:
                positions = self.ibkr_executor.get_current_positions()
            except AttributeError as e:
                logger.error(f"FAIL-FAST: _get_current_positions() - IBKR executor missing get_current_positions method: {e}")
                raise CriticalBotError(f"IBKR executor method missing: get_current_positions - {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: _get_current_positions() - IBKR get_current_positions call failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"IBKR positions retrieval failed: {e}")
            
            if positions is None:
                logger.error("FAIL-FAST: _get_current_positions() - IBKR returned None for positions")
                raise CriticalBotError("IBKR returned None for current positions")
            
            if not isinstance(positions, dict):
                logger.error(f"FAIL-FAST: _get_current_positions() - Invalid positions type: {type(positions)}")
                raise CriticalBotError(f"Invalid positions data type from IBKR: {type(positions)}")
            
            self._cached_positions = positions
            self._last_positions_check = now
            
            return positions
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: _get_current_positions() - Position retrieval failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Position retrieval failed: {e}")

    def _get_market_price(self, ticker: str) -> Optional[float]:
        """Get current market price for ticker."""
        try:
            if ticker is None:
                logger.error("FAIL-FAST: _get_market_price() - ticker parameter is None")
                raise CriticalBotError("Ticker parameter is None for price lookup")
            
            if not isinstance(ticker, str):
                logger.error(f"FAIL-FAST: _get_market_price() - Invalid ticker type: {type(ticker)}")
                raise CriticalBotError(f"Invalid ticker type for price lookup: {type(ticker)}")
            
            if len(ticker.strip()) == 0:
                logger.error("FAIL-FAST: _get_market_price() - Empty ticker string")
                raise CriticalBotError("Empty ticker string for price lookup")
            
            try:
                price = self.ibkr_executor.get_market_data(ticker)
            except AttributeError as e:
                logger.error(f"FAIL-FAST: _get_market_price() - IBKR executor missing get_market_data method: {e}")
                raise CriticalBotError(f"IBKR executor method missing: get_market_data - {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: _get_market_price() - IBKR get_market_data call failed for {ticker}: {type(e).__name__}: {e}")
                raise CriticalBotError(f"IBKR market data retrieval failed for {ticker}: {e}")
            
            if price is None:
                logger.warning(f"No market data available for {ticker}")
                return None
            
            try:
                price = float(price)
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _get_market_price() - Invalid price type for {ticker}: {type(price)}, value: {price}, error: {e}")
                raise CriticalBotError(f"Invalid price type from IBKR for {ticker}: {price}")
            
            if price <= 0:
                logger.error(f"FAIL-FAST: _get_market_price() - Non-positive price for {ticker}: {price}")
                raise CriticalBotError(f"Invalid market price for {ticker}: {price}")
            
            if np.isnan(price) or np.isinf(price):
                logger.error(f"FAIL-FAST: _get_market_price() - NaN/Inf price for {ticker}: {price}")
                raise CriticalBotError(f"NaN/Inf market price for {ticker}: {price}")
            
            return price
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: _get_market_price() - Price retrieval failed for {ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Market price retrieval failed for {ticker}: {e}")

    def validate_portfolio(self, positions: List[PortfolioPosition]) -> bool:
        """Validate portfolio meets risk constraints."""
        try:
            if positions is None:
                logger.error("FAIL-FAST: validate_portfolio() - positions parameter is None")
                raise CriticalBotError("Cannot validate portfolio - positions list is None")
            
            if not isinstance(positions, list):
                logger.error(f"FAIL-FAST: validate_portfolio() - Invalid positions type: {type(positions)}")
                raise CriticalBotError(f"Invalid positions type for validation: {type(positions)}")
            
            if not positions:
                logger.info("Empty portfolio - validation passed")
                return True
            
            # Validate each position object
            for i, position in enumerate(positions):
                try:
                    if position is None:
                        logger.error(f"FAIL-FAST: validate_portfolio() - Position {i} is None")
                        raise CriticalBotError(f"Position {i} is None in portfolio")
                    
                    if not isinstance(position, PortfolioPosition):
                        logger.error(f"FAIL-FAST: validate_portfolio() - Invalid position type at index {i}: {type(position)}")
                        raise CriticalBotError(f"Invalid position type at index {i}: {type(position)}")
                    
                    # Validate position attributes
                    if not hasattr(position, 'ticker') or position.ticker is None:
                        logger.error(f"FAIL-FAST: validate_portfolio() - Position {i} missing or None ticker")
                        raise CriticalBotError(f"Position {i} missing or None ticker")
                    
                    if not hasattr(position, 'weight') or position.weight is None:
                        logger.error(f"FAIL-FAST: validate_portfolio() - Position {i} missing or None weight")
                        raise CriticalBotError(f"Position {i} missing or None weight")
                    
                except Exception as e:
                    logger.error(f"FAIL-FAST: validate_portfolio() - Position validation failed at index {i}: {type(e).__name__}: {e}")
                    raise CriticalBotError(f"Position validation failed at index {i}: {e}")
            
            try:
                current_capital = self._get_current_capital()
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"FAIL-FAST: validate_portfolio() - Cannot get capital for validation: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Portfolio validation failed - cannot access capital: {e}")
            
            # Comprehensive validation
            try:
                validation_result = self._validate_geological_portfolio(positions, current_capital)
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"FAIL-FAST: validate_portfolio() - Geological validation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Geological portfolio validation failed: {e}")
            
            if validation_result is None:
                logger.error("FAIL-FAST: validate_portfolio() - Validation result is None")
                raise CriticalBotError("Portfolio validation returned None")
            
            if 'valid' not in validation_result:
                logger.error("FAIL-FAST: validate_portfolio() - Missing 'valid' field in validation result")
                raise CriticalBotError("Portfolio validation missing 'valid' field")
            
            if not validation_result['valid']:
                reasons = validation_result.get('reasons', ['Unknown validation failure'])
                logger.error(f"FAIL-FAST: validate_portfolio() - Portfolio validation failed: {reasons}")
                return False
            
            logger.info("Portfolio validation passed")
            return True
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: validate_portfolio() - Portfolio validation error: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Portfolio validation system error: {e}")

    def _validate_trailhead_signals(self, signals: List[TrailheadSignal]) -> List[TrailheadSignal]:
        """Validate and filter geological pressure point signals."""
        try:
            if signals is None:
                logger.error("FAIL-FAST: _validate_trailhead_signals() - signals parameter is None")
                raise CriticalBotError("Cannot validate trailhead signals - signals list is None")
            
            if not isinstance(signals, list):
                logger.error(f"FAIL-FAST: _validate_trailhead_signals() - Invalid signals type: {type(signals)}")
                raise CriticalBotError(f"Invalid signals type for validation: {type(signals)}")
            
            if not signals:
                logger.info("Empty signals list - returning empty validated list")
                return []
            
            validated_signals = []
            
            for i, signal in enumerate(signals):
                try:
                    if signal is None:
                        logger.debug(f"Skipping None signal at index {i}")
                        continue
                    
                    if not isinstance(signal, TrailheadSignal):
                        logger.warning(f"Invalid signal type at index {i}: {type(signal)}")
                        continue
                    
                    # Validate ticker
                    if not hasattr(signal, 'ticker') or signal.ticker is None:
                        logger.warning(f"Signal at index {i} missing or None ticker")
                        continue
                    
                    if not isinstance(signal.ticker, str) or len(signal.ticker.strip()) == 0:
                        logger.warning(f"Signal at index {i} has invalid ticker: {signal.ticker}")
                        continue
                    
                    # Validate composite score
                    if not hasattr(signal, 'composite_score') or signal.composite_score is None:
                        logger.warning(f"Signal {signal.ticker} missing or None composite_score")
                        continue
                    
                    try:
                        composite_score = float(signal.composite_score)
                    except (ValueError, TypeError):
                        logger.warning(f"Signal {signal.ticker} has invalid composite_score: {signal.composite_score}")
                        continue
                    
                    if np.isnan(composite_score) or np.isinf(composite_score):
                        logger.warning(f"Signal {signal.ticker} has NaN/Inf composite_score: {composite_score}")
                        continue
                    
                    if composite_score < self.min_confidence:
                        logger.debug(f"Signal {signal.ticker} below confidence threshold: {composite_score}")
                        continue
                    
                    # Validate pressure score
                    if not hasattr(signal, 'pressure_score') or signal.pressure_score is None:
                        logger.warning(f"Signal {signal.ticker} missing or None pressure_score")
                        continue
                    
                    try:
                        pressure_score = float(signal.pressure_score)
                    except (ValueError, TypeError):
                        logger.warning(f"Signal {signal.ticker} has invalid pressure_score: {signal.pressure_score}")
                        continue
                    
                    if not (0 <= pressure_score <= 1):
                        logger.warning(f"Signal {signal.ticker} pressure_score out of bounds: {pressure_score}")
                        continue
                    
                    # Validate fragility score
                    if not hasattr(signal, 'fragility_score') or signal.fragility_score is None:
                        logger.warning(f"Signal {signal.ticker} missing or None fragility_score")
                        continue
                    
                    try:
                        fragility_score = float(signal.fragility_score)
                    except (ValueError, TypeError):
                        logger.warning(f"Signal {signal.ticker} has invalid fragility_score: {signal.fragility_score}")
                        continue
                    
                    if not (0 <= fragility_score <= 1):
                        logger.warning(f"Signal {signal.ticker} fragility_score out of bounds: {fragility_score}")
                        continue
                    
                    # Validate trigger type
                    if not hasattr(signal, 'trigger_type') or signal.trigger_type is None:
                        logger.warning(f"Signal {signal.ticker} missing or None trigger_type")
                        continue
                    
                    valid_triggers = {'squeeze', 'breakout', 'reversal', 'cascade'}
                    if signal.trigger_type not in valid_triggers:
                        logger.warning(f"Signal {signal.ticker} invalid trigger_type: {signal.trigger_type}")
                        continue
                    
                    validated_signals.append(signal)
                    
                except Exception as e:
                    logger.warning(f"Signal validation failed at index {i}: {type(e).__name__}: {e}")
                    continue
            
            logger.info(f"Validated {len(validated_signals)} out of {len(signals)} trailhead signals")
            return validated_signals
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: _validate_trailhead_signals() - Signal validation failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Trailhead signal validation failed: {e}")

    def _calculate_geological_diversification(
        self,
        signals: List[TrailheadSignal],
        chemistry_map: Dict[str, AssetChemistry],
        current_state: Tuple[float, float],
        target_state: Tuple[float, float]
    ) -> Dict[str, float]:
        """Calculate geological diversification matrix for optimal formation spread."""
        try:
            if signals is None:
                logger.error("FAIL-FAST: _calculate_geological_diversification() - signals parameter is None")
                raise CriticalBotError("Cannot calculate diversification - signals list is None")
            
            if not isinstance(signals, list):
                logger.error(f"FAIL-FAST: _calculate_geological_diversification() - Invalid signals type: {type(signals)}")
                raise CriticalBotError(f"Invalid signals type for diversification: {type(signals)}")
            
            if chemistry_map is None:
                logger.error("FAIL-FAST: _calculate_geological_diversification() - chemistry_map parameter is None")
                raise CriticalBotError("Cannot calculate diversification - chemistry_map is None")
            
            if not isinstance(chemistry_map, dict):
                logger.error(f"FAIL-FAST: _calculate_geological_diversification() - Invalid chemistry_map type: {type(chemistry_map)}")
                raise CriticalBotError(f"Invalid chemistry_map type for diversification: {type(chemistry_map)}")
            
            if current_state is None or target_state is None:
                logger.error("FAIL-FAST: _calculate_geological_diversification() - State parameters are None")
                raise CriticalBotError("Cannot calculate diversification - state parameters are None")
            
            if not signals:
                logger.info("Empty signals list - returning empty diversification matrix")
                return {}
            
            diversification_matrix = {}
            
            # Count chemistry and trigger types
            chemistry_counts = {}
            trigger_counts = {}
            
            for signal in signals:
                try:
                    if signal is None or not hasattr(signal, 'ticker'):
                        continue
                    
                    chemistry = chemistry_map.get(signal.ticker)
                    if chemistry and hasattr(chemistry, 'chemistry_type'):
                        chemistry_type = chemistry.chemistry_type
                        chemistry_counts[chemistry_type] = chemistry_counts.get(chemistry_type, 0) + 1
                    
                    if hasattr(signal, 'trigger_type') and signal.trigger_type:
                        trigger_counts[signal.trigger_type] = trigger_counts.get(signal.trigger_type, 0) + 1
                        
                except Exception as e:
                    logger.warning(f"Error counting chemistry/trigger for signal: {type(e).__name__}: {e}")
                    continue
            
            total_signals = len(signals)
            if total_signals == 0:
                logger.error("FAIL-FAST: _calculate_geological_diversification() - No valid signals for diversification")
                raise CriticalBotError("No valid signals for diversification calculation")
            
            # Extract state coordinates
            try:
                current_risk, current_momentum = current_state
                target_risk, target_momentum = target_state
                
                current_risk = float(current_risk)
                current_momentum = float(current_momentum)
                target_risk = float(target_risk)
                target_momentum = float(target_momentum)
                
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _calculate_geological_diversification() - Invalid state coordinates: {e}")
                raise CriticalBotError(f"Invalid state coordinates for diversification: {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_geological_diversification() - State extraction failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"State extraction failed for diversification: {e}")
            
            for signal in signals:
                try:
                    if signal is None or not hasattr(signal, 'ticker'):
                        continue
                    
                    chemistry = chemistry_map.get(signal.ticker)
                    if chemistry is None:
                        diversification_matrix[signal.ticker] = 0.5
                        continue
                    
                    diversification_factor = 1.0
                    
                    # Chemistry type diversification penalty
                    if hasattr(chemistry, 'chemistry_type') and chemistry.chemistry_type in chemistry_counts:
                        chemistry_concentration = chemistry_counts[chemistry.chemistry_type] / total_signals
                        if chemistry_concentration > 0.4:
                            diversification_factor *= (1.0 - (chemistry_concentration - 0.4))
                    
                    # Trigger type diversification penalty
                    if hasattr(signal, 'trigger_type') and signal.trigger_type in trigger_counts:
                        trigger_concentration = trigger_counts[signal.trigger_type] / total_signals
                        if trigger_concentration > 0.5:
                            diversification_factor *= (1.0 - (trigger_concentration - 0.5))
                    
                    # Market state alignment bonus
                    if hasattr(chemistry, 'chemistry_type'):
                        if chemistry.chemistry_type == 'volatile_compound' and target_risk < current_risk:
                            diversification_factor *= 1.1
                        elif chemistry.chemistry_type == 'catalyst_accelerant' and abs(target_momentum) > abs(current_momentum):
                            diversification_factor *= 1.1
                        elif chemistry.chemistry_type == 'phase_change' and abs(target_risk - 0.5) < abs(current_risk - 0.5):
                            diversification_factor *= 1.1
                    
                    diversification_matrix[signal.ticker] = max(0.3, min(1.5, diversification_factor))
                    
                except Exception as e:
                    logger.warning(f"Diversification calculation failed for signal {getattr(signal, 'ticker', 'unknown')}: {type(e).__name__}: {e}")
                    diversification_matrix[signal.ticker] = 1.0  # Default factor
                    continue
            
            if not diversification_matrix:
                logger.error("FAIL-FAST: _calculate_geological_diversification() - Empty diversification matrix")
                raise CriticalBotError("Diversification matrix calculation produced no results")
            
            return diversification_matrix
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: _calculate_geological_diversification() - Diversification calculation failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Geological diversification calculation failed: {e}")

    def _calculate_geological_position_sizing(
        self,
        signal: TrailheadSignal,
        chemistry: AssetChemistry,
        current_state: Tuple[float, float],
        target_state: Tuple[float, float],
        available_capital: float,
        total_capital: float,
        diversification_matrix: Dict[str, float],
        current_price: float  # CORRECTED: Real price parameter
    ) -> Dict:
        """Calculate precise geological position sizing using advanced geological physics."""
        try:
            # Validate inputs
            if signal is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - signal parameter is None")
                raise CriticalBotError("Cannot calculate position sizing - signal is None")
            
            if chemistry is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - chemistry parameter is None")
                raise CriticalBotError("Cannot calculate position sizing - chemistry is None")
            
            if current_state is None or target_state is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - State parameters are None")
                raise CriticalBotError("Cannot calculate position sizing - state parameters are None")
            
            if diversification_matrix is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - diversification_matrix parameter is None")
                raise CriticalBotError("Cannot calculate position sizing - diversification_matrix is None")
            
            # Validate numeric inputs
            try:
                available_capital = float(available_capital)
                total_capital = float(total_capital)
                current_price = float(current_price)
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Invalid numeric parameters: {e}")
                raise CriticalBotError(f"Invalid numeric parameters for position sizing: {e}")
            
            if available_capital < 0 or total_capital <= 0 or current_price <= 0:
                logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Invalid parameter values: available={available_capital}, total={total_capital}, price={current_price}")
                raise CriticalBotError(f"Invalid parameter values for position sizing")
            
            # Validate signal attributes
            if not hasattr(signal, 'composite_score') or signal.composite_score is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - Signal missing composite_score")
                raise CriticalBotError("Signal missing composite_score for position sizing")
            
            if not hasattr(signal, 'pressure_score') or signal.pressure_score is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - Signal missing pressure_score")
                raise CriticalBotError("Signal missing pressure_score for position sizing")
            
            if not hasattr(signal, 'fragility_score') or signal.fragility_score is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - Signal missing fragility_score")
                raise CriticalBotError("Signal missing fragility_score for position sizing")
            
            if not hasattr(signal, 'trigger_type') or signal.trigger_type is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - Signal missing trigger_type")
                raise CriticalBotError("Signal missing trigger_type for position sizing")
            
            if not hasattr(signal, 'ticker') or signal.ticker is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - Signal missing ticker")
                raise CriticalBotError("Signal missing ticker for position sizing")
            
            # Validate chemistry attributes
            if not hasattr(chemistry, 'chemistry_type') or chemistry.chemistry_type is None:
                logger.error("FAIL-FAST: _calculate_geological_position_sizing() - Chemistry missing chemistry_type")
                raise CriticalBotError("Chemistry missing chemistry_type for position sizing")
            
            # Base Kelly Criterion calculation
            try:
                win_probability = float(signal.composite_score)
                pressure_score = float(signal.pressure_score)
                fragility_score = float(signal.fragility_score)
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Invalid signal scores: {e}")
                raise CriticalBotError(f"Invalid signal scores for position sizing: {e}")
            
            # Calculate expected payoff based on pressure and fragility
            try:
                pressure_energy = pressure_score * fragility_score
                expected_payoff_ratio = 1.0 + (pressure_energy * 2.0)  # Up to 3:1 payoff
                
                # Kelly fraction with geological adjustments
                if expected_payoff_ratio > 1.0:
                    kelly_fraction = (win_probability * expected_payoff_ratio - (1 - win_probability)) / expected_payoff_ratio
                else:
                    kelly_fraction = 0.0
                
                # Apply geological safety factor
                kelly_fraction = min(kelly_fraction, win_probability * 0.5)
                
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Kelly calculation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Kelly fraction calculation failed: {e}")
            
            # Chemical reaction amplification
            try:
                chemistry_multiplier = {
                    'volatile_compound': 1.2,
                    'catalyst_accelerant': 1.1,
                    'phase_change': 1.0,
                    'noble_gas': 0.8
                }.get(chemistry.chemistry_type, 1.0)
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Chemistry multiplier failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Chemistry multiplier calculation failed: {e}")
            
            # Market state adjustments
            try:
                current_risk, current_momentum = current_state
                target_risk, target_momentum = target_state
                
                current_risk = float(current_risk)
                current_momentum = float(current_momentum)
                target_risk = float(target_risk)
                target_momentum = float(target_momentum)
                
                risk_adjustment = 1.0 - (current_risk * 0.3)
                
                momentum_alignment = 1.0
                if signal.trigger_type in ['breakout', 'cascade']:
                    if target_momentum * current_momentum > 0:
                        momentum_alignment = 1.1
                        
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - State adjustment calculation failed: {e}")
                raise CriticalBotError(f"State adjustment calculation failed: {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Market state adjustment failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Market state adjustment failed: {e}")
            
            # Final position size calculation
            try:
                base_allocation = kelly_fraction * chemistry_multiplier * risk_adjustment * momentum_alignment
                
                # Apply diversification factor
                diversification_factor = diversification_matrix.get(signal.ticker, 1.0)
                if diversification_factor is None:
                    logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - None diversification factor for {signal.ticker}")
                    raise CriticalBotError(f"None diversification factor for {signal.ticker}")
                
                diversification_factor = float(diversification_factor)
                final_allocation = base_allocation * diversification_factor
                
                # Enforce maximum position size limit
                final_allocation = min(final_allocation, self.max_position_size)
                
                # Calculate dollar amount and shares - CORRECTED: Use real price
                dollar_amount = min(final_allocation * total_capital, available_capital)
                shares = int(dollar_amount / current_price) if current_price > 0 else 0
                
                # Minimum position validation
                if shares < 1 or dollar_amount < 100:
                    shares = 0
                    dollar_amount = 0.0
                    final_allocation = 0.0
                else:
                    # Recalculate exact dollar amount based on shares
                    dollar_amount = shares * current_price
                    final_allocation = dollar_amount / total_capital
                
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Final calculation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Final position sizing calculation failed: {e}")
            
            try:
                result = {
                    'shares': shares,
                    'dollar_amount': dollar_amount,
                    'weight': final_allocation,
                    'strategy': {
                        'kelly_fraction': kelly_fraction,
                        'chemistry_multiplier': chemistry_multiplier,
                        'risk_adjustment': risk_adjustment,
                        'momentum_alignment': momentum_alignment,
                        'diversification_factor': diversification_factor,
                        'expected_payoff_ratio': expected_payoff_ratio
                    }
                }
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Result construction failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Position sizing result construction failed: {e}")
            
            return result
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Position sizing failed for {getattr(signal, 'ticker', 'unknown')}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Geological position sizing failed: {e}")

    def _calculate_precision_entry_points(
        self,
        signal: TrailheadSignal,
        chemistry: AssetChemistry,
        current_price: float,
        current_state: Tuple[float, float],
        target_state: Tuple[float, float]
    ) -> Dict:
        """Calculate precision entry and exit points based on geological analysis."""
        try:
            # Validate inputs
            if signal is None:
                logger.error("FAIL-FAST: _calculate_precision_entry_points() - signal parameter is None")
                raise CriticalBotError("Cannot calculate entry points - signal is None")
            
            if chemistry is None:
                logger.error("FAIL-FAST: _calculate_precision_entry_points() - chemistry parameter is None")
                raise CriticalBotError("Cannot calculate entry points - chemistry is None")
            
            if current_state is None or target_state is None:
                logger.error("FAIL-FAST: _calculate_precision_entry_points() - State parameters are None")
                raise CriticalBotError("Cannot calculate entry points - state parameters are None")
            
            # Validate current price
            try:
                current_price = float(current_price)
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Invalid current_price: {current_price}, error: {e}")
                raise CriticalBotError(f"Invalid current_price for entry points: {current_price}")
            
            if current_price <= 0:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Non-positive current_price: {current_price}")
                raise CriticalBotError(f"Non-positive current_price for entry points: {current_price}")
            
            if np.isnan(current_price) or np.isinf(current_price):
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - NaN/Inf current_price: {current_price}")
                raise CriticalBotError(f"NaN/Inf current_price for entry points: {current_price}")
            
            # Validate signal attributes
            if not hasattr(signal, 'trigger_type') or signal.trigger_type is None:
                logger.error("FAIL-FAST: _calculate_precision_entry_points() - Signal missing trigger_type")
                raise CriticalBotError("Signal missing trigger_type for entry points")
            
            if not hasattr(signal, 'pressure_score') or signal.pressure_score is None:
                logger.error("FAIL-FAST: _calculate_precision_entry_points() - Signal missing pressure_score")
                raise CriticalBotError("Signal missing pressure_score for entry points")
            
            if not hasattr(signal, 'fragility_score') or signal.fragility_score is None:
                logger.error("FAIL-FAST: _calculate_precision_entry_points() - Signal missing fragility_score")
                raise CriticalBotError("Signal missing fragility_score for entry points")
            
            if not hasattr(signal, 'composite_score') or signal.composite_score is None:
                logger.error("FAIL-FAST: _calculate_precision_entry_points() - Signal missing composite_score")
                raise CriticalBotError("Signal missing composite_score for entry points")
            
            # Validate chemistry attributes
            if not hasattr(chemistry, 'chemistry_type') or chemistry.chemistry_type is None:
                logger.error("FAIL-FAST: _calculate_precision_entry_points() - Chemistry missing chemistry_type")
                raise CriticalBotError("Chemistry missing chemistry_type for entry points")
            
            # Entry price based on trigger type
            try:
                entry_adjustment = 0.0
                
                if signal.trigger_type == 'breakout':
                    entry_adjustment = 0.002  # 0.2% above current price
                elif signal.trigger_type == 'squeeze':
                    entry_adjustment = 0.0
                elif signal.trigger_type == 'reversal':
                    entry_adjustment = -0.005  # 0.5% below current price
                elif signal.trigger_type == 'cascade':
                    entry_adjustment = 0.001  # 0.1% above for immediate entry
                else:
                    logger.warning(f"Unknown trigger_type {signal.trigger_type}, using default adjustment")
                    entry_adjustment = 0.0
                
                entry_price = current_price * (1.0 + entry_adjustment)
                
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Entry price calculation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Entry price calculation failed: {e}")
            
            # Target price based on pressure and fragility
            try:
                pressure_score = float(signal.pressure_score)
                fragility_score = float(signal.fragility_score)
                pressure_energy = pressure_score * fragility_score
                base_target_multiplier = 1.0 + (pressure_energy * 0.15)  # Up to 15% gain
                
                # Chemistry-specific target adjustments
                chemistry_target_multiplier = {
                    'volatile_compound': 1.3,
                    'catalyst_accelerant': 1.2,
                    'phase_change': 1.15,
                    'noble_gas': 1.05
                }.get(chemistry.chemistry_type, 1.1)
                
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Target multiplier calculation failed: {e}")
                raise CriticalBotError(f"Target multiplier calculation failed: {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Target price base calculation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Target price base calculation failed: {e}")
            
            # Market state target adjustments
            try:
                current_risk, current_momentum = current_state
                target_risk, target_momentum = target_state
                
                current_risk = float(current_risk)
                target_risk = float(target_risk)
                
                if target_risk < current_risk:
                    state_multiplier = 1.0 + ((current_risk - target_risk) * 0.2)
                else:
                    state_multiplier = 1.0
                
                final_target_multiplier = base_target_multiplier * chemistry_target_multiplier * state_multiplier
                target_price = entry_price * final_target_multiplier
                
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - State adjustment calculation failed: {e}")
                raise CriticalBotError(f"State adjustment calculation failed: {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Target price calculation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Target price calculation failed: {e}")
            
            # Stop loss based on confidence and chemistry
            try:
                composite_score = float(signal.composite_score)
                base_stop_loss_pct = 0.08 - (composite_score * 0.03)  # 5% to 8% based on confidence
                
                chemistry_stop_multiplier = {
                    'volatile_compound': 1.4,
                    'catalyst_accelerant': 1.1,
                    'phase_change': 0.9,
                    'noble_gas': 0.8
                }.get(chemistry.chemistry_type, 1.0)
                
                final_stop_loss_pct = base_stop_loss_pct * chemistry_stop_multiplier
                stop_loss = entry_price * (1.0 - final_stop_loss_pct)
                
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Stop loss calculation failed: {e}")
                raise CriticalBotError(f"Stop loss calculation failed: {e}")
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Stop loss setup failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Stop loss setup failed: {e}")
            
            # Validate entry points
            try:
                if target_price <= entry_price:
                    target_price = entry_price * 1.05  # Minimum 5% target
                    logger.warning(f"Adjusted target price to minimum 5% above entry for signal")
                
                if stop_loss >= entry_price:
                    stop_loss = entry_price * 0.95  # Minimum 5% stop
                    logger.warning(f"Adjusted stop loss to minimum 5% below entry for signal")
                
                # Calculate risk-reward ratio
                risk_reward_ratio = (target_price - entry_price) / (entry_price - stop_loss)
                
                if risk_reward_ratio <= 0:
                    logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Invalid risk-reward ratio: {risk_reward_ratio}")
                    raise CriticalBotError(f"Invalid risk-reward ratio calculated: {risk_reward_ratio}")
                
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Entry point validation failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Entry point validation failed: {e}")
            
            try:
                result = {
                    'entry_price': round(entry_price, 2),
                    'target_price': round(target_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'risk_reward_ratio': risk_reward_ratio,
                    'entry_strategy': {
                        'trigger_type': signal.trigger_type,
                        'entry_adjustment': entry_adjustment,
                        'pressure_energy': pressure_energy,
                        'final_target_multiplier': final_target_multiplier,
                        'final_stop_loss_pct': final_stop_loss_pct
                    }
                }
            except Exception as e:
                logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Result construction failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Entry points result construction failed: {e}")
            
            return result
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Entry point calculation failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Precision entry point calculation failed: {e}")

    def _validate_geological_portfolio(self, positions: List[PortfolioPosition], total_capital: float) -> Dict:
        """Validate geological portfolio formation meets all safety constraints."""
        try:
            if positions is None:
                logger.error("FAIL-FAST: _validate_geological_portfolio() - positions parameter is None")
                raise CriticalBotError("Cannot validate geological portfolio - positions list is None")
            
            if not isinstance(positions, list):
                logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Invalid positions type: {type(positions)}")
                raise CriticalBotError(f"Invalid positions type for geological validation: {type(positions)}")
            
            try:
                total_capital = float(total_capital)
            except (ValueError, TypeError) as e:
                logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Invalid total_capital: {total_capital}, error: {e}")
                raise CriticalBotError(f"Invalid total_capital for geological validation: {total_capital}")
            
            if total_capital <= 0:
                logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Non-positive total_capital: {total_capital}")
                raise CriticalBotError(f"Non-positive total_capital for geological validation: {total_capital}")
            
            validation_result = {'valid': True, 'reasons': []}
            
            if not positions:
                return validation_result
            
            # Check total allocation
            try:
                total_weight = 0.0
                for position in positions:
                    if position is None:
                        logger.error("FAIL-FAST: _validate_geological_portfolio() - None position in positions list")
                        raise CriticalBotError("None position found in portfolio positions list")
                    
                    if not hasattr(position, 'weight') or position.weight is None:
                        logger.error("FAIL-FAST: _validate_geological_portfolio() - Position missing or None weight")
                        raise CriticalBotError("Position missing or None weight in geological validation")
                    
                    try:
                        weight = float(position.weight)
                    except (ValueError, TypeError) as e:
                        logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Invalid position weight: {position.weight}, error: {e}")
                        raise CriticalBotError(f"Invalid position weight in geological validation: {position.weight}")
                    
                    total_weight += weight
                
                if total_weight > 1.0:
                    validation_result['valid'] = False
                    validation_result['reasons'].append(f"Portfolio over-allocated: {total_weight:.1%}")
                    
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Total allocation check failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Total allocation validation failed: {e}")
            
            # Check individual position sizes
            for i, position in enumerate(positions):
                try:
                    if not hasattr(position, 'ticker') or position.ticker is None:
                        logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Position {i} missing or None ticker")
                        raise CriticalBotError(f"Position {i} missing or None ticker")
                    
                    # Validate weight
                    if not hasattr(position, 'weight') or position.weight is None:
                        logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Position {position.ticker} missing or None weight")
                        raise CriticalBotError(f"Position {position.ticker} missing or None weight")
                    
                    weight = float(position.weight)
                    if weight > 0.20:  # Max 20% per position
                        validation_result['valid'] = False
                        validation_result['reasons'].append(f"{position.ticker} position too large: {weight:.1%}")
                    
                    # Validate prices
                    if not hasattr(position, 'target_price') or position.target_price is None:
                        logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Position {position.ticker} missing or None target_price")
                        raise CriticalBotError(f"Position {position.ticker} missing or None target_price")
                    
                    if not hasattr(position, 'entry_price') or position.entry_price is None:
                        logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Position {position.ticker} missing or None entry_price")
                        raise CriticalBotError(f"Position {position.ticker} missing or None entry_price")
                    
                    if not hasattr(position, 'stop_loss') or position.stop_loss is None:
                        logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Position {position.ticker} missing or None stop_loss")
                        raise CriticalBotError(f"Position {position.ticker} missing or None stop_loss")
                    
                    target_price = float(position.target_price)
                    entry_price = float(position.entry_price)
                    stop_loss = float(position.stop_loss)
                    
                    if target_price <= entry_price:
                        validation_result['valid'] = False
                        validation_result['reasons'].append(f"{position.ticker} invalid target price")
                    
                    if stop_loss >= entry_price:
                        validation_result['valid'] = False
                        validation_result['reasons'].append(f"{position.ticker} invalid stop loss")
                    
                    # Validate position size and value
                    if not hasattr(position, 'position_size') or position.position_size is None:
                        logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Position {position.ticker} missing or None position_size")
                        raise CriticalBotError(f"Position {position.ticker} missing or None position_size")
                    
                    position_size = int(position.position_size)
                    position_value = position_size * entry_price
                    
                    if position_value < 10:  # IBKR minimum
                        validation_result['valid'] = False
                        validation_result['reasons'].append(f"{position.ticker} below minimum order value")
                    
                except CriticalBotError:
                    raise
                except (ValueError, TypeError) as e:
                    logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Position validation failed for {getattr(position, 'ticker', f'position_{i}')}: {e}")
                    raise CriticalBotError(f"Position validation failed: {e}")
                except Exception as e:
                    logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Position validation error for {getattr(position, 'ticker', f'position_{i}')}: {type(e).__name__}: {e}")
                    raise CriticalBotError(f"Position validation system error: {e}")
            
            # Check chemistry diversification
            try:
                chemistry_counts = {}
                for position in positions:
                    if not hasattr(position, 'chemistry_type') or position.chemistry_type is None:
                        logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Position {getattr(position, 'ticker', 'unknown')} missing or None chemistry_type")
                        raise CriticalBotError(f"Position missing or None chemistry_type")
                    
                    chemistry_type = position.chemistry_type
                    chemistry_counts[chemistry_type] = chemistry_counts.get(chemistry_type, 0) + 1
                
                if chemistry_counts:
                    max_chemistry_count = max(chemistry_counts.values())
                    if max_chemistry_count > len(positions) * 0.6:  # No more than 60% in one chemistry
                        validation_result['valid'] = False
                        validation_result['reasons'].append("Excessive concentration in one chemistry type")
                        
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Chemistry diversification check failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Chemistry diversification validation failed: {e}")
            
            return validation_result
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Portfolio validation failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Geological portfolio validation failed: {e}")
