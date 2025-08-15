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
        if ibkr_executor is None:
            logger.error("FAIL-FAST: PortfolioSynthesizer.__init__() - ibkr_executor is None")
            raise CriticalBotError("Cannot initialize PortfolioSynthesizer - IBKR executor is None")
        
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
            
            # Validate state bounds
            for name, value in [
                ('current_risk', current_risk), ('target_risk', target_risk)
            ]:
                if not (0 <= value <= 1):
                    logger.error(f"FAIL-FAST: synthesize_portfolio() - {name} out of bounds: {value}")
                    raise CriticalBotError(f"Geological {name} exceeds safe limits: {value}")
            
            for name, value in [
                ('current_momentum', current_momentum), ('target_momentum', target_momentum)
            ]:
                if not (-1 <= value <= 1):
                    logger.error(f"FAIL-FAST: synthesize_portfolio() - {name} out of bounds: {value}")
                    raise CriticalBotError(f"Geological {name} exceeds safe limits: {value}")
            
            # Get real capital and positions from IBKR
            try:
                current_capital = self._get_current_capital()
                buying_power = self._get_buying_power()
                current_positions = self._get_current_positions()
            except Exception as e:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - IBKR account data retrieval failed: {type(e).__name__}: {e}")
                raise CriticalBotError(f"Cannot access account data for portfolio synthesis: {e}")
            
            logger.info(f"Geological portfolio synthesis: Capital=${current_capital:,.2f}, "
                       f"Buying Power=${buying_power:,.2f}, "
                       f"Current Positions={len(current_positions)}, "
                       f"Pressure Points={len(trailhead_signals)}")
            
            # Filter and validate signals
            validated_signals = self._validate_trailhead_signals(trailhead_signals)
            if not validated_signals:
                logger.warning("No validated geological pressure points found")
                return []
            
            # Calculate geological diversification strategy
            diversification_matrix = self._calculate_geological_diversification(
                validated_signals, chemistry_map, current_state, target_state
            )
            
            # Synthesize portfolio positions
            portfolio_positions = []
            total_allocated = 0.0
            
            for signal in validated_signals[:self.max_positions]:
                try:
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
                    current_price = self._get_market_price(signal.ticker)
                    if current_price is None or current_price <= 0:
                        logger.warning(f"No market data for geological formation {signal.ticker}")
                        continue
                    
                    # Calculate geological position sizing
                    remaining_capital = buying_power - total_allocated
                    if remaining_capital < 100:  # Minimum $100 position
                        logger.info("Insufficient geological survey funding for additional formations")
                        break
                    
                    position_allocation = self._calculate_geological_position_sizing(
                        signal, chemistry, current_state, target_state,
                        remaining_capital, current_capital, diversification_matrix,
                        current_price  # CORRECTED: Pass real price
                    )
                    
                    if position_allocation['shares'] == 0:
                        logger.debug(f"Zero position size calculated for {signal.ticker}")
                        continue
                    
                    # Calculate precision entry and exit points
                    entry_points = self._calculate_precision_entry_points(
                        signal, chemistry, current_price, current_state, target_state
                    )
                    
                    # Create portfolio position
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
                    
                    portfolio_positions.append(position)
                    total_allocated += position_allocation['dollar_amount']
                    
                    logger.info(f"Geological formation synthesized: {signal.ticker} "
                               f"({position_allocation['shares']} shares @ ${entry_points['entry_price']:.2f})")
                    
                except Exception as e:
                    logger.error(f"FAIL-FAST: synthesize_portfolio() - Position synthesis failed for {signal.ticker}: {type(e).__name__}: {e}")
                    continue
            
            # Validate final portfolio
            portfolio_validation = self._validate_geological_portfolio(portfolio_positions, current_capital)
            if not portfolio_validation['valid']:
                logger.error(f"FAIL-FAST: synthesize_portfolio() - Portfolio validation failed: {portfolio_validation['reasons']}")
                raise CriticalBotError(f"Geological portfolio formation invalid: {portfolio_validation['reasons']}")
            
            # Portfolio synthesis summary
            total_weight = sum(p.weight for p in portfolio_positions)
            avg_confidence = np.mean([p.confidence for p in portfolio_positions]) if portfolio_positions else 0
            
            logger.info(f"Geological portfolio synthesized: {len(portfolio_positions)} formations, "
                       f"${total_allocated:,.2f} allocated ({total_weight:.1%} of capital), "
                       f"Avg confidence: {avg_confidence:.2f}")
            
            return portfolio_positions
            
        except CriticalBotError:
            raise  # Re-raise critical errors
        except Exception as e:
            logger.error(f"FAIL-FAST: _validate_geological_portfolio() - Portfolio validation failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Geological portfolio validation failed: {e}")

    def _get_current_capital(self) -> float:
        """Get current portfolio value from IBKR account."""
        try:
            now = datetime.now()
            if (self._last_capital_check and 
                (now - self._last_capital_check).seconds < 60 and
                self._cached_capital is not None):
                return self._cached_capital
            
            capital = self.ibkr_executor.get_account_value()
            if capital is None or capital <= 0:
                logger.error(f"FAIL-FAST: _get_current_capital() - Invalid capital value: {capital}")
                raise CriticalBotError(f"Invalid account capital value: {capital}")
            
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
            buying_power = self.ibkr_executor.get_buying_power()
            if buying_power is None or buying_power < 0:
                logger.error(f"FAIL-FAST: _get_buying_power() - Invalid buying power: {buying_power}")
                raise CriticalBotError(f"Invalid buying power value: {buying_power}")
            
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
            
            positions = self.ibkr_executor.get_current_positions()
            if positions is None:
                logger.error("FAIL-FAST: _get_current_positions() - Positions data is None")
                raise CriticalBotError("Current positions data is None")
            
            if not isinstance(positions, dict):
                logger.error(f"FAIL-FAST: _get_current_positions() - Invalid positions type: {type(positions)}")
                raise CriticalBotError(f"Invalid positions data type: {type(positions)}")
            
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
            if ticker is None or not isinstance(ticker, str):
                logger.error(f"FAIL-FAST: _get_market_price() - Invalid ticker: {ticker}")
                raise CriticalBotError(f"Invalid ticker for price lookup: {ticker}")
            
            if len(ticker.strip()) == 0:
                logger.error("FAIL-FAST: _get_market_price() - Empty ticker string")
                raise CriticalBotError("Empty ticker string for price lookup")
            
            price = self.ibkr_executor.get_market_data(ticker)
            if price is None:
                logger.warning(f"No market data available for {ticker}")
                return None
            
            if not isinstance(price, (int, float)) or price <= 0:
                logger.error(f"FAIL-FAST: _get_market_price() - Invalid price for {ticker}: {price}")
                raise CriticalBotError(f"Invalid market price for {ticker}: {price}")
            
            if np.isnan(price) or np.isinf(price):
                logger.error(f"FAIL-FAST: _get_market_price() - NaN/Inf price for {ticker}: {price}")
                raise CriticalBotError(f"NaN/Inf market price for {ticker}: {price}")
            
            return float(price)
            
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
                if position is None:
                    logger.error(f"FAIL-FAST: validate_portfolio() - Position {i} is None")
                    raise CriticalBotError(f"Position {i} is None in portfolio")
                
                if not isinstance(position, PortfolioPosition):
                    logger.error(f"FAIL-FAST: validate_portfolio() - Invalid position type at index {i}: {type(position)}")
                    raise CriticalBotError(f"Invalid position type at index {i}: {type(position)}")
            
            try:
                current_capital = self._get_current_capital()
            except Exception as e:
                logger.error(f"FAIL-FAST: validate_portfolio() - Cannot get capital for validation: {e}")
                raise CriticalBotError(f"Portfolio validation failed - cannot access capital: {e}")
            
            # Comprehensive validation
            validation_result = self._validate_geological_portfolio(positions, current_capital)
            
            if not validation_result['valid']:
                logger.error(f"FAIL-FAST: validate_portfolio() - Portfolio validation failed: {validation_result['reasons']}")
                return False
            
            logger.info("Portfolio validation passed")
            return True
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"FAIL-FAST: validate_portfolio() - Portfolio validation error: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Portfolio validation system error: {e}")FAST: synthesize_portfolio() - Catastrophic portfolio synthesis failure: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Portfolio synthesis system failure: {e}")

    def _validate_trailhead_signals(self, signals: List[TrailheadSignal]) -> List[TrailheadSignal]:
        """Validate and filter geological pressure point signals."""
        try:
            if not signals:
                return []
            
            validated_signals = []
            
            for signal in signals:
                try:
                    if signal is None:
                        continue
                    
                    if not isinstance(signal, TrailheadSignal):
                        continue
                    
                    if signal.ticker is None or not isinstance(signal.ticker, str):
                        continue
                    
                    if signal.composite_score is None or np.isnan(signal.composite_score):
                        continue
                    
                    if signal.composite_score < self.min_confidence:
                        continue
                    
                    if not (0 <= signal.pressure_score <= 1):
                        continue
                    
                    if not (0 <= signal.fragility_score <= 1):
                        continue
                    
                    valid_triggers = {'squeeze', 'breakout', 'reversal', 'cascade'}
                    if signal.trigger_type not in valid_triggers:
                        continue
                    
                    validated_signals.append(signal)
                    
                except Exception:
                    continue
            
            return validated_signals
            
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
            diversification_matrix = {}
            
            # Count chemistry and trigger types
            chemistry_counts = {}
            trigger_counts = {}
            
            for signal in signals:
                chemistry = chemistry_map.get(signal.ticker)
                if chemistry:
                    chemistry_type = chemistry.chemistry_type
                    chemistry_counts[chemistry_type] = chemistry_counts.get(chemistry_type, 0) + 1
                    trigger_counts[signal.trigger_type] = trigger_counts.get(signal.trigger_type, 0) + 1
            
            total_signals = len(signals)
            
            for signal in signals:
                chemistry = chemistry_map.get(signal.ticker)
                if chemistry is None:
                    diversification_matrix[signal.ticker] = 0.5
                    continue
                
                diversification_factor = 1.0
                
                # Chemistry type diversification penalty
                chemistry_concentration = chemistry_counts[chemistry.chemistry_type] / total_signals
                if chemistry_concentration > 0.4:
                    diversification_factor *= (1.0 - (chemistry_concentration - 0.4))
                
                # Trigger type diversification penalty
                trigger_concentration = trigger_counts[signal.trigger_type] / total_signals
                if trigger_concentration > 0.5:
                    diversification_factor *= (1.0 - (trigger_concentration - 0.5))
                
                # Market state alignment bonus
                current_risk, current_momentum = current_state
                target_risk, target_momentum = target_state
                
                if chemistry.chemistry_type == 'volatile_compound' and target_risk < current_risk:
                    diversification_factor *= 1.1
                elif chemistry.chemistry_type == 'catalyst_accelerant' and abs(target_momentum) > abs(current_momentum):
                    diversification_factor *= 1.1
                elif chemistry.chemistry_type == 'phase_change' and abs(target_risk - 0.5) < abs(current_risk - 0.5):
                    diversification_factor *= 1.1
                
                diversification_matrix[signal.ticker] = max(0.3, min(1.5, diversification_factor))
            
            return diversification_matrix
            
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
            # Base Kelly Criterion calculation
            win_probability = signal.composite_score
            
            # Calculate expected payoff based on pressure and fragility
            pressure_energy = signal.pressure_score * signal.fragility_score
            expected_payoff_ratio = 1.0 + (pressure_energy * 2.0)  # Up to 3:1 payoff
            
            # Kelly fraction with geological adjustments
            if expected_payoff_ratio > 1.0:
                kelly_fraction = (win_probability * expected_payoff_ratio - (1 - win_probability)) / expected_payoff_ratio
            else:
                kelly_fraction = 0.0
            
            # Apply geological safety factor
            kelly_fraction = min(kelly_fraction, win_probability * 0.5)
            
            # Chemical reaction amplification
            chemistry_multiplier = {
                'volatile_compound': 1.2,
                'catalyst_accelerant': 1.1,
                'phase_change': 1.0,
                'noble_gas': 0.8
            }.get(chemistry.chemistry_type, 1.0)
            
            # Market state adjustments
            current_risk, current_momentum = current_state
            target_risk, target_momentum = target_state
            
            risk_adjustment = 1.0 - (current_risk * 0.3)
            
            momentum_alignment = 1.0
            if signal.trigger_type in ['breakout', 'cascade']:
                if target_momentum * current_momentum > 0:
                    momentum_alignment = 1.1
            
            # Final position size calculation
            base_allocation = kelly_fraction * chemistry_multiplier * risk_adjustment * momentum_alignment
            
            # Apply diversification factor
            diversification_factor = diversification_matrix.get(signal.ticker, 1.0)
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
            
            return {
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
            logger.error(f"FAIL-FAST: _calculate_geological_position_sizing() - Position sizing failed for {signal.ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Geological position sizing failed for {signal.ticker}: {e}")

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
            # Entry price based on trigger type
            entry_adjustment = 0.0
            
            if signal.trigger_type == 'breakout':
                entry_adjustment = 0.002  # 0.2% above current price
            elif signal.trigger_type == 'squeeze':
                entry_adjustment = 0.0
            elif signal.trigger_type == 'reversal':
                entry_adjustment = -0.005  # 0.5% below current price
            elif signal.trigger_type == 'cascade':
                entry_adjustment = 0.001  # 0.1% above for immediate entry
            
            entry_price = current_price * (1.0 + entry_adjustment)
            
            # Target price based on pressure and fragility
            pressure_energy = signal.pressure_score * signal.fragility_score
            base_target_multiplier = 1.0 + (pressure_energy * 0.15)  # Up to 15% gain
            
            # Chemistry-specific target adjustments
            chemistry_target_multiplier = {
                'volatile_compound': 1.3,
                'catalyst_accelerant': 1.2,
                'phase_change': 1.15,
                'noble_gas': 1.05
            }.get(chemistry.chemistry_type, 1.1)
            
            # Market state target adjustments
            current_risk, current_momentum = current_state
            target_risk, target_momentum = target_state
            
            if target_risk < current_risk:
                state_multiplier = 1.0 + ((current_risk - target_risk) * 0.2)
            else:
                state_multiplier = 1.0
            
            final_target_multiplier = base_target_multiplier * chemistry_target_multiplier * state_multiplier
            target_price = entry_price * final_target_multiplier
            
            # Stop loss based on confidence and chemistry
            base_stop_loss_pct = 0.08 - (signal.composite_score * 0.03)  # 5% to 8% based on confidence
            
            chemistry_stop_multiplier = {
                'volatile_compound': 1.4,
                'catalyst_accelerant': 1.1,
                'phase_change': 0.9,
                'noble_gas': 0.8
            }.get(chemistry.chemistry_type, 1.0)
            
            final_stop_loss_pct = base_stop_loss_pct * chemistry_stop_multiplier
            stop_loss = entry_price * (1.0 - final_stop_loss_pct)
            
            # Validate entry points
            if target_price <= entry_price:
                target_price = entry_price * 1.05  # Minimum 5% target
            
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.95  # Minimum 5% stop
            
            return {
                'entry_price': round(entry_price, 2),
                'target_price': round(target_price, 2),
                'stop_loss': round(stop_loss, 2),
                'risk_reward_ratio': (target_price - entry_price) / (entry_price - stop_loss),
                'entry_strategy': {
                    'trigger_type': signal.trigger_type,
                    'entry_adjustment': entry_adjustment,
                    'pressure_energy': pressure_energy,
                    'final_target_multiplier': final_target_multiplier,
                    'final_stop_loss_pct': final_stop_loss_pct
                }
            }
            
        except Exception as e:
            logger.error(f"FAIL-FAST: _calculate_precision_entry_points() - Entry point calculation failed for {signal.ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Precision entry point calculation failed for {signal.ticker}: {e}")

    def _validate_geological_portfolio(self, positions: List[PortfolioPosition], total_capital: float) -> Dict:
        """Validate geological portfolio formation meets all safety constraints."""
        try:
            validation_result = {'valid': True, 'reasons': []}
            
            if not positions:
                return validation_result
            
            # Check total allocation
            total_weight = sum(p.weight for p in positions)
            if total_weight > 1.0:
                validation_result['valid'] = False
                validation_result['reasons'].append(f"Portfolio over-allocated: {total_weight:.1%}")
            
            # Check individual position sizes
            for position in positions:
                if position.weight > 0.20:  # Max 20% per position
                    validation_result['valid'] = False
                    validation_result['reasons'].append(f"{position.ticker} position too large: {position.weight:.1%}")
                
                if position.target_price <= position.entry_price:
                    validation_result['valid'] = False
                    validation_result['reasons'].append(f"{position.ticker} invalid target price")
                
                if position.stop_loss >= position.entry_price:
                    validation_result['valid'] = False
                    validation_result['reasons'].append(f"{position.ticker} invalid stop loss")
                
                position_value = position.position_size * position.entry_price
                if position_value < 10:  # IBKR minimum
                    validation_result['valid'] = False
                    validation_result['reasons'].append(f"{position.ticker} below minimum order value")
            
            # Check chemistry diversification
            chemistry_counts = {}
            for position in positions:
                chemistry_counts[position.chemistry_type] = chemistry_counts.get(position.chemistry_type, 0) + 1
            
            max_chemistry_count = max(chemistry_counts.values()) if chemistry_counts else 0
            if max_chemistry_count > len(positions) * 0.6:  # No more than 60% in one chemistry
                validation_result['valid'] = False
                validation_result['reasons'].append("Excessive concentration in one chemistry type")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"FAIL-
