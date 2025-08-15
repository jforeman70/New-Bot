import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from collections import deque

from Market_State_Calculator import CriticalBotError

logger = logging.getLogger(__name__)

@dataclass
class StatePrediction:
    """Predicted future market state with confidence."""
    target_risk: float
    target_momentum: float
    confidence: float
    time_horizon: int  # Hours until predicted state
    cycle_phase: str
    metadata: Dict

class StatePredictor:
    """Predicts future market states based on cyclical patterns and regime analysis."""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.state_history = deque(maxlen=lookback_periods)
        self.cycle_memory = {}
        
    def predict_next_state(
        self,
        current_state: Tuple[float, float],
        state_history: Optional[List[Tuple[float, float, datetime]]] = None
    ) -> StatePrediction:
        """
        Predict where market state will move next based on cycles and patterns.
        
        Args:
            current_state: Current (risk, momentum) from calculate_market_state
            state_history: Optional historical states for pattern matching
            
        Returns:
            StatePrediction with target coordinates and confidence
        """
        
        try:
            # Input validation
            if current_state is None:
                logger.error("Null current_state in predict_next_state")
                raise ValueError("Current state cannot be None")
                
            if not isinstance(current_state, tuple):
                logger.error(f"Invalid current_state type: {type(current_state)}")
                raise ValueError(f"Current state must be tuple, got {type(current_state)}")
                
            if len(current_state) != 2:
                logger.error(f"Invalid current_state length: {len(current_state)}")
                raise ValueError(f"Current state must have 2 elements, got {len(current_state)}")
            
            current_risk, current_momentum = current_state
            
            if current_risk is None or current_momentum is None:
                logger.error(f"Null values in state: risk={current_risk}, momentum={current_momentum}")
                raise ValueError("State values cannot be None")
                
            if not isinstance(current_risk, (int, float)) or not isinstance(current_momentum, (int, float)):
                logger.error(f"Non-numeric state values: risk={type(current_risk)}, momentum={type(current_momentum)}")
                raise ValueError("State values must be numeric")
            
            if not (0 <= current_risk <= 1):
                logger.error(f"Risk out of range: {current_risk}")
                raise ValueError(f"Risk must be 0-1, got {current_risk}")
                
            if not (-1 <= current_momentum <= 1):
                logger.error(f"Momentum out of range: {current_momentum}")
                raise ValueError(f"Momentum must be -1 to 1, got {current_momentum}")
            
            # Update history if provided
            if state_history is not None:
                if not isinstance(state_history, list):
                    logger.error(f"Invalid state_history type: {type(state_history)}")
                    raise ValueError("State history must be a list")
                    
                try:
                    self.state_history.extend(state_history)
                except Exception as e:
                    logger.error(f"Failed to update state history: {e}")
                    raise ValueError(f"Invalid state history format: {e}")
            
            # Identify current cycle phase
            try:
                cycle_phase = self._identify_cycle_phase(current_risk, current_momentum)
            except Exception as e:
                logger.error(f"Failed to identify cycle phase: {e}")
                raise CriticalBotError(f"Cycle phase identification failed: {e}")
            
            # Calculate orbital trajectory
            try:
                next_position = self._calculate_orbital_position(
                    current_risk, current_momentum, cycle_phase
                )
            except Exception as e:
                logger.error(f"Failed to calculate orbital position: {e}")
                raise CriticalBotError(f"Orbital calculation failed: {e}")
            
            # Apply mean reversion forces
            try:
                reversion_adjusted = self._apply_mean_reversion(
                    next_position, current_risk, current_momentum
                )
            except Exception as e:
                logger.error(f"Failed to apply mean reversion: {e}")
                raise CriticalBotError(f"Mean reversion calculation failed: {e}")
            
            # Detect regime changes
            try:
                regime_shift = self._detect_regime_shift(current_risk, current_momentum)
            except Exception as e:
                logger.error(f"Failed to detect regime shift: {e}")
                raise CriticalBotError(f"Regime detection failed: {e}")
            
            # Final prediction with regime adjustment
            try:
                if regime_shift:
                    target_risk, target_momentum = self._apply_regime_shift(
                        reversion_adjusted, regime_shift
                    )
                    confidence = 0.65
                else:
                    target_risk, target_momentum = reversion_adjusted
                    confidence = 0.80
            except Exception as e:
                logger.error(f"Failed to apply regime shift: {e}")
                raise CriticalBotError(f"Regime adjustment failed: {e}")
            
            # Validate outputs before clipping
            if target_risk is None or target_momentum is None:
                logger.error(f"Null prediction values: risk={target_risk}, momentum={target_momentum}")
                raise CriticalBotError("Prediction calculation returned null values")
            
            # Bound predictions to valid ranges
            target_risk = np.clip(target_risk, 0, 1)
            target_momentum = np.clip(target_momentum, -1, 1)
            
            # Adjust confidence based on state extremes
            if current_risk > 0.8 or current_risk < 0.2:
                confidence *= 1.15
            if abs(current_momentum) > 0.7:
                confidence *= 1.10
            
            confidence = min(0.95, confidence)
            
            # Time horizon based on volatility
            try:
                time_horizon = self._calculate_time_horizon(current_risk)
            except Exception as e:
                logger.error(f"Failed to calculate time horizon: {e}")
                raise CriticalBotError(f"Time horizon calculation failed: {e}")
            
            return StatePrediction(
                target_risk=target_risk,
                target_momentum=target_momentum,
                confidence=confidence,
                time_horizon=time_horizon,
                cycle_phase=cycle_phase,
                metadata={
                    'current_risk': current_risk,
                    'current_momentum': current_momentum,
                    'risk_delta': target_risk - current_risk,
                    'momentum_delta': target_momentum - current_momentum,
                    'regime_shift': regime_shift,
                    'predicted_at': datetime.now().isoformat()
                }
            )
            
        except CriticalBotError:
            raise
        except ValueError as e:
            logger.error(f"Validation error in predict_next_state: {e}")
            logger.error(f"Current state: {current_state if 'current_state' in locals() else 'Not available'}")
            raise CriticalBotError(f"Invalid state prediction input: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in predict_next_state: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise CriticalBotError(f"State prediction system failure: {e}")
    
    def _identify_cycle_phase(self, risk: float, momentum: float) -> str:
        """Identify which phase of market cycle we're in."""
        
        try:
            if risk is None or momentum is None:
                raise ValueError(f"Null values: risk={risk}, momentum={momentum}")
                
            if risk < 0.3 and momentum > 0.3:
                return 'euphoria'
            elif risk < 0.3 and momentum < -0.3:
                return 'disbelief'
            elif risk > 0.7 and momentum < -0.3:
                return 'capitulation'
            elif risk > 0.7 and momentum > 0.3:
                return 'recovery'
            elif risk > 0.5 and abs(momentum) < 0.3:
                return 'anxiety'
            else:
                return 'distribution'
                
        except Exception as e:
            logger.error(f"Error in _identify_cycle_phase: {e}")
            raise CriticalBotError(f"Cycle phase identification failed: {e}")
    
    def _calculate_orbital_position(
        self, 
        risk: float, 
        momentum: float, 
        phase: str
    ) -> Tuple[float, float]:
        """Calculate next position in market's orbital cycle."""
        
        try:
            if any(x is None for x in [risk, momentum, phase]):
                raise ValueError(f"Null parameters: risk={risk}, momentum={momentum}, phase={phase}")
            
            angular_velocities = {
                'euphoria': 0.15,
                'disbelief': 0.25,
                'capitulation': 0.35,
                'recovery': 0.30,
                'anxiety': 0.20,
                'distribution': 0.18
            }
            
            velocity = angular_velocities.get(phase)
            if velocity is None:
                logger.warning(f"Unknown phase: {phase}, using default velocity")
                velocity = 0.20
            
            # Convert to polar coordinates
            radius = np.sqrt(risk**2 + momentum**2)
            angle = np.arctan2(momentum, risk)
            
            # Rotate by velocity
            new_angle = angle + velocity
            
            # Adjust radius based on phase
            if phase in ['capitulation', 'recovery']:
                radius *= 1.05
            elif phase == 'euphoria':
                radius *= 0.95
            
            # Convert back to cartesian
            new_risk = radius * np.cos(new_angle)
            new_momentum = radius * np.sin(new_angle)
            
            if new_risk is None or new_momentum is None:
                raise ValueError("Calculation returned null values")
            
            return new_risk, new_momentum
            
        except Exception as e:
            logger.error(f"Error in _calculate_orbital_position: {e}")
            logger.error(f"Inputs: risk={risk}, momentum={momentum}, phase={phase}")
            raise CriticalBotError(f"Orbital calculation failed: {e}")
    
    def _apply_mean_reversion(
        self,
        predicted: Tuple[float, float],
        current_risk: float,
        current_momentum: float
    ) -> Tuple[float, float]:
        """Apply mean reversion forces to prediction."""
        
        try:
            if predicted is None or len(predicted) != 2:
                raise ValueError(f"Invalid predicted state: {predicted}")
                
            predicted_risk, predicted_momentum = predicted
            
            if any(x is None for x in [predicted_risk, predicted_momentum, current_risk, current_momentum]):
                raise ValueError("Null values in mean reversion calculation")
            
            risk_mean = 0.4
            momentum_mean = 0.0
            
            risk_reversion_force = 0.15 * (risk_mean - current_risk)
            momentum_reversion_force = 0.20 * (momentum_mean - current_momentum)
            
            adjusted_risk = predicted_risk + risk_reversion_force
            adjusted_momentum = predicted_momentum + momentum_reversion_force
            
            return adjusted_risk, adjusted_momentum
            
        except Exception as e:
            logger.error(f"Error in _apply_mean_reversion: {e}")
            raise CriticalBotError(f"Mean reversion calculation failed: {e}")
    
    def _detect_regime_shift(self, risk: float, momentum: float) -> Optional[str]:
        """Detect potential regime changes."""
        
        try:
            if risk is None or momentum is None:
                raise ValueError(f"Null values: risk={risk}, momentum={momentum}")
            
            if risk > 0.85 and momentum < -0.5:
                return 'crash_imminent'
            elif risk < 0.15 and momentum > 0.7:
                return 'bubble_forming'
            elif risk > 0.7 and momentum > 0.5:
                return 'relief_rally'
            elif risk < 0.3 and momentum < -0.5:
                return 'correction_starting'
            
            return None
            
        except Exception as e:
            logger.error(f"Error in _detect_regime_shift: {e}")
            raise CriticalBotError(f"Regime detection failed: {e}")
    
    def _apply_regime_shift(
        self,
        base_prediction: Tuple[float, float],
        regime: str
    ) -> Tuple[float, float]:
        """Adjust prediction for regime changes."""
        
        try:
            if base_prediction is None or len(base_prediction) != 2:
                raise ValueError(f"Invalid base prediction: {base_prediction}")
                
            risk, momentum = base_prediction
            
            if risk is None or momentum is None:
                raise ValueError("Null values in base prediction")
            
            adjustments = {
                'crash_imminent': (0.95, -0.8),
                'bubble_forming': (0.1, 0.9),
                'relief_rally': (0.5, 0.7),
                'correction_starting': (0.6, -0.4)
            }
            
            if regime in adjustments:
                target_risk, target_momentum = adjustments[regime]
                risk = 0.3 * risk + 0.7 * target_risk
                momentum = 0.3 * momentum + 0.7 * target_momentum
            
            return risk, momentum
            
        except Exception as e:
            logger.error(f"Error in _apply_regime_shift: {e}")
            logger.error(f"Regime: {regime}")
            raise CriticalBotError(f"Regime adjustment failed: {e}")
    
    def _calculate_time_horizon(self, risk: float) -> int:
        """Calculate prediction time horizon based on volatility."""
        
        try:
            if risk is None:
                raise ValueError("Risk cannot be None")
                
            if risk > 0.8:
                return 4
            elif risk > 0.6:
                return 12
            elif risk > 0.4:
                return 24
            else:
                return 48
                
        except Exception as e:
            logger.error(f"Error in _calculate_time_horizon: {e}")
            raise CriticalBotError(f"Time horizon calculation failed: {e}")