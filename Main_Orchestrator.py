import logging
import time
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os

from Market_State_Calculator import calculate_market_state, CriticalBotError
from Chemistry_Classifier import classify_asset_chemistry, AssetChemistry
from Trailhead_Detector import detect_trailheads, TrailheadSignal
from Portfolio_Synthesizer import PortfolioSynthesizer, PortfolioPosition
from State_Predictor import StatePredictor, StatePrediction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('catalyst_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """Tracks current system state and positions."""
    market_state: Tuple[float, float]
    predicted_state: Optional[StatePrediction]
    active_positions: List[PortfolioPosition]
    last_update: datetime
    cycle_count: int
    errors_count: int
    total_pnl: float = 0.0
    daily_pnl: float = 0.0

class CatalystTradingSystem:
    """
    Main trading system orchestrator - the geological market navigator.
    
    Core Philosophy: Markets are geological systems with predictable state transitions.
    Position at "trailheads" (pressure points) before major moves occur.
    """
    
    def __init__(
        self,
        fmp_key: str,
        finnhub_key: Optional[str] = None,
        capital: float = 5000.0,
        universe_size: int = 500,
        cycle_interval: int = 30,
        max_positions: int = 10,
        max_position_size: float = 0.15
    ):
        try:
            if not fmp_key:
                logger.error("CatalystTradingSystem.__init__: FMP API key is None or empty")
                raise CriticalBotError("FMP API key is required")
                
            if not isinstance(capital, (int, float)) or capital <= 0:
                logger.error(f"CatalystTradingSystem.__init__: Invalid capital value: {capital}")
                raise CriticalBotError("Capital must be a positive number")
                
            if not isinstance(universe_size, int) or universe_size <= 0:
                logger.error(f"CatalystTradingSystem.__init__: Invalid universe_size: {universe_size}")
                raise CriticalBotError("Universe size must be a positive integer")
            
            self.fmp_key = fmp_key
            self.finnhub_key = finnhub_key
            self.capital = capital
            self.universe_size = universe_size
            self.cycle_interval = cycle_interval
            self.max_positions = max_positions
            self.max_position_size = max_position_size
            
            # Initialize components with fail-fast validation
            try:
                self.state_predictor = StatePredictor()
                if self.state_predictor is None:
                    logger.error("CatalystTradingSystem.__init__: StatePredictor initialization returned None")
                    raise CriticalBotError("StatePredictor initialization failed")
            except Exception as e:
                logger.error(f"CatalystTradingSystem.__init__: StatePredictor initialization failed: {e}")
                raise CriticalBotError(f"StatePredictor initialization failed: {e}")
            
            try:
                self.portfolio_synthesizer = PortfolioSynthesizer(
                    capital=capital,
                    fmp_key=fmp_key,
                    max_positions=max_positions,
                    max_position_size=max_position_size
                )
                if self.portfolio_synthesizer is None:
                    logger.error("CatalystTradingSystem.__init__: PortfolioSynthesizer initialization returned None")
                    raise CriticalBotError("PortfolioSynthesizer initialization failed")
            except Exception as e:
                logger.error(f"CatalystTradingSystem.__init__: PortfolioSynthesizer initialization failed: {e}")
                raise CriticalBotError(f"PortfolioSynthesizer initialization failed: {e}")
            
            # System state
            self.system_state = SystemState(
                market_state=(0.5, 0.0),
                predicted_state=None,
                active_positions=[],
                last_update=datetime.now(),
                cycle_count=0,
                errors_count=0
            )
            
            # Universe tracking
            self.universe_tickers = []
            self.chemistry_cache = {}
            self.last_chemistry_update = None
            self.last_universe_update = None
            
            # Performance tracking
            self.trade_log = []
            self.daily_start_capital = capital
            
            logger.info(f"Catalyst Trading System initialized with ${capital:,.0f}")
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"CatalystTradingSystem.__init__: Unexpected initialization error: {e}")
            raise CriticalBotError(f"System initialization failed: {e}")
    
    def _chemistry_needs_update(self) -> bool:
        """Check if chemistry data needs refresh."""
        try:
            if not self.last_chemistry_update:
                return True
            
            if self.last_chemistry_update is None:
                logger.error("_chemistry_needs_update: last_chemistry_update is None when it shouldn't be")
                raise CriticalBotError("Chemistry update timestamp invalid")
                
            age = datetime.now() - self.last_chemistry_update
            return age > timedelta(minutes=30)  # Refresh every 30 minutes
            
        except Exception as e:
            logger.error(f"_chemistry_needs_update: Error checking chemistry update status: {e}")
            raise CriticalBotError(f"Chemistry update check failed: {e}")
        
    def _shutdown(self):
        """Clean shutdown procedures."""
        try:
            logger.info("🛑 Shutting down Catalyst Trading System")
            
            # Log final statistics
            try:
                runtime = datetime.now() - self.system_state.last_update
                logger.info(f"Final Statistics:")
                logger.info(f"  Cycles: {self.system_state.cycle_count}")
                logger.info(f"  Errors: {self.system_state.errors_count}")
                logger.info(f"  Runtime: {runtime}")
                logger.info(f"  Active Positions: {len(self.system_state.active_positions)}")
            except Exception as e:
                logger.error(f"_shutdown: Error calculating final statistics: {e}")
                # Don't raise here - shutdown should continue
            
            # Save system state for potential recovery
            try:
                if self.system_state is None:
                    logger.error("_shutdown: system_state is None")
                    raise CriticalBotError("System state invalid during shutdown")
                
                state_data = {
                    'timestamp': datetime.now().isoformat(),
                    'market_state': self.system_state.market_state,
                    'cycle_count': self.system_state.cycle_count,
                    'errors_count': self.system_state.errors_count,
                    'active_positions': [],
                    'universe_size': len(self.universe_tickers) if self.universe_tickers else 0,
                    'chemistry_cache_size': len(self.chemistry_cache) if self.chemistry_cache else 0
                }
                
                # Safely process active positions
                if self.system_state.active_positions:
                    for pos in self.system_state.active_positions:
                        try:
                            if not hasattr(pos, 'ticker') or not hasattr(pos, 'chemistry'):
                                logger.error(f"_shutdown: Invalid position object: {pos}")
                                continue
                            
                            pos_data = {
                                'ticker': pos.ticker,
                                'position_size': pos.position_size,
                                'entry_price': pos.entry_price,
                                'confidence': pos.confidence,
                                'chemistry': pos.chemistry.value if hasattr(pos.chemistry, 'value') else str(pos.chemistry)
                            }
                            state_data['active_positions'].append(pos_data)
                        except Exception as e:
                            logger.error(f"_shutdown: Error processing position {pos}: {e}")
                            continue
                
                with open('system_state.json', 'w') as f:
                    json.dump(state_data, f, indent=2)
                
                logger.info("✅ System state saved to system_state.json")
                
            except Exception as e:
                logger.error(f"_shutdown: Failed to save system state: {e}")
                # Don't raise here - shutdown should complete
            
            logger.info("🏁 Catalyst Trading System shutdown complete")
            
        except Exception as e:
            logger.error(f"_shutdown: Critical error during shutdown: {e}")
            # Even shutdown errors shouldn't prevent program termination

    def run_loop(self, max_cycles: Optional[int] = None):
        """
        Main execution loop - runs every 30 seconds.
        
        Trading Logic:
        1. Get current market state [risk, momentum]
        2. Predict destination state (4-48hr horizon)
        3. Update chemistry classifications
        4. Detect trailheads (pressure points)
        5. Synthesize optimal portfolio
        6. Execute trades (placeholder for IBKR)
        7. Monitor risk and performance
        
        Args:
            max_cycles: Maximum cycles to run (None for infinite)
        """
        try:
            logger.info(f"Starting Catalyst Trading System main loop")
            cycles_run = 0
            
            # Initialize universe on startup
            self._update_universe()
            
            while max_cycles is None or cycles_run < max_cycles:
                try:
                    cycle_start = time.time()
                    
                    # Execute complete trading cycle
                    self._execute_trading_cycle()
                    
                    cycles_run += 1
                    self.system_state.cycle_count = cycles_run
                    
                    # Calculate cycle timing
                    elapsed = time.time() - cycle_start
                    sleep_time = max(0, self.cycle_interval - elapsed)
                    
                    if sleep_time > 0:
                        logger.debug(f"Cycle {cycles_run} complete in {elapsed:.1f}s, sleeping {sleep_time:.1f}s")
                        time.sleep(sleep_time)
                    else:
                        logger.warning(f"Cycle {cycles_run} took {elapsed:.1f}s, exceeding {self.cycle_interval}s interval")
                        
                except CriticalBotError as e:
                    logger.error(f"run_loop: Critical error in cycle {cycles_run}: {e}")
                    self.system_state.errors_count += 1
                    
                    # Fail fast on too many critical errors
                    if self.system_state.errors_count > 3:
                        logger.critical(f"run_loop: Too many critical errors ({self.system_state.errors_count}), shutting down system")
                        raise CriticalBotError(f"System instability: {self.system_state.errors_count} critical errors")
                    
                    # Wait 1 minute before retry on critical error
                    logger.info("Waiting 60 seconds before retry...")
                    time.sleep(60)
                    
                except KeyboardInterrupt:
                    logger.info("Shutdown requested by user")
                    break
                    
                except Exception as e:
                    logger.error(f"run_loop: Unexpected error in cycle {cycles_run}: {e}")
                    self.system_state.errors_count += 1
                    
                    if self.system_state.errors_count > 5:
                        logger.critical(f"run_loop: Too many unexpected errors ({self.system_state.errors_count}), shutting down")
                        raise CriticalBotError(f"System instability: {self.system_state.errors_count} unexpected errors")
                    
                    time.sleep(30)  # Short wait for unexpected errors
                    
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"run_loop: Fatal error in main loop: {e}")
            raise CriticalBotError(f"Main loop failed: {e}")
        finally:
            self._shutdown()
            
    def _execute_trading_cycle(self):
        """Execute one complete trading cycle - the core geological analysis."""
        
        try:
            logger.info(f"=== CYCLE {self.system_state.cycle_count + 1} ===")
            
            # Step 1: Calculate current market state (risk, momentum coordinates)
            logger.info("📊 Calculating market state...")
            try:
                market_data = calculate_market_state(self.fmp_key)
                if market_data is None:
                    logger.error("_execute_trading_cycle: calculate_market_state returned None")
                    raise CriticalBotError("Market state calculation returned None")
                
                if not isinstance(market_data, tuple) or len(market_data) < 2:
                    logger.error(f"_execute_trading_cycle: Invalid market_data format: {market_data}")
                    raise CriticalBotError("Market state data format invalid")
                
                current_state = market_data[:2]  # (risk, momentum)
                if current_state is None or len(current_state) != 2:
                    logger.error(f"_execute_trading_cycle: Invalid current_state: {current_state}")
                    raise CriticalBotError("Current state extraction failed")
                
                self.system_state.market_state = current_state
                logger.info(f"Market State: Risk={current_state[0]:.3f}, Momentum={current_state[1]:.3f}")
                
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"_execute_trading_cycle: Market state calculation failed: {e}")
                raise CriticalBotError(f"Market state calculation error: {e}")
            
            # Step 2: Predict destination state using orbital mechanics
            logger.info("🔮 Predicting destination state...")
            try:
                prediction = self.state_predictor.predict_next_state(current_state)
                # Note: prediction can be None for valid reasons, so we don't fail-fast here
                self.system_state.predicted_state = prediction
                
                if prediction:
                    if not hasattr(prediction, 'target_risk') or not hasattr(prediction, 'target_momentum'):
                        logger.error(f"_execute_trading_cycle: Invalid prediction object attributes: {prediction}")
                        raise CriticalBotError("Prediction object missing required attributes")
                    
                    logger.info(f"Predicted State: Risk={prediction.target_risk:.3f}, "
                               f"Momentum={prediction.target_momentum:.3f} "
                               f"(confidence: {prediction.confidence:.2%})")
                else:
                    logger.info("No state prediction available")
                
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"_execute_trading_cycle: State prediction failed: {e}")
                raise CriticalBotError(f"State prediction error: {e}")
            
            # Step 3: Update chemistry classifications (every 5 cycles or on startup)
            if (self.system_state.cycle_count % 5 == 0 or 
                not self.chemistry_cache or 
                self._chemistry_needs_update()):
                logger.info("🧪 Updating chemistry classifications...")
                self._update_chemistry()
            
            # Step 4: Update universe if needed (every 20 cycles)
            if self.system_state.cycle_count % 20 == 0:
                logger.info("🌌 Refreshing trading universe...")
                self._update_universe()
            
            # Step 5: Detect trailheads (pressure points)
            logger.info("🏔️ Detecting trailheads...")
            try:
                signals = self._detect_trailheads()
                if signals is None:
                    logger.error("_execute_trading_cycle: _detect_trailheads returned None")
                    raise CriticalBotError("Trailhead detection returned None")
                
                if not isinstance(signals, list):
                    logger.error(f"_execute_trading_cycle: Invalid signals type: {type(signals)}")
                    raise CriticalBotError("Trailhead signals format invalid")
                
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"_execute_trading_cycle: Trailhead detection failed: {e}")
                raise CriticalBotError(f"Trailhead detection error: {e}")
            
            if not signals:
                logger.info("No trailhead signals detected - market stable")
                return
            
            logger.info(f"Found {len(signals)} trailhead signals")
            for i, signal in enumerate(signals[:3]):  # Log top 3
                try:
                    if not hasattr(signal, 'ticker') or not hasattr(signal, 'chemistry'):
                        logger.error(f"_execute_trading_cycle: Invalid signal object: {signal}")
                        raise CriticalBotError("Signal object missing required attributes")
                    
                    logger.info(f"  #{i+1}: {signal.ticker} - {signal.chemistry.value} "
                               f"(strength: {signal.signal_strength:.2f}, "
                               f"confidence: {signal.confidence:.2%})")
                except Exception as e:
                    logger.error(f"_execute_trading_cycle: Error logging signal {i}: {e}")
                    raise CriticalBotError(f"Signal validation error: {e}")
            
            # Step 6: Synthesize optimal portfolio
            logger.info("⚗️ Synthesizing portfolio...")
            try:
                target_positions = self._synthesize_portfolio(signals, prediction)
                if target_positions is None:
                    logger.error("_execute_trading_cycle: _synthesize_portfolio returned None")
                    raise CriticalBotError("Portfolio synthesis returned None")
                
                if not isinstance(target_positions, list):
                    logger.error(f"_execute_trading_cycle: Invalid target_positions type: {type(target_positions)}")
                    raise CriticalBotError("Portfolio positions format invalid")
                
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"_execute_trading_cycle: Portfolio synthesis failed: {e}")
                raise CriticalBotError(f"Portfolio synthesis error: {e}")
            
            if not target_positions:
                logger.info("No viable positions identified")
                return
            
            # Step 7: Execute trades (placeholder for IBKR integration)
            logger.info("📈 Would execute trades:")
            total_allocation = 0
            try:
                for pos in target_positions:
                    if not hasattr(pos, 'position_size') or not hasattr(pos, 'entry_price'):
                        logger.error(f"_execute_trading_cycle: Invalid position object: {pos}")
                        raise CriticalBotError("Position object missing required attributes")
                    
                    allocation = pos.position_size * pos.entry_price
                    total_allocation += allocation
                    logger.info(f"  {pos.ticker}: {pos.position_size} shares @ ${pos.entry_price:.2f} "
                               f"= ${allocation:.0f} ({pos.confidence:.2%} confidence)")
                
                logger.info(f"Total allocation: ${total_allocation:.0f} ({total_allocation/self.capital:.1%} of capital)")
                
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"_execute_trading_cycle: Trade execution validation failed: {e}")
                raise CriticalBotError(f"Trade execution error: {e}")
            
            # Update system state
            self.system_state.active_positions = target_positions
            self.system_state.last_update = datetime.now()
            
            # Log cycle summary
            logger.info(f"Cycle complete - {len(target_positions)} positions, "
                       f"${total_allocation:.0f} allocated")
                       
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"_execute_trading_cycle: Unexpected cycle error: {e}")
            raise CriticalBotError(f"Trading cycle failed: {e}")
            
    def _update_chemistry(self):
        """Update chemistry classifications for trading universe."""
        try:
            if not self.universe_tickers:
                logger.error("_update_chemistry: No universe tickers available")
                raise CriticalBotError("Universe tickers required for chemistry update")
            
            if not isinstance(self.universe_tickers, list):
                logger.error(f"_update_chemistry: Invalid universe_tickers type: {type(self.universe_tickers)}")
                raise CriticalBotError("Universe tickers format invalid")
                
            # Process in batches to respect API limits
            batch_size = 100
            updated_count = 0
            
            for i in range(0, len(self.universe_tickers), batch_size):
                try:
                    batch = self.universe_tickers[i:i+batch_size]
                    if not batch:
                        logger.error(f"_update_chemistry: Empty batch at index {i}")
                        raise CriticalBotError("Empty ticker batch encountered")
                    
                    chemistry = classify_asset_chemistry(
                        batch, 
                        self.fmp_key,
                        self.system_state.market_state
                    )
                    
                    if chemistry is None:
                        logger.error(f"_update_chemistry: classify_asset_chemistry returned None for batch {i}")
                        raise CriticalBotError("Chemistry classification returned None")
                    
                    if not isinstance(chemistry, dict):
                        logger.error(f"_update_chemistry: Invalid chemistry type: {type(chemistry)}")
                        raise CriticalBotError("Chemistry classification format invalid")
                    
                    if chemistry:
                        self.chemistry_cache.update(chemistry)
                        updated_count += len(chemistry)
                        
                    # Small delay between batches to be API-friendly
                    if i + batch_size < len(self.universe_tickers):
                        time.sleep(0.5)
                        
                except CriticalBotError:
                    raise
                except Exception as e:
                    logger.error(f"_update_chemistry: Batch {i} processing failed: {e}")
                    raise CriticalBotError(f"Chemistry batch processing error: {e}")
            
            if updated_count == 0:
                logger.error("_update_chemistry: No chemistry data updated")
                raise CriticalBotError("Chemistry update produced no results")
            
            self.last_chemistry_update = datetime.now()
            logger.info(f"Updated chemistry for {updated_count} assets")
            
            # Log chemistry distribution
            try:
                chemistry_counts = {}
                for chem in self.chemistry_cache.values():
                    if hasattr(chem, 'value'):
                        chemistry_counts[chem.value] = chemistry_counts.get(chem.value, 0) + 1
                    else:
                        logger.error(f"_update_chemistry: Invalid chemistry object: {chem}")
                        raise CriticalBotError("Chemistry object missing value attribute")
                
                logger.debug(f"Chemistry distribution: {chemistry_counts}")
                
            except CriticalBotError:
                raise
            except Exception as e:
                logger.error(f"_update_chemistry: Chemistry distribution logging failed: {e}")
                raise CriticalBotError(f"Chemistry validation error: {e}")
                
        except Exception as e:
            logger.error(f"_update_chemistry: Unexpected chemistry update error: {e}")
            raise CriticalBotError(f"Chemistry update failed: {e}")
            
    def _detect_trailheads(self) -> List[TrailheadSignal]:
        """Detect trailhead signals using pressure point analysis."""
        try:
            if not self.chemistry_cache:
                logger.error("_detect_trailheads: No chemistry data available")
                raise CriticalBotError("Chemistry data required for trailhead detection")
            
            if not isinstance(self.chemistry_cache, dict):
                logger.error(f"_detect_trailheads: Invalid chemistry_cache type: {type(self.chemistry_cache)}")
                raise CriticalBotError("Chemistry cache format invalid")
                
            signals = detect_trailheads(
                self.chemistry_cache,
                self.system_state.market_state,
                self.fmp_key,
                self.finnhub_key
            )
            
            if signals is None:
                logger.error("_detect_trailheads: detect_trailheads returned None")
                raise CriticalBotError("Trailhead detection returned None")
            
            if not isinstance(signals, list):
                logger.error(f"_detect_trailheads: Invalid signals type: {type(signals)}")
                raise CriticalBotError("Trailhead signals format invalid")
            
            # Filter and sort signals
            valid_signals = []
            for signal in signals:
                try:
                    if not hasattr(signal, 'confidence'):
                        logger.error(f"_detect_trailheads: Signal missing confidence attribute: {signal}")
                        raise CriticalBotError("Signal object missing confidence")
                    
                    if signal.confidence > 0.1:
                        valid_signals.append(signal)
                except Exception as e:
                    logger.error(f"_detect_trailheads: Signal validation failed: {e}")
                    raise CriticalBotError(f"Signal validation error: {e}")
            
            try:
                valid_signals.sort(key=lambda x: x.signal_strength * x.confidence, reverse=True)
            except Exception as e:
                logger.error(f"_detect_trailheads: Signal sorting failed: {e}")
                raise CriticalBotError(f"Signal sorting error: {e}")
            
            return valid_signals[:20]  # Return top 20 signals
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"_detect_trailheads: Unexpected trailhead detection error: {e}")
            raise CriticalBotError(f"Trailhead detection failed: {e}")
            
    def _synthesize_portfolio(
        self,
        signals: List[TrailheadSignal],
        prediction: StatePrediction
    ) -> List[PortfolioPosition]:
        """Synthesize optimal portfolio using geological principles."""
        try:
            if signals is None:
                logger.error("_synthesize_portfolio: signals parameter is None")
                raise CriticalBotError("Signals required for portfolio synthesis")
            
            if not isinstance(signals, list):
                logger.error(f"_synthesize_portfolio: Invalid signals type: {type(signals)}")
                raise CriticalBotError("Signals format invalid")
            
            if not prediction:
                logger.error("_synthesize_portfolio: No state prediction available")
                raise CriticalBotError("State prediction required for portfolio synthesis")
            
            if not hasattr(prediction, 'target_risk') or not hasattr(prediction, 'target_momentum'):
                logger.error(f"_synthesize_portfolio: Invalid prediction object: {prediction}")
                raise CriticalBotError("Prediction object missing required attributes")
                
            target_state = (prediction.target_risk, prediction.target_momentum)
            
            try:
                positions = self.portfolio_synthesizer.synthesize_portfolio(
                    signals,
                    self.chemistry_cache,
                    self.system_state.market_state,
                    target_state
                )
            except Exception as e:
                logger.error(f"_synthesize_portfolio: Portfolio synthesizer failed: {e}")
                raise CriticalBotError(f"Portfolio synthesizer error: {e}")
            
            if positions is None:
                logger.error("_synthesize_portfolio: Portfolio synthesizer returned None")
                raise CriticalBotError("Portfolio synthesis returned None")
            
            if not isinstance(positions, list):
                logger.error(f"_synthesize_portfolio: Invalid positions type: {type(positions)}")
                raise CriticalBotError("Portfolio positions format invalid")
            
            # Validate positions
            validated_positions = []
            for pos in positions:
                try:
                    if self._validate_position(pos):
                        validated_positions.append(pos)
                    else:
                        logger.warning(f"Position {pos.ticker} failed validation")
                except Exception as e:
                    logger.error(f"_synthesize_portfolio: Position validation failed for {pos}: {e}")
                    raise CriticalBotError(f"Position validation error: {e}")
            
            return validated_positions
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"_synthesize_portfolio: Unexpected portfolio synthesis error: {e}")
            raise CriticalBotError(f"Portfolio synthesis failed: {e}")
    
    def _validate_position(self, position: PortfolioPosition) -> bool:
        """Validate position meets risk constraints."""
        try:
            if position is None:
                logger.error("_validate_position: Position is None")
                raise CriticalBotError("Position cannot be None")
            
            if not hasattr(position, 'position_size') or not hasattr(position, 'entry_price'):
                logger.error(f"_validate_position: Position missing required attributes: {position}")
                raise CriticalBotError("Position object missing required attributes")
            
            if not hasattr(position, 'ticker') or not hasattr(position, 'confidence'):
                logger.error(f"_validate_position: Position missing ticker or confidence: {position}")
                raise CriticalBotError("Position object missing ticker or confidence")
            
            # Check position size limits
            try:
                position_value = position.position_size * position.entry_price
                position_weight = position_value / self.capital
            except (TypeError, ZeroDivisionError) as e:
                logger.error(f"_validate_position: Position calculation failed for {position.ticker}: {e}")
                raise CriticalBotError(f"Position calculation error: {e}")
            
            if position_weight > self.max_position_size:
                logger.warning(f"Position {position.ticker} exceeds max size: {position_weight:.2%}")
                return False
                
            # Check minimum confidence
            if position.confidence < 0.2:
                logger.warning(f"Position {position.ticker} below confidence threshold: {position.confidence:.2%}")
                return False
                
            return True
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"_validate_position: Unexpected validation error: {e}")
            raise CriticalBotError(f"Position validation failed: {e}")
            
    def _update_universe(self):
        """Load/refresh trading universe from FMP screener."""
        try:
            # Use FMP's stock screener for liquid, large-cap stocks
            url = (f"https://financialmodelingprep.com/api/v3/stock-screener?"
                   f"marketCapMoreThan=1000000000&"  # $1B+ market cap
                   f"volumeMoreThan=1000000&"         # 1M+ daily volume
                   f"limit={self.universe_size}&"
                   f"apikey={self.fmp_key}")
            
            try:
                response = requests.get(url, timeout=30)
                if response is None:
                    logger.error("_update_universe: API request returned None")
                    raise CriticalBotError("API request failed - no response")
                
                response.raise_for_status()
                
            except requests.exceptions.Timeout as e:
                logger.error(f"_update_universe: API timeout: {e}")
                raise CriticalBotError(f"API timeout error: {e}")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"_update_universe: API connection error: {e}")
                raise CriticalBotError(f"API connection error: {e}")
            except requests.exceptions.HTTPError as e:
                logger.error(f"_update_universe: API HTTP error: {e}")
                raise CriticalBotError(f"API HTTP error: {e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"_update_universe: API request error: {e}")
                raise CriticalBotError(f"API request error: {e}")
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"_update_universe: JSON decode error: {e}")
                raise CriticalBotError(f"API response JSON decode error: {e}")
            
            if data is None:
                logger.error("_update_universe: API returned None data")
                raise CriticalBotError("API returned no data")
            
            if not isinstance(data, list):
                logger.error(f"_update_universe: Invalid data type from API: {type(data)}")
                raise CriticalBotError("Invalid screener data format from FMP")
            
            if len(data) == 0:
                logger.error("_update_universe: API returned empty data list")
                raise CriticalBotError("API returned empty screener results")
            
            new_tickers = []
            for item in data:
                try:
                    if not isinstance(item, dict):
                        logger.error(f"_update_universe: Invalid item type: {type(item)}")
                        continue
                    
                    if 'symbol' not in item:
                        logger.error(f"_update_universe: Item missing symbol: {item}")
                        continue
                    
                    ticker = item['symbol']
                    if ticker is None:
                        logger.error(f"_update_universe: Ticker is None in item: {item}")
                        continue
                    
                    # Filter out problematic tickers
                    if (len(ticker) <= 5 and 
                        '.' not in ticker and 
                        ticker.isalpha()):
                        new_tickers.append(ticker)
                        
                except Exception as e:
                    logger.error(f"_update_universe: Error processing ticker item: {e}")
                    continue
            
            if len(new_tickers) < 50:
                logger.error(f"_update_universe: Insufficient valid tickers: {len(new_tickers)}")
                raise CriticalBotError(f"Universe too small: {len(new_tickers)} tickers")
            
            self.universe_tickers = new_tickers[:self.universe_size]
            self.last_universe_update = datetime.now()
            
            logger.info(f"Universe updated: {len(self.universe_tickers)} tickers")
            logger.debug(f"Sample tickers: {self.universe_tickers[:10]}")
            
        except CriticalBotError:
            raise
        except Exception as e:
            logger.error(f"_update_universe: Unexpected universe update error: {e}")
            raise CriticalBotError(f"Universe update failed: {e}")

def main():
    """Entry point for the Catalyst Trading System."""
    try:
        # Load configuration from environment
        fmp_key = os.getenv('FMP_API_KEY')
        finnhub_key = os.getenv('FINNHUB_API_KEY')
        
        if not fmp_key:
            logger.error("main: FMP_API_KEY environment variable not set")
            raise CriticalBotError("FMP_API_KEY environment variable required")
        
        if fmp_key is None or fmp_key.strip() == "":
            logger.error("main: FMP_API_KEY is None or empty")
            raise CriticalBotError("Valid FMP_API_KEY required")
        
        # Configuration with validation
        try:
            config = {
                'capital': float(os.getenv('CAPITAL', '5000')),
                'universe_size': int(os.getenv('UNIVERSE_SIZE', '500')),
                'cycle_interval': int(os.getenv('CYCLE_INTERVAL', '30')),
                'max_positions': int(os.getenv('MAX_POSITIONS', '10')),
                'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.15'))
            }
            
            # Validate configuration values
            if config['capital'] <= 0:
                logger.error(f"main: Invalid capital value: {config['capital']}")
                raise CriticalBotError("Capital must be positive")
            
            if config['universe_size'] <= 0:
                logger.error(f"main: Invalid universe_size: {config['universe_size']}")
                raise CriticalBotError("Universe size must be positive")
            
            if config['cycle_interval'] <= 0:
                logger.error(f"main: Invalid cycle_interval: {config['cycle_interval']}")
                raise CriticalBotError("Cycle interval must be positive")
            
        except ValueError as e:
            logger.error(f"main: Configuration value error: {e}")
            raise CriticalBotError(f"Invalid configuration values: {e}")
        except Exception as e:
            logger.error(f"main: Configuration error: {e}")
            raise CriticalBotError(f"Configuration failed: {e}")
        
        logger.info("🚀 Initializing Catalyst Trading System")
        logger.info(f"Configuration: {config}")
        
        # Create and run the trading system
        try:
            catalyst = CatalystTradingSystem(
                fmp_key=fmp_key,
                finnhub_key=finnhub_key,
                **config
            )
            
            if catalyst is None:
                logger.error("main: CatalystTradingSystem initialization returned None")
                raise CriticalBotError("System initialization failed")
            
        except Exception as e:
            logger.error(f"main: System initialization failed: {e}")
            raise CriticalBotError(f"System initialization error: {e}")
        
        try:
            catalyst.run_loop()
        except KeyboardInterrupt:
            logger.info("System interrupted by user")
        except CriticalBotError as e:
            logger.critical(f"System failed with critical error: {e}")
            raise
        except Exception as e:
            logger.critical(f"main: System failed with unexpected error: {e}")
            raise CriticalBotError(f"System runtime failure: {e}")
            
    except CriticalBotError:
        raise
    except Exception as e:
        logger.critical(f"main: Fatal error in main function: {e}")
        raise CriticalBotError(f"Main function failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except CriticalBotError as e:
        logger.critical(f"SYSTEM HALT: {e}")
        exit(1)
    except Exception as e:
        logger.critical(f"UNEXPECTED SYSTEM FAILURE: {e}")
        exit(1)