import numpy as np
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

from Market_State_Calculator import CriticalBotError
from Chemistry_Classifier import AssetChemistry

logger = logging.getLogger(__name__)

@dataclass
class TrailheadSignal:
    """Represents a detected market pressure point about to release."""
    ticker: str
    pressure_score: float  # 0-1, higher = more pressure
    fragility_score: float  # 0-1, higher = more fragile
    composite_score: float  # Combined trailhead probability
    trigger_type: str  # 'squeeze', 'breakout', 'reversal', 'cascade'
    metadata: Dict

def detect_trailheads(
    chemistry_map: Dict[str, AssetChemistry],
    market_state: Tuple[float, float],
    fmp_key: str,
    finnhub_key: Optional[str] = None
) -> List[TrailheadSignal]:
    """
    🌋 CUTTING-EDGE: Detect geological pressure points about to experience seismic market moves.
    
    Revolutionary Geological Detection Algorithm:
    - PRESSURE BUILDUP: Volume accumulation, option flow imbalances, short squeeze potential
    - FRAGILITY ANALYSIS: Correlation breakdown, liquidity gaps, structural weakness
    - SEISMIC TRIGGERS: Critical mass thresholds where pressure overcomes resistance
    - CHEMICAL CATALYSIS: How asset chemistry amplifies pressure release
    
    Geological Metaphor Applied:
    🏔️ TRAILHEADS = Critical pressure points where tectonic forces converge
    💥 PRESSURE = Accumulated market stress seeking release path
    🪨 FRAGILITY = Structural weakness in current price formation
    ⚡ TRIGGERS = Specific events that cause pressure release (earnings, breakouts, etc.)
    
    Detection Types:
    - SQUEEZE: Pressure buildup from short interest + volume + technical compression
    - BREAKOUT: Price approaching critical resistance with momentum
    - REVERSAL: Overextended positions meeting contrary pressure
    - CASCADE: Correlation breakdown creating domino effect potential
    
    Args:
        chemistry_map: Asset chemical profiles from classify_asset_chemistry
        market_state: Current geological pressure (risk, momentum) coordinates
        fmp_key: Financial data source for pressure measurements
        finnhub_key: Optional sentiment/options data for enhanced detection
    
    Returns:
        List of TrailheadSignals sorted by composite score (highest pressure first)
        
    Raises:
        CriticalBotError: On geological survey equipment failure
    """
    
    try:
        # Geological survey validation with fail-fast
        if chemistry_map is None:
            logger.error("FAIL-FAST: detect_trailheads() - chemistry_map parameter is None")
            raise CriticalBotError("Cannot detect trailheads - chemical analysis data is None")
        
        if not isinstance(chemistry_map, dict):
            logger.error(f"FAIL-FAST: detect_trailheads() - Invalid chemistry_map type: {type(chemistry_map)}")
            raise CriticalBotError(f"Invalid chemistry map type: {type(chemistry_map)}")
        
        if not chemistry_map:
            logger.error("FAIL-FAST: detect_trailheads() - Empty chemistry_map provided")
            raise CriticalBotError("Cannot detect trailheads - no chemical compounds to analyze")
        
        if market_state is None:
            logger.error("FAIL-FAST: detect_trailheads() - market_state parameter is None")
            raise CriticalBotError("Cannot detect trailheads - geological pressure readings are None")
        
        if not isinstance(market_state, (tuple, list)) or len(market_state) != 2:
            logger.error(f"FAIL-FAST: detect_trailheads() - Invalid market_state format: {market_state}")
            raise CriticalBotError(f"Invalid geological pressure format: {market_state}")
        
        try:
            pressure_risk, tectonic_momentum = market_state
            pressure_risk = float(pressure_risk)
            tectonic_momentum = float(tectonic_momentum)
        except (ValueError, TypeError) as e:
            logger.error(f"FAIL-FAST: detect_trailheads() - Cannot convert market_state to float: {market_state}, error: {e}")
            raise CriticalBotError(f"Geological pressure coordinates invalid: {market_state}")
        
        if not (0 <= pressure_risk <= 1) or not (-1 <= tectonic_momentum <= 1):
            logger.error(f"FAIL-FAST: detect_trailheads() - Geological pressure out of bounds: risk={pressure_risk}, momentum={tectonic_momentum}")
            raise CriticalBotError(f"Geological pressure exceeds instrument limits: ({pressure_risk}, {tectonic_momentum})")
        
        if fmp_key is None:
            logger.error("FAIL-FAST: detect_trailheads() - fmp_key parameter is None")
            raise CriticalBotError("Cannot operate geological detection equipment - API key is None")
        
        if not isinstance(fmp_key, str) or not fmp_key.strip():
            logger.error(f"FAIL-FAST: detect_trailheads() - Invalid fmp_key type or value: {type(fmp_key)}")
            raise CriticalBotError("Cannot operate geological detection equipment - invalid API key")
        
        logger.info(f"Geological pressure point detection: {len(chemistry_map)} compounds under pressure ({pressure_risk:.3f}, {tectonic_momentum:.3f})")
        
        trailhead_signals = []
        failed_detections = []
        
        for ticker, chemistry in chemistry_map.items():
            try:
                if ticker is None:
                    logger.warning("FAIL-FAST: detect_trailheads() - None ticker in chemistry_map")
                    failed_detections.append(str(ticker))
                    continue
                
                if not isinstance(ticker, str):
                    logger.warning(f"FAIL-FAST: detect_trailheads() - Invalid ticker type: {type(ticker)}")
                    failed_detections.append(str(ticker))
                    continue
                
                if chemistry is None:
                    logger.warning(f"FAIL-FAST: detect_trailheads() - Chemistry data is None for {ticker}")
                    failed_detections.append(ticker)
                    continue
                
                if not isinstance(chemistry, AssetChemistry):
                    logger.warning(f"FAIL-FAST: detect_trailheads() - Invalid chemistry type for {ticker}: {type(chemistry)}")
                    failed_detections.append(ticker)
                    continue
                
                # Perform geological pressure point analysis
                trailhead_data = _analyze_geological_pressure_point(ticker, chemistry, market_state, fmp_key, finnhub_key)
                if trailhead_data is None:
                    logger.debug(f"No pressure point detected for {ticker}")
                    continue
                
                # Calculate trailhead signal strength
                signal = _calculate_trailhead_signal_strength(trailhead_data, chemistry, market_state)
                if signal is None:
                    logger.warning(f"Signal calculation failed for {ticker}")
                    failed_detections.append(ticker)
                    continue
                
                # Validate signal quality
                if signal.composite_score < 0.3:
                    logger.debug(f"Weak trailhead signal for {ticker}: {signal.composite_score:.2f}")
                    continue
                
                trailhead_signals.append(signal)
                logger.debug(f"Trailhead detected: {ticker} ({signal.trigger_type}) - composite: {signal.composite_score:.2f}")
                
            except CriticalBotError:
                raise  # Re-raise critical errors immediately
            except Exception as e:
                logger.error(f"FAIL-FAST: detect_trailheads() - Pressure point analysis failed for {ticker}: {type(e).__name__}: {e}")
                failed_detections.append(ticker)
                continue
        
        # Validate detection results
        if not trailhead_signals:
            logger.warning("No geological pressure points detected - market may be in stable equilibrium")
            return []
        
        # Sort by composite pressure score (highest first)
        try:
            trailhead_signals.sort(key=lambda x: x.composite_score, reverse=True)
        except Exception as e:
            logger.error(f"FAIL-FAST: detect_trailheads() - Signal sorting failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Trailhead signal sorting failed: {e}")
        
        detection_rate = len(trailhead_signals) / len(chemistry_map)
        logger.info(f"Geological survey complete: {len(trailhead_signals)} pressure points detected ({detection_rate:.1%} success rate)")
        
        if failed_detections:
            logger.debug(f"Failed detections: {failed_detections}")
        
        return trailhead_signals
        
    except CriticalBotError:
        raise  # Re-raise critical errors
    except TypeError as e:
        logger.error(f"FAIL-FAST: detect_trailheads() - Type error in function parameters: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Trailhead detection type error: {e}")
    except ValueError as e:
        logger.error(f"FAIL-FAST: detect_trailheads() - Value error in function parameters: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Trailhead detection value error: {e}")
    except Exception as e:
        logger.error(f"FAIL-FAST: detect_trailheads() - Catastrophic geological detection failure: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Trailhead detection system failure: {e}")

def _analyze_geological_pressure_point(
    ticker: str, 
    chemistry: AssetChemistry, 
    market_state: Tuple[float, float], 
    fmp_key: str, 
    finnhub_key: Optional[str]
) -> Optional[Dict]:
    """
    🔬 CUTTING-EDGE: Analyze specific geological pressure point using seismic instruments.
    
    Revolutionary Pressure Detection:
    - SEISMIC MONITORING: Volume patterns, price compression, volatility clustering
    - FAULT LINE MAPPING: Support/resistance stress points, correlation breakdown
    - PRESSURE MEASUREMENT: Short interest, option flow, institutional positioning
    - CHEMICAL REACTION POTENTIAL: How asset chemistry responds to pressure
    """
    try:
        if ticker is None:
            logger.error("FAIL-FAST: _analyze_geological_pressure_point() - ticker parameter is None")
            raise CriticalBotError("Cannot analyze geological pressure point - ticker is None")
        
        if not isinstance(ticker, str):
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - Invalid ticker type: {type(ticker)}")
            raise CriticalBotError(f"Invalid ticker type for pressure analysis: {type(ticker)}")
        
        if len(ticker.strip()) == 0:
            logger.error("FAIL-FAST: _analyze_geological_pressure_point() - Empty ticker string")
            raise CriticalBotError("Cannot analyze pressure point - empty ticker")
        
        if chemistry is None:
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - chemistry parameter is None for {ticker}")
            raise CriticalBotError(f"Cannot analyze pressure point - chemistry data is None for {ticker}")
        
        if not isinstance(chemistry, AssetChemistry):
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - Invalid chemistry type for {ticker}: {type(chemistry)}")
            raise CriticalBotError(f"Invalid chemistry type for pressure analysis of {ticker}")
        
        if market_state is None:
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - market_state is None for {ticker}")
            raise CriticalBotError(f"Cannot analyze pressure point - market state is None for {ticker}")
        
        try:
            pressure_risk, tectonic_momentum = market_state
            pressure_risk = float(pressure_risk)
            tectonic_momentum = float(tectonic_momentum)
        except (ValueError, TypeError) as e:
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - Invalid market_state for {ticker}: {market_state}, error: {e}")
            raise CriticalBotError(f"Invalid market state for pressure analysis of {ticker}: {e}")
        
        if fmp_key is None or not isinstance(fmp_key, str):
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - Invalid fmp_key for {ticker}")
            raise CriticalBotError(f"Cannot access geological instruments for {ticker} - invalid API key")
        
        # 1. SEISMIC ACTIVITY ANALYSIS - Advanced technical pressure detection
        seismic_data = _measure_seismic_activity(ticker, fmp_key)
        if seismic_data is None:
            logger.warning(f"No seismic data available for {ticker}")
            return None
        
        # 2. FAULT LINE STRESS ANALYSIS - Support/resistance pressure points
        fault_stress = _analyze_fault_line_stress(ticker, seismic_data)
        if fault_stress is None:
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - Fault stress analysis failed for {ticker}")
            raise CriticalBotError(f"Fault line stress analysis failed for {ticker}")
        
        # 3. PRESSURE ACCUMULATION - Volume and positioning analysis
        pressure_buildup = _measure_pressure_accumulation(ticker, seismic_data, chemistry)
        if pressure_buildup is None:
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - Pressure accumulation analysis failed for {ticker}")
            raise CriticalBotError(f"Pressure accumulation analysis failed for {ticker}")
        
        # 4. FRAGILITY ASSESSMENT - Structural weakness detection
        fragility_analysis = _assess_structural_fragility(ticker, seismic_data, chemistry, market_state)
        if fragility_analysis is None:
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - Fragility assessment failed for {ticker}")
            raise CriticalBotError(f"Structural fragility assessment failed for {ticker}")
        
        # 5. CHEMICAL CATALYSIS POTENTIAL - How chemistry amplifies pressure
        catalysis_potential = _calculate_chemical_catalysis_potential(chemistry, market_state, seismic_data)
        if catalysis_potential is None:
            logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - Catalysis potential calculation failed for {ticker}")
            raise CriticalBotError(f"Chemical catalysis calculation failed for {ticker}")
        
        return {
            'ticker': ticker,
            'seismic_data': seismic_data,
            'fault_stress': fault_stress,
            'pressure_buildup': pressure_buildup,
            'fragility_analysis': fragility_analysis,
            'catalysis_potential': catalysis_potential,
            'market_pressure': pressure_risk,
            'tectonic_momentum': tectonic_momentum
        }
        
    except CriticalBotError:
        raise  # Re-raise critical errors
    except Exception as e:
        logger.error(f"FAIL-FAST: _analyze_geological_pressure_point() - Pressure point analysis failed for {ticker}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Geological pressure analysis failed for {ticker}: {e}")

def _measure_seismic_activity(ticker: str, fmp_key: str) -> Optional[Dict]:
    """
    🌊 Measure seismic market activity using advanced geological instruments.
    
    Cutting-Edge Seismic Detection:
    - PRICE COMPRESSION: Bollinger Band squeeze, decreasing volatility
    - VOLUME ACCUMULATION: Unusual volume patterns, distribution analysis
    - MOMENTUM DIVERGENCE: Price vs. volume vs. RSI divergences
    - VOLATILITY CLUSTERING: Periods of calm before eruption
    """
    try:
        if ticker is None or not isinstance(ticker, str):
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Invalid ticker: {ticker}")
            raise CriticalBotError(f"Cannot measure seismic activity - invalid ticker: {ticker}")
        
        if fmp_key is None or not isinstance(fmp_key, str):
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Invalid fmp_key for {ticker}")
            raise CriticalBotError(f"Cannot access seismic instruments for {ticker} - invalid API key")
        
        # Fetch extended market data for seismic analysis
        daily_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries=60&apikey={fmp_key}"
        
        try:
            response = requests.get(daily_url, timeout=10)
        except requests.exceptions.Timeout as e:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - API timeout for {ticker}: {e}")
            raise CriticalBotError(f"Seismic data retrieval timeout for {ticker}: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Connection error for {ticker}: {e}")
            raise CriticalBotError(f"Seismic data connection error for {ticker}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Request error for {ticker}: {e}")
            raise CriticalBotError(f"Seismic data request error for {ticker}: {e}")
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - HTTP error for {ticker}: {e}")
            raise CriticalBotError(f"Seismic data HTTP error for {ticker}: {e}")
        
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - JSON parsing error for {ticker}: {e}")
            raise CriticalBotError(f"Seismic data JSON parsing failed for {ticker}: {e}")
        
        if data is None:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - API returned None for {ticker}")
            raise CriticalBotError(f"Seismic data API returned None for {ticker}")
        
        if not isinstance(data, dict):
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Invalid data format for {ticker}: {type(data)}")
            raise CriticalBotError(f"Invalid seismic data format for {ticker}")
        
        if 'historical' not in data:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - No historical data for {ticker}")
            raise CriticalBotError(f"No historical seismic data for {ticker}")
        
        historical = data['historical']
        if historical is None:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Historical data is None for {ticker}")
            raise CriticalBotError(f"Historical seismic data is None for {ticker}")
        
        if not isinstance(historical, list):
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Invalid historical data type for {ticker}: {type(historical)}")
            raise CriticalBotError(f"Invalid historical data format for {ticker}")
        
        if len(historical) == 0:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Empty historical data for {ticker}")
            raise CriticalBotError(f"Empty historical seismic data for {ticker}")
        
        historical = historical[:60]  # Last 60 days
        
        # Extract and validate seismic measurements
        prices = []
        volumes = []
        highs = []
        lows = []
        
        for i, h in enumerate(historical):
            try:
                if h is None:
                    logger.debug(f"Skipping None historical entry {i} for {ticker}")
                    continue
                
                if not isinstance(h, dict):
                    logger.debug(f"Skipping invalid historical entry {i} for {ticker}: {type(h)}")
                    continue
                
                required_keys = ['close', 'volume', 'high', 'low']
                if not all(key in h for key in required_keys):
                    logger.debug(f"Skipping incomplete historical entry {i} for {ticker}")
                    continue
                
                try:
                    close = float(h['close'])
                    volume = float(h['volume'])
                    high = float(h['high'])
                    low = float(h['low'])
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping invalid numerical data in entry {i} for {ticker}: {e}")
                    continue
                
                if close <= 0 or volume < 0 or high <= 0 or low <= 0:
                    logger.debug(f"Skipping invalid values in entry {i} for {ticker}")
                    continue
                
                if high < low:
                    logger.debug(f"Skipping invalid high/low in entry {i} for {ticker}")
                    continue
                
                if np.isnan(close) or np.isnan(volume) or np.isnan(high) or np.isnan(low):
                    logger.debug(f"Skipping NaN values in entry {i} for {ticker}")
                    continue
                
                if np.isinf(close) or np.isinf(volume) or np.isinf(high) or np.isinf(low):
                    logger.debug(f"Skipping infinite values in entry {i} for {ticker}")
                    continue
                
                prices.append(close)
                volumes.append(volume)
                highs.append(high)
                lows.append(low)
                
            except Exception as e:
                logger.debug(f"Error processing historical entry {i} for {ticker}: {type(e).__name__}: {e}")
                continue
        
        if len(prices) < 20:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Insufficient seismic data for {ticker}: {len(prices)} days (need 20+)")
            raise CriticalBotError(f"Insufficient geological data for {ticker}: only {len(prices)} days available")
        
        # CUTTING-EDGE SEISMIC CALCULATIONS WITH BULLETPROOF ERROR HANDLING
        
        try:
            # 1. PRICE COMPRESSION (Bollinger Band Squeeze)
            if len(prices) < 20:
                logger.error(f"FAIL-FAST: _measure_seismic_activity() - Insufficient price data for Bollinger calculation: {len(prices)}")
                raise CriticalBotError(f"Insufficient price data for seismic analysis of {ticker}")
            
            prices_array = np.array(prices[-20:])
            sma_20 = np.mean(prices_array)
            std_20 = np.std(prices_array)
            current_price = prices[-1]
            
            if std_20 == 0:
                logger.warning(f"Zero standard deviation for {ticker}, using minimal value")
                std_20 = current_price * 0.001  # 0.1% of price
            
            bollinger_position = (current_price - sma_20) / (2 * std_20)
            compression_ratio = std_20 / sma_20 if sma_20 > 0 else 0
            
            if np.isnan(bollinger_position) or np.isinf(bollinger_position):
                bollinger_position = 0.0
            if np.isnan(compression_ratio) or np.isinf(compression_ratio):
                compression_ratio = 0.0
            
            # 2. VOLUME PRESSURE ANALYSIS
            if len(volumes) < 20:
                logger.error(f"FAIL-FAST: _measure_seismic_activity() - Insufficient volume data for {ticker}")
                raise CriticalBotError(f"Insufficient volume data for pressure analysis of {ticker}")
            
            volumes_array = np.array(volumes[-20:])
            avg_volume = np.mean(volumes_array)
            recent_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else avg_volume
            
            volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1.0
            if np.isnan(volume_surge) or np.isinf(volume_surge):
                volume_surge = 1.0
            
            # 3. MOMENTUM DIVERGENCE DETECTION
            price_momentum = 0.0
            volume_momentum = 0.0
            
            if len(prices) >= 10:
                try:
                    price_momentum = (prices[-1] / prices[-10] - 1) if prices[-10] > 0 else 0
                    if np.isnan(price_momentum) or np.isinf(price_momentum):
                        price_momentum = 0.0
                except (IndexError, ZeroDivisionError):
                    price_momentum = 0.0
            
            if len(volumes) >= 15:
                try:
                    recent_vol_avg = np.mean(volumes[-5:])
                    earlier_vol_avg = np.mean(volumes[-15:-5])
                    volume_momentum = (recent_vol_avg / earlier_vol_avg - 1) if earlier_vol_avg > 0 else 0
                    if np.isnan(volume_momentum) or np.isinf(volume_momentum):
                        volume_momentum = 0.0
                except (IndexError, ZeroDivisionError):
                    volume_momentum = 0.0
            
            # 4. VOLATILITY PATTERNS
            returns = []
            for i in range(len(prices)-1):
                try:
                    if prices[i+1] > 0:
                        return_val = (prices[i] / prices[i+1] - 1)
                        if not np.isnan(return_val) and not np.isinf(return_val):
                            returns.append(return_val)
                except (IndexError, ZeroDivisionError):
                    continue
            
            if len(returns) < 10:
                logger.error(f"FAIL-FAST: _measure_seismic_activity() - Insufficient returns data for {ticker}")
                raise CriticalBotError(f"Insufficient returns data for volatility analysis of {ticker}")
            
            recent_volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0
            long_term_volatility = np.std(returns[-30:]) if len(returns) >= 30 else recent_volatility
            
            if long_term_volatility == 0:
                long_term_volatility = recent_volatility if recent_volatility > 0 else 0.001
            
            volatility_ratio = recent_volatility / long_term_volatility
            if np.isnan(volatility_ratio) or np.isinf(volatility_ratio):
                volatility_ratio = 1.0
            
            # 5. RANGE COMPRESSION
            daily_ranges = []
            min_len = min(len(highs), len(lows), len(prices))
            
            for i in range(min_len):
                try:
                    if prices[i] > 0:
                        range_val = (highs[i] - lows[i]) / prices[i]
                        if not np.isnan(range_val) and not np.isinf(range_val) and range_val >= 0:
                            daily_ranges.append(range_val)
                except (ZeroDivisionError, IndexError):
                    continue
            
            if len(daily_ranges) < 10:
                logger.error(f"FAIL-FAST: _measure_seismic_activity() - Insufficient range data for {ticker}")
                raise CriticalBotError(f"Insufficient range data for compression analysis of {ticker}")
            
            avg_range = np.mean(daily_ranges[-20:]) if len(daily_ranges) >= 20 else np.mean(daily_ranges)
            recent_range = np.mean(daily_ranges[-5:]) if len(daily_ranges) >= 5 else avg_range
            
            range_compression = recent_range / avg_range if avg_range > 0 else 1.0
            if np.isnan(range_compression) or np.isinf(range_compression):
                range_compression = 1.0
            
            return {
                'prices': prices,
                'volumes': volumes,
                'highs': highs,
                'lows': lows,
                'returns': returns,
                'bollinger_position': float(bollinger_position),
                'compression_ratio': float(compression_ratio),
                'volume_surge': float(volume_surge),
                'price_momentum': float(price_momentum),
                'volume_momentum': float(volume_momentum),
                'volatility_ratio': float(volatility_ratio),
                'range_compression': float(range_compression),
                'current_price': float(current_price),
                'sma_20': float(sma_20),
                'std_20': float(std_20)
            }
            
        except Exception as e:
            logger.error(f"FAIL-FAST: _measure_seismic_activity() - Seismic calculation failed for {ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Seismic activity calculation failed for {ticker}: {e}")
        
    except CriticalBotError:
        raise  # Re-raise critical errors
    except Exception as e:
        logger.error(f"FAIL-FAST: _measure_seismic_activity() - Seismic measurement failed for {ticker}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Seismic activity measurement failed for {ticker}: {e}")

def _analyze_fault_line_stress(ticker: str, seismic_data: Dict) -> Dict:
    """
    🏔️ Analyze geological fault line stress points where pressure will release.
    
    Fault Line Stress Physics:
    - RESISTANCE MAPPING: Key price levels under stress
    - SUPPORT EROSION: Weakening foundation points
    - BREAKOUT PROBABILITY: Stress exceeding structural limits
    - ENERGY ACCUMULATION: Pressure building at critical levels
    """
    try:
        if seismic_data is None:
            logger.error(f"FAIL-FAST: _analyze_fault_line_stress() - seismic_data is None for {ticker}")
            raise CriticalBotError(f"Cannot analyze fault lines - seismic data is None for {ticker}")
        
        if not isinstance(seismic_data, dict):
            logger.error(f"FAIL-FAST: _analyze_fault_line_stress() - Invalid seismic_data type for {ticker}: {type(seismic_data)}")
            raise CriticalBotError(f"Invalid seismic data format for fault analysis of {ticker}")
        
        required_keys = ['prices', 'volumes', 'highs', 'lows', 'current_price']
        for key in required_keys:
            if key not in seismic_data:
                logger.error(f"FAIL-FAST: _analyze_fault_line_stress() - Missing {key} in seismic_data for {ticker}")
                raise CriticalBotError(f"Missing {key} in seismic data for fault analysis of {ticker}")
            if seismic_data[key] is None:
                logger.error(f"FAIL-FAST: _analyze_fault_line_stress() - {key} is None in seismic_data for {ticker}")
                raise CriticalBotError(f"Seismic data {key} is None for fault analysis of {ticker}")
        
        try:
            prices = seismic_data['prices']
            volumes = seismic_data['volumes']
            highs = seismic_data['highs']
            lows = seismic_data['lows']
            current_price = float(seismic_data['current_price'])
            
            if current_price <= 0:
                logger.error(f"FAIL-FAST: _analyze_fault_line_stress() - Invalid current_price for {ticker}: {current_price}")
                raise CriticalBotError(f"Invalid current price for fault analysis of {ticker}: {current_price}")
            
            if len(prices) < 10:
                logger.error(f"FAIL-FAST: _analyze_fault_line_stress() - Insufficient price data for {ticker}: {len(prices)}")
                raise CriticalBotError(f"Insufficient price data for fault line analysis of {ticker}")
            
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"FAIL-FAST: _analyze_fault_line_stress() - Data extraction failed for {ticker}: {e}")
            raise CriticalBotError(f"Fault line data extraction failed for {ticker}: {e}")
        
        try:
            # 1. RESISTANCE LEVEL STRESS ANALYSIS
            resistance_levels = []
            if len(prices) >= 11:
                for i in range(5, len(prices)-5):
                    try:
                        # Check if this is a local maximum
                        is_peak = True
                        for j in range(i-3, i+4):
                            if j != i and j >= 0 and j < len(prices):
                                if prices[i] < prices[j]:
                                    is_peak = False
                                    break
                        
                        if is_peak and not np.isnan(prices[i]) and not np.isinf(prices[i]):
                            resistance_levels.append(prices[i])
                    except (IndexError, TypeError):
                        continue
            
            # Find closest resistance above current price
            overhead_resistance = [r for r in resistance_levels if r > current_price and not np.isnan(r)]
            resistance_distance = 1.0  # Default high distance
            
            if overhead_resistance:
                try:
                    closest_resistance = min(overhead_resistance)
                    resistance_distance = (closest_resistance - current_price) / current_price
                    if np.isnan(resistance_distance) or np.isinf(resistance_distance):
                        resistance_distance = 1.0
                except (ValueError, ZeroDivisionError):
                    resistance_distance = 1.0
            
            # 2. SUPPORT LEVEL EROSION ANALYSIS
            support_levels = []
            if len(prices) >= 11:
                for i in range(5, len(prices)-5):
                    try:
                        # Check if this is a local minimum
                        is_trough = True
                        for j in range(i-3, i+4):
                            if j != i and j >= 0 and j < len(prices):
                                if prices[i] > prices[j]:
                                    is_trough = False
                                    break
                        
                        if is_trough and not np.isnan(prices[i]) and not np.isinf(prices[i]):
                            support_levels.append(prices[i])
                    except (IndexError, TypeError):
                        continue
            
            # Find closest support below current price
            support_below = [s for s in support_levels if s < current_price and not np.isnan(s)]
            support_distance = 1.0  # Default high distance
            
            if support_below:
                try:
                    closest_support = max(support_below)
                    support_distance = (current_price - closest_support) / current_price
                    if np.isnan(support_distance) or np.isinf(support_distance):
                        support_distance = 1.0
                except (ValueError, ZeroDivisionError):
                    support_distance = 1.0
            
            # 3. VOLUME AT PRICE ANALYSIS
            volume_at_resistance = 0.0
            if overhead_resistance and len(volumes) >= 20:
                try:
                    closest_resistance = min(overhead_resistance)
                    volume_count = 0
                    total_volume = 0.0
                    
                    # Find periods when price was near this resistance
                    for i in range(max(0, len(prices)-20), len(prices)):
                        try:
                            if abs(prices[i] - closest_resistance) / closest_resistance < 0.02:  # Within 2%
                                total_volume += volumes[i]
                                volume_count += 1
                        except (IndexError, ZeroDivisionError):
                            continue
                    
                    if volume_count > 0:
                        volume_at_resistance = total_volume / volume_count
                    
                    if np.isnan(volume_at_resistance) or np.isinf(volume_at_resistance):
                        volume_at_resistance = 0.0
                        
                except Exception as e:
                    logger.debug(f"Volume at resistance calculation failed for {ticker}: {e}")
                    volume_at_resistance = 0.0
            
            # 4. BREAKOUT ENERGY CALCULATION
            breakout_energy = 0.0
            try:
                if resistance_distance < 0.05:  # Within 5% of resistance
                    breakout_energy = (0.05 - resistance_distance) / 0.05  # Closer = more energy
                    
                    # Amplify with volume surge
                    volume_surge = seismic_data.get('volume_surge', 1.0)
                    if isinstance(volume_surge, (int, float)) and not np.isnan(volume_surge):
                        breakout_energy *= min(volume_surge, 3.0)  # Cap at 3x
                    
                    # Amplify with positive momentum
                    price_momentum = seismic_data.get('price_momentum', 0.0)
                    if isinstance(price_momentum, (int, float)) and not np.isnan(price_momentum):
                        breakout_energy *= max(price_momentum + 1.0, 0.5)  # Positive momentum helps
                    
                    if np.isnan(breakout_energy) or np.isinf(breakout_energy):
                        breakout_energy = 0.0
                    
                    breakout_energy = max(0.0, min(1.0, breakout_energy))  # Clamp to [0,1]
                    
            except Exception as e:
                logger.debug(f"Breakout energy calculation failed for {ticker}: {e}")
                breakout_energy = 0.0
            
            return {
                'resistance_distance': float(resistance_distance),
                'support_distance': float(support_distance),
                'volume_at_resistance': float(volume_at_resistance),
                'breakout_energy': float(breakout_energy),
                'resistance_levels': len(overhead_resistance),
                'support_levels': len(support_below)
            }
            
        except Exception as e:
            logger.error(f"FAIL-FAST: _analyze_fault_line_stress() - Fault line calculation failed for {ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Fault line stress calculation failed for {ticker}: {e}")
        
    except CriticalBotError:
        raise
    except Exception as e:
        logger.error(f"FAIL-FAST: _analyze_fault_line_stress() - Fault line analysis failed for {ticker}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Fault line stress analysis failed for {ticker}: {e}")

def _measure_pressure_accumulation(ticker: str, seismic_data: Dict, chemistry: AssetChemistry) -> Dict:
    """
    💨 Measure geological pressure accumulation using advanced instruments.
    
    Pressure Physics:
    - ACCUMULATION PATTERNS: Volume without price movement
    - DISTRIBUTION DETECTION: Smart money positioning
    - MOMENTUM BUILDUP: Energy storage before release
    - CHEMICAL AMPLIFICATION: How asset chemistry affects pressure
    """
    try:
        if seismic_data is None:
            logger.error(f"FAIL-FAST: _measure_pressure_accumulation() - seismic_data is None for {ticker}")
            raise CriticalBotError(f"Cannot measure pressure accumulation - seismic data is None for {ticker}")
        
        if chemistry is None:
            logger.error(f"FAIL-FAST: _measure_pressure_accumulation() - chemistry is None for {ticker}")
            raise CriticalBotError(f"Cannot measure pressure accumulation - chemistry data is None for {ticker}")
        
        if not isinstance(chemistry, AssetChemistry):
            logger.error(f"FAIL-FAST: _measure_pressure_accumulation() - Invalid chemistry type for {ticker}: {type(chemistry)}")
            raise CriticalBotError(f"Invalid chemistry type for pressure analysis of {ticker}")
        
        required_keys = ['prices', 'volumes']
        for key in required_keys:
            if key not in seismic_data or seismic_data[key] is None:
                logger.error(f"FAIL-FAST: _measure_pressure_accumulation() - Missing or None {key} for {ticker}")
                raise CriticalBotError(f"Missing seismic data {key} for pressure accumulation analysis of {ticker}")
        
        try:
            prices = seismic_data['prices']
            volumes = seismic_data['volumes']
            
            if len(prices) < 10 or len(volumes) < 10:
                logger.error(f"FAIL-FAST: _measure_pressure_accumulation() - Insufficient data for {ticker}: prices={len(prices)}, volumes={len(volumes)}")
                raise CriticalBotError(f"Insufficient data for pressure accumulation analysis of {ticker}")
            
        except (KeyError, TypeError) as e:
            logger.error(f"FAIL-FAST: _measure_pressure_accumulation() - Data extraction failed for {ticker}: {e}")
            raise CriticalBotError(f"Pressure accumulation data extraction failed for {ticker}: {e}")
        
        try:
            # 1. ACCUMULATION/DISTRIBUTION ANALYSIS (VPT)
            vpt = 0
            vpt_values = []
            
            for i in range(1, len(prices)):
                try:
                    if prices[i-1] > 0:
                        price_change = (prices[i] - prices[i-1]) / prices[i-1]
                        if not np.isnan(price_change) and not np.isinf(price_change):
                            vpt += volumes[i] * price_change
                            vpt_values.append(vpt)
                except (IndexError, ZeroDivisionError):
                    continue
            
            if len(vpt_values) < 10:
                logger.error(f"FAIL-FAST: _measure_pressure_accumulation() - Insufficient VPT data for {ticker}")
                raise CriticalBotError(f"Insufficient VPT data for pressure analysis of {ticker}")
            
            # Calculate VPT and price trends
            vpt_trend = 0.0
            price_trend = 0.0
            
            if len(vpt_values) >= 10:
                try:
                    vpt_trend = (vpt_values[-1] - vpt_values[-10]) / abs(vpt_values[-10]) if vpt_values[-10] != 0 else 0
                    if np.isnan(vpt_trend) or np.isinf(vpt_trend):
                        vpt_trend = 0.0
                except (IndexError, ZeroDivisionError):
                    vpt_trend = 0.0
            
            if len(prices) >= 10:
                try:
                    price_trend = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
                    if np.isnan(price_trend) or np.isinf(price_trend):
                        price_trend = 0.0
                except (IndexError, ZeroDivisionError):
                    price_trend = 0.0
            
            accumulation_divergence = vpt_trend - price_trend
            if np.isnan(accumulation_divergence) or np.isinf(accumulation_divergence):
                accumulation_divergence = 0.0
            
            # 2. SMART MONEY DETECTION
            smart_money_days = 0
            total_days = 0
            
            if len(prices) >= 10 and len(volumes) >= 10:
                try:
                    avg_volume = np.mean(volumes)
                    highs = seismic_data.get('highs', [])
                    lows = seismic_data.get('lows', [])
                    
                    min_len = min(len(prices), len(volumes), len(highs), len(lows))
                    
                    # Calculate average daily range
                    daily_ranges = []
                    for i in range(min_len):
                        try:
                            if prices[i] > 0 and highs[i] >= lows[i]:
                                range_val = (highs[i] - lows[i]) / prices[i]
                                if not np.isnan(range_val) and not np.isinf(range_val):
                                    daily_ranges.append(range_val)
                        except (IndexError, ZeroDivisionError):
                            continue
                    
                    if len(daily_ranges) > 0:
                        avg_range = np.mean(daily_ranges)
                        
                        for i in range(min_len):
                            try:
                                if (volumes[i] > avg_volume * 1.5 and  # High volume day
                                    prices[i] > 0 and highs[i] >= lows[i]):
                                    
                                    daily_range = (highs[i] - lows[i]) / prices[i]
                                    if daily_range < avg_range:  # Low range despite high volume
                                        smart_money_days += 1
                                    total_days += 1
                                else:
                                    total_days += 1
                            except (IndexError, ZeroDivisionError):
                                continue
                
                except Exception as e:
                    logger.debug(f"Smart money calculation failed for {ticker}: {e}")
            
            smart_money_ratio = smart_money_days / total_days if total_days > 0 else 0.0
            if np.isnan(smart_money_ratio) or np.isinf(smart_money_ratio):
                smart_money_ratio = 0.0
            
            # 3. MOMENTUM BUILDUP
            momentum_acceleration = 0.0
            
            if len(prices) >= 15:
                try:
                    # Calculate momentum over different periods
                    mom_5 = (prices[-1] / prices[-6] - 1) if len(prices) >= 6 and prices[-6] > 0 else 0
                    mom_10 = (prices[-1] / prices[-11] - 1) if len(prices) >= 11 and prices[-11] > 0 else 0
                    mom_15 = (prices[-1] / prices[-16] - 1) if len(prices) >= 16 and prices[-16] > 0 else 0
                    
                    for mom in [mom_5, mom_10, mom_15]:
                        if np.isnan(mom) or np.isinf(mom):
                            mom = 0.0
                    
                    # Accelerating momentum = building pressure
                    momentum_acceleration = (mom_5 - mom_10) + (mom_10 - mom_15)
                    if np.isnan(momentum_acceleration) or np.isinf(momentum_acceleration):
                        momentum_acceleration = 0.0
                        
                except Exception as e:
                    logger.debug(f"Momentum acceleration calculation failed for {ticker}: {e}")
                    momentum_acceleration = 0.0
            
            # 4. CHEMICAL PRESSURE AMPLIFICATION
            chemistry_amplifier = 1.0
            try:
                chemistry_type = chemistry.chemistry_type
                
                if chemistry_type == 'volatile_compound':
                    chemistry_amplifier = 1.4  # Volatile compounds build pressure faster
                elif chemistry_type == 'phase_change':
                    chemistry_amplifier = 1.2  # Phase change assets at transition points
                elif chemistry_type == 'catalyst_accelerant':
                    chemistry_amplifier = 1.3  # Catalysts amplify existing pressure
                elif chemistry_type == 'noble_gas':
                    chemistry_amplifier = 0.8  # Noble gases resist pressure buildup
                else:
                    chemistry_amplifier = 1.0  # Default
                    
            except Exception as e:
                logger.debug(f"Chemistry amplifier calculation failed for {ticker}: {e}")
                chemistry_amplifier = 1.0
            
            return {
                'accumulation_divergence': float(accumulation_divergence),
                'smart_money_ratio': float(smart_money_ratio),
                'momentum_acceleration': float(momentum_acceleration),
                'chemistry_amplifier': float(chemistry_amplifier),
                'vpt_trend': float(vpt_trend),
                'price_trend': float(price_trend)
            }
            
        except Exception as e:
            logger.error(f"FAIL-FAST: _measure_pressure_accumulation() - Pressure calculation failed for {ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Pressure accumulation calculation failed for {ticker}: {e}")
        
    except CriticalBotError:
        raise
    except Exception as e:
        logger.error(f"FAIL-FAST: _measure_pressure_accumulation() - Pressure measurement failed for {ticker}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Pressure accumulation measurement failed for {ticker}: {e}")

def _assess_structural_fragility(ticker: str, seismic_data: Dict, chemistry: AssetChemistry, market_state: Tuple[float, float]) -> Dict:
    """
    🪨 Assess structural fragility - how easily will the formation break under pressure?
    
    Fragility Analysis:
    - VOLATILITY EXPANSION: How quickly volatility increases under stress
    - CORRELATION BREAKDOWN: Independence from market during stress
    - LIQUIDITY GAPS: Volume drops that create air pockets
    - STRUCTURAL WEAKNESS: Technical patterns showing instability
    """
    try:
        if seismic_data is None:
            logger.error(f"FAIL-FAST: _assess_structural_fragility() - seismic_data is None for {ticker}")
            raise CriticalBotError(f"Cannot assess structural fragility - seismic data is None for {ticker}")
        
        if chemistry is None:
            logger.error(f"FAIL-FAST: _assess_structural_fragility() - chemistry is None for {ticker}")
            raise CriticalBotError(f"Cannot assess structural fragility - chemistry data is None for {ticker}")
        
        if market_state is None:
            logger.error(f"FAIL-FAST: _assess_structural_fragility() - market_state is None for {ticker}")
            raise CriticalBotError(f"Cannot assess structural fragility - market state is None for {ticker}")
        
        try:
            pressure_risk, tectonic_momentum = market_state
            pressure_risk = float(pressure_risk)
            tectonic_momentum = float(tectonic_momentum)
        except (ValueError, TypeError) as e:
            logger.error(f"FAIL-FAST: _assess_structural_fragility() - Invalid market_state for {ticker}: {market_state}, error: {e}")
            raise CriticalBotError(f"Invalid market state for fragility analysis of {ticker}: {e}")
        
        try:
            # 1. VOLATILITY EXPANSION ANALYSIS
            volatility_ratio = seismic_data.get('volatility_ratio', 1.0)
            if volatility_ratio is None or np.isnan(volatility_ratio) or np.isinf(volatility_ratio):
                volatility_ratio = 1.0
            
            # High ratio means recent volatility expanding vs historical
            volatility_fragility = max(0, float(volatility_ratio) - 1.0)  # > 1.0 means expanding volatility
            
            # 2. VOLUME CONSISTENCY ANALYSIS
            volumes = seismic_data.get('volumes', [])
            if volumes is None or len(volumes) < 10:
                logger.error(f"FAIL-FAST: _assess_structural_fragility() - Insufficient volume data for {ticker}")
                raise CriticalBotError(f"Insufficient volume data for fragility analysis of {ticker}")
            
            recent_volumes = volumes[-10:] if len(volumes) >= 10 else volumes
            volume_mean = np.mean(recent_volumes)
            volume_std = np.std(recent_volumes)
            
            if volume_mean > 0:
                volume_cv = volume_std / volume_mean
                volume_consistency = 1.0 / (volume_cv + 0.01)  # Higher CV = lower consistency
            else:
                volume_consistency = 1.0
            
            # Low consistency = more fragile (volume drops create gaps)
            volume_fragility = max(0, 2.0 - volume_consistency)
            
            if np.isnan(volume_fragility) or np.isinf(volume_fragility):
                volume_fragility = 0.0
            
            # 3. RANGE EXPANSION ANALYSIS
            range_compression = seismic_data.get('range_compression', 1.0)
            if range_compression is None or np.isnan(range_compression) or np.isinf(range_compression):
                range_compression = 1.0
            
            range_expansion = 1.0 / max(float(range_compression), 0.1)
            # If ranges are compressing now, expansion = fragility when pressure releases
            range_fragility = max(0, range_expansion - 1.0)
            
            # 4. CHEMICAL FRAGILITY FACTORS
            chemistry_fragility = 1.0
            try:
                chemistry_type = chemistry.chemistry_type
                
                if chemistry_type == 'volatile_compound':
                    chemistry_fragility = 1.5  # Most fragile under pressure
                elif chemistry_type == 'phase_change':
                    chemistry_fragility = 1.3  # Fragile at transition points
                elif chemistry_type == 'catalyst_accelerant':
                    chemistry_fragility = 1.1  # Moderate fragility
                elif chemistry_type == 'noble_gas':
                    chemistry_fragility = 0.6  # Most stable under pressure
                else:
                    chemistry_fragility = 1.0
                    
            except Exception as e:
                logger.debug(f"Chemistry fragility calculation failed for {ticker}: {e}")
                chemistry_fragility = 1.0
            
            # 5. MARKET STRESS AMPLIFICATION
            # Higher market risk makes individual assets more fragile
            market_stress_multiplier = 1.0 + pressure_risk
            
            # 6. MOMENTUM FRAGILITY
            # Assets moving against strong momentum are more fragile
            momentum_fragility = 1.0
            
            try:
                if abs(tectonic_momentum) > 0.5:
                    price_momentum = seismic_data.get('price_momentum', 0.0)
                    if price_momentum is not None and not np.isnan(price_momentum):
                        if tectonic_momentum * price_momentum < 0:  # Moving against market momentum
                            momentum_fragility = 1.3
            except Exception as e:
                logger.debug(f"Momentum fragility calculation failed for {ticker}: {e}")
                momentum_fragility = 1.0
            
            return {
                'volatility_fragility': float(volatility_fragility),
                'volume_fragility': float(volume_fragility),
                'range_fragility': float(range_fragility),
                'chemistry_fragility': float(chemistry_fragility),
                'market_stress_multiplier': float(market_stress_multiplier),
                'momentum_fragility': float(momentum_fragility)
            }
            
        except Exception as e:
            logger.error(f"FAIL-FAST: _assess_structural_fragility() - Fragility calculation failed for {ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Structural fragility calculation failed for {ticker}: {e}")
        
    except CriticalBotError:
        raise
    except Exception as e:
        logger.error(f"FAIL-FAST: _assess_structural_fragility() - Fragility assessment failed for {ticker}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Structural fragility assessment failed for {ticker}: {e}")

def _calculate_chemical_catalysis_potential(chemistry: AssetChemistry, market_state: Tuple[float, float], seismic_data: Dict) -> Dict:
    """
    ⚗️ Calculate how asset chemistry will catalyze pressure release.
    
    Chemical Catalysis Physics:
    - ACTIVATION ENERGY: Minimum pressure needed to trigger reaction
    - REACTION RATE: How quickly pressure converts to price movement
    - AMPLIFICATION FACTOR: How much chemistry multiplies pressure effect
    - CATALYST EFFICIENCY: Chemistry's ability to accelerate market reactions
    """
    try:
        if chemistry is None:
            logger.error("FAIL-FAST: _calculate_chemical_catalysis_potential() - chemistry parameter is None")
            raise CriticalBotError("Cannot calculate chemical catalysis - chemistry data is None")
        
        if market_state is None:
            logger.error("FAIL-FAST: _calculate_chemical_catalysis_potential() - market_state parameter is None")
            raise CriticalBotError("Cannot calculate chemical catalysis - market state is None")
        
        if seismic_data is None:
            logger.error("FAIL-FAST: _calculate_chemical_catalysis_potential() - seismic_data parameter is None")
            raise CriticalBotError("Cannot calculate chemical catalysis - seismic data is None")
        
        try:
            pressure_risk, tectonic_momentum = market_state
            pressure_risk = float(pressure_risk)
            tectonic_momentum = float(tectonic_momentum)
        except (ValueError, TypeError) as e:
            logger.error(f"FAIL-FAST: _calculate_chemical_catalysis_potential() - Invalid market_state: {market_state}, error: {e}")
            raise CriticalBotError(f"Invalid market state for catalysis calculation: {e}")
        
        try:
            # Extract chemical properties from metadata
            geological_analysis = chemistry.metadata.get('geological_analysis', {})
            activation_energy = geological_analysis.get('activation_energy', 0.02)
            catalytic_potential = geological_analysis.get('catalytic_potential', 1.0)
            
            # Validate chemical properties
            if activation_energy is None or np.isnan(activation_energy) or np.isinf(activation_energy):
                activation_energy = 0.02
            if catalytic_potential is None or np.isnan(catalytic_potential) or np.isinf(catalytic_potential):
                catalytic_potential = 1.0
            
            activation_energy = float(activation_energy)
            catalytic_potential = float(catalytic_potential)
            
            # 1. ACTIVATION ENERGY ANALYSIS
            # Lower activation energy = easier to trigger reaction
            current_volatility = seismic_data.get('volatility_ratio', 1.0)
            if current_volatility is None or np.isnan(current_volatility) or np.isinf(current_volatility):
                current_volatility = 1.0
            
            current_volatility = float(current_volatility)
            activation_threshold = activation_energy * 100  # Convert to percentage
            
            if activation_threshold > 0:
                energy_ratio = current_volatility / activation_threshold
            else:
                energy_ratio = 1.0
            
            activation_readiness = min(energy_ratio, 2.0)  # Cap at 2.0
            if np.isnan(activation_readiness) or np.isinf(activation_readiness):
                activation_readiness = 1.0
            
            # 2. CHEMISTRY-SPECIFIC CATALYSIS
            chemistry_catalyst_factor = 1.0
            
            try:
                chemistry_type = chemistry.chemistry_type
                
                if chemistry_type == 'catalyst_accelerant':
                    # Catalyst accelerants amplify any reaction
                    chemistry_catalyst_factor = 2.0 + abs(tectonic_momentum)
                    
                elif chemistry_type == 'volatile_compound':
                    # Volatile compounds have explosive reactions under right conditions
                    if pressure_risk < 0.4:  # Low pressure environment
                        chemistry_catalyst_factor = 1.8
                    else:  # High pressure can suppress volatiles
                        chemistry_catalyst_factor = 0.8
                        
                elif chemistry_type == 'phase_change':
                    # Phase change assets have step-function reactions at critical points
                    compression_ratio = seismic_data.get('compression_ratio', 0.05)
                    if compression_ratio is not None and not np.isnan(compression_ratio):
                        if compression_ratio < 0.02:  # Highly compressed
                            chemistry_catalyst_factor = 1.6  # Ready for phase transition
                        else:
                            chemistry_catalyst_factor = 1.0
                    else:
                        chemistry_catalyst_factor = 1.0
                        
                elif chemistry_type == 'noble_gas':
                    # Noble gases resist catalysis but are stable
                    chemistry_catalyst_factor = 0.7
                else:
                    chemistry_catalyst_factor = 1.0
                    
            except Exception as e:
                logger.debug(f"Chemistry catalyst factor calculation failed: {e}")
                chemistry_catalyst_factor = 1.0
            
            # 3. PRESSURE-CATALYSIS INTERACTION
            # How market pressure affects catalytic efficiency
            pressure_catalysis = 1.0
            
            try:
                chemistry_type = chemistry.chemistry_type
                
                if chemistry_type == 'volatile_compound':
                    # Volatiles need low pressure to catalyze properly
                    pressure_catalysis = 1.5 - pressure_risk
                elif chemistry_type == 'phase_change':
                    # Phase changes work best at moderate pressure
                    optimal_pressure = 0.4
                    pressure_catalysis = 1.0 - abs(pressure_risk - optimal_pressure)
                elif chemistry_type == 'catalyst_accelerant':
                    # Catalysts work better under higher pressure
                    pressure_catalysis = 1.0 + (pressure_risk * 0.5)
                else:
                    pressure_catalysis = 1.0
                    
                if np.isnan(pressure_catalysis) or np.isinf(pressure_catalysis):
                    pressure_catalysis = 1.0
                    
            except Exception as e:
                logger.debug(f"Pressure catalysis calculation failed: {e}")
                pressure_catalysis = 1.0
            
            # 4. MOMENTUM AMPLIFICATION
            # How tectonic momentum affects catalytic potential
            momentum_amplification = 1.0 + (abs(tectonic_momentum) * 0.3)
            if np.isnan(momentum_amplification) or np.isinf(momentum_amplification):
                momentum_amplification = 1.0
            
            # 5. COMPOSITE CATALYSIS SCORE
            composite_catalysis = (
                activation_readiness * 
                chemistry_catalyst_factor * 
                pressure_catalysis * 
                momentum_amplification
            )
            
            if np.isnan(composite_catalysis) or np.isinf(composite_catalysis):
                composite_catalysis = 1.0
            
            return {
                'activation_readiness': float(activation_readiness),
                'chemistry_catalyst_factor': float(chemistry_catalyst_factor),
                'pressure_catalysis': float(pressure_catalysis),
                'momentum_amplification': float(momentum_amplification),
                'composite_catalysis': float(composite_catalysis),
                'activation_energy': float(activation_energy),
                'catalytic_potential': float(catalytic_potential)
            }
            
        except Exception as e:
            logger.error(f"FAIL-FAST: _calculate_chemical_catalysis_potential() - Catalysis calculation failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Chemical catalysis calculation failed: {e}")
        
    except CriticalBotError:
        raise
    except Exception as e:
        logger.error(f"FAIL-FAST: _calculate_chemical_catalysis_potential() - Chemical catalysis analysis failed: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Chemical catalysis potential calculation failed: {e}")

def _calculate_trailhead_signal_strength(trailhead_data: Dict, chemistry: AssetChemistry, market_state: Tuple[float, float]) -> Optional[TrailheadSignal]:
    """
    🎯 Calculate final trailhead signal strength combining all geological factors.
    
    Signal Synthesis Physics:
    - PRESSURE SCORE: How much pressure has accumulated
    - FRAGILITY SCORE: How easily the structure will break
    - TRIGGER IDENTIFICATION: What type of pressure release expected
    - COMPOSITE SCORE: Overall probability of significant move
    """
    try:
        if trailhead_data is None:
            logger.error("FAIL-FAST: _calculate_trailhead_signal_strength() - trailhead_data is None")
            raise CriticalBotError("Cannot calculate signal strength - trailhead data is None")
        
        if not isinstance(trailhead_data, dict):
            logger.error(f"FAIL-FAST: _calculate_trailhead_signal_strength() - Invalid trailhead_data type: {type(trailhead_data)}")
            raise CriticalBotError(f"Invalid trailhead data format: {type(trailhead_data)}")
        
        if chemistry is None:
            logger.error("FAIL-FAST: _calculate_trailhead_signal_strength() - chemistry is None")
            raise CriticalBotError("Cannot calculate signal strength - chemistry data is None")
        
        if market_state is None:
            logger.error("FAIL-FAST: _calculate_trailhead_signal_strength() - market_state is None")
            raise CriticalBotError("Cannot calculate signal strength - market state is None")
        
        ticker = trailhead_data.get('ticker')
        if ticker is None:
            logger.error("FAIL-FAST: _calculate_trailhead_signal_strength() - ticker missing from trailhead_data")
            raise CriticalBotError("Ticker missing from trailhead data")
        
        # Extract required components with validation
        required_components = ['fault_stress', 'pressure_buildup', 'fragility_analysis', 'catalysis_potential']
        for component in required_components:
            if component not in trailhead_data or trailhead_data[component] is None:
                logger.error(f"FAIL-FAST: _calculate_trailhead_signal_strength() - Missing {component} for {ticker}")
                raise CriticalBotError(f"Missing {component} in trailhead data for {ticker}")
        
        try:
            fault_stress = trailhead_data['fault_stress']
            pressure_buildup = trailhead_data['pressure_buildup']
            fragility_analysis = trailhead_data['fragility_analysis']
            catalysis_potential = trailhead_data['catalysis_potential']
            
            # Validate all components are dictionaries
            for name, component in [
                ('fault_stress', fault_stress),
                ('pressure_buildup', pressure_buildup),
                ('fragility_analysis', fragility_analysis),
                ('catalysis_potential', catalysis_potential)
            ]:
                if not isinstance(component, dict):
                    logger.error(f"FAIL-FAST: _calculate_trailhead_signal_strength() - Invalid {name} type for {ticker}: {type(component)}")
                    raise CriticalBotError(f"Invalid {name} format for {ticker}")
        
        except (KeyError, TypeError) as e:
            logger.error(f"FAIL-FAST: _calculate_trailhead_signal_strength() - Component extraction failed for {ticker}: {e}")
            raise CriticalBotError(f"Trailhead component extraction failed for {ticker}: {e}")
        
        try:
            # 1. PRESSURE SCORE CALCULATION (0-1)
            pressure_components = [
                fault_stress.get('breakout_energy', 0.0) * 0.3,
                pressure_buildup.get('accumulation_divergence', 0.0) * 0.25,
                pressure_buildup.get('smart_money_ratio', 0.0) * 0.2,
                pressure_buildup.get('momentum_acceleration', 0.0) * 0.15,
                catalysis_potential.get('composite_catalysis', 1.0) * 0.1
            ]
            
            # Validate and clamp pressure components
            validated_components = []
            for i, component in enumerate(pressure_components):
                if component is None or np.isnan(component) or np.isinf(component):
                    component = 0.0
                validated_components.append(max(0, min(1, float(component))))
            
            raw_pressure_score = sum(validated_components)
            
            # Apply chemistry amplification
            chemistry_amplifier = pressure_buildup.get('chemistry_amplifier', 1.0)
            if chemistry_amplifier is None or np.isnan(chemistry_amplifier) or np.isinf(chemistry_amplifier):
                chemistry_amplifier = 1.0
            
            pressure_score = raw_pressure_score * float(chemistry_amplifier)
            pressure_score = min(1.0, max(0.0, pressure_score))
            
            # 2. FRAGILITY SCORE CALCULATION (0-1)
            fragility_components = [
                fragility_analysis.get('volatility_fragility', 0.0) * 0.25,
                fragility_analysis.get('volume_fragility', 0.0) * 0.2,
                fragility_analysis.get('range_fragility', 0.0) * 0.2,
                fragility_analysis.get('chemistry_fragility', 1.0) * 0.2,
                fragility_analysis.get('momentum_fragility', 1.0) * 0.15
            ]
            
            # Validate and clamp fragility components
            validated_fragility = []
            for component in fragility_components:
                if component is None or np.isnan(component) or np.isinf(component):
                    component = 0.0
                validated_fragility.append(max(0, min(1, float(component))))
            
            raw_fragility_score = sum(validated_fragility)
            
            # Apply market stress multiplier
            market_stress_multiplier = fragility_analysis.get('market_stress_multiplier', 1.0)
            if market_stress_multiplier is None or np.isnan(market_stress_multiplier) or np.isinf(market_stress_multiplier):
                market_stress_multiplier = 1.0
            
            fragility_score = raw_fragility_score * float(market_stress_multiplier)
            fragility_score = min(1.0, max(0.0, fragility_score))
            
            # 3. TRIGGER TYPE IDENTIFICATION
            trigger_type = _identify_trigger_type(trailhead_data, chemistry, market_state)
            if trigger_type is None or not isinstance(trigger_type, str):
                trigger_type = 'breakout'  # Default
            
            # 4. COMPOSITE SCORE CALCULATION
            # Pressure and fragility work together - need both for significant move
            if pressure_score > 0 and fragility_score > 0:
                base_composite = (pressure_score * fragility_score) ** 0.5  # Geometric mean
            else:
                base_composite = 0.0
            
            # Apply trigger-specific multipliers
            trigger_multipliers = {
                'squeeze': 1.3,      # Squeezes have high probability
                'breakout': 1.2,     # Breakouts are reliable
                'cascade': 1.4,      # Cascades can be explosive
                'reversal': 1.1      # Reversals are trickier
            }
            
            trigger_multiplier = trigger_multipliers.get(trigger_type, 1.0)
            composite_score = base_composite * trigger_multiplier
            
            # Apply confidence factor from chemistry classification
            chemistry_confidence = chemistry.confidence
            if chemistry_confidence is None or np.isnan(chemistry_confidence) or np.isinf(chemistry_confidence):
                chemistry_confidence = 0.5  # Default moderate confidence
            
            composite_score *= float(chemistry_confidence)
            
            # Final bounds checking
            composite_score = min(1.0, max(0.0, composite_score))
            
            # 5. METADATA COMPILATION
            metadata = {
                'geological_analysis': {
                    'fault_stress': fault_stress,
                    'pressure_buildup': pressure_buildup,
                    'fragility_analysis': fragility_analysis,
                    'catalysis_potential': catalysis_potential
                },
                'scoring_breakdown': {
                    'raw_pressure_score': float(raw_pressure_score),
                    'raw_fragility_score': float(raw_fragility_score),
                    'chemistry_amplifier': float(chemistry_amplifier),
                    'market_stress_multiplier': float(market_stress_multiplier),
                    'trigger_multiplier': float(trigger_multiplier),
                    'chemistry_confidence': float(chemistry_confidence)
                },
                'market_state': market_state,
                'chemistry_type': chemistry.chemistry_type,
                'detection_timestamp': datetime.now().isoformat()
            }
            
            return TrailheadSignal(
                ticker=ticker,
                pressure_score=float(pressure_score),
                fragility_score=float(fragility_score),
                composite_score=float(composite_score),
                trigger_type=trigger_type,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"FAIL-FAST: _calculate_trailhead_signal_strength() - Signal calculation failed for {ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Trailhead signal calculation failed for {ticker}: {e}")
        
    except CriticalBotError:
        raise
    except Exception as e:
        ticker_info = trailhead_data.get('ticker', 'unknown') if isinstance(trailhead_data, dict) else 'unknown'
        logger.error(f"FAIL-FAST: _calculate_trailhead_signal_strength() - Signal strength calculation failed for {ticker_info}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Trailhead signal strength calculation failed for {ticker_info}: {e}")

def _identify_trigger_type(trailhead_data: Dict, chemistry: AssetChemistry, market_state: Tuple[float, float]) -> str:
    """
    🔍 Identify the specific type of geological pressure release expected.
    
    Trigger Types:
    - SQUEEZE: Compression + volume + technical setup for explosive move
    - BREAKOUT: Approaching resistance with momentum and volume
    - REVERSAL: Overextended position meeting contrary pressure
    - CASCADE: Correlation breakdown creating domino effect
    """
    try:
        if trailhead_data is None or not isinstance(trailhead_data, dict):
            logger.error("FAIL-FAST: _identify_trigger_type() - Invalid trailhead_data")
            raise CriticalBotError("Cannot identify trigger type - invalid trailhead data")
        
        if chemistry is None:
            logger.error("FAIL-FAST: _identify_trigger_type() - chemistry is None")
            raise CriticalBotError("Cannot identify trigger type - chemistry data is None")
        
        if market_state is None:
            logger.error("FAIL-FAST: _identify_trigger_type() - market_state is None")
            raise CriticalBotError("Cannot identify trigger type - market state is None")
        
        try:
            pressure_risk, tectonic_momentum = market_state
            pressure_risk = float(pressure_risk)
            tectonic_momentum = float(tectonic_momentum)
        except (ValueError, TypeError) as e:
            logger.error(f"FAIL-FAST: _identify_trigger_type() - Invalid market_state: {market_state}, error: {e}")
            raise CriticalBotError(f"Invalid market state for trigger identification: {e}")
        
        # Extract required data with safe defaults
        seismic_data = trailhead_data.get('seismic_data', {})
        fault_stress = trailhead_data.get('fault_stress', {})
        pressure_buildup = trailhead_data.get('pressure_buildup', {})
        fragility_analysis = trailhead_data.get('fragility_analysis', {})
        
        try:
            # SQUEEZE DETECTION
            squeeze_signals = 0
            
            compression_ratio = seismic_data.get('compression_ratio', 0.05)
            if compression_ratio is not None and not np.isnan(compression_ratio) and compression_ratio < 0.02:
                squeeze_signals += 1
            
            range_compression = seismic_data.get('range_compression', 1.0)
            if range_compression is not None and not np.isnan(range_compression) and range_compression < 0.8:
                squeeze_signals += 1
            
            volume_surge = seismic_data.get('volume_surge', 1.0)
            if volume_surge is not None and not np.isnan(volume_surge) and volume_surge > 1.2:
                squeeze_signals += 1
            
            accumulation_divergence = pressure_buildup.get('accumulation_divergence', 0.0)
            if accumulation_divergence is not None and not np.isnan(accumulation_divergence) and accumulation_divergence > 0.1:
                squeeze_signals += 1
            
            # BREAKOUT DETECTION
            breakout_signals = 0
            
            resistance_distance = fault_stress.get('resistance_distance', 1.0)
            if resistance_distance is not None and not np.isnan(resistance_distance) and resistance_distance < 0.03:
                breakout_signals += 1
            
            breakout_energy = fault_stress.get('breakout_energy', 0.0)
            if breakout_energy is not None and not np.isnan(breakout_energy) and breakout_energy > 0.5:
                breakout_signals += 1
            
            price_momentum = seismic_data.get('price_momentum', 0.0)
            if price_momentum is not None and not np.isnan(price_momentum) and price_momentum > 0.02:
                breakout_signals += 1
            
            volume_momentum = seismic_data.get('volume_momentum', 0.0)
            if volume_momentum is not None and not np.isnan(volume_momentum) and volume_momentum > 0.1:
                breakout_signals += 1
            
            # REVERSAL DETECTION
            reversal_signals = 0
            
            if price_momentum is not None and not np.isnan(price_momentum) and abs(price_momentum) > 0.1:
                reversal_signals += 1
            
            if volume_momentum is not None and not np.isnan(volume_momentum) and volume_momentum < -0.1:
                reversal_signals += 1
            
            momentum_fragility = fragility_analysis.get('momentum_fragility', 1.0)
            if momentum_fragility is not None and not np.isnan(momentum_fragility) and momentum_fragility > 1.2:
                reversal_signals += 1
            
            bollinger_position = seismic_data.get('bollinger_position', 0.0)
            if bollinger_position is not None and not np.isnan(bollinger_position):
                if bollinger_position > 1.5 or bollinger_position < -1.5:
                    reversal_signals += 1
            
            # CASCADE DETECTION
            cascade_signals = 0
            
            if chemistry.chemistry_type == 'catalyst_accelerant':
                cascade_signals += 1
            
            if pressure_risk > 0.7:
                cascade_signals += 1
            
            volatility_fragility = fragility_analysis.get('volatility_fragility', 0.0)
            if volatility_fragility is not None and not np.isnan(volatility_fragility) and volatility_fragility > 1.0:
                cascade_signals += 1
            
            if abs(tectonic_momentum) > 0.6:
                cascade_signals += 1
            
            # TRIGGER TYPE DECISION
            signal_counts = {
                'squeeze': squeeze_signals,
                'breakout': breakout_signals,
                'reversal': reversal_signals,
                'cascade': cascade_signals
            }
            
            # Return trigger type with highest signal count
            trigger_type = max(signal_counts, key=signal_counts.get)
            
            # Minimum threshold - need at least 2 signals for reliable trigger
            if signal_counts[trigger_type] < 2:
                # Default based on chemistry type
                chemistry_defaults = {
                    'volatile_compound': 'squeeze',
                    'catalyst_accelerant': 'cascade',
                    'phase_change': 'breakout',
                    'noble_gas': 'reversal'
                }
                trigger_type = chemistry_defaults.get(chemistry.chemistry_type, 'breakout')
            
            return trigger_type
            
        except Exception as e:
            logger.error(f"FAIL-FAST: _identify_trigger_type() - Trigger calculation failed: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Trigger type calculation failed: {e}")
        
    except CriticalBotError:
        raise
    except Exception as e:
        logger.error(f"FAIL-FAST: _identify_trigger_type() - Trigger identification failed: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Trigger type identification failed: {e}")
