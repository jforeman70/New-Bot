import numpy as np
import requests
from typing import Dict, List, Literal, Tuple
from dataclasses import dataclass
import logging

from Market_State_Calculator import CriticalBotError

logger = logging.getLogger(__name__)

ChemistryType = Literal['noble_gas', 'volatile_compound', 'phase_change', 'catalyst_accelerant']

@dataclass
class AssetChemistry:
    """Represents an asset's chemical properties and market behavior profile."""
    ticker: str
    chemistry_type: ChemistryType
    beta: float
    confidence: float
    metadata: Dict

def classify_asset_chemistry(
    tickers: List[str], 
    fmp_key: str,
    market_state: Tuple[float, float]
) -> Dict[str, AssetChemistry]:
    """
    Classify assets based on their chemical reaction to geological market states.
    
    Revolutionary Geological Framework:
    - Markets are geological systems with tectonic pressure and fault lines
    - Assets are chemical compounds that react to pressure/temperature changes
    - State transitions occur at critical pressure points (trailheads)
    - Each asset has unique activation energy and catalytic properties
    
    Chemistry Types Based on Pressure Response:
    - noble_gas: Inert under pressure, stable molecular structure (defensive stocks)
    - volatile_compound: Explosive reaction under low pressure, needs containment (growth)
    - phase_change: Solid→liquid→gas transitions at pressure thresholds (value stocks)  
    - catalyst_accelerant: Accelerates reactions without being consumed (momentum)
    
    Args:
        tickers: Chemical compounds (stock symbols) to analyze
        fmp_key: Data source for molecular analysis
        market_state: Current geological pressure (risk, momentum) coordinates
    
    Returns:
        Dict mapping ticker → Chemical reaction profile
        
    Raises:
        CriticalBotError: On geological survey failure
    """
    
    try:
        # Geological survey validation with fail-fast
        if tickers is None:
            logger.error("FAIL-FAST: classify_asset_chemistry() - tickers parameter is None")
            raise CriticalBotError("Cannot perform chemical analysis - tickers parameter is None")
        
        if not tickers:
            logger.error("FAIL-FAST: classify_asset_chemistry() - Empty ticker list provided")
            raise CriticalBotError("Cannot perform chemical analysis - no compounds provided")
        
        if not isinstance(tickers, list):
            logger.error(f"FAIL-FAST: classify_asset_chemistry() - Invalid tickers type: {type(tickers)}")
            raise CriticalBotError(f"Invalid tickers parameter type: {type(tickers)}")
        
        if fmp_key is None:
            logger.error("FAIL-FAST: classify_asset_chemistry() - fmp_key parameter is None")
            raise CriticalBotError("Cannot access geological survey data - API key is None")
        
        if not fmp_key or fmp_key.strip() == "":
            logger.error("FAIL-FAST: classify_asset_chemistry() - Invalid geological data source")
            raise CriticalBotError("Cannot access geological survey data - invalid API key")
        
        if market_state is None:
            logger.error("FAIL-FAST: classify_asset_chemistry() - market_state parameter is None")
            raise CriticalBotError("Geological pressure readings are None")
        
        if not isinstance(market_state, (tuple, list)) or len(market_state) != 2:
            logger.error(f"FAIL-FAST: classify_asset_chemistry() - Invalid market_state format: {market_state}")
            raise CriticalBotError(f"Geological pressure readings invalid format: {market_state}")
        
        try:
            pressure_risk, tectonic_momentum = market_state
            pressure_risk = float(pressure_risk)
            tectonic_momentum = float(tectonic_momentum)
        except (ValueError, TypeError) as e:
            logger.error(f"FAIL-FAST: classify_asset_chemistry() - Cannot convert market_state to float: {market_state}, error: {e}")
            raise CriticalBotError(f"Market state coordinates cannot be converted to numbers: {market_state}")
        
        # Validate geological pressure bounds
        if not (0 <= pressure_risk <= 1):
            logger.error(f"FAIL-FAST: classify_asset_chemistry() - Pressure risk out of bounds: {pressure_risk}")
            raise CriticalBotError(f"Geological pressure risk exceeds safe limits: {pressure_risk} (must be 0-1)")
        
        if not (-1 <= tectonic_momentum <= 1):
            logger.error(f"FAIL-FAST: classify_asset_chemistry() - Tectonic momentum out of bounds: {tectonic_momentum}")
            raise CriticalBotError(f"Tectonic momentum exceeds safe limits: {tectonic_momentum} (must be -1 to 1)")
        
        logger.info(f"Geological analysis of {len(tickers)} compounds under pressure: risk={pressure_risk:.3f}, momentum={tectonic_momentum:.3f}")
        
        classifications = {}
        failed_compounds = []
        
        for ticker in tickers:
            try:
                if ticker is None:
                    logger.warning("FAIL-FAST: classify_asset_chemistry() - None ticker in list")
                    failed_compounds.append(str(ticker))
                    continue
                
                if not isinstance(ticker, str):
                    logger.warning(f"FAIL-FAST: classify_asset_chemistry() - Invalid ticker type: {type(ticker)}")
                    failed_compounds.append(str(ticker))
                    continue
                
                if len(ticker.strip()) == 0:
                    logger.warning(f"FAIL-FAST: classify_asset_chemistry() - Empty ticker string")
                    failed_compounds.append(ticker)
                    continue
                
                ticker = ticker.strip().upper()
                
                # Perform molecular geological analysis
                molecular_data = _analyze_molecular_structure(ticker, fmp_key)
                if molecular_data is None:
                    logger.warning(f"Molecular structure unknown for compound {ticker}")
                    failed_compounds.append(ticker)
                    continue
                
                # Calculate chemical reaction profile under current pressure
                chemistry = _determine_pressure_reaction_profile(molecular_data, market_state)
                if chemistry is None:
                    logger.error(f"FAIL-FAST: classify_asset_chemistry() - Chemistry calculation returned None for {ticker}")
                    failed_compounds.append(ticker)
                    continue
                
                # Validate reaction confidence (activation energy threshold)
                if chemistry.confidence < 0.4:
                    logger.warning(f"Chemical reaction uncertain for {ticker}: {chemistry.confidence:.2f}")
                    failed_compounds.append(ticker)
                    continue
                
                classifications[ticker] = chemistry
                logger.debug(f"{ticker} classified as {chemistry.chemistry_type} (reaction confidence: {chemistry.confidence:.2f})")
                
            except CriticalBotError:
                raise  # Re-raise critical errors immediately
            except Exception as e:
                logger.error(f"FAIL-FAST: classify_asset_chemistry() - Molecular analysis failed for {ticker}: {type(e).__name__}: {e}")
                failed_compounds.append(ticker)
                continue
        
        # Validate geological survey results
        if not classifications:
            logger.error("FAIL-FAST: classify_asset_chemistry() - Complete geological survey failure - no successful classifications")
            raise CriticalBotError("Chemical analysis failed - no stable reactions identified")
        
        success_rate = len(classifications) / len(tickers)
        if success_rate < 0.5:
            logger.error(f"FAIL-FAST: classify_asset_chemistry() - Critical geological survey failure rate: {success_rate:.1%} ({len(classifications)}/{len(tickers)})")
            raise CriticalBotError(f"Chemical analysis critical failure - success rate too low: {success_rate:.1%}")
        
        if success_rate < 0.6:
            logger.warning(f"Low geological survey success: {success_rate:.1%} ({len(classifications)}/{len(tickers)})")
        
        logger.info(f"Geological survey complete: {len(classifications)} compounds classified")
        return classifications
        
    except CriticalBotError:
        raise  # Re-raise our custom exceptions
    except TypeError as e:
        logger.error(f"FAIL-FAST: classify_asset_chemistry() - Type error in function parameters: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Chemical classification type error: {e}")
    except ValueError as e:
        logger.error(f"FAIL-FAST: classify_asset_chemistry() - Value error in function parameters: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Chemical classification value error: {e}")
    except Exception as e:
        logger.error(f"FAIL-FAST: classify_asset_chemistry() - Unexpected geological survey catastrophic failure: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Chemical classification system failure: {e}")

def _analyze_molecular_structure(ticker: str, fmp_key: str) -> Dict:
    """
    Analyze molecular structure and pressure response characteristics.
    BULLETPROOF: Uses only numpy and built-in functions, no external dependencies.
    """
    try:
        if ticker is None:
            logger.error("FAIL-FAST: _analyze_molecular_structure() - ticker parameter is None")
            raise CriticalBotError("Cannot analyze molecular structure - ticker is None")
        
        if not isinstance(ticker, str):
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Invalid ticker type: {type(ticker)}")
            raise CriticalBotError(f"Invalid ticker type for molecular analysis: {type(ticker)}")
        
        if fmp_key is None:
            logger.error("FAIL-FAST: _analyze_molecular_structure() - fmp_key parameter is None")
            raise CriticalBotError("Cannot access molecular data - API key is None")
        
        # Fetch geological survey data
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={fmp_key}"
        
        try:
            profile_response = requests.get(profile_url, timeout=10)
        except requests.exceptions.Timeout as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Profile API timeout for {ticker}: {e}")
            raise CriticalBotError(f"Molecular data retrieval timeout for {ticker}: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Profile API connection error for {ticker}: {e}")
            raise CriticalBotError(f"Molecular data connection error for {ticker}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Profile API request error for {ticker}: {e}")
            raise CriticalBotError(f"Molecular data request error for {ticker}: {e}")
        
        try:
            profile_response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Profile API HTTP error for {ticker}: {e}")
            raise CriticalBotError(f"Molecular data HTTP error for {ticker}: {e}")
        
        try:
            profile_data = profile_response.json()
        except ValueError as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Profile JSON parsing error for {ticker}: {e}")
            raise CriticalBotError(f"Molecular data JSON parsing failed for {ticker}: {e}")
        
        if profile_data is None:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Profile data is None for {ticker}")
            raise CriticalBotError(f"Molecular profile data is None for {ticker}")
        
        if not isinstance(profile_data, list):
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Profile data not a list for {ticker}: {type(profile_data)}")
            raise CriticalBotError(f"Invalid molecular profile data format for {ticker}")
        
        if len(profile_data) == 0:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Empty profile data for {ticker}")
            raise CriticalBotError(f"Empty molecular profile data for {ticker}")
        
        profile = profile_data[0]
        if profile is None:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Profile object is None for {ticker}")
            raise CriticalBotError(f"Molecular profile object is None for {ticker}")
        
        # Extended price history for geological analysis
        prices_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries=120&apikey={fmp_key}"
        
        try:
            prices_response = requests.get(prices_url, timeout=10)
        except requests.exceptions.Timeout as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Price API timeout for {ticker}: {e}")
            raise CriticalBotError(f"Price data retrieval timeout for {ticker}: {e}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Price API connection error for {ticker}: {e}")
            raise CriticalBotError(f"Price data connection error for {ticker}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Price API request error for {ticker}: {e}")
            raise CriticalBotError(f"Price data request error for {ticker}: {e}")
        
        try:
            prices_response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Price API HTTP error for {ticker}: {e}")
            raise CriticalBotError(f"Price data HTTP error for {ticker}: {e}")
        
        try:
            prices_data = prices_response.json()
        except ValueError as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Price JSON parsing error for {ticker}: {e}")
            raise CriticalBotError(f"Price data JSON parsing failed for {ticker}: {e}")
        
        if prices_data is None:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Price data is None for {ticker}")
            raise CriticalBotError(f"Price data is None for {ticker}")
        
        if not isinstance(prices_data, dict):
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Price data not a dict for {ticker}: {type(prices_data)}")
            raise CriticalBotError(f"Invalid price data format for {ticker}")
        
        if 'historical' not in prices_data:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - No historical data for {ticker}")
            raise CriticalBotError(f"No historical price data for {ticker}")
        
        historical = prices_data['historical']
        if historical is None:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Historical data is None for {ticker}")
            raise CriticalBotError(f"Historical price data is None for {ticker}")
        
        if not isinstance(historical, list):
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Historical data not a list for {ticker}: {type(historical)}")
            raise CriticalBotError(f"Invalid historical data format for {ticker}")
        
        historical = historical[:120]
        
        # Robust data extraction with validation
        prices = []
        volumes = []
        highs = []
        lows = []
        
        for i, h in enumerate(historical):
            try:
                if h is None:
                    logger.debug(f"Skipping None historical entry {i} for {ticker}")
                    continue
                
                close = h.get('close')
                volume = h.get('volume')
                high = h.get('high')
                low = h.get('low')
                
                if close is None or volume is None or high is None or low is None:
                    logger.debug(f"Skipping incomplete historical entry {i} for {ticker}")
                    continue
                
                try:
                    close_val = float(close)
                    volume_val = float(volume)
                    high_val = float(high)
                    low_val = float(low)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping invalid numerical data in entry {i} for {ticker}: {e}")
                    continue
                
                if close_val <= 0 or volume_val < 0 or high_val <= 0 or low_val <= 0:
                    logger.debug(f"Skipping invalid values in entry {i} for {ticker}")
                    continue
                
                if high_val < low_val:
                    logger.debug(f"Skipping invalid high/low in entry {i} for {ticker}")
                    continue
                
                prices.append(close_val)
                volumes.append(volume_val)
                highs.append(high_val)
                lows.append(low_val)
                
            except Exception as e:
                logger.debug(f"Error processing historical entry {i} for {ticker}: {e}")
                continue
        
        if len(prices) < 60:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Insufficient price data for {ticker}: {len(prices)} days (need 60+)")
            raise CriticalBotError(f"Insufficient geological data for {ticker}: only {len(prices)} days available")
        
        # GEOLOGICAL CALCULATIONS - BULLETPROOF VERSION
        
        # 1. Calculate returns with bounds checking
        returns = []
        for i in range(min(len(prices)-1, 119)):
            try:
                if prices[i+1] <= 0:
                    logger.debug(f"Skipping zero/negative price for {ticker} at index {i+1}")
                    continue
                return_val = (prices[i] / prices[i+1]) - 1
                if np.isnan(return_val) or np.isinf(return_val):
                    logger.debug(f"Skipping invalid return for {ticker} at index {i}")
                    continue
                returns.append(return_val)
            except (ZeroDivisionError, OverflowError) as e:
                logger.debug(f"Math error calculating return for {ticker} at index {i}: {e}")
                continue
        
        if len(returns) < 30:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Insufficient returns data for {ticker}: {len(returns)} (need 30+)")
            raise CriticalBotError(f"Insufficient return data for geological analysis of {ticker}")
        
        try:
            returns_array = np.array(returns)
        except Exception as e:
            logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Cannot create returns array for {ticker}: {e}")
            raise CriticalBotError(f"Failed to process returns data for {ticker}: {e}")
        
        # 2. Volume pressure analysis (safe correlation)
        volume_changes = []
        min_length = min(len(volumes)-1, len(returns))
        
        for i in range(min_length):
            try:
                if volumes[i+1] <= 0:
                    continue
                vol_change = (volumes[i] / volumes[i+1]) - 1
                if np.isnan(vol_change) or np.isinf(vol_change):
                    continue
                volume_changes.append(vol_change)
            except (ZeroDivisionError, OverflowError):
                continue
        
        # Safe correlation calculation
        pressure_volume_correlation = 0.0
        if len(volume_changes) >= 20 and len(returns) >= 20:
            try:
                # Manual correlation to avoid dependency issues
                abs_returns = np.abs(returns_array[:len(volume_changes)])
                vol_changes_array = np.array(volume_changes)
                
                abs_returns_std = np.std(abs_returns)
                vol_changes_std = np.std(vol_changes_array)
                
                if abs_returns_std > 0 and vol_changes_std > 0:
                    try:
                        corr_matrix = np.corrcoef(abs_returns, vol_changes_array)
                        if corr_matrix is not None and corr_matrix.shape == (2, 2):
                            pressure_volume_correlation = corr_matrix[0,1]
                            if np.isnan(pressure_volume_correlation) or np.isinf(pressure_volume_correlation):
                                pressure_volume_correlation = 0.0
                    except Exception as e:
                        logger.debug(f"Correlation calculation failed for {ticker}: {e}")
                        pressure_volume_correlation = 0.0
            except Exception as e:
                logger.debug(f"Pressure-volume analysis failed for {ticker}: {e}")
                pressure_volume_correlation = 0.0
        
        # 3. Volatility clustering (safe calculation)
        volatility_clustering = 0.0
        if len(returns) >= 30:
            try:
                # Calculate rolling volatility windows
                window_size = 10
                volatility_windows = []
                
                for i in range(len(returns) - window_size):
                    try:
                        window_data = returns_array[i:i+window_size]
                        window_vol = np.std(window_data)
                        if not np.isnan(window_vol) and not np.isinf(window_vol):
                            volatility_windows.append(window_vol)
                    except Exception:
                        continue
                
                if len(volatility_windows) >= 10:
                    try:
                        vol_array = np.array(volatility_windows)
                        vol_std_prev = np.std(vol_array[:-1])
                        vol_std_next = np.std(vol_array[1:])
                        
                        if vol_std_prev > 0 and vol_std_next > 0:
                            corr_matrix = np.corrcoef(vol_array[:-1], vol_array[1:])
                            if corr_matrix is not None and corr_matrix.shape == (2, 2):
                                volatility_clustering = corr_matrix[0,1]
                                if np.isnan(volatility_clustering) or np.isinf(volatility_clustering):
                                    volatility_clustering = 0.0
                    except Exception as e:
                        logger.debug(f"Volatility clustering calculation failed for {ticker}: {e}")
                        volatility_clustering = 0.0
            except Exception as e:
                logger.debug(f"Volatility clustering analysis failed for {ticker}: {e}")
                volatility_clustering = 0.0
        
        # 4. Fault line analysis (simplified peak detection)
        fault_line_strength = 0.0
        if len(prices) >= 20:
            try:
                price_array = np.array(prices)
                peaks = 0
                troughs = 0
                
                # Find local maxima (peaks) - safe indexing
                for i in range(5, len(price_array)-5):
                    try:
                        if (price_array[i] > price_array[i-1] and 
                            price_array[i] > price_array[i+1] and
                            price_array[i] > price_array[i-2] and 
                            price_array[i] > price_array[i+2]):
                            peaks += 1
                    except IndexError:
                        break
                
                # Find local minima (troughs) - safe indexing
                for i in range(5, len(price_array)-5):
                    try:
                        if (price_array[i] < price_array[i-1] and 
                            price_array[i] < price_array[i+1] and
                            price_array[i] < price_array[i-2] and 
                            price_array[i] < price_array[i+2]):
                            troughs += 1
                    except IndexError:
                        break
                
                fault_line_strength = min((peaks + troughs) / 20.0, 1.0)
                if np.isnan(fault_line_strength) or np.isinf(fault_line_strength):
                    fault_line_strength = 0.0
                    
            except Exception as e:
                logger.debug(f"Fault line analysis failed for {ticker}: {e}")
                fault_line_strength = 0.0
        
        # 5. Activation energy (safe calculation)
        activation_energy = 0.02  # Default value
        if len(prices) >= 20:
            try:
                daily_ranges = []
                min_len = min(len(highs), len(lows), len(prices))
                
                for i in range(min_len):
                    try:
                        if prices[i] > 0 and highs[i] >= lows[i]:
                            range_val = (highs[i] - lows[i]) / prices[i]
                            if not np.isnan(range_val) and not np.isinf(range_val) and range_val >= 0:
                                daily_ranges.append(range_val)
                    except (ZeroDivisionError, OverflowError):
                        continue
                
                if len(daily_ranges) >= 10:
                    try:
                        activation_energy = np.percentile(daily_ranges, 75)
                        if np.isnan(activation_energy) or np.isinf(activation_energy) or activation_energy < 0:
                            activation_energy = 0.02
                    except Exception as e:
                        logger.debug(f"Activation energy calculation failed for {ticker}: {e}")
                        activation_energy = 0.02
            except Exception as e:
                logger.debug(f"Activation energy analysis failed for {ticker}: {e}")
                activation_energy = 0.02
        
        # 6. Molecular stability (safe calculation)
        stability_ratio = 1.0
        if len(returns) >= 30:
            try:
                returns_abs = np.abs(returns_array)
                high_stress_threshold = np.percentile(returns_abs, 75)
                low_stress_threshold = np.percentile(returns_abs, 25)
                
                high_stress_returns = returns_array[returns_abs > high_stress_threshold]
                low_stress_returns = returns_array[returns_abs < low_stress_threshold]
                
                if len(high_stress_returns) > 0 and len(low_stress_returns) > 0:
                    try:
                        high_stress_vol = np.std(high_stress_returns)
                        low_stress_vol = np.std(low_stress_returns)
                        
                        if high_stress_vol > 0:
                            stability_ratio = low_stress_vol / high_stress_vol
                            if np.isnan(stability_ratio) or np.isinf(stability_ratio) or stability_ratio < 0:
                                stability_ratio = 1.0
                    except (ZeroDivisionError, OverflowError):
                        stability_ratio = 1.0
            except Exception as e:
                logger.debug(f"Stability ratio calculation failed for {ticker}: {e}")
                stability_ratio = 1.0
        
        # 7. Seismic frequency (safe calculation)
        seismic_frequency = 0.0
        if len(returns) > 0:
            try:
                returns_std = np.std(returns_array)
                if returns_std > 0 and not np.isnan(returns_std) and not np.isinf(returns_std):
                    extreme_threshold = 2 * returns_std
                    extreme_moves = np.sum(np.abs(returns_array) > extreme_threshold)
                    seismic_frequency = extreme_moves / len(returns_array)
                    if np.isnan(seismic_frequency) or np.isinf(seismic_frequency):
                        seismic_frequency = 0.0
            except Exception as e:
                logger.debug(f"Seismic frequency calculation failed for {ticker}: {e}")
                seismic_frequency = 0.0
        
        # Extract basic profile data with validation
        beta = profile.get('beta')
        if beta is None or not isinstance(beta, (int, float)) or np.isnan(beta) or np.isinf(beta):
            beta = 1.0
        
        sector = profile.get('sector', 'Unknown')
        if not isinstance(sector, str) or sector is None:
            sector = 'Unknown'
        
        market_cap = profile.get('mktCap', 0)
        if market_cap is None or not isinstance(market_cap, (int, float)) or np.isnan(market_cap) or np.isinf(market_cap):
            market_cap = 0.0
        
        return {
            'ticker': ticker,
            'beta': float(beta),
            'sector': sector,
            'market_cap': float(market_cap),
            'prices': prices,
            'returns': returns,
            'volumes': volumes,
            # BULLETPROOF GEOLOGICAL METRICS
            'pressure_volume_correlation': float(pressure_volume_correlation),
            'volatility_clustering': float(volatility_clustering),
            'fault_line_strength': float(fault_line_strength),
            'activation_energy': float(activation_energy),
            'stability_ratio': float(stability_ratio),
            'catalytic_potential': abs(float(beta)),
            'seismic_frequency': float(seismic_frequency)
        }
        
    except CriticalBotError:
        raise  # Re-raise critical errors immediately
    except requests.RequestException as e:
        logger.error(f"FAIL-FAST: _analyze_molecular_structure() - API request failed for {ticker}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Molecular data API request failed for {ticker}: {e}")
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Data parsing error for {ticker}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Molecular data parsing failed for {ticker}: {e}")
    except Exception as e:
        logger.error(f"FAIL-FAST: _analyze_molecular_structure() - Unexpected error for {ticker}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Molecular structure analysis failed for {ticker}: {e}")

def _determine_pressure_reaction_profile(molecular_data: Dict, market_state: Tuple[float, float]) -> AssetChemistry:
    """
    Determine chemical reaction profile under geological pressure.
    BULLETPROOF: All calculations validated and bounded.
    """
    try:
        if molecular_data is None:
            logger.error("FAIL-FAST: _determine_pressure_reaction_profile() - molecular_data is None")
            raise CriticalBotError("Cannot determine reaction profile - molecular data is None")
        
        if not isinstance(molecular_data, dict):
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Invalid molecular_data type: {type(molecular_data)}")
            raise CriticalBotError(f"Invalid molecular data type: {type(molecular_data)}")
        
        if market_state is None:
            logger.error("FAIL-FAST: _determine_pressure_reaction_profile() - market_state is None")
            raise CriticalBotError("Cannot determine reaction profile - market state is None")
        
        try:
            pressure_risk, tectonic_momentum = market_state
            pressure_risk = float(pressure_risk)
            tectonic_momentum = float(tectonic_momentum)
        except (ValueError, TypeError) as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Cannot convert market_state: {market_state}, error: {e}")
            raise CriticalBotError(f"Invalid market state format: {market_state}")
        
        ticker = molecular_data.get('ticker')
        if ticker is None:
            logger.error("FAIL-FAST: _determine_pressure_reaction_profile() - ticker missing from molecular_data")
            raise CriticalBotError("Ticker missing from molecular data")
        
        # Extract geological properties with validation
        pressure_volume_corr = molecular_data.get('pressure_volume_correlation')
        if pressure_volume_corr is None or not isinstance(pressure_volume_corr, (int, float)) or np.isnan(pressure_volume_corr):
            logger.warning(f"Invalid pressure_volume_correlation for {ticker}: {pressure_volume_corr}, using default 0.0")
            pressure_volume_corr = 0.0
        
        volatility_clustering = molecular_data.get('volatility_clustering')
        if volatility_clustering is None or not isinstance(volatility_clustering, (int, float)) or np.isnan(volatility_clustering):
            logger.warning(f"Invalid volatility_clustering for {ticker}: {volatility_clustering}, using default 0.0")
            volatility_clustering = 0.0
        
        fault_line_strength = molecular_data.get('fault_line_strength')
        if fault_line_strength is None or not isinstance(fault_line_strength, (int, float)) or np.isnan(fault_line_strength):
            logger.warning(f"Invalid fault_line_strength for {ticker}: {fault_line_strength}, using default 0.0")
            fault_line_strength = 0.0
        
        activation_energy = molecular_data.get('activation_energy')
        if activation_energy is None or not isinstance(activation_energy, (int, float)) or np.isnan(activation_energy):
            logger.warning(f"Invalid activation_energy for {ticker}: {activation_energy}, using default 0.02")
            activation_energy = 0.02
        
        stability_ratio = molecular_data.get('stability_ratio')
        if stability_ratio is None or not isinstance(stability_ratio, (int, float)) or np.isnan(stability_ratio):
            logger.warning(f"Invalid stability_ratio for {ticker}: {stability_ratio}, using default 1.0")
            stability_ratio = 1.0
        
        catalytic_potential = molecular_data.get('catalytic_potential')
        if catalytic_potential is None or not isinstance(catalytic_potential, (int, float)) or np.isnan(catalytic_potential):
            logger.warning(f"Invalid catalytic_potential for {ticker}: {catalytic_potential}, using default 1.0")
            catalytic_potential = 1.0
        
        seismic_frequency = molecular_data.get('seismic_frequency')
        if seismic_frequency is None or not isinstance(seismic_frequency, (int, float)) or np.isnan(seismic_frequency):
            logger.warning(f"Invalid seismic_frequency for {ticker}: {seismic_frequency}, using default 0.0")
            seismic_frequency = 0.0
        
        beta = molecular_data.get('beta')
        if beta is None or not isinstance(beta, (int, float)) or np.isnan(beta):
            logger.warning(f"Invalid beta for {ticker}: {beta}, using default 1.0")
            beta = 1.0
        
        # Convert to float and validate ranges
        try:
            pressure_volume_corr = float(pressure_volume_corr)
            volatility_clustering = float(volatility_clustering)
            fault_line_strength = float(fault_line_strength)
            activation_energy = float(activation_energy)
            stability_ratio = float(stability_ratio)
            catalytic_potential = float(catalytic_potential)
            seismic_frequency = float(seismic_frequency)
            beta = float(beta)
        except (ValueError, TypeError) as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Cannot convert geological properties to float for {ticker}: {e}")
            raise CriticalBotError(f"Invalid geological property values for {ticker}: {e}")
        
        # Validate ranges
        if not (-1 <= pressure_volume_corr <= 1):
            logger.warning(f"Pressure-volume correlation out of range for {ticker}: {pressure_volume_corr}, clamping to [-1,1]")
            pressure_volume_corr = max(-1, min(1, pressure_volume_corr))
        
        if not (0 <= volatility_clustering <= 1):
            logger.warning(f"Volatility clustering out of range for {ticker}: {volatility_clustering}, clamping to [0,1]")
            volatility_clustering = max(0, min(1, volatility_clustering))
        
        if not (0 <= fault_line_strength <= 1):
            logger.warning(f"Fault line strength out of range for {ticker}: {fault_line_strength}, clamping to [0,1]")
            fault_line_strength = max(0, min(1, fault_line_strength))
        
        if activation_energy < 0:
            logger.warning(f"Negative activation energy for {ticker}: {activation_energy}, using absolute value")
            activation_energy = abs(activation_energy)
        
        if stability_ratio < 0:
            logger.warning(f"Negative stability ratio for {ticker}: {stability_ratio}, using absolute value")
            stability_ratio = abs(stability_ratio)
        
        if seismic_frequency < 0:
            logger.warning(f"Negative seismic frequency for {ticker}: {seismic_frequency}, using absolute value")
            seismic_frequency = abs(seismic_frequency)
        
        # CUTTING-EDGE CHEMISTRY SCORING BASED ON PHYSICS
        
        # NOBLE GAS: Inert, stable molecular structure
        noble_gas_score = 0.0
        try:
            if stability_ratio > 0.8:
                noble_gas_score += 0.4
            if activation_energy < 0.02:
                noble_gas_score += 0.3
            if abs(pressure_volume_corr) < 0.3:
                noble_gas_score += 0.3
            if seismic_frequency < 0.05:
                noble_gas_score += 0.2
        except Exception as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Noble gas scoring failed for {ticker}: {e}")
            raise CriticalBotError(f"Noble gas scoring calculation failed for {ticker}: {e}")
        
        # VOLATILE COMPOUND: Explosive under specific conditions
        volatile_compound_score = 0.0
        try:
            if volatility_clustering > 0.4:
                volatile_compound_score += 0.4
            if activation_energy > 0.04:
                volatile_compound_score += 0.3
            if pressure_volume_corr > 0.5:
                volatile_compound_score += 0.3
            if seismic_frequency > 0.15:
                volatile_compound_score += 0.2
        except Exception as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Volatile compound scoring failed for {ticker}: {e}")
            raise CriticalBotError(f"Volatile compound scoring calculation failed for {ticker}: {e}")
        
        # PHASE CHANGE: Undergoes state transitions at critical points
        phase_change_score = 0.0
        try:
            if fault_line_strength > 0.6:
                phase_change_score += 0.4
            if 0.4 <= stability_ratio <= 0.8:
                phase_change_score += 0.3
            if 0.02 <= activation_energy <= 0.04:
                phase_change_score += 0.3
            if 0.3 <= abs(pressure_volume_corr) <= 0.6:
                phase_change_score += 0.2
        except Exception as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Phase change scoring failed for {ticker}: {e}")
            raise CriticalBotError(f"Phase change scoring calculation failed for {ticker}: {e}")
        
        # CATALYST ACCELERANT: Accelerates reactions without being consumed
        catalyst_accelerant_score = 0.0
        try:
            if catalytic_potential > 1.5:
                catalyst_accelerant_score += 0.4
            if pressure_volume_corr < -0.3:
                catalyst_accelerant_score += 0.3
            if volatility_clustering < 0.2:
                catalyst_accelerant_score += 0.3
            if 0.05 <= seismic_frequency <= 0.12:
                catalyst_accelerant_score += 0.2
        except Exception as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Catalyst accelerant scoring failed for {ticker}: {e}")
            raise CriticalBotError(f"Catalyst accelerant scoring calculation failed for {ticker}: {e}")
        
        # GEOLOGICAL PRESSURE ADJUSTMENTS
        try:
            if pressure_risk > 0.7:  # High pressure market
                noble_gas_score *= 1.3
                volatile_compound_score *= 0.7
            elif pressure_risk < 0.3:  # Low pressure market
                volatile_compound_score *= 1.4
                catalyst_accelerant_score *= 1.2
            
            # Tectonic momentum adjustments
            if abs(tectonic_momentum) > 0.5:
                catalyst_accelerant_score *= 1.3
                phase_change_score *= 1.1
        except Exception as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Pressure adjustment calculation failed for {ticker}: {e}")
            raise CriticalBotError(f"Pressure adjustment calculation failed for {ticker}: {e}")
        
        # Calculate final chemistry type
        try:
            scores = {
                'noble_gas': noble_gas_score,
                'volatile_compound': volatile_compound_score,
                'phase_change': phase_change_score,
                'catalyst_accelerant': catalyst_accelerant_score
            }
            
            # Validate all scores are numbers
            for chem_type, score in scores.items():
                if not isinstance(score, (int, float)) or np.isnan(score) or np.isinf(score):
                    logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Invalid score for {chem_type} in {ticker}: {score}")
                    raise CriticalBotError(f"Invalid chemistry score calculated for {ticker}")
            
            chemistry_type = max(scores, key=scores.get)
            raw_confidence = scores[chemistry_type]
            
            # Advanced confidence calculation
            max_score = max(scores.values())
            second_max = sorted(scores.values())[-2] if len(scores.values()) > 1 else 0
            separation = max_score - second_max
            
            confidence = min((raw_confidence + separation) / 2.0, 1.0)
            confidence = max(confidence, 0.1)  # Minimum confidence
            
            if not isinstance(confidence, (int, float)) or np.isnan(confidence) or np.isinf(confidence):
                logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Invalid confidence calculated for {ticker}: {confidence}")
                raise CriticalBotError(f"Invalid confidence value calculated for {ticker}")
            
        except Exception as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Final scoring calculation failed for {ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Final chemistry scoring failed for {ticker}: {e}")
        
        # Enhanced metadata
        try:
            metadata = {
                'geological_analysis': {
                    'pressure_volume_correlation': pressure_volume_corr,
                    'volatility_clustering': volatility_clustering,
                    'fault_line_strength': fault_line_strength,
                    'activation_energy': activation_energy,
                    'stability_ratio': stability_ratio,
                    'catalytic_potential': catalytic_potential,
                    'seismic_frequency': seismic_frequency
                },
                'reaction_scores': scores,
                'market_pressure': pressure_risk,
                'tectonic_momentum': tectonic_momentum,
                'confidence_factors': {
                    'raw_score': raw_confidence,
                    'separation': separation
                }
            }
        except Exception as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Metadata creation failed for {ticker}: {e}")
            raise CriticalBotError(f"Metadata creation failed for {ticker}: {e}")
        
        try:
            return AssetChemistry(
                ticker=ticker,
                chemistry_type=chemistry_type,
                beta=beta,
                confidence=confidence,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - AssetChemistry creation failed for {ticker}: {type(e).__name__}: {e}")
            raise CriticalBotError(f"Asset chemistry object creation failed for {ticker}: {e}")
        
    except CriticalBotError:
        raise  # Re-raise critical errors immediately
    except Exception as e:
        ticker_info = molecular_data.get('ticker', 'unknown') if isinstance(molecular_data, dict) else 'unknown'
        logger.error(f"FAIL-FAST: _determine_pressure_reaction_profile() - Unexpected reaction analysis failure for {ticker_info}: {type(e).__name__}: {e}")
        raise CriticalBotError(f"Chemical reaction profile analysis failed for {ticker_info}: {e}")
