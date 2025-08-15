"""
Risk Manager - Production Ready with OH SHIT Handle
Zero tolerance for risk breaches. Fail-safe position sizing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    """Risk level classifications"""
    SAFE = "SAFE"
    ELEVATED = "ELEVATED" 
    DANGEROUS = "DANGEROUS"
    EMERGENCY = "EMERGENCY"
    LOCKDOWN = "LOCKDOWN"

class PositionSize(NamedTuple):
    """Immutable position sizing decision"""
    symbol: str
    direction: str  # 'LONG', 'SHORT', 'FLAT'
    size_usd: float
    size_percent: float  # Percent of account
    leverage: float
    max_loss_usd: float
    confidence_level: float
    risk_level: RiskLevel
    timestamp: datetime
    rationale: str

@dataclass
class AccountState:
    """Current account state"""
    total_equity: float
    available_cash: float
    total_exposure: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    max_drawdown: float
    open_positions: Dict[str, float]  # symbol -> position_value
    
class EmergencyProtocol:
    """OH SHIT HANDLE - Emergency position management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # HARD LIMITS - these trigger immediate action
        self.max_daily_loss_pct = 3.0      # 3% daily loss = emergency close
        self.max_weekly_loss_pct = 8.0     # 8% weekly loss = lockdown
        self.max_monthly_loss_pct = 15.0   # 15% monthly = full reset
        self.max_single_position_pct = 5.0 # 5% max per position
        self.max_total_exposure_pct = 25.0 # 25% max total exposure
        
        # Black swan triggers
        self.vix_panic_level = 45          # VIX > 45 = market panic
        self.correlation_break_threshold = 0.8  # Major correlation breakdown
        
        self.emergency_active = False
        self.lockdown_until = None
        
    def check_emergency_conditions(self, account: AccountState, 
                                 market_state: Dict) -> Tuple[RiskLevel, str]:
        """
        Check if emergency protocols should activate
        Returns: (risk_level, reason)
        """
        reasons = []
        
        # Check account-based triggers
        if account.daily_pnl / account.total_equity * 100 <= -self.max_daily_loss_pct:
            self.emergency_active = True
            return RiskLevel.EMERGENCY, f"Daily loss limit breached: {account.daily_pnl/account.total_equity*100:.1f}%"
            
        if account.weekly_pnl / account.total_equity * 100 <= -self.max_weekly_loss_pct:
            self.lockdown_until = datetime.now() + timedelta(days=7)
            return RiskLevel.LOCKDOWN, f"Weekly loss limit breached: {account.weekly_pnl/account.total_equity*100:.1f}%"
            
        if account.monthly_pnl / account.total_equity * 100 <= -self.max_monthly_loss_pct:
            self.lockdown_until = datetime.now() + timedelta(days=30)
            return RiskLevel.LOCKDOWN, f"Monthly loss limit breached: {account.monthly_pnl/account.total_equity*100:.1f}%"
            
        # Check exposure limits
        exposure_pct = account.total_exposure / account.total_equity * 100
        if exposure_pct > self.max_total_exposure_pct:
            reasons.append(f"Exposure limit: {exposure_pct:.1f}% > {self.max_total_exposure_pct}%")
            
        # Check market-based triggers
        current_vix = market_state.get('vix', 20)
        if current_vix > self.vix_panic_level:
            self.emergency_active = True
            return RiskLevel.EMERGENCY, f"VIX panic level: {current_vix} > {self.vix_panic_level}"
            
        # Check if still in lockdown
        if self.lockdown_until and datetime.now() < self.lockdown_until:
            return RiskLevel.LOCKDOWN, f"In lockdown until {self.lockdown_until}"
            
        # Determine risk level
        if reasons:
            return RiskLevel.DANGEROUS, "; ".join(reasons)
        elif exposure_pct > self.max_total_exposure_pct * 0.8:
            return RiskLevel.ELEVATED, f"High exposure: {exposure_pct:.1f}%"
        else:
            return RiskLevel.SAFE, "All systems normal"
            
    def emergency_flatten(self, account: AccountState) -> List[str]:
        """
        Emergency position flattening - close everything NOW
        Returns list of symbols to close
        """
        self.logger.critical("🚨 EMERGENCY FLATTEN ACTIVATED 🚨")
        
        symbols_to_close = list(account.open_positions.keys())
        
        self.logger.critical(f"Closing all positions: {symbols_to_close}")
        
        return symbols_to_close
        
    def reset_emergency(self):
        """Reset emergency state after manual review"""
        self.emergency_active = False
        self.lockdown_until = None
        self.logger.warning("Emergency state manually reset")

class RiskManager:
    """
    Production-ready risk manager with emergency protocols
    Calculates position sizes and enforces hard limits
    """
    
    def __init__(self, account_equity: float):
        self.account_equity = account_equity
        self.logger = self._setup_logging()
        self.emergency = EmergencyProtocol()
        
        # Position sizing parameters
        self.base_position_pct = 1.0      # 1% base position
        self.max_position_pct = 4.0       # 4% max position (conservative)
        self.confidence_multiplier = 2.0  # Scale by confidence
        
        # Leverage constraints by asset class
        self.max_leverage = {
            'crypto': 5.0,    # 5x max on crypto
            'forex': 10.0,    # 10x max on forex  
            'equity': 2.0,    # 2x max on equities
            'default': 1.0    # 1x default
        }
        
        # Risk tracking
        self.position_history = []
        self.daily_trades = []
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
        
    def _classify_asset(self, symbol: str) -> str:
        """Classify asset for leverage limits"""
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'USDT', 'BUSD']
        forex_symbols = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF']
        
        symbol_upper = symbol.upper()
        
        for crypto in crypto_symbols:
            if crypto in symbol_upper:
                return 'crypto'
                
        for forex in forex_symbols:
            if forex in symbol_upper and len(symbol) == 6:  # EURUSD format
                return 'forex'
                
        return 'equity'
        
    def _calculate_kelly_fraction(self, win_rate: float, avg_win: float, 
                                avg_loss: float) -> float:
        """
        Calculate Kelly criterion for position sizing
        Conservative implementation with caps
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 0.0
            
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly at 25% (quarter Kelly for safety)
        kelly_capped = min(0.25, max(0, kelly_fraction))
        
        return kelly_capped
        
    def calculate_position_size(self, symbol: str, signal_strength: float,
                              confidence: float, direction: str,
                              account_state: AccountState,
                              market_state: Dict) -> PositionSize:
        """
        Calculate optimal position size with all safety checks
        """
        try:
            # Check emergency conditions first
            risk_level, risk_reason = self.emergency.check_emergency_conditions(
                account_state, market_state
            )
            
            # Emergency protocols
            if risk_level == RiskLevel.LOCKDOWN:
                return PositionSize(
                    symbol=symbol,
                    direction='FLAT',
                    size_usd=0.0,
                    size_percent=0.0,
                    leverage=1.0,
                    max_loss_usd=0.0,
                    confidence_level=0.0,
                    risk_level=risk_level,
                    timestamp=datetime.now(),
                    rationale=f"LOCKDOWN: {risk_reason}"
                )
                
            if risk_level == RiskLevel.EMERGENCY:
                # Only allow closing positions, no new positions
                if direction != 'FLAT':
                    return PositionSize(
                        symbol=symbol,
                        direction='FLAT',
                        size_usd=0.0,
                        size_percent=0.0,
                        leverage=1.0,
                        max_loss_usd=0.0,
                        confidence_level=0.0,
                        risk_level=risk_level,
                        timestamp=datetime.now(),
                        rationale=f"EMERGENCY - No new positions: {risk_reason}"
                    )
                    
            # Base position sizing
            base_size_pct = self.base_position_pct
            
            # Scale by signal strength (0-100)
            signal_multiplier = signal_strength / 100.0
            
            # Scale by confidence (0-1)
            confidence_multiplier = confidence * self.confidence_multiplier
            
            # Calculate raw position size
            raw_size_pct = base_size_pct * signal_multiplier * confidence_multiplier
            
            # Apply caps
            max_allowed_pct = self.max_position_pct
            if risk_level == RiskLevel.DANGEROUS:
                max_allowed_pct *= 0.5  # Half size in dangerous conditions
            elif risk_level == RiskLevel.ELEVATED:
                max_allowed_pct *= 0.75  # Reduce size in elevated risk
                
            final_size_pct = min(raw_size_pct, max_allowed_pct)
            
            # Check if we already have position in this symbol
            existing_position = account_state.open_positions.get(symbol, 0)
            if abs(existing_position) > 0:
                # Reduce new position size if we already have exposure
                final_size_pct *= 0.5
                
            # Calculate USD size
            size_usd = account_state.total_equity * final_size_pct / 100
            
            # Determine leverage
            asset_class = self._classify_asset(symbol)
            max_lev = self.max_leverage.get(asset_class, 1.0)
            
            # Reduce leverage in risky conditions
            if risk_level in [RiskLevel.DANGEROUS, RiskLevel.ELEVATED]:
                max_lev = min(max_lev, 2.0)
                
            # Conservative leverage for low confidence
            if confidence < 0.7:
                max_lev = min(max_lev, 1.5)
                
            leverage = min(max_lev, confidence * max_lev)
            
            # Calculate max loss (2% of position)
            max_loss_usd = size_usd * 0.02
            
            # Final validation
            if size_usd < 50:  # Minimum position size
                return PositionSize(
                    symbol=symbol,
                    direction='FLAT',
                    size_usd=0.0,
                    size_percent=0.0,
                    leverage=1.0,
                    max_loss_usd=0.0,
                    confidence_level=confidence,
                    risk_level=risk_level,
                    timestamp=datetime.now(),
                    rationale="Position too small - below minimum"
                )
                
            rationale = f"Signal:{signal_strength:.0f}, Conf:{confidence:.2f}, Risk:{risk_level.value}"
            
            position = PositionSize(
                symbol=symbol,
                direction=direction,
                size_usd=size_usd,
                size_percent=final_size_pct,
                leverage=leverage,
                max_loss_usd=max_loss_usd,
                confidence_level=confidence,
                risk_level=risk_level,
                timestamp=datetime.now(),
                rationale=rationale
            )
            
            self.logger.info(f"Position calculated: {position}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Position sizing failed: {e}")
            # Return safe default
            return PositionSize(
                symbol=symbol,
                direction='FLAT',
                size_usd=0.0,
                size_percent=0.0,
                leverage=1.0,
                max_loss_usd=0.0,
                confidence_level=0.0,
                risk_level=RiskLevel.EMERGENCY,
                timestamp=datetime.now(),
                rationale=f"SYSTEM ERROR: {e}"
            )
            
    def check_stop_loss(self, symbol: str, entry_price: float, 
                       current_price: float, position_size: PositionSize) -> bool:
        """
        Check if stop loss should trigger
        """
        if position_size.direction == 'FLAT':
            return False
            
        # Calculate current P&L
        if position_size.direction == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price
            
        current_loss = position_size.size_usd * abs(pnl_pct)
        
        # Trigger stop if max loss exceeded
        if current_loss >= position_size.max_loss_usd:
            self.logger.warning(f"Stop loss triggered for {symbol}: "
                              f"Loss ${current_loss:.2f} >= ${position_size.max_loss_usd:.2f}")
            return True
            
        return False
        
    def get_emergency_status(self) -> Dict:
        """Get current emergency protocol status"""
        return {
            'emergency_active': self.emergency.emergency_active,
            'lockdown_until': self.emergency.lockdown_until,
            'daily_loss_limit': self.emergency.max_daily_loss_pct,
            'weekly_loss_limit': self.emergency.max_weekly_loss_pct,
            'vix_panic_level': self.emergency.vix_panic_level,
            'max_position_pct': self.max_position_pct,
            'timestamp': datetime.now()
        }
        
    def manual_emergency_override(self, override_code: str = "MANUAL_RESET"):
        """Manual emergency reset - use with extreme caution"""
        if override_code == "MANUAL_RESET":
            self.emergency.reset_emergency()
            self.logger.critical("🔓 EMERGENCY MANUALLY RESET BY OPERATOR")
        else:
            self.logger.error("Invalid override code")

# Usage example
if __name__ == "__main__":
    # Initialize with $10k account
    risk_manager = RiskManager(account_equity=10000)
    
    # Mock account state
    account = AccountState(
        total_equity=10000,
        available_cash=8000,
        total_exposure=2000,
        daily_pnl=-100,
        weekly_pnl=50,
        monthly_pnl=200,
        max_drawdown=500,
        open_positions={'SPY': 1000}
    )
    
    # Mock market state
    market = {
        'vix': 25,
        'regime': 'RISK_ON'
    }
    
    # Calculate position size
    position = risk_manager.calculate_position_size(
        symbol='QQQ',
        signal_strength=85,
        confidence=0.8,
        direction='LONG',
        account_state=account,
        market_state=market
    )
    
    print(f"Calculated Position: {position}")
    
    # Check emergency status
    emergency_status = risk_manager.get_emergency_status()
    print(f"Emergency Status: {emergency_status}")
