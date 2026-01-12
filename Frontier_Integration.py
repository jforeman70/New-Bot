"""
Frontier_Integration.py
🔗 Integration layer between Optimal_Frontier and Portfolio_Synthesizer

This module provides methods to integrate efficient frontier optimization
with the existing Catalyst Framework's Portfolio_Synthesizer, enabling
the system to find optimal portfolio allocations using Modern Portfolio Theory.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

from Optimal_Frontier import OptimalFrontier, PortfolioMetrics
from Portfolio_Synthesizer import PortfolioPosition
from Trailhead_Detector import TrailheadSignal

logger = logging.getLogger(__name__)


class FrontierIntegration:
    """
    🔗 Integration layer for Optimal Frontier + Portfolio Synthesizer.
    
    This class bridges the gap between:
    1. Trailhead signals (geological pressure points)
    2. Efficient frontier optimization (MPT)
    3. Portfolio positions (execution-ready allocations)
    """
    
    def __init__(
        self,
        historical_data_days: int = 252,  # 1 year
        risk_free_rate: float = 0.02,
        optimization_method: str = 'max_sharpe'  # 'max_sharpe', 'min_variance', 'risk_parity'
    ):
        """
        Initialize the integration layer.
        
        Args:
            historical_data_days: Number of days of historical data to use
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
            optimization_method: Portfolio optimization method to use
        """
        self.historical_data_days = historical_data_days
        self.risk_free_rate = risk_free_rate
        self.optimization_method = optimization_method
        
        # Initialize optimal frontier calculator
        self.frontier_calculator = OptimalFrontier(
            risk_free_rate=risk_free_rate,
            max_position_size=0.30,  # 30% max per position for frontier
            min_position_size=0.0,
            allow_short=False
        )
        
        logger.info(f"Frontier Integration initialized: method={optimization_method}, "
                   f"data_days={historical_data_days}")
    
    def calculate_optimal_weights(
        self,
        trailhead_signals: List[TrailheadSignal],
        market_data: Optional[pd.DataFrame] = None,
        target_return: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights using efficient frontier optimization.
        
        Args:
            trailhead_signals: List of validated trailhead signals
            market_data: Optional historical price data (if None, uses CSV files)
            target_return: Optional target return level (for frontier portfolios)
            
        Returns:
            Dictionary mapping ticker -> optimal weight
        """
        try:
            logger.info(f"Calculating optimal weights for {len(trailhead_signals)} signals")
            
            # Extract tickers from signals
            tickers = [signal.ticker for signal in trailhead_signals]
            
            if not tickers:
                logger.warning("No tickers provided, returning empty weights")
                return {}
            
            # Load or generate returns data
            if market_data is not None:
                returns_data = self._calculate_returns_from_prices(market_data, tickers)
            else:
                # Try to load from CSV files
                try:
                    returns_data = pd.read_csv('market_returns.csv', index_col=0, parse_dates=True)
                    returns_data = returns_data[tickers]  # Filter to relevant tickers
                except Exception as e:
                    logger.warning(f"Could not load market data: {e}")
                    # Fallback: use equal weights
                    return self._equal_weights_fallback(tickers)
            
            # Load returns data into frontier calculator
            self.frontier_calculator.load_data(returns_data, tickers)
            
            # Calculate optimal portfolio based on method
            if self.optimization_method == 'max_sharpe':
                optimal_portfolio = self.frontier_calculator.find_maximum_sharpe_portfolio()
            elif self.optimization_method == 'min_variance':
                optimal_portfolio = self.frontier_calculator.find_minimum_variance_portfolio()
            elif self.optimization_method == 'risk_parity':
                optimal_portfolio = self.frontier_calculator.find_risk_parity_portfolio()
            elif self.optimization_method == 'target_return' and target_return is not None:
                optimal_portfolio = self.frontier_calculator.find_target_return_portfolio(target_return)
            else:
                logger.warning(f"Unknown optimization method: {self.optimization_method}")
                return self._equal_weights_fallback(tickers)
            
            # Convert to dictionary
            optimal_weights = dict(zip(optimal_portfolio.tickers, optimal_portfolio.weights))
            
            logger.info(f"Optimal weights calculated: {optimal_portfolio.expected_return:.2%} return, "
                       f"{optimal_portfolio.volatility:.2%} vol, "
                       f"{optimal_portfolio.sharpe_ratio:.2f} Sharpe")
            
            return optimal_weights
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal weights: {e}")
            return self._equal_weights_fallback(tickers)
    
    def enhance_portfolio_positions(
        self,
        trailhead_signals: List[TrailheadSignal],
        baseline_positions: List[PortfolioPosition],
        market_data: Optional[pd.DataFrame] = None,
        blend_factor: float = 0.5  # 0=pure baseline, 1=pure frontier
    ) -> List[PortfolioPosition]:
        """
        Enhance existing portfolio positions by blending with frontier optimization.
        
        This method takes the baseline positions from Portfolio_Synthesizer
        and blends them with optimal frontier weights for improved risk-return.
        
        Args:
            trailhead_signals: Trailhead signals used for baseline
            baseline_positions: Positions from Portfolio_Synthesizer
            market_data: Optional historical price data
            blend_factor: How much to blend frontier (0-1)
            
        Returns:
            Enhanced list of PortfolioPosition objects
        """
        try:
            logger.info(f"Enhancing {len(baseline_positions)} positions with frontier optimization")
            
            if not baseline_positions:
                return baseline_positions
            
            # Calculate optimal weights
            optimal_weights = self.calculate_optimal_weights(trailhead_signals, market_data)
            
            if not optimal_weights:
                logger.warning("No optimal weights calculated, returning baseline positions")
                return baseline_positions
            
            # Create weight adjustment mapping
            enhanced_positions = []
            
            for position in baseline_positions:
                ticker = position.ticker
                baseline_weight = position.weight
                
                # Get optimal weight (default to baseline if not in optimal)
                optimal_weight = optimal_weights.get(ticker, baseline_weight)
                
                # Blend weights
                blended_weight = (1 - blend_factor) * baseline_weight + blend_factor * optimal_weight
                
                # Create enhanced position with blended weight
                enhanced_position = PortfolioPosition(
                    ticker=position.ticker,
                    weight=blended_weight,
                    position_size=position.position_size,  # Will be recalculated
                    entry_price=position.entry_price,
                    target_price=position.target_price,
                    stop_loss=position.stop_loss,
                    chemistry_type=position.chemistry_type,
                    confidence=position.confidence,
                    metadata={
                        **position.metadata,
                        'frontier_optimization': {
                            'baseline_weight': baseline_weight,
                            'optimal_weight': optimal_weight,
                            'blended_weight': blended_weight,
                            'blend_factor': blend_factor,
                            'optimization_method': self.optimization_method,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                )
                
                enhanced_positions.append(enhanced_position)
            
            # Normalize weights to sum to 1
            total_weight = sum(p.weight for p in enhanced_positions)
            if total_weight > 0:
                for position in enhanced_positions:
                    position.weight = position.weight / total_weight
            
            logger.info(f"Enhanced portfolio: {len(enhanced_positions)} positions with frontier optimization")
            
            return enhanced_positions
            
        except Exception as e:
            logger.error(f"Failed to enhance positions: {e}")
            return baseline_positions
    
    def get_frontier_recommendation(
        self,
        trailhead_signals: List[TrailheadSignal],
        market_data: Optional[pd.DataFrame] = None,
        risk_tolerance: float = 0.5  # 0=min variance, 1=max return
    ) -> Dict:
        """
        Get frontier-based portfolio recommendation.
        
        Args:
            trailhead_signals: List of trailhead signals
            market_data: Optional historical price data
            risk_tolerance: Risk tolerance level (0-1)
            
        Returns:
            Dictionary with recommendation details
        """
        try:
            tickers = [signal.ticker for signal in trailhead_signals]
            
            # Calculate optimal weights
            optimal_weights = self.calculate_optimal_weights(trailhead_signals, market_data)
            
            if not optimal_weights:
                return {
                    'success': False,
                    'message': 'Could not calculate optimal weights',
                    'weights': {}
                }
            
            # Get all key portfolios
            min_var = self.frontier_calculator.find_minimum_variance_portfolio()
            max_sharpe = self.frontier_calculator.find_maximum_sharpe_portfolio()
            
            # Generate efficient frontier for visualization
            frontier_portfolios = self.frontier_calculator.generate_efficient_frontier(n_portfolios=10)
            
            return {
                'success': True,
                'optimal_weights': optimal_weights,
                'min_variance_portfolio': min_var.to_dict(),
                'max_sharpe_portfolio': max_sharpe.to_dict(),
                'frontier_portfolios': [p.to_dict() for p in frontier_portfolios],
                'recommendation': {
                    'method': self.optimization_method,
                    'expected_return': max_sharpe.expected_return,
                    'volatility': max_sharpe.volatility,
                    'sharpe_ratio': max_sharpe.sharpe_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get frontier recommendation: {e}")
            return {
                'success': False,
                'message': str(e),
                'weights': {}
            }
    
    def _calculate_returns_from_prices(
        self,
        price_data: pd.DataFrame,
        tickers: List[str]
    ) -> pd.DataFrame:
        """Calculate returns from price data."""
        # Filter to relevant tickers
        prices = price_data[tickers]
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Limit to specified number of days
        if len(returns) > self.historical_data_days:
            returns = returns.tail(self.historical_data_days)
        
        return returns
    
    def _equal_weights_fallback(self, tickers: List[str]) -> Dict[str, float]:
        """Fallback to equal weights if optimization fails."""
        logger.warning("Using equal weights fallback")
        n = len(tickers)
        if n == 0:
            return {}
        return {ticker: 1.0 / n for ticker in tickers}


def demonstrate_integration():
    """Demonstrate the frontier integration with sample data."""
    
    print("=" * 80)
    print("🔗 FRONTIER INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create mock trailhead signals
    from Trailhead_Detector import TrailheadSignal
    
    # Create mock trailhead signals using dataclass constructor
    mock_signals = [
        TrailheadSignal(
            ticker='AAPL',
            pressure_score=0.75,
            fragility_score=0.65,
            composite_score=0.80,
            trigger_type='breakout',
            metadata={'timestamp': datetime.now().isoformat()}
        ),
        TrailheadSignal(
            ticker='MSFT',
            pressure_score=0.70,
            fragility_score=0.60,
            composite_score=0.75,
            trigger_type='squeeze',
            metadata={'timestamp': datetime.now().isoformat()}
        ),
        TrailheadSignal(
            ticker='AMZN',
            pressure_score=0.80,
            fragility_score=0.70,
            composite_score=0.85,
            trigger_type='cascade',
            metadata={'timestamp': datetime.now().isoformat()}
        ),
    ]
    
    # Initialize integration
    integration = FrontierIntegration(
        optimization_method='max_sharpe',
        risk_free_rate=0.02
    )
    
    # Get recommendation
    recommendation = integration.get_frontier_recommendation(mock_signals)
    
    if recommendation['success']:
        print("\n✅ Frontier Recommendation:")
        print(f"   Method: {recommendation['recommendation']['method']}")
        print(f"   Expected Return: {recommendation['recommendation']['expected_return']:.2%}")
        print(f"   Volatility: {recommendation['recommendation']['volatility']:.2%}")
        print(f"   Sharpe Ratio: {recommendation['recommendation']['sharpe_ratio']:.2f}")
        
        print("\n📊 Optimal Weights:")
        for ticker, weight in sorted(recommendation['optimal_weights'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"   {ticker}: {weight:.2%}")
    
    print("\n" + "=" * 80)
    print("✅ INTEGRATION DEMONSTRATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_integration()
