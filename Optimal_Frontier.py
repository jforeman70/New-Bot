"""
Optimal_Frontier.py
🏔️ GEOLOGICAL PORTFOLIO OPTIMIZATION: Finding the Efficient Frontier

This module implements Modern Portfolio Theory (MPT) and advanced optimization
techniques to find the optimal portfolio allocation along the efficient frontier.

The efficient frontier represents the set of optimal portfolios that offer the
highest expected return for a defined level of risk, or equivalently, the lowest
risk for a given level of expected return.

Key Features:
- Calculate efficient frontier using Markowitz optimization
- Find maximum Sharpe ratio portfolio (tangency portfolio)
- Find minimum variance portfolio
- Calculate risk parity allocation
- Support for constraints (position limits, sector concentration)
- Integration with existing Catalyst Framework geology
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    """Portfolio risk-return metrics."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    tickers: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'weights': dict(zip(self.tickers, self.weights)),
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio
        }


class OptimalFrontier:
    """
    🏔️ CUTTING-EDGE: Geological Portfolio Optimization System
    
    This class finds the efficient frontier - the optimal set of portfolios
    that maximize return for given risk levels. Integrates with the Catalyst
    Framework's geological metaphor where portfolios are terrain formations.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 2% risk-free rate
        max_position_size: float = 0.25,  # 25% max per position
        min_position_size: float = 0.0,  # 0% min (can be 0)
        allow_short: bool = False  # No shorting by default
    ):
        """
        Initialize the Optimal Frontier calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            max_position_size: Maximum weight per asset (0-1)
            min_position_size: Minimum weight per asset (0-1)
            allow_short: Whether to allow short positions (negative weights)
        """
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.allow_short = allow_short
        
        # Cached data
        self.returns_data = None
        self.mean_returns = None
        self.cov_matrix = None
        self.tickers = None
        
    def load_data_from_csv(self, returns_csv: str) -> None:
        """
        Load historical returns data from CSV file.
        
        Args:
            returns_csv: Path to CSV file with returns data
        """
        try:
            logger.info(f"Loading returns data from {returns_csv}")
            self.returns_data = pd.read_csv(returns_csv, index_col=0, parse_dates=True)
            self.tickers = list(self.returns_data.columns)
            
            # Calculate mean returns and covariance matrix
            self.mean_returns = self.returns_data.mean().values  # Daily returns
            self.cov_matrix = self.returns_data.cov().values  # Daily covariance
            
            logger.info(f"Loaded {len(self.tickers)} assets: {self.tickers}")
            logger.info(f"Mean daily returns: {self.mean_returns * 252}")  # Annualized
            logger.info(f"Data shape: {self.returns_data.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load data from {returns_csv}: {e}")
            raise
    
    def load_data(
        self, 
        returns: pd.DataFrame,
        tickers: Optional[List[str]] = None
    ) -> None:
        """
        Load returns data directly from DataFrame.
        
        Args:
            returns: DataFrame with returns data (rows=dates, cols=tickers)
            tickers: Optional list of ticker names (uses DataFrame columns if None)
        """
        try:
            self.returns_data = returns
            self.tickers = tickers if tickers else list(returns.columns)
            
            # Calculate statistics
            self.mean_returns = returns.mean().values
            self.cov_matrix = returns.cov().values
            
            logger.info(f"Loaded {len(self.tickers)} assets: {self.tickers}")
            
        except Exception as e:
            logger.error(f"Failed to load returns data: {e}")
            raise
    
    def calculate_portfolio_metrics(
        self,
        weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        
        Args:
            weights: Asset weights (must sum to 1)
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        # Annualize daily returns and volatility
        expected_return = np.dot(weights, self.mean_returns) * 252
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights)) * 252
        volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        return expected_return, volatility, sharpe_ratio
    
    def find_minimum_variance_portfolio(self) -> PortfolioMetrics:
        """
        Find the minimum variance portfolio (lowest risk portfolio).
        
        This is the leftmost point on the efficient frontier.
        
        Returns:
            PortfolioMetrics for the minimum variance portfolio
        """
        logger.info("Finding minimum variance portfolio...")
        
        n_assets = len(self.tickers)
        
        # Objective: minimize portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights)) * 252
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds for each weight
        if self.allow_short:
            bounds = [(-1, 1) for _ in range(n_assets)]
        else:
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Minimum variance optimization did not converge: {result.message}")
        
        # Calculate metrics
        weights = result.x
        expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(weights)
        
        logger.info(f"Minimum variance portfolio: Return={expected_return:.2%}, Vol={volatility:.2%}")
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            tickers=self.tickers
        )
    
    def find_maximum_sharpe_portfolio(self) -> PortfolioMetrics:
        """
        Find the maximum Sharpe ratio portfolio (tangency portfolio).
        
        This portfolio has the best risk-adjusted returns and represents
        the optimal portfolio for a risk-averse investor.
        
        Returns:
            PortfolioMetrics for the maximum Sharpe ratio portfolio
        """
        logger.info("Finding maximum Sharpe ratio portfolio (tangency portfolio)...")
        
        n_assets = len(self.tickers)
        
        # Objective: minimize negative Sharpe ratio (= maximize Sharpe ratio)
        def negative_sharpe(weights):
            expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(weights)
            return -sharpe_ratio  # Minimize negative = maximize positive
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        if self.allow_short:
            bounds = [(-1, 1) for _ in range(n_assets)]
        else:
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Maximum Sharpe optimization did not converge: {result.message}")
        
        # Calculate metrics
        weights = result.x
        expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(weights)
        
        logger.info(f"Maximum Sharpe portfolio: Return={expected_return:.2%}, Vol={volatility:.2%}, Sharpe={sharpe_ratio:.2f}")
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            tickers=self.tickers
        )
    
    def find_target_return_portfolio(self, target_return: float) -> PortfolioMetrics:
        """
        Find the minimum variance portfolio for a target return level.
        
        Args:
            target_return: Target annual return (e.g., 0.15 for 15%)
            
        Returns:
            PortfolioMetrics for the efficient portfolio at target return
        """
        logger.info(f"Finding efficient portfolio for target return: {target_return:.2%}")
        
        n_assets = len(self.tickers)
        
        # Objective: minimize variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights)) * 252
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, self.mean_returns) * 252 - target_return}  # Target return
        ]
        
        # Bounds
        if self.allow_short:
            bounds = [(-1, 1) for _ in range(n_assets)]
        else:
            bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Target return optimization did not converge: {result.message}")
            # If it fails, return max Sharpe portfolio instead
            logger.info("Falling back to maximum Sharpe portfolio")
            return self.find_maximum_sharpe_portfolio()
        
        # Calculate metrics
        weights = result.x
        expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(weights)
        
        logger.info(f"Efficient portfolio: Return={expected_return:.2%}, Vol={volatility:.2%}")
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            tickers=self.tickers
        )
    
    def generate_efficient_frontier(
        self,
        n_portfolios: int = 50
    ) -> List[PortfolioMetrics]:
        """
        Generate the efficient frontier by calculating optimal portfolios
        across a range of return levels.
        
        Args:
            n_portfolios: Number of portfolios to generate along the frontier
            
        Returns:
            List of PortfolioMetrics representing the efficient frontier
        """
        logger.info(f"Generating efficient frontier with {n_portfolios} portfolios...")
        
        # Find min variance and max Sharpe portfolios to determine range
        min_var_portfolio = self.find_minimum_variance_portfolio()
        max_sharpe_portfolio = self.find_maximum_sharpe_portfolio()
        
        # Determine return range
        min_return = min(min_var_portfolio.expected_return, max_sharpe_portfolio.expected_return)
        max_return = max(self.mean_returns) * 252 * 1.2  # 120% of best single asset
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                portfolio = self.find_target_return_portfolio(target_return)
                efficient_portfolios.append(portfolio)
            except Exception as e:
                logger.warning(f"Failed to find portfolio for return {target_return:.2%}: {e}")
                continue
        
        logger.info(f"Generated {len(efficient_portfolios)} efficient portfolios")
        
        return efficient_portfolios
    
    def find_risk_parity_portfolio(self) -> PortfolioMetrics:
        """
        Find the risk parity portfolio where each asset contributes equally
        to portfolio risk.
        
        This is an alternative allocation strategy that focuses on risk
        diversification rather than return optimization.
        
        Returns:
            PortfolioMetrics for the risk parity portfolio
        """
        logger.info("Finding risk parity portfolio...")
        
        n_assets = len(self.tickers)
        
        # Objective: minimize difference in risk contributions
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # We want all risk contributions to be equal (= portfolio_vol / n_assets)
            target_risk = portfolio_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds (risk parity doesn't work well with shorts)
        bounds = [(0.01, self.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization did not converge: {result.message}")
        
        # Calculate metrics
        weights = result.x
        expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(weights)
        
        logger.info(f"Risk parity portfolio: Return={expected_return:.2%}, Vol={volatility:.2%}")
        
        return PortfolioMetrics(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            tickers=self.tickers
        )
    
    def save_frontier_to_csv(
        self,
        efficient_portfolios: List[PortfolioMetrics],
        filename: str = 'efficient_frontier.csv'
    ) -> None:
        """
        Save efficient frontier portfolios to CSV file.
        
        Args:
            efficient_portfolios: List of PortfolioMetrics from generate_efficient_frontier
            filename: Output CSV filename
        """
        try:
            # Create DataFrame with results
            data = []
            for i, portfolio in enumerate(efficient_portfolios):
                row = {
                    'Portfolio': i + 1,
                    'Expected_Return': portfolio.expected_return,
                    'Volatility': portfolio.volatility,
                    'Sharpe_Ratio': portfolio.sharpe_ratio
                }
                # Add weights for each ticker
                for ticker, weight in zip(portfolio.tickers, portfolio.weights):
                    row[f'Weight_{ticker}'] = weight
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logger.info(f"Saved efficient frontier to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save frontier to CSV: {e}")
            raise


def demo_optimal_frontier():
    """Demonstration of the Optimal Frontier calculator with sample data."""
    
    print("=" * 80)
    print("🏔️ OPTIMAL FRONTIER CALCULATOR - Demonstration")
    print("=" * 80)
    
    # Initialize calculator
    frontier = OptimalFrontier(
        risk_free_rate=0.02,
        max_position_size=0.30,
        min_position_size=0.0,
        allow_short=False
    )
    
    # Load sample data
    print("\n📊 Loading market data from CSV...")
    frontier.load_data_from_csv('market_returns.csv')
    
    # Find key portfolios
    print("\n🎯 Finding Key Portfolios:")
    print("-" * 80)
    
    min_var = frontier.find_minimum_variance_portfolio()
    print(f"\n✅ Minimum Variance Portfolio:")
    print(f"   Expected Return: {min_var.expected_return:.2%}")
    print(f"   Volatility: {min_var.volatility:.2%}")
    print(f"   Sharpe Ratio: {min_var.sharpe_ratio:.2f}")
    print(f"   Top 3 Holdings: {sorted(zip(min_var.tickers, min_var.weights), key=lambda x: x[1], reverse=True)[:3]}")
    
    max_sharpe = frontier.find_maximum_sharpe_portfolio()
    print(f"\n✅ Maximum Sharpe Ratio Portfolio (Optimal):")
    print(f"   Expected Return: {max_sharpe.expected_return:.2%}")
    print(f"   Volatility: {max_sharpe.volatility:.2%}")
    print(f"   Sharpe Ratio: {max_sharpe.sharpe_ratio:.2f}")
    print(f"   Top 3 Holdings:")
    for ticker, weight in sorted(zip(max_sharpe.tickers, max_sharpe.weights), key=lambda x: x[1], reverse=True)[:3]:
        print(f"      {ticker}: {weight:.2%}")
    
    risk_parity = frontier.find_risk_parity_portfolio()
    print(f"\n✅ Risk Parity Portfolio:")
    print(f"   Expected Return: {risk_parity.expected_return:.2%}")
    print(f"   Volatility: {risk_parity.volatility:.2%}")
    print(f"   Sharpe Ratio: {risk_parity.sharpe_ratio:.2f}")
    
    # Generate efficient frontier
    print("\n📈 Generating Efficient Frontier...")
    efficient_portfolios = frontier.generate_efficient_frontier(n_portfolios=20)
    
    # Save to CSV
    frontier.save_frontier_to_csv(efficient_portfolios, 'efficient_frontier.csv')
    
    print(f"\n✅ Generated {len(efficient_portfolios)} efficient portfolios")
    print("\n📁 Results saved to:")
    print("   - efficient_frontier.csv")
    
    print("\n" + "=" * 80)
    print("✅ OPTIMAL FRONTIER CALCULATION COMPLETE!")
    print("=" * 80)
    
    return frontier, max_sharpe


if __name__ == "__main__":
    # Run demonstration
    frontier_calculator, optimal_portfolio = demo_optimal_frontier()
