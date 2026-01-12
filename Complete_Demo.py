#!/usr/bin/env python3
"""
Complete_Demo.py
🎯 Complete demonstration of the Optimal Frontier system

This script demonstrates the entire workflow:
1. Load sample market data
2. Calculate efficient frontier
3. Find optimal portfolios
4. Visualize results
5. Show integration with Catalyst Framework
"""

import sys
import logging
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def demonstrate_complete_workflow():
    """Run the complete demonstration."""
    
    print_header("🏔️ OPTIMAL FRONTIER - COMPLETE DEMONSTRATION")
    
    print("This demonstration shows:")
    print("  1. ✅ Load sample market data from CSV files")
    print("  2. ✅ Calculate the efficient frontier")
    print("  3. ✅ Find optimal portfolios (max Sharpe, min variance, risk parity)")
    print("  4. ✅ Visualize the results")
    print("  5. ✅ Integrate with the Catalyst Framework")
    print()
    
    try:
        # Step 1: Load and verify data
        print_header("STEP 1: Load Sample Market Data")
        
        import Optimal_Frontier
        
        print("📂 Loading market data...")
        returns_df = pd.read_csv('market_returns.csv', index_col=0, parse_dates=True)
        prices_df = pd.read_csv('market_prices.csv', index_col=0, parse_dates=True)
        
        print(f"✅ Loaded {len(returns_df)} days of returns for {len(returns_df.columns)} assets")
        print(f"   Assets: {', '.join(returns_df.columns.tolist())}")
        print(f"   Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
        
        # Step 2: Calculate efficient frontier
        print_header("STEP 2: Calculate Efficient Frontier")
        
        frontier = Optimal_Frontier.OptimalFrontier(
            risk_free_rate=0.02,
            max_position_size=0.30,
            min_position_size=0.0,
            allow_short=False
        )
        
        frontier.load_data(returns_df)
        print("✅ Frontier calculator initialized")
        
        # Step 3: Find key portfolios
        print_header("STEP 3: Find Optimal Portfolios")
        
        print("🔍 Finding minimum variance portfolio...")
        min_var = frontier.find_minimum_variance_portfolio()
        print(f"✅ Minimum Variance Portfolio:")
        print(f"   Expected Return: {min_var.expected_return:.2%}")
        print(f"   Volatility: {min_var.volatility:.2%}")
        print(f"   Sharpe Ratio: {min_var.sharpe_ratio:.2f}")
        
        print("\n🔍 Finding maximum Sharpe ratio portfolio...")
        max_sharpe = frontier.find_maximum_sharpe_portfolio()
        print(f"✅ Maximum Sharpe Portfolio (OPTIMAL):")
        print(f"   Expected Return: {max_sharpe.expected_return:.2%}")
        print(f"   Volatility: {max_sharpe.volatility:.2%}")
        print(f"   Sharpe Ratio: {max_sharpe.sharpe_ratio:.2f}")
        print(f"   Top 3 Holdings:")
        weights_sorted = sorted(zip(max_sharpe.tickers, max_sharpe.weights), 
                               key=lambda x: x[1], reverse=True)
        for i, (ticker, weight) in enumerate(weights_sorted[:3], 1):
            print(f"      {i}. {ticker}: {weight:.2%}")
        
        print("\n🔍 Finding risk parity portfolio...")
        risk_parity = frontier.find_risk_parity_portfolio()
        print(f"✅ Risk Parity Portfolio:")
        print(f"   Expected Return: {risk_parity.expected_return:.2%}")
        print(f"   Volatility: {risk_parity.volatility:.2%}")
        print(f"   Sharpe Ratio: {risk_parity.sharpe_ratio:.2f}")
        
        # Step 4: Generate and save frontier
        print_header("STEP 4: Generate Complete Efficient Frontier")
        
        print("📈 Generating 20 portfolios along the efficient frontier...")
        efficient_portfolios = frontier.generate_efficient_frontier(n_portfolios=20)
        print(f"✅ Generated {len(efficient_portfolios)} efficient portfolios")
        
        frontier.save_frontier_to_csv(efficient_portfolios, 'efficient_frontier.csv')
        print("✅ Saved to: efficient_frontier.csv")
        
        # Step 5: Visualize
        print_header("STEP 5: Visualize Results")
        
        print("🎨 Creating visualization...")
        import Visualize_Frontier
        Visualize_Frontier.visualize_efficient_frontier(
            'efficient_frontier.csv',
            'market_returns.csv',
            'efficient_frontier_plot.png'
        )
        print("✅ Visualization saved to: efficient_frontier_plot.png")
        
        # Step 6: Integration demonstration
        print_header("STEP 6: Catalyst Framework Integration")
        
        from Frontier_Integration import FrontierIntegration
        from Trailhead_Detector import TrailheadSignal
        
        print("🔗 Creating integration layer...")
        integration = FrontierIntegration(
            optimization_method='max_sharpe',
            risk_free_rate=0.02
        )
        
        # Create sample trailhead signals
        print("📊 Creating sample trailhead signals...")
        sample_signals = [
            TrailheadSignal(
                ticker='AAPL',
                pressure_score=0.75,
                fragility_score=0.65,
                composite_score=0.80,
                trigger_type='breakout',
                metadata={}
            ),
            TrailheadSignal(
                ticker='MSFT',
                pressure_score=0.70,
                fragility_score=0.60,
                composite_score=0.75,
                trigger_type='squeeze',
                metadata={}
            ),
            TrailheadSignal(
                ticker='AMZN',
                pressure_score=0.80,
                fragility_score=0.70,
                composite_score=0.85,
                trigger_type='cascade',
                metadata={}
            ),
        ]
        
        print("🎯 Calculating optimal weights for signals...")
        optimal_weights = integration.calculate_optimal_weights(sample_signals)
        print(f"✅ Optimal Weights Calculated:")
        for ticker, weight in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {ticker}: {weight:.2%}")
        
        # Summary
        print_header("✅ DEMONSTRATION COMPLETE!")
        
        print("📊 SUMMARY OF RESULTS:")
        print()
        print("Generated Files:")
        print("  ✅ efficient_frontier.csv - Complete frontier data")
        print("  ✅ efficient_frontier_plot.png - Visual representation")
        print("  ✅ market_prices.csv - Sample price data")
        print("  ✅ market_returns.csv - Sample returns data")
        print("  ✅ market_summary.csv - Summary statistics")
        print()
        print("Key Portfolios Found:")
        print(f"  🏆 Maximum Sharpe: {max_sharpe.expected_return:.2%} return, {max_sharpe.volatility:.2%} vol, {max_sharpe.sharpe_ratio:.2f} Sharpe")
        print(f"  🛡️  Minimum Variance: {min_var.expected_return:.2%} return, {min_var.volatility:.2%} vol, {min_var.sharpe_ratio:.2f} Sharpe")
        print(f"  ⚖️  Risk Parity: {risk_parity.expected_return:.2%} return, {risk_parity.volatility:.2%} vol, {risk_parity.sharpe_ratio:.2f} Sharpe")
        print()
        print("Next Steps:")
        print("  1. Review FRONTIER_README.md for detailed documentation")
        print("  2. Integrate with your Portfolio_Synthesizer workflow")
        print("  3. Use your own market data by replacing CSV files")
        print("  4. Customize optimization parameters in Optimal_Frontier.py")
        print()
        print("🏔️ The efficient frontier is where geology meets mathematics!")
        print()
        
        return True
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure all required files are present:")
        print("  - market_returns.csv")
        print("  - market_prices.csv")
        print("\nYou can generate them by running:")
        print("  python3 Optimal_Frontier.py")
        return False


if __name__ == "__main__":
    success = demonstrate_complete_workflow()
    sys.exit(0 if success else 1)
