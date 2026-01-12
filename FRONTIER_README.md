# Optimal Frontier Calculator

## 🏔️ Finding the Efficient Frontier for Portfolio Optimization

This module implements Modern Portfolio Theory (MPT) to find the **efficient frontier** - the set of optimal portfolios that offer the highest expected return for a given level of risk.

## Overview

The efficient frontier represents the Holy Grail of portfolio optimization:
- **Maximize returns** for a given risk level
- **Minimize risk** for a given return target
- Find the **optimal risk-adjusted portfolio** (maximum Sharpe ratio)

## Quick Start

### 1. Generate Sample Data (if you don't have your own)

```bash
python3 Optimal_Frontier.py
```

This creates:
- `market_prices.csv` - Historical price data
- `market_returns.csv` - Daily returns
- `market_summary.csv` - Summary statistics
- `efficient_frontier.csv` - Optimal portfolios along the frontier

### 2. Visualize the Results

```bash
python3 Visualize_Frontier.py
```

Creates `efficient_frontier_plot.png` showing:
- The efficient frontier curve
- Individual asset positions
- Maximum Sharpe ratio portfolio (⭐ gold star)
- Minimum variance portfolio (⭐ green star)
- Portfolio allocation breakdown

### 3. Integrate with Existing System

```python
from Frontier_Integration import FrontierIntegration
from Trailhead_Detector import TrailheadSignal

# Initialize integration
integration = FrontierIntegration(
    optimization_method='max_sharpe',  # or 'min_variance', 'risk_parity'
    risk_free_rate=0.02
)

# Get optimal weights for your signals
optimal_weights = integration.calculate_optimal_weights(trailhead_signals)

# Get full recommendation with frontier data
recommendation = integration.get_frontier_recommendation(trailhead_signals)
```

## Key Features

### 1. Multiple Optimization Methods

**Maximum Sharpe Ratio (Recommended)**
- Best risk-adjusted returns
- Optimal for most investors
- Default method

```python
frontier = OptimalFrontier(risk_free_rate=0.02)
frontier.load_data_from_csv('market_returns.csv')
optimal = frontier.find_maximum_sharpe_portfolio()
```

**Minimum Variance**
- Lowest possible risk
- Conservative approach
- Good for risk-averse investors

```python
min_var = frontier.find_minimum_variance_portfolio()
```

**Risk Parity**
- Equal risk contribution from each asset
- Diversification-focused
- Alternative to traditional MPT

```python
risk_parity = frontier.find_risk_parity_portfolio()
```

**Target Return**
- Find optimal portfolio for specific return goal
- Useful for constrained objectives

```python
target_portfolio = frontier.find_target_return_portfolio(target_return=0.15)  # 15%
```

### 2. Efficient Frontier Generation

Generate multiple portfolios along the efficient frontier:

```python
frontier_portfolios = frontier.generate_efficient_frontier(n_portfolios=50)
frontier.save_frontier_to_csv(frontier_portfolios, 'my_frontier.csv')
```

### 3. Integration with Catalyst Framework

The system integrates seamlessly with the existing Catalyst Framework:

```python
# Enhance Portfolio_Synthesizer positions with frontier optimization
enhanced_positions = integration.enhance_portfolio_positions(
    trailhead_signals=signals,
    baseline_positions=synthesizer_positions,
    blend_factor=0.5  # 50% baseline, 50% frontier
)
```

## Sample Results

Based on the demo data:

### Maximum Sharpe Portfolio (Optimal)
- **Expected Return:** 53.55%
- **Volatility:** 10.95%
- **Sharpe Ratio:** 4.71
- **Top Holdings:**
  - AMZN: 25.10%
  - JPM: 24.92%
  - MSFT: 14.15%

### Minimum Variance Portfolio (Safest)
- **Expected Return:** 20.26%
- **Volatility:** 7.02%
- **Sharpe Ratio:** 2.60
- **Top Holdings:**
  - JNJ: 20.62%
  - PG: 19.09%
  - WMT: 13.36%

### Risk Parity Portfolio (Balanced)
- **Expected Return:** 30.21%
- **Volatility:** 8.50%
- **Sharpe Ratio:** 3.32

## Advanced Usage

### Custom Constraints

```python
frontier = OptimalFrontier(
    risk_free_rate=0.02,
    max_position_size=0.20,  # 20% max per position
    min_position_size=0.05,  # 5% minimum
    allow_short=False  # No shorting
)
```

### Using Your Own Data

```python
import pandas as pd

# Load your own returns data
returns = pd.read_csv('my_returns.csv', index_col=0, parse_dates=True)

frontier = OptimalFrontier()
frontier.load_data(returns)

# Calculate optimal portfolio
optimal = frontier.find_maximum_sharpe_portfolio()
print(f"Optimal weights: {dict(zip(optimal.tickers, optimal.weights))}")
```

### Custom Risk-Free Rate

Adjust for current market conditions:

```python
# Example: Current 10-year Treasury yield
frontier = OptimalFrontier(risk_free_rate=0.045)  # 4.5%
```

## CSV File Formats

### market_prices.csv
```csv
Date,AAPL,MSFT,GOOGL,...
2025-01-01,150.00,300.00,140.00,...
2025-01-02,151.26,304.03,136.61,...
```

### market_returns.csv
```csv
Date,AAPL,MSFT,GOOGL,...
2025-01-02,0.0084,0.0134,-0.0242,...
2025-01-03,-0.0016,0.0301,0.0104,...
```

### efficient_frontier.csv
```csv
Portfolio,Expected_Return,Volatility,Sharpe_Ratio,Weight_AAPL,Weight_MSFT,...
1,0.2026,0.0702,2.6027,0.0742,0.1067,...
2,0.2582,0.0716,3.3292,0.0709,0.1141,...
```

## Integration with Portfolio_Synthesizer

The frontier optimization enhances the existing geological portfolio synthesis:

1. **Baseline Portfolio:** Portfolio_Synthesizer creates positions based on trailhead signals and chemistry
2. **Frontier Optimization:** Optimal_Frontier calculates MPT-optimal weights
3. **Blended Result:** Combine both approaches for best of both worlds

```python
# In Portfolio_Synthesizer workflow:
baseline_positions = synthesizer.synthesize_portfolio(...)

# Apply frontier optimization
integration = FrontierIntegration(optimization_method='max_sharpe')
optimized_positions = integration.enhance_portfolio_positions(
    trailhead_signals=signals,
    baseline_positions=baseline_positions,
    blend_factor=0.7  # 70% frontier, 30% baseline
)
```

## Theoretical Foundation

### Modern Portfolio Theory (MPT)

The efficient frontier is calculated using **Markowitz Portfolio Optimization**:

```
Maximize: E[R_p] = w^T μ
Subject to: σ_p^2 = w^T Σ w ≤ σ_target^2
            w^T 1 = 1
            0 ≤ w_i ≤ max_weight
```

Where:
- `w` = portfolio weights
- `μ` = expected returns vector
- `Σ` = covariance matrix
- `σ_p` = portfolio volatility

### Sharpe Ratio

The maximum Sharpe ratio portfolio maximizes:

```
Sharpe = (E[R_p] - R_f) / σ_p
```

Where `R_f` is the risk-free rate.

## Performance Considerations

- Optimization uses **scipy.optimize.minimize** with SLSQP method
- Typical runtime: <1 second for 10 assets
- Scales well up to 100+ assets
- Covariance matrix calculation is the bottleneck for large universes

## Troubleshooting

### "Optimization did not converge"
- Try reducing the number of constraints
- Increase `maxiter` in the optimize call
- Check for singular covariance matrix (highly correlated assets)

### "No optimal weights calculated"
- Ensure returns data has sufficient history (recommended: 252+ days)
- Check for NaN or infinite values in returns
- Verify tickers match between signals and returns data

### Negative Sharpe Ratios
- May occur if expected returns are below risk-free rate
- Consider adjusting the risk-free rate
- May indicate poor asset selection

## Dependencies

```bash
pip install numpy pandas scipy matplotlib
```

## Files

- `Optimal_Frontier.py` - Core frontier calculation engine
- `Frontier_Integration.py` - Integration with Portfolio_Synthesizer
- `Visualize_Frontier.py` - Visualization tools
- `market_*.csv` - Sample data files
- `efficient_frontier.csv` - Output frontier portfolios

## Next Steps

1. **Backtest** the optimal portfolios against historical data
2. **Monte Carlo** simulation for portfolio stress testing
3. **Dynamic rebalancing** based on changing market conditions
4. **Multi-period optimization** for longer horizons
5. **Factor models** (Fama-French) for enhanced returns estimation

## References

- Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*
- Sharpe, W. F. (1964). "Capital Asset Prices". *Journal of Finance*
- Merton, R. C. (1972). "An Analytic Derivation of the Efficient Portfolio Frontier"

---

**🏔️ The efficient frontier is where geological market analysis meets mathematical optimization!**
