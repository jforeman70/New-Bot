"""
Visualize_Frontier.py
📊 Efficient Frontier Visualization Tool

Creates visualizations of the efficient frontier and portfolio allocations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Use non-interactive backend for server environments
# Change to 'TkAgg' or remove for interactive use (Jupyter, desktop GUI)
matplotlib.use('Agg')
from pathlib import Path

def visualize_efficient_frontier(
    frontier_csv: str = 'efficient_frontier.csv',
    returns_csv: str = 'market_returns.csv',
    output_file: str = 'efficient_frontier_plot.png'
):
    """
    Create visualization of the efficient frontier.
    
    Args:
        frontier_csv: Path to efficient frontier CSV file
        returns_csv: Path to market returns CSV file
        output_file: Output image filename
    """
    print("=" * 80)
    print("📊 EFFICIENT FRONTIER VISUALIZATION")
    print("=" * 80)
    
    # Load data
    print("\n📂 Loading data...")
    frontier_df = pd.read_csv(frontier_csv)
    returns_df = pd.read_csv(returns_csv, index_col=0, parse_dates=True)
    
    # Calculate individual asset metrics
    tickers = list(returns_df.columns)
    asset_returns = returns_df.mean() * 252  # Annualized
    asset_volatilities = returns_df.std() * np.sqrt(252)  # Annualized
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Efficient Frontier
    ax1.scatter(
        frontier_df['Volatility'],
        frontier_df['Expected_Return'],
        c=frontier_df['Sharpe_Ratio'],
        cmap='viridis',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5,
        label='Efficient Frontier'
    )
    
    # Add individual assets
    ax1.scatter(
        asset_volatilities,
        asset_returns,
        color='red',
        s=80,
        alpha=0.7,
        marker='D',
        edgecolors='darkred',
        linewidth=1,
        label='Individual Assets'
    )
    
    # Label individual assets
    for i, ticker in enumerate(tickers):
        ax1.annotate(
            ticker,
            (asset_volatilities[i], asset_returns[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    # Highlight key portfolios
    min_vol_idx = frontier_df['Volatility'].idxmin()
    max_sharpe_idx = frontier_df['Sharpe_Ratio'].idxmax()
    
    ax1.scatter(
        frontier_df.loc[min_vol_idx, 'Volatility'],
        frontier_df.loc[min_vol_idx, 'Expected_Return'],
        color='green',
        s=300,
        marker='*',
        edgecolors='darkgreen',
        linewidth=2,
        label=f'Min Variance (Sharpe: {frontier_df.loc[min_vol_idx, "Sharpe_Ratio"]:.2f})',
        zorder=5
    )
    
    ax1.scatter(
        frontier_df.loc[max_sharpe_idx, 'Volatility'],
        frontier_df.loc[max_sharpe_idx, 'Expected_Return'],
        color='gold',
        s=300,
        marker='*',
        edgecolors='orange',
        linewidth=2,
        label=f'Max Sharpe (Sharpe: {frontier_df.loc[max_sharpe_idx, "Sharpe_Ratio"]:.2f})',
        zorder=5
    )
    
    # Add colorbar
    colorbar = plt.colorbar(ax1.collections[0], ax=ax1)
    colorbar.set_label('Sharpe Ratio', rotation=270, labelpad=15)
    
    ax1.set_xlabel('Volatility (Risk)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Expected Return', fontsize=12, fontweight='bold')
    ax1.set_title('🏔️ Efficient Frontier\nOptimal Risk-Return Portfolios', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Format axes as percentages
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Plot 2: Portfolio Allocations for Max Sharpe Portfolio
    weight_cols = [col for col in frontier_df.columns if col.startswith('Weight_')]
    max_sharpe_weights = frontier_df.loc[max_sharpe_idx, weight_cols].values
    tickers_from_weights = [col.replace('Weight_', '') for col in weight_cols]
    
    # Filter out near-zero weights for cleaner visualization
    significant_weights = [(t, w) for t, w in zip(tickers_from_weights, max_sharpe_weights) if w > 0.01]
    significant_weights.sort(key=lambda x: x[1], reverse=True)
    
    tickers_sig = [t for t, w in significant_weights]
    weights_sig = [w for t, w in significant_weights]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(tickers_sig)))
    bars = ax2.bar(range(len(tickers_sig)), weights_sig, color=colors, edgecolor='black', linewidth=1.5)
    
    ax2.set_xticks(range(len(tickers_sig)))
    ax2.set_xticklabels(tickers_sig, rotation=45, ha='right')
    ax2.set_ylabel('Portfolio Weight', fontsize=12, fontweight='bold')
    ax2.set_title(f'🎯 Maximum Sharpe Ratio Portfolio\nReturn: {frontier_df.loc[max_sharpe_idx, "Expected_Return"]:.1%} | ' +
                  f'Vol: {frontier_df.loc[max_sharpe_idx, "Volatility"]:.1%} | ' +
                  f'Sharpe: {frontier_df.loc[max_sharpe_idx, "Sharpe_Ratio"]:.2f}',
                  fontsize=14, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, weight in zip(bars, weights_sig):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.1%}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")
    
    # Print summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print("-" * 80)
    print(f"Number of Portfolios on Frontier: {len(frontier_df)}")
    print(f"\n🎯 Maximum Sharpe Ratio Portfolio:")
    print(f"   Expected Return: {frontier_df.loc[max_sharpe_idx, 'Expected_Return']:.2%}")
    print(f"   Volatility: {frontier_df.loc[max_sharpe_idx, 'Volatility']:.2%}")
    print(f"   Sharpe Ratio: {frontier_df.loc[max_sharpe_idx, 'Sharpe_Ratio']:.2f}")
    print(f"   Top 3 Holdings:")
    for i, (ticker, weight) in enumerate(significant_weights[:3], 1):
        print(f"      {i}. {ticker}: {weight:.2%}")
    
    print(f"\n🛡️ Minimum Variance Portfolio:")
    print(f"   Expected Return: {frontier_df.loc[min_vol_idx, 'Expected_Return']:.2%}")
    print(f"   Volatility: {frontier_df.loc[min_vol_idx, 'Volatility']:.2%}")
    print(f"   Sharpe Ratio: {frontier_df.loc[min_vol_idx, 'Sharpe_Ratio']:.2f}")
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    visualize_efficient_frontier()
