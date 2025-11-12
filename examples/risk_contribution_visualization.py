"""
Enhanced Risk Contribution Visualization with Buy/Sell Signals

This module provides an enhanced visualization for risk contribution analysis
that shows position changes and trading signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from portwine.analyzers import RiskContributionAnalyzer


def plot_risk_contribution_with_signals(results, title="Risk Contribution with Position Signals"):
    """
    Enhanced visualization showing:
    1. Risk contribution over time
    2. Position weights (signals) over time
    3. Buy/Sell signals based on position changes
    4. Risk contribution vs position weight correlation
    
    Parameters
    ----------
    results : dict
        Results dictionary from Backtester.run_backtest() containing:
        - 'signals_df': DataFrame of daily portfolio weights
        - 'tickers_returns': DataFrame of daily ticker returns
    title : str, optional
        Title for the overall figure (default: "Risk Contribution with Position Signals")
    
    Returns
    -------
    None
        Displays the visualization and prints actionable insights
    """
    signals_df = results.get('signals_df')
    tickers_returns = results.get('tickers_returns')
    
    if signals_df is None or tickers_returns is None:
        print("Error: signals_df or tickers_returns missing.")
        return
    
    # Calculate risk contributions
    risk_analyzer = RiskContributionAnalyzer(cov_window=60)
    risk_dict = risk_analyzer.analyze(results)
    
    if risk_dict is None:
        print("Could not calculate risk contributions.")
        return
    
    pct_risk_contrib = risk_dict['pct_risk_contrib']
    
    # Calculate position changes (buy/sell signals)
    position_changes = signals_df.diff()  # Positive = buy/increase, Negative = sell/decrease
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3, height_ratios=[2, 2, 2, 1.5])
    
    tickers = signals_df.columns.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
    color_map = {tkr: colors[i] for i, tkr in enumerate(tickers)}
    
    # 1. Risk Contribution Over Time (Stacked Area)
    ax1 = fig.add_subplot(gs[0, :])
    pct_risk_contrib.plot.area(ax=ax1, linewidth=0, alpha=0.7, color=[color_map[t] for t in tickers])
    ax1.set_title('Risk Contribution Over Time (Percentage)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Risk Contribution %', fontsize=11)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, ncol=min(4, len(tickers)))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Position Weights Over Time (Stacked Area)
    ax2 = fig.add_subplot(gs[1, :])
    signals_df.plot.area(ax=ax2, linewidth=0, alpha=0.7, color=[color_map[t] for t in tickers])
    ax2.set_title('Position Weights Over Time (Portfolio Allocation)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Weight (0-1)', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9, ncol=min(4, len(tickers)))
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Position Changes (Buy/Sell Signals) - Heatmap style with proper dates
    ax3 = fig.add_subplot(gs[2, :])
    
    # Create a matrix for position changes
    change_matrix = position_changes.fillna(0)
    
    # Use pcolormesh with matplotlib date handling
    dates = change_matrix.index
    # Convert dates to matplotlib date numbers using date2num
    date_nums = mdates.date2num(dates.to_pydatetime())
    
    # Create meshgrid for pcolormesh
    # Need one extra point for pcolormesh boundaries
    y_positions = np.arange(len(tickers) + 1)
    
    # Extend dates by one to create proper boundaries for pcolormesh
    if len(date_nums) > 1:
        # Calculate average spacing between dates
        date_diff = np.mean(np.diff(date_nums))
        date_nums_extended = np.append(date_nums, date_nums[-1] + date_diff)
    else:
        date_nums_extended = np.append(date_nums, date_nums[0] + 1)
    
    X, Y = np.meshgrid(date_nums_extended, y_positions)
    
    # Plot as heatmap with dates
    im = ax3.pcolormesh(X, Y, change_matrix.T.values, cmap='RdYlGn', 
                        vmin=-1, vmax=1, shading='auto')
    
    # Set y-axis to show tickers (centered on each row)
    ax3.set_yticks(y_positions[:-1] + 0.5)
    ax3.set_yticklabels(tickers)
    
    # Format x-axis with dates
    ax3.xaxis_date()  # Tell matplotlib these are dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    
    # Rotate date labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax3.set_title('Position Changes Over Time (Green=Buy/Increase, Red=Sell/Decrease)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Ticker', fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Position Change (Weight Δ)', fontsize=10)
    
    # 4. Risk vs Position Correlation (Scatter for each ticker)
    ax4 = fig.add_subplot(gs[3, 0])
    
    # Calculate correlation between risk contribution and position weight for each ticker
    correlations = []
    for tkr in tickers:
        risk_series = pct_risk_contrib[tkr].dropna()
        weight_series = signals_df[tkr].dropna()
        
        # Align indices
        common_idx = risk_series.index.intersection(weight_series.index)
        if len(common_idx) > 10:  # Need enough data points
            corr = risk_series.loc[common_idx].corr(weight_series.loc[common_idx])
            correlations.append({'Ticker': tkr, 'Correlation': corr})
    
    if correlations:
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('Correlation', ascending=True)
        
        bars = ax4.barh(corr_df['Ticker'], corr_df['Correlation'], 
                        color=[color_map[t] for t in corr_df['Ticker']], alpha=0.7)
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_title('Risk Contribution vs Position Weight Correlation', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Correlation Coefficient', fontsize=10)
        ax4.set_ylabel('Ticker', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (tkr, corr) in enumerate(zip(corr_df['Ticker'], corr_df['Correlation'])):
            ax4.text(corr, i, f' {corr:.2f}', va='center', fontsize=9)
    
    # 5. Summary Statistics Table
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis('off')
    
    # Calculate summary stats
    summary_data = []
    for tkr in tickers:
        risk_mean = pct_risk_contrib[tkr].mean()
        risk_max = pct_risk_contrib[tkr].max()
        weight_mean = signals_df[tkr].mean()
        weight_max = signals_df[tkr].max()
        
        # Count buy/sell events
        changes = position_changes[tkr].dropna()
        buy_events = (changes > 0.01).sum()  # Significant increase
        sell_events = (changes < -0.01).sum()  # Significant decrease
        
        summary_data.append({
            'Ticker': tkr,
            'Avg Risk %': f"{risk_mean*100:.1f}%",
            'Max Risk %': f"{risk_max*100:.1f}%",
            'Avg Weight': f"{weight_mean*100:.1f}%",
            'Buy Events': buy_events,
            'Sell Events': sell_events
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table
    table = ax5.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    # Print actionable insights
    print("\n" + "="*80)
    print("RISK CONTRIBUTION ANALYSIS - ACTIONABLE INSIGHTS")
    print("="*80)
    
    print("\n📊 INTERPRETATION GUIDE:")
    print("-" * 80)
    print("1. RISK CONTRIBUTION: Shows how much each ticker contributes to portfolio risk")
    print("   - High risk contribution = ticker is driving portfolio volatility")
    print("   - Low risk contribution = ticker is stabilizing the portfolio")
    
    print("\n2. POSITION WEIGHTS: Shows actual portfolio allocation")
    print("   - Weight = 1.0 (100%) = fully allocated to that ticker")
    print("   - Weight = 0.0 = not holding that ticker")
    
    print("\n3. POSITION CHANGES (Buy/Sell Signals):")
    print("   - GREEN areas = BUY/INCREASE position (weight going up)")
    print("   - RED areas = SELL/DECREASE position (weight going down)")
    print("   - This shows when the strategy rotates between tickers")
    
    print("\n4. RISK vs POSITION CORRELATION:")
    print("   - Positive correlation = risk increases when position increases (expected)")
    print("   - Negative correlation = risk decreases when position increases (unusual)")
    
    print("\n" + "="*80)
    print("💡 TRADING SIGNALS:")
    print("-" * 80)
    
    # Identify recent signals
    if len(position_changes) > 0:
        last_date = position_changes.index[-1]
        last_changes = position_changes.loc[last_date]
        
        buy_signals = last_changes[last_changes > 0.01].sort_values(ascending=False)
        sell_signals = last_changes[last_changes < -0.01].sort_values(ascending=True)
        
        if len(buy_signals) > 0:
            print(f"\n🟢 BUY/INCREASE SIGNALS (as of {last_date.date()}):")
            for tkr, change in buy_signals.items():
                current_weight = signals_df.loc[last_date, tkr]
                risk_contrib = pct_risk_contrib.loc[last_date, tkr] if last_date in pct_risk_contrib.index else 0
                print(f"   • {tkr}: Weight increased by {change*100:.1f}% "
                      f"(Current: {current_weight*100:.1f}%, Risk: {risk_contrib*100:.1f}%)")
        
        if len(sell_signals) > 0:
            print(f"\n🔴 SELL/DECREASE SIGNALS (as of {last_date.date()}):")
            for tkr, change in sell_signals.items():
                current_weight = signals_df.loc[last_date, tkr]
                risk_contrib = pct_risk_contrib.loc[last_date, tkr] if last_date in pct_risk_contrib.index else 0
                print(f"   • {tkr}: Weight decreased by {abs(change)*100:.1f}% "
                      f"(Current: {current_weight*100:.1f}%, Risk: {risk_contrib*100:.1f}%)")
        
        if len(buy_signals) == 0 and len(sell_signals) == 0:
            print(f"\n⚪ NO SIGNIFICANT CHANGES (as of {last_date.date()})")
            print("   Strategy is maintaining current positions")
    
    print("\n" + "="*80)

