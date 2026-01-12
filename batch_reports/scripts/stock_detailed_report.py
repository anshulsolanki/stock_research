"""
STOCK DETAILED REPORT GENERATOR
================================

Generates a comprehensive PDF report for an individual stock, replicating the
"Stock Deepdive" UI section with all technical and fundamental analyses.

This is a REPORTING LAYER ONLY - it calls existing analysis modules and formats
their outputs into PDF. No new analysis logic is implemented here.
"""

import sys
import os
import datetime
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import yfinance as yf
import io

# Add parent directories to path
# Scripts are now in batch_reports/scripts, so we need to go up two levels to reach root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'fundamental_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lagging_indicator_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'leading_indicator_analysis'))

# Import Analysis Modules
try:
    import fundamental_analysis
    import macd_analysis
    import supertrend_analysis
    import bollinger_band_analysis
    import crossover_analysis
    import crossover_analysis
    import rsi_volume_divergence
    import volatility_squeeze_analysis
    import rs_analysis
    import volume_analysis
    import multi_timeframe_analysis
    from website_screen_shot_automation.trendlyne_snapshot import get_trendlyne_snapshots
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ... (rest of code)


def create_title_page(pdf, ticker, current_price=None):
    """Creates title page with stock info."""
    plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    
    plt.text(0.5, 0.7, "Stock Detailed Analysis Report", 
             ha='center', va='center', fontsize=28, weight='bold', color='#1e293b')
    
    plt.text(0.5, 0.55, ticker, 
             ha='center', va='center', fontsize=36, weight='bold', color='#2563eb')
    
    if current_price:
        plt.text(0.5, 0.45, f"Current Price: ₹{current_price:.2f}", 
                 ha='center', va='center', fontsize=18, color='#475569')
    
    plt.text(0.5, 0.25, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", 
             ha='center', va='center', fontsize=12, color='#94a3b8')
    
    pdf.savefig()
    plt.close()


def fetch_shared_data(ticker):
    """
    Fetches 4 years of daily data to specific shared needs.
    This covers:
    - 3-year Price Chart (needs ~1095 days)
    - 200 EMA (needs ~300 days)
    - General Technical Analysis
    """
    try:
        print(f"  Fetching shared data for {ticker} (4 years)...")
        stock = yf.Ticker(ticker)
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=4*365)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            print(f"  Warning: No data fetched for {ticker}")
            return None
            
        return df
    except Exception as e:
        print(f"  Error fetching shared data: {e}")
        return None


def add_price_chart(pdf, ticker, save_plot=True, df=None):
    """Fetches and plots 3-year daily price chart."""
    try:
        # Use shared dataframe if provided
        if df is not None:
             # Filter for last 3 years
             end_date = df.index[-1]
             start_date = end_date - datetime.timedelta(days=3*365)
             hist = df.loc[start_date:end_date]
        else:
            print(f"  Fetching 3-year price data for {ticker}...")
            stock = yf.Ticker(ticker)
            
            # Fetch 3 years of daily data
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=3*365)
            hist = stock.history(start=start_date, end=end_date, interval='1d')
        
        if hist.empty:
            print(f"  Warning: No price data available for {ticker}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(hist.index, hist['Close'], linewidth=1.5, color='#2563eb')
        ax.fill_between(hist.index, hist['Close'], alpha=0.2, color='#2563eb')
        
        ax.set_title(f"{ticker} - 3 Year Daily Price Chart", 
                     fontsize=16, weight='bold', pad=20, color='#1e293b')
        ax.set_xlabel('Date', fontsize=12, color='#475569')
        ax.set_ylabel('Price (₹)', fontsize=12, color='#475569')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        plt.tight_layout()
        if save_plot:
            pdf.savefig(fig)
        plt.close(fig)
        
        return hist['Close'].iloc[-1]  # Return current price
        
    except Exception as e:
        print(f"  Error fetching price chart: {e}")
        return None


def render_styled_table_page(pdf, df, title, col_formats=None):
    """
    Renders a dataframe as a styled table (reused from weekly_analysis.py).
    """
    if df.empty:
        return

    if not col_formats:
        col_formats = {}

    rows_per_page = 20
    num_pages = (len(df) // rows_per_page) + 1
    
    for i in range(num_pages):
        start_idx = i * rows_per_page
        end_idx = min((i + 1) * rows_per_page, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()
        
        if chunk.empty:
            continue
            
        # Pre-process data for display
        display_chunk = chunk.copy()
        
        # Helper to replace unsupported emojis
        def replace_icons(val):
            if isinstance(val, str):
                val = val.replace('🟢', '●').replace('🔴', '●').replace('🟡', '●')
                val = val.replace('⬆️', '▲').replace('⬇️', '▼')
                val = val.replace('➖', '-').replace('⚠️', '!').replace('💥', '*')
                return val.strip()
            return val

        for col in chunk.columns:
            display_chunk[col] = display_chunk[col].apply(replace_icons)
            
            fmt = col_formats.get(col)
            if fmt == 'bool':
                display_chunk[col] = chunk[col].apply(lambda x: '✔' if x else '-')
            elif fmt == 'float':
                display_chunk[col] = chunk[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            elif fmt == 'percent':
                display_chunk[col] = chunk[col].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        ax.set_title(f"{title} (Page {i+1}/{num_pages})", 
                     fontsize=16, weight='bold', pad=20, color='#1e293b')
        
        # Create table
        table = ax.table(cellText=display_chunk.values, colLabels=display_chunk.columns, 
                        loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style cells
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='#475569')
                cell.set_facecolor('#f8fafc')
                cell.set_edgecolor('#e2e8f0')
                cell.set_linewidth(1)
            else:
                cell.set_edgecolor('#e2e8f0')
                cell.set_linewidth(0.5)
                
                col_name = display_chunk.columns[col]
                val = chunk.iloc[row-1][col_name]
                
                # Apply color coding based on value
                if isinstance(val, str):
                    val_lower = val.lower()
                    if 'yes' in val_lower or 'growing' in val_lower:
                        cell.set_text_props(color='#16a34a', weight='bold')
                    elif 'no' in val_lower or 'declining' in val_lower:
                        cell.set_text_props(color='#ef4444', weight='bold')

        pdf.savefig(fig)
        plt.close(fig)


def render_fundamentals_page(pdf, ticker):
    """Calls existing fundamental analysis functions and formats results."""
    print("  Running Fundamental Analysis...")
    
    # Long-term (4Y) Analysis
    fund_4y_data = []
    
    metrics_4y = [
        ('Revenue', fundamental_analysis.analyze_revenue_growth_4y),
        ('Net Income', fundamental_analysis.analyze_profit_growth_4y),
        ('ROE', fundamental_analysis.analyze_roe_growth_4y),
        ('EPS', fundamental_analysis.analyze_eps_growth_4y)
    ]
    
    for name, func in metrics_4y:
        try:
            res = func(ticker)
            if res.get('success'):
                fund_4y_data.append({
                    'Metric': name,
                    'Is Growing?': 'Yes' if res.get('is_growing') else 'No',
                    'Accelerating?': 'Yes' if res.get('has_accelerating_trend') else 'No',
                    '1Y Growth': f"{res.get('growth_1y', 0):.2f}%",
                    '3Y CAGR': f"{res.get('growth_3y_cagr', 0):.2f}%"
                })
        except Exception as e:
            print(f"    Error in 4Y {name}: {e}")
    
    if not fund_4y_data and not fund_6q_data:
        return

    # Create one figure for both tables
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    # Render 4Y Table (Top)
    if fund_4y_data:
        df_4y = pd.DataFrame(fund_4y_data)
        ax1.axis('tight')
        ax1.axis('off')
        ax1.set_title("Long-term Fundamental Analysis (4 Years)", fontsize=16, weight='bold', pad=10, color='#1e293b')
        
        # Helper to render table on an axis
        def render_table_on_ax(ax, df):
            # Pre-process data
            display_df = df.copy()
            
            # Helper to replace icons
            def replace_icons(val):
                if isinstance(val, str):
                    val = val.replace('🟢', '●').replace('🔴', '●').replace('🟡', '●')
                    val = val.replace('⬆️', '▲').replace('⬇️', '▼')
                    val = val.replace('➖', '-').replace('⚠️', '!').replace('💥', '*')
                    return val.strip()
                return val

            for col in df.columns:
                display_df[col] = display_df[col].apply(replace_icons)

            table = ax.table(cellText=display_df.values, colLabels=display_df.columns, 
                            loc='center', cellLoc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Style cells
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='#475569')
                    cell.set_facecolor('#f8fafc')
                    cell.set_edgecolor('#e2e8f0')
                    cell.set_linewidth(1)
                else:
                    cell.set_edgecolor('#e2e8f0')
                    cell.set_linewidth(0.5)
                    col_name = display_df.columns[col]
                    val = df.iloc[row-1][col_name]
                    # Apply color coding
                    if isinstance(val, str):
                        val_lower = val.lower()
                        if 'yes' in val_lower or 'growing' in val_lower:
                            cell.set_text_props(color='#16a34a', weight='bold')
                        elif 'no' in val_lower or 'declining' in val_lower:
                            cell.set_text_props(color='#ef4444', weight='bold')
        
        render_table_on_ax(ax1, df_4y)
    else:
        ax1.axis('off')
        ax1.text(0.5, 0.5, "No Long-term Data Available", ha='center', va='center')

    # Calculate 6Q Data (Missing Block Restored)
    fund_6q_data = []
    metrics_6q = [
        ('Revenue', fundamental_analysis.analyze_revenue_growth_6q),
        ('Net Income', fundamental_analysis.analyze_profit_growth_6q),
        ('ROE', fundamental_analysis.analyze_roe_growth_6q),
        ('EPS', fundamental_analysis.analyze_eps_growth_6q)
    ]
    
    for name, func in metrics_6q:
        try:
            res = func(ticker)
            if res.get('success'):
                fund_6q_data.append({
                    'Metric': name,
                    'Is Growing?': 'Yes' if res.get('is_growing') else 'No',
                    'Recent QoQ': f"{res.get('recent_quarter_growth', 0):.2f}%",
                    'Avg QoQ': f"{res.get('average_qoq_growth', 0):.2f}%"
                })
        except Exception as e:
            print(f"    Error in 6Q {name}: {e}")

    # Render 6Q Table (Bottom)
    if fund_6q_data:
        df_6q = pd.DataFrame(fund_6q_data)
        ax2.axis('tight')
        ax2.axis('off')
        ax2.set_title("Short-term Fundamental Analysis (6 Quarters)", fontsize=16, weight='bold', pad=10, color='#1e293b')
        render_table_on_ax(ax2, df_6q)
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, "No Short-term Data Available", ha='center', va='center')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def render_summary_page(pdf, title, summary_data):
    """Renders a text summary page with key metrics."""
    if not summary_data:
        return
    
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.92, title, ha='center', va='top', 
             fontsize=18, weight='bold', color='#1e293b')
    
    # Summary box
    y_position = 0.82
    line_height = 0.04
    
    for key, value in summary_data.items():
        # Key in bold, value in regular
        text = f"{key}: "
        fig.text(0.15, y_position, text, ha='left', va='top',
                 fontsize=12, weight='bold', color='#475569')
        
        # Determine color and weight based on value content
        color = '#1e293b'  # Default Slate-800
        weight = 'normal'
        
        if isinstance(value, str):
            value_lower = str(value).lower()
            
            # RED keywords (Negative/Bearish) - Check FIRST to catch "Strong Downtrend" correctly
            if any(word in value_lower for word in ['bearish', 'sell', 'downtrend', 'negative', 'weakening', 'lagging', 'declining', 'no', 'underperforming']):
                color = '#dc2626'  # Red-600
                weight = 'bold'
                
            # GREEN keywords (Positive/Bullish)
            elif any(word in value_lower for word in ['bullish', 'buy', 'uptrend', 'positive', 'strengthening', 'strong', 'leader', 'emerging', 'accelerating', 'detected', 'growing', 'yes']):
                color = '#16a34a'  # Green-600
                weight = 'bold'
                
            # ORANGE/AMBER keywords (Neutral/Warning)
            elif any(word in value_lower for word in ['neutral', 'mixed', 'monitor']):
                color = '#d97706'  # Amber-600
                weight = 'bold'
        
        # Increased x-offset to 0.50 to prevent overlap with long keys
        fig.text(0.50, y_position, str(value), ha='left', va='top',
                 fontsize=12, color=color, weight=weight)
        
        y_position -= line_height
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_analysis_chart(pdf, figure, title=None):
    """Adds a matplotlib figure to PDF."""
    if figure:
        # Title handling removed to prevent overlay as charts already have titles
        # if title:
        #     figure.suptitle(title, fontsize=16, weight='bold', y=0.98)
        pdf.savefig(figure)
        plt.close(figure)


def render_rsi_volume_divergence(pdf, ticker, df=None):
    try:
        print("    - RSI-Volume Divergence")
        
        # Filter for last 2 years of data if df is provided
        local_df = df
        if local_df is not None:
             cutoff_date = local_df.index[-1] - datetime.timedelta(days=2*365)
             local_df = local_df[local_df.index > cutoff_date].copy()
             
        res = rsi_volume_divergence.run_analysis(ticker, show_plot=False, df=local_df)
        if res.get('success'):
            # Add summary page
            summary = {
                'Current RSI': f"{res.get('current_rsi', 0):.2f}",
                'Current Volume': f"{res.get('current_volume', 0):.0f}",
                'Volume MA 20': f"{res.get('volume_ma_20', 0):.0f}",
                'Bullish Divergences': len(res.get('bullish_divergences', [])),
                'Bearish Divergences': len(res.get('bearish_divergences', []))
            }
            
            # Add last 4 divergences (2 bullish + 2 bearish)
            bullish = res.get('bullish_divergences', [])[-2:] if res.get('bullish_divergences') else []
            bearish = res.get('bearish_divergences', [])[-2:] if res.get('bearish_divergences') else []
            
            for idx, div in enumerate(bullish, 1):
                summary[f'Bullish Div {idx}'] = f"Date: {div.get('Date', 'N/A')}, Price: {div.get('Price', 0):.2f}"
            for idx, div in enumerate(bearish, 1):
                summary[f'Bearish Div {idx}'] = f"Date: {div.get('Date', 'N/A')}, Price: {div.get('Price', 0):.2f}"
            
            render_summary_page(pdf, f"{ticker} - RSI-Volume Summary", summary)
            
            # Add chart
            if res.get('figure'):
                add_analysis_chart(pdf, res['figure'], f"{ticker} - RSI-Volume Divergence Analysis")
    except Exception as e:
        print(f"    Error in RSI-Volume: {e}")


def render_volatility_squeeze(pdf, ticker, df=None):
    try:
        print("    - Volatility Squeeze")
        
        # Filter for last 2 years of data if df is provided
        local_df = df
        if local_df is not None:
             cutoff_date = local_df.index[-1] - datetime.timedelta(days=2*365)
             local_df = local_df[local_df.index > cutoff_date].copy()
             
        res = volatility_squeeze_analysis.run_analysis(ticker, show_plot=False, df=local_df)
        if res.get('success'):
            # Add summary page
            summary = {
                'BB Width': f"{res.get('current_bb_width', 0):.4f}" if res.get('current_bb_width') else 'N/A',
                'ATR': f"{res.get('current_atr', 0):.4f}" if res.get('current_atr') else 'N/A',
                'Total Signals': len(res.get('signals', []))
            }
            # Add last 5 signals with details
            if res.get('signals'):
                recent_signals = res.get('signals', [])[-5:]  # Last 5
                for idx, sig in enumerate(recent_signals, 1):
                    sig_type = sig.get('Type', 'Unknown')
                    sig_date = sig.get('Date', 'N/A')
                    sig_price = sig.get('Price', 0)
                    summary[f'Signal {idx}'] = f"{sig_type} at {sig_date} (Price: {sig_price:.2f})"
            
            render_summary_page(pdf, f"{ticker} - Volatility Squeeze Summary", summary)
            
            # Add chart
            if res.get('figure'):
                add_analysis_chart(pdf, res['figure'], f"{ticker} - Volatility Squeeze Analysis")
    except Exception as e:
        print(f"    Error in Volatility Squeeze: {e}")


def render_macd(pdf, ticker, df=None):
    try:
        print("    - MACD")
        
        # Filter for last 2 years of data if df is provided
        local_df = df
        if local_df is not None:
             cutoff_date = local_df.index[-1] - datetime.timedelta(days=2*365)
             local_df = local_df[local_df.index > cutoff_date].copy()
             
        res = macd_analysis.run_analysis(ticker, show_plot=False, df=local_df)
        if res.get('success'):
            # Add summary page - matching UI format exactly
            summary = {
                'MACD Line': f"{res.get('macd_line', 0):.2f}",
                'Signal Line': f"{res.get('signal_line', 0):.2f}",
                'Histogram': f"{res.get('histogram', 0):.2f}",
                'Trend': res.get('trend', 'Unknown'),
                'Momentum': res.get('momentum', 'Unknown'),
                'Signal': res.get('crossover_signal', 'No Recent Signal')
            }
            
            # Add divergences section below
            divergences = res.get('divergences', [])
            if divergences:
                summary['Divergences Detected'] = len(divergences)
                for idx, div in enumerate(divergences[-3:], 1):
                    div_date = div.get('Date', 'N/A')
                    div_price = div.get('Price', 0)
                    div_details = div.get('Details', '')
                    summary[f'  └ {div.get("Type", "Unknown")}'] = f"Date: {div_date}, Price: {div_price:.2f}"
            
            render_summary_page(pdf, f"{ticker} - MACD Summary", summary)
            
            # Add chart
            if res.get('figure'):
                add_analysis_chart(pdf, res['figure'], f"{ticker} - MACD Analysis")
    except Exception as e:
        print(f"    Error in MACD: {e}")


def render_supertrend(pdf, ticker, df=None):
    try:
        print("    - Supertrend")
        
        # Filter for last 2 years of data if df is provided
        local_df = df
        if local_df is not None:
             cutoff_date = local_df.index[-1] - datetime.timedelta(days=2*365)
             local_df = local_df[local_df.index > cutoff_date].copy()
             
        res = supertrend_analysis.run_analysis(ticker, show_plot=False, df=local_df)
        if res.get('success'):
            # Add summary page
            summary = {
                'Status': res.get('status', 'Unknown'),
                'Supertrend Value': f"{res.get('supertrend_value', 0):.2f}"
            }
            if res.get('signal_date'):
                # Format the date nicely
                sig_date = res.get('signal_date')
                if isinstance(sig_date, (datetime.datetime, pd.Timestamp)):
                    sig_date_str = sig_date.strftime('%Y-%m-%d')
                else:
                    sig_date_str = str(sig_date)
                summary['Signal Identified On'] = sig_date_str
            
            render_summary_page(pdf, f"{ticker} - Supertrend Summary", summary)
            
            # Add chart
            if res.get('figure'):
                add_analysis_chart(pdf, res['figure'], f"{ticker} - Supertrend Analysis")
    except Exception as e:
        print(f"    Error in Supertrend: {e}")


def render_bollinger_bands(pdf, ticker, df=None):
    try:
        print("    - Bollinger Bands")
        
        # Filter for last 2 years of data if df is provided
        local_df = df
        if local_df is not None:
             cutoff_date = local_df.index[-1] - datetime.timedelta(days=2*365)
             local_df = local_df[local_df.index > cutoff_date].copy()
             
        res = bollinger_band_analysis.run_analysis(ticker, show_plot=False, df=local_df)
        if res.get('success'):
            # Add summary page
            summary = {
                'Current Price': f"{res.get('last_price', 0):.2f}",
                'Upper Band': f"{res.get('bb_upper', 0):.2f}",
                'Middle Band (SMA 20)': f"{res.get('sma_20', 0):.2f}",
                'Lower Band': f"{res.get('bb_lower', 0):.2f}",
                '%B': f"{res.get('pct_b', 0):.4f}",
                'Band Width': f"{res.get('bandwidth', 0):.4f}",
                'Status': res.get('status', 'Unknown'),
                'Signal': res.get('signal', 'Unknown')
            }
            
            # Add recent signals
            if res.get('signals'):
                recent_signals = res.get('signals', [])[-3:]  # Last 3
                for idx, sig in enumerate(recent_signals, 1):
                    summary[f'Recent Signal {idx}'] = f"{sig.get('Type', 'Unknown')} at {sig.get('Date', 'N/A')}"
            
            render_summary_page(pdf, f"{ticker} - Bollinger Bands Summary", summary)
            
            # Add chart
            if res.get('figure'):
                add_analysis_chart(pdf, res['figure'], f"{ticker} - Bollinger Bands Analysis")
    except Exception as e:
        print(f"    Error in Bollinger Bands: {e}")


def render_ema_crossover(pdf, ticker, df=None):
    try:
        print("    - EMA Crossover")
        res = crossover_analysis.run_analysis(ticker, show_plot=False, df=df)
        if res.get('success'):
            # Add summary page
            summary = {
                'EMA 20': f"{res.get('ema_20', 0):.2f}",
                'EMA 50': f"{res.get('ema_50', 0):.2f}",
                'EMA 200': f"{res.get('ema_200', 0):.2f}",
                'Trend Status': res.get('trend_status', 'Unknown')
            }
            if res.get('gc_date'):
                summary['Golden Cross Date'] = str(res.get('gc_date'))
            if res.get('gc_price'):
                summary['Golden Cross Price'] = f"{res.get('gc_price', 0):.2f}"
            
            render_summary_page(pdf, f"{ticker} - EMA Crossover Summary", summary)
            
            # Add chart
            if res.get('figure'):
                add_analysis_chart(pdf, res['figure'], f"{ticker} - EMA Crossover Analysis")
    except Exception as e:
        print(f"    Error in Crossover: {e}")


        print(f"    Error in Crossover: {e}")


def render_rs_analysis(pdf, ticker, df=None):
    """
    Runs RS analysis and adds summary + chart to PDF.
    Uses sector-specific index for comparison if available.
    """
    try:
        print("  Running Relative Strength Analysis...")
        # Run RS analysis with sector index
        res = rs_analysis.run_analysis(ticker, show_plot=False, use_sector_index=True, df=df)
        
        if res.get('success'):
            # Create summary page with better organization
            summary = {}
            
            # === SECTION 1: Overview ===
            summary['═══ OVERVIEW ═══'] = ''
            summary['Benchmark Index'] = res.get('benchmark', 'N/A')
            if res.get('sector'):
                summary['Sector'] = res.get('sector')
            summary['Classification'] = res.get('classification', 'Unknown')
            summary['RS Score'] = f"{res.get('rs_score', 0):.1f}/100"
            
            # === SECTION 2: RS Ratios ===
            summary[''] = ''  # Spacer
            summary['═══ RS RATIOS ═══'] = ''
            rs_ratios = res.get('rs_ratios', {})
            for name in ['1M', '3M', '6M', '1Y']:
                if name in rs_ratios and rs_ratios[name] is not None:
                    value = rs_ratios[name]
                    # Add text label instead of emoji for PDF compatibility
                    if value >= 1.2:
                        indicator = '[Strong]'
                    elif value >= 1.0:
                        indicator = '[Leader]'
                    elif value >= 0.8:
                        indicator = '[Neutral]'
                    else:
                        indicator = '[Lagging]'
                    summary[f'  {name} RS'] = f"{value:.3f}  {indicator}"
            
            # === SECTION 3: Turnaround Analysis ===
            turnaround_info = res.get('turnaround_info', {})
            if turnaround_info:
                summary['  '] = ''  # Spacer
                summary['═══ TURNAROUND ANALYSIS ═══'] = ''
                
                # Check each condition
                emerging_rs = turnaround_info.get('emerging_rs', False)
                medium_lagging = turnaround_info.get('medium_term_lagging', False)
                perf_improving = turnaround_info.get('absolute_perf_improving', False)
                is_turnaround = turnaround_info.get('is_turnaround', False)
                
                # Overall status
                if is_turnaround:
                    summary['Turnaround Status'] = '!! DETECTED !!'
                else:
                    summary['Turnaround Status'] = 'Not Detected'
                
                # Detailed breakdown
                summary['   '] = ''  # Spacer
                summary['SIGNAL CRITERIA:'] = ''
                
                # Criterion 1: Emerging RS
                if emerging_rs:
                    summary['  ✓ Emerging RS'] = f"1M ({rs_ratios.get('1M', 0):.3f}) > 3M ({rs_ratios.get('3M', 0):.3f})"
                else:
                    summary['  ✗ Emerging RS'] = 'Not accelerating'
                
                # Criterion 2: Medium-term lagging
                if medium_lagging:
                    lagging_details = []
                    if '6M' in rs_ratios and rs_ratios['6M'] <= 1.0:
                        lagging_details.append(f"6M={rs_ratios['6M']:.3f}")
                    if '1Y' in rs_ratios and rs_ratios['1Y'] <= 1.0:
                        lagging_details.append(f"1Y={rs_ratios['1Y']:.3f}")
                    summary['  ✓ Medium-term Lagging'] = f"{', '.join(lagging_details)} (≤1.0)"
                else:
                    summary['  ✗ Medium-term Lagging'] = 'Not lagging'
                
                # Criterion 3: Performance improving
                if perf_improving and 'perf_3m' in turnaround_info and 'perf_6m' in turnaround_info:
                    perf_3m = turnaround_info['perf_3m']
                    perf_6m = turnaround_info['perf_6m']
                    summary['  ✓ Performance Improving'] = f"3M ({perf_3m:+.1f}%) > 6M ({perf_6m:+.1f}%)"
                else:
                    summary['  ✗ Performance Improving'] = 'Not improving'
            
            # Render summary page
            render_summary_page(pdf, f"{ticker} - Relative Strength Analysis", summary)
            
            # Add RS analysis chart
            if res.get('figure'):
                add_analysis_chart(pdf, res['figure'])
    except Exception as e:
        print(f"    Error in RS Analysis: {e}")



def render_volume_analysis(pdf, ticker, df=None):
    """
    Runs Volume Analysis and adds summary + chart to PDF.
    """
    try:
        print("  Running Volume Analysis...")
        # Note: volume_analysis.run_analysis now supports return_figure=True
        res = volume_analysis.run_analysis(ticker, show_plot=False, df=df, return_figure=True)
        
        if res.get('success'):
            summary = {}
            summary['═══ VOLUME ANALYSIS ═══'] = ''
            
            # Signals
            divergences = res.get('divergences', [])
            if not divergences:
                summary['Status'] = 'No significant volume anomalies detected.'
            else:
                summary['Status'] = f"{len(divergences)} Signals Detected"
                summary[''] = '' # Spacer
                
                # List last 5 signals
                for div in divergences[-5:]:
                    date = div['Date']
                    signal_type = div['Type']
                    # Add emoji/marker based on type
                    if 'Buying' in signal_type or 'Bullish' in signal_type:
                         marker = '[+]'
                    elif 'Selling' in signal_type or 'Bearish' in signal_type or 'Distribution' in signal_type or 'Climax' in signal_type:
                         marker = '[-]'
                    else:
                         marker = '[?]'
                         
                    summary[f"{date} {marker}"] = signal_type

            render_summary_page(pdf, f"{ticker} - Volume Analysis", summary)
            
            if res.get('figure'):
                add_analysis_chart(pdf, res['figure'], f"{ticker} - Volume Analysis")
                
    except Exception as e:
        print(f"    Error in Volume Analysis: {e}")


def render_technical_indicators(pdf, ticker, df=None):
    """
    Renders all technical indicators in the specified order:
    1. EMA Crossover
    2. Bollinger Bands
    3. Supertrend
    4. MACD
    5. Volatility Squeeze
    6. RSI-Volume Divergence
    """
    print("  Running Technical Analysis...")
    render_ema_crossover(pdf, ticker, df=df)
    render_bollinger_bands(pdf, ticker, df=df)
    render_supertrend(pdf, ticker, df=df)
    render_macd(pdf, ticker, df=df)
    render_volatility_squeeze(pdf, ticker, df=df)
    render_rsi_volume_divergence(pdf, ticker, df=df)
    render_volume_analysis(pdf, ticker, df=df)


def render_multi_timeframe_analysis(pdf, ticker, df=None):
    """
    Renders Multi-Timeframe Analysis (Supertrend, MACD, Charts).
    """
    try:
        print("  Running Multi-Timeframe Analysis...")
        # Note: Multi-timeframe analysis currently handles its own multi-timeframe fetching
        # We can pass specific timeframe dataframes if refactored, but for now it might re-fetch
        # or we update run_analysis to accept a map of DFs.
        # Checking multi_timeframe_analysis.py, it fetches data internally.
        # To optimize, we should refactor multi_timeframe_analysis.run_analysis to accept data.
        # For now, let's keep it as is or do a quick check.
        # The prompt asked to optimize this file. 
        # Ideally, we pass the data.
        # Let's assume multi_timeframe_analysis.run_analysis will be updated or handles it.
        # Actually, multi_timeframe_analysis.py fetches data for '1wk', '1d', '15m'.
        # Our fetch_shared_data only gets '1d'.
        # So we can pass '1d' data to avoid ONE fetch, but it still needs '1wk' and '15m'.
        # However, looking at previous edits, we didn't update multi_timeframe_analysis to accept df yet.
        # Let's stick to the plan: pass df where possible.
        res = multi_timeframe_analysis.run_analysis(ticker, show_plot=False)
        
        if res.get('success'):
            # --- Page 1: Tables ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
            
            # Supertrend Table
            st_data = res.get('supertrend_results', [])
            if st_data:
                df_st = pd.DataFrame(st_data)
                # Rename columns for display
                df_st.columns = ['Timeframe', 'Status', 'Value', 'Signal Date', 'Last Price']
                # Reorder
                df_st = df_st[['Timeframe', 'Status', 'Value', 'Last Price', 'Signal Date']]
                
                ax1.axis('tight')
                ax1.axis('off')
                ax1.set_title("Multi-Timeframe Supertrend Analysis", fontsize=16, weight='bold', pad=10, color='#1e293b')
                
                table1 = ax1.table(cellText=df_st.values, colLabels=df_st.columns, loc='center', cellLoc='center')
                table1.auto_set_font_size(False)
                table1.set_fontsize(10)
                table1.scale(1.2, 1.8)
                
                # Styling
                for (row, col), cell in table1.get_celld().items():
                    if row == 0:
                        cell.set_text_props(weight='bold', color='#475569')
                        cell.set_facecolor('#f8fafc')
                    else:
                        # Color coding for Status
                        if col == 1: # Status column
                            val = df_st.iloc[row-1]['Status']
                            if 'UPTREND' in str(val):
                                cell.set_text_props(color='#16a34a', weight='bold') # Green
                            elif 'DOWNTREND' in str(val):
                                cell.set_text_props(color='#dc2626', weight='bold') # Red
            else:
                ax1.axis('off')
                ax1.text(0.5, 0.5, "No Supertrend Data", ha='center', va='center')

            # MACD Table
            macd_data = res.get('macd_results', [])
            if macd_data:
                df_macd = pd.DataFrame(macd_data)
                # Rename columns
                df_macd.columns = ['Timeframe', 'Trend', 'Momentum', 'Signal']
                
                ax2.axis('tight')
                ax2.axis('off')
                ax2.set_title("Multi-Timeframe MACD Analysis", fontsize=16, weight='bold', pad=10, color='#1e293b')
                
                table2 = ax2.table(cellText=df_macd.values, colLabels=df_macd.columns, loc='center', cellLoc='center')
                table2.auto_set_font_size(False)
                table2.set_fontsize(10)
                table2.scale(1.2, 1.8)
                
                # Styling
                for (row, col), cell in table2.get_celld().items():
                    if row == 0:
                        cell.set_text_props(weight='bold', color='#475569')
                        cell.set_facecolor('#f8fafc')
                    else:
                        # Color coding for Trend
                        if col == 1: # Trend column
                            val = df_macd.iloc[row-1]['Trend']
                            if 'Bullish' in str(val):
                                cell.set_text_props(color='#16a34a', weight='bold')
                            elif 'Bearish' in str(val):
                                cell.set_text_props(color='#dc2626', weight='bold')

            else:
                ax2.axis('off')
                ax2.text(0.5, 0.5, "No MACD Data", ha='center', va='center')
                
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # --- Page 2: Charts ---
            if res.get('figure'):
                # The figure is already created by multi_timeframe_analysis with 3 subplots
                # We just need to save it
                fig_charts = res['figure']
                # Ensure size is appropriate for PDF landscape/portrait
                fig_charts.set_size_inches(11, 8.5) # Landscape
                pdf.savefig(fig_charts)
                plt.close(fig_charts)

    except Exception as e:
        print(f"    Error in Multi-Timeframe Analysis: {e}")



def render_trendlyne_snapshots(pdf, ticker):
    """
    Captures and renders Trendlyne screenshots.
    This uses browser automation and might take some time.
    """
    try:
        print("  Running Trendlyne Snapshot Automation...")
        # Clean ticker for Trendlyne search (remove .NS/.BO and common suffixes)
        search_name = ticker.replace('.NS', '').replace('.BO', '')
        
        # Use a temporary output directory or valid location
        # The script defaults to its own dir, but returns absolute paths.
        # We can pass "." to let it manage files, or handle it here.
        # Let's just pass the ticker name.
        
        # Call with save_to_file=False to get bytes in memory
        snapshots = get_trendlyne_snapshots(stock_name=search_name, headless=True, save_to_file=False)
        
        if not snapshots:
            print("    No Trendlyne snapshots generated.")
            return

        print(f"    Captured {len(snapshots)} snapshots.")
        
        for item in snapshots:
            try:
                img = None
                if item.get('type') == 'memory':
                     # Create image from bytes
                     img_bytes = item.get('content')
                     if img_bytes:
                         img = plt.imread(io.BytesIO(img_bytes), format='png')
                elif item.get('type') == 'file':
                    img_path = item.get('content')
                    if os.path.exists(img_path):
                        img = plt.imread(img_path)
                    else:
                        print(f"    Image file not found: {img_path}")
                
                if img is not None:
                    fig, ax = plt.subplots(figsize=(11, 8.5)) # Standard landscape
                    ax.imshow(img)
                    ax.axis('off')
                    
                    # Extract type from filename for title
                    filename = item.get('name', 'Snapshot')
                    title = f"Trendlyne Snapshot - {filename}"
                    ax.set_title(title, fontsize=12, color='#475569', pad=5)
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
            except Exception as e:
                print(f"    Error rendering image {item.get('name')}: {e}")

    except Exception as e:
        print(f"    Error in Trendlyne Snapshots: {e}")


# Backward compatibility wrappers (if needed)
def render_leading_indicators(pdf, ticker):
    """Warning: Deprecated. Use render_technical_indicators instead."""
    render_volatility_squeeze(pdf, ticker)
    render_rsi_volume_divergence(pdf, ticker)

def render_lagging_indicators(pdf, ticker):
    """Warning: Deprecated. Use render_technical_indicators instead."""
    render_ema_crossover(pdf, ticker)
    render_bollinger_bands(pdf, ticker)
    render_supertrend(pdf, ticker)
    render_macd(pdf, ticker)


def generate_stock_report(ticker, output_dir=None):
    """
    Generate comprehensive PDF report for a stock.
    
    Args:
        ticker: Stock symbol (e.g., "HDFCBANK.NS")
        output_dir: Optional output directory (default: batch_reports/reports)
    
    Returns:
        Path to generated PDF file
    """
    print(f"\n{'='*60}")
    print(f"Generating Stock Detailed Report for {ticker}")
    print(f"{'='*60}\n")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_dir is None:
        # Default to ../reports relative to this script
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
        # Ensure reports directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    report_filename = os.path.join(output_dir, f'Stock_Report_{ticker.replace(".", "_")}_{timestamp}.pdf')
    
    # Fetch data once for all components
    df = fetch_shared_data(ticker)

    with PdfPages(report_filename) as pdf:
        # 1. Title Page (Must be first)
        # We need current price for title page, so we run price chart logic first but save plot later
        current_price = add_price_chart(pdf, ticker, save_plot=False, df=df)
        create_title_page(pdf, ticker, current_price)
        
        # 2. Price Chart (Add to PDF now)
        add_price_chart(pdf, ticker, save_plot=True, df=df)
        
        # 3. Fundamental Analysis
        render_fundamentals_page(pdf, ticker)
        
        # 4. Relative Strength Analysis
        render_rs_analysis(pdf, ticker, df=df)
        
        # 5. Technical Indicators (New Order)
        # Order: EMA -> BB -> Supertrend -> MACD -> Vol Squeeze -> RSI-Vol
        render_technical_indicators(pdf, ticker, df=df)

        # 5b. Multi-Timeframe Analysis
        # Pass df if multi_timeframe_analysis supports it, otherwise it fetches internals
        render_multi_timeframe_analysis(pdf, ticker, df=df)
        
        # 6. Trendlyne Snapshots
        render_trendlyne_snapshots(pdf, ticker)
    
    print(f"\n{'='*60}")
    print(f"Report generated successfully: {report_filename}")
    print(f"{'='*60}\n")
    
    return report_filename


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stock_detailed_report.py <TICKER>")
        print("Example: python stock_detailed_report.py HDFCBANK.NS")
        sys.exit(1)
    
    ticker = sys.argv[1]
    pdf_path = generate_stock_report(ticker)
