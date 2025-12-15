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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import yfinance as yf

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fundamental_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lagging_indicator_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'leading_indicator_analysis'))

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


def add_price_chart(pdf, ticker, save_plot=True):
    """Fetches and plots 3-year daily price chart."""
    try:
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
        
        # Determine color based on value
        color = '#1e293b'  # Default
        if isinstance(value, str):
            value_lower = str(value).lower()
            if any(word in value_lower for word in ['bullish', 'buy', 'uptrend', 'positive', 'strengthening']):
                color = '#16a34a'  # Green
            elif any(word in value_lower for word in ['bearish', 'sell', 'downtrend', 'negative', 'weakening']):
                color = '#ef4444'  # Red
        
        fig.text(0.35, y_position, str(value), ha='left', va='top',
                 fontsize=12, color=color)
        
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


def render_leading_indicators(pdf, ticker):
    """Calls existing leading indicator modules and adds charts."""
    print("  Running Leading Indicator Analyses...")
    
    # RSI Divergence - REMOVED as per request (keeping only RSI-Volume)
    
    # RSI-Volume Divergence
    try:
        print("    - RSI-Volume Divergence")
        res = rsi_volume_divergence.run_analysis(ticker, show_plot=False)
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
    
    # Volatility Squeeze
    try:
        print("    - Volatility Squeeze")
        res = volatility_squeeze_analysis.run_analysis(ticker, show_plot=False)
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


def render_lagging_indicators(pdf, ticker):
    """Calls existing lagging indicator modules and adds charts."""
    print("  Running Lagging Indicator Analyses...")
    
    # MACD
    try:
        print("    - MACD")
        res = macd_analysis.run_analysis(ticker, show_plot=False)
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
    
    # Supertrend
    try:
        print("    - Supertrend")
        res = supertrend_analysis.run_analysis(ticker, show_plot=False)
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
    
    # Bollinger Bands
    try:
        print("    - Bollinger Bands")
        res = bollinger_band_analysis.run_analysis(ticker, show_plot=False)
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
    
    # EMA Crossover
    try:
        print("    - EMA Crossover")
        res = crossover_analysis.run_analysis(ticker, show_plot=False)
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
    
    except Exception as e:
        print(f"    Error in Crossover: {e}")


def generate_stock_report(ticker, output_dir=None):
    """
    Generate comprehensive PDF report for a stock.
    
    Args:
        ticker: Stock symbol (e.g., "HDFCBANK.NS")
        output_dir: Optional output directory (default: current directory)
    
    Returns:
        Path to generated PDF file
    """
    print(f"\n{'='*60}")
    print(f"Generating Stock Detailed Report for {ticker}")
    print(f"{'='*60}\n")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    report_filename = os.path.join(output_dir, f'Stock_Report_{ticker.replace(".", "_")}_{timestamp}.pdf')
    
    with PdfPages(report_filename) as pdf:
        # 1. Title Page (Must be first)
        # We need current price for title page, so we run price chart logic first but save plot later
        current_price = add_price_chart(pdf, ticker, save_plot=False)
        create_title_page(pdf, ticker, current_price)
        
        # 2. Price Chart (Add to PDF now)
        add_price_chart(pdf, ticker, save_plot=True)
        
        # 2. Fundamental Analysis
        render_fundamentals_page(pdf, ticker)
        
        # 3. Leading Indicators
        render_leading_indicators(pdf, ticker)
        
        # 4. Lagging Indicators
        render_lagging_indicators(pdf, ticker)
    
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
