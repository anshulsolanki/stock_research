"""
WEEKLY ANALYSIS REPORT GENERATOR
================================

Generates a comprehensive PDF report containing:
1. Sector Analysis
2. Stocks in Sector Analysis (Top Sectors + Emerging)
3. Batch Analysis of Top Stocks
4. Deepdive Analysis of Individual Stocks
"""

import sys
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

# Add parent directories to path
# Scripts are now in batch_reports/scripts, so we need to go up two levels to reach root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'market_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'fundamental_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lagging_indicator_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'leading_indicator_analysis'))

# Import Analysis Modules
try:
    import sector_analysis
    import stock_in_sector_analysis
    import batch_analysis
    import fundamental_analysis
    import macd_analysis
    import supertrend_analysis
    import bollinger_band_analysis
    import crossover_analysis
    import donchian_channel_analysis
    import rsi_divergence_analysis
    import rsi_volume_divergence
    import volatility_squeeze_analysis
    # Import stock_detailed_report for deep dive formatting
    import stock_detailed_report
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def create_title_page(pdf, title, subtitle=""):
    """Creates a simple title page."""
    plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    plt.text(0.5, 0.6, title, ha='center', va='center', fontsize=24, weight='bold')
    plt.text(0.5, 0.4, subtitle, ha='center', va='center', fontsize=14)
    plt.text(0.5, 0.2, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', va='center', fontsize=10)
    pdf.savefig()
    plt.close()

def render_table_page(pdf, df, title, col_widths=None):
    """Renders a dataframe as a table on a PDF page."""
    if df.empty:
        return

    # Split into chunks if too many rows (approx 25 rows per page)
    rows_per_page = 25
    num_pages = (len(df) // rows_per_page) + 1
    
    for i in range(num_pages):
        start_idx = i * rows_per_page
        end_idx = min((i + 1) * rows_per_page, len(df))
        chunk = df.iloc[start_idx:end_idx]
        
        if chunk.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        ax.set_title(f"{title} (Page {i+1}/{num_pages})", fontsize=16, pad=20)
        
        # Create table
        table = ax.table(cellText=chunk.values, colLabels=chunk.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)
        
        pdf.savefig(fig)
        plt.close(fig)

def render_styled_table_page(pdf, df, title, col_formats=None):
    """
    Renders a dataframe as a styled table with UI-like elements.
    
    Args:
        col_formats: Dict mapping column names to style types:
            - 'score': Blue background, white text
            - 'bool': Green check for True, ' - ' for False
            - 'float': Round to 2 decimals
            - 'rs': Red/Green text based on value > 0
            - 'trend': Color coded text for trend (Uptrend=Green, Downtrend=Red)
            - 'signal': Color coded text for signal (Buy=Green, Sell=Red)
    """
    if df.empty:
        return

    # Default styling if not provided
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
            
        # Pre-process data for display (icons, rounding)
        display_chunk = chunk.copy()
        
        # Helper to replace unsupported emojis with supported shapes
        def replace_icons(val):
            if isinstance(val, str):
                # Replace common emojis with supported geometric shapes
                # Circles: 🟢 🔴 🟡 -> ● (U+25CF)
                val = val.replace('🟢', '●').replace('🔴', '●').replace('🟡', '●') 
                # Arrows: ⬆️ -> ▲, ⬇️ -> ▼
                val = val.replace('⬆️', '▲').replace('⬇️', '▼')
                # Others
                val = val.replace('➖', '-').replace('⚠️', '!').replace('💥', '*')
                return val.strip()
            return val

        for col in chunk.columns:
            # First apply icon replacement to all
            display_chunk[col] = display_chunk[col].apply(replace_icons)

            fmt = col_formats.get(col)
            if fmt == 'bool':
                # Replace with symbols
                display_chunk[col] = chunk[col].apply(lambda x: '✔' if x else '-')
            elif fmt == 'float' or fmt == 'rs':
                display_chunk[col] = chunk[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            elif fmt == 'score':
                # Round score
                display_chunk[col] = chunk[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        ax.set_title(f"{title} (Page {i+1}/{num_pages})", fontsize=16, weight='bold', pad=20, color='#1e293b')
        
        # Create table
        table = ax.table(cellText=display_chunk.values, colLabels=display_chunk.columns, loc='center', cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5) # More vertical padding
        
        # Access cells to style them
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                # Header Style
                cell.set_text_props(weight='bold', color='#475569')
                cell.set_facecolor('#f8fafc')
                cell.set_edgecolor('#e2e8f0')
                cell.set_linewidth(1)
            else:
                # Data Row Style
                cell.set_edgecolor('#e2e8f0')
                cell.set_linewidth(0.5)
                
                # Column specific styling
                col_name = display_chunk.columns[col]
                fmt = col_formats.get(col_name)
                val = chunk.iloc[row-1][col_name] # Original value for logic
                
                if fmt == 'score':
                     cell.set_facecolor('#2563eb') # Blue
                     cell.set_text_props(color='white', weight='bold')
                
                elif fmt == 'bool':
                    if val: # True -> Green check
                         cell.set_text_props(color='#16a34a', weight='bold') # Green
                    else:
                         cell.set_text_props(color='#94a3b8') # Grey
                         
                elif fmt == 'rs':
                    try:
                        if isinstance(val, (int, float)):
                            if val > 0:
                                cell.set_text_props(color='#16a34a', weight='bold') # Green
                            elif val < 0:
                                cell.set_text_props(color='#ef4444', weight='bold') # Red
                    except:
                        pass
                        
                elif fmt == 'trend':
                    val_str = str(val).lower()
                    if 'up' in val_str or 'bull' in val_str:
                        cell.set_text_props(color='#16a34a', weight='bold')
                    elif 'down' in val_str or 'bear' in val_str:
                        cell.set_text_props(color='#ef4444', weight='bold')
                        
                elif fmt == 'signal':
                    val_str = str(val).lower()
                    if 'buy' in val_str:
                        cell.set_text_props(color='#16a34a', weight='bold') # Green
                        cell.set_facecolor('#ecfdf5') # Light Green BG
                    elif 'sell' in val_str:
                        cell.set_text_props(color='#ef4444', weight='bold') # Red
                        cell.set_facecolor('#fef2f2') # Light Red BG

        pdf.savefig(fig)
        plt.close(fig)

def run_weekly_analysis(include_deepdive=False, output_dir=None):
    print("Starting Weekly Analysis Report Generation...")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define output directory
    if output_dir is None:
        # Default: batch_reports/reports
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    report_filename = os.path.join(output_dir, f'Weekly_Report_{timestamp}.pdf')
    
    with PdfPages(report_filename) as pdf:
        # Title Page
        create_title_page(pdf, "Weekly Market Analysis Report", "Comprehensive Technical & Fundamental Review")
        
        # ==================================================================================
        # SECTION 1: SECTOR ANALYSIS
        # ==================================================================================
        print("\n[Section 1] Running Sector Analysis...")
        sector_res = sector_analysis.run_analysis(show_plot=False)
        
        if sector_res['success']:
            # Save Plot
            if sector_res['figure']:
                pdf.savefig(sector_res['figure'])
                plt.close(sector_res['figure'])
            
            # Save Results Table
            results_df = sector_res['results']
            # Prepare Data for Table
            display_cols = ['Score', '1M', '3M', '6M', '1Y', 'Consistent', 'Emerging', 'Early_Turnaround', 'MA_Breakout', 'Volume_Surge']
            display_cols = [c for c in display_cols if c in results_df.columns]
            
            table_df = results_df[display_cols].reset_index().rename(columns={
                'index': 'SECTOR',
                'Score': 'SCORE',
                '1M': '1M RS',
                '3M': '3M RS',
                '6M': '6M RS',
                '1Y': '1Y RS',
                'Consistent': 'CONSISTENT',
                'Emerging': 'EMERGING',
                'Early_Turnaround': 'TURNAROUND',
                'MA_Breakout': 'MA BREAKOUT',
                'Volume_Surge': 'VOL SURGE'
            })
            
            # Formats
            sector_formats = {
                'SCORE': 'score',
                '1M RS': 'rs', '3M RS': 'rs', '6M RS': 'rs', '1Y RS': 'rs',
                'CONSISTENT': 'bool', 'EMERGING': 'bool', 'TURNAROUND': 'bool', 'MA BREAKOUT': 'bool', 'VOL SURGE': 'bool'
            }
            render_styled_table_page(pdf, table_df, "Sector Relative Strength Matrix", sector_formats)
            
            # Identify Top Sectors & Emerging
            top_sectors = results_df.sort_values(by='Score', ascending=False).head(3).index.tolist()
            emerging_sectors = results_df[results_df['Emerging'] == True].index.tolist()
            
            raw_sectors_to_analyze = list(set(top_sectors + emerging_sectors))
            print(f"  Selected Sectors (Raw): {raw_sectors_to_analyze}")
            
            # Map to config keys
            SECTOR_MAPPING = {
                "Bank Nifty": "Bank",
                "Nifty IT": "IT",
                "Nifty Pharma": "Pharma",
                "Nifty Auto": "Auto",
                "Nifty FMCG": "FMCG",
                "Nifty Metal": "Metal",
                "Nifty Realty": "Realty",
                "Nifty Infra": "Infra"
            }
            
            sectors_to_analyze = []
            for s in raw_sectors_to_analyze:
                if s in SECTOR_MAPPING:
                    sectors_to_analyze.append(SECTOR_MAPPING[s])
                elif s in SECTOR_MAPPING.values():
                    sectors_to_analyze.append(s) # Already correct
                else:
                    print(f"  Warning: Could not map sector '{s}' to a config key.")
            
            print(f"  Mapped Sectors: {sectors_to_analyze}")

        else:
            print(f"  Sector Analysis Failed: {sector_res.get('error')}")
            sectors_to_analyze = []

        # ==================================================================================
        # SECTION 2: STOCKS IN SECTOR
        # ==================================================================================
        print("\n[Section 2] Running Stocks in Sector Analysis...")
        
        all_top_stocks = []
        
        for sector in sectors_to_analyze:
            print(f"  Analyzing Sector: {sector}")
            sis_res = stock_in_sector_analysis.run_analysis(sector, show_plot=False)
            
            if sis_res['success']:
                # Save Plot
                if sis_res['figure']:
                    sis_res['figure'].suptitle(f"Sector: {sector} - relative Strength", fontsize=16)
                    pdf.savefig(sis_res['figure'])
                    plt.close(sis_res['figure'])
                
                # Save Table
                sis_df = sis_res['results']
                display_cols = ['Score', '1M', '3M', '6M', '1Y', 'Consistent', 'Emerging', 'Early_Turnaround', 'MA_Breakout', 'Volume_Surge']
                display_cols = [c for c in display_cols if c in sis_df.columns]
                
                table_df = sis_df[display_cols].reset_index().head(15).rename(columns={
                    'index': 'STOCK',
                    'Score': 'SCORE',
                    '1M': '1M RS',
                    '3M': '3M RS',
                    '6M': '6M RS',
                    '1Y': '1Y RS',
                    'Consistent': 'CONSISTENT',
                    'Emerging': 'EMERGING',
                    'Early_Turnaround': 'TURNAROUND',
                    'MA_Breakout': 'MA BREAKOUT',
                    'Volume_Surge': 'VOL SURGE'
                })
                
                sis_formats = {
                    'SCORE': 'score',
                    '1M RS': 'rs', '3M RS': 'rs', '6M RS': 'rs', '1Y RS': 'rs',
                    'CONSISTENT': 'bool', 'EMERGING': 'bool', 'TURNAROUND': 'bool', 'MA BREAKOUT': 'bool', 'VOL SURGE': 'bool'
                }
                render_styled_table_page(pdf, table_df, f"Top Stocks in {sector}", sis_formats)
                
                # Pick top 4
                top_4_names = sis_df.head(4).index.tolist()
                
                # We need to map Name -> Ticker
                # Load config again or use a helper if available, or just iterate to find it
                # Since we are iterating sectors, let's load the specific sector config to find tickers
                # But easiest is to load all configs at once at start of script.
                # For now, let's just re-load inside the loop or use a global map.
                
                # Quick load of tickers Grouped for mapping
                import json
                possible_paths = [
                    os.path.join(os.path.dirname(__file__), '..', 'data', 'tickers_grouped.json'),
                     '/Users/solankianshul/Documents/projects/stock_research/data/tickers_grouped.json'
                ]
                config_file = next((p for p in possible_paths if os.path.exists(p)), None)
                if config_file:
                    with open(config_file, 'r') as f:
                        full_config = json.load(f)
                    
                    if sector in full_config:
                        sector_stocks = full_config[sector]['stocks'] # Name -> Ticker
                        for name in top_4_names:
                            if name in sector_stocks:
                                all_top_stocks.append(sector_stocks[name])
                            else:
                                print(f"  Warning: Could not find ticker for {name}")
                else:
                    print("  Error: Config file not found, cannot map names to tickers.")

            else:
                 print(f"  Failed to analyze sector {sector}: {sis_res.get('error')}")

        # Unique Stocks for Deepdive
        final_stock_list = list(set(all_top_stocks))
        print(f"\n  Identified {len(final_stock_list)} unique tickers for deepdive: {final_stock_list}")

        # ==================================================================================
        # SECTION 3: BATCH ANALYSIS
        # ==================================================================================
        print("\n[Section 3] Running Batch Analysis...")
        
        if final_stock_list:
            batch_res = batch_analysis.run_batch_analysis(final_stock_list)
            
            # Convert to DataFrame for easier printing
            batch_data = []
            for item in batch_res:
                if item['success']:
                    cols = item['columns']
                    
                    row = {
                        'Ticker': item['ticker'],
                        'Score': item['score'],
                        'Price': item['price'],
                        'Trend': cols['trend_direction'],
                        'Signal': cols['trend_signal'],
                        'RSI': cols['rsi_value'],
                        'Squeeze': cols['squeeze'],
                        'RS Score': cols['rs_score']
                    }
                    batch_data.append(row)
            
            if batch_data:
                batch_df = pd.DataFrame(batch_data).sort_values(by='Score', ascending=False)
                
                batch_formats = {
                    'Score': 'score',
                    'Price': 'float',
                    'Trend': 'trend',
                    'Signal': 'signal',
                    'RSI': 'float',
                    'RS Score': 'rs'
                }
                render_styled_table_page(pdf, batch_df, "Batch Analysis Summary", batch_formats)
            else:
                print("  No successful batch analysis results.")
        else:
            print("  No stocks selected for batch analysis.")

        # ==================================================================================
        # SECTION 4: INDIVIDUAL DEEPDIVE
        # ==================================================================================
        if include_deepdive:
            print("\n[Section 4] Running Individual Deepdives...")
            
            for ticker in final_stock_list:
                print(f"  Deepdive for {ticker}...")
                
                # Use functions from stock_detailed_report.py for consistent formatting
                # 1. Title Page
                current_price = stock_detailed_report.add_price_chart(pdf, ticker, save_plot=False)
                stock_detailed_report.create_title_page(pdf, ticker, current_price)
                
                # 2. Price Chart
                stock_detailed_report.add_price_chart(pdf, ticker, save_plot=True)
                
                # 3. Fundamental Analysis
                stock_detailed_report.render_fundamentals_page(pdf, ticker)
                
                # 4. Technical Indicators (New Order)
                stock_detailed_report.render_technical_indicators(pdf, ticker)
        else:
            print("\n[Section 4] Individual Deepdives skipped (include_deepdive=False).")
                
    print(f"\nReport generated successfully: {report_filename}")
    
    return {
        'report_path': report_filename,
        'final_stock_list': final_stock_list
    }

if __name__ == "__main__":
    run_weekly_analysis(False)
