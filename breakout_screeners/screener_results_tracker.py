import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = os.path.dirname(__file__)
SCREENER_RESULTS_DIR = os.path.join(BASE_DIR, 'screener_results')

SCREENERS = ['CAMSLIM_breakouts', 'fw_breakouts', 'Minervini_breakouts', 'qullamaggie_breakouts', 'darvas_boxes']
OUTPUT_EXCEL = os.path.join(SCREENER_RESULTS_DIR, 'consolidated_results.xlsx')
OUTPUT_PDF = os.path.join(SCREENER_RESULTS_DIR, 'consolidated_results.pdf')

def get_results():
    data = {} # date -> {screener -> [tickers]}
    
    for screener in SCREENERS:
        screener_path = os.path.join(SCREENER_RESULTS_DIR, screener)
        if not os.path.exists(screener_path):
            print(f"Directory not found: {screener_path}")
            continue
            
        for ts_folder in os.listdir(screener_path):
            if ts_folder.startswith('.'):
                continue
            folder_path = os.path.join(screener_path, ts_folder)
            if not os.path.isdir(folder_path):
                continue
                
            csv_path = os.path.join(folder_path, 'results.csv')
            if not os.path.exists(csv_path):
                continue
                
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                
                # We assume column 'Ticker' exists
                if 'Ticker' not in df.columns:
                    print(f"Missing Ticker column in {csv_path}")
                    continue
                    
                # Extract date from folder name (YYYY-MM-DD)
                folder_date = ts_folder.split('_')[0]
                for _, row in df.iterrows():
                    date = folder_date
                    ticker = row['Ticker']
                    
                    if date not in data:
                        data[date] = {s: [] for s in SCREENERS}
                    
                    if ticker not in data[date][screener]:
                        data[date][screener].append(ticker)
                        
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                
    return data

def create_dataframe(data):
    rows = []
    for date in sorted(data.keys()):
        row = {'Date': date}
        for screener in SCREENERS:
            tickers = data[date][screener]
            # Map folder names to requested column names
            if screener == 'CAMSLIM_breakouts':
                col_name = 'CAMSLIM'
            elif screener == 'fw_breakouts':
                col_name = 'fw'
            elif screener == 'Minervini_breakouts':
                col_name = 'Minervini'
            elif screener == 'qullamaggie_breakouts':
                col_name = 'Qullamaggie'
            elif screener == 'darvas_boxes':
                col_name = 'Darvas'
            else:
                col_name = screener
                
            row[col_name] = ', '.join(tickers) if tickers else 'NA'
        rows.append(row)
        
    df = pd.DataFrame(rows)
    # Ensure column order
    columns = ['Date', 'CAMSLIM', 'fw', 'Minervini', 'Qullamaggie', 'Darvas']
    # If any column is missing, add it with 'NA'
    for col in columns:
        if col not in df.columns:
            df[col] = 'NA'
    return df[columns]

def render_pdf_styled_table(pdf, df, title):
    if df.empty:
        return
    # Calculate line count for each row and group into pages
    pages = []
    current_page = []
    current_lines = 0
    MAX_LINES_PER_PAGE = 43

    for idx, row in df.iterrows():
        h = 1
        for col in df.columns:
            if col == 'Date':
                continue
            val = str(row[col])
            if val and val != 'NA':
                lines = len(val.split(', '))
                if lines > h:
                    h = lines

        if current_lines + h > MAX_LINES_PER_PAGE and current_page:
            pages.append(pd.DataFrame(current_page))
            current_page = []
            current_lines = 0

        current_page.append(row)
        current_lines += h

    if current_page:
        pages.append(pd.DataFrame(current_page))

    num_pages = len(pages)
    for i, chunk in enumerate(pages):
        # Replace comma with newline for PDF rendering to avoid overlap
        for col in chunk.columns:
            if col == 'Date':
                continue
            chunk[col] = chunk[col].astype(str).str.replace(', ', '\n')
        
        fig = plt.figure(figsize=(14, 8.5)) 
        ax = fig.add_axes([0, 0.05, 1, 0.85])
        ax.axis('off')
        
        # Standard header
        PRIMARY_COLOR = '#1e293b'
        BORDER_COLOR = '#94a3b8'
        fig.text(0.5, 0.96, f"{title} - Page {i+1}/{num_pages}", ha='center', va='center', fontsize=16, weight='bold', color=PRIMARY_COLOR)
        copyright_text = f"© {datetime.now().year} Stock Research. All Rights Reserved."
        fig.text(0.5, 0.935, copyright_text, ha='center', va='center', fontsize=9, style='italic', color=BORDER_COLOR)
        from matplotlib.lines import Line2D
        line = Line2D([0.08, 0.92], [0.91, 0.91], transform=fig.transFigure, color=BORDER_COLOR, linewidth=1.0, alpha=0.5)
        fig.add_artist(line)
        
        table = ax.table(cellText=chunk.values, colLabels=chunk.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8.5) 
        
        # Dynamically calculate row heights based on number of lines
        row_max_lines = {}
        for (row, col), cell in table.get_celld().items():
            if row > 0: # Skip header
                text = cell.get_text().get_text()
                lines = len(text.split('\n'))
                if row not in row_max_lines:
                    row_max_lines[row] = 1
                if lines > row_max_lines[row]:
                    row_max_lines[row] = lines
        
        # Get default height from a sample cell (assumes at least one row exists)
        sample_cell = table.get_celld().get((1, 0))
        if sample_cell:
            default_height = sample_cell.get_height()
            
            # Set height for each cell based on row's max lines
            for (row, col), cell in table.get_celld().items():
                if row > 0:
                    cell.set_height(default_height * row_max_lines[row])        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white', fontsize=8.5)
                cell.set_facecolor('#1e293b')
            else:
                cell.set_linewidth(0.3)
                cell.set_facecolor('#f8fafc' if row % 2 == 0 else 'white')
                if col == 0:
                     cell.set_text_props(weight='bold', color='#2563eb')
        pdf.savefig(fig)
        plt.close(fig)

def main():
    print("Scanning results...")
    data = get_results()
    if not data:
        print("No data found.")
        return
        
    df = create_dataframe(data)
    print(f"Found results for {len(df)} unique dates.")
    
    # Save Excel
    try:
        df.to_excel(OUTPUT_EXCEL, index=False)
        print(f"Saved Excel to {OUTPUT_EXCEL}")
    except Exception as e:
        print(f"Error saving Excel: {e}")
        # Try fallback to CSV if Excel fails
        csv_fallback = OUTPUT_EXCEL.replace('.xlsx', '.csv')
        df.to_csv(csv_fallback, index=False)
        print(f"Fallback: Saved CSV to {csv_fallback}")
        
    # Save PDF
    try:
        with PdfPages(OUTPUT_PDF) as pdf:
            render_pdf_styled_table(pdf, df, "Consolidated Screener Results")
        print(f"Saved PDF to {OUTPUT_PDF}")
    except Exception as e:
        print(f"Error saving PDF: {e}")

if __name__ == "__main__":
    main()
