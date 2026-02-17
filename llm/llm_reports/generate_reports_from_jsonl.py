# -------------------------------------------------------------------------------
# Project: Stock Analysis (https://github.com/anshulsolanki/stock_analysis)
# Author:  Anshul Solanki
# License: MIT License
# 
# DISCLAIMER: 
# This software is for educational purposes only. It is not financial advice.
# Stock trading involves risks. The author is not responsible for any losses.
# -------------------------------------------------------------------------------

"""
Copyright (c) 2026 Anshul Solanki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import pandas as pd
from fpdf import FPDF
import json
import os
import datetime
import argparse

def load_jsonl(file_path):
    """Loads data from a JSONL file."""
    results = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    results.append(data)
                except:
                    pass
    return results

def create_pdf_report(df, output_path):
    """Creates a PDF report with a table layout."""
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Trade Analysis Results - All Recommendations", ln=True, align='C')
    pdf.ln(5)
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    # Define column widths
    w_stock = 45 # Increased slightly
    w_rec = 25   # New column for Recommendation
    w_score = 15
    w_reason = 190 # Adjusted
    
    # Header
    pdf.cell(w_stock, 10, "Stock", border=1, align='C')
    pdf.cell(w_rec, 10, "Rec.", border=1, align='C')
    pdf.cell(w_score, 10, "Score", border=1, align='C')
    pdf.cell(w_reason, 10, "Reasoning", border=1, align='C')
    pdf.ln()
    
    # Helper to calculate lines
    def get_lines(pdf, text, width):
        lines = 0
        for paragraph in text.split('\n'):
            words = paragraph.split()
            current_line = ""
            for word in words:
                # Check width with space
                test_line = current_line + word + " "
                if pdf.get_string_width(test_line) < width:
                    current_line = test_line
                else:
                    lines += 1
                    current_line = word + " "
            if current_line:
                lines += 1
        return max(1, lines)

    # Rows
    pdf.set_font("Arial", size=9)
    for _, row in df.iterrows():
        stock = str(row.get('stock_name', 'Unknown'))[:25]
        rec = str(row.get('recommendation', ''))
        score = str(row.get('confidence_score', 0))
        reason = str(row.get('reasoning', ''))
        
        # Calculate needed height
        num_lines = get_lines(pdf, reason, w_reason)
        # Assuming 5mm per line
        row_height = num_lines * 5 
        
        # Check space
        # A4 Landscape height ~210mm. Bottom margin ~20mm => limit 190mm
        if pdf.get_y() + row_height > 190:
            pdf.add_page()
            # Reprint header
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(w_stock, 10, "Stock", border=1, align='C')
            pdf.cell(w_rec, 10, "Rec.", border=1, align='C')
            pdf.cell(w_score, 10, "Score", border=1, align='C')
            pdf.cell(w_reason, 10, "Reasoning", border=1, align='C')
            pdf.ln()
            pdf.set_font("Arial", size=9)

        # Color coding for Recommendation
        if rec.upper() == 'BUY':
            pdf.set_text_color(0, 100, 0) # Dark Green
        elif rec.upper() == 'SELL':
            pdf.set_text_color(150, 0, 0) # Dark Red
        else:
            pdf.set_text_color(0, 0, 0)   # Black

        # Get start coordinates
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        
        # Print Reasoning first to be safe or Columns first?
        # Actually standard cells don't wrap, so they take fixed height.
        # We want all to have 'row_height'.
        # But multi_cell height might vary slightly if our calc is off.
        # Safer: Print MultiCell, get actual y_end, then print others.
        # Since we ensured it fits on page, we can print MultiCell safely.
        
        pdf.set_xy(x_start + w_stock + w_rec + w_score, y_start)
        pdf.set_text_color(0, 0, 0) # Reset to black for long text
        pdf.multi_cell(w_reason, 5, reason, border=1, align='L')
        y_end = pdf.get_y()
        row_height = y_end - y_start
        
        # Restore color for the single line columns
        if rec.upper() == 'BUY':
            pdf.set_text_color(0, 100, 0)
        elif rec.upper() == 'SELL':
            pdf.set_text_color(150, 0, 0)
        else:
            pdf.set_text_color(0, 0, 0)

        # Print other cells
        pdf.set_xy(x_start, y_start)
        pdf.cell(w_stock, row_height, stock, border=1, align='L')
        
        pdf.set_xy(x_start + w_stock, y_start)
        pdf.cell(w_rec, row_height, rec, border=1, align='C')
        
        pdf.set_xy(x_start + w_stock + w_rec, y_start)
        pdf.cell(w_score, row_height, score, border=1, align='C')
        
        # Move cursor to next row
        pdf.set_text_color(0, 0, 0) # Reset color
        pdf.set_xy(x_start, y_end)
        
        # Page break check
        if pdf.get_y() > 190:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(w_stock, 10, "Stock", border=1, align='C')
            pdf.cell(w_rec, 10, "Rec.", border=1, align='C')
            pdf.cell(w_score, 10, "Score", border=1, align='C')
            pdf.cell(w_reason, 10, "Reasoning", border=1, align='C')
            pdf.ln()
            pdf.set_font("Arial", size=9)
    try:
        pdf.output(output_path)
        print(f"Saved PDF: {output_path}")
    except Exception as e:
        print(f"Failed to save PDF: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_file", help="Path to input JSONL file")
    parser.add_argument("output_dir", help="Directory to save reports")
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_file):
        print(f"Error: JSONL file not found at {args.jsonl_file}")
        return
        
    results = load_jsonl(args.jsonl_file)
    if not results:
        print("No results found in JSONL file.")
        return
        
    df = pd.DataFrame(results)
    
    # Include ALL recommendations
    # Just select relevant columns
    cols = ['stock_name', 'recommendation', 'confidence_score', 'reasoning']
    # Ensure columns exist
    cols = [c for c in cols if c in df.columns]
    export_df = df[cols].copy()
    
    # Sort by Recommendation (BUY first) then Confidence Score?
    # Or just Confidence Score? User wants to see all.
    # Let's sort by Confidence Score desc.
    if 'confidence_score' in export_df.columns:
        export_df = export_df.sort_values(by='confidence_score', ascending=False)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Excel
    xlsx_path = os.path.join(args.output_dir, f"trade_analysis_report_{timestamp}.xlsx")
    try:
        export_df.to_excel(xlsx_path, index=False) 
        print(f"Saved Excel: {xlsx_path}")
    except Exception as e:
        print(f"Failed to save Excel: {e}")
        # Fallback to CSV
        csv_path = os.path.join(args.output_dir, f"trade_analysis_report_{timestamp}.csv")
        export_df.to_csv(csv_path, index=False)
        print(f"Fallback: Saved CSV: {csv_path}")

    # Save PDF
    pdf_path = os.path.join(args.output_dir, f"trade_analysis_report_{timestamp}.pdf")
    create_pdf_report(export_df, pdf_path)

if __name__ == "__main__":
    main()
