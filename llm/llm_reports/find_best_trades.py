"""
Stock Trade Analysis & Screening Tool

This script automates the screening of multiple stock trend reports to identify high-probability trade setups.
It uses Google's Gemini models to analyze PDF reports in parallel and generates a ranked summary of recommendations.

Key Features:
-   **Parallel Processing**: Analyzes multiple PDFs concurrently for high throughput.
-   **Resume Capability**: Skips already processed files to handle interruptions gracefully.
-   **Structured Output**: Generates a JSONL log, an Excel spreadsheet, and a formatted PDF summary.
-   **Ranking System**: Filters and ranks 'BUY' recommendations based on a confidence score (1-10).

Usage:
    python find_best_trades.py <target_directory> [--model <model_name>] [--workers <num_workers>]

Requirements:
    - GEMINI_API_KEY environment variable.
    - Python dependencies: google-genai, pandas, fpdf2, openpyxl.
"""

import os
import time
import json
import argparse
# import google.generativeai as genai # Deprecated
from google import genai
from google.genai import types
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
DEFAULT_MODEL_NAME = "gemini-3-flash-preview"

def get_api_key():
    """
    Retrieves the Gemini API key from the environment or a local .env file.

    It first checks `os.environ`. If not found, it attempts to parse a `.env` file 
    located in the same directory as the script.

    Returns:
        str or None: The API key if found, otherwise None.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('GEMINI_API_KEY='):
                            api_key = line.strip().split('=', 1)[1].strip().strip("'").strip('"')
                            os.environ['GEMINI_API_KEY'] = api_key
                            break
        except Exception:
            pass
    return api_key

def upload_to_gemini(client, path, mime_type="application/pdf"):
    """
    Uploads a file to the Gemini API.

    Args:
        client (genai.Client): Authenticated Gemini client.
        path (str): Local path to the file.
        mime_type (str, optional): MIME type of the file. Defaults to "application/pdf".

    Returns:
        types.File or None: The uploaded file object, or None if upload fails or file doesn't exist.
    """
    if not os.path.exists(path):
        return None
    try:
        # V2 SDK upload
        file = client.files.upload(file=path, config={'mime_type': mime_type})
        return file
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

def wait_for_file_active(client, file):
    """
    Polls the file status until it becomes active or fails.

    Args:
        client (genai.Client): Authenticated Gemini client.
        file (types.File): The file object to monitor.

    Returns:
        bool: True if the file becomes 'ACTIVE', False otherwise.
    """
    try:
        while file.state.name == "PROCESSING":
            time.sleep(1)
            file = client.files.get(name=file.name)
        return file.state.name == "ACTIVE"
    except Exception:
        return False

def analyze_report(client, model_name, file, filename):
    """
    Sends the file to Gemini for strategic trade analysis.

    Uses a specific prompt to extract a structured JSON recommendation (BUY/SELL/HOLD),
    confidence score, and reasoning from the stock report.

    Args:
        client (genai.Client): Authenticated Gemini client.
        model_name (str): Name of the Gemini model to use.
        file (types.File): The uploaded file object.
        filename (str): Name of the file (used for error reporting).

    Returns:
        dict: A dictionary containing the analysis result (stock_name, recommendation, etc.)
              or an error message.
    """
    prompt = """
    You are a Senior Portfolio Manager. Analyze the attached stock report (focusing on the first 3 pages and the "Formulate the Recommendation" section).

    Determine the final recommendation.
    Return ONLY a valid JSON object with the following fields:
    {
      "stock_name": "Name of the stock",
      "recommendation": "BUY" | "SELL" | "HOLD",
      "confidence_score": <integer 1-10, where 10 is the strongest conviction>,
      "reasoning": "A detailed explanation (2-5 sentences) of why this is a good/bad setup, citing specific technical details from the report."
    }
    """
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[file, prompt],
            config={
                'response_mime_type': 'application/json'
            }
        )
        result = json.loads(response.text)
        if isinstance(result, list):
            result = result[0]
        return result
    except Exception as e:
        return {"error": str(e), "stock_name": filename}

def load_processed_files(results_file):
    """
    Loads the state of previously processed files to enable resuming.

    Args:
        results_file (str): Path to the JSONL results file.

    Returns:
        tuple: A set of processed filenames and a list of existing result dictionaries.
    """
    processed = set()
    results = []
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'filename' in data:
                        processed.add(data['filename'])
                    results.append(data)
                except:
                    pass
    return processed, results

def save_result(result, results_file):
    with open(results_file, 'a') as f:
        f.write(json.dumps(result) + "\n")

def print_table(data):
    """
    Prints a formatted summary table of analysis results to the console.

    Args:
        data (list): List of analysis result dictionaries.
    """
    if not data:
        return

    # Columns: Rank, Stock, Score, Reasoning
    w_rank = 6
    w_stock = 15
    w_score = 8
    
    print("\n" + "="*120)
    print(f"{'Rank':<{w_rank}} | {'Stock':<{w_stock}} | {'Score':<{w_score}} | {'Reasoning'}")
    print("-" * 120)

    for i, res in enumerate(data, 1):
        score = res.get('confidence_score', 0)
        score_str = f"{score}/10"
        stock = res.get('stock_name', 'Unknown')[:w_stock-1]
        reason = res.get('reasoning', '').replace('\n', ' ')
        print(f"{i:<{w_rank}} | {stock:<{w_stock}} | {score_str:<{w_score}} | {reason}")
    
    print("="*120 + "\n")

def process_file_wrapper(args):
    """
    Worker function for processing a single file in a thread.

    Handles the full lifecycle: Client setup -> Upload -> Wait -> Analyze -> Cleanup.
    
    Args:
        args (tuple): A tuple containing (filename, target_dir, model_name).

    Returns:
        dict: The analysis result or an error dictionary.
    """
    filename, target_dir, model_name = args
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    file_path = os.path.join(target_dir, filename)
    
    # Upload
    pdf_file = upload_to_gemini(client, file_path)
    if not pdf_file:
        return {"error": "Upload Failed", "filename": filename}
        
    # Wait
    if not wait_for_file_active(client, pdf_file):
        return {"error": "Processing Failed", "filename": filename}
        
    # Analyze
    result = analyze_report(client, model_name, pdf_file, filename)
    
    # Cleanup (V2 uses client.files.delete or similar? Check docs. Usually client.files.delete(name=...))
    try:
        client.files.delete(name=pdf_file.name)
    except:
        pass
        
    return result

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

def sanitize_text(text):
    """
    Sanitizes text for PDF generation, removing incompatible Unicode characters.
    
    Replaces common special characters (dashes, quotes) with Latin-1 compatible equivalents
    and forcefully encodes/decodes to remove remaining unsupported characters.

    Args:
        text (str): Input text.

    Returns:
        str: Sanitized text safe for fpdf2 (Latin-1).
    """
    if not isinstance(text, str):
        return str(text)
    replacements = {
        '\u2013': '-',   # en-dash
        '\u2014': '--',  # em-dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...', # ellipsis
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Fallback for any other non-latin-1 characters
    return text.encode('latin-1', 'replace').decode('latin-1')

def save_to_formats(results, output_dir):
    """
    Exports the aggregated results to Excel and PDF formats.

    Creates a timestamped Excel file and a formatted PDF report with color-coded
    recommendations (Green for BUY, Red for SELL).

    Args:
        results (list): List of analysis result dictionaries.
        output_dir (str): Directory where output files will be saved.
    """
    if not results:
        return
        
    df = pd.DataFrame(results)
    
    # Select and rename columns for clarity
    cols = ['stock_name', 'recommendation', 'confidence_score', 'reasoning']
    # Filter for existing columns only
    cols = [c for c in cols if c in df.columns]
    
    df_export = df[cols].copy()
    if 'confidence_score' in df_export.columns:
        df_export = df_export.sort_values(by='confidence_score', ascending=False)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Excel
    xlsx_path = os.path.join(output_dir, f"trade_analysis_results_{timestamp}.xlsx")
    try:
        df_export.to_excel(xlsx_path, index=False)
        print(f"Saved Excel: {xlsx_path}")
    except Exception as e:
        print(f"Failed to save Excel: {e}")

    # PDF
    pdf_path = os.path.join(output_dir, f"trade_analysis_results_{timestamp}.pdf")
    try:
        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 10, "Trade Analysis Results - All Recommendations", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(5)
        
        # Table Header
        pdf.set_font("Helvetica", 'B', 10)
        # Define column widths
        w_stock = 45
        w_rec = 25
        w_score = 15
        w_reason = 190
        
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
        pdf.set_font("Helvetica", size=9)
        for _, row in df_export.iterrows():
            stock = sanitize_text(row.get('stock_name', 'Unknown'))[:25]
            rec = sanitize_text(row.get('recommendation', ''))
            score = sanitize_text(row.get('confidence_score', 0))
            reason = sanitize_text(row.get('reasoning', ''))
            
            # Calculate needed height
            num_lines = get_lines(pdf, reason, w_reason)
            row_height = num_lines * 5 
            
            # Check space (A4 Landscape height ~210mm. Bottom margin ~20mm => limit 190mm)
            if pdf.get_y() + row_height > 190:
                pdf.add_page()
                # Reprint header
                pdf.set_font("Helvetica", 'B', 10)
                pdf.cell(w_stock, 10, "Stock", border=1, align='C')
                pdf.cell(w_rec, 10, "Rec.", border=1, align='C')
                pdf.cell(w_score, 10, "Score", border=1, align='C')
                pdf.cell(w_reason, 10, "Reasoning", border=1, align='C')
                pdf.ln()
                pdf.set_font("Helvetica", size=9)

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
            
            # Print Reasoning first to measure height/print
            pdf.set_xy(x_start + w_stock + w_rec + w_score, y_start)
            pdf.set_text_color(0, 0, 0) # Reset to black for long text
            pdf.multi_cell(w_reason, 5, reason, border=1, align='L')
            
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
            pdf.set_xy(x_start, y_start + row_height)
            
        pdf.output(pdf_path)
        print(f"Saved PDF: {pdf_path}")
    except Exception as e:
        print(f"Failed to save PDF: {e}")

def main():
    """
    Main execution entry point.

    Parses command-line arguments, sets up the worker pool, and orchestrates 
    the batch analysis of PDF reports.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", help="Directory containing PDF reports")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Gemini model name")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    args = parser.parse_args()

    if not get_api_key():
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    target_dir = args.target_dir
    if not os.path.exists(target_dir):
        print(f"Error: Target directory not found: {target_dir}")
        return

    files = [f for f in os.listdir(target_dir) if f.lower().endswith('.pdf')]
    if not files:
        print(f"No PDF files found in {target_dir}")
        return

    results_file = os.path.join(target_dir, "trade_analysis_results.jsonl")

    # Load Resume State
    processed_files, all_results = load_processed_files(results_file)
    files_to_process = [f for f in files if f not in processed_files]
    
    print(f"Found {len(files)} reports. {len(processed_files)} already processed. {len(files_to_process)} to go.")
    print(f"Analysis with {args.model} using {args.workers} workers...\n")

    if files_to_process:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            task_args = [(f, target_dir, args.model) for f in files_to_process]
            future_to_file = {executor.submit(process_file_wrapper, arg): arg[0] for arg in task_args}
            
            completed = 0
            total = len(files_to_process)
            
            for future in as_completed(future_to_file):
                completed += 1
                filename = future_to_file[future]
                try:
                    result = future.result()
                    if "error" not in result:
                        result['filename'] = filename
                        save_result(result, results_file)
                        all_results.append(result)
                        print(f"[{completed}/{total}] {filename[:30]}... Done")
                    else:
                        print(f"[{completed}/{total}] {filename[:30]}... Failed: {result['error']}")
                except Exception as exc:
                    print(f"[{completed}/{total}] {filename[:30]}... Error: {exc}")

    print("\nAnalysis complete.")

    # Filter and Rank
    buy_recommendations = [r for r in all_results if r.get('recommendation', '').upper() == 'BUY']
    buy_recommendations.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
    
    # Save all results to XLSX/PDF
    # User said "in same directory" -> likely same as where results are saved or target_dir.
    # Let's save to target_dir to keep it together with reports.
    # Actually, saving to current directory is safer if target_dir is read-only or messy.
    # But usually "same directory" implies with the reports.
    # Let's use target_dir.
    print(f"\nSaving detailed results to {target_dir}...")
    save_to_formats(all_results, target_dir)

if __name__ == "__main__":
    main()
