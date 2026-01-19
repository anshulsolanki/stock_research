import os
import time
import sys
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
import markdown
from fpdf import FPDF
from io import BytesIO
from pypdf import PdfWriter

# Load environment variables
load_dotenv()

# Configuration
#gemini-2.5-flash
#DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_MODEL_NAME = "gemini-3-pro-preview" 

def setup_gemini_client():
    """Configures and returns a Gemini Client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please export your API key or set it in a .env file.")
        sys.exit(1)
    return genai.Client(api_key=api_key)

def save_analysis_to_pdf(text, output_path):
    """
    Saves the analysis text as a PDF file using fpdf2.
    """
    # ... (Keep existing text processing logic) ...
    text = text.replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"').replace("₹", "INR ")
    
    text = text.replace("* **Data Present:**", "\n\n* **Data Present:**")
    text = text.replace("* **Data Missing:**", "\n\n* **Data Missing:**")
    text = text.replace("* **Confidence Level:**", "\n\n* **Confidence Level:**")
    text = text.replace("* **Strengths:**", "\n\n* **Strengths:**")
    text = text.replace("* **Weaknesses:**", "\n\n* **Weaknesses:**")
    text = text.replace("* **What is present:**", "\n\n* **What is present:**")
    text = text.replace("* **What is missing:**", "\n\n* **What is missing:**")
    
    text = text.replace("* **Entry Price:**", "\n\n* **Entry Price:**")
    text = text.replace("* **Stop Loss:**", "\n\n* **Stop Loss:**")
    text = text.replace("* **Take Profit:**", "\n\n* **Take Profit:**")
    text = text.replace("* **Rationale:**", "\n\n* **Rationale:**")
    text = text.replace("* **Condition to Buy:**", "\n\n* **Condition to Buy:**")
    
    replacements = {
        "**BUY**": "<b><font color='#27ae60'>BUY</font></b>",
        "**YES**": "<b><font color='#27ae60'>YES</font></b>",
        "**NO**": "<b><font color='#c0392b'>NO</font></b>",
        "**SELL**": "<b><font color='#c0392b'>SELL</font></b>",
        "Stage 4": "<b><font color='#c0392b'>Stage 4</font></b>",
        "Downtrend": "<b><font color='#c0392b'>Downtrend</font></b>",
        "Uptrend": "<b><font color='#27ae60'>Uptrend</font></b>",
        "Strong Momentum": "<b><font color='#27ae60'>Strong Momentum</font></b>",
        "Bearish": "<b><font color='#c0392b'>Bearish</font></b>",
        "Bullish": "<b><font color='#27ae60'>Bullish</font></b>",
        "Stop Loss": "<b><font color='#c0392b'>Stop Loss</font></b>",
        "Take Profit": "<b><font color='#27ae60'>Take Profit</font></b>"
    }
    
    for key, value in replacements.items():
        text = text.replace(key, value)

    html_content = markdown.markdown(text, extensions=['tables'])
    
    # Add table styling
    html_content = html_content.replace("<table>", '<table border="1" align="center" width="100%" style="border-collapse: collapse;">')
    html_content = html_content.replace("<th>", '<th style="background-color: #f0f0f0; padding: 5px; font-weight: bold;">')
    html_content = html_content.replace("<td>", '<td style="padding: 5px;">')
    
    class PDF(FPDF):
        def header(self):
            self.set_font("helvetica", "B", 16)
            self.set_text_color(44, 62, 80)
            title = "Gemini Stock Analysis Report"
            width = self.get_string_width(title) + 6
            self.set_x((210 - width) / 2)
            self.cell(width, 10, title, border=0, new_x="LMARGIN", new_y="NEXT", align="C")
            self.ln(5)
            self.set_draw_color(44, 62, 80)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            self.set_x(-50)
            self.cell(0, 10, timestamp, align="R")

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    pdf.set_text_color(51, 51, 51)
    
    html_content = html_content.replace("<p>", "<p style='margin-bottom: 10px; line-height: 1.6;'>")
    html_content = html_content.replace("<h1>", "<h1 style='color: #2c3e50; font-size: 18pt; margin-top: 20px; border-bottom: 1px solid #eee;'>")
    html_content = html_content.replace("<h2>", "<h2 style='color: #34495e; font-size: 14pt; margin-top: 15px;'>")
    html_content = html_content.replace("<h3>", "<h3 style='color: #7f8c8d; font-size: 12pt; margin-top: 10px;'>")
    html_content = html_content.replace("<ul>", "<ul style='margin-bottom: 10px;'>")
    html_content = html_content.replace("<li>", "<li style='margin-bottom: 5px;'>")
    
    try:
        import re
        
        # 1. Base Markdown conversion
        html_content = markdown.markdown(text, extensions=['tables'])

        # 2. Semantic styling (Typography)
        html_content = html_content.replace("<p>", "<p style='margin-bottom: 10px; line-height: 1.6;'>")
        html_content = html_content.replace("<h1>", "<h1 style='color: #2c3e50; font-size: 18pt; margin-top: 20px; border-bottom: 1px solid #eee;'>")
        html_content = html_content.replace("<h2>", "<h2 style='color: #34495e; font-size: 14pt; margin-top: 15px;'>")
        html_content = html_content.replace("<h3>", "<h3 style='color: #7f8c8d; font-size: 12pt; margin-top: 10px;'>")
        html_content = html_content.replace("<ul>", "<ul style='margin-bottom: 10px;'>")
        html_content = html_content.replace("<li>", "<li style='margin-bottom: 5px;'>")

        html_content = re.sub(r'<table>', '<table border="1" align="center" width="100%" style="border-collapse: collapse;">', html_content)
        html_content = re.sub(r'<th>', '<th style="background-color: #f0f0f0; padding: 5px; font-weight: bold;">', html_content)
        html_content = re.sub(r'<td>', '<td style="padding: 5px;">', html_content)

        try:
            pdf.write_html(html_content)
        except Exception as html_error:
            print(f"Warning: PDF generation with rich HTML failed: {html_error}")
            print("Attempting fallback with simplified HTML...")
            
            # Fallback 1: Try without the heavy inline styles which might confuse fpdf2
            simple_html = markdown.markdown(text, extensions=['tables'])
            # Only essential table borders
            simple_html = simple_html.replace("<table>", '<table border="1">')
            pdf = PDF()
            pdf.add_page()
            pdf.set_font("helvetica", size=11)
            pdf.write_html(simple_html)
        
        # Save
        print(f"Saving analysis to PDF: {output_path}...")
        pdf.output(output_path)
        print(f"PDF Analysis saved successfully to: {output_path}")

    except Exception as e:
        print(f"Error generating PDF section with fpdf2: {e}")
        # Fallback dump if HTML fails entirely
        txt_path = output_path + ".txt"
        with open(txt_path, "w") as f:
            f.write(text)
        print(f"Saved raw text to {txt_path} due to error.")
        
        # Attempt to create a simple PDF from the text
        try:
            print("Attempting to generate text-only fallback PDF...")
            simple_pdf = FPDF()
            simple_pdf.add_page()
            simple_pdf.set_font("helvetica", size=10)
            # Sanitize text for latin-1
            safe_text = text.encode('latin-1', 'replace').decode('latin-1')
            simple_pdf.multi_cell(0, 5, safe_text)
            simple_pdf.output(output_path)
            print(f"Fallback PDF generated successfully: {output_path}")
        except Exception as fallback_error:
            print(f"Failed to generate fallback PDF: {fallback_error}")

def upload_to_gemini(client, path, mime_type="application/pdf"):
    """
    Uploads the given file to Gemini using the client.
    """
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        sys.exit(1)
        
    print(f"Uploading file: {path}...")
    # Use client.files.upload logic
    # The new SDK might use client.files.upload(file=path)
    # Check if 'path' should be opened file or path string. 
    # Usually SDK v2 takes path=path.
    try:
        # Assuming google-genai v1.0+ conventions
        # client.files.upload(path=...) returns a File object
        file = client.files.upload(file=path) 
    except Exception as e:
         print(f"Upload failed: {e}")
         sys.exit(1)

    print(f"File uploaded: {file.display_name} ({file.uri})")
    return file

def wait_for_files_active(client, files):
    """
    Waits for the given files to be active.
    """
    print("Waiting for file processing...", end="")
    for file in files:
        # file is the object returned by upload
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            # Refresh file status
            file = client.files.get(name=file.name)
        
        if file.state.name != "ACTIVE":
            print(f"\nError: File {file.name} failed to process. State: {file.state.name}")
            sys.exit(1)
    print("\nAll files ready.")

def analyze_stock_report(pdf_path, model_name=DEFAULT_MODEL_NAME):
    """
    Analyzes the stock report using Gemini.
    """
    client = setup_gemini_client()
    
    # 1. Upload File
    pdf_file = upload_to_gemini(client, pdf_path)
    
    # 2. Wait for processing
    wait_for_files_active(client, [pdf_file])
    
    # 3. Generate Content
    prompt = """
    Act as a seasoned Positional Equity Trader specializing in intermediate-term trends. Your goal is to identify trade setups with a probability of 5%-15% upside over a 1-4 month time horizon.

    Please analyze the attached report using a step-by-step approach:

    Step 1: Audit the Data.
    Please evaluate the data quality in a bulleted list format:
    * **Data Present:** (e.g., daily charts, moving averages, RSI).
    * **Data Missing:** (e.g., weekly charts, volume analysis).
    * **Confidence Level:** (High/Medium/Low) and brief explanation.

    Step 2: Analyze the Setup.
    Review the price structure, trend direction, and momentum. Look for confluence that supports a multi-month move.

    Step 3: Formulate the Recommendation.
    Based on Step 2, advise if I should enter this stock now.

    If YES: Provide the following in a bulleted list:
    * **Entry Price:** (Exact level)
    * **Stop Loss:** (Hard level)
    * **Take Profit:** (Target level)

    If NO: Explain why invalid. If it's a potential setup, provide a bulleted list for the conditional plan:
    * **Condition to Buy:** (e.g., Close above EMA 50)
    * **Entry Price:** (Hypothetical trigger)
    * **Stop Loss:** (Hypothetical risk)
    * **Take Profit:** (Hypothetical target)
    Please keep your tone objective, professional, and risk-averse.
    """
    
    print("\nSending request to Gemini (this may take a moment)...")
    try:
        # client.models.generate_content
        response = client.models.generate_content(
            model=model_name,
            contents=[pdf_file, prompt],
            config=types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=8192
            )
        )
        
        print("\n" + "="*80)
        print("GEMINI ANALYSIS REPORT")
        print("="*80 + "\n")
        print(response.text)
        print("\n" + "="*80)
        
        # 5. News & Analyst Targets
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            parts = base_name.split('_')
            # Check if format matches Stock_Report_<SYMBOL>...
            if len(parts) >= 3 and parts[0] == "Stock" and parts[1] == "Report":
                stock_name = parts[2]
            else:
                # Fallback: Just use the filename
                stock_name = base_name
        except Exception:
            stock_name = base_name

        news_prompt = f"""
        OK , now Act as a stock research analyst working in investment firm. Your goal is to search for latest news and information on stock: {stock_name}.
        Please search on internet and anywhere you can answer following:
                
        1) what are the latest news for {stock_name} this week?
        2) Any upgrade or downgrade from brokerages for {stock_name}?
        3) Any other news directly or indirectly impacting {stock_name}?
        4) What are the latest targets from analysts community and brokerages for {stock_name}? Provide this as a Markdown Table.
        
        Please format the news bullets clearly and ensure the targets are in a table.
        """
        
        print(f"\nSending request for News & Analyst Targets for {stock_name}...")
        
        try:
            # Google Search Tool using GoogleGenAI SDK V2
            google_search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            # Note regarding model: 'gemini-1.5-pro' generally supports tools. 
            # If default model fails with tools, we might need a fallback.
            news_response = client.models.generate_content(
                model=model_name,
                contents=news_prompt,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                )
            )
            news_text = news_response.text

        except Exception as tool_error:
            print(f"Warning: Could not use Google Search tool: {tool_error}. Falling back to standard generation.")
            news_response = client.models.generate_content(
                model=model_name,
                contents=news_prompt
            )
            news_text = news_response.text

        print("\n" + "="*80)
        print("NEWS & ANALYST TARGETS")
        print("="*80 + "\n")
        print(news_text)
        print("\n" + "="*80)
        
        # Combine reports
        full_report = response.text + "\n\n" + "# News & Analyst Targets\n" + news_text
        
        output_pdf_path = f"Gemini_Analysis_{base_name}.pdf"
        save_analysis_to_pdf(full_report, output_pdf_path)
        
        if not os.path.exists(output_pdf_path):
             print(f"Skipping merge: {output_pdf_path} was not generated.")
        else:
            try:
                print(f"Merging original report into {output_pdf_path}...")
                temp_gemini_path = f"temp_{output_pdf_path}"
                os.rename(output_pdf_path, temp_gemini_path)
                
                merger = PdfWriter()
                # 1. Gemini Analysis
                merger.append(temp_gemini_path)
                # 2. Original Report
                merger.append(pdf_path)
                
                # Write combined PDF
                merger.write(output_pdf_path)
                merger.close()
                
                # Cleanup temp file
                os.remove(temp_gemini_path)
                print(f"Successfully merged original report. Final output: {output_pdf_path}")
                
            except Exception as merge_error:
                print(f"Error merging PDFs: {merge_error}")
                # Restore original if merge fails
                if os.path.exists(temp_gemini_path):
                    os.rename(temp_gemini_path, output_pdf_path)
        
    except Exception as e:
        print(f"\nError generating content: {e}")
        # Note: clean up file if possible?
    
def analyze_folder(folder_path, model_name=DEFAULT_MODEL_NAME):
    """
    Analyzes all PDF stock reports in a folder.
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    print(f"Scanning folder: {folder_path}...")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not files:
        print("No PDF files found in the folder.")
        return
        
    print(f"Found {len(files)} PDF reports. Starting batch analysis...")
    
    for i, filename in enumerate(files, 1):
        pdf_path = os.path.join(folder_path, filename)
        print(f"\n[{i}/{len(files)}] Processing: {filename}")
        try:
            analyze_stock_report(pdf_path, model_name)
        except Exception as e:
            print(f"Failed to analyze {filename}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a Stock Report PDF using Gemini.")
    parser.add_argument("path", help="Path to the PDF report file or a folder containing PDFs")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help=f"Gemini model to use (default: {DEFAULT_MODEL_NAME})")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        analyze_folder(args.path, args.model)
    else:
        analyze_stock_report(args.path, args.model)
