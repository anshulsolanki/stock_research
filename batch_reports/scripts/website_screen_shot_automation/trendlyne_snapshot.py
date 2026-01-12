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

import argparse
import time
import os
from playwright.sync_api import sync_playwright

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Config parameters for login
USERNAME = "your_username_here" 
PASSWORD = "your_password_here"

def login_if_needed(page):
    """
    Checks if the user is logged in. If not, performs the login action.
    """
    try:
        print("Checking login status...")
        # Check if the profile dropdown exists (indicator of being logged in)
        # We use a short timeout because we expect it to be there immediately if logged in
        if page.locator(".tl-profile-dropdown").count() > 0:
            print("User is already logged in.")
            return

        print("User is NOT logged in. Attempting to log in...")
        
        # Click the Login/Signup button
        if page.locator("#login-signup-btn").count() > 0:
            page.click("#login-signup-btn")
        else:
            print("Login button not found, and not logged in. unexpected state.")
            return

        # Wait for the login modal/form
        page.wait_for_selector("#id_login")
        
        # Fill credentials
        # Check if USERNAME/PASSWORD are set to actual values
        if USERNAME == "your_username_here" or PASSWORD == "your_password_here":
            print("WARNING: Default credentials found. Please update USERNAME and PASSWORD constants in the script.")
            return

        print(f"Logging in as {USERNAME}...")
        page.fill("#id_login", USERNAME)
        page.fill("#id_password", PASSWORD)
        
        # Click the Login button
        # The modal usually has a submit button. Based on inspection, it's often a button with class 'login-btn'
        page.click(".login-btn")
        
        # Wait for login to complete (profile dropdown should appear)
        page.wait_for_selector(".tl-profile-dropdown", timeout=25000)
        print("Login successful.")
        
    except Exception as e:
        print(f"Login process failed or encountered an error: {e}")
        # We continue even if login fails, as some public info might still be accessible
        pass

def get_trendlyne_snapshots(stock_name="Dabur", output_dir=".", headless=True, save_to_file=True):
    """
    Automates taking screenshots of Trendlyne for a specific stock.
    1. Log in if needed.
    2. Searches for the stock.
    3. Takes a screenshot of the Main Page.
    4. Navigates to the Forecaster Page and takes a screenshot.
    
    Returns:
        List of dictionaries with keys: 'name', 'type' ('file' or 'memory'), 'content' (path or bytes)
    """
    
    # Ensure output directory exists (only if saving to file)
    if save_to_file:
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(SCRIPT_DIR, output_dir)
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Path for persistent browser data
    user_data_dir = os.path.join(SCRIPT_DIR, "trendlyne_user_data")

    generated_files = []

    with sync_playwright() as p:
        # Launch persistent context to reuse sessions
        # Viewport and User Agent are set here
        context = p.chromium.launch_persistent_context(
            user_data_dir,
            headless=headless,
            channel="chrome",
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()

        try:
            print(f"Navigating to Trendlyne...")
            page.goto("https://trendlyne.com/")
            
            # Perform login check
            login_if_needed(page)
            
            # Wait for search bar
            print(f"Searching for '{stock_name}'...")
            page.wait_for_selector("#navbar-desktop-search")
            page.fill("#navbar-desktop-search", stock_name)
            
            # Wait for autocomplete results
            # The selector .ui-menu-item-wrapper usually contains the results
            page.wait_for_selector(".ui-menu-item-wrapper")
            
            # Give a small buffer for the list to fully populate/settle
            time.sleep(1)
            
            # Click the first result
            print("Clicking the first result...")
            page.click(".ui-menu-item-wrapper >> nth=0")
            
            # Wait for navigation to the stock page
            page.wait_for_load_state("networkidle")
            
            # 1. Main Page Screenshot
            main_url = page.url
            print(f"Arrived at: {main_url}")
            
            # User requested "content that appears in full screen mode" -> Viewport screenshot
            if save_to_file:
                main_screenshot_path = os.path.join(output_dir, f"{stock_name}_main.png")
                page.screenshot(path=main_screenshot_path)
                print(f"Saved Main Page screenshot to: {main_screenshot_path}")
                generated_files.append({'name': f"{stock_name}_main", 'type': 'file', 'content': main_screenshot_path})
            else:
                img_bytes = page.screenshot()
                generated_files.append({'name': f"{stock_name}_main", 'type': 'memory', 'content': img_bytes})
                print(f"Captured Main Page screenshot (in memory)")
            
            # 2. Forecaster Page Screenshot
            # Construct URL: https://trendlyne.com/equity/303/DABUR/dabur-india-ltd/
            #             -> https://trendlyne.com/equity/consensus-estimates/303/DABUR/dabur-india-ltd/
            
            if "/equity/" in main_url:
                forecaster_url = main_url.replace("/equity/", "/equity/consensus-estimates/")
                print(f"Navigating to Forecaster page: {forecaster_url}")
                
                page.goto(forecaster_url)
                page.wait_for_load_state("networkidle")
                
                if save_to_file:
                    forecaster_screenshot_path = os.path.join(output_dir, f"{stock_name}_forecaster.png")
                    page.screenshot(path=forecaster_screenshot_path)
                    print(f"Saved Forecaster Page screenshot to: {forecaster_screenshot_path}")
                    generated_files.append({'name': f"{stock_name}_forecaster", 'type': 'file', 'content': forecaster_screenshot_path})
                else:
                    img_bytes = page.screenshot()
                    generated_files.append({'name': f"{stock_name}_forecaster", 'type': 'memory', 'content': img_bytes})
                    print(f"Captured Forecaster Page screenshot (in memory)")
            else:
                print("Could not construct Forecaster URL (unexpected URL format).")
                
        except Exception as e:
            print(f"An error occurred: {e}")
            if save_to_file:
                # Take a debug screenshot if something fails
                debug_path = os.path.join(output_dir, "debug_error.png")
                page.screenshot(path=debug_path)
            # We don't append debug screenshot to generated_files as we probably don't want it in the report

        finally:
            context.close()
            
    return generated_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take screenshots from Trendlyne for a given stock.")
    parser.add_argument("stock_name", type=str, nargs='?', default="DABUR", help="Name of the stock (e.g., DABUR, TCS)")
    parser.add_argument("--output", type=str, default=".", help="Output directory for screenshots (relative to script or absolute)")
    parser.add_argument("--no-headless", action="store_false", dest="headless", help="Run browser in non-headless mode (useful for manual login)")
    parser.set_defaults(headless=True)
    
    args = parser.parse_args()
    
    get_trendlyne_snapshots(args.stock_name, args.output, args.headless)
