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

"""
Prompts Module for Gemini Stock Analysis Workflow

This module centralizes all the large instructional prompts and persona definitions 
used by the Gemini AI in `analyze_report.py`. 

Extracting these prompts into a separate file keeps the main application logic clean 
and makes it much easier to tweak, tune, or update the AI's trading personas and 
instructions in the future without risking accidental logic breaks.
"""

# ===============================================================================
# 1. STRATEGIC ANALYSIS PROMPT
# ===============================================================================
# Usage:   Passed directly to Gemini alongside the uploaded PDF stock report.
# Purpose: Instructs the AI to act as a "Positional Equity Trader". It provides a 
#          strict 3-step framework (Audit, Analyze, Recommend) ensuring the AI 
#          always outputs consistent, risk-averse Entry, Stop Loss, and Take 
#          Profit targets.
STRATEGIC_ANALYSIS_PROMPT = """
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

# ===============================================================================
# 2. CANDLESTICK ANALYSIS PROMPT
# ===============================================================================
# Usage:   Passed to Gemini alongside the uploaded PDF stock report.
# Purpose: Provides the AI with the exact mathematical algorithm (the 5-step, 
#          -2 to +2 scoring system) used by the local scripts to generate the 
#          Candlestick Classification charts. 
#          Giving the AI this "internal documentation" allows it to correctly 
#          interpret the chart visually and predict short-term buying/selling 
#          momentum accurately.
CANDLESTICK_ANALYSIS_PROMPT = """
Act as a seasoned Candlestick Trader specializing in understanding the price action using candlestick classification. 
In the attached report, please look at the page containing "Candlestick Classification".
Then interpret the Candlestick Classification chart for the stock price movement and predict which side the future price should be.
The chart is made with following logic on candlesticks 

* **Candlestick Classification System**

This script classifies daily candlesticks into 5 categories based on professional price action analysis:

1. Seller Strong Control (-2): Large bearish body, small wicks, close near low, high volume
2. Seller Control (-1): Moderate bearish candle or strong candle with low volume
3. No Control (0): Small body, long wicks, indecision patterns
4. Buyer Control (+1): Moderate bullish candle or strong candle with low volume
5. Buyer Strong Control (+2): Large bullish body, small wicks, close near high, high volume

* **Key Principles**:
- Body size is normalized by ATR to handle varying volatility
- Volume acts as a validator/multiplier for the signal
- Close position shows where price settled (conviction)
- Wicks show failed attempts and rejections

* **CLASSIFICATION ALGORITHM DOCUMENTATION**:
========================================

The classification uses a 5-step scoring system:

* **STEP 1 - BASE SCORE (Body Strength)**:  
- Measure body size relative to ATR (volatility normalization)
- Strong body (>1.5 ATR) → ±2 points
- Moderate body (0.8-1.5 ATR) → ±1 points  
- Weak body (<0.5 ATR) → 0 points

WHY? Body size shows conviction, but must be relative to volatility.
A ₹50 move means different things for ₹200 vs ₹2000 stock.

* **STEP 2 - CLOSE POSITION ADJUSTMENT**:
- Check where price closed within the day's range
- Bullish candle closing near low → downgrade (weak conviction)
- Bearish candle closing near high → upgrade (weak conviction)

WHY? Close is more important than open. It shows where traders
were comfortable holding overnight. A bullish candle closing
near its low shows buyers lost control by end of day.

* **STEP 3 - WICK ANALYSIS**:
- Long lower wick → bullish adjustment (buyers defended)
- Long upper wick → bearish adjustment (sellers defended)
- Both wicks long → strong indecision signal
- Both wicks small → amplify existing signal

WHY? Wicks show FAILED attempts. Price tried to go there but
got rejected. This is critical for support/resistance analysis.

* **STEP 4 - VOLUME MULTIPLIER (Most Critical)**:
- High volume (>1.3x avg) → multiply by 1.2 (confirm)
- Low volume (<0.8x avg) → multiply by 0.6 (weaken)

WHY? "Price without volume is half the story." Volume shows
HOW MANY participants agreed. High volume = institutional
participation. Low volume = weak hands, may not sustain.

* **STEP 5 - FINAL MAPPING**:
- Clamp score to -2 to +2 range
- Map to one of 5 discrete categories based on thresholds

* **EXAMPLES**:
=========

Example 1: Strong Bullish Day
Input: Body=2.0 ATR, Close Position=0.85, Small wicks, Volume=1.5x avg
Step 1: +2 (strong bullish body)
Step 2: +2 (closed near high, no change)
Step 3: +2.4 (small wicks amplify by 1.2)
Step 4: +2.88 (high volume multiplies by 1.2)
Step 5: Clamp to +2 → "Buyer Strong Control"

Example 2: Weak Bullish Day
Input: Body=1.0 ATR, Close Position=0.6, Normal wicks, Volume=0.7x avg
Step 1: +1 (moderate bullish body)
Step 2: +1 (closed above mid, no change)
Step 3: +1 (normal wicks, no change)
Step 4: +0.6 (low volume weakens by 0.6)
Step 5: +0.6 → "Buyer Control" (downgraded from potential strong)

Example 3: Indecision Day
Input: Body=0.3 ATR, Close Position=0.5, Large wicks both sides, Volume=0.9x avg
Step 1: 0 (weak body)
Step 2: 0 (closed at middle)
Step 3: 0 (both wicks large confirms indecision)
Step 4: 0 (average volume, no change)
Step 5: 0 → "No Control"
"""

# ===============================================================================
# 3. NEWS & ANALYST TARGETS PROMPT TEMPLATE
# ===============================================================================
# Usage:   Passed to Gemini alongside the Google Search grounding Tool.
# Note:    This is a Python format string. You MUST call `.format(stock_name=...)`
#          on this string before sending it to the Gemini API.
# Purpose: Instructs the AI to browse the live internet to aggregate the latest 
#          weekly news, broker upgrades/downgrades, and price targets for a 
#          specific stock, outputting the targets neatly in a Markdown table.
NEWS_ANALYSIS_PROMPT_TEMPLATE = """
OK , now Act as a stock research analyst working in investment firm. Your goal is to search for latest news and information on stock: {stock_name}.
Please search on internet and anywhere you can answer following:
        
1) what are the latest news for {stock_name} this week?
2) Any upgrade or downgrade from brokerages for {stock_name}?
3) Any other news directly or indirectly impacting {stock_name}?
4) What are the latest targets from analysts community and brokerages for {stock_name}? Provide this as a Markdown Table.

Please format the news bullets clearly and ensure the targets are in a table.
"""
