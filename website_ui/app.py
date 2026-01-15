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

from flask import Flask, render_template, request, jsonify
import sys
import os
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    print("\n" + "="*80)
    print("ERROR: 'python-dotenv' not found!")
    print("It looks like you are NOT using the project's virtual environment.")
    print("Please run the app using:")
    print("  ./.venv/bin/python website_ui/app.py")
    print("="*80 + "\n")
    # We can continue if they don't have .env needs, but Flask might complain later
    # For now, let's just let it fail gracefully later or just pass
    load_dotenv = None
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Add parent directory to path to import analysis modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lagging_indicator_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'leading_indicator_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'market_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fundamental_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'batch_reports', 'scripts'))

from macd_analysis import run_analysis as run_macd_analysis
from supertrend_analysis import run_analysis as run_supertrend_analysis
from multi_timeframe_analysis import run_analysis as run_multi_timeframe_analysis
from bollinger_band_analysis import run_analysis as run_bollinger_analysis
from crossover_analysis import run_analysis as run_crossover_analysis
from donchian_channel_analysis import run_analysis as run_donchian_analysis
from rsi_divergence_analysis import run_analysis as run_rsi_analysis
from rsi_volume_divergence import run_analysis as run_rsi_volume_analysis
from volatility_squeeze_analysis import run_analysis as run_volatility_squeeze_analysis
from rs_analysis import run_analysis as run_rs_analysis
from volume_analysis import run_analysis as run_volume_analysis
from sector_analysis import run_analysis as run_sector_analysis
from stock_in_sector_analysis import run_analysis as run_stock_in_sector_analysis
from batch_analysis import run_batch_analysis
from fundamental_analysis import run_analysis as run_fundamental_analysis
from stock_detailed_report import generate_stock_report

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')

@app.route('/market_analysis')
def market_analysis():
    """Serve the market analysis page"""
    return render_template('market_analysis.html')

@app.route('/batch')
def batch_analysis_page():
    """Serve the batch analysis page"""
    return render_template('batch_analysis.html')

@app.route('/settings')
def settings():
    """Serve the settings page"""
    return render_template('settings.html')

@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    """Get list of tickers from tickers_list.json"""
    try:
        tickers_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickers_list.json')
        if not os.path.exists(tickers_file):
            return jsonify({'success': False, 'error': 'tickers_list.json not found'})
            
        import json
        with open(tickers_file, 'r') as f:
            data = json.load(f)
            
        return jsonify({'success': True, 'tickers': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    """Get list of watchlist stocks from watchlist.json"""
    try:
        watchlist_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'watchlist.json')
        if not os.path.exists(watchlist_file):
            return jsonify({'success': False, 'error': 'watchlist.json not found'})
            
        import json
        with open(watchlist_file, 'r') as f:
            data = json.load(f)
            
        return jsonify({'success': True, 'watchlist': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def fetch_stock_data(ticker, interval='1d', lookback_days=730):
    """
    Fetch stock data once for reuse across multiple analysis functions.
    """
    try:
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Adjust for 15m limit (max 60 days)
        if interval == '15m':
            actual_lookback = min(lookback_days, 59)
            start_date = end_date - timedelta(days=actual_lookback)
            
        print(f"Fetching shared data for {ticker} ({interval}, {lookback_days}d)...")
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, 
                         progress=False, auto_adjust=False, multi_level_index=False)
        
        if df.empty:
            print(f"Warning: Fetched empty dataframe for {ticker}")
            return None
            
        return df
    except Exception as e:
        print(f"Error fetching shared data: {e}")
        return None


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a stock ticker using MACD, Supertrend, and Bollinger Bands"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        analysis_type = data.get('analysis_type', 'all')  # 'all', 'macd', 'supertrend', 'bollinger', 'crossover', or 'donchian'
        macd_config = data.get('macd_config', {})
        supertrend_config = data.get('supertrend_config', {})
        bollinger_config = data.get('bollinger_config', {})
        crossover_config = data.get('crossover_config', {})
        donchian_config = data.get('donchian_config', {})
        rsi_config = data.get('rsi_config', {})
        rsi_volume_config = data.get('rsi_volume_config', {})
        volatility_squeeze_config = data.get('volatility_squeeze_config', {})
        rs_config = data.get('rs_config', {})
        volume_config = data.get('volume_config', {}) # Added volume_config
        use_sector_index = data.get('use_sector_index', False)
        
        if not ticker:
            return jsonify({'success': False, 'error': 'Please enter a valid ticker symbol'})
        
        response_data = {
            'success': True,
            'ticker': ticker,
            'analysis_type': analysis_type
        }

        # Centralized Data Fetching
        # Determine interval and lookback based on configs
        # Default to 1d and 2 years (730 days)
        interval = '1d'
        lookback_period = 730
        
        # Check if any config initiates a different interval (simple check)
        # We prioritize the interval found in the first active config
        if macd_config.get('INTERVAL'): interval = macd_config['INTERVAL']
        elif supertrend_config.get('INTERVAL'): interval = supertrend_config['INTERVAL']
        elif bollinger_config.get('INTERVAL'): interval = bollinger_config['INTERVAL']
        elif rsi_config.get('INTERVAL'): interval = rsi_config['INTERVAL']
        
        # Fetch shared dataframe
        df = fetch_stock_data(ticker, interval=interval, lookback_days=lookback_period)
        
        # Fetch benchmark data if RS analysis is requested
        df_benchmark = None
        if analysis_type in ['all', 'rs']:
            benchmark = rs_config.get('BENCHMARK_TICKER')
            if not benchmark and use_sector_index:
                # Need to detect benchmark or sector
                # Since we can't easily auto-detect here without repeating logic, 
                # we rely on rs_analysis to auto-detect if we pass None, BUT
                # to optimize, we should try to fetch it here if we can.
                # For now, let's let rs_analysis fetch it if we can't easily determine it,
                # OR we accept that we only optimize the main ticker data here.
                # However, if benchmark IS provided:
                pass
            
            if benchmark:
                df_benchmark = fetch_stock_data(benchmark, interval=interval, lookback_days=lookback_period)

        # Run MACD analysis
        if analysis_type in ['all', 'macd']:
            macd_results = run_macd_analysis(ticker=ticker, show_plot=False, config=macd_config, df=df)
            
            if not macd_results['success']:
                return jsonify(macd_results)
            
            # Convert MACD figure to base64 image
            macd_fig = macd_results['figure']
            macd_buf = BytesIO()
            macd_fig.savefig(macd_buf, format='png', dpi=100, bbox_inches='tight')
            macd_buf.seek(0)
            macd_image_base64 = base64.b64encode(macd_buf.getvalue()).decode('utf-8')
            macd_buf.close()
            
            # Close the figure
            plt.close(macd_fig)
            
            # Format MACD divergences
            divergences_formatted = []
            if macd_results['divergences']:
                for div in macd_results['divergences']:
                    divergences_formatted.append({
                        'type': div['Type'],
                        'date': div['Date'].strftime('%Y-%m-%d'),
                        'price': float(div['Price']),
                        'details': div['Details']
                    })
            
            response_data['macd'] = {
                'macd_line': round(macd_results['macd_line'], 2),
                'signal_line': round(macd_results['signal_line'], 2),
                'histogram': round(macd_results['histogram'], 2),
                'trend': macd_results['trend'],
                'momentum': macd_results['momentum'],
                'crossover_signal': macd_results['crossover_signal'],
                'divergences': divergences_formatted,
                'chart_image': macd_image_base64
            }
        
        # Run Supertrend analysis
        if analysis_type in ['all', 'supertrend']:
            supertrend_results = run_supertrend_analysis(ticker=ticker, show_plot=False, config=supertrend_config, df=df)
            
            if not supertrend_results['success']:
                return jsonify(supertrend_results)
            
            # Convert Supertrend figure to base64 image
            supertrend_fig = supertrend_results['figure']
            supertrend_buf = BytesIO()
            supertrend_fig.savefig(supertrend_buf, format='png', dpi=100, bbox_inches='tight')
            supertrend_buf.seek(0)
            supertrend_image_base64 = base64.b64encode(supertrend_buf.getvalue()).decode('utf-8')
            supertrend_buf.close()
            
            # Close the figure
            plt.close(supertrend_fig)
            
            response_data['supertrend'] = {
                'last_trend': supertrend_results['last_trend'],
                'last_price': round(supertrend_results['last_price'], 2),
                'supertrend_value': round(supertrend_results['supertrend_value'], 2),
                'status': supertrend_results['status'],
                'last_date': supertrend_results['last_date'].strftime('%Y-%m-%d'),
                'signal_date': supertrend_results['signal_date'].strftime('%Y-%m-%d'),
                'chart_image': supertrend_image_base64
            }
        
        # Run Bollinger Band analysis
        if analysis_type in ['all', 'bollinger']:
            bollinger_results = run_bollinger_analysis(ticker=ticker, show_plot=False, config=bollinger_config, df=df)
            
            if not bollinger_results['success']:
                return jsonify(bollinger_results)
            
            # Convert Bollinger figure to base64 image
            bollinger_fig = bollinger_results['figure']
            bollinger_buf = BytesIO()
            bollinger_fig.savefig(bollinger_buf, format='png', dpi=100, bbox_inches='tight')
            bollinger_buf.seek(0)
            bollinger_image_base64 = base64.b64encode(bollinger_buf.getvalue()).decode('utf-8')
            bollinger_buf.close()
            
            # Close the figure
            plt.close(bollinger_fig)
            
            # Format signals
            signals_formatted = []
            if bollinger_results['signals']:
                for sig in bollinger_results['signals']:
                    signals_formatted.append({
                        'type': sig['Type'],
                        'date': sig['Date'].strftime('%Y-%m-%d'),
                        'price': float(sig['Price']),
                        'reason': sig.get('Reason', '')
                    })
            
            response_data['bollinger'] = {
                'bb_upper': round(bollinger_results['bb_upper'], 2),
                'bb_lower': round(bollinger_results['bb_lower'], 2),
                'sma_20': round(bollinger_results['sma_20'], 2),
                'last_price': round(bollinger_results['last_price'], 2),
                'pct_b': round(bollinger_results['pct_b'], 2),
                'bandwidth': round(bollinger_results['bandwidth'], 4),
                'status': bollinger_results['status'],
                'signal': bollinger_results['signal'],
                'signals': signals_formatted,
                'chart_image': bollinger_image_base64
            }

        # Run Crossover Analysis
        if analysis_type in ['all', 'crossover']:
            crossover_results = run_crossover_analysis(ticker=ticker, show_plot=False, config=crossover_config, df=df)
            
            if not crossover_results['success']:
                return jsonify(crossover_results)
            
            # Convert Crossover figure to base64 image
            crossover_fig = crossover_results['figure']
            crossover_buf = BytesIO()
            crossover_fig.savefig(crossover_buf, format='png', dpi=100, bbox_inches='tight')
            crossover_buf.seek(0)
            crossover_image_base64 = base64.b64encode(crossover_buf.getvalue()).decode('utf-8')
            crossover_buf.close()
            
            # Close the figure
            plt.close(crossover_fig)
            
            # Format Golden Cross Date
            gc_date_str = None
            if crossover_results['gc_date']:
                gc_date_str = crossover_results['gc_date'].strftime('%Y-%m-%d')
            
            response_data['crossover'] = {
                'ema_20': round(crossover_results['ema_20'], 2) if crossover_results['ema_20'] else None,
                'ema_50': round(crossover_results['ema_50'], 2) if crossover_results['ema_50'] else None,
                'ema_200': round(crossover_results['ema_200'], 2) if crossover_results['ema_200'] else None,
                'trend_status': crossover_results['trend_status'],
                'gc_date': gc_date_str,
                'gc_price': round(crossover_results['gc_price'], 2) if crossover_results['gc_price'] else None,
                'chart_image': crossover_image_base64
            }

        # Run Donchian Channel Analysis
        if analysis_type in ['all', 'donchian']:
            donchian_results = run_donchian_analysis(ticker=ticker, show_plot=False, config=donchian_config, df=df)
            
            if not donchian_results['success']:
                return jsonify(donchian_results)
            
            # Convert Donchian figure to base64 image
            donchian_fig = donchian_results['figure']
            donchian_buf = BytesIO()
            donchian_fig.savefig(donchian_buf, format='png', dpi=100, bbox_inches='tight')
            donchian_buf.seek(0)
            donchian_image_base64 = base64.b64encode(donchian_buf.getvalue()).decode('utf-8')
            donchian_buf.close()
            
            # Close the figure
            plt.close(donchian_fig)
            
            # Format signals
            signals_formatted = []
            if donchian_results['signals']:
                for sig in donchian_results['signals']:
                    signals_formatted.append({
                        'type': sig['Type'],
                        'date': sig['Date'].strftime('%Y-%m-%d'),
                        'price': float(sig['Price']),
                        'upper': float(sig['Upper']),
                        'lower': float(sig['Lower'])
                    })
            
            response_data['donchian'] = {
                'dc_upper': round(donchian_results['dc_upper'], 2),
                'dc_lower': round(donchian_results['dc_lower'], 2),
                'dc_middle': round(donchian_results['dc_middle'], 2),
                'last_price': round(donchian_results['last_price'], 2),
                'status': donchian_results['status'],
                'breakout_signal': donchian_results['breakout_signal'],
                'signals': signals_formatted,
                'chart_image': donchian_image_base64
            }

        # Run Volume Analysis
        if analysis_type in ['all', 'volume']:
            print(f"Starting Volume analysis for {ticker}...")
            try:
                vol_config = data.get('volume_config', {})
                if not vol_config:
                    vol_config = {'ORDER': 5, 'LOOKBACK_PERIODS': 365}
                    
                vol_result = run_volume_analysis(ticker, show_plot=False, config=vol_config, df=df)
                if vol_result['success']:
                    print(f"Volume analysis result: {vol_result['success']}")
                    
                    # Ensure divergences is list of dicts
                    divs = vol_result.get('divergences', [])
                    
                    response_data['volume'] = {
                        'divergences': divs,
                        'chart_image': vol_result.get('chart_image')
                    }
                else:
                    print(f"Volume analysis failed: {vol_result.get('error')}")
            except Exception as e:
                print(f"Error in Volume analysis: {e}")
                traceback.print_exc()

        # Run RSI Divergence Analysis
        if analysis_type in ['all', 'rsi']:
            print(f"Starting RSI analysis for {ticker}...")
            rsi_results = run_rsi_analysis(ticker=ticker, show_plot=False, config=rsi_config, df=df)
            print(f"RSI analysis result: {rsi_results['success']}")
            
            if not rsi_results['success']:
                return jsonify(rsi_results)
            
            # Convert RSI figure to base64 image
            rsi_fig = rsi_results['figure']
            rsi_buf = BytesIO()
            rsi_fig.savefig(rsi_buf, format='png', dpi=100, bbox_inches='tight')
            rsi_buf.seek(0)
            rsi_image_base64 = base64.b64encode(rsi_buf.getvalue()).decode('utf-8')
            rsi_buf.close()
            print("RSI chart generated successfully")
            
            # Close the figure
            plt.close(rsi_fig)
            
            # Format divergences
            divergences_formatted = []
            if rsi_results['divergences']:
                for div in rsi_results['divergences']:
                    divergences_formatted.append({
                        'type': div['Type'],
                        'date': div['Date'].strftime('%Y-%m-%d'),
                        'price': float(div['Price']),
                        'details': div['Details']
                    })
            
            response_data['rsi'] = {
                'current_rsi': round(rsi_results['current_rsi'], 2),
                'divergences': divergences_formatted,
                'chart_image': rsi_image_base64
            }
        
        # Run RSI-Volume Divergence Analysis
        if analysis_type in ['all', 'rsi_volume']:
            print(f"Starting RSI-Volume analysis for {ticker}...")
            rsi_volume_results = run_rsi_volume_analysis(ticker=ticker, show_plot=False, config=rsi_volume_config, df=df)
            print(f"RSI-Volume analysis result: {rsi_volume_results['success']}")
            
            if not rsi_volume_results['success']:
                return jsonify(rsi_volume_results)
            
            # Convert RSI-Volume figure to base64 image
            rsi_volume_fig = rsi_volume_results['figure']
            rsi_volume_buf = BytesIO()
            rsi_volume_fig.savefig(rsi_volume_buf, format='png', dpi=100, bbox_inches='tight')
            rsi_volume_buf.seek(0)
            rsi_volume_image_base64 = base64.b64encode(rsi_volume_buf.getvalue()).decode('utf-8')
            rsi_volume_buf.close()
            print("RSI-Volume chart generated successfully")
            
            # Close the figure
            plt.close(rsi_volume_fig)
            
            # Format divergences for JSON
            bullish_divs_formatted = []
            bearish_divs_formatted = []
            early_reversals_formatted = []
            
            if rsi_volume_results['bullish_divergences']:
                for div in rsi_volume_results['bullish_divergences']:
                    bullish_divs_formatted.append({
                        'type': div['Type'],
                        'date': div['Date'].strftime('%Y-%m-%d'),
                        'price': float(div['Price']),
                        'rsi': float(div['RSI']),
                        'volume': int(div['Volume']),
                        'details': div['Details']
                    })
            
            if rsi_volume_results['bearish_divergences']:
                for div in rsi_volume_results['bearish_divergences']:
                    bearish_divs_formatted.append({
                        'type': div['Type'],
                        'date': div['Date'].strftime('%Y-%m-%d'),
                        'price': float(div['Price']),
                        'rsi': float(div['RSI']),
                        'volume': int(div['Volume']),
                        'details': div['Details']
                    })
            
            if rsi_volume_results['early_reversals']:
                for rev in rsi_volume_results['early_reversals']:
                    early_reversals_formatted.append({
                        'type': rev['Type'],
                        'date': rev['Date'].strftime('%Y-%m-%d'),
                        'price': float(rev['Price']),
                        'rsi': float(rev['RSI']),
                        'volume': int(rev['Volume']),
                        'details': rev['Details']
                    })
            
            response_data['rsi_volume'] = {
                'current_rsi': round(rsi_volume_results['current_rsi'], 2),
                'current_volume': int(rsi_volume_results['current_volume']),
                'volume_ma_20': int(rsi_volume_results['volume_ma_20']),
                'volume_ma_50': int(rsi_volume_results['volume_ma_50']),
                'bullish_divergences': bullish_divs_formatted,
                'bearish_divergences': bearish_divs_formatted,
                'early_reversals': early_reversals_formatted,
                'chart_image': rsi_volume_image_base64
            }
        
        # Run Volatility Squeeze Analysis
        if analysis_type in ['all', 'volatility_squeeze']:
            print(f"Starting Volatility Squeeze analysis for {ticker}...")
            volatility_squeeze_results = run_volatility_squeeze_analysis(ticker=ticker, show_plot=False, config=volatility_squeeze_config, df=df)
            print(f"Volatility Squeeze analysis result: {volatility_squeeze_results['success']}")
            
            if not volatility_squeeze_results['success']:
                return jsonify(volatility_squeeze_results)
            
            # Convert Volatility Squeeze figure to base64 image
            volatility_squeeze_fig = volatility_squeeze_results['figure']
            volatility_squeeze_buf = BytesIO()
            volatility_squeeze_fig.savefig(volatility_squeeze_buf, format='png', dpi=100, bbox_inches='tight')
            volatility_squeeze_buf.seek(0)
            volatility_squeeze_image_base64 = base64.b64encode(volatility_squeeze_buf.getvalue()).decode('utf-8')
            volatility_squeeze_buf.close()
            print("Volatility Squeeze chart generated successfully")
            
            # Close the figure
            plt.close(volatility_squeeze_fig)
            
            # Format signals for JSON
            signals_formatted = []
            
            if volatility_squeeze_results['signals']:
                for sig in volatility_squeeze_results['signals']:
                    signals_formatted.append({
                        'type': sig['Type'],
                        'date': sig['Date'].strftime('%Y-%m-%d'),
                        'price': float(sig['Price']),
                        'bb_width': float(sig['BB_Width']),
                        'atr': float(sig['ATR'])
                    })
            
            response_data['volatility_squeeze'] = {
                'current_bb_width': round(volatility_squeeze_results['current_bb_width'], 4) if volatility_squeeze_results['current_bb_width'] else None,
                'current_atr': round(volatility_squeeze_results['current_atr'], 2) if volatility_squeeze_results['current_atr'] else None,
                'signals': signals_formatted,
                'chart_image': volatility_squeeze_image_base64
            }
        
        # Run RS Analysis
        if analysis_type in ['all', 'rs']:
            print(f"Starting RS analysis for {ticker}...")
            
            # Get benchmark from config
            benchmark = rs_config.get('BENCHMARK_TICKER', None)
            
            rs_results = run_rs_analysis(
                ticker=ticker, 
                benchmark=benchmark,
                show_plot=False, 
                config=rs_config, 
                use_sector_index=use_sector_index,
                df=df,
                df_benchmark=df_benchmark
            )
            print(f"RS analysis result: {rs_results['success']}")
            
            if not rs_results['success']:
                return jsonify(rs_results)
            
            # Convert RS figure to base64 image
            rs_fig = rs_results['figure']
            rs_buf = BytesIO()
            rs_fig.savefig(rs_buf, format='png', dpi=100, bbox_inches='tight')
            rs_buf.seek(0)
            rs_image_base64 = base64.b64encode(rs_buf.getvalue()).decode('utf-8')
            rs_buf.close()
            print("RS chart generated successfully")
            
            # Close the figure
            plt.close(rs_fig)
            
            # Format signals for JSON
            signals_formatted = []
            
            if rs_results['signals']:
                for sig in rs_results['signals']:
                    signals_formatted.append({
                        'type': sig['type'],
                        'date': sig['date'],
                        'description': sig['description']
                    })
            
            response_data['rs'] = {
                'benchmark': rs_results['benchmark'],
                'sector': rs_results.get('sector', None),
                'rs_ratios': rs_results['rs_ratios'],
                'rs_score': round(rs_results['rs_score'], 1),
                'classification': rs_results['classification'],
                'trading_summary': rs_results['trading_summary'],
                'signals': signals_formatted,
                'chart_image': rs_image_base64
            }
        
        # Run Multi-Timeframe Analysis
        if analysis_type in ['all', 'multi_timeframe']:
            print(f"Starting Multi-Timeframe analysis for {ticker}...")
            multi_tf_results = run_multi_timeframe_analysis(ticker=ticker, show_plot=False)
            print(f"Multi-Timeframe analysis result: {multi_tf_results['success']}")
            
            if not multi_tf_results['success']:
                return jsonify(multi_tf_results)
            
            # Convert candlestick figure to base64 image
            multi_tf_fig = multi_tf_results['figure']
            multi_tf_buf = BytesIO()
            multi_tf_fig.savefig(multi_tf_buf, format='png', dpi=100, bbox_inches='tight')
            multi_tf_buf.seek(0)
            multi_tf_image_base64 = base64.b64encode(multi_tf_buf.getvalue()).decode('utf-8')
            multi_tf_buf.close()
            print("Multi-Timeframe chart generated successfully")
            
            # Close the figure
            plt.close(multi_tf_fig)
            
            response_data['multi_timeframe'] = {
                'supertrend_results': multi_tf_results['supertrend_results'],
                'macd_results': multi_tf_results['macd_results'],
                'chart_image': multi_tf_image_base64
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    """Get list of available benchmark indices from tickers_grouped.json"""
    try:
        # Load tickers_grouped.json
        tickers_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickers_grouped.json')
        
        if not os.path.exists(tickers_file):
            return jsonify({'success': False, 'error': 'tickers_grouped.json not found'})
            
        import json
        with open(tickers_file, 'r') as f:
            data = json.load(f)
            
        benchmarks = []
        
        # Extract indices from "Sector" -> "stocks"
        if "Sector" in data and "stocks" in data["Sector"]:
            for name, symbol in data["Sector"]["stocks"].items():
                benchmarks.append({
                    'name': name,
                    'symbol': symbol
                })
        
        # Add standard benchmarks if not present
        standard_benchmarks = [
            {'name': 'Nifty 50', 'symbol': '^NSEI'},
            {'name': 'S&P 500', 'symbol': '^GSPC'}
        ]
        
        for std in standard_benchmarks:
            if not any(b['symbol'] == std['symbol'] for b in benchmarks):
                benchmarks.append(std)
                
        return jsonify({'success': True, 'benchmarks': benchmarks})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sectors', methods=['GET'])
def get_sectors():
    """Get list of available sectors from tickers_grouped.json"""
    try:
        # Load tickers_grouped.json
        tickers_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickers_grouped.json')
        
        if not os.path.exists(tickers_file):
            return jsonify({'success': False, 'error': 'tickers_grouped.json not found'})
            
        import json
        with open(tickers_file, 'r') as f:
            data = json.load(f)
            
        sectors = []
        
        # Extract all sectors except "Sector" which is for indices
        for sector_name, sector_data in data.items():
            if sector_name != "Sector":
                sectors.append({
                    'name': sector_name,
                    'index_symbol': sector_data['index_symbol']
                })
        
        # Check for optional batch analysis file
        batch_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickers_batch_analysis.json')
        if os.path.exists(batch_file):
            sectors.append({
                'name': 'Custom Batch List',
                'index_symbol': '^NSEI' # Default fallback
            })
        
        return jsonify({'success': True, 'sectors': sectors})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market_analysis/sector', methods=['GET'])
def get_sector_analysis():
    """Run and return Sector Analysis results"""
    try:
        results = run_sector_analysis(show_plot=False)
        
        if not results['success']:
            return jsonify(results)
            
        # Convert figure to base64
        fig = results['figure']
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        # Format table data
        df = results['results']
        table_data = []
        
        # Columns to include
        cols = ['1M', '3M', '6M', '1Y', 'Consistent', 'Emerging', 'Early_Turnaround', 'MA_Breakout', 'Volume_Surge', 'Score']
        
        for idx, row in df.iterrows():
            item = {'Sector': idx}
            for col in cols:
                if col in row:
                    val = row[col]
                    # Handle boolean values
                    if isinstance(val, bool) or isinstance(val, np.bool_):
                        item[col] = bool(val)
                    # Handle float values
                    elif isinstance(val, float) or isinstance(val, np.float64):
                        item[col] = round(float(val), 3)
                    else:
                        item[col] = val
            table_data.append(item)
            
        return jsonify({
            'success': True,
            'chart_image': image_base64,
            'data': table_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market_analysis/stocks_in_sector', methods=['POST'])
def get_stocks_in_sector_analysis():
    """Run and return Stocks in Sector Analysis for a given sector"""
    try:
        data = request.get_json()
        sector_name = data.get('sector', '').strip()
        
        if not sector_name:
            return jsonify({'success': False, 'error': 'Please provide a sector name'})
        
        results = run_stock_in_sector_analysis(sector_name, show_plot=False)
        
        if not results['success']:
            return jsonify(results)
            
        # Convert figure to base64
        fig = results['figure']
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        # Format table data
        df = results['results']
        table_data = []
        
        # Columns to include
        cols = ['1M', '3M', '6M', '1Y', 'Consistent', 'Emerging', 'Early_Turnaround', 'MA_Breakout', 'Volume_Surge', 'Score']
        
        for idx, row in df.iterrows():
            item = {'Stock': idx}
            for col in cols:
                if col in row:
                    val = row[col]
                    # Handle boolean values
                    if isinstance(val, bool) or isinstance(val, np.bool_):
                        item[col] = bool(val)
                    # Handle float values
                    elif isinstance(val, float) or isinstance(val, np.float64):
                        item[col] = round(float(val), 3)
                    else:
                        item[col] = val
            table_data.append(item)
            
        return jsonify({
            'success': True,
            'chart_image': image_base64,
            'data': table_data,
            'sector_name': results['sector_name'],
            'sector_index': results['sector_index']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/fundamental-analysis', methods=['POST'])
def get_fundamental_analysis():
    """Run fundamental analysis on a stock ticker"""
    
    def convert_to_json_serializable(obj):
        """Convert numpy/pandas types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalar types
            return obj.item()
        elif obj is None:
            return None
        else:
            return obj
    
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper()
        
        if not ticker:
            return jsonify({'success': False, 'error': 'Please enter a valid ticker symbol'})
        
        print(f"Starting fundamental analysis for {ticker}...")
        results = run_fundamental_analysis(ticker)
        print(f"Fundamental analysis result: {results['success']}")
        
        if not results['success']:
            return jsonify(results)
        
        # Convert all numpy/pandas types to native Python types
        formatted_response = convert_to_json_serializable({
            'success': True,
            'ticker': results['ticker'],
            'analysis_date': results['analysis_date'],
            'long_term': results['long_term'],
            'short_term': results['short_term']
        })
        
        return jsonify(formatted_response)
        
    except Exception as e:
        import traceback
        print(f"Error in fundamental analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/batch_analysis', methods=['POST'])
def get_batch_analysis():
    """Run Batch Analysis using tickers_batch_analysis.json"""
    try:
        # Get request data
        data = request.get_json() or {}
        interval = data.get('interval', '1d')  # Default to daily
        
        # Always use the dedicated batch file
        batch_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickers_batch_analysis.json')
        
        if not os.path.exists(batch_file):
             return jsonify({'success': False, 'error': 'tickers_batch_analysis.json not found'})
             
        import json
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            tickers = list(batch_data.values())
            
        if not tickers:
            return jsonify({'success': False, 'error': 'No tickers found in batch file'})
            
        # Limit batch size to prevent timeout/overload
        MAX_BATCH = 50
        if len(tickers) > MAX_BATCH:
            tickers = tickers[:MAX_BATCH]
            
        # Run batch analysis with interval parameter
        results = run_batch_analysis(tickers, interval=interval)
        
        return jsonify({
            'success': True,
            'data': results,
            'interval': interval
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download_report', methods=['GET'])
def download_report():
    """Generate and serve the stock detailed report PDF"""
    from flask import send_file
    import tempfile
    
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'success': False, 'error': 'Ticker is required'}), 400
    
    try:
        ticker = ticker.strip().upper()
        # Use a fixed temporary directory or the reports directory
        # The generate_stock_report fn has a default output_dir, but we might want to segregate web downloads
        # Let's use a specific directory in the project for easier persistence/debugging if needed, 
        # or just use the system temp if we want to be clean.
        
        # Using a 'downloads' folder in website_ui to keep it contained
        download_dir = os.path.join(os.path.dirname(__file__), 'downloads')
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
            
        # Generate the report
        print(f"Generating PDF report for {ticker}...")
        pdf_path = generate_stock_report(ticker, output_dir=download_dir)
        
        if not os.path.exists(pdf_path):
             return jsonify({'success': False, 'error': 'Failed to generate PDF report'}), 500
             
        # Serve the file
        # as_attachment=True forces download
        # download_name is available in newer flask, or use attachment_filename in older
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=os.path.basename(pdf_path),
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
