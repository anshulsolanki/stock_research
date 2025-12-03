from flask import Flask, render_template, request, jsonify
import sys
import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path to import analysis modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lagging_indicator_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'leading_indicator_analysis'))

from macd_analysis import run_analysis as run_macd_analysis
from supertrend_analysis import run_analysis as run_supertrend_analysis
from bollinger_band_analysis import run_analysis as run_bollinger_analysis
from crossover_analysis import run_analysis as run_crossover_analysis
from donchian_channel_analysis import run_analysis as run_donchian_analysis
from rsi_divergence_analysis import run_analysis as run_rsi_analysis
from rsi_volume_divergence import run_analysis as run_rsi_volume_analysis
from volatility_squeeze_analysis import run_analysis as run_volatility_squeeze_analysis
from rs_analysis import run_analysis as run_rs_analysis

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')

@app.route('/batch')
def batch_analysis():
    """Serve the batch analysis page"""
    return render_template('batch_analysis.html')

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
        use_sector_index = data.get('use_sector_index', False)
        
        if not ticker:
            return jsonify({'success': False, 'error': 'Please enter a valid ticker symbol'})
        
        response_data = {
            'success': True,
            'ticker': ticker,
            'analysis_type': analysis_type
        }

        # Run MACD analysis
        if analysis_type in ['all', 'macd']:
            macd_results = run_macd_analysis(ticker=ticker, show_plot=False, config=macd_config)
            
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
            supertrend_results = run_supertrend_analysis(ticker=ticker, show_plot=False, config=supertrend_config)
            
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
                'chart_image': supertrend_image_base64
            }
        
        # Run Bollinger Band analysis
        if analysis_type in ['all', 'bollinger']:
            bollinger_results = run_bollinger_analysis(ticker=ticker, show_plot=False, config=bollinger_config)
            
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
            crossover_results = run_crossover_analysis(ticker=ticker, show_plot=False, config=crossover_config)
            
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
            donchian_results = run_donchian_analysis(ticker=ticker, show_plot=False, config=donchian_config)
            
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

        # Run RSI Divergence Analysis
        if analysis_type in ['all', 'rsi']:
            print(f"Starting RSI analysis for {ticker}...")
            rsi_results = run_rsi_analysis(ticker=ticker, show_plot=False, config=rsi_config)
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
            rsi_volume_results = run_rsi_volume_analysis(ticker=ticker, show_plot=False, config=rsi_volume_config)
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
            volatility_squeeze_results = run_volatility_squeeze_analysis(ticker=ticker, show_plot=False, config=volatility_squeeze_config)
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
                use_sector_index=use_sector_index
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

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
