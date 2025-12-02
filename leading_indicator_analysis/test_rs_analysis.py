"""
Quick test script for RS Analysis - runs without interactive plot.
"""
import sys
sys.path.append('/Users/solankianshul/Documents/projects/stock_research/leading_indicator_analysis')

from rs_analysis import run_analysis

# Test 1: Indian Stock (should detect Nifty 50 as benchmark)
print("="*70)
print("TEST 1: Indian Stock - RELIANCE.NS")
print("="*70)
result1 = run_analysis("RELIANCE.NS", show_plot=False)

if result1['success']:
    print(f"✓ Analysis completed successfully")
    print(f"✓ Benchmark auto-detected: {result1['benchmark']}")
    print(f"✓ Classification: {result1['classification']}")
    print(f"✓ RS Score: {result1['rs_score']:.1f}/100")
    print(f"✓ RS Ratios:")
    for period, value in result1['rs_ratios'].items():
        if value:
            status = "📈" if value > 1.0 else "📉"
            print(f"    {period}: {value:.3f} {status}")
    print(f"✓ Signals detected: {len(result1['signals'])}")
else:
    print(f"✗ Analysis failed: {result1.get('error', 'Unknown error')}")

# Test 2: US Stock (should detect S&P 500 as benchmark)
print("\n" + "="*70)
print("TEST 2: US Stock - AAPL")
print("="*70)
result2 = run_analysis("AAPL", show_plot=False)

if result2['success']:
    print(f"✓ Analysis completed successfully")
    print(f"✓ Benchmark auto-detected: {result2['benchmark']}")
    print(f"✓ Classification: {result2['classification']}")
    print(f"✓ RS Score: {result2['rs_score']:.1f}/100")
    print(f"✓ RS Ratios:")
    for period, value in result2['rs_ratios'].items():
        if value:
            status = "📈" if value > 1.0 else "📉"
            print(f"    {period}: {value:.3f} {status}")
    print(f"✓ Signals detected: {len(result2['signals'])}")
else:
    print(f"✗ Analysis failed: {result2.get('error', 'Unknown error')}")

print("\n" + "="*70)
print("TESTS COMPLETED")
print("="*70)
