"""
Test script for RS Analysis with sector index detection
"""
import sys
sys.path.append('/Users/solankianshul/Documents/projects/stock_research/leading_indicator_analysis')

from rs_analysis import run_analysis

print("="*80)
print("TEST: Sector-Based RS Analysis")
print("="*80)

# Test with LT.NS using sector index (should use Infra index ^CNXINFRA)
print("\n1. Testing LT.NS with use_sector_index=True")
print("-" * 80)
result1 = run_analysis("LT.NS", show_plot=False, use_sector_index=True)

if result1['success']:
    print(f"\n✓ Analysis completed successfully")
    print(f"✓ Ticker: {result1['ticker']}")
    print(f"✓ Benchmark: {result1['benchmark']}")
    if 'sector' in result1:
        print(f"✓ Sector: {result1['sector']}")
    print(f"✓ Classification: {result1['classification']}")
    print(f"✓ RS Score: {result1['rs_score']:.1f}/100")
else:
    print(f"✗ Analysis failed: {result1.get('error', 'Unknown error')}")

# Test with same ticker using broad market index
print("\n\n2. Testing LT.NS with use_sector_index=False (Broad Market)")
print("-" * 80)
result2 = run_analysis("LT.NS", show_plot=False, use_sector_index=False)

if result2['success']:
    print(f"\n✓ Analysis completed successfully")
    print(f"✓ Ticker: {result2['ticker']}")
    print(f"✓ Benchmark: {result2['benchmark']}")
    print(f"✓ Classification: {result2['classification']}")
    print(f"✓ RS Score: {result2['rs_score']:.1f}/100")
else:
    print(f"✗ Analysis failed: {result2.get('error', 'Unknown error')}")

# Compare the results
print("\n\n3. Comparison")
print("-" * 80)
if result1['success'] and result2['success']:
    print(f"Sector Index Benchmark: {result1['benchmark']}")
    print(f"Broad Market Benchmark: {result2['benchmark']}")
    print(f"\nSector Index RS Score: {result1['rs_score']:.1f}/100")
    print(f"Broad Market RS Score: {result2['rs_score']:.1f}/100")
    print(f"\nSector Index Classification: {result1['classification']}")
    print(f"Broad Market Classification: {result2['classification']}")

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)
