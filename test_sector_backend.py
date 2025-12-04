
import sys
import os

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'market_analysis'))

try:
    from sector_analysis import run_analysis as run_sector_analysis
    print("Successfully imported run_sector_analysis")
    
    result = run_sector_analysis(show_plot=False)
    
    if result['success']:
        print("Sector Analysis Run Successful")
        print("Keys in result:", result.keys())
        print("Data rows:", len(result['results']))
        print("Figure object:", result['figure'])
    else:
        print("Sector Analysis Failed:", result.get('error'))

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
