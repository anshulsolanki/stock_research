import requests
import json
import time

def test_analysis():
    url = 'http://127.0.0.1:5000/analyze'
    
    # Test Case 1: Default parameters (1d, 365)
    print("Testing Default Parameters...")
    payload_default = {
        "ticker": "TCS.NS",
        "macd_config": {"FAST": 12, "SLOW": 26, "SIGNAL": 9, "INTERVAL": "1d", "LOOKBACK_PERIODS": 365},
        "supertrend_config": {"PERIOD": 10, "MULTIPLIER": 3.0, "INTERVAL": "1d", "LOOKBACK_PERIODS": 365}
    }
    try:
        response = requests.post(url, json=payload_default)
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("  Success! Default parameters working.")
                print(f"  MACD Date: {data['macd']['divergences'][-1]['date'] if data['macd']['divergences'] else 'No divergences'}")
                print(f"  Supertrend Date: {data['supertrend']['last_date']}")
            else:
                print(f"  Failed: {data.get('error')}")
        else:
            print(f"  Error: Status Code {response.status_code}")
    except Exception as e:
        print(f"  Exception: {e}")

    print("-" * 30)

    # Test Case 2: Weekly Interval, Longer Lookback
    print("Testing Weekly Interval & 730 Days Lookback...")
    payload_weekly = {
        "ticker": "TCS.NS",
        "macd_config": {"FAST": 12, "SLOW": 26, "SIGNAL": 9, "INTERVAL": "1wk", "LOOKBACK_PERIODS": 730},
        "supertrend_config": {"PERIOD": 10, "MULTIPLIER": 3.0, "INTERVAL": "1wk", "LOOKBACK_PERIODS": 730}
    }
    try:
        response = requests.post(url, json=payload_weekly)
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("  Success! Weekly parameters working.")
                # Verify dates are roughly weekly or just check success for now
                print(f"  Supertrend Date: {data['supertrend']['last_date']}")
            else:
                print(f"  Failed: {data.get('error')}")
        else:
            print(f"  Error: Status Code {response.status_code}")
    except Exception as e:
        print(f"  Exception: {e}")

if __name__ == "__main__":
    # Wait a bit for server to start if running immediately after
    time.sleep(2)
    test_analysis()
