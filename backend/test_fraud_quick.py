import requests
import random
from datetime import datetime

API_URL = "http://localhost:8000/api/v1"

# Test with high-value transaction (should be fraud)
print("Testing HIGH VALUE transaction (should be FRAUD)...")
high_value_data = {
    "transaction_id": "test_high_001",
    "time": datetime.now().isoformat(),
    "amount": 150000.00,  # High amount - should trigger fraud
    **{f"v{i}": random.uniform(2, 4) for i in range(1, 29)}
}

response = requests.post(f"{API_URL}/fraud/predict", json=high_value_data)
print(f"Status: {response.status_code}")
if response.ok:
    result = response.json()
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Probability: {result['fraud_probability']:.2%}")
    print(f"Model: {result['model_used']}")
else:
    print(f"Error: {response.text}")

print("\n" + "="*50 + "\n")

# Test with low-value transaction (should be safe)
print("Testing LOW VALUE transaction (should be SAFE)...")
low_value_data = {
    "transaction_id": "test_low_001",
    "time": datetime.now().isoformat(),
    "amount": 500.00,  # Low amount - should be safe
    **{f"v{i}": random.uniform(-0.5, 0.5) for i in range(1, 29)}
}

response = requests.post(f"{API_URL}/fraud/predict", json=low_value_data)
print(f"Status: {response.status_code}")
if response.ok:
    result = response.json()
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Probability: {result['fraud_probability']:.2%}")
    print(f"Model: {result['model_used']}")
else:
    print(f"Error: {response.text}")
