#!/usr/bin/env python3
"""
Simple test script to verify the API is working
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_auth():
    """Test authentication"""
    print("\n🔐 Testing authentication...")
    try:
        # Login with demo user
        login_data = {
            "username": "demo",
            "password": "demo123"
        }
        response = requests.post(f"{BASE_URL}/api/v1/auth/login", data=login_data)
        if response.status_code == 200:
            token = response.json()["access_token"]
            print(f"✅ Login successful, token: {token[:20]}...")
            return token
        else:
            print(f"❌ Login failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Auth test failed: {e}")
        return None

def test_fraud_prediction():
    """Test fraud prediction endpoint"""
    print("\n🤖 Testing fraud prediction...")
    try:
        # Sample transaction data
        transaction_data = {
            "transaction_id": "test_txn_001",
            "amount": 1500.0,
            "time": datetime.utcnow().isoformat(),
            "v1": 0.5, "v2": -1.2, "v3": 0.8, "v4": -0.3, "v5": 1.1,
            "v6": -0.7, "v7": 0.2, "v8": 0.9, "v9": -0.4, "v10": 0.6,
            "v11": -0.8, "v12": 0.3, "v13": 1.0, "v14": -0.5, "v15": 0.7,
            "v16": -0.2, "v17": 0.4, "v18": -0.9, "v19": 0.1, "v20": 0.8,
            "v21": -0.6, "v22": 0.5, "v23": -0.1, "v24": 0.3, "v25": -0.4,
            "v26": 0.2, "v27": -0.7, "v28": 0.6
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/fraud/predict", json=transaction_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Fraud prediction successful:")
            print(f"   Transaction ID: {result['transaction_id']}")
            print(f"   Is Fraud: {result['is_fraud']}")
            print(f"   Probability: {result['fraud_probability']:.3f}")
            print(f"   Confidence: {result['confidence_score']:.3f}")
            print(f"   Model: {result['model_used']}")
            return True
        else:
            print(f"❌ Fraud prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Fraud prediction test failed: {e}")
        return False

def test_analytics():
    """Test analytics endpoint"""
    print("\n📊 Testing analytics...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/analytics/dashboard")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analytics successful:")
            print(f"   Transactions today: {data['total_transactions_today']}")
            print(f"   Fraud detected: {data['fraud_detected_today']}")
            print(f"   Fraud rate: {data['fraud_rate_today']}%")
            return True
        else:
            print(f"❌ Analytics failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Analytics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting API Tests...")
    print("=" * 50)
    
    # Test basic connectivity
    if not test_health():
        print("\n❌ Basic connectivity failed. Make sure the server is running.")
        return
    
    # Test authentication
    token = test_auth()
    
    # Test fraud prediction
    test_fraud_prediction()
    
    # Test analytics
    test_analytics()
    
    print("\n" + "=" * 50)
    print("✅ API tests completed!")
    print("\n💡 Next steps:")
    print("   - Open http://localhost:8000/docs for interactive API docs")
    print("   - Test WebSocket at ws://localhost:8000/ws/test_client")
    print("   - Check real-time fraud detection with /api/v1/fraud/predict/realtime")

if __name__ == "__main__":
    main()