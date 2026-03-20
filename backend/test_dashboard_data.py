"""
Quick test script to verify dashboard data flow
"""
import requests
import time

API_URL = "http://localhost:8000/api/v1"

def test_api_connection():
    """Test if API is running"""
    try:
        response = requests.get(f"{API_URL.replace('/api/v1', '')}/health", timeout=5)
        if response.ok:
            print("✅ Backend API is running")
            return True
        else:
            print("❌ Backend API not responding")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False

def create_test_transaction():
    """Create a single test transaction"""
    data = {
        "user_id": "test_user_1",
        "amount": 15000.00,
        "merchant_name": "Test Merchant",
        "merchant_category": "retail",
        "location": "Mumbai",
        "description": "Test transaction"
    }
    
    try:
        response = requests.post(f"{API_URL}/transactions/", json=data, timeout=5)
        response.raise_for_status()
        transaction = response.json()
        print(f"✅ Created transaction: {transaction['id'][:8]}")
        return transaction
    except Exception as e:
        print(f"❌ Failed to create transaction: {e}")
        return None

def get_transactions():
    """Get all transactions"""
    try:
        response = requests.get(f"{API_URL}/transactions/", timeout=5)
        response.raise_for_status()
        transactions = response.json()
        print(f"✅ Retrieved {len(transactions)} transactions")
        return transactions
    except Exception as e:
        print(f"❌ Failed to get transactions: {e}")
        return []

def get_dashboard_metrics():
    """Get dashboard metrics"""
    try:
        response = requests.get(f"{API_URL}/analytics/dashboard", timeout=5)
        response.raise_for_status()
        metrics = response.json()
        print(f"✅ Dashboard metrics:")
        print(f"   Total transactions today: {metrics.get('total_transactions_today', 0)}")
        print(f"   Fraud detected today: {metrics.get('fraud_detected_today', 0)}")
        print(f"   Fraud rate: {metrics.get('fraud_rate_today', 0)}%")
        return metrics
    except Exception as e:
        print(f"❌ Failed to get dashboard metrics: {e}")
        return None

def main():
    print("="*60)
    print("  DASHBOARD DATA FLOW TEST")
    print("="*60)
    print()
    
    # Test 1: API connection
    print("Test 1: API Connection")
    if not test_api_connection():
        print("\n❌ Backend not running. Start it with:")
        print("   cd backend && python -m uvicorn app.main:app --reload")
        return
    print()
    
    # Test 2: Create transaction
    print("Test 2: Create Transaction")
    transaction = create_test_transaction()
    if not transaction:
        return
    print()
    
    # Test 3: Retrieve transactions
    print("Test 3: Retrieve Transactions")
    transactions = get_transactions()
    print()
    
    # Test 4: Dashboard metrics
    print("Test 4: Dashboard Metrics")
    metrics = get_dashboard_metrics()
    print()
    
    print("="*60)
    if transactions and metrics:
        print("✅ ALL TESTS PASSED!")
        print("   Your dashboard should now show data")
        print("   Visit: http://localhost:5173")
    else:
        print("❌ SOME TESTS FAILED")
        print("   Check the errors above")
    print("="*60)

if __name__ == "__main__":
    main()
