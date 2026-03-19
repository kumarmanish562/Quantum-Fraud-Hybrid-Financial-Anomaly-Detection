"""
Script to add sample transactions to the system via API
Run this to populate your dashboard with real transaction data
"""
import requests
import random
from datetime import datetime, timedelta

API_URL = "http://localhost:8000/api/v1"

# Sample merchant data
MERCHANTS = [
    {"name": "Supermarket", "category": "grocery"},
    {"name": "North Star Logistics", "category": "retail"},
    {"name": "Online Store", "category": "online"},
    {"name": "Velvet Lounge Paris", "category": "restaurant"},
    {"name": "Tech Store Online", "category": "online"},
    {"name": "Gas Station 24/7", "category": "gas"},
    {"name": "Coffee Shop", "category": "restaurant"},
    {"name": "Fashion Retail", "category": "retail"},
    {"name": "Electronics Hub", "category": "retail"},
    {"name": "CryptoX Direct", "category": "online"},
]

LOCATIONS = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Miami", "Seattle"]

def create_transaction(user_id, amount, merchant_name, merchant_category, location, description):
    """Create a transaction via API"""
    data = {
        "user_id": user_id,
        "amount": amount,
        "merchant_name": merchant_name,
        "merchant_category": merchant_category,
        "location": location,
        "description": description
    }
    
    try:
        response = requests.post(f"{API_URL}/transactions/", json=data, timeout=5)
        response.raise_for_status()
        transaction = response.json()
        print(f"✓ Created transaction: {transaction['id'][:8]} - ₹{amount:.2f} at {merchant_name}")
        return transaction
    except requests.exceptions.Timeout:
        print(f"✗ Request timeout - is the backend running?")
        return None
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error - cannot reach backend at {API_URL}")
        return None
    except Exception as e:
        print(f"✗ Failed to create transaction: {e}")
        return None

def predict_fraud(transaction_id, amount, merchant_category):
    """Predict fraud for a transaction"""
    # Note: This is a simplified prediction call
    # The actual fraud prediction endpoint may require different parameters
    print(f"  ℹ️  Fraud prediction skipped (implement based on your ML model requirements)")
    return None

def generate_sample_transactions(count=20):
    """Generate sample transactions"""
    print(f"\n🚀 Generating {count} sample transactions...\n")
    
    for i in range(count):
        # Random transaction data
        merchant = random.choice(MERCHANTS)
        # Use rupee amounts (multiply by 80 for INR conversion)
        amount = round(random.uniform(400, 400000), 2)
        
        # Make some transactions suspicious (high amounts)
        if random.random() < 0.1:  # 10% chance of suspicious transaction
            amount = round(random.uniform(400000, 1200000), 2)
        
        user_id = f"user_{random.randint(1, 50)}"
        location = random.choice(LOCATIONS)
        description = f"Purchase at {merchant['name']}"
        
        # Create transaction
        transaction = create_transaction(
            user_id=user_id,
            amount=amount,
            merchant_name=merchant["name"],
            merchant_category=merchant["category"],
            location=location,
            description=description
        )
        
        if transaction:
            # Predict fraud (optional - implement based on your needs)
            # predict_fraud(
            #     transaction_id=transaction["id"],
            #     amount=amount,
            #     merchant_category=merchant["category"]
            # )
            pass
        
        print()  # Empty line between transactions

    print(f"\n✅ Finished generating {count} transactions!")
    print(f"📊 Check your dashboard at http://localhost:5173")
    print(f"💰 All amounts are in Indian Rupees (₹)")

if __name__ == "__main__":
    import sys
    
    # Get count from command line or use default
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    
    print("=" * 60)
    print("  QUANTUM FRAUD DETECTION - Transaction Generator")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL.replace('/api/v1', '')}/health")
        if response.ok:
            print("✓ Backend API is running")
        else:
            print("✗ Backend API is not responding correctly")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Cannot connect to backend API at {API_URL}")
        print(f"  Make sure the backend is running: cd backend && python run_server.py")
        sys.exit(1)
    
    generate_sample_transactions(count)
