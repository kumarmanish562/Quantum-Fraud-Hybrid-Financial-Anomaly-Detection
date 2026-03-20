"""
Script to add sample transactions to the system via API
Run this to populate your dashboard with real transaction data
"""
import requests
import random
from datetime import datetime, timedelta

API_URL = "http://localhost:8000/api/v1"

# Sample merchant data (Indian merchants)
MERCHANTS = [
    {"name": "Big Bazaar", "category": "grocery"},
    {"name": "Reliance Digital", "category": "retail"},
    {"name": "Flipkart", "category": "online"},
    {"name": "Barbeque Nation", "category": "restaurant"},
    {"name": "Amazon India", "category": "online"},
    {"name": "Indian Oil Petrol Pump", "category": "gas"},
    {"name": "Cafe Coffee Day", "category": "restaurant"},
    {"name": "Westside Fashion", "category": "retail"},
    {"name": "Croma Electronics", "category": "retail"},
    {"name": "Myntra", "category": "online"},
    {"name": "DMart", "category": "grocery"},
    {"name": "Swiggy", "category": "food_delivery"},
    {"name": "Zomato", "category": "food_delivery"},
    {"name": "BookMyShow", "category": "entertainment"},
    {"name": "PVR Cinemas", "category": "entertainment"},
    {"name": "Tanishq Jewellers", "category": "jewelry"},
    {"name": "Apollo Pharmacy", "category": "pharmacy"},
    {"name": "Decathlon Sports", "category": "sports"},
    {"name": "Haldiram's", "category": "restaurant"},
    {"name": "Paytm Mall", "category": "online"},
]

LOCATIONS = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", 
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Indore", "Kochi", "Surat", "Nagpur"
]

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
        # Indian rupee amounts - realistic ranges
        amount = round(random.uniform(50, 15000), 2)
        
        # Make some transactions suspicious (high amounts)
        if random.random() < 0.1:  # 10% chance of suspicious transaction
            amount = round(random.uniform(50000, 500000), 2)
        
        # Some very small transactions (also suspicious)
        if random.random() < 0.05:  # 5% chance of micro transaction
            amount = round(random.uniform(1, 10), 2)
        
        user_id = f"user_{random.randint(1, 100)}"
        location = random.choice(LOCATIONS)
        description = f"Purchase at {merchant['name']}, {location}"
        
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
    print(f"🇮🇳 Locations: {', '.join(LOCATIONS[:5])} and more...")

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
