"""
Enhanced script to generate demo transactions with fraud detection
This will populate your dashboard with realistic data including fraud alerts
"""
import requests
import random
import numpy as np
from datetime import datetime, timedelta
import time

API_URL = "http://localhost:8000/api/v1"

# Indian merchants
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

def generate_features(amount, time_hour, merchant_category, is_intended_fraud=False):
    """Generate 30 features for ML model (simplified)"""
    # Generate V1-V28 (PCA features) - random but correlated with amount
    features = {}
    features["time"] = time_hour * 3600  # Convert to seconds
    
    # Generate V1-V28 features with realistic patterns
    for i in range(1, 29):
        # Base value from normal distribution
        base_value = np.random.randn()
        
        # Only adjust features if this is INTENDED to be fraud
        if is_intended_fraud:
            # Adjust based on fraud indicators
            if amount > 100000:  # Very high amount - likely fraud
                base_value += random.uniform(1.5, 3.0)
            elif amount > 50000:  # High amount - suspicious
                base_value += random.uniform(0.5, 1.5)
            elif amount < 10:  # Very low amount - suspicious
                base_value += random.uniform(-2.0, -0.5)
            
            # Late night transactions are more suspicious
            if time_hour < 5 or time_hour > 23:
                base_value += random.uniform(0.3, 1.0)
        else:
            # Normal transactions - keep features neutral
            base_value += random.uniform(-0.2, 0.2)
        
        features[f"v{i}"] = round(base_value, 6)
    
    features["amount"] = amount
    return features

def create_transaction_with_fraud_check(user_id, amount, merchant_name, merchant_category, location, description, time_hour, is_intended_fraud=False):
    """Create transaction and run fraud detection"""
    
    # Step 1: Create transaction
    transaction_data = {
        "user_id": user_id,
        "amount": amount,
        "merchant_name": merchant_name,
        "merchant_category": merchant_category,
        "location": location,
        "description": description
    }
    
    try:
        # Create transaction
        response = requests.post(f"{API_URL}/transactions/", json=transaction_data, timeout=5)
        response.raise_for_status()
        transaction = response.json()
        transaction_id = transaction["id"]
        
        # Step 2: Generate features for fraud detection
        features = generate_features(amount, time_hour, merchant_category, is_intended_fraud)
        
        # Step 3: Run fraud prediction
        fraud_data = {
            "transaction_id": transaction_id,
            **features
        }
        
        try:
            fraud_response = requests.post(
                f"{API_URL}/fraud/predict",
                json=fraud_data,
                timeout=10
            )
            
            if fraud_response.ok:
                fraud_result = fraud_response.json()
                is_fraud = fraud_result.get("is_fraud", False)
                probability = fraud_result.get("fraud_probability", 0)
                
                # Step 4: Update transaction with fraud results
                update_response = requests.patch(
                    f"{API_URL}/transactions/{transaction_id}/fraud-status",
                    params={
                        "is_fraud": is_fraud,
                        "fraud_probability": probability,
                        "risk_score": probability
                    },
                    timeout=5
                )
                
                status_icon = "🚨" if is_fraud else "✅"
                print(f"{status_icon} Transaction {transaction_id[:8]}: ₹{amount:,.2f} at {merchant_name}, {location}")
                print(f"   Fraud: {is_fraud} | Probability: {probability:.2%} | Model: {fraud_result.get('model_used', 'N/A')}")
                
                if is_fraud:
                    risk_factors = fraud_result.get("risk_factors", [])
                    if risk_factors:
                        print(f"   Risk Factors: {', '.join(risk_factors)}")
                
                return transaction, fraud_result
            else:
                print(f"⚠️  Transaction created but fraud check failed: {transaction_id[:8]}")
                print(f"   Error: {fraud_response.text}")
                return transaction, None
                
        except Exception as e:
            print(f"⚠️  Transaction created but fraud check error: {str(e)}")
            return transaction, None
            
    except requests.exceptions.Timeout:
        print(f"✗ Request timeout - is the backend running?")
        return None, None
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error - cannot reach backend")
        return None, None
    except Exception as e:
        print(f"✗ Failed to create transaction: {e}")
        return None, None

def generate_demo_data(count=50):
    """Generate demo transactions with varied patterns"""
    print(f"\n🚀 Generating {count} demo transactions with fraud detection...\n")
    
    stats = {
        "total": 0,
        "fraud": 0,
        "suspicious": 0,
        "legitimate": 0,
        "errors": 0
    }
    
    for i in range(count):
        # Random transaction data
        merchant = random.choice(MERCHANTS)
        location = random.choice(LOCATIONS)
        user_id = f"user_{random.randint(1, 100)}"
        time_hour = random.randint(0, 23)
        
        # Generate amount with realistic distribution
        rand = random.random()
        is_intended_fraud = False
        
        if rand < 0.75:  # 75% normal/safe transactions
            amount = round(random.uniform(100, 5000), 2)
            time_hour = random.randint(8, 22)  # Normal business hours
            is_intended_fraud = False
            
        elif rand < 0.90:  # 15% medium transactions (might be suspicious)
            amount = round(random.uniform(5000, 25000), 2)
            time_hour = random.randint(6, 23)
            is_intended_fraud = False
            
        elif rand < 0.95:  # 5% high transactions (suspicious)
            amount = round(random.uniform(25000, 100000), 2)
            time_hour = random.choice([1, 2, 3, 22, 23])  # Late hours
            is_intended_fraud = True  # Mark as intended fraud
            
        else:  # 5% very high (likely fraud)
            amount = round(random.uniform(100000, 500000), 2)
            time_hour = random.choice([0, 1, 2, 3, 4])  # Very late night
            is_intended_fraud = True  # Mark as intended fraud
        
        description = f"Purchase at {merchant['name']}, {location}"
        
        # Create transaction with fraud check
        transaction, fraud_result = create_transaction_with_fraud_check(
            user_id=user_id,
            amount=amount,
            merchant_name=merchant["name"],
            merchant_category=merchant["category"],
            location=location,
            description=description,
            time_hour=time_hour,
            is_intended_fraud=is_intended_fraud
        )
        
        if transaction:
            stats["total"] += 1
            if fraud_result:
                prob = fraud_result.get("fraud_probability", 0)
                if fraud_result.get("is_fraud") or prob > 0.7:
                    stats["fraud"] += 1
                elif prob > 0.3:
                    stats["suspicious"] += 1
                else:
                    stats["legitimate"] += 1
            else:
                stats["legitimate"] += 1
        else:
            stats["errors"] += 1
        
        print()  # Empty line
        time.sleep(0.1)  # Small delay to avoid overwhelming the server
    
    # Print summary
    print("\n" + "="*60)
    print("📊 GENERATION SUMMARY")
    print("="*60)
    print(f"✅ Total Transactions: {stats['total']}")
    print(f"✅ Legitimate (Safe): {stats['legitimate']} ({stats['legitimate']/stats['total']*100:.1f}%)")
    print(f"⚠️  Suspicious (Medium): {stats['suspicious']} ({stats['suspicious']/stats['total']*100:.1f}%)")
    print(f"🚨 Fraud (High Risk): {stats['fraud']} ({stats['fraud']/stats['total']*100:.1f}%)")
    if stats['errors'] > 0:
        print(f"❌ Errors: {stats['errors']}")
    print("="*60)
    print(f"\n📊 Check your dashboard at http://localhost:5173")
    print(f"🇮🇳 All amounts in Indian Rupees (₹)")
    print(f"\n💡 Realistic Distribution:")
    print(f"   - Most transactions are safe (normal shopping)")
    print(f"   - Some are suspicious (high amounts)")
    print(f"   - Few are fraud (very high amounts at odd times)")

if __name__ == "__main__":
    import sys
    
    # Get count from command line or use default
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    print("=" * 60)
    print("  QUANTUM FRAUD DETECTION - Demo Data Generator")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_URL.replace('/api/v1', '')}/health", timeout=5)
        if response.ok:
            print("✓ Backend API is running")
        else:
            print("✗ Backend API is not responding correctly")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Cannot connect to backend API")
        print(f"  Make sure the backend is running:")
        print(f"  cd backend && python -m uvicorn app.main:app --reload")
        sys.exit(1)
    
    generate_demo_data(count)
