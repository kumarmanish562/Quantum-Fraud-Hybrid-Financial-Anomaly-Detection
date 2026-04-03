import requests
import json

# Test registration
url = "http://localhost:8000/api/v1/user-auth/register"
data = {
    "email": "testuser@example.com",
    "password": "test123",
    "full_name": "Test User",
    "company_name": "Test Company",
    "phone": "+91 9876543210"
}

print("🧪 Testing Registration API...")
print(f"URL: {url}")
print(f"Data: {json.dumps(data, indent=2)}")

try:
    response = requests.post(url, json=data)
    print(f"\n✅ Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        print("\n✅ Registration successful!")
        print(f"OTP: {response.json().get('otp')}")
except Exception as e:
    print(f"\n❌ Error: {e}")
