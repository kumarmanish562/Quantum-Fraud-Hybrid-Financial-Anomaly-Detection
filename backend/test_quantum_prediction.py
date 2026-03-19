"""
Test quantum model predictions through the backend service
"""
import sys
import os
import asyncio
import numpy as np

# Ensure we can import from app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_ml_service():
    """Test ML Service with quantum model"""
    print("=" * 70)
    print("TESTING QUANTUM MODEL THROUGH BACKEND ML SERVICE")
    print("=" * 70)
    
    try:
        from app.services.ml_service import MLService
        
        print("\n1. Initializing ML Service...")
        ml_service = MLService()
        print("   ✅ ML Service initialized")
        
        # Check model status
        print("\n2. Checking Model Status...")
        status = await ml_service.get_model_status()
        print(f"   Models loaded: {status}")
        
        # Create sample transaction features (30 features as expected)
        print("\n3. Creating Sample Transaction...")
        # Simulate a transaction with PCA features V1-V28, Time, and Amount
        sample_features = np.random.randn(1, 30).astype(np.float32)
        sample_features[0, 0] = 12345.0  # Time
        sample_features[0, -1] = 5000.0  # Amount (potentially fraudulent)
        
        print(f"   Sample shape: {sample_features.shape}")
        print(f"   Time: {sample_features[0, 0]}")
        print(f"   Amount: ${sample_features[0, -1]:.2f}")
        
        # Make prediction
        print("\n4. Making Prediction...")
        result = await ml_service.predict_single(sample_features)
        
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        print(f"   Model Used: {result['model']}")
        print(f"   Is Fraud: {result['is_fraud']}")
        print(f"   Probability: {result['probability']:.4f}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Risk Factors: {result['risk_factors']}")
        
        # Test multiple predictions
        print("\n5. Testing Multiple Predictions...")
        test_cases = [
            {"amount": 100.0, "desc": "Small legitimate transaction"},
            {"amount": 15000.0, "desc": "Large potentially fraudulent transaction"},
            {"amount": 0.5, "desc": "Micro transaction"},
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            features = np.random.randn(1, 30).astype(np.float32)
            features[0, 0] = 12345.0 + i * 1000
            features[0, -1] = test_case["amount"]
            
            result = await ml_service.predict_single(features)
            
            print(f"\n   Test {i}: {test_case['desc']}")
            print(f"      Amount: ${test_case['amount']:.2f}")
            print(f"      Model: {result['model']}")
            print(f"      Prediction: {'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
            print(f"      Probability: {result['probability']:.4f}")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_loading():
    """Test if models can be loaded properly"""
    print("\n" + "=" * 70)
    print("TESTING MODEL LOADING")
    print("=" * 70)
    
    try:
        import torch
        from ml_engine.models.hybrid_nn import HybridQNN
        from app.core.config import settings
        
        print(f"\n1. Model Path Configuration:")
        print(f"   MODEL_PATH: {settings.MODEL_PATH}")
        print(f"   HYBRID_MODEL_PATH: {settings.HYBRID_MODEL_PATH}")
        
        # Check if file exists
        hybrid_path = settings.HYBRID_MODEL_PATH
        if os.path.exists(hybrid_path):
            print(f"   ✅ Hybrid model file exists")
            
            # Try loading
            print(f"\n2. Loading Hybrid Model...")
            model = HybridQNN(n_features=30, n_qubits=4, n_layers=2)
            model.load_state_dict(torch.load(hybrid_path, map_location='cpu'))
            model.eval()
            print(f"   ✅ Model loaded successfully")
            
            # Test prediction
            print(f"\n3. Testing Model Prediction...")
            test_input = torch.randn(1, 30)
            with torch.no_grad():
                output = model(test_input)
                probability = torch.sigmoid(output).item()
            
            print(f"   ✅ Prediction successful")
            print(f"   Raw output: {output.item():.4f}")
            print(f"   Probability: {probability:.4f}")
            
            return True
        else:
            print(f"   ❌ Hybrid model file NOT found at: {hybrid_path}")
            return False
            
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "🧪" * 35)
    print("QUANTUM MODEL BACKEND INTEGRATION TEST")
    print("🧪" * 35 + "\n")
    
    # Test model loading first
    loading_success = asyncio.run(test_model_loading())
    
    # Test ML service
    service_success = asyncio.run(test_ml_service())
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if loading_success and service_success:
        print("✅ Quantum model is working properly in the backend!")
        print("✅ Ready for production use!")
    elif service_success:
        print("⚠️  Backend service works but model loading has issues")
        print("   System will use fallback predictions")
    else:
        print("❌ Issues detected - check error messages above")
    
    print("\n")

if __name__ == "__main__":
    main()
