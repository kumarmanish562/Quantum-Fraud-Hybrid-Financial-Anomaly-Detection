"""
Test script to verify quantum model functionality
"""
import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pytorch_import():
    """Test if PyTorch can be imported and used"""
    print("=" * 60)
    print("TEST 1: PyTorch Import and Basic Operations")
    print("=" * 60)
    try:
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ PyTorch CUDA available: {torch.cuda.is_available()}")
        
        # Test basic tensor operations
        test_tensor = torch.randn(3, 3)
        print(f"✅ Created test tensor: {test_tensor.shape}")
        
        # Test computation
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✅ Matrix multiplication works: {result.shape}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_pennylane_import():
    """Test if PennyLane can be imported"""
    print("\n" + "=" * 60)
    print("TEST 2: PennyLane Import")
    print("=" * 60)
    try:
        import pennylane as qml
        print(f"✅ PennyLane version: {qml.__version__}")
        
        # Test basic quantum circuit
        dev = qml.device('default.qubit', wires=2)
        
        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        test_params = np.array([0.5, 0.3])
        result = circuit(test_params)
        print(f"✅ Basic quantum circuit works: output = {result}")
        
        return True
    except Exception as e:
        print(f"❌ PennyLane test failed: {e}")
        return False

def test_hybrid_model_import():
    """Test if hybrid model can be imported"""
    print("\n" + "=" * 60)
    print("TEST 3: Hybrid Model Import")
    print("=" * 60)
    try:
        from ml_engine.models.hybrid_nn import HybridQNN
        print("✅ HybridQNN class imported successfully")
        return True
    except Exception as e:
        print(f"❌ Hybrid model import failed: {e}")
        return False

def test_hybrid_model_creation():
    """Test if hybrid model can be created"""
    print("\n" + "=" * 60)
    print("TEST 4: Hybrid Model Creation")
    print("=" * 60)
    try:
        from ml_engine.models.hybrid_nn import HybridQNN
        
        # Create model with test parameters
        model = HybridQNN(n_features=30, n_qubits=4, n_layers=2)
        print(f"✅ Model created successfully")
        print(f"   - Input features: 30")
        print(f"   - Quantum qubits: 4")
        print(f"   - Quantum layers: 2")
        
        # Check model structure
        print(f"✅ Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        return True, model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False, None

def test_hybrid_model_forward():
    """Test if hybrid model can perform forward pass"""
    print("\n" + "=" * 60)
    print("TEST 5: Hybrid Model Forward Pass")
    print("=" * 60)
    try:
        from ml_engine.models.hybrid_nn import HybridQNN
        
        model = HybridQNN(n_features=30, n_qubits=4, n_layers=2)
        model.eval()
        
        # Create dummy input (batch_size=2, features=30)
        dummy_input = torch.randn(2, 30)
        print(f"✅ Created dummy input: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output values: {output.squeeze().numpy()}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_saved_model_loading():
    """Test if saved quantum model can be loaded"""
    print("\n" + "=" * 60)
    print("TEST 6: Saved Model Loading")
    print("=" * 60)
    
    model_path = "ml_engine/saved_models/quantum_hqnn.pth"
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print("   This is OK - model needs to be trained first")
        return False
    
    try:
        from ml_engine.models.hybrid_nn import HybridQNN
        
        # Load model
        model = HybridQNN(n_features=30, n_qubits=4, n_layers=2)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        print(f"✅ Saved model loaded successfully from: {model_path}")
        
        # Test with dummy data
        dummy_input = torch.randn(1, 30)
        with torch.no_grad():
            output = model(dummy_input)
            probability = torch.sigmoid(output).item()
        
        print(f"✅ Model prediction works")
        print(f"   - Raw output: {output.item():.4f}")
        print(f"   - Probability: {probability:.4f}")
        print(f"   - Prediction: {'FRAUD' if probability > 0.5 else 'LEGITIMATE'}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_metrics():
    """Check if model metrics are available"""
    print("\n" + "=" * 60)
    print("TEST 7: Model Metrics")
    print("=" * 60)
    
    metrics_path = "ml_engine/saved_models/hybrid_metrics.json"
    
    if not os.path.exists(metrics_path):
        print(f"⚠️  Metrics file not found: {metrics_path}")
        return False
    
    try:
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"✅ Metrics loaded successfully")
        print(f"\nModel Performance:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   - {key}: {value:.4f}")
            else:
                print(f"   - {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Metrics loading failed: {e}")
        return False

def test_backend_integration():
    """Test if backend can use the quantum model"""
    print("\n" + "=" * 60)
    print("TEST 8: Backend Integration")
    print("=" * 60)
    try:
        # Change to backend directory context
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))
        
        from app.services.ml_service import MLService
        
        print("✅ ML Service imported successfully")
        
        # Create service instance
        ml_service = MLService()
        print("✅ ML Service initialized")
        
        # Check model status
        print(f"\nModel Status:")
        for model_name, status in ml_service.model_metadata.items():
            if isinstance(status, dict):
                print(f"   - {model_name}:")
                for key, value in status.items():
                    print(f"      • {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Backend integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "🔬" * 30)
    print("QUANTUM MODEL VERIFICATION TEST SUITE")
    print("🔬" * 30 + "\n")
    
    results = {}
    
    # Run tests
    results['pytorch'] = test_pytorch_import()
    results['pennylane'] = test_pennylane_import()
    results['import'] = test_hybrid_model_import()
    
    if results['import']:
        success, model = test_hybrid_model_creation()
        results['creation'] = success
        
        if success:
            results['forward'] = test_hybrid_model_forward()
    else:
        results['creation'] = False
        results['forward'] = False
    
    results['saved_model'] = test_saved_model_loading()
    results['metrics'] = test_model_metrics()
    results['backend'] = test_backend_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name.upper()}")
    
    print("\n" + "=" * 60)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your quantum model is working properly!")
    elif passed >= total * 0.7:
        print("\n⚠️  MOSTLY WORKING - Some issues detected but core functionality works")
    else:
        print("\n❌ ISSUES DETECTED - Quantum model needs attention")
    
    print("\n")

if __name__ == "__main__":
    main()
