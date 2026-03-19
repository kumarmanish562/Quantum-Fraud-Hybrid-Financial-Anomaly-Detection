import os
import sys
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    torch = None
import joblib
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

# Add parent directory to path to import ml_engine
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from ml_engine.models.classical import ClassicalModel
from ml_engine.models.hybrid_nn import HybridQNN
from app.core.config import settings

class MLService:
    def __init__(self):
        self.classical_model = None
        self.hybrid_model = None
        self.model_metadata = {}
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load classical model
            classical_path = os.path.join(settings.MODEL_PATH, "classical_model.joblib")
            if os.path.exists(classical_path):
                self.classical_model = joblib.load(classical_path)
                self.model_metadata["classical"] = {
                    "loaded": True,
                    "last_loaded": datetime.utcnow(),
                    "path": classical_path
                }
            else:
                # Initialize new classical model if no saved model exists
                self.classical_model = ClassicalModel(model_type='xgboost')
                self.model_metadata["classical"] = {
                    "loaded": False,
                    "needs_training": True
                }
            
            # Load hybrid quantum model
            hybrid_path = os.path.join(settings.MODEL_PATH, "quantum_hqnn.pth")
            if TORCH_AVAILABLE and os.path.exists(hybrid_path):
                # Initialize hybrid model architecture
                self.hybrid_model = HybridQNN(n_features=30, n_qubits=4, n_layers=2)
                self.hybrid_model.load_state_dict(torch.load(hybrid_path, map_location='cpu'))
                self.hybrid_model.eval()
                self.model_metadata["hybrid"] = {
                    "loaded": True,
                    "last_loaded": datetime.utcnow(),
                    "path": hybrid_path
                }
            else:
                self.model_metadata["hybrid"] = {
                    "loaded": False,
                    "needs_training": True,
                    "torch_available": TORCH_AVAILABLE
                }
                
        except Exception as e:
            print(f"Error loading models: {e}")
            self.model_metadata["error"] = str(e)
    
    async def predict_single(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction
        """
        try:
            # Use hybrid model if available, otherwise classical
            if (TORCH_AVAILABLE and self.hybrid_model and 
                self.model_metadata.get("hybrid", {}).get("loaded")):
                return await self._predict_hybrid(features)
            elif self.classical_model and self.model_metadata.get("classical", {}).get("loaded"):
                return await self._predict_classical(features)
            else:
                # Fallback to simple rule-based prediction
                return await self._predict_fallback(features)
                
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    async def _predict_hybrid(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict using hybrid quantum-classical model"""
        if not TORCH_AVAILABLE:
            raise Exception("PyTorch not available for hybrid predictions")
            
        try:
            # Convert to torch tensor
            features_tensor = torch.FloatTensor(features)
            
            # Get prediction
            with torch.no_grad():
                output = self.hybrid_model(features_tensor)
                probability = torch.sigmoid(output).item()
            
            is_fraud = probability > 0.5
            confidence = abs(probability - 0.5) * 2  # Convert to confidence score
            
            # Analyze risk factors based on feature importance
            risk_factors = self._analyze_risk_factors(features, "hybrid")
            
            return {
                "is_fraud": is_fraud,
                "probability": probability,
                "confidence": confidence,
                "model": "hybrid_quantum",
                "risk_factors": risk_factors
            }
            
        except Exception as e:
            raise Exception(f"Hybrid prediction failed: {str(e)}")
    
    async def _predict_classical(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict using classical model"""
        try:
            # Get prediction and probability
            prediction = self.classical_model.model.predict(features)[0]
            probability = self.classical_model.model.predict_proba(features)[0][1]
            
            is_fraud = bool(prediction)
            confidence = abs(probability - 0.5) * 2
            
            # Analyze risk factors
            risk_factors = self._analyze_risk_factors(features, "classical")
            
            return {
                "is_fraud": is_fraud,
                "probability": float(probability),
                "confidence": float(confidence),
                "model": "classical_xgboost",
                "risk_factors": risk_factors
            }
            
        except Exception as e:
            raise Exception(f"Classical prediction failed: {str(e)}")
    
    async def _predict_fallback(self, features: np.ndarray) -> Dict[str, Any]:
        """Fallback rule-based prediction when models are not available"""
        try:
            # Simple rule-based approach
            amount = features[0][-1]  # Amount is the last feature
            
            # Basic rules for demonstration
            is_fraud = False
            probability = 0.1
            
            if amount > 10000:  # Large transactions
                probability += 0.3
            if amount < 1:  # Very small transactions
                probability += 0.2
                
            # Add some randomness for demonstration
            import random
            probability += random.uniform(0, 0.2)
            probability = min(probability, 0.95)
            
            is_fraud = probability > 0.5
            confidence = abs(probability - 0.5) * 2
            
            return {
                "is_fraud": is_fraud,
                "probability": probability,
                "confidence": confidence,
                "model": "rule_based_fallback",
                "risk_factors": ["high_amount"] if amount > 10000 else []
            }
            
        except Exception as e:
            raise Exception(f"Fallback prediction failed: {str(e)}")
    
    def _analyze_risk_factors(self, features: np.ndarray, model_type: str) -> list:
        """Analyze and return risk factors based on feature values"""
        risk_factors = []
        
        try:
            # Extract key features (assuming last feature is amount)
            amount = features[0][-1]
            
            # Amount-based risk factors
            if amount > 10000:
                risk_factors.append("high_amount")
            elif amount < 1:
                risk_factors.append("micro_transaction")
            
            # Time-based risk factors (first feature is timestamp)
            timestamp = features[0][0]
            hour = datetime.fromtimestamp(timestamp).hour
            if hour < 6 or hour > 22:
                risk_factors.append("unusual_time")
            
            # Feature-based risk factors (simplified)
            feature_values = features[0][1:-1]  # Exclude time and amount
            if np.any(np.abs(feature_values) > 3):  # High PCA values
                risk_factors.append("unusual_pattern")
                
        except Exception:
            pass  # Ignore errors in risk factor analysis
            
        return risk_factors
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        return {
            "models": self.model_metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def retrain_models(self):
        """Retrain models (placeholder for background task)"""
        # This would implement model retraining logic
        # For now, just update metadata
        self.model_metadata["last_retrain_attempt"] = datetime.utcnow()
        return {"status": "retraining_completed"}