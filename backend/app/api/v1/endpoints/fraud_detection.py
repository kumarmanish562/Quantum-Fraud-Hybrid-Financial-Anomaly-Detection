from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import asyncio
import numpy as np

from app.schemas.fraud import TransactionInput, FraudPrediction, BatchPredictionRequest
from app.services.ml_service import MLService
from app.websocket.manager import ConnectionManager

router = APIRouter()
ml_service = MLService()
manager = ConnectionManager()

@router.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict fraud for a single transaction using hybrid quantum-classical model
    """
    try:
        # Convert transaction to feature array
        features = transaction.to_feature_array()
        
        # Get prediction from ML service
        prediction = await ml_service.predict_single(features)
        
        return FraudPrediction(
            transaction_id=transaction.transaction_id,
            is_fraud=prediction["is_fraud"],
            fraud_probability=prediction["probability"],
            confidence_score=prediction["confidence"],
            model_used=prediction["model"],
            risk_factors=prediction.get("risk_factors", [])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/batch", response_model=List[FraudPrediction])
async def predict_fraud_batch(request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions
    """
    try:
        predictions = []
        
        for transaction in request.transactions:
            features = transaction.to_feature_array()
            prediction = await ml_service.predict_single(features)
            
            predictions.append(FraudPrediction(
                transaction_id=transaction.transaction_id,
                is_fraud=prediction["is_fraud"],
                fraud_probability=prediction["probability"],
                confidence_score=prediction["confidence"],
                model_used=prediction["model"],
                risk_factors=prediction.get("risk_factors", [])
            ))
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.post("/predict/realtime")
async def predict_fraud_realtime(
    transaction: TransactionInput,
    background_tasks: BackgroundTasks
):
    """
    Real-time fraud prediction with WebSocket notification
    """
    try:
        # Get prediction
        features = transaction.to_feature_array()
        prediction = await ml_service.predict_single(features)
        
        result = FraudPrediction(
            transaction_id=transaction.transaction_id,
            is_fraud=prediction["is_fraud"],
            fraud_probability=prediction["probability"],
            confidence_score=prediction["confidence"],
            model_used=prediction["model"],
            risk_factors=prediction.get("risk_factors", [])
        )
        
        # Send real-time notification if fraud detected
        if result.is_fraud and result.fraud_probability > 0.7:
            background_tasks.add_task(
                notify_fraud_alert,
                result.dict()
            )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Real-time prediction failed: {str(e)}")

@router.get("/models/status")
async def get_model_status():
    """
    Get status of loaded ML models
    """
    try:
        status = await ml_service.get_model_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@router.post("/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """
    Trigger model retraining (background task)
    """
    try:
        background_tasks.add_task(ml_service.retrain_models)
        return {"message": "Model retraining started", "status": "in_progress"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")

async def notify_fraud_alert(prediction_data: Dict[str, Any]):
    """
    Send fraud alert via WebSocket to connected clients
    """
    alert_message = {
        "type": "fraud_alert",
        "data": prediction_data,
        "timestamp": prediction_data.get("timestamp")
    }
    
    # Broadcast to all connected clients
    await manager.broadcast(alert_message)