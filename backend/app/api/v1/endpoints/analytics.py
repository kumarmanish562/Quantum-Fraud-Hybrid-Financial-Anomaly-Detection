from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from datetime import datetime, timedelta

from app.services.analytics_service import AnalyticsService

router = APIRouter()
analytics_service = AnalyticsService()

@router.get("/dashboard")
async def get_dashboard_metrics():
    """
    Get key metrics for the dashboard
    """
    try:
        return await analytics_service.get_dashboard_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard metrics: {str(e)}")

@router.get("/fraud-trends")
async def get_fraud_trends(
    days: int = Query(30, ge=1, le=365),
    granularity: str = Query("daily", regex="^(hourly|daily|weekly|monthly)$")
):
    """
    Get fraud detection trends over time
    """
    try:
        return await analytics_service.get_fraud_trends(days=days, granularity=granularity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get fraud trends: {str(e)}")

@router.get("/model-performance")
async def get_model_performance():
    """
    Get ML model performance metrics
    """
    try:
        return await analytics_service.get_model_performance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")

@router.get("/risk-distribution")
async def get_risk_distribution():
    """
    Get distribution of risk scores
    """
    try:
        return await analytics_service.get_risk_distribution()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get risk distribution: {str(e)}")

@router.get("/transaction-patterns")
async def get_transaction_patterns(
    pattern_type: str = Query("amount", regex="^(amount|time|location|merchant)$"),
    days: int = Query(7, ge=1, le=90)
):
    """
    Get transaction pattern analysis
    """
    try:
        return await analytics_service.get_transaction_patterns(
            pattern_type=pattern_type,
            days=days
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transaction patterns: {str(e)}")

@router.get("/alerts/summary")
async def get_alerts_summary():
    """
    Get summary of fraud alerts
    """
    try:
        return await analytics_service.get_alerts_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts summary: {str(e)}")

@router.get("/real-time/metrics")
async def get_realtime_metrics():
    """
    Get real-time system metrics
    """
    try:
        return await analytics_service.get_realtime_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time metrics: {str(e)}")