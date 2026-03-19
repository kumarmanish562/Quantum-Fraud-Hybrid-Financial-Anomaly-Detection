from fastapi import APIRouter

from app.api.v1.endpoints import fraud_detection, transactions, analytics, auth

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(fraud_detection.router, prefix="/fraud", tags=["fraud-detection"])
api_router.include_router(transactions.router, prefix="/transactions", tags=["transactions"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])