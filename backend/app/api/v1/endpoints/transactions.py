from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta

from app.schemas.transaction import Transaction, TransactionCreate, TransactionFilter
from app.services.transaction_service import TransactionService

router = APIRouter()
transaction_service = TransactionService()

@router.post("/", response_model=Transaction)
async def create_transaction(transaction: TransactionCreate):
    """
    Create a new transaction record
    """
    try:
        return await transaction_service.create_transaction(transaction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create transaction: {str(e)}")

@router.get("/", response_model=List[Transaction])
async def get_transactions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    is_fraud: Optional[bool] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None
):
    """
    Get transactions with optional filtering
    """
    try:
        filters = TransactionFilter(
            start_date=start_date,
            end_date=end_date,
            is_fraud=is_fraud,
            min_amount=min_amount,
            max_amount=max_amount
        )
        
        return await transaction_service.get_transactions(
            skip=skip,
            limit=limit,
            filters=filters
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transactions: {str(e)}")

@router.get("/{transaction_id}", response_model=Transaction)
async def get_transaction(transaction_id: str):
    """
    Get a specific transaction by ID
    """
    try:
        transaction = await transaction_service.get_transaction_by_id(transaction_id)
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return transaction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transaction: {str(e)}")

@router.get("/stats/summary")
async def get_transaction_stats():
    """
    Get transaction statistics summary
    """
    try:
        return await transaction_service.get_transaction_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get transaction stats: {str(e)}")

@router.get("/fraud/recent")
async def get_recent_fraud_transactions(
    hours: int = Query(24, ge=1, le=168),  # Last 24 hours by default, max 1 week
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get recent fraud transactions
    """
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)
        
        filters = TransactionFilter(
            start_date=start_date,
            end_date=end_date,
            is_fraud=True
        )
        
        return await transaction_service.get_transactions(
            skip=0,
            limit=limit,
            filters=filters
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent fraud transactions: {str(e)}")