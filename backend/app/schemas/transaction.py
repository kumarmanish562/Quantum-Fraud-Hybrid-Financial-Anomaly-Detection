from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TransactionBase(BaseModel):
    amount: float = Field(..., gt=0)
    merchant_category: Optional[str] = None
    merchant_name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None

class TransactionCreate(TransactionBase):
    user_id: str
    
class Transaction(TransactionBase):
    id: str
    user_id: str
    timestamp: datetime
    is_fraud: Optional[bool] = None
    fraud_probability: Optional[float] = None
    risk_score: Optional[float] = None
    
    class Config:
        from_attributes = True

class TransactionFilter(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_fraud: Optional[bool] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    user_id: Optional[str] = None