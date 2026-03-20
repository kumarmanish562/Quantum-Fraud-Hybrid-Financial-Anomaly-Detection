from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

from app.schemas.transaction import Transaction, TransactionCreate, TransactionFilter

# Shared in-memory storage (singleton pattern for demo)
_TRANSACTION_STORAGE: Dict[str, Transaction] = {}

class TransactionService:
    def __init__(self):
        # Use shared storage so all instances see the same data
        self.transactions = _TRANSACTION_STORAGE
    
    async def create_transaction(self, transaction_data: TransactionCreate) -> Transaction:
        """Create a new transaction"""
        transaction_id = str(uuid.uuid4())
        
        transaction = Transaction(
            id=transaction_id,
            user_id=transaction_data.user_id,
            amount=transaction_data.amount,
            timestamp=datetime.utcnow(),
            merchant_category=transaction_data.merchant_category,
            merchant_name=transaction_data.merchant_name,
            location=transaction_data.location,
            description=transaction_data.description,
            is_fraud=None,  # Will be set by fraud detection
            fraud_probability=None,
            risk_score=None
        )
        
        self.transactions[transaction_id] = transaction
        return transaction
    
    async def update_transaction_fraud_status(
        self, 
        transaction_id: str, 
        is_fraud: bool, 
        fraud_probability: float,
        risk_score: float = None
    ) -> Optional[Transaction]:
        """Update transaction with fraud detection results"""
        transaction = self.transactions.get(transaction_id)
        if transaction:
            transaction.is_fraud = is_fraud
            transaction.fraud_probability = fraud_probability
            transaction.risk_score = risk_score or fraud_probability
            self.transactions[transaction_id] = transaction
        return transaction
    
    async def get_transaction_by_id(self, transaction_id: str) -> Optional[Transaction]:
        """Get a transaction by ID"""
        return self.transactions.get(transaction_id)
    
    async def get_transactions(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[TransactionFilter] = None
    ) -> List[Transaction]:
        """Get transactions with optional filtering"""
        transactions = list(self.transactions.values())
        
        # Apply filters
        if filters:
            if filters.start_date:
                transactions = [t for t in transactions if t.timestamp >= filters.start_date]
            if filters.end_date:
                transactions = [t for t in transactions if t.timestamp <= filters.end_date]
            if filters.is_fraud is not None:
                transactions = [t for t in transactions if t.is_fraud == filters.is_fraud]
            if filters.min_amount is not None:
                transactions = [t for t in transactions if t.amount >= filters.min_amount]
            if filters.max_amount is not None:
                transactions = [t for t in transactions if t.amount <= filters.max_amount]
            if filters.user_id:
                transactions = [t for t in transactions if t.user_id == filters.user_id]
        
        # Sort by timestamp (newest first)
        transactions.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply pagination
        return transactions[skip:skip + limit]
    
    async def get_transaction_stats(self) -> Dict[str, Any]:
        """Get transaction statistics"""
        transactions = list(self.transactions.values())
        
        if not transactions:
            return {
                "total_transactions": 0,
                "total_amount": 0,
                "fraud_count": 0,
                "fraud_rate": 0,
                "avg_amount": 0
            }
        
        total_transactions = len(transactions)
        total_amount = sum(t.amount for t in transactions)
        fraud_count = sum(1 for t in transactions if t.is_fraud)
        fraud_rate = fraud_count / total_transactions if total_transactions > 0 else 0
        avg_amount = total_amount / total_transactions if total_transactions > 0 else 0
        
        # Recent stats (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_transactions = [t for t in transactions if t.timestamp >= recent_cutoff]
        recent_fraud_count = sum(1 for t in recent_transactions if t.is_fraud)
        
        return {
            "total_transactions": total_transactions,
            "total_amount": round(total_amount, 2),
            "fraud_count": fraud_count,
            "fraud_rate": round(fraud_rate * 100, 2),
            "avg_amount": round(avg_amount, 2),
            "recent_24h": {
                "transactions": len(recent_transactions),
                "fraud_count": recent_fraud_count,
                "fraud_rate": round(recent_fraud_count / len(recent_transactions) * 100, 2) if recent_transactions else 0
            }
        }