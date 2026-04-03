from sqlalchemy import Column, String, Boolean, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, nullable=False)
    company_name = Column(String)
    phone = Column(String)
    password_hash = Column(String, nullable=False)
    is_active = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    otp_code = Column(String)
    otp_expiry = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    api_key = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, default="Default API Key")
    is_active = Column(Boolean, default=True)
    requests_count = Column(Integer, default=0)
    requests_limit = Column(Integer, default=1000)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)
