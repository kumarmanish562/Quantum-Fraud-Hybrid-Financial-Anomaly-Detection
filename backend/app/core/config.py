from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Quantum Fraud Detection API"
    VERSION: str = "1.0.0"
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./fraud_detection.db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # JWT
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080"
    ]
    
    # ML Models
    MODEL_PATH: str = "../ml_engine/saved_models/"
    CLASSICAL_MODEL_PATH: str = "../ml_engine/saved_models/classical_model.joblib"
    HYBRID_MODEL_PATH: str = "../ml_engine/saved_models/quantum_hqnn.pth"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()