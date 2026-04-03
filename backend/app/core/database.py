from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.user import Base
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# PostgreSQL connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:0328@localhost:5432/fraud_detection")

# Create engine with PostgreSQL specific settings
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
