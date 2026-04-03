"""
Database initialization script
Run this after PostgreSQL setup to create tables using SQLAlchemy
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import init_db, engine
from sqlalchemy import text

def check_connection():
    """Check if database connection is working"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def create_tables():
    """Create all tables"""
    try:
        init_db()
        print("✅ All tables created successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False

def show_tables():
    """Show all created tables"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))
            tables = result.fetchall()
            print("\n📋 Created tables:")
            for table in tables:
                print(f"   - {table[0]}")
        return True
    except Exception as e:
        print(f"❌ Failed to show tables: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting database initialization...\n")
    
    # Step 1: Check connection
    if not check_connection():
        print("\n⚠️  Please check your DATABASE_URL in .env file")
        sys.exit(1)
    
    # Step 2: Create tables
    if not create_tables():
        sys.exit(1)
    
    # Step 3: Show created tables
    show_tables()
    
    print("\n✨ Database initialization completed!")
    print("🎉 You can now start the backend server!")
