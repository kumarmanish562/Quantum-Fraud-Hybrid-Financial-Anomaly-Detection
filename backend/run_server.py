#!/usr/bin/env python3
"""
Development server runner for Quantum Fraud Detection API
"""
import uvicorn
import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Run the development server"""
    print("🚀 Starting Quantum Fraud Detection API Server...")
    print("📊 Dashboard will be available at: http://localhost:8000/docs")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws/{client_id}")
    print("⚡ Real-time fraud detection ready!")
    print("-" * 50)
    
    # Check if .env file exists
    env_file = current_dir / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found. Using default configuration.")
        print("   Copy .env.example to .env and configure your settings.")
        print("-" * 50)
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()