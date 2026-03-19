#!/usr/bin/env python3
"""
WebSocket test client for real-time fraud detection
"""
import asyncio
import websockets
import json
from datetime import datetime

async def test_websocket():
    """Test WebSocket connection and real-time features"""
    uri = "ws://localhost:8000/ws/test_client"
    
    try:
        print("🔌 Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket!")
            
            # Send a test message
            test_message = {
                "type": "test",
                "message": "Hello from test client",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket.send(json.dumps(test_message))
            print(f"📤 Sent: {test_message}")
            
            # Listen for messages
            print("👂 Listening for messages (press Ctrl+C to stop)...")
            
            try:
                while True:
                    message = await websocket.recv()
                    print(f"📥 Received: {message}")
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping WebSocket client...")
                
    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

def main():
    """Run WebSocket test"""
    print("🚀 WebSocket Test Client")
    print("=" * 30)
    asyncio.run(test_websocket())

if __name__ == "__main__":
    main()