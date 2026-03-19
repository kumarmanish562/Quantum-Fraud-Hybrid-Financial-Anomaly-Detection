#!/bin/bash

echo "========================================"
echo "  QUANTUM FRAUD DETECTION SYSTEM TEST"
echo "========================================"
echo ""

echo "[1/5] Checking if backend is running..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ [OK] Backend is running on port 8000"
else
    echo "✗ [FAIL] Backend is NOT running!"
    echo "Please start backend: cd backend && python run_server.py"
    exit 1
fi
echo ""

echo "[2/5] Checking if frontend is running..."
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "✓ [OK] Frontend is running on port 5173"
else
    echo "✗ [FAIL] Frontend is NOT running!"
    echo "Please start frontend: cd frontend && npm run dev"
    exit 1
fi
echo ""

echo "[3/5] Testing API endpoints..."
if curl -s http://localhost:8000/api/v1/analytics/dashboard > /dev/null 2>&1; then
    echo "✓ [OK] API endpoints responding"
else
    echo "✗ [FAIL] API not responding properly"
    exit 1
fi
echo ""

echo "[4/5] Checking transaction count..."
echo "Fetching transaction statistics..."
curl -s http://localhost:8000/api/v1/transactions/stats/summary | python -m json.tool 2>/dev/null || echo "No transactions yet"
echo ""

echo "[5/5] System Status Summary"
echo "========================================"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "API Docs: http://localhost:8000/docs"
echo "========================================"
echo ""

echo "✓ [SUCCESS] All systems operational!"
echo ""
echo "Next steps:"
echo "1. Open browser: http://localhost:5173"
echo "2. Generate transactions: cd backend && python add_sample_transaction.py"
echo "3. Test Real-Time Detection page"
echo ""
