# Quantum Fraud Detection Backend API

A real-time fraud detection API using quantum-classical hybrid machine learning models.

## Features

- 🔮 **Quantum-Classical Hybrid ML**: Advanced fraud detection using quantum neural networks
- ⚡ **Real-time Processing**: WebSocket-based real-time fraud alerts
- 🛡️ **Secure Authentication**: JWT-based user authentication
- 📊 **Analytics Dashboard**: Comprehensive fraud analytics and reporting
- 🚀 **High Performance**: Async FastAPI with optimized ML inference
- 📈 **Scalable Architecture**: Modular design for easy scaling

## Quick Start

### 1. Setup Virtual Environment

**Windows:**
```bash
# Run the setup script
setup_venv.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Run the setup script
chmod +x setup_venv.sh
./setup_venv.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
# (Optional - defaults work for development)
```

### 3. Start the Server

```bash
# Using the runner script
python run_server.py

# Or directly with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **WebSocket**: ws://localhost:8000/ws/{client_id}

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get token
- `GET /api/v1/auth/me` - Get current user info

### Fraud Detection
- `POST /api/v1/fraud/predict` - Single transaction prediction
- `POST /api/v1/fraud/predict/batch` - Batch predictions
- `POST /api/v1/fraud/predict/realtime` - Real-time prediction with alerts
- `GET /api/v1/fraud/models/status` - Model status

### Transactions
- `GET /api/v1/transactions/` - List transactions
- `POST /api/v1/transactions/` - Create transaction
- `GET /api/v1/transactions/{id}` - Get specific transaction
- `GET /api/v1/transactions/stats/summary` - Transaction statistics

### Analytics
- `GET /api/v1/analytics/dashboard` - Dashboard metrics
- `GET /api/v1/analytics/fraud-trends` - Fraud trends over time
- `GET /api/v1/analytics/model-performance` - ML model metrics
- `GET /api/v1/analytics/real-time/metrics` - Real-time system metrics

## Real-time Features

### WebSocket Connection

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/client123');

// Listen for fraud alerts
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'fraud_alert') {
        console.log('Fraud detected:', data.data);
    }
};
```

### Real-time Fraud Detection

```python
import requests

# Send transaction for real-time analysis
response = requests.post('http://localhost:8000/api/v1/fraud/predict/realtime', 
    json={
        "transaction_id": "txn_123",
        "amount": 5000.0,
        "time": "2024-01-01T12:00:00Z",
        "v1": 0.5, "v2": -1.2, # ... other PCA features
        # ... (include all required V1-V28 features)
    }
)
```

## ML Models

The API supports multiple ML models:

1. **Classical XGBoost**: Traditional gradient boosting
2. **Hybrid Quantum-Classical**: Quantum neural network with classical preprocessing
3. **Rule-based Fallback**: Simple rules when models aren't available

Models are automatically loaded from `../ml_engine/saved_models/` directory.

## Development

### Project Structure

```
backend/
├── app/
│   ├── api/v1/endpoints/    # API route handlers
│   ├── core/                # Configuration
│   ├── schemas/             # Pydantic models
│   ├── services/            # Business logic
│   └── websocket/           # WebSocket management
├── requirements.txt         # Dependencies
├── run_server.py           # Development server
└── README.md               # This file
```

### Adding New Features

1. **New API Endpoint**: Add to `app/api/v1/endpoints/`
2. **New Schema**: Add to `app/schemas/`
3. **New Service**: Add to `app/services/`
4. **Configuration**: Update `app/core/config.py`

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

## Production Deployment

### Environment Variables

Key environment variables for production:

```bash
# Security
SECRET_KEY=your-secure-secret-key
ENVIRONMENT=production
DEBUG=False

# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Redis (for real-time features)
REDIS_URL=redis://host:port

# CORS
BACKEND_CORS_ORIGINS=["https://yourdomain.com"]
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring

The API includes built-in monitoring endpoints:

- `/health` - Basic health check
- `/api/v1/analytics/real-time/metrics` - System metrics
- WebSocket connection count tracking

## Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS protection
- Input validation with Pydantic
- SQL injection protection (when using database)

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the logs for error details
3. Ensure all dependencies are installed correctly
4. Verify ML models are available in the expected paths