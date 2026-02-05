# Quantum-Fraud-Hybrid-Financial-Anomaly-Detection

This project implements a Hybrid Quantum Neural Network (HQNN) for detecting financial fraud.

## Project Structure

- **ml_engine/**: Core machine learning logic (Dataset, Classical Models, Hybrid QNN).
- **backend/**: FastAPI application to serve the model.
- **frontend/**: Modern Dashboard to visualize system status and accuracy.
- **notebooks/**: Jupyter notebooks for exploration and evaluation.
- **saved_models/**: Trained models and metrics artifacts.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the models (if not already trained):
   ```bash
   python main.py train-hybrid
   ```

3. Run the Backend API:
   ```bash
   cd backend
   python app.py
   ```
   Server will start at `http://localhost:8000`.

4. Run the Frontend:
   - Simply open `frontend/index.html` in your browser.
   - Or serve it using a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 3000
     ```

## Features
- **Accurate**: >96% accuracy on Credit Card Fraud Detection dataset (Imbalanced).
- **Hybrid**: Uses Variational Quantum Circuits (PennyLane) + PyTorch.
- **Interactive**: Real-time fraud probability simulation.
