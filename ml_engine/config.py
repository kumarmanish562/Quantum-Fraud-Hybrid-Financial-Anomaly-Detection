import os

# Paths
# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data and Models are now inside ml_engine
DATA_DIR = os.path.join(ML_ENGINE_DIR, 'data')
SAVED_MODELS_DIR = os.path.join(ML_ENGINE_DIR, 'saved_models')

DATA_FILE_PATH = os.path.join(DATA_DIR, 'creditcard.csv')
CLASSICAL_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'classical_xgboost.pkl')
QUANTUM_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'quantum_hqnn.pth')

# Hyperparameters - Classical
XGB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 4,
    'random_state': 42
}

# Hyperparameters - Hybrid
HQNN_PARAMS = {
    'n_qubits': 4,
    'n_layers': 2,
    'learning_rate': 0.01,
    'epochs': 5,
    'batch_size': 32
}

# Data Settings
TEST_SIZE = 0.2
RANDOM_SEED = 42
