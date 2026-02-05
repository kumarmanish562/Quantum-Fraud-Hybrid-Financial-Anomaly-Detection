from ..models.classical import ClassicalModel
from ..dataset import CreditCardDataset
from ..config import XGB_PARAMS, CLASSICAL_MODEL_PATH

def run_classical_training():
    print("Loading Data...")
    try:
        dataset = CreditCardDataset()
        X_train, X_test, y_train, y_test = dataset.preprocess_and_split(use_smote=True)
    except FileNotFoundError as e:
        print(e)
        return
    
    print("Initializing Classical Model (XGBoost)...")
    model = ClassicalModel('xgboost', XGB_PARAMS)
    
    print("Training...")
    model.train(X_train, y_train)
    
    print("Evaluating...")
    acc = model.evaluate(X_test, y_test)
    print(f"Classical Model Accuracy: {acc}")
    
    print(f"Saving model to {CLASSICAL_MODEL_PATH}...")
    model.save(CLASSICAL_MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    run_classical_training()
