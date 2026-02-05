import torch
import torch.nn as nn
import torch.optim as optim
from ..models.hybrid_nn import HybridQNN
from ..dataset import CreditCardDataset
from ..config import HQNN_PARAMS, QUANTUM_MODEL_PATH

def run_hybrid_training():
    print("Loading Data for Hybrid Model...")
    try:
        dataset = CreditCardDataset()
        train_loader, test_loader = dataset.get_torch_dataloaders(batch_size=HQNN_PARAMS['batch_size'])
    except FileNotFoundError as e:
        print(e)
        return
    
    # Get n_features from dataset
    X_sample, _ = next(iter(train_loader))
    n_features = X_sample.shape[1]
    
    print(f"Features: {n_features}")
    
    print("Initializing Hybrid QNN...")
    model = HybridQNN(n_features, HQNN_PARAMS['n_qubits'], HQNN_PARAMS['n_layers'])
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=HQNN_PARAMS['learning_rate'])
    
    print("Training...")
    for epoch in range(HQNN_PARAMS['epochs']):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{HQNN_PARAMS['epochs']}, Loss: {total_loss/len(train_loader)}")
        
    print("Evaluating...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted.squeeze() == y_batch).sum().item()
            
    acc = correct / total
    print(f"Hybrid Model Accuracy: {acc}")
    
    print(f"Saving model to {QUANTUM_MODEL_PATH}...")
    torch.save(model.state_dict(), QUANTUM_MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    run_hybrid_training()
