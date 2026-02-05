import torch
import torch.nn as nn
import pennylane as qml

class HybridQNN(nn.Module):
    def __init__(self, n_features, n_qubits, n_layers, n_classes=1):
        super(HybridQNN, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical Input Layer (Compress features to n_qubits)
        self.cl_layer1 = nn.Linear(n_features, 64)
        self.cl_layer2 = nn.Linear(64, n_qubits)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Quantum Layer logic
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)
        
        # Classical Output Layer
        self.cl_output = nn.Linear(n_qubits, n_classes)
        self.sigmoid = nn.Sigmoid()

    def _circuit(self, inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
        qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        x = self.cl_layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cl_layer2(x)
        x = self.relu(x) 
        
        x = 3.14 * torch.tanh(x) # Normalize to [-pi, pi] roughly for AngleEmbedding
        
        x = self.qlayer(x)
        
        x = self.cl_output(x)
        x = self.sigmoid(x)
        return x
