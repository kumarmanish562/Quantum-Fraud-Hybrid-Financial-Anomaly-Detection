# Hybrid Quantum-Classical Fraud Detection: Presentation Guide

This guide is designed to help you present your project step-by-step. It breaks down the technical implementation into logical segments with key talking points for each.

---

## 🏗️ Project Overview
**Goal:** To develop a robust financial anomaly detection system using a hybrid quantum-classical neural network (HQCNN) to identify fraudulent credit card transactions.

**Key Highlights:**
- **Problem:** Financial fraud is rare (imbalanced data), making detection difficult for standard models.
- **Solution:** Leverage the expressive power of Quantum Variational Circuits alongside Classical Deep Learning.
- **Tools:** Pennylane (Quantum), PyTorch (Classical), Scikit-Learn (Analytics).

---

## 📊 Step 1: Data Preprocessing
*How we prepared the data for a Hybrid Model.*

- **Dataset:** Kaggle Credit Card Fraud Detection (284,807 transactions, only 492 frauds).
- **Scaling:** Uses `StandardScaler` to normalize features (Time, Amount, and V1-V28).
- **Addressing Imbalance:** 
    - **SMOTE (Synthetic Minority Over-sampling Technique):** Synthetically generates new fraud cases to balance the training set.
    - **Stratified Splitting:** Ensures both training and testing sets maintain the same ratio of fraud cases.

---

## 🛠️ Step 2: Feature Engineering & Dimensionality Reduction
*Bridging the gap between 30 features and a limited number of qubits.*

- **Classical Compression:** Since we use a limited number of qubits (e.g., 4-8), we use a classical `nn.Linear` layer to compress the 30 input features into a smaller representation.
- **Normalization:** The features are normalized to a range suitable for quantum embedding (using `tanh` to map values to $[-\pi, \pi]$).

---

## ⚛️ Step 3: Hybrid Quantum Architecture
*The "Secret Sauce" - How Quantum and Classical layers interact.*

1.  **Classical Layer:** A standard Feed-Forward Neural Network layer for initial feature extraction.
2.  **Quantum Embedding:** `AngleEmbedding` maps compressed classical features into the Hilbert space of the qubits.
3.  **Variational Quantum Circuit (VQC):** 
    - `StronglyEntanglingLayers` apply rotations and CNOT gates to create complex patterns.
    - This is where "Quantum Advantage" potentially resides: the ability to represent high-dimensional non-linear boundaries.
4.  **Measurement:** Measuring expectation values ($PauliZ$) to convert quantum states back into classical numerical data.
5.  **Classification Layer:** A final `Linear` + `Sigmoid` layer to output the fraud probability (0 to 1).

---

## 🚀 Step 4: Model Training & Tuning
*The learning process.*

- **Loss Function:** Binary Cross Entropy (BCE) Loss - ideal for fraud vs. non-fraud.
- **Optimizer:** Adam Optimizer for efficient weight updates in both classical and quantum layers.
- **Challenge:** Quantum simulation is computationally expensive. We focus on mini-batch training to speed up convergence.

---

## 📈 Step 5: Performance Evaluation & Results
*Interpreting the Paper Metrics Chart.*

- **Compare Models:** We compare the Hybrid Model against a standard Classical Neural Network and a Random Forest.
- **Key Metrics:**
    - **Accuracy:** Overall correctness (can be misleading due to imbalance).
    - **Precision:** Of all predicted frauds, how many were actually fraud?
    - **Recall (Sensitivity):** How many of the actual frauds did we catch? (Crucial for banking).
    - **F1-Score:** The harmonic mean of Precision and Recall.

---

## 💡 Potential Q&A / Discussion Points
1.  **Why use Quantum?** Quantum circuits can explore feature spaces that are exponentially large, potentially finding patterns classical models miss.
2.  **Is it faster?** Not yet. Current quantum hardware is simulated. The goal is "higher accuracy" or "better generalization," not speed in the current phase.
3.  **Scaling:** Future work involves deploying on real NISQ (Noisy Intermediate-Scale Quantum) hardware.

---

*This guide was generated to assist in the final phase of the Hybrid Financial Anomaly Detection project.*
