import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('heart_disease.csv')

# Handle missing values (simple imputation with mean for this example)
data.fillna(data.mean(), inplace=True)

# Feature scaling methods
def min_max_scaling(features):
    return (features - features.min()) / (features.max() - features.min())

# Prepare data
features = data.drop(columns=['HeartDisease'])
labels = data['HeartDisease'].values

# Min-max scaling
features_min_max = min_max_scaling(features)
X_train_mm, X_val_mm, y_train_mm, y_val_mm = train_test_split(features_min_max.values, labels, test_size=0.3, random_state=42)

# Initialize weights and bias
def initialize_weights(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss function with regularization
def compute_loss(y_true, y_pred, weights, l1_reg=0, l2_reg=0):
    m = len(y_true)
    loss = -1/m * (np.dot(y_true, np.log(y_pred)) + np.dot((1 - y_true), np.log(1 - y_pred)))
    l1_term = l1_reg * np.sum(np.abs(weights))
    l2_term = l2_reg * np.sum(weights**2)
    return loss + l1_term + l2_term

# Gradient descent algorithm with early stopping
def gradient_descent(X, y, X_val, y_val, weights, bias, learning_rate, iterations, patience, l1_reg=0, l2_reg=0):
    m = len(y)
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history = []
    val_loss_history = []
    
    for i in range(iterations):
        # Forward propagation
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        
        # Compute loss
        loss = compute_loss(y, y_pred, weights, l1_reg, l2_reg)
        loss_history.append(loss)
        
        # Compute gradients
        dw = 1/m * np.dot(X.T, (y_pred - y)) + l1_reg * np.sign(weights) + 2 * l2_reg * weights
        db = 1/m * np.sum(y_pred - y)
        
        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Validation loss
        z_val = np.dot(X_val, weights) + bias
        y_val_pred = sigmoid(z_val)
        val_loss = compute_loss(y_val, y_val_pred, weights, l1_reg, l2_reg)
        val_loss_history.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at iteration {i}")
                break
    
    return weights, bias, loss_history, val_loss_history

# Metrics functions
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])

def precision_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def recall_score(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Training parameters
learning_rate = 0.01
iterations = 1000
patience = 10

# Train model with Min-max scaling and early stopping
weights_mm, bias_mm = initialize_weights(X_train_mm.shape[1])
weights_mm, bias_mm, loss_history_mm, val_loss_history_mm = gradient_descent(
    X_train_mm, y_train_mm, X_val_mm, y_val_mm, weights_mm, bias_mm, learning_rate, iterations, patience, l1_reg=0.01, l2_reg=0.01)

# Predict on the validation set
z_val = np.dot(X_val_mm, weights_mm) + bias_mm
y_val_pred_prob = sigmoid(z_val)
y_val_pred = [1 if i > 0.5 else 0 for i in y_val_pred_prob]

# Calculate metrics
accuracy = accuracy_score(y_val_mm, y_val_pred)
precision = precision_score(y_val_mm, y_val_pred)
recall = recall_score(y_val_mm, y_val_pred)
f1 = f1_score(y_val_mm, y_val_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plotting results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss_history_mm, label='Training Loss')
plt.plot(val_loss_history_mm, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration with Early Stopping')
plt.legend()

plt.tight_layout()
plt.show()

# Analysis
print("\nAnalysis:")
print("Early stopping helps to prevent overfitting by stopping the training process when the validation loss stops improving.")
print("Regularization techniques like L1 and L2 help to prevent overfitting by adding a penalty to the loss function.")
print("The combination of early stopping and regularization can improve the generalization performance of the model.")