import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('heart_disease.csv')

# Handle missing values (simple imputation with mean for this example)
data.fillna(data.mean(), inplace=True)

# Split the dataset into train, validation, and test sets (70:15:15)
def train_test_val_split(features, labels, train_size=0.7, val_size=0.15):
    total_size = len(features)
    train_end = int(train_size * total_size)
    val_end = int((train_size + val_size) * total_size)
    
    X_train = features[:train_end]
    y_train = labels[:train_end]
    X_val = features[train_end:val_end]
    y_val = labels[train_end:val_end]
    X_test = features[val_end:]
    y_test = labels[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Initialize weights and bias
def initialize_weights(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss function
def compute_loss(y_true, y_pred):
    m = len(y_true)
    loss = -1/m * (np.dot(y_true, np.log(y_pred)) + np.dot((1 - y_true), np.log(1 - y_pred)))
    return loss

# Gradient descent algorithm
def gradient_descent(X, y, X_val, y_val, weights, bias, learning_rate, iterations):
    m = len(y)
    loss_history = []
    val_loss_history = []
    
    for i in range(iterations):
        # Forward propagation
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        
        # Compute loss
        loss = compute_loss(y, y_pred)
        loss_history.append(loss)
        
        # Compute gradients
        dw = 1/m * np.dot(X.T, (y_pred - y))
        db = 1/m * np.sum(y_pred - y)
        
        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # Validation loss
        z_val = np.dot(X_val, weights) + bias
        y_val_pred = sigmoid(z_val)
        val_loss = compute_loss(y_val, y_val_pred)
        val_loss_history.append(val_loss)
    
    return weights, bias, loss_history, val_loss_history

# Feature scaling methods
def min_max_scaling(features):
    return (features - features.min()) / (features.max() - features.min())

def no_scaling(features):
    return features

# Prepare data for different scaling methods
features = data.drop(columns=['HeartDisease'])
labels = data['HeartDisease'].values

# Min-max scaling
features_min_max = min_max_scaling(features)
X_train_mm, y_train_mm, X_val_mm, y_val_mm, X_test_mm, y_test_mm = train_test_val_split(features_min_max.values, labels)

# No scaling
features_no_scaling = no_scaling(features)
X_train_ns, y_train_ns, X_val_ns, y_val_ns, X_test_ns, y_test_ns = train_test_val_split(features_no_scaling.values, labels)

# Training parameters
learning_rate = 0.01
iterations = 1000

# Train model with Min-max scaling
weights_mm, bias_mm = initialize_weights(X_train_mm.shape[1])
weights_mm, bias_mm, loss_history_mm, val_loss_history_mm = gradient_descent(
    X_train_mm, y_train_mm, X_val_mm, y_val_mm, weights_mm, bias_mm, learning_rate, iterations)

# Train model with No scaling
weights_ns, bias_ns = initialize_weights(X_train_ns.shape[1])
weights_ns, bias_ns, loss_history_ns, val_loss_history_ns = gradient_descent(
    X_train_ns, y_train_ns, X_val_ns, y_val_ns, weights_ns, bias_ns, learning_rate, iterations)

# Plotting results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss_history_mm, label='Min-max Scaling - Training Loss')
plt.plot(val_loss_history_mm, label='Min-max Scaling - Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration (Min-max Scaling)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_history_ns, label='No Scaling - Training Loss')
plt.plot(val_loss_history_ns, label='No Scaling - Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration (No Scaling)')
plt.legend()

plt.tight_layout()
plt.show()

# Analysis
print("Final Training Loss (Min-max Scaling):", loss_history_mm[-1])
print("Final Validation Loss (Min-max Scaling):", val_loss_history_mm[-1])
print("Final Training Loss (No Scaling):", loss_history_ns[-1])
print("Final Validation Loss (No Scaling):", val_loss_history_ns[-1])