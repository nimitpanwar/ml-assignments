import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('heart_disease.csv')

# Handle missing values (simple imputation with mean for this example)
data.fillna(data.mean(), inplace=True)

# Normalize the features
features = data.drop(columns=['HeartDisease'])
features = (features - features.mean()) / features.std()
labels = data['HeartDisease'].values

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

X_train, y_train, X_val, y_val, X_test, y_test = train_test_val_split(features.values, labels)

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
def gradient_descent(X, y, weights, bias, learning_rate, iterations):
    m = len(y)
    loss_history = []
    accuracy_history = []
    
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
        
        # Compute accuracy
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        accuracy = np.mean(y_pred_class == y)
        accuracy_history.append(accuracy)
        
        # Validation loss and accuracy
        if i % 10 == 0:
            z_val = np.dot(X_val, weights) + bias
            y_val_pred = sigmoid(z_val)
            val_loss = compute_loss(y_val, y_val_pred)
            val_loss_history.append(val_loss)
            y_val_pred_class = [1 if i > 0.5 else 0 for i in y_val_pred]
            val_accuracy = np.mean(y_val_pred_class == y_val)
            val_accuracy_history.append(val_accuracy)
    
    return weights, bias, loss_history, accuracy_history, val_loss_history, val_accuracy_history

# Training the model
learning_rate = 0.01
iterations = 1000
weights, bias = initialize_weights(X_train.shape[1])
val_loss_history = []
val_accuracy_history = []

weights, bias, loss_history, accuracy_history, val_loss_history, val_accuracy_history = gradient_descent(
    X_train, y_train, weights, bias, learning_rate, iterations)

# Plotting results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss vs. Iteration')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Validation Loss vs. Iteration')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(accuracy_history, label='Training Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs. Iteration')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy vs. Iteration')
plt.legend()

plt.tight_layout()
plt.show()

# Analysis
print("Final Training Loss:", loss_history[-1])
print("Final Validation Loss:", val_loss_history[-1])
print("Final Training Accuracy:", accuracy_history[-1])
print("Final Validation Accuracy:", val_accuracy_history[-1])