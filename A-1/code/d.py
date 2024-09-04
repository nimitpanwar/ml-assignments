import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('heart_disease.csv')

# Handle missing values (simple imputation with mean for this example)
data.fillna(data.mean(), inplace=True)

# Feature scaling methods
def min_max_scaling(features):
    return (features - features.min()) / (features.max() - features.min())

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

# Prepare data
features = data.drop(columns=['HeartDisease'])
labels = data['HeartDisease'].values

# Min-max scaling
features_min_max = min_max_scaling(features)
X_train_mm, y_train_mm, X_val_mm, y_val_mm, X_test_mm, y_test_mm = train_test_val_split(features_min_max.values, labels)

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

# Accuracy function
def compute_accuracy(y_true, y_pred):
    y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
    return np.mean(y_pred_class == y_true)

# Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(X, y, X_val, y_val, weights, bias, learning_rate, iterations):
    m = len(y)
    loss_history = []
    val_loss_history = []
    accuracy_history = []
    val_accuracy_history = []
    
    for i in range(iterations):
        for j in range(m):
            # Select one sample
            X_sample = X[j].reshape(1, -1)
            y_sample = y[j].reshape(1, )
            
            # Forward propagation
            z = np.dot(X_sample, weights) + bias
            y_pred = sigmoid(z)
            
            # Compute gradients
            dw = np.dot(X_sample.T, (y_pred - y_sample))
            db = np.sum(y_pred - y_sample)
            
            # Update weights and bias
            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        # Compute loss and accuracy for the entire dataset
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        loss = compute_loss(y, y_pred)
        accuracy = compute_accuracy(y, y_pred)
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        
        # Validation loss and accuracy
        z_val = np.dot(X_val, weights) + bias
        y_val_pred = sigmoid(z_val)
        val_loss = compute_loss(y_val, y_val_pred)
        val_accuracy = compute_accuracy(y_val, y_val_pred)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
    
    return weights, bias, loss_history, val_loss_history, accuracy_history, val_accuracy_history

# Mini-Batch Gradient Descent (MBGD)
def mini_batch_gradient_descent(X, y, X_val, y_val, weights, bias, learning_rate, iterations, batch_size):
    m = len(y)
    loss_history = []
    val_loss_history = []
    accuracy_history = []
    val_accuracy_history = []
    
    for i in range(iterations):
        # Shuffle the data
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        
        for j in range(0, m, batch_size):
            # Select mini-batch
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]
            
            # Forward propagation
            z = np.dot(X_batch, weights) + bias
            y_pred = sigmoid(z)
            
            # Compute gradients
            dw = 1/batch_size * np.dot(X_batch.T, (y_pred - y_batch))
            db = 1/batch_size * np.sum(y_pred - y_batch)
            
            # Update weights and bias
            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        # Compute loss and accuracy for the entire dataset
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        loss = compute_loss(y, y_pred)
        accuracy = compute_accuracy(y, y_pred)
        loss_history.append(loss)
        accuracy_history.append(accuracy)
        
        # Validation loss and accuracy
        z_val = np.dot(X_val, weights) + bias
        y_val_pred = sigmoid(z_val)
        val_loss = compute_loss(y_val, y_val_pred)
        val_accuracy = compute_accuracy(y_val, y_val_pred)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)
    
    return weights, bias, loss_history, val_loss_history, accuracy_history, val_accuracy_history

# Training parameters
learning_rate = 0.01
iterations = 1000

# Train model with Stochastic Gradient Descent (SGD)
weights_sgd, bias_sgd = initialize_weights(X_train_mm.shape[1])
weights_sgd, bias_sgd, loss_history_sgd, val_loss_history_sgd, accuracy_history_sgd, val_accuracy_history_sgd = stochastic_gradient_descent(
    X_train_mm, y_train_mm, X_val_mm, y_val_mm, weights_sgd, bias_sgd, learning_rate, iterations)

# Train model with Mini-Batch Gradient Descent (MBGD) with batch size 32
batch_size = 32
weights_mbgd, bias_mbgd = initialize_weights(X_train_mm.shape[1])
weights_mbgd, bias_mbgd, loss_history_mbgd, val_loss_history_mbgd, accuracy_history_mbgd, val_accuracy_history_mbgd = mini_batch_gradient_descent(
    X_train_mm, y_train_mm, X_val_mm, y_val_mm, weights_mbgd, bias_mbgd, learning_rate, iterations, batch_size)

# Plotting results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(loss_history_sgd, label='SGD - Training Loss')
plt.plot(val_loss_history_sgd, label='SGD - Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration (SGD)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(accuracy_history_sgd, label='SGD - Training Accuracy')
plt.plot(val_accuracy_history_sgd, label='SGD - Validation Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Iteration (SGD)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(loss_history_mbgd, label='MBGD - Training Loss')
plt.plot(val_loss_history_mbgd, label='MBGD - Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration (MBGD)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(accuracy_history_mbgd, label='MBGD - Training Accuracy')
plt.plot(val_accuracy_history_mbgd, label='MBGD - Validation Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Iteration (MBGD)')
plt.legend()

plt.tight_layout()
plt.show()

# Analysis
print("\nAnalysis:")
print("Stochastic Gradient Descent (SGD) updates the weights after each training example, which can lead to faster convergence but also more noise in the updates.")
print("Mini-Batch Gradient Descent (MBGD) updates the weights after a batch of training examples, which can provide a balance between the convergence speed of SGD and the stability of Batch Gradient Descent.")
print("The plots show the trade-offs in terms of convergence speed and stability between these methods.")