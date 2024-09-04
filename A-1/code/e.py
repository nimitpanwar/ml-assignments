import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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
    for i in range(iterations):
        # Forward propagation
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        
        # Compute gradients
        dw = 1/m * np.dot(X.T, (y_pred - y))
        db = 1/m * np.sum(y_pred - y)
        
        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias

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

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, val_index in kf.split(features_min_max):
    X_train, X_val = features_min_max.values[train_index], features_min_max.values[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    
    # Initialize weights and bias
    weights, bias = initialize_weights(X_train.shape[1])
    
    # Train the model
    weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate=0.01, iterations=1000)
    
    # Predict on the validation set
    z_val = np.dot(X_val, weights) + bias
    y_val_pred_prob = sigmoid(z_val)
    y_val_pred = [1 if i > 0.5 else 0 for i in y_val_pred_prob]
    
    # Calculate metrics
    accuracy_scores.append(accuracy_score(y_val, y_val_pred))
    precision_scores.append(precision_score(y_val, y_val_pred))
    recall_scores.append(recall_score(y_val, y_val_pred))
    f1_scores.append(f1_score(y_val, y_val_pred))

# Calculate average and standard deviation for each metric
accuracy_mean = np.mean(accuracy_scores)
accuracy_std = np.std(accuracy_scores)
precision_mean = np.mean(precision_scores)
precision_std = np.std(precision_scores)
recall_mean = np.mean(recall_scores)
recall_std = np.std(recall_scores)
f1_mean = np.mean(f1_scores)
f1_std = np.std(f1_scores)

print(f"Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
print(f"F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")

# Analysis
print("\nAnalysis:")
print("The average and standard deviation of the metrics across the folds provide insight into the model's stability and variance.")
print("A low standard deviation indicates that the model's performance is consistent across different folds, suggesting good stability.")
print("A high standard deviation indicates that the model's performance varies significantly across different folds, suggesting potential issues with robustness.")